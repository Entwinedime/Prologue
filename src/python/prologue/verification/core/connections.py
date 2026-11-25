"""Verify producer and consumer links, model collectives and CPU communication."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from math import prod

from prologue.common.io import read_json
from prologue.verification.core.actions import ActionIdentity
from prologue.verification.core.adapters import framework_evidence
from prologue.verification.core.protocol import PdPollingEvidence
from prologue.verification.engine import CausalIndex, FindingLedger


class BatchSynchronization:
    def __init__(self, finding):
        self.finding = finding
        self.counts = Counter()

    @staticmethod
    def owns(work):
        positions = [*work.get("callers", ()), work.get("continuation") or {}]
        return any(position["owner"] in {"batch_sync", "batch_sync_exchange"}
                   for position in positions if "owner" in position)

    @staticmethod
    def terminal(work):
        position = work.get("continuation") or {}
        return position.get("owner") == "batch_sync_exchange" and position.get("exit") == "call.return"

    @staticmethod
    def span(identities):
        positions = [identity["work"]["continuation"] for identity in identities]
        if (positions[0]["entry"] != "call.enter" or positions[-1]["exit"] != "call.return"
                or any(position["owner"] != "batch_sync_exchange" for position in positions)):
            raise ValueError("batch synchronization span lacks its complete native invocation")
        enclosing = {name: value for name, value in identities[0]["work"].items() if name != "continuation"}
        if any({name: value for name, value in identity["work"].items() if name != "continuation"} != enclosing
               or identity["group"] != identities[0]["group"] for identity in identities):
            raise ValueError("batch synchronization span crosses different native work")
        for previous, following in pairwise(positions):
            if (previous["exit"] == "call.return" or previous["exit"] != following["entry"]
                    or previous["exit_context"] != following["entry_context"]):
                raise ValueError("batch synchronization segments do not form one continuation")

    def attach(self, payload, endpoints, identity, origin):
        layout = payload["layout"]
        if len(layout) != 3 or any(type(size) is not int or size < 1 for size in layout):
            raise ValueError("batch synchronization requires positive DP/TP/CP dimensions")
        candidates = [
            endpoint
            for endpoint in endpoints
            if endpoint["operation"] == "all_gather_into_tensor" and endpoint["group_role"] == payload["group"]
        ]
        if prod(layout) == 1 and not candidates:
            self.check_values([payload], identity, origin)
            return
        if len(candidates) != 1 or "batch_sync" in candidates[0] or len(candidates[0]["members"]) != prod(layout):
            raise ValueError("batch synchronization observation has no unique matching communication")
        candidates[0]["batch_sync"] = payload

    def check(self, endpoints, origin):
        relevant = [endpoint["operation"] == "all_gather_into_tensor" and self.owns(endpoint["work"].get("work", {}))
                    for endpoint in endpoints]
        if not any(relevant):
            return
        if not all(relevant):
            self.finding("batch_sync_protocol", endpoints[0]["work"], origin=origin,
                         reason="participants disagree on batch synchronization ownership")
            return
        if any("batch_sync" not in endpoint for endpoint in endpoints):
            self.counts[f"{origin}_unobserved"] += 1
            return
        self.check_values([endpoint["batch_sync"] for endpoint in endpoints], endpoints[0]["work"], origin)

    def check_values(self, records, identity, origin):
        first = records[0]
        if (prod(first["layout"]) != len(records)
                or any(record["layout"] != first["layout"] or record["group"] != first["group"] for record in records)):
            raise ValueError("batch synchronization values disagree on participant layout")
        if any(record["active"] != [1] * len(records) for record in records):

            # CPU activity mirrors do not prove device-side fault-recovery masks; no fallback values are invented.
            self.counts[f"{origin}_inactive_unverified"] += 1
            return
        for record in records:
            fields = record["input"]
            counts = ("num_tokens", "num_tokens_for_logprob", "local_forward_mode")
            flags = ("can_cuda_graph", "is_extend_in_batch", "local_can_run_tbo", "can_run_breakable_cuda_graph")
            if (fields.keys() != set(counts + flags)
                    or any(type(fields[name]) is not int or fields[name] < 0 for name in counts)
                    or any(type(fields[name]) is not bool for name in flags)):
                raise ValueError("batch synchronization input lacks the seven native scalar fields")

        # Native coordination consumes the first attention-TP/CP participant of each DP group.
        _, tp, cp = first["layout"]
        rows = [record["input"] for record in records[::tp * cp]]
        expected = {
            "global_num_tokens": [row["num_tokens"] for row in rows],
            "global_num_tokens_for_logprob": [row["num_tokens_for_logprob"] for row in rows],
            "can_cuda_graph": all(row["can_cuda_graph"] for row in rows),
            "is_extend_in_batch": any(row["is_extend_in_batch"] for row in rows),
            "can_run_breakable_cuda_graph": all(row["can_run_breakable_cuda_graph"] for row in rows),
        }
        self.counts[f"{origin}_checked"] += 1
        for record in records:
            if record["output"] != expected:
                self.finding("batch_sync_values", identity, origin=origin, expected=expected, actual=record["output"])
                return
        self.counts[f"{origin}_matched"] += 1


class Connections:
    def __init__(self):
        self.values, self.reads, self.published = {}, {}, {}
        self.ports, self.ready, self.complete = {}, {}, {}
        self.channels = defaultdict(lambda: defaultdict(list))
        self.predicted_channels = defaultdict(lambda: defaultdict(list))
        self.communication_edges = set()
        self.native_communication_edges = set()
        self.metadata_returns = {}
        self.endpoints = {}
        self.action_returns = {}
        self.batch_sync = BatchSynchronization(self.finding)
        self.signal_inputs, self.signal_reads, self.signal_outputs = {}, {}, {}
        self.signal_events, self.signal_edges = {}, set()
        self.query_consumers = {}
        self.findings = FindingLedger()
        self.counts = {}
        self.value_evidence_complete = True

    def finding(self, category, identity, **detail):
        self.findings.add(category, identity, detail)

    @staticmethod
    def communication_bounds(contents, index):
        for piece in contents.work[index]:
            if piece["kind"] == "collective":
                return piece["start"], piece["finish"]
        port = next(port for port in contents.graph["scopes"][index]["ports"] if "rendezvous" in port)
        return port["rendezvous"], port["event"]

    @staticmethod
    def transfer_payload(scope):
        if scope["communication"]["operation"] != "transfer":
            return None
        port, = (port for port in scope["ports"] if port["use"] in {"data", "metadata"})
        tensor = port["tensor"]
        return port["direction"], tensor["dtype"], tensor["shape"]

    def endpoint(self, contents, index, label, identity, event_ids, values):
        scope = contents.graph["scopes"][index]
        entered = next(port["event"] for port in scope["ports"] if port["name"] == "call.enter")
        ready, complete = self.communication_bounds(contents, index)
        return {**scope["communication"], "scope": label, "work": identity, "entered": event_ids[entered],
                "ready": event_ids[ready], "complete": event_ids[complete], "values": values,
                "metadata": scope["role"] == "metadata_collective", "elements": scope["parameters"].get("elements"),
                "group_role": scope["bindings"].get("collective_group"), "payload": self.transfer_payload(scope)}

    def native(self, contents, raw, rank, instance, dependencies, identities):
        graph, owners = contents.graph, {}
        native_events = range(len(graph["events"]))
        endpoints = []
        metadata = {}
        for record in raw.get("action_values", ()):
            if "collective" not in record["values"]:
                continue
            index = record["scope"]
            if (
                type(index) is not int
                or not 0 < index < len(raw["scopes"])
                or raw["scopes"][index]["role"] != "metadata_collective"
            ):
                raise ValueError("metadata evidence requires a unique observed collective")
            index = contents.source_ids[index]
            if index in metadata:
                raise ValueError("metadata collective has competing native observations")
            metadata[index] = record["values"]["collective"]
        for root, scope in enumerate(graph["scopes"]):
            if not ActionIdentity.selected(root, scope):
                continue
            identity = identities[root]
            key = (identity["group"], root)
            for index, path in contents.paths(root).items():
                if index in owners:
                    raise ValueError("native scope belongs to more than one action")
                owners[index] = key, path, identity

        # Native handles or declared scope boundaries identify producers independently of projected signal names.
        signals, inputs = defaultdict(set), []
        for index, (key, path, identity) in owners.items():
            projected = {(port["name"], port["direction"]): port for port in graph["scopes"][index]["ports"]}
            origins = graph["scopes"][index]["bindings"]["source_scopes"]
            native_ports = (port for origin in origins for port in raw["scopes"][origin]["ports"])
            for port in native_ports:
                if port["use"] != "signal":
                    continue
                label = key, path, port["direction"], port["name"]
                if "boundary" in port:
                    reference = port["boundary"]
                    generation = ("boundary", reference["scope"], reference["interval"], reference["side"])
                else:
                    generation = ("point", port["point"]) if "point" in port else ("producer", port["producer"])
                boundary = projected[port["name"], port["direction"]]
                if boundary["use"] != "signal":
                    raise ValueError("native completion source lost its signal port")
                if port["direction"] == "output":
                    signals[generation].add(label)
                else:
                    inputs.append((label, generation, identity))

        # Queries observe a generation without waiting for it; the native record point still identifies its producer.
        for query in raw["state_queries"]:
            if query["kind"] != "event":
                continue
            key, path, identity = owners[contents.source_ids[query["scope"]]]
            inputs.append(((key, path, "query", "completion"), ("point", query["point"]), identity))
        for label, generation, identity in inputs:
            producers = signals[generation]
            if not producers:
                self.finding("native_evidence", identity, scope=label[1], port=label[3],
                             reason="completion handle has no observed producer")
            self.signal_inputs[label] = producers, identity

        # The writing child and the scope exporting its value need not be the same scope.
        exports = defaultdict(list)
        for index, (key, path, _) in owners.items():
            for port in graph["scopes"][index]["ports"]:
                if port["use"] != "data" or port["direction"] != "output":
                    continue
                tensor = port["tensor"]
                writer = owners.get(tensor["writer"])
                if writer is not None and writer[0] == key:
                    exports[tensor["writer"], tensor["storage"], tensor["version"]].append(
                        ((key, path, "output", port["name"]), port["event"]))

        for index, (key, path, identity) in owners.items():
            scope = graph["scopes"][index]
            for port in scope["ports"]:
                if port["use"] != "data" or port["direction"] != "input" or 0 in port["tensor"]["shape"]:
                    continue
                tensor = port["tensor"]
                writer = owners.get(tensor["writer"])
                if writer is None:
                    self.finding("native_evidence", identity, scope=path, port=port["name"],
                        reason="data writer has no observed action")
                    self.value_evidence_complete = False
                    continue
                if writer[0] == key:
                    continue

                outputs = [label for label, event in exports[tensor["writer"], tensor["storage"], tensor["version"]]
                           if dependencies.reaches(event, port["event"])]
                if not outputs:
                    self.finding("native_evidence", identity, scope=path, port=port["name"],
                        reason="writer action has no connected export for this value")
                    self.value_evidence_complete = False
                    continue
                consumer = key, path, "input", port["name"]
                self.values[consumer] = (writer[:2], outputs, rank, identity)

            if communication := scope.get("communication"):
                endpoint = self.endpoint(contents, index, (key, path), identity, native_events, metadata.get(index))
                self.channels[instance, communication["channel"]][rank].append(endpoint)
                endpoints.append(endpoint)

        observed = set()
        for record in raw.get("action_values", ()):
            if "batch_sync" not in record["values"]:
                continue
            parts = record["values"]["segments"]
            if (not parts or any(type(part) is not int or not 0 < part < len(raw["scopes"]) for part in parts)
                    or parts != sorted(set(parts)) or record["scope"] != parts[-1]
                        or contents.source_ids[parts[-1]] in observed):
                raise ValueError("batch synchronization requires its actual ordered segment span")
            parent = raw["scopes"][parts[0]]["parent"]
            if parts != [i for i, scope in enumerate(raw["scopes"]) if scope["parent"] == parent
                and scope["role"] == "segment"]:
                raise ValueError("batch synchronization observation omits or borrows an invocation segment")
            parts = [contents.source_ids[part] for part in parts]
            self.batch_sync.span([identities[part] for part in parts])
            identity = identities[parts[-1]]
            candidates = [endpoint for endpoint in endpoints if parts[0] <= endpoint["scope"][0][1] <= parts[-1]]
            self.batch_sync.attach(record["values"]["batch_sync"], candidates, identity, "native")
            observed.add(parts[-1])
        expected = {scope for scope, identity in identities.items()
            if self.batch_sync.terminal(identity.get("work", {}))}
        self.batch_sync.counts["native_unobserved_returns"] += len(expected - observed)

    def predicted(self, contents, identity, events, key):
        graph, paths = contents.graph, contents.paths(0)
        group = identity["group"]
        continuation = identity.get("work", {}).get("continuation") or {}
        if continuation.get("owner") == "batch_sync_exchange":
            returned = next(port["event"] for port in graph["scopes"][0]["ports"] if port["name"] == "call.return")
            self.action_returns[events[returned]] = {"key": key, "work": identity, "completed_ns": None}
        for index, path in paths.items():
            scope = graph["scopes"][index]
            if scope["role"] == "query" and scope["work"]["context"]["query"] == "event":
                returned = next(port["event"] for port in scope["ports"] if port["name"] == "call.return")
                self.query_consumers[events[returned]] = (key, path, "query", "completion"), identity
            if communication := scope.get("communication"):
                entered = next(port["event"] for port in scope["ports"] if port["name"] == "call.enter")
                rank = graph["resources"][graph["events"][entered]["resource"]]["rank"]
                if communication["rank"] != rank:
                    raise ValueError("communication endpoint belongs to another worker")
                endpoint = self.endpoint(contents, index, (key, path), identity, events, None)
                self.ready[key, path], self.complete[key, path] = endpoint["ready"], endpoint["complete"]
                self.endpoints[endpoint["scope"]] = endpoint
                self.predicted_channels[communication["channel"]][rank].append(endpoint)
                if endpoint["metadata"]:
                    returned = next(port["event"] for port in scope["ports"] if port["name"] == "call.return")
                    self.metadata_returns[events[returned]] = {"endpoint": endpoint, "completed_ns": None}
            for port in scope["ports"]:
                if port["use"] == "signal":
                    label = key, path, port["direction"], port["name"]
                    self.signal_events[label] = events[port["event"]]
                    if port["direction"] == "output":
                        if port["signal"] in self.signal_outputs:
                            raise ValueError("predicted completion generation was published twice")
                        self.signal_outputs[port["signal"]] = label
                    else:
                        self.signal_reads[label] = self.signal_outputs.get(port["signal"]), identity
                if port["use"] != "data":
                    continue
                label = key, path, port["direction"], port["name"]
                self.ports[label] = events[port["event"]]
                tensor = port["tensor"]
                if port["direction"] != "input" or 0 in tensor["shape"]:
                    continue
                writer = tensor["writer"]
                if writer is not None:
                    producer, output = (key, paths[writer]), None
                else:
                    producer, output = self.published.get((group, tensor["storage"], tensor["version"]), (None, None))
                self.reads[label] = producer, output, identity

        # Only root exports cross an invocation boundary; internal or pass-through ports cannot replace the producer.
        for port in graph["scopes"][0]["ports"]:
            if port["use"] != "data" or port["direction"] != "output":
                continue
            tensor = port["tensor"]
            if tensor["writer"] is not None:
                self.published[group, tensor["storage"], tensor["version"]] = (
                    (key, paths[tensor["writer"]]), (key, paths[0], "output", port["name"]),
                )

    def observe(self, record):
        if record["kind"] == "completion" and record["event"] in self.action_returns:
            self.action_returns[record["event"]]["completed_ns"] = record["time_ns"]
        if record["kind"] == "action_values" and "batch_sync" in record["values"]:
            parts = record["values"]["segments"]
            if not parts or len(parts) != len(set(parts)) or record["boundary"] != parts[-1]:
                raise ValueError("batch synchronization requires its own completed segments")
            span = [self.action_returns[part] for part in parts]
            if any(part["work"]["group"] != record["group"] or part["completed_ns"] is None
                   or record["time_ns"] < part["completed_ns"] for part in span):
                raise ValueError("batch synchronization result precedes its invocation return")
            self.batch_sync.span([part["work"] for part in span])
            keys = [part["key"][1] for part in span]
            if keys != sorted(set(keys)):
                raise ValueError("batch synchronization segments are out of invocation order")
            first, last = keys[0], keys[-1]
            expected = {event for event, part in self.action_returns.items()
                        if part["work"]["group"] == record["group"] and first <= part["key"][1] <= last
                        and (part["work"].get("work", {}).get("continuation")
                            or {}).get("owner") == "batch_sync_exchange"}
            if set(parts) != expected:
                raise ValueError("batch synchronization observation omits an invocation segment")
            candidates = [endpoint for (key, _), endpoint in self.endpoints.items()
                          if key[0] == record["group"] and first <= key[1] <= last]
            self.batch_sync.attach(record["values"]["batch_sync"], candidates, span[-1]["work"], "predicted")
            for part in parts:
                del self.action_returns[part]
        if record["kind"] == "completion_read" and record["boundary"] in self.query_consumers:
            consumer, identity = self.query_consumers.pop(record["boundary"])
            self.signal_reads[consumer] = self.signal_outputs[record["signal"]], identity
        elif record["kind"] == "completion" and record["event"] in self.metadata_returns:
            self.metadata_returns[record["event"]]["completed_ns"] = record["time_ns"]
        elif record["kind"] == "action_values" and "collective" in record["values"]:
            pending = self.metadata_returns.pop(record["boundary"])
            endpoint = pending["endpoint"]
            if (record["group"] != endpoint["work"]["group"] or pending["completed_ns"] is None
                    or record["time_ns"] < pending["completed_ns"]):
                raise ValueError("metadata result precedes its own completed native call")
            endpoint["values"] = record["values"]["collective"]

    def check_metadata(self, endpoints, origin):
        self.batch_sync.check(endpoints, origin)
        if not any(endpoint["metadata"] for endpoint in endpoints):
            return
        if not all(endpoint["metadata"] for endpoint in endpoints):
            self.finding("metadata_protocol", endpoints[0]["work"],
                reason="participants disagree on metadata ownership")
            return

        # A position-wise polling MIN must refer to the same ordered requests on every peer.
        queues = [PdPollingEvidence.request_order(endpoint["work"]["work"])
                  if PdPollingEvidence.in_poll(endpoint["work"]["work"]) else None for endpoint in endpoints]
        if any(queue != queues[0] for queue in queues):
            self.finding("metadata_protocol", endpoints[0]["work"], origin=origin,
                         reason="PD polling operands refer to different peer request queues")
            return
        field = f"{origin}_metadata_exchanges"
        self.counts[field] = self.counts.get(field, 0) + 1
        if any(endpoint["values"] is None for endpoint in endpoints):
            field = f"{origin}_metadata_unobserved"
            self.counts[field] = self.counts.get(field, 0) + 1
            return

        collective = endpoints[0]["work"]["work"]["context"]["collective"]
        root = collective["root"]
        if root is not None and (type(root) is not int or not 0 <= root < len(endpoints)):
            raise ValueError("metadata broadcast root is outside its native group")
        for position, endpoint in enumerate(endpoints):
            values = endpoint["values"]
            size = endpoint["elements"] if root is None or position == root else 0
            if (endpoint["work"]["work"]["context"]["collective"] != collective
                    or not isinstance(values["input"], list) or not isinstance(values["output"], list)
                    or endpoint["elements"] != endpoints[0]["elements"] or len(values["input"]) != size
                    or any(type(value) not in (bool, int, float) for row in values.values() for value in row)):
                raise ValueError("metadata values disagree with their declared scalar collective")
        operands = [endpoint["values"]["input"] for endpoint in endpoints]
        operation = collective["operation"]
        if operation == "broadcast":
            expected = operands[collective["root"]]
        elif operation == "all_gather_into_tensor":
            expected = [value for operand in operands for value in operand]
        elif operation == "all_reduce" and collective["reduction"] == "RedOpType.MIN":
            expected = [min(values) for values in zip(*operands, strict=True)]
        else:
            raise ValueError("metadata evidence uses an unsupported scalar collective")

        # Compute from each run's actual operands; no core/broker reduction or cross-run polling index is reused.
        for endpoint in endpoints:
            if endpoint["values"]["output"] != expected:
                self.finding("metadata_values", endpoint["work"], expected=expected,
                    actual=endpoint["values"]["output"], origin=origin)
                return
        field = f"{origin}_metadata_matched"
        self.counts[field] = self.counts.get(field, 0) + 1

    @staticmethod
    def mapped(label, pairs):
        key = pairs.get(label[0])
        return None if key is None else (key, *label[1:])

    def check_predicted_readiness(self, dependencies, event_ranks):
        checked, ready = 0, 0
        for consumer, (writer, output, identity) in self.reads.items():
            if writer is not None and writer[0] == consumer[0]:
                continue
            checked += 1
            if output is None:
                self.finding("data_producer", identity, scope=consumer[1], port=consumer[3],
                             origin="predicted", reason="external data input has no published storage/version producer")
                continue
            source, target = self.ports[output], self.ports[consumer]
            if not dependencies.reaches((event_ranks[source], source), (event_ranks[target], target)):
                self.finding("data_readiness", identity, scope=consumer[1], port=consumer[3], origin="predicted",
                             producer=output, reason="external data input is not gated by its own producer export")
            else:
                ready += 1
        self.counts.update(predicted_data_inputs_checked=checked, predicted_data_inputs_ready=ready)

        checked, ready = 0, 0
        for consumer, (producer, identity) in self.signal_reads.items():

            # A query observes completion; only an explicit wait requires a dependency on that generation.
            if consumer[2] == "query":
                continue
            checked += 1
            if producer is None:
                self.finding("signal_producer", identity, scope=consumer[1], port=consumer[3],
                             origin="predicted", reason="completion wait has no published generation")
            elif (self.signal_events[producer], self.signal_events[consumer]) not in self.signal_edges:
                self.finding("signal_dependency", identity, scope=consumer[1], port=consumer[3],
                             origin="predicted", reason="wait has no direct dependency on its bound completion event")
            else:
                ready += 1
        self.counts.update(predicted_signal_inputs_checked=checked, predicted_signal_inputs_ready=ready)

    def compare_signals(self, pairs):
        matched, unpaired = 0, 0
        targets = set(pairs.values())
        for native_consumer, (native_producers, identity) in self.signal_inputs.items():
            if not native_producers:
                continue
            consumer = self.mapped(native_consumer, pairs)
            producers = {self.mapped(producer, pairs) for producer in native_producers}
            read = self.signal_reads.get(consumer)
            producer = read[0] if read is not None else None
            if consumer is None or None in producers or (producer is not None and producer[0] not in targets):
                unpaired += 1
                continue
            if producer not in producers:
                self.finding("signal_producer", identity, scope=consumer[1], port=consumer[3],
                             expected_producers=sorted(producers), predicted_producer=producer)
            else:
                matched += 1
        self.counts.update(signal_inputs=len(self.signal_inputs), signal_inputs_matched=matched,
            signal_inputs_unpaired=unpaired)

    def compare_values(self, dependencies, pairs):
        matched, unpaired = 0, 0
        targets = set(pairs.values())
        for native_consumer, (native_writer, native_outputs, rank, identity) in self.values.items():
            consumer, writer = self.mapped(native_consumer, pairs), self.mapped(native_writer, pairs)
            outputs = [self.mapped(output, pairs) for output in native_outputs]
            read = self.reads.get(consumer)
            predicted_writer = read[0] if read is not None else None
            if (consumer is None or writer is None or None in outputs
                    or (predicted_writer is not None and predicted_writer[0] not in targets)):
                unpaired += 1
                continue
            if predicted_writer != writer:
                self.finding("data_producer", identity, scope=consumer[1], port=consumer[3],
                             expected_writer=writer, predicted_writer=predicted_writer)
            elif consumer not in self.ports or any(output not in self.ports for output in outputs):
                self.finding("data_readiness", identity, scope=consumer[1], port=consumer[3],
                    reason="mapped data boundary is absent")
            elif any(not dependencies.reaches((rank, self.ports[output]), (rank, self.ports[consumer]))
                for output in outputs):
                self.finding("data_readiness", identity, scope=consumer[1], port=consumer[3],
                    reason="consumer is not gated by its producer exports")
            else:
                matched += 1
        self.counts.update(data_inputs=len(self.values), data_inputs_matched=matched, data_inputs_unpaired=unpaired,
                           data_evidence_complete=self.value_evidence_complete)

    def edge_details(self, edges):
        result = []
        for source, target in sorted(edges)[:3]:
            record = {}
            for name, event, ports in (("source", source, self.ready), ("target", target, self.complete)):
                record[name] = {"event": event,
                    "ports": [{"invocation": {"group": key[0], "event": key[1]}, "scope": path}
                                                          for (key, path), value in ports.items() if value == event]}
            result.append(record)
        return result

    def exchanges(self, channels, dependencies, category):
        for channel, ranks in channels.items():
            first = next(iter(ranks.values()))[0]
            members = first["members"]
            if set(ranks) != set(members) or len({len(values) for values in ranks.values()}) != 1:
                self.finding(category, first["work"], channel=channel,
                    reason="communication participants or call counts are incomplete")
                yield None
                continue
            ordered = []
            valid = True
            for rank in members:
                calls = sorted(ranks[rank],
                    key=lambda value: (dependencies[rank].positions[value["entered"]], value["order"]))

                # Protocol order is run-local: never align separate executions by a polling ordinal.
                if any(not dependencies[rank].reaches(left["entered"], right["entered"]) for left,
                    right in pairwise(calls)):
                    self.finding(category, first["work"], channel=channel,
                        reason="calls have no established protocol order")
                    valid = False
                ordered.append(calls)
            if not valid:
                yield None
                continue

            for endpoints in zip(*ordered, strict=True):
                if any(value["members"] != members or value["operation"] != endpoints[0]["operation"]
                       or not set(value["required"]) <= set(members) for value in endpoints):
                    self.finding(category, first["work"], channel=channel,
                        reason="peers disagree on operation or members")
                    yield None
                else:
                    yield endpoints

    def compare_edges(self, category, expected, unresolved):
        missing, extra = expected - self.communication_edges, self.communication_edges - expected

        # Unobserved peers can explain extra target edges. A proven prerequisite remains mandatory.
        for name, edges in (("missing", missing), ("extra", extra)):
            if edges and (name == "missing" or not unresolved):
                self.finding(category, {}, **{name: len(edges)}, examples=self.edge_details(edges))
        return not (unresolved or missing or extra)

    def compare_predicted_communication(self, dependencies):
        expected, exchanges, unresolved = set(), 0, False
        for endpoints in self.exchanges(self.predicted_channels, dependencies, "communication_protocol"):
            if endpoints is None:
                unresolved = True
                continue
            exchanges += 1
            self.check_metadata(endpoints, "predicted")
            peers = {endpoint["rank"]: endpoint for endpoint in endpoints}
            for endpoint in endpoints:
                destination = self.complete.get(endpoint["scope"])
                for rank in endpoint["required"]:
                    source = self.ready.get(peers[rank]["scope"])
                    if source is None or destination is None:
                        self.finding("communication_endpoint", endpoint["work"], scope=endpoint["scope"][1],
                                     reason="declared communication boundary is absent")
                        unresolved = True
                    else:
                        expected.add((source, destination))

        complete = self.compare_edges("communication_protocol", expected, unresolved)
        self.counts.update(predicted_communication_exchanges=exchanges,
                           predicted_communication_edges_expected=len(expected),
                           predicted_communication_edges_matched=len(expected & self.communication_edges),
                           predicted_communication_protocol_complete=complete)

    def native_communication(self, dependencies):
        self.native_rounds = list(self.exchanges(self.channels, dependencies, "native_evidence"))
        complete = True
        exchanges, row_pairs = 0, 0
        for endpoints in self.native_rounds:
            if endpoints is None:
                complete = False
                continue
            exchanges += 1
            self.check_metadata(endpoints, "native")
            if endpoints[0]["operation"] == "transfer":
                send, receive = (endpoint["payload"] for endpoint in endpoints)
                if (send[0], receive[0]) != ("input", "output") or send[1:] != receive[1:]:
                    complete = False
                    self.finding("communication_payload", endpoints[0]["work"], origin="native", send=send,
                        receive=receive)
            peers = {endpoint["rank"]: endpoint for endpoint in endpoints}
            for endpoint in endpoints:

                # Actual send/receive splits must agree across peers, independently of target route derivation.
                if traffic := endpoint.get("peer_rows"):
                    for index, peer in enumerate(endpoints):
                        incoming = peer.get("peer_rows")
                        if incoming is None:
                            complete = False
                            self.finding("native_evidence", endpoint["work"],
                                reason="peer exchange has no observed row counts", peer=peer["rank"])
                        elif traffic["send"][index] != incoming["receive"][peer["members"].index(endpoint["rank"])]:
                            complete = False
                            self.finding("communication_payload", endpoint["work"],
                                reason="native send and receive row counts disagree", peer=peer["rank"])
                        else:
                            row_pairs += 1

                for rank in endpoint["required"]:
                    self.native_communication_edges.add(((rank, peers[rank]["ready"]),
                        (endpoint["rank"], endpoint["complete"])))
        self.counts.update(communication_channels=len(self.channels), communication_exchanges=exchanges,
                           native_communication_protocol_complete=complete, native_peer_row_pairs=row_pairs)

    def compare_communication(self, pairs):
        expected, unresolved, unpaired = set(), False, 0
        payloads, payloads_compared, payloads_matched = 0, 0, 0
        for endpoints in self.native_rounds:
            if endpoints is None:
                unresolved = True
                continue
            peers = {endpoint["rank"]: endpoint for endpoint in endpoints}
            for endpoint in endpoints:
                destination_scope = self.mapped(endpoint["scope"], pairs)
                destination = self.complete.get(destination_scope)
                target = self.endpoints.get(destination_scope)
                if endpoint["payload"] is not None:
                    payloads += 1
                    if target is not None:
                        payloads_compared += 1
                        if endpoint["payload"] == target["payload"]:
                            payloads_matched += 1
                        else:
                            self.finding("communication_payload", endpoint["work"], native=endpoint["payload"],
                                predicted=target["payload"])
                if target is not None:
                    if endpoint["metadata"] and endpoint["values"] is not None and target["values"] is not None:
                        field = "metadata_operands_compared"
                        self.counts[field] = self.counts.get(field, 0) + 1
                        if endpoint["values"] != target["values"]:
                            self.finding("metadata_operands", endpoint["work"], native=endpoint["values"],
                                predicted=target["values"])
                    if "batch_sync" in endpoint and "batch_sync" in target:
                        self.batch_sync.counts["endpoints_compared"] += 1
                        if endpoint["batch_sync"] != target["batch_sync"]:
                            self.finding("batch_sync_operands", endpoint["work"], native=endpoint["batch_sync"],
                                predicted=target["batch_sync"])
                for rank in endpoint["required"]:
                    source_scope = self.mapped(peers[rank]["scope"], pairs)
                    if source_scope is None or destination_scope is None:
                        unpaired += 1
                        unresolved = True
                        continue
                    source = self.ready.get(source_scope)
                    if source is None or destination is None:
                        self.finding("communication_endpoint", endpoint["work"], scope=endpoint["scope"][1],
                            reason="mapped communication boundary is absent")
                        unresolved = True
                    else:
                        expected.add((source, destination))

        self.compare_edges("communication_dependency", expected, unresolved)
        self.counts.update(communication_edges_expected=len(expected),
            communication_edges_observed=len(self.communication_edges),
                           communication_edges_matched=len(expected & self.communication_edges),
                               communication_edges_unpaired=unpaired,
                           communication_evidence_complete=not unresolved, device_transfer_payloads=payloads,
                           device_transfer_payloads_compared=payloads_compared,
                               device_transfer_payloads_matched=payloads_matched)


@dataclass(frozen=True, slots=True)
class NativeCpuCall:
    key: tuple
    phase: str
    thread: int
    entered: tuple[int, int]
    posted: tuple[int, int] | None
    returned: tuple[int, int]
    counters: tuple[int, int]
    execution: dict
    descriptor: dict

    def identity(self):
        work = self.execution["work"]
        context = work["context"]
        normalized = {
            **work,
            "context": {**context, "cpu_transfer": framework_evidence().cpu_transfer.identity(context["cpu_transfer"])},
        }
        return {"operation": self.execution["operation"], "work": ActionIdentity.work_identity(normalized)}

    def source(self):
        return {"post": list(self.key), "phase": self.phase}

    def services(self):
        operation = self.execution["operation"]
        if self.posted is None:
            return ((self.entered[1], self.returned[1], operation),)
        before, after = framework_evidence().cpu_protocol.post_operations(operation)
        return ((self.entered[1], self.posted[1], before), (self.posted[1], self.returned[1], after))


class CpuProtocolEvidence:
    def __init__(self, capture, owners):
        self.capture = capture
        self.counts, self.findings = Counter(), FindingLedger()
        self.native_threads = defaultdict(lambda: defaultdict(list))
        self.native_calls, self.native_times = defaultdict(list), {}
        self.native_groups, self.predicted_groups, self.actors = owners, {}, {}
        self.messages = {origin: defaultdict(dict) for origin in ("native", "predicted")}
        self.points = {origin: set() for origin in self.messages}
        self.edges = {origin: set() for origin in self.messages}
        self.last_cpu, self.returns, self.horizons = {}, {}, {}
        self.services, self.service_returns = {}, {}
        self.posts, self.post_times = set(), {}
        self.readiness, self.ready_events = {}, set()
        self.ready_inputs = defaultdict(dict)
        self.ready_bounds = {}
        self.service_gates = set()
        self.communication_edges = set()
        self.finished_services, self.finished_ready = set(), set()
        self.terminals = []
        self.terminals_checked = False
        self.sequence = Counter()
        self.predicted_scope = False

    def missing(self, reason, **detail):
        self.counts[reason] += 1
        self.findings.add("cpu_protocol_evidence", {}, {"reason": reason, **detail})

    def point(self, origin, rank):
        self.sequence[origin, rank] -= 1
        point = rank, self.sequence[origin, rank]
        self.points[origin].add(point)
        return point

    def native_graph(self, rank, graph):
        for event, (node, evidence) in enumerate(zip(graph["events"], graph["evidence"]["events"], strict=True)):
            resource = graph["resources"][node["resource"]]
            if resource["kind"] == "thread":
                thread = int(resource["identity"].removeprefix("thread/"))
                positions = self.native_threads[rank, thread]
                positions[evidence["order"]].append((rank, event))
        self.read_native_calls(rank)

    @staticmethod
    def execution(interval, operation):
        execution = interval["execution"]
        if (execution["operation"] != operation or type(execution["in_flow"]) is not bool
                or not isinstance(execution["work"], dict)):
            raise ValueError("CPU protocol interval changes its native operation or entering context")
        return execution

    def native_interval(self, rank, key, phase, interval, descriptor, transport=None):
        begin, end = interval["begin_counter"], interval["end_counter"]
        entered, returned = interval["begin_raw_ns"], interval["end_raw_ns"]
        if (
            any(type(value) is not int for value in (begin, end, entered, returned))
            or begin > end
            or entered > returned
        ):
            raise ValueError("CPU protocol has an invalid native interval")
        thread = interval["thread"]
        positions = self.native_threads[rank, thread]
        endpoints = []
        for counter, stamp in ((begin, entered), (end, returned)):
            if any(event >= 0 for _, event in positions[counter]):
                self.missing("native_protocol_counter_ties_coarse_boundary", rank=rank, thread=thread, counter=counter)
            point = self.point("native", rank)
            positions[counter].append(point)
            self.native_times[point] = stamp
            endpoints.append(point)
        first, last = endpoints
        posted = None
        if transport is None:
            self.edges["native"].add((first, last))
        else:
            posted = self.point("native", rank)
            self.native_times[posted] = transport["post_raw_ns"]
            self.edges["native"].update(((first, posted), (posted, last)))
        call = NativeCpuCall(key, phase, thread, first, posted, last, (begin, end), interval["execution"], descriptor)
        self.native_calls[rank].append(call)
        return call

    def post(self, origin, rank, key, descriptor, posted):
        group, source, target, tag, position = key
        route = group, source, target, tag
        if (route != tuple(descriptor[name] for name in ("group", "source", "target", "tag"))
                or source == target or rank != (target if descriptor["receiving"] else source)
                or type(descriptor["receiving"]) is not bool
                or type(position) is not int or position < 0
                or any(type(size) is not int or size < 0 for size in descriptor["shape"])
                or type(descriptor["elements"]) is not int or descriptor["elements"] != prod(descriptor["shape"])
                or type(descriptor["element_bytes"]) is not int or descriptor["element_bytes"] <= 0):
            raise ValueError("CPU protocol post has inconsistent route, actor or tensor geometry")
        peers = self.messages[origin][key]
        if rank in peers:
            raise ValueError("CPU protocol repeats a local post identity")
        wire = tuple(descriptor[name] for name in ("dtype", "elements", "element_bytes"))
        peers[rank] = {"posted": posted, "consumed": None, "wire": wire}
        self.counts[origin + "_posts"] += 1
        return peers[rank]

    def native_transport(self, post):
        transport = framework_evidence().cpu_protocol.transport(post)
        if transport is None:
            self.counts["native_unobserved_transport_posts"] += 1
        else:
            self.counts["native_transport_posts"] += 1
            self.counts["native_transport_completions"] += transport["completion_raw_ns"] is not None
        return transport

    def native_readiness(self, peers):
        if len(peers) != 2 or any(peer["transport"] is None for peer in peers.values()):
            return None
        last_post = max(peer["transport"]["post_raw_ns"] for peer in peers.values())
        windows = {
            rank: framework_evidence().cpu_protocol.ready_window(
                peer["transport"], last_post, None if peer["consumed"] is None else self.native_times[peer["consumed"]]
            )
            for rank, peer in peers.items()
        }
        return last_post, windows

    def read_native_calls(self, rank):
        path = self.capture / "protocol" / f"rank-{rank}.json"
        if not path.is_file():
            self.missing("missing_native_protocol", rank=rank)
            return
        document = read_json(path)
        if document["backend"] != "gloo" or set(document["operations"]) != {"send", "recv", "isend", "irecv"}:
            raise ValueError("CPU protocol observation declares a different operation scope")
        transport_clock = document.get("transport_clock")
        if transport_clock is not None and transport_clock != "monotonic_raw_ns":
            raise ValueError("Gloo transport observation requires the shared RAW clock")
        start = {tuple(row["route"]): row["position"] for row in document["start_positions"]}
        end = {tuple(row["route"]): row["position"] for row in document["end_positions"]}
        observed = Counter()
        posting_threads = defaultdict(set)
        for route, floor in start.items():
            if not 0 <= floor <= end[route]:
                raise ValueError("CPU protocol has a reversed observation horizon")
        self.horizons[rank] = {route: (start.get(route, 0), limit) for route, limit in end.items()}
        for post in document["posts"]:
            key = tuple(post["key"])
            route, position = key[:-1], key[-1]
            posting_threads[route].add(post["interval"]["thread"])
            if not position < end[route]:
                raise ValueError("CPU protocol post lies outside its declared channel history")
            source, target = key[1:3]
            if (
                self.native_groups[source] != self.native_groups[target]
                or self.native_groups[rank] != self.native_groups[source]
            ):
                raise ValueError("CPU protocol crosses independent native worlds")
            operation = post["operation"]
            if operation not in {"send", "recv", "isend", "irecv"} or post["descriptor"]["receiving"] != (
                operation in {"recv", "irecv"}
            ):
                raise ValueError("CPU protocol operation disagrees with its receive/send descriptor")
            self.execution(post["interval"], operation)
            descriptor = post["descriptor"]
            transport = self.native_transport(post) if transport_clock is not None else None
            call = self.native_interval(rank, key, "post", post["interval"], descriptor, transport)
            endpoint = self.post("native", rank, key, descriptor, call.posted or call.entered)
            endpoint["transport"] = transport
            if post["consumed"] is not None:
                if operation in {"isend", "irecv"}:
                    self.execution(post["consumed"], operation + "_wait")
                    wait = self.native_interval(rank, key, "wait", post["consumed"], descriptor)
                    if post["consumed"]["begin_counter"] < post["interval"]["end_counter"]:
                        raise ValueError("CPU Work wait starts before its post returns")
                    self.edges["native"].add((call.returned, wait.entered))
                    consumed = wait.returned
                else:

                    # Blocking operations consume the instrumented call itself.
                    if post["consumed"] != post["interval"]:
                        raise ValueError("blocking CPU work disagrees with its recorded return")
                    consumed = call.returned
                endpoint["consumed"] = consumed
            observed[route] += position >= start.get(route, 0)
        for route, limit in end.items():
            if observed[route] != limit - start.get(route, 0):
                self.missing("native_posts_missing_from_horizon", rank=rank, route=route)
            if len(posting_threads[route]) > 1:
                self.missing("native_channel_has_multiple_posting_threads", rank=rank, route=route)

    def observe(self, row):
        kind = row["kind"]
        if kind == "resources":
            self.predicted_groups = {group["id"]: (group["rank"], instance["id"])
                                     for instance in row["instances"] for group in instance["groups"]}
        elif kind == "graph":
            graph, events = row["graph"], row["event_ids"]
            ports = {port["name"]: events[port["event"]] for port in graph["scopes"][0]["ports"]
                     if port["name"] in {"call.enter", "call.return", "transport.post"}}
            entered = next(port for port in graph["scopes"][0]["ports"] if port["name"] == "call.enter")
            rank = graph["resources"][graph["events"][entered["event"]]["resource"]]["rank"]
            if rank in self.last_cpu:
                self.edges["predicted"].add((self.last_cpu[rank], (rank, ports["call.enter"])))
            if graph["scopes"][0]["role"] == "cpu_service":
                self.service(graph, events, rank, ports)
            else:
                self.returns[ports["call.return"]] = rank
        elif kind == "completion":
            event = row["event"]
            if event in self.service_returns:
                rank, _, _ = self.service_returns[event]
                self.finished_services.add(event)
                self.last_cpu[rank] = rank, event
            if event in self.posts:
                self.post_times[event] = row["time_ns"]
            if event in self.ready_events:
                self.finished_ready.add(event)
        elif kind == "edge":
            edge = row["source"], row["target"]
            if row["target"] in self.ready_events:
                if row["relation"] != "communication" or row["source"] in self.ready_inputs[row["target"]]:
                    raise ValueError("CPU readiness has a duplicate or noncommunication prerequisite")
                self.ready_inputs[row["target"]][row["source"]] = row["delay_ns"]
                self.communication_edges.add(edge)
            if edge in self.service_gates:
                if row["relation"] != "communication" or row["delay_ns"] != 0:
                    raise ValueError("CPU API return changes its local readiness dependency")
                self.service_gates.remove(edge)
                self.communication_edges.add(edge)
        elif kind == "boundary_time" and row["event"] in self.ready_events:
            self.ready_bounds[row["event"]] = row["begin_ns"], row["end_ns"]
        elif kind == "action_values" and "outputs" in row["values"]:
            rank, _ = self.predicted_groups[row["group"]]
            if self.returns.pop(row["boundary"]) != rank:
                raise ValueError("CPU protocol action return belongs to another participant")
            self.last_cpu[rank] = rank, row["boundary"]
        elif kind == "native_protocol":
            record, group = row["record"], row["group"]
            if record["phase"] == "scope":
                if (
                    self.predicted_scope
                    or record["backend"] != "gloo"
                    or set(record["operations"]) != {"send", "recv", "isend", "irecv"}
                ):
                    raise ValueError("CPU broker observation has a competing or different protocol scope")
                self.actors = {actor["rank"]: actor["group"] for actor in record["participants"]}
                if (len(self.actors) != len(record["participants"])
                        or set(self.actors.values()) != set(self.predicted_groups)
                        or len(self.actors) != len(self.predicted_groups)):
                    raise ValueError("CPU broker observation omits or repeats a native participant")
                self.predicted_scope = True
                return
            rank, _ = self.predicted_groups[group]
            actor = record["rank"]
            if not self.predicted_scope or self.actors[actor] != group:
                raise ValueError("CPU broker actor changes its native participant")
            key = tuple(record["key"])
            phase = record["phase"]
            if phase == "readiness":
                if (key, rank) in self.readiness or type(record["event"]) is not int:
                    raise ValueError("CPU protocol repeats or omits its local readiness event")
                self.readiness[key, rank] = (record["event"], False)
                self.ready_events.add(record["event"])
                return
            if phase == "ready":
                self.predicted_ready(key, rank, actor, record, row["time_ns"])
                return
            execution = record["execution"]
            operation = execution["operation"]
            self.execution(record, operation)
            if record["event"] is None:
                if execution["in_flow"]:
                    raise ValueError("CPU serving operation has no execution service")
                point = self.point("predicted", rank)
                if rank in self.last_cpu:
                    self.edges["predicted"].add((self.last_cpu[rank], point))
                self.last_cpu[rank] = point
            else:
                point = rank, record["event"]
            if phase == "post":
                if self.predicted_groups[self.actors[key[1]]][1] != self.predicted_groups[self.actors[key[2]]][1]:
                    raise ValueError("CPU broker P2P crosses independent native worlds")
                if (operation not in {"send", "recv", "isend", "irecv"}
                        or record["descriptor"]["receiving"] != (operation in {"recv", "irecv"})):
                    raise ValueError("CPU post changes its native operation")
                posted = point
                if record["event"] is not None:
                    expected_rank, expected_key, descriptor, work, boundary = self.services.pop(record["event"])
                    if (expected_rank, expected_key, descriptor, work) != (rank, key, record["descriptor"], execution):
                        raise ValueError("CPU post differs from its submitted service")
                    posted = rank, boundary
                endpoint = self.post("predicted", actor, key, record["descriptor"], posted)
                endpoint.update(time_ns=row["time_ns"], execution=execution)
            elif phase == "consumed":
                peers = self.messages["predicted"][key]
                if len(peers) != 2 or "ready_ns" not in peers[actor] or peers[actor]["ready_ns"] > row["time_ns"]:
                    raise ValueError("CPU broker consumption precedes the matching peer post")
                endpoint = peers[actor]
                if endpoint["consumed"] is not None:
                    raise ValueError("CPU Work consumption is recorded twice")
                post_operation = endpoint["execution"]["operation"]
                if operation != post_operation + ("_wait" if post_operation in {"isend", "irecv"} else ""):
                    raise ValueError("CPU consumption changes its posted operation")
                if record["event"] is not None:
                    if record["event"] not in self.finished_services or self.service_returns[record["event"]] != (
                        rank,
                        key,
                        execution,
                    ):
                        raise ValueError("CPU consumption precedes or changes its own service return")
                endpoint["consumed"] = point
            else:
                raise ValueError("CPU protocol has an unknown observation phase")

    def predicted_ready(self, key, rank, actor, record, stamp):
        peers = self.messages["predicted"][key]
        if len(peers) != 2 or "ready_ns" in peers[actor]:
            raise ValueError("CPU readiness is repeated or lacks either post")
        parents = {peer["posted"][1] for peer in peers.values() if peer["posted"][1] >= 0}
        event, service = record["event"], record["service_ns"]
        if (service is not None and (type(service) is not int or service < 0)) or set(
            record["prerequisites"]
        ) != parents:
            raise ValueError("CPU readiness changes its post prerequisites or service duration")
        if event is None:
            if parents or service != 0:
                raise ValueError("Untimed CPU readiness includes modeled transfer work")
        else:
            if self.readiness[key, rank] != (event, False) or event not in self.finished_ready:
                raise ValueError("CPU readiness was published before its declared event completed")
            delay = 0 if service is None else service
            if self.ready_inputs[event] != dict.fromkeys(parents, delay):
                raise ValueError("CPU readiness differs from the executor's actual dependency edges")
            if service is None and event not in self.ready_bounds:
                raise ValueError("Unknown CPU readiness has no explicit completion condition")
            posted = [self.post_times[parent] for parent in parents]
            if posted and stamp < max(posted) + delay:
                raise ValueError("CPU readiness completed before its supplied service lower bound")
            self.readiness[key, rank] = (event, True)
        peers[actor]["ready_ns"] = stamp
        self.counts["predicted_ready_endpoints"] += 1

    def service(self, graph, events, rank, ports):
        root = graph["scopes"][0]
        binding = root["bindings"]["cpu_protocol"]
        key, descriptor, execution = tuple(binding["key"]), binding["descriptor"], binding["execution"]
        operation = execution["operation"]
        self.execution(binding, operation)
        posting = operation in {"send", "recv", "isend", "irecv"}
        if not posting and operation not in {"isend_wait", "irecv_wait"}:
            raise ValueError("CPU service declares an unobserved protocol operation")
        number = 2 if posting else 1
        expected_ports = {"call.enter": events[0], "call.return": events[number]}
        if posting:
            expected_ports["transport.post"] = events[1]
        work = [{"scope": 0, "resource": 0, "kind": "host", "start": first, "finish": first + 1}
                for first in range(number)]
        if (len(graph["scopes"]) != 1 or len(graph["resources"]) != 1 or len(graph["events"]) != number + 1
                or ports != expected_ports
                or graph["resources"][0]["kind"] != "thread" or graph["resources"][0]["role"] != "control"
                or not execution["in_flow"] or root["work"] != execution["work"]
                or graph["work"] != work
                or root["kv_accesses"] or root["slot_accesses"] or graph["edges"]):
            raise ValueError("CPU service changes its host work or protocol boundaries")
        if posting:
            self.services[events[0]] = rank, key, descriptor, execution, events[1]
            self.posts.add(events[1])
        if operation not in {"isend", "irecv"} and (key, rank) in self.readiness:

            # Check actual executor edges, not only the service's claimed prerequisites or observed timestamps.
            self.service_gates.add((self.readiness[key, rank][0], events[number]))
        self.service_returns[events[number]] = rank, key, execution
        self.counts["predicted_service_intervals"] += number

    def connect_messages(self, origin):
        for key, peers in self.messages[origin].items():
            self.counts[origin + "_consumptions"] += sum(peer["consumed"] is not None for peer in peers.values())
            self.counts[origin + "_unconsumed_posts"] += sum(peer["consumed"] is None for peer in peers.values())
            if len({peer["wire"] for peer in peers.values()}) != 1:
                raise ValueError("CPU protocol endpoints disagree on wire dtype or element count")
            if len(peers) == 2:
                transports = [row["transport"] for row in peers.values() if row.get("transport") is not None]
                if len(transports) == 2:
                    if transports[0]["slot"] != transports[1]["slot"]:
                        raise ValueError("Gloo transport endpoints disagree on the native message slot")
                    self.counts["native_transport_pairs"] += 1
                self.counts[origin + "_messages"] += 1
                for destination in peers.values():
                    if destination["consumed"] is not None:

                        # Both posts precede either completion; the peer need not have consumed its Work.
                        for source in peers.values():
                            if source["posted"] != destination["consumed"]:
                                self.edges[origin].add((source["posted"], destination["consumed"]))
                continue
            actor, endpoint = next(iter(peers.items()))
            if endpoint["consumed"] is None:
                self.counts[origin + "_unconsumed_tail_posts"] += 1
                continue
            peer = key[2] if actor == key[1] else key[1]
            first, last = self.horizons.get(peer, {}).get(key[:-1], (0, 0))
            if origin == "native" and peer in self.horizons and key[-1] < first:
                self.counts["native_initial_external_completions"] += 1
            elif origin == "native" and peer in self.horizons and key[-1] >= last:
                self.terminals.append((key, endpoint["consumed"]))
                self.counts["native_terminal_external_completions"] += 1
            else:
                self.missing(origin + "_missing_peer_post", key=key, actor=actor)

    def native(self, native, dependencies):
        for (rank, thread), positions in self.native_threads.items():
            if rank in native and positions and all(event < 0 for points in positions.values() for _, event in points):
                self.missing("native_protocol_thread_has_no_coarse_anchor", rank=rank, thread=thread)

            # Equal clock readings do not identify an event or order two different events.
            previous = ()
            for _, current in sorted(positions.items()):
                self.edges["native"].update((source, target) for source in previous for target in current)
                previous = current
        self.connect_messages("native")
        if self.findings:
            return None
        graph = self.joined_graph("native", native, dependencies)

        # An unobserved suffix is harmless only if it cannot lead back to any retained serving event.
        relevant = [False] * len(graph.positions)
        for rank, original in native.items():
            for event in original.next:
                relevant[graph.positions[rank, event]] = True
        for position in range(len(relevant) - 1, -1, -1):
            relevant[position] |= any(relevant[target] for target in graph.successors[position])
        for key, consumed in self.terminals:
            if relevant[graph.positions[consumed]]:
                self.missing("native_terminal_message_can_reach_serving", key=key)
            else:
                self.counts["native_terminal_outside_serving"] += 1
        self.terminals_checked = True
        return graph

    def joined_graph(self, origin, graphs, dependencies):
        graph = CausalIndex.joined(graphs, dependencies)
        graph.add_events(self.points[origin])
        for source, target in self.edges[origin]:
            if source != target:
                graph.connect(source, target, "communication" if source[0] != target[0] else "control")
        graph.prepare()
        return graph

    def prediction(self, predicted, predicted_edges, *, complete):
        if not self.predicted_scope:
            self.missing("missing_predicted_protocol_scope")
        pending = sum(not published for _, published in self.readiness.values())
        if self.services or self.service_gates or self.service_returns.keys() != self.finished_services or pending:
            if complete:
                raise ValueError("CPU protocol has an unobserved service or unfinished readiness")

            # A stopped prefix can still have live work; it cannot qualify the missing protocol paths.
            self.missing("unfinished_predicted_protocol", services=len(self.services), gates=len(self.service_gates),
                         returns=len(self.service_returns.keys() - self.finished_services),
                         pending_readiness=pending)
        self.connect_messages("predicted")

        # Keep each run's observed paths available; protocol completeness gates cross-run negative conclusions.
        return self.joined_graph("predicted", predicted, predicted_edges)

    def compare_wire_payloads(self, pairs):
        if len(set(pairs.values())) != len(pairs):
            raise ValueError("multiple predicted CPU messages use the same native post witness")
        routes, differences = defaultdict(Counter), []
        for predicted_key, native_key in pairs.items():
            native = self.messages["native"][native_key]
            predicted = self.messages["predicted"].get(predicted_key, {})
            if len(native) != 2 or len(predicted) != 2:
                continue

            # The witness identifies the message. Each run supplies its own wire descriptor.
            _, source, target, _, _ = native_key
            execution = next(peer["execution"] for peer in predicted.values() if peer["execution"]["in_flow"])
            channel = execution["work"]["context"]["cpu_transfer"]["group"]
            counts = routes[channel, source, target]
            expected, actual = native[source]["wire"], predicted[source]["wire"]
            counts["checked"] += 1
            counts["matched"] += expected == actual
            if expected != actual:
                differences.append({"channel": channel, "source": source, "target": target,
                                    "native_post": native_key, "predicted_post": predicted_key,
                                    "native": self.payload(expected), "predicted": self.payload(actual)})

        checked = sum(counts["checked"] for counts in routes.values())
        return {"witnessed": len(pairs), "checked": checked, "matched": checked - len(differences),
                "unresolved": len(pairs) - checked,
                "routes": [{"channel": channel, "source": source, "target": target, **counts}
                           for (channel, source, target), counts in sorted(routes.items())]}, differences

    @staticmethod
    def payload(wire):
        dtype, elements, element_bytes = wire
        return {"dtype": dtype, "elements": elements, "element_bytes": element_bytes, "bytes": elements * element_bytes}

    def report(self):
        return {
            "complete": not self.findings and not self.returns and (not self.terminals or self.terminals_checked),
            "unreturned_actions": len(self.returns),
            **self.counts,
            "scope": (
                "Observed CPU posts, readiness and consumption within each run. Native call boundaries and "
                "predicted action returns anchor the protocol paths. Prefix posts and unconsumed tails have "
                "explicit counts. Cross-run payloads use checked post witnesses when available."
            ),
        }
