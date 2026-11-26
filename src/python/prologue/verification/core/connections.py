"""Track value exports and completion generations across paired invocations."""

from collections import defaultdict

from prologue.verification.core.action_identity import ActionIdentity
from prologue.verification.core.device_communication import CommunicationEvidence
from prologue.verification.engine import FindingLedger


class Connections:
    def __init__(self):
        self.values, self.reads, self.published = {}, {}, {}
        self.ports = {}
        self.signal_inputs, self.signal_reads, self.signal_outputs = {}, {}, {}
        self.signal_events, self.signal_edges = {}, set()
        self.query_consumers = {}
        self.findings = FindingLedger()
        self.counts = {}
        self.communication = CommunicationEvidence(self.counts, self.finding)
        self.value_evidence_complete = True

    def finding(self, category, identity, **detail):
        self.findings.add(category, identity, detail)

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

        self.native_signals(raw, contents, owners)

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
                        ((key, path, "output", port["name"]), port["event"])
                    )

        for index, (key, path, identity) in owners.items():
            scope = graph["scopes"][index]
            for port in scope["ports"]:
                if port["use"] != "data" or port["direction"] != "input" or 0 in port["tensor"]["shape"]:
                    continue
                tensor = port["tensor"]
                writer = owners.get(tensor["writer"])

                if writer is None:
                    self.finding(
                        "native_evidence",
                        identity,
                        scope=path,
                        port=port["name"],
                        reason="data writer has no observed action",
                    )
                    self.value_evidence_complete = False
                    continue

                if writer[0] == key:
                    continue

                outputs = [
                    label
                    for label, event in exports[tensor["writer"], tensor["storage"], tensor["version"]]
                    if dependencies.reaches(event, port["event"])
                ]

                if not outputs:
                    self.finding(
                        "native_evidence",
                        identity,
                        scope=path,
                        port=port["name"],
                        reason="writer action has no connected export for this value",
                    )
                    self.value_evidence_complete = False
                    continue

                consumer = key, path, "input", port["name"]
                self.values[consumer] = (writer[:2], outputs, rank, identity)

            if communication := scope.get("communication"):
                endpoint = self.communication.endpoint(
                    contents, index, (key, path), identity, native_events, metadata.get(index)
                )
                self.communication.channels[instance, communication["channel"]][rank].append(endpoint)
                endpoints.append(endpoint)

        self.communication.native_batch_returns(raw, contents, identities, endpoints)

    def native_signals(self, raw, contents, owners):
        graph = contents.graph

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
                self.finding(
                    "native_evidence",
                    identity,
                    scope=label[1],
                    port=label[3],
                    reason="completion handle has no observed producer",
                )

            self.signal_inputs[label] = producers, identity

    def predicted(self, contents, identity, events, key):
        graph, paths = contents.graph, contents.paths(0)
        group = identity["group"]
        self.communication.begin_action(graph, identity, events, key)

        for index, path in paths.items():
            scope = graph["scopes"][index]
            if scope["role"] == "query" and scope["work"]["context"]["query"] == "event":
                returned = next(port["event"] for port in scope["ports"] if port["name"] == "call.return")
                self.query_consumers[events[returned]] = (key, path, "query", "completion"), identity

            if "communication" in scope:
                self.communication.add_predicted(contents, index, key, path, identity, events)

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
                    (key, paths[tensor["writer"]]),
                    (key, paths[0], "output", port["name"]),
                )

    def observe(self, record):
        self.communication.observe(record)
        if record["kind"] == "completion_read" and record["boundary"] in self.query_consumers:
            consumer, identity = self.query_consumers.pop(record["boundary"])
            self.signal_reads[consumer] = self.signal_outputs[record["signal"]], identity

    def check_predicted_readiness(self, dependencies, event_ranks):
        checked, ready = 0, 0
        for consumer, (writer, output, identity) in self.reads.items():
            if writer is not None and writer[0] == consumer[0]:
                continue
            checked += 1

            if output is None:
                self.finding(
                    "data_producer",
                    identity,
                    scope=consumer[1],
                    port=consumer[3],
                    origin="predicted",
                    reason="external data input has no published storage/version producer",
                )
                continue

            source, target = self.ports[output], self.ports[consumer]
            if not dependencies.reaches((event_ranks[source], source), (event_ranks[target], target)):
                self.finding(
                    "data_readiness",
                    identity,
                    scope=consumer[1],
                    port=consumer[3],
                    origin="predicted",
                    producer=output,
                    reason="external data input is not gated by its own producer export",
                )
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
                self.finding(
                    "signal_producer",
                    identity,
                    scope=consumer[1],
                    port=consumer[3],
                    origin="predicted",
                    reason="completion wait has no published generation",
                )
            elif (self.signal_events[producer], self.signal_events[consumer]) not in self.signal_edges:
                self.finding(
                    "signal_dependency",
                    identity,
                    scope=consumer[1],
                    port=consumer[3],
                    origin="predicted",
                    reason="wait has no direct dependency on its bound completion event",
                )
            else:
                ready += 1

        self.counts.update(predicted_signal_inputs_checked=checked, predicted_signal_inputs_ready=ready)

    def compare_signals(self, pairs):
        matched, unpaired = 0, 0
        targets = set(pairs.values())
        for native_consumer, (native_producers, identity) in self.signal_inputs.items():
            if not native_producers:
                continue
            consumer = ActionIdentity.mapped(native_consumer, pairs)
            producers = {ActionIdentity.mapped(producer, pairs) for producer in native_producers}
            read = self.signal_reads.get(consumer)
            producer = read[0] if read is not None else None

            if consumer is None or None in producers or (producer is not None and producer[0] not in targets):
                unpaired += 1
                continue

            if producer not in producers:
                self.finding(
                    "signal_producer",
                    identity,
                    scope=consumer[1],
                    port=consumer[3],
                    expected_producers=sorted(producers),
                    predicted_producer=producer,
                )
            else:
                matched += 1

        self.counts.update(
            signal_inputs=len(self.signal_inputs), signal_inputs_matched=matched, signal_inputs_unpaired=unpaired
        )

    def compare_values(self, dependencies, pairs):
        matched, unpaired = 0, 0
        targets = set(pairs.values())
        for native_consumer, (native_writer, native_outputs, rank, identity) in self.values.items():
            consumer = ActionIdentity.mapped(native_consumer, pairs)
            writer = ActionIdentity.mapped(native_writer, pairs)
            outputs = [ActionIdentity.mapped(output, pairs) for output in native_outputs]
            read = self.reads.get(consumer)
            predicted_writer = read[0] if read is not None else None

            if (
                consumer is None
                or writer is None
                or None in outputs
                or (predicted_writer is not None and predicted_writer[0] not in targets)
            ):
                unpaired += 1
                continue

            if predicted_writer != writer:
                self.finding(
                    "data_producer",
                    identity,
                    scope=consumer[1],
                    port=consumer[3],
                    expected_writer=writer,
                    predicted_writer=predicted_writer,
                )

            elif consumer not in self.ports or any(output not in self.ports for output in outputs):
                self.finding(
                    "data_readiness",
                    identity,
                    scope=consumer[1],
                    port=consumer[3],
                    reason="mapped data boundary is absent",
                )

            elif any(
                not dependencies.reaches((rank, self.ports[output]), (rank, self.ports[consumer])) for output in outputs
            ):
                self.finding(
                    "data_readiness",
                    identity,
                    scope=consumer[1],
                    port=consumer[3],
                    reason="consumer is not gated by its producer exports",
                )
            else:
                matched += 1

        self.counts.update(
            data_inputs=len(self.values),
            data_inputs_matched=matched,
            data_inputs_unpaired=unpaired,
            data_evidence_complete=self.value_evidence_complete,
        )
