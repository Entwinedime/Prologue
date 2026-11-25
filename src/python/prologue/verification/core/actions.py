"""Identify native actions and compare the structure inside their named boundaries."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice

from prologue.verification.core.adapters import observed_scopes
from prologue.verification.engine import FindingLedger


class QueryResults:
    """Check reads against this execution's producers, without pairing native polling counts."""

    result_types = {"event": bool, "transport": int, "metadata": bool, "resource": int, "clock": int}

    def __init__(self):
        self.groups, self.pending = {}, {}
        self.caller_returns = {}
        self.clock_epoch_ns = None
        self.signals, self.signal_completions = {}, {}
        self.results = {}
        self.counts = Counter()
        self.completion_reads = 0
        self.notifications = 0
        self.notices = {}
        self.notice_inputs = defaultdict(set)
        self.observation_edges = set()

    @staticmethod
    def clock(record, time_ns=None, epoch_ns=None):
        read, value = record["read"], record["result"]
        if (read.keys() != {"kind", "clock", "consumer"} or read["kind"] != "clock"
                or read["clock"] not in {"time", "monotonic", "perf_counter"}
                or not isinstance(read["consumer"], str) or not read["consumer"]
                or type(value) is not int or value < 0):
            raise ValueError("clock read requires its native API, policy consumer and integer nanoseconds")
        if time_ns is not None and value != time_ns + epoch_ns:
            raise ValueError("native policy clock differs from the completed core read and declared epoch")

    def observe(self, record):
        kind = record["kind"]
        if kind == "resources":
            self.clock_epoch_ns = record["clock_epoch_ns"]
            if type(self.clock_epoch_ns) is not int or self.clock_epoch_ns <= 0:
                raise ValueError("native control clock requires a positive declared epoch")
            self.groups = {
                group["rank"]: group["id"] for instance in record["instances"] for group in instance["groups"]
            }
        elif kind == "graph":
            graph = record["graph"]
            scope = graph["scopes"][0]
            ports = {port["name"]: port["event"] for port in scope["ports"]}
            rank = graph["resources"][graph["events"][ports["call.return"]]["resource"]]["rank"]
            group = self.groups[rank]
            if group in self.caller_returns.values() or any(query["group"] == group for query in self.pending.values()):
                raise ValueError(
                    "native action was submitted before its preceding read and caller continuation completed"
                )
            for port in scope["ports"]:
                if port["use"] == "signal" and port["direction"] == "output":
                    event = record["event_ids"][port["event"]]
                    self.signals[port["signal"]] = event
                    self.signal_completions[event] = None
            if scope["role"] != "query":
                return
            event, entered = ports["call.return"], ports["call.enter"]
            self.caller_returns[record["event_ids"][ports["caller.return"]]] = group
            self.pending[record["event_ids"][event]] = {
                "kind": scope["work"]["context"]["query"], "group": group,
                "returned_ns": None, "entered": record["event_ids"][entered], "reads": 0, "last_read": None,
                "work": scope["work"],
            }
        elif kind == "notification_declared":
            event = record["event"]
            if event in self.signal_completions or record["identity"]["group"] not in self.groups.values():
                raise ValueError("notification has a duplicate event or unknown participant")
            self.notices[event] = record["identity"]
            self.signal_completions[event] = None
        elif kind == "edge" and record["target"] in self.notices:
            self.notice_inputs[record["target"]].add(record["source"])
        elif kind == "edge" and record["relation"] == "observation":
            self.observation_edges.add((record["source"], record["target"]))
        elif kind == "completion":
            event = record["event"]
            self.caller_returns.pop(event, None)
            if event in self.pending:
                self.pending[event]["returned_ns"] = record["time_ns"]
            if event in self.signal_completions:
                self.signal_completions[event] = record["time_ns"]
        elif kind == "notification_ready":
            events = record["prerequisites"]
            if (
                self.notices.pop(record["event"]) != record["identity"]
                or self.signal_completions[record["event"]] != record["time_ns"]
                or not events
                or len(events) != len(set(events))
                or set(events) != self.notice_inputs.pop(record["event"])
                or any(
                    self.signal_completions[event] is None or self.signal_completions[event] > record["time_ns"]
                    for event in events
                )
            ):
                raise ValueError("notification was released before its actual producers completed")
            self.notifications += 1
        elif kind == "completion_read":
            query = self.pending[record["boundary"]]
            event = self.signals[record["signal"]]
            completed_ns = self.signal_completions[event]

            # Recorded order distinguishes a read from a completion at the same simulated time.
            if (query["returned_ns"] is None or record["time_ns"] != query["returned_ns"]
                    or record["group"] != query["group"] or record["event"] != event
                    or type(record["result"]) is not bool or record["result"] != (completed_ns is not None)
                    or (completed_ns is not None and completed_ns > record["time_ns"])):
                raise ValueError("completion read disagrees with its query, signal producer or completion state")
            query["reads"] += 1
            query["last_read"] = record
            self.completion_reads += 1
            if ((event, record["boundary"]) in self.observation_edges) != record["result"]:
                raise ValueError(
                    "completion read lacks its actual observation edge or turns an unsuccessful read into a wait"
                )
            return query
        elif kind == "state_query":
            query = self.pending.pop(record["boundary"])
            expected_type = self.result_types.get(query["kind"])

            # A protocol poll can use cached state or multiple reads; an event query reads exactly one signal.
            last_read = query["last_read"]
            if (type(record["result"]) is not expected_type
                    or record["group"] != query["group"] or query["returned_ns"] is None
                    or record["time_ns"] != query["returned_ns"] or record["completion_reads"] != query["reads"]
                    or (last_read is not None and record["time_ns"] < last_read["time_ns"])
                    or (query["kind"] == "event" and (query["reads"] != 1 or record["result"] != last_read["result"]))):
                raise ValueError("query result disagrees with its worker, result type, boundary or actual reads")
            for clock in record["clocks"]:
                self.clock(clock, query["returned_ns"], self.clock_epoch_ns)
            if query["kind"] == "clock":
                self.clock(
                    {"read": query["work"]["context"]["read"], "result": record["result"]},
                    query["returned_ns"],
                    self.clock_epoch_ns,
                )
            else:
                self.results[query["group"], query["entered"]] = record["result"]
            self.counts["nested_clocks"] += len(record["clocks"])
            self.counts[query["kind"]] += 1
            return query

    def finish(self, *, complete):
        if complete and (self.pending or self.notices or self.caller_returns):
            raise ValueError("execution has unfinished query or notification observations")
        return {**self.counts, "pending_queries": len(self.pending), "pending_callers": len(self.caller_returns),
                "pending_notifications": len(self.notices)}


class ActionScopes:
    def __init__(self, graph):
        self.graph = graph
        self.children, self.work = defaultdict(list), defaultdict(list)
        self.events = defaultdict(list)
        for index, scope in enumerate(graph["scopes"]):
            self.children[scope["parent"]].append(index)
        for piece in graph["work"]:
            self.work[piece["scope"]].append(piece)

        # Repeated action cuts share one event-ownership index over this immutable graph.
        for index, event in enumerate(graph["events"]):
            self.events[event["scope"]].append(index)
        self.placements = {}

    @staticmethod
    def expression_key(expression):
        if isinstance(expression, dict):
            return {name: ActionScopes.expression_key(value) for name, value in expression.items()}
        if not isinstance(expression, list):
            return type(expression).__name__, expression
        operation, *arguments = expression
        values = tuple(ActionScopes.expression_key(value) for value in arguments)
        if operation in {"add", "mul"}:
            identity = 0 if operation == "add" else 1
            values = tuple(value for value in values if value != ("int", identity))
            if not values:
                return "int", identity
        if operation == "floor_div" and values[1] == ("int", 1):
            return values[0]
        if len(values) == 1 and operation in {"add", "mul", "min", "max", "and", "or"}:
            return values[0]

        # Only binary commutation preserves the DSL's checked arithmetic without changing overflow order.
        return operation, frozenset(values) if len(values) == 2 and operation in {"add", "mul", "eq"} else values

    @classmethod
    def expression_list_key(cls, expressions):
        return None if expressions is None else tuple(cls.expression_key(value) for value in expressions)

    @classmethod
    def packing_key(cls, tensor):
        if packing := tensor.get("packing"):
            rules = {key: packing[key] for key in ("member_count", "row_extent", "range")}
            return packing["table"], cls.expression_key(rules), packing["slices"]

    @classmethod
    def declaration_expressions(cls, scope):
        return {"arguments": cls.expression_key(scope.get("arguments", {})),
                "guards": cls.expression_list_key(scope.get("guards", []))}

    @staticmethod
    def tensor_publications(raw):
        ends = {int(row["name"].split("/")[1]): row["end_counter"]
                for row in raw["host_observation"]["intervals"] if row["name"].startswith("prologue.scope/")}
        ends.update(
            {scope["id"]: scope["host_interval"]["end_counter"] for scope in raw["scopes"] if "service" in scope}
        )
        outputs = [
            (ends[scope["id"]], scope["id"], port)
            for scope in raw["scopes"]
            for port in scope["ports"]
            if port["direction"] == "output" and port["use"] in {"host", "device"} and port.get("tensor") is not None
        ]
        published = {}
        for _, scope, port in sorted(outputs, key=lambda row: (row[0], row[1])):
            tensor = port["tensor"]
            published.setdefault((tensor["storage"], tensor["version"]), (scope, port["name"]))
        return published

    def validate_source(self, raw, rank):
        if raw["environment"]["parallel"]["rank"] != rank or self.graph["environment"] != raw["environment"]:
            raise ValueError("graph and recording belong to different worker environments")
        raw = {**raw, "scopes": observed_scopes(raw)}
        self.source_ids = {}
        for index, scope in enumerate(self.graph["scopes"]):
            origins = scope["bindings"]["source_scopes"]
            if len(origins) != 1:
                raise ValueError("coarse scope requires one exact native declaration")
            origin, = origins
            if type(origin) is not int or not 0 <= origin < len(raw["scopes"]) or origin in self.source_ids:
                raise ValueError("native scope has an invalid or competing coarse owner")
            self.source_ids[origin] = index
        if self.source_ids.keys() != set(range(len(raw["scopes"]))):
            raise ValueError("coarse scopes omit native observations")
        publications = self.tensor_publications(raw)

        # Each interpreted call keeps its native declaration; private CPU suppression belongs to capture.
        for index, scope in enumerate(self.graph["scopes"]):
            origins = scope["bindings"]["source_scopes"]
            observed = raw["scopes"][origins[0]]
            communication = observed.get("communication")
            if "required_peers" in observed["bindings"]:
                group, counts = communication, observed["peer_rows"]
                sent, received = counts["send"], counts["receive"]
                if (len(sent) != len(group["members"]) or len(received) != len(sent)
                        or any(type(value) is not int or value < 0 for value in sent + received)):
                    raise ValueError("native expert exchange has invalid peer-wise row counts")

                required = [rank for rank, send, receive in zip(group["members"], sent, received, strict=True)
                            if rank == group["rank"] or send or receive]
                tensors = {
                    port["direction"]: port["tensor"] for port in observed["ports"] if port["name"] == "hidden_states"
                }
                parent = raw["scopes"][observed["parent"]]
                topk, = [port["tensor"]["shape"][1] for port in parent["ports"] if port["name"] == "topk_ids"]

                before, after = (tensors[edge]["shape"][0] for edge in ("input", "output"))
                geometry = {"dispatch": (before * topk, after), "combine": (before, after * topk)}
                if group["required"] != required or (sum(sent), sum(received)) != geometry[group["operation"]]:
                    raise ValueError("native expert exchange prerequisites disagree with its observed traffic")
                communication = {**group, "peer_rows": counts}

            bindings = dict(observed["bindings"])
            bindings.update({name: observed[name] for name in ("effects", "publication") if name in observed})
            bindings["source_scopes"] = origins
            parent = None if observed["parent"] is None else self.source_ids[observed["parent"]]
            if (
                not observed["complete"]
                or scope["bindings"] != bindings
                or scope["parent"] != parent
                or scope.get("row_call") != observed.get("row_call")
                or scope.get("communication") != communication
                or any(
                    scope[field] != observed[field]
                    for field in ("role", "site", "implementation", "parameters", "constants", "work")
                )
                or self.declaration_expressions(scope) != self.declaration_expressions(observed)
            ):
                raise ValueError(f"graph scope {index} differs from its native recording")

            # Check native tensor facts before either interpretation or projection can become the comparison's truth.
            ports = {(port["name"], port["direction"]): port for port in scope["ports"]}
            native_tensors = {
                (port["name"], port["direction"]) for port in observed["ports"] if port.get("tensor") is not None
            }
            graph_tensors = {key for key, port in ports.items() if port.get("tensor") is not None}
            if len(ports) != len(scope["ports"]) or native_tensors != graph_tensors:
                raise ValueError(f"graph scope {index} omits, duplicates or invents native tensor ports")
            for port in observed["ports"]:
                tensor = port.get("tensor")
                if tensor is None:
                    continue
                boundary = ports[port["name"], port["direction"]]
                use = {"host": "data", "device": "data", "metadata": "metadata", "state": "metadata"}[port["use"]]
                relation = "data" if use == "data" else "control"
                interval = "device" if port["use"] in {"device", "state"} else "call"
                side = "return" if interval == "call" else "complete"
                if port["direction"] == "input" and not port.get("at_return", False):
                    side = "enter"
                endpoint = f"{interval}.{side}"
                if observed.get("asynchronous", False) and port["use"] == "device" and port["direction"] == "output":
                    endpoint = "communication.complete"
                direction = "input" if side == "enter" else "output"
                ready = ports[endpoint, direction]["event"]
                if port["direction"] == "output" and use == "data":
                    producer, name = publications[tensor["storage"], tensor["version"]]
                    ancestor = producer
                    while ancestor is not None and ancestor != observed["id"]:
                        ancestor = raw["scopes"][ancestor]["parent"]
                    if producer != observed["id"] and ancestor is not None:

                        # Re-exporting a descendant's published version keeps that child's readiness, not the parent's
                        # return.
                        outputs = self.graph["scopes"][self.source_ids[producer]]["ports"]
                        (ready,) = [
                            value["event"]
                            for value in outputs
                            if value["name"] == name and value["direction"] == "output"
                        ]
                if boundary["use"] != use or boundary["relation"] != relation or boundary["event"] != ready:
                    raise ValueError(
                        f"graph scope {index} changes native tensor use or readiness: "
                        f"{port['name']}/{port['direction']}"
                    )
                projected = boundary["tensor"]
                if (
                    projected is None
                    or any(
                        tensor[field] != projected[field]
                        for field in ("dtype", "shape", "value", "storage", "version", "buffer")
                    )
                    or self.packing_key(tensor) != self.packing_key(projected)
                    or projected["allocation"]
                    != (None if tensor["allocation"] is None else self.source_ids[tensor["allocation"]])
                    or projected["writer"] != (None if tensor["writer"] is None else self.source_ids[tensor["writer"]])
                    or self.expression_list_key(tensor.get("shape_rule"))
                    != self.expression_list_key(projected.get("shape_rule"))
                ):
                    raise ValueError(
                        f"graph scope {index} changes native tensor port {port['name']}/{port['direction']}"
                    )

                # A blocking host read consumes by return, regardless of the tensor's storage device.
                if port.get("at_return", False):
                    if observed["role"] == "segment":
                        raise ValueError("completed host reads require a separate native materialization action")
                    if (port["direction"] != "input" or port["use"] != "host" or boundary["use"] != "data"
                            or boundary["event"] != ports["call.return", "output"]["event"]):
                        raise ValueError(f"graph scope {index} changes completed host read {port['name']}")

            if "native_transfer" not in observed:
                self.validate_state_accesses(index, observed, scope, ports)

        # Each read retains its own observed action and result, even when its logical identity repeats.
        reads = set()
        for query in raw["state_queries"]:
            scope = query["scope"]
            if type(scope) is not int or not 0 < scope < len(raw["scopes"]) or scope in reads:
                raise ValueError("query requires its own unique observed action")
            reads.add(scope)
            observed = raw["scopes"][scope]
            kind = query["kind"]
            condition = {"query": kind}
            if kind in {"resource", "clock"}:
                condition["read"] = query["read"]
                if observed["work"]["context"]["read"] != query["read"]:
                    raise ValueError("query changes its native resource, clock or consumer")
            if (type(query["result"]) is not QueryResults.result_types.get(kind)
                    or observed["role"] != "query" or not observed["bindings"].get("action")
                    or observed["bindings"]["condition"] != condition
                    or observed["work"]["context"]["query"] != kind):
                raise ValueError("query observation disagrees with its action or result contract")
            for clock in query["clocks"]:
                QueryResults.clock(clock)
            if kind == "clock":
                QueryResults.clock({"read": query["read"], "result": query["result"]})
            if kind in {"transport", "metadata"} and not observed["work"]["context"]["request"]["id"]:
                raise ValueError("protocol and metadata queries require a logical request")
        if reads != {index for index, scope in enumerate(raw["scopes"]) if scope["role"] == "query"}:
            raise ValueError("query actions lack observed results")
        self.bind_pipeline_placements(raw)

    def validate_state_accesses(self, index, observed, scope, ports):
        for space, field in (("kv", "kv_accesses"), ("slot", "slot_accesses")):
            native = [access for access in observed.get("state_accesses", ()) if access["space"] == space]
            if len(native) != len(scope[field]):
                raise ValueError(f"graph scope {index} changes its declared {space} accesses")
            for access, projected in zip(native, scope[field], strict=True):
                expected = {key: value for key, value in access.items() if key != "interval"}
                actual = {key: value for key, value in projected.items() if key not in {"start", "finish"}}
                if space == "kv":

                    # Only range rules are expressions; request lists and regions retain literal order and multiplicity.
                    for value in (expected, actual):
                        if "rule" in value:
                            value["rule"] = self.expression_key(value["rule"])
                if expected != actual:
                    raise ValueError(f"graph scope {index} changes its declared {space} access contents")
                interval = access["interval"]
                if interval == "service":
                    for side, endpoint in (("begin", "start"), ("end", "finish")):
                        event = projected[endpoint]
                        resource = self.graph["resources"][self.graph["events"][event]["resource"]]
                        if (
                            resource["kind"] != "service"
                            or resource["identity"] != f"service/{observed['thread']}"
                            or self.graph["evidence"]["events"][event]["order"]
                            != observed["host_interval"][side + "_counter"]
                        ):
                            raise ValueError(f"graph scope {index} changes its declared service access interval")
                    continue
                entered, returned = {
                    "call": ("call.enter", "call.return"),
                    "device": ("device.enter", "device.complete"),
                }[interval]
                begin, end = ports[entered, "input"]["event"], ports[returned, "output"]["event"]
                if (projected["start"], projected["finish"]) != (begin, end):
                    raise ValueError(f"graph scope {index} changes its declared {space} access interval")

    def bind_pipeline_placements(self, raw):
        contexts = {}
        queries = {query["scope"]: query["point"] for query in raw["state_queries"] if query["kind"] == "event"}
        for scope in raw["scopes"]:
            producer = scope["work"].get("context", {}).get("producer", {})
            if "pipeline" not in producer:
                continue
            points = [port["point"] for port in scope["ports"]
                      if port["use"] == "signal" and port["direction"] == "input" and "point" in port]
            if scope["id"] in queries:
                points.append(queries[scope["id"]])
            for point in points:
                if contexts.setdefault(point, producer["pipeline"]) != producer["pipeline"]:
                    raise ValueError("native signal consumers disagree on their producer's pipeline placement")

        # A recorded event generation carries its creator's position, independently of a later wait's current position.
        for scope in raw["scopes"]:
            if scope["role"] not in {"launch", "output_skip"}:
                continue
            for port in scope["ports"]:
                if (port["name"] == "completion" and port["direction"] == "output" and "point" in port
                        and port["point"] in contexts):
                    self.placements[self.source_ids[scope["id"]]] = contexts[port["point"]]

        # Preparation and actual materialization consume the PP value produced in that native loop position.
        for index, scope in enumerate(self.graph["scopes"]):
            materialization = scope["role"] == "host_read"
            if scope["role"] != "output_prepare" and not materialization:
                continue
            for port in scope["ports"]:
                if port["name"] != "next_token_ids" or port["direction"] != "input":
                    continue
                writer = port["tensor"]["writer"]
                while writer is not None:
                    owner = self.graph["scopes"][writer]
                    placement = self.placements.get(writer, owner["work"].get("context", {}).get("pipeline"))
                    if placement is not None:
                        self.placements[index] = placement
                        if materialization:
                            self.placements[scope["parent"]] = placement
                        break
                    writer = owner["parent"]

    def record(self, index, group):
        record = ActionIdentity.record(self.graph["scopes"][index], group)
        if record["work"]["observation_context"] is not None:
            return record

        # Nested calls inherit the envelope's entry position; their own continuation and logical progress remain
        # separate.
        owner = index
        while owner is not None:
            if owner in self.placements:
                record["work"]["observation_context"] = {"pipeline": self.placements[owner]}
                break
            owner = self.graph["scopes"][owner]["parent"]
        return record


class ActionIdentity:
    @classmethod
    def request_progress(cls, value):
        if isinstance(value, list):
            return any(cls.request_progress(item) for item in value)
        if not isinstance(value, dict):
            return False

        # Request progress follows the shared ABI; its presence in a snapshot does not establish work ownership.
        return {"id", "generated_tokens", "committed_tokens", "extend"} <= value.keys() or any(
            cls.request_progress(item) for item in value.values()
        )

    @classmethod
    def request_work(cls, record):
        work = record["work"]

        # Queue/caller snapshots observe requests; only this call's declared request or batch operands own work.
        return bool(work["members"]) or any(
            cls.request_progress(work["context"].get(operand)) for operand in ("request", "members", "left", "right")
        )

    @classmethod
    def occurrence_anchored(cls, record):

        # A readiness read can recur for the same work; one observation does not make it a unique occurrence.
        return record["action"] != "query" and cls.request_work(record)

    @staticmethod
    def selected(index, scope):
        return bool(index) and (scope["role"] == "segment" or scope["bindings"].get("action", False)
                                or (scope["parent"] == 0 and scope["role"] != "invocation"))

    @staticmethod
    def work_identity(work):

        # Preserve submitted token ranges; later request progress can be shared by different in-flight batches.
        logical = {
            "members": [{name: value for name, value in member.items() if name != "slot"}
                        for member in work.get("members", ())],
            "context": work.get("context"), "callers": work.get("callers", []),
            "continuation": work.get("continuation"), "observation_context": work.get("observation_context"),
        }
        if not isinstance(logical["context"], dict) or not isinstance(logical["callers"], list):
            raise ValueError("work has no valid logical work/caller context")
        if logical["continuation"] is not None and not isinstance(logical["continuation"], dict):
            raise ValueError("work has invalid continuation context")
        if logical["observation_context"] is not None and not isinstance(logical["observation_context"], dict):
            raise ValueError("work has invalid observation context")
        return logical

    @classmethod
    def record(cls, scope, group):
        return {
            "group": group,
            "action": scope["role"],
            "parameters": scope["parameters"],
            "work": cls.work_identity(scope["work"]),
        }

    @staticmethod
    def logical_key(record, contexts):
        if "work" not in record:
            return json.dumps(record, sort_keys=True)

        # Placement qualifies absolute clocks; it cannot distinguish repeated logical work across runs.
        work = {name: value for name, value in record["work"].items() if name != "observation_context"}
        return json.dumps(contexts.encode({**record, "work": work}), sort_keys=True)

    @staticmethod
    def structure_key(record, contexts):
        if "work" not in record:
            return json.dumps(record, sort_keys=True)

        # Shared structure ignores caller state; logical pairing retains it and only absolute timing keeps placement.
        work = record["work"]
        callers = [{"owner": caller["owner"], "entry": caller["entry"]} for caller in work["callers"]]
        return json.dumps(
            contexts.encode(
                {
                    **record,
                    "work": {
                        "members": work["members"],
                        "context": work["context"],
                        "callers": callers,
                        "continuation": work["continuation"],
                    },
                }
            ),
            sort_keys=True,
        )

    @staticmethod
    def group(graph, instance):
        port = next(port for port in graph["scopes"][0]["ports"] if port["name"] == "call.enter")
        resource = graph["resources"][graph["events"][port["event"]]["resource"]]
        return f"{instance}/{resource['rank']}"


class GraphContents(ActionScopes):
    def __init__(self, graph):
        super().__init__(graph)
        self.pools = {}

    @classmethod
    def value_differences(cls, native, predicted, path=""):
        if isinstance(native, dict) and isinstance(predicted, dict):
            for key in sorted(native.keys() | predicted.keys()):
                if key in native and key in predicted:
                    yield from cls.value_differences(native[key], predicted[key], f"{path}/{key}")
                else:
                    yield {"field": f"{path}/{key}", "native": native.get(key), "predicted": predicted.get(key)}
        elif native != predicted:
            yield {"field": path, "native": native, "predicted": predicted}

    @classmethod
    def differences(cls, native, predicted, details=False):
        for path in sorted(native.keys() | predicted.keys()):
            if path not in native or path not in predicted:
                yield "scope_tree", {"scope": path, "native": path in native, "predicted": path in predicted}
                continue
            for field in sorted(native[path].keys() | predicted[path].keys()):
                expected, actual = native[path].get(field), predicted[path].get(field)
                if expected != actual:
                    changes = cls.value_differences(expected, actual)
                    yield field, {"scope": path, "changes": list(changes if details else islice(changes, 3))}

    def control_key(self, root, identity):
        scope = self.graph["scopes"][root]
        return json.dumps({"control": {
            **{name: identity[name] for name in ("group", "action", "parameters")},
            **{name: scope[name] for name in ("site", "implementation")},
            "owner": (identity["work"]["continuation"] or {}).get("owner"),
        }}, sort_keys=True)

    def declared_control(self, root):
        scope = self.graph["scopes"][root]
        if not scope["bindings"].get("segment", {}).get("host_only", False):
            return False
        ports = {(port["name"], port["direction"]): port for port in scope["ports"]}
        host_ports = len(scope["ports"]) == 2 and ports.keys() == {("call.enter", "input"), ("call.return", "output")}
        for port in scope["ports"]:
            resource = self.graph["resources"][self.graph["events"][port["event"]]["resource"]]
            host_ports &= (resource["kind"] == "thread" and resource["role"] == "control"
                           and port["use"] == "control" and port["relation"] == "control"
                           and not any(port.get(name) for name in ("tensor", "wait", "signal"))
                               and "rendezvous" not in port)
        pieces = self.work[root]
        resource = self.graph["resources"][pieces[0]["resource"]] if len(pieces) == 1 else {}

        # A native declaration licenses shared CPU contents, never hidden children, device work or resource effects.
        if (scope["role"] != "segment" or self.children[root] or not host_ports or len(pieces) != 1
                or pieces[0]["kind"] != "host" or resource.get("kind") != "thread" or resource.get("role") != "control"
                or scope.get("communication") or scope["kv_accesses"] or scope["slot_accesses"]):
            raise ValueError("native host-only declaration contains undeclared execution structure")
        return True

    def structure_key(self, root, identity, controls, contexts):
        key = self.control_key(root, identity)
        return key if key in controls else ActionIdentity.structure_key(identity, contexts)

    def observed_pools(self, raw, registrations, groups):

        # The native session identifies the actual destination independently of the predictor's routing rules.
        targets = sorted({registrations[scope["native_transfer"]["session"]]
                          for scope in raw["scopes"] if "native_transfer" in scope})
        self.pools = {f"destination/{position}": groups[rank] for position, rank in enumerate(targets)}
        bindings = {rank: f"destination/{position}" for position, rank in enumerate(targets)}
        for scope in self.graph["scopes"]:
            if scope["role"] == "kv_transfer":
                expected = set()
                for origin in scope["bindings"]["source_scopes"]:
                    transfer = raw["scopes"][origin]["native_transfer"]
                    field = "kv_accesses" if transfer["kind"] == "kv" else "slot_accesses"
                    if transfer["kind"] == "kv" and not transfer["source_pages"]:
                        continue
                    destination = bindings[registrations[transfer["session"]]]
                    expected.update(((field, "local", "read"), (field, destination, "write")))
                actual = {(field, access["pool"], access["access"])
                          for field in ("kv_accesses", "slot_accesses") for access in scope[field]}
                if actual != expected:
                    raise ValueError("coarse transfer loses or invents a native payload endpoint")

    def pool(self, name, group):
        return group if name == "local" else self.pools.get(name, name)

    def invocation_bindings(self, root, group):
        order = []
        children = [child for child in self.children[root] if self.graph["scopes"][child]["role"] == "kv_transfer"]
        if "evidence" in self.graph:
            entries = {child: next(port["event"] for port in self.graph["scopes"][child]["ports"]
                if port["name"] == "call.enter")
                       for child in children}

            # Native service entries are measured CPU calls; graph scope numbering is not an ordering witness.
            children.sort(key=lambda child: self.graph["evidence"]["events"][entries[child]]["time_ns"])
        for child in children:
            scope = self.graph["scopes"][child]
            peers = {self.pool(access["pool"], group)
                     for field in ("kv_accesses", "slot_accesses") for access in scope[field]
                         if access["access"] == "write"}
            if not peers:
                continue
            peer, = peers

            # Original synchronous service order supplies a native choice, never a predicted graph or an ordinal peer
            # ID.
            if not order or order[-1] != peer:
                if peer in order:
                    raise ValueError("native transport returns to an already completed destination")
                order.append(peer)
        return {"transport_order": order} if order else {}

    def resource(self, index):
        resource = self.graph["resources"][index]
        return {name: resource[name] for name in ("kind", "role", "rank", "serial")}

    def paths(self, root):
        paths, pending = {root: "root"}, [root]
        for index in pending:
            for child, scope in enumerate(self.children[index]):
                paths[scope] = f"{paths[index]}/{child}"
                pending.append(scope)
        return paths

    def serial_tails(self, root):
        graph = self.graph
        scopes = [root]
        for scope in scopes:
            scopes.extend(self.children[scope])
        retained = {event for scope in scopes for event in self.events[scope]}
        for scope in scopes:
            for work in self.work[scope]:
                retained.update((work["start"], work["finish"]))
            for port in graph["scopes"][scope]["ports"]:
                retained.add(port["event"])
                if "wait" in port:
                    retained.add(port["wait"]["begin"])
                if "rendezvous" in port:
                    retained.add(port["rendezvous"])
            for access in graph["scopes"][scope]["kv_accesses"] + graph["scopes"][scope]["slot_accesses"]:
                retained.update((access["start"], access["finish"]))

        tails = {}
        for event in retained:
            resource = graph["events"][event]["resource"]
            if not graph["resources"][resource]["serial"]:
                continue
            previous = tails.get(resource)
            if (
                previous is None
                or graph["evidence"]["events"][event]["order"] > graph["evidence"]["events"][previous]["order"]
            ):
                tails[resource] = event
        return tails

    def completion_boundaries(self, root, *, named_only=False):
        graph = self.graph
        ports = graph["scopes"][root]["ports"]
        result = [{"at": {"port": port["name"], "direction": port["direction"]}, "event": port["event"]}
            for port in ports]
        if named_only:
            return result

        named_resources = {graph["events"][port["event"]]["resource"] for port in ports}
        tails = self.serial_tails(root)
        for resource, event in sorted(tails.items()):
            names = sorted((port["name"], port["direction"]) for port in ports
                           if graph["events"][port["event"]]["resource"] == resource)
            if names:
                name, direction = names[0]
                identity = {"port": name, "direction": direction}
            else:
                identity = {key: graph["resources"][resource][key] for key in ("kind", "role")}
                if sum(all(graph["resources"][peer][key] == value for key, value in identity.items())
                       for peer in tails.keys() - named_resources) != 1:
                    raise ValueError(f"scope {root} ({graph['scopes'][root]['site']}) resource tail {identity} "
                                     "has no unique named interface or resource role")

            # A semantic resource identity selects a tail; native thread/stream IDs never cross the run boundary.
            result.append({"at": {"resource": identity}, "event": event})
        return result

    def host_tails(self, root):
        graph = self.graph
        ports = graph["scopes"][root]["ports"]
        if "evidence" in graph:
            tails = self.serial_tails(root)
        else:
            tails = {graph["events"][port["event"]]["resource"]: port["event"] for port in ports
                     if port["name"].startswith("resource.exit/")}
        named = {port["event"] for port in ports
                 if not port["name"].startswith(("resource.enter/", "resource.exit/"))}
        host_finishes = {piece["finish"] for piece in self.work[root] if piece["kind"] == "host"}
        scope = tuple(graph["scopes"][root][field] for field in ("role", "site", "implementation"))
        identities = {}
        for lane, tail in tails.items():
            resource = graph["resources"][lane]
            if resource["kind"] == "thread" and tail not in named and tail in host_finishes:
                identities[tail] = (*scope, resource["role"])
        return identities

    def boundary_event(self, root, selector):
        ports = self.graph["scopes"][root]["ports"]
        identity = selector.get("resource", selector)
        if "port" in identity:
            port, = [port for port in ports
                if (port["name"], port["direction"]) == (identity["port"], identity["direction"])]
            if "resource" not in selector:
                return port["event"]
            resource = self.graph["events"][port["event"]]["resource"]
            candidates = [port for port in ports if port["name"].startswith("resource.exit/")
                          and self.graph["events"][port["event"]]["resource"] == resource]
        else:
            named_resources = {self.graph["events"][port["event"]]["resource"] for port in ports
                               if not port["name"].startswith(("resource.enter/", "resource.exit/"))}
            candidates = [port for port in ports if port["name"].startswith("resource.exit/")
                          and self.graph["events"][port["event"]]["resource"] not in named_resources
                          and all(self.graph["resources"][self.graph["events"][port["event"]]["resource"]][key] == value
                                  for key, value in identity.items())]
        if len(candidates) > 1:
            raise ValueError("execution resource tail has an ambiguous semantic interface")
        return candidates[0]["event"] if candidates else None

    def snapshot(self, root, group):
        graph, paths = self.graph, self.paths(root)

        # Relative versions and alias partitions survive different addresses and allocation histories.
        versions = {}
        for index in paths:
            for port in graph["scopes"][index]["ports"]:
                if port["use"] == "data" and (tensor := port.get("tensor")):
                    storage = tensor["storage"]
                    versions[storage] = min(versions.get(storage, tensor["version"]), tensor["version"])
        storages, resources, channels = {}, {}, {}
        channel_origins = {}
        for index in paths:
            if communication := graph["scopes"][index].get("communication"):
                channel = communication["channel"]
                channel_origins[channel] = min(channel_origins.get(channel, communication["order"]),
                    communication["order"])

        result = {}
        for index in paths:
            scope = graph["scopes"][index]
            contents = {name: scope[name] for name in ("role", "site", "implementation", "parameters")}
            if index != root and (row := scope.get("row_call")):
                contents["row_call"] = {**row, "request": scope["work"]["members"][0]["id"]}
            contents["weights"] = scope["bindings"].get("weights")
            ports = {}
            for port in sorted(scope["ports"], key=lambda value: (value["name"], value["direction"])):

                # These ports are added during action cutting; cross-action serial frontiers need their own check.
                if index == root and port["name"].startswith(("resource.enter/", "resource.exit/")):
                    continue
                boundary = {name: port[name] for name in ("use", "relation")}
                resource = graph["events"][port["event"]]["resource"]
                resources.setdefault(resource, len(resources))
                boundary["resource"] = {**self.resource(resource), "alias": resources[resource]}
                if "wait" in port:
                    waited = port["wait"]["resource"]
                    resources.setdefault(waited, len(resources))
                    boundary["wait_resource"] = {**self.resource(waited), "alias": resources[waited]}
                boundary["rendezvous"] = "rendezvous" in port
                if tensor := port.get("tensor"):
                    boundary["tensor"] = {"dtype": tensor["dtype"], "shape": tensor["shape"]}
                    if packing := tensor.get("packing"):
                        boundary["tensor"]["packing"] = {key: packing[key] for key in ("table", "slices")}

                    # Metadata describes layout; its runtime buffer lifecycle is not a graph data dependency.
                    if port["use"] == "data":
                        storage = tensor["storage"]
                        storages.setdefault(storage, len(storages))
                        boundary["tensor"].update(
                            storage_alias=storages[storage], version=tensor["version"] - versions[storage],
                            writer=paths.get(tensor["writer"], "external"),
                            allocation=paths.get(tensor["allocation"], "external"),
                            buffer=tensor["buffer"],
                        )
                ports[f"{port['direction']}:{port['name']}"] = boundary
            contents["ports"] = ports

            for field in ("kv_accesses", "slot_accesses"):
                accesses = []
                for access in scope[field]:
                    value = {name: value for name, value in access.items() if name not in {"start", "finish", "rule"}}
                    value["pool"] = self.pool(value["pool"], group)
                    accesses.append(value)
                contents[field] = accesses
            communication = scope.get("communication")
            if communication:

                # Tensor ports and execution edges supply the evidence for agreements and service order.
                communication = {name: value for name, value in communication.items()
                    if name not in {"agreements", "ordered_service"}}
                channel = communication["channel"]
                channels.setdefault(channel, len(channels))
                communication = {**communication, "channel": channels[channel],
                    "order": communication["order"] - channel_origins[channel]}
            contents["communication"] = communication
            work = Counter()
            for item in self.work[index]:
                service = {"kind": item["kind"], "resource": self.resource(item["resource"])}
                if "copy" in item:
                    service["copy"] = item["copy"]
                work[json.dumps(service, sort_keys=True)] += 1
            contents["work"] = dict(work)
            result[paths[index]] = contents
        return result


@dataclass(frozen=True, slots=True)
class RankedBoundaries:
    events: dict
    work: list
    resources: dict


class ActionBoundaries:
    def __init__(self, contents, root, event_ids=None):
        graph = contents.graph
        event_ids = range(len(graph["events"])) if event_ids is None else event_ids
        paths, resources = contents.paths(root), {}
        self.events, self.work, self.resources = {}, [], {}
        for scope, path in paths.items():
            node = graph["scopes"][scope]
            boundaries = {}
            for port in sorted(node["ports"], key=lambda value: (value["name"], value["direction"])):

                # Generated action frontiers have no source port; their edges still participate in reachability.
                if scope == root and port["name"].startswith(("resource.enter/", "resource.exit/")):
                    continue
                name = f"{path}/{port['direction']}:{port['name']}"
                boundaries[name] = event_ids[port["event"]]
                resource = graph["events"][port["event"]]["resource"]
                resources.setdefault(resource, len(resources))
                if "wait" in port:
                    boundaries[name + "/wait.begin"] = event_ids[port["wait"]["begin"]]
                    resources.setdefault(port["wait"]["resource"], len(resources))
                if "rendezvous" in port:
                    boundaries[name + "/rendezvous"] = event_ids[port["rendezvous"]]
            for field in ("kv_accesses", "slot_accesses"):
                for index, access in enumerate(node[field]):
                    for side in ("start", "finish"):
                        boundaries[f"{path}/{field}/{index}/{side}"] = event_ids[access[side]]

            self.events.update(boundaries)

        for scope, path in paths.items():
            for work in contents.work[scope]:
                resource = work["resource"]
                descriptor = contents.resource(resource)
                self.resources[resource] = (*descriptor.values(), resources.get(resource))
                identity = (path, work["kind"], resource)
                self.work.append((identity, event_ids[work["start"]], event_ids[work["finish"]]))

    def ranked(self, rank):
        return RankedBoundaries(
            events={name: (rank, event) for name, event in self.events.items()},
            work=[(identity, (rank, start), (rank, finish)) for identity, start, finish in self.work],
            resources=self.resources,
        )


class ActionOrderComparison:
    def __init__(self, contexts, details=False):
        self.contexts, self.details = contexts, details
        self.counts, self.findings = Counter(), FindingLedger()
        self.repeated_counts = Counter()
        self.qualified = False


    def compare_repeated(self, actual, predicted, actions, repeated):
        if not repeated:
            return
        names = ("root/input:call.enter", "root/output:call.return")
        anchors = [(native.events[name], target.events[name]) for _, native, target in actions for name in names]
        if not anchors:
            self.repeated_counts["unqualified_contexts"] += len(repeated)
            self.repeated_counts["unqualified_calls"] += sum(len(targets) for _, _, targets in repeated.values())
            return
        frontiers = []
        for side, graph in enumerate((actual, predicted)):
            events = set()
            for row in repeated.values():
                for call in row[side + 1]:
                    events.update(call.events[name] for name in names)
            frontiers.append(graph.frontiers([row[side] for row in anchors], events))
        for identity, native, targets in repeated.values():
            if not native:
                self.repeated_counts.update(unqualified_contexts=1, unqualified_calls=len(targets))
                continue
            positions = [{tuple(value for name in names for value in masks[call.events[name]]) for call in calls}
                         for masks, calls in zip(frontiers, (native, targets), strict=True)]
            missing, extra = positions[0] - positions[1], positions[1] - positions[0]
            self.repeated_counts.update(
                contexts_checked=1,
                native_positions=len(positions[0]),
                predicted_positions=len(positions[1]),
                matched_positions=len(positions[0] & positions[1]),
                calls_checked=len(targets),
            )
            if not missing and not extra:
                self.repeated_counts.update(contexts_matched=1, calls_matched=len(targets))
                continue

            # Compare whole position sets. No member is selected to supply a source occurrence, body or duration.
            examples = {}
            for label, differences, calls, masks in (
                ("native", missing, native, frontiers[0]),
                ("predicted", extra, targets, frontiers[1]),
            ):
                for call in calls:
                    signature = tuple(value for name in names for value in masks[call.events[name]])
                    if signature in differences:
                        examples[label] = {"events": {name: call.events[name] for name in names},
                                           "anchor_counts": [mask.bit_count() for mask in signature]}
                        break
            self.findings.add("repeated_call_order", identity, {"native_only_positions": len(missing),
                                  "predicted_only_positions": len(extra), "examples": examples})

    def compare(self, actual, predicted, actions):
        labels, identities = defaultdict(list), {}
        for identity, native, estimate in actions:
            if native.events.keys() != estimate.events.keys():
                self.findings.add("ordering_evidence", identity, {"reason": "action boundary sets differ"})
                return

            # Occurrence pairs are already established; repeated work still needs distinct order labels.
            key = ActionIdentity.logical_key(identity, self.contexts), native.events["root/input:call.enter"]
            identities[key] = identity
            for name, event in native.events.items():
                labels[event, estimate.events[name]].append((key, name))
        pairs = list(labels)
        self.counts.update(actions_compared=len(identities), boundaries_compared=sum(map(len, labels.values())),
                           paired_events=len(pairs))
        if not pairs:
            return

        # Equal ports may alias one event; their partitions must agree before comparing partial orders.
        for side, origin in enumerate(("native", "predicted")):
            aliases = defaultdict(list)
            for pair in pairs:
                aliases[pair[side]].append(pair)
            for event, entries in aliases.items():
                if len(entries) == 1:
                    continue
                self.counts["alias_conflicts"] += 1
                key, _ = labels[entries[0]][0]
                self.findings.add("action_order_alias", identities[key], {
                    "origin": origin, "event": event, "opposite_events": len(entries),
                    "reason": "named boundaries alias one event in only one run",
                })
        if self.findings:
            return

        projections = [
            graph.projected([pair[side] for pair in pairs]) for side, graph in enumerate((actual, predicted))
        ]
        covers = [
            {(source, target) for source, targets in graph.next.items() for target in targets} for graph in projections
        ]
        names = [min(labels[pair]) for pair in pairs]
        differences = []
        for side, kind in enumerate(("missing", "extra")):
            other = projections[1 - side]
            self.counts[("native" if side == 0 else "predicted") + "_relations"] = len(covers[side])
            for source, target in covers[side]:
                if other.reaches(source, target):
                    continue
                reverse = other.reaches(target, source)
                differences.append((side, source, target, reverse))
                self.counts[kind + "_relations"] += 1
                self.counts[kind + "_reversed_relations"] += reverse
                self.counts[kind + "_cross_rank_relations"] += (
                    identities[names[source][0]]["group"] != identities[names[target][0]]["group"]
                )
        self.qualified = True

        # Covers generate the whole order. Checking both directions preserves strict reachability equality.
        # The first native topological position is a representative, not a unique first cause under concurrency.
        differences.sort(key=lambda item: (actual.positions[pairs[item[2]][0]], item[0], item[1]))
        shown = set()
        for side, source, target, reverse in differences:
            source_key, source_name = names[source]
            target_key, target_name = names[target]
            kind = "missing" if side == 0 else "extra"
            detail = {"source": {"work": identities[source_key], "boundary": source_name}, "boundary": target_name,
                      "opposite_order_reversed": reverse}
            if self.details or kind not in shown:
                detail["path"] = (actual, predicted)[side].witness(pairs[source][side], pairs[target][side])
                shown.add(kind)
            self.findings.add(kind + "_order", identities[target_key], detail)
