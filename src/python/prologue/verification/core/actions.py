"""Identify native actions and compare the structure inside their named boundaries."""

from collections import defaultdict

from prologue.verification.core.action_identity import ActionIdentity, QueryResults
from prologue.verification.core.adapters import observed_scopes


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
        return {
            "arguments": cls.expression_key(scope.get("arguments", {})),
            "guards": cls.expression_list_key(scope.get("guards", [])),
        }

    @staticmethod
    def tensor_publications(raw):
        ends = {
            int(row["name"].split("/")[1]): row["end_counter"]
            for row in raw["host_observation"]["intervals"]
            if row["name"].startswith("prologue.scope/")
        }
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

            (origin,) = origins
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
            communication = self.native_communication(raw, observed)

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

            ports = self.validate_tensor_ports(index, scope, observed, raw, publications)

            if "native_transfer" not in observed:
                self.validate_state_accesses(index, observed, scope, ports)

        self.validate_queries(raw)
        self.bind_pipeline_placements(raw)

    def native_communication(self, raw, observed):
        communication = observed.get("communication")
        if "required_peers" in observed["bindings"]:
            group, counts = communication, observed["peer_rows"]
            sent, received = counts["send"], counts["receive"]
            if (
                len(sent) != len(group["members"])
                or len(received) != len(sent)
                or any(type(value) is not int or value < 0 for value in sent + received)
            ):
                raise ValueError("native expert exchange has invalid peer-wise row counts")

            required = [
                rank
                for rank, send, receive in zip(group["members"], sent, received, strict=True)
                if rank == group["rank"] or send or receive
            ]
            tensors = {
                port["direction"]: port["tensor"] for port in observed["ports"] if port["name"] == "hidden_states"
            }
            parent = raw["scopes"][observed["parent"]]
            (topk,) = [port["tensor"]["shape"][1] for port in parent["ports"] if port["name"] == "topk_ids"]

            before, after = (tensors[edge]["shape"][0] for edge in ("input", "output"))
            geometry = {"dispatch": (before * topk, after), "combine": (before, after * topk)}
            if group["required"] != required or (sum(sent), sum(received)) != geometry[group["operation"]]:
                raise ValueError("native expert exchange prerequisites disagree with its observed traffic")

            communication = {**group, "peer_rows": counts}

        return communication

    def validate_tensor_ports(self, index, scope, observed, raw, publications):

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
                        value["event"] for value in outputs if value["name"] == name and value["direction"] == "output"
                    ]

            if boundary["use"] != use or boundary["relation"] != relation or boundary["event"] != ready:
                raise ValueError(
                    f"graph scope {index} changes native tensor use or readiness: {port['name']}/{port['direction']}"
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
                raise ValueError(f"graph scope {index} changes native tensor port {port['name']}/{port['direction']}")

            # A blocking host read consumes by return, regardless of the tensor's storage device.
            if port.get("at_return", False):
                if observed["role"] == "segment":
                    raise ValueError("completed host reads require a separate native materialization action")
                if (
                    port["direction"] != "input"
                    or port["use"] != "host"
                    or boundary["use"] != "data"
                    or boundary["event"] != ports["call.return", "output"]["event"]
                ):
                    raise ValueError(f"graph scope {index} changes completed host read {port['name']}")

        return ports

    def validate_queries(self, raw):

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

            if (
                type(query["result"]) is not QueryResults.result_types.get(kind)
                or observed["role"] != "query"
                or not observed["bindings"].get("action")
                or observed["bindings"]["condition"] != condition
                or observed["work"]["context"]["query"] != kind
            ):
                raise ValueError("query observation disagrees with its action or result contract")

            for clock in query["clocks"]:
                QueryResults.clock(clock)
            if kind == "clock":
                QueryResults.clock({"read": query["read"], "result": query["result"]})

            if kind in {"transport", "metadata"} and not observed["work"]["context"]["request"]["id"]:
                raise ValueError("protocol and metadata queries require a logical request")

        if reads != {index for index, scope in enumerate(raw["scopes"]) if scope["role"] == "query"}:
            raise ValueError("query actions lack observed results")

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

            points = [
                port["point"]
                for port in scope["ports"]
                if port["use"] == "signal" and port["direction"] == "input" and "point" in port
            ]

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
                if (
                    port["name"] == "completion"
                    and port["direction"] == "output"
                    and "point" in port
                    and port["point"] in contexts
                ):
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
