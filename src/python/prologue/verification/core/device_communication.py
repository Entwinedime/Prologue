"""Qualify device collective, transfer and batch synchronization endpoints."""

from collections import defaultdict
from itertools import pairwise

from prologue.verification.core.action_identity import ActionIdentity
from prologue.verification.core.batch_sync import BatchSynchronization
from prologue.verification.core.metadata import compare_metadata


class CommunicationEvidence:
    """Track the native and predicted protocols in their own order before comparing paired actions."""

    def __init__(self, counts, finding):
        self.counts, self.finding = counts, finding
        self.ready, self.complete = {}, {}
        self.channels = defaultdict(lambda: defaultdict(list))
        self.predicted_channels = defaultdict(lambda: defaultdict(list))
        self.communication_edges = set()
        self.native_communication_edges = set()
        self.metadata_returns = {}
        self.endpoints = {}
        self.action_returns = {}
        self.batch_sync = BatchSynchronization(self.finding)

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
        (port,) = (port for port in scope["ports"] if port["use"] in {"data", "metadata"})
        tensor = port["tensor"]
        return port["direction"], tensor["dtype"], tensor["shape"]

    def endpoint(self, contents, index, label, identity, event_ids, values):
        scope = contents.graph["scopes"][index]
        entered = next(port["event"] for port in scope["ports"] if port["name"] == "call.enter")
        ready, complete = self.communication_bounds(contents, index)
        return {
            **scope["communication"],
            "scope": label,
            "work": identity,
            "entered": event_ids[entered],
            "ready": event_ids[ready],
            "complete": event_ids[complete],
            "values": values,
            "metadata": scope["role"] == "metadata_collective",
            "elements": scope["parameters"].get("elements"),
            "group_role": scope["bindings"].get("collective_group"),
            "payload": self.transfer_payload(scope),
        }

    def native_batch_returns(self, raw, contents, identities, endpoints):
        observed = set()
        for record in raw.get("action_values", ()):
            if "batch_sync" not in record["values"]:
                continue
            parts = record["values"]["segments"]

            if (
                not parts
                or any(type(part) is not int or not 0 < part < len(raw["scopes"]) for part in parts)
                or parts != sorted(set(parts))
                or record["scope"] != parts[-1]
                or contents.source_ids[parts[-1]] in observed
            ):
                raise ValueError("batch synchronization requires its actual ordered segment span")

            parent = raw["scopes"][parts[0]]["parent"]
            if parts != [
                i for i, scope in enumerate(raw["scopes"]) if scope["parent"] == parent and scope["role"] == "segment"
            ]:
                raise ValueError("batch synchronization observation omits or borrows an invocation segment")

            parts = [contents.source_ids[part] for part in parts]
            self.batch_sync.span([identities[part] for part in parts])
            identity = identities[parts[-1]]
            candidates = [endpoint for endpoint in endpoints if parts[0] <= endpoint["scope"][0][1] <= parts[-1]]
            self.batch_sync.attach(record["values"]["batch_sync"], candidates, identity, "native")
            observed.add(parts[-1])

        expected = {
            scope for scope, identity in identities.items() if self.batch_sync.terminal(identity.get("work", {}))
        }
        self.batch_sync.counts["native_unobserved_returns"] += len(expected - observed)

    def begin_action(self, graph, identity, events, key):
        continuation = identity.get("work", {}).get("continuation") or {}

        if continuation.get("owner") == "batch_sync_exchange":
            returned = next(port["event"] for port in graph["scopes"][0]["ports"] if port["name"] == "call.return")
            self.action_returns[events[returned]] = {"key": key, "work": identity, "completed_ns": None}

    def add_predicted(self, contents, index, key, path, identity, events):
        graph = contents.graph
        scope = graph["scopes"][index]
        communication = scope["communication"]
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


    def observe(self, record):
        if record["kind"] == "completion" and record["event"] in self.action_returns:
            self.action_returns[record["event"]]["completed_ns"] = record["time_ns"]
        if record["kind"] == "action_values" and "batch_sync" in record["values"]:
            parts = record["values"]["segments"]
            if not parts or len(parts) != len(set(parts)) or record["boundary"] != parts[-1]:
                raise ValueError("batch synchronization requires its own completed segments")

            span = [self.action_returns[part] for part in parts]
            if any(
                part["work"]["group"] != record["group"]
                or part["completed_ns"] is None
                or record["time_ns"] < part["completed_ns"]
                for part in span
            ):
                raise ValueError("batch synchronization result precedes its invocation return")

            self.batch_sync.span([part["work"] for part in span])
            keys = [part["key"][1] for part in span]
            if keys != sorted(set(keys)):
                raise ValueError("batch synchronization segments are out of invocation order")

            first, last = keys[0], keys[-1]
            expected = {
                event
                for event, part in self.action_returns.items()
                if part["work"]["group"] == record["group"]
                and first <= part["key"][1] <= last
                and (part["work"].get("work", {}).get("continuation") or {}).get("owner") == "batch_sync_exchange"
            }

            if set(parts) != expected:
                raise ValueError("batch synchronization observation omits an invocation segment")
            candidates = [
                endpoint
                for (key, _), endpoint in self.endpoints.items()
                if key[0] == record["group"] and first <= key[1] <= last
            ]
            self.batch_sync.attach(record["values"]["batch_sync"], candidates, span[-1]["work"], "predicted")

            for part in parts:
                del self.action_returns[part]

        if record["kind"] == "completion" and record["event"] in self.metadata_returns:
            self.metadata_returns[record["event"]]["completed_ns"] = record["time_ns"]
        elif record["kind"] == "action_values" and "collective" in record["values"]:
            pending = self.metadata_returns.pop(record["boundary"])
            endpoint = pending["endpoint"]
            if (
                record["group"] != endpoint["work"]["group"]
                or pending["completed_ns"] is None
                or record["time_ns"] < pending["completed_ns"]
            ):
                raise ValueError("metadata result precedes its own completed native call")

            endpoint["values"] = record["values"]["collective"]

    def edge_details(self, edges):
        result = []
        for source, target in sorted(edges)[:3]:
            record = {}
            for name, event, ports in (("source", source, self.ready), ("target", target, self.complete)):
                record[name] = {
                    "event": event,
                    "ports": [
                        {"invocation": {"group": key[0], "event": key[1]}, "scope": path}
                        for (key, path), value in ports.items()
                        if value == event
                    ],
                }

            result.append(record)

        return result

    def exchanges(self, channels, dependencies, category):
        for channel, ranks in channels.items():
            first = next(iter(ranks.values()))[0]
            members = first["members"]
            if set(ranks) != set(members) or len({len(values) for values in ranks.values()}) != 1:
                self.finding(
                    category,
                    first["work"],
                    channel=channel,
                    reason="communication participants or call counts are incomplete",
                )
                yield None
                continue

            ordered = []
            valid = True
            for rank in members:
                calls = sorted(
                    ranks[rank], key=lambda value: (dependencies[rank].positions[value["entered"]], value["order"])
                )

                # Protocol order is run-local: never align separate executions by a polling ordinal.
                if any(
                    not dependencies[rank].reaches(left["entered"], right["entered"]) for left, right in pairwise(calls)
                ):
                    self.finding(
                        category, first["work"], channel=channel, reason="calls have no established protocol order"
                    )
                    valid = False

                ordered.append(calls)

            if not valid:
                yield None
                continue

            for endpoints in zip(*ordered, strict=True):
                if any(
                    value["members"] != members
                    or value["operation"] != endpoints[0]["operation"]
                    or not set(value["required"]) <= set(members)
                    for value in endpoints
                ):
                    self.finding(
                        category, first["work"], channel=channel, reason="peers disagree on operation or members"
                    )
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
            self.batch_sync.check(endpoints, "predicted")
            compare_metadata(endpoints, "predicted", self.counts, self.finding)
            peers = {endpoint["rank"]: endpoint for endpoint in endpoints}

            for endpoint in endpoints:
                destination = self.complete.get(endpoint["scope"])
                for rank in endpoint["required"]:
                    source = self.ready.get(peers[rank]["scope"])
                    if source is None or destination is None:
                        self.finding(
                            "communication_endpoint",
                            endpoint["work"],
                            scope=endpoint["scope"][1],
                            reason="declared communication boundary is absent",
                        )
                        unresolved = True
                    else:
                        expected.add((source, destination))

        complete = self.compare_edges("communication_protocol", expected, unresolved)
        self.counts.update(
            predicted_communication_exchanges=exchanges,
            predicted_communication_edges_expected=len(expected),
            predicted_communication_edges_matched=len(expected & self.communication_edges),
            predicted_communication_protocol_complete=complete,
        )

    def native_communication(self, dependencies):
        self.native_rounds = list(self.exchanges(self.channels, dependencies, "native_evidence"))
        complete = True
        exchanges, row_pairs = 0, 0

        for endpoints in self.native_rounds:
            if endpoints is None:
                complete = False
                continue

            exchanges += 1
            self.batch_sync.check(endpoints, "native")
            compare_metadata(endpoints, "native", self.counts, self.finding)
            if endpoints[0]["operation"] == "transfer":
                send, receive = (endpoint["payload"] for endpoint in endpoints)
                if (send[0], receive[0]) != ("input", "output") or send[1:] != receive[1:]:
                    complete = False
                    self.finding(
                        "communication_payload", endpoints[0]["work"], origin="native", send=send, receive=receive
                    )

            peers = {endpoint["rank"]: endpoint for endpoint in endpoints}
            for endpoint in endpoints:

                # Actual send/receive splits must agree across peers, independently of target route derivation.
                if traffic := endpoint.get("peer_rows"):
                    for index, peer in enumerate(endpoints):
                        incoming = peer.get("peer_rows")
                        if incoming is None:
                            complete = False
                            self.finding(
                                "native_evidence",
                                endpoint["work"],
                                reason="peer exchange has no observed row counts",
                                peer=peer["rank"],
                            )
                        elif traffic["send"][index] != incoming["receive"][peer["members"].index(endpoint["rank"])]:
                            complete = False
                            self.finding(
                                "communication_payload",
                                endpoint["work"],
                                reason="native send and receive row counts disagree",
                                peer=peer["rank"],
                            )
                        else:
                            row_pairs += 1

                for rank in endpoint["required"]:
                    self.native_communication_edges.add(
                        ((rank, peers[rank]["ready"]), (endpoint["rank"], endpoint["complete"]))
                    )

        self.counts.update(
            communication_channels=len(self.channels),
            communication_exchanges=exchanges,
            native_communication_protocol_complete=complete,
            native_peer_row_pairs=row_pairs,
        )

    def compare_communication(self, pairs):
        expected, unresolved, unpaired = set(), False, 0
        payloads, payloads_compared, payloads_matched = 0, 0, 0
        for endpoints in self.native_rounds:
            if endpoints is None:
                unresolved = True
                continue

            peers = {endpoint["rank"]: endpoint for endpoint in endpoints}
            for endpoint in endpoints:
                destination_scope = ActionIdentity.mapped(endpoint["scope"], pairs)
                destination = self.complete.get(destination_scope)
                target = self.endpoints.get(destination_scope)

                if endpoint["payload"] is not None:
                    payloads += 1
                    if target is not None:
                        payloads_compared += 1
                        if endpoint["payload"] == target["payload"]:
                            payloads_matched += 1
                        else:
                            self.finding(
                                "communication_payload",
                                endpoint["work"],
                                native=endpoint["payload"],
                                predicted=target["payload"],
                            )

                if target is not None:
                    if endpoint["metadata"] and endpoint["values"] is not None and target["values"] is not None:
                        field = "metadata_operands_compared"
                        self.counts[field] = self.counts.get(field, 0) + 1
                        if endpoint["values"] != target["values"]:
                            self.finding(
                                "metadata_operands",
                                endpoint["work"],
                                native=endpoint["values"],
                                predicted=target["values"],
                            )

                    if "batch_sync" in endpoint and "batch_sync" in target:
                        self.batch_sync.counts["endpoints_compared"] += 1
                        if endpoint["batch_sync"] != target["batch_sync"]:
                            self.finding(
                                "batch_sync_operands",
                                endpoint["work"],
                                native=endpoint["batch_sync"],
                                predicted=target["batch_sync"],
                            )

                for rank in endpoint["required"]:
                    source_scope = ActionIdentity.mapped(peers[rank]["scope"], pairs)
                    if source_scope is None or destination_scope is None:
                        unpaired += 1
                        unresolved = True
                        continue

                    source = self.ready.get(source_scope)
                    if source is None or destination is None:
                        self.finding(
                            "communication_endpoint",
                            endpoint["work"],
                            scope=endpoint["scope"][1],
                            reason="mapped communication boundary is absent",
                        )
                        unresolved = True
                    else:
                        expected.add((source, destination))

        self.compare_edges("communication_dependency", expected, unresolved)
        self.counts.update(
            communication_edges_expected=len(expected),
            communication_edges_observed=len(self.communication_edges),
            communication_edges_matched=len(expected & self.communication_edges),
            communication_edges_unpaired=unpaired,
            communication_evidence_complete=not unresolved,
            device_transfer_payloads=payloads,
            device_transfer_payloads_compared=payloads_compared,
            device_transfer_payloads_matched=payloads_matched,
        )
