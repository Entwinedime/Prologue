"""Verify request ingress, PD polling prerequisites and transferred KV and metadata."""

from collections import defaultdict


class PdStateEvidence:
    """Check native state visibility and transport prerequisites independently of polling counts."""

    def __init__(self):
        self.sessions, self.peer_counts, self.services = {}, {}, {}
        self.queries = defaultdict(list)
        self.states = defaultdict(list)
        self.admission_members = {}
        self.addresses, self.publications = {}, {}
        self.receipts = {}

    def register(self, registrations, groups):
        for session, recording in registrations.items():
            group = groups[recording["parallel"]["rank"]]
            if self.sessions.setdefault(session, group) != group:
                raise ValueError("PD session disagrees with its initialized worker")
            endpoint = tuple(recording["registration"]["control_endpoint"])
            if self.addresses.setdefault(endpoint, group) != group:
                raise ValueError("PD control endpoint disagrees with its initialized worker")

    def observe(self, raw, group):
        if raw["kv_session"] is not None:
            session = raw["kv_session"]
            if session in self.sessions:
                raise ValueError("PD workers share a native session identity")

            self.sessions[session] = group

        if raw["kv_endpoint"] is not None:
            address = tuple(raw["kv_endpoint"])
            if address in self.addresses:
                raise ValueError("PD workers share a native control endpoint")

            self.addresses[address] = group

        self.peer_counts[group] = len(raw["environment"].get("kv_peers", {}))
        environment = raw["environment"]
        if environment["implementation"]["disaggregation_mode"] == "decode":

            # DecodeTransferQueue uses this GroupCoordinator's CPU group, with the same declared members.
            role = environment["worker_roles"][str(environment["parallel"]["rank"])]
            self.admission_members[group] = tuple(role["groups"]["attn_tp"]["members"])

        for update in raw["protocol_states"]:
            self.states[group, update["request"]].append(update)
        for receipt in raw.get("reservation_receipts", ()):
            key = group, receipt["request"], receipt["session"]
            if key in self.receipts:
                raise ValueError(f"native reservation received twice at {key}")

            self.receipts[key] = receipt["received_raw_ns"]

        for query in raw["state_queries"]:
            if query["kind"] != "transport":
                continue
            request = raw["scopes"][query["scope"]]["work"]["context"]["request"]["id"]
            self.queries[group, request].append(query)

        for scope in raw["scopes"]:
            if "reservation_publication" in scope:
                publication = scope["reservation_publication"]
                context = scope["work"]["context"]
                key = group, publication["request"], context["reservation_peer"]

                if (key in self.publications or scope["role"] != "kv_reservation" or not scope["complete"]
                        or context["request"]["id"] != publication["request"]
                        or not 0 <= publication["begin_raw_ns"] <= publication["end_raw_ns"]):
                    raise ValueError(f"PD reservation lacks a unique completed native send at {key}")

                self.publications[key] = publication

            if scope["role"] == "metadata_collective" and any(
                caller["owner"] == "decode_receive" for caller in scope["work"].get("callers", [])
            ):
                members = tuple(scope["communication"]["members"])
                if self.admission_members[group] != members:
                    raise ValueError(f"native Decode admission group changes within {group}")

            if "native_transfer" not in scope:
                continue
            parent = raw["scopes"][scope["submitted_by"]]
            member, = parent["work"]["members"]
            key = group, member["id"]
            service = self.services.setdefault(key, {"end_ns": 0, "targets": set(), "metadata": set()})
            interval = scope["host_interval"]

            if not scope["complete"] or interval["begin_raw_ns"] > interval["end_raw_ns"]:
                raise ValueError(f"PD service {group}/{scope['id']} has no ordered completed interval")
            service["end_ns"] = max(service["end_ns"], interval["end_raw_ns"])
            transfer = scope["native_transfer"]
            target = transfer["session"]
            service["targets"].add(target)

            if transfer["kind"] == "metadata":
                if not parent["parameters"]["last_chunk"] or target in service["metadata"]:
                    raise ValueError(f"PD request {key} has repeated or non-final metadata")
                service["metadata"].add(target)

    def requirements(self):
        required = defaultdict(set)
        for key, service in self.services.items():
            sender, request = key
            if service["targets"] != service["metadata"] or len(service["metadata"]) != self.peer_counts[sender]:
                raise ValueError(f"PD request {key} lacks final metadata for its observed peer interfaces")

            required[key].add(key)
            for session in service["metadata"]:

                # A sender publishes Success only after its whole destination cohort has returned.
                required[self.sessions[session], request].add(key)

        return required

    def causal_dependencies(self, sources, groups):
        services, reservations = defaultdict(set), defaultdict(set)
        for rank, (contents, raw, _) in sources.items():
            group = groups[rank]
            for observed in raw["scopes"]:
                scope = contents.graph["scopes"][contents.source_ids[observed["id"]]]
                if "native_transfer" in observed:
                    member, = raw["scopes"][observed["submitted_by"]]["work"]["members"]
                    (port,) = [
                        port
                        for port in scope["ports"]
                        if port["name"] == "service.complete" and port["direction"] == "output"
                    ]
                    services[group, member["id"]].add((rank, port["event"]))

                if "reservation_publication" in observed:
                    publication = observed["reservation_publication"]
                    recipient = self.addresses[tuple(publication["target"])]
                    (port,) = [
                        port
                        for port in scope["ports"]
                        if port["name"] == "reservation.ready" and port["direction"] == "output"
                    ]
                    reservations[recipient, publication["request"]].add((rank, port["event"]))

        required = self.requirements()
        dependencies = set()
        for rank, (contents, raw, _) in sources.items():
            for query in raw["state_queries"]:
                if query["kind"] != "transport" or query["result"] < 2:
                    continue
                observed = raw["scopes"][query["scope"]]
                key = groups[rank], observed["work"]["context"]["request"]["id"]
                scope = contents.graph["scopes"][contents.source_ids[query["scope"]]]
                (returned,) = [
                    port for port in scope["ports"] if port["name"] == "call.return" and port["direction"] == "output"
                ]

                # A successful native read proves these prerequisites, without inventing a notification timestamp.
                predecessors = set(reservations[key])
                if query["result"] == 4:
                    for sender in required[key]:
                        predecessors.update(services[sender])

                dependencies.update((source, (rank, returned["event"]), "observation") for source in predecessors)

        return dependencies

    @staticmethod
    def state_windows(key, records):
        windows = {}
        for record in records:
            begin, end = record["begin_raw_ns"], record["end_raw_ns"]
            before, after = record["before"], record["after"]

            # CommonKVManager takes max(old, new) for nonfailed states; snapshots may enclose concurrent writers.
            if (
                type(begin) is not int
                or type(end) is not int
                or not 0 <= begin <= end
                or type(after) is not int
                or after not in (1, 2, 3, 4)
                or (before is not None and (type(before) is not int or before not in (1, 2, 3, 4) or before > after))
            ):
                raise ValueError(f"PD state write lacks an ordered nonfailed monotonic interval at {key}")

            for state in (2, 3, 4):
                if (0 if before is None else before) < state <= after:

                    # Several overlapping observations can bound the same transition, not create several transitions.
                    previous = windows.get(state, {"begin_raw_ns": begin, "end_raw_ns": end})
                    window = {"begin_raw_ns": max(begin, previous["begin_raw_ns"]),
                              "end_raw_ns": min(end, previous["end_raw_ns"])}

                    if window["begin_raw_ns"] > window["end_raw_ns"]:
                        raise ValueError(f"PD state {state} has incompatible transition intervals at {key}")
                    windows[state] = window

        # A later monotonic state cannot become visible before an earlier state's feasible interval.
        earliest = 0
        for state in sorted(windows):
            earliest = max(earliest, windows[state]["begin_raw_ns"])
            if earliest > windows[state]["end_raw_ns"]:
                raise ValueError(f"PD state transitions contradict their monotonic order at {key}")

        return windows

    @staticmethod
    def check_read(identity, result, begin, end, transitions):
        if type(result) is not int or result not in (1, 2, 3, 4) or not 0 <= begin <= end:
            raise ValueError(f"PD read has an invalid nonfailed state or interval at {identity}")

        # Reads constrain one shared monotonic transition, rather than choosing a different time for every poll.
        bounds = {state: (lower, upper) for state, lower, upper in transitions}
        feasible, earliest = [], 0
        for state in (2, 3, 4):

            # An unmeasured intermediate transition still has to satisfy every read of this endpoint.
            lower, upper = bounds.get(state, (0, None))
            if result < state:
                lower = max(lower, begin)
            else:
                upper = end if upper is None else min(upper, end)

            earliest = max(earliest, lower)
            if upper is not None and earliest > upper:
                raise ValueError(f"PD read at {identity} contradicts state {state} visibility interval")
            feasible.append((state, earliest, upper))

        return feasible

    def inspect_endpoint(self, group, request, sources, windows, updates):
        """Validate a native endpoint from its observed writes, reads and required senders."""

        queries = self.queries.get((group, request), ())
        service_end = max(self.services[source]["end_ns"] for source in sources)
        update = updates.get((group, request))

        if update is not None:
            if update["end_raw_ns"] < service_end:
                raise ValueError(f"PD completion update at {group}/{request} precedes required service return")
            for source in sources:
                if (
                    source != (group, request)
                    and source in updates
                    and update["end_raw_ns"] < updates[source]["begin_raw_ns"]
                ):
                    raise ValueError(f"PD receiver completion at {group}/{request} precedes sender publication")

        successes, pending_after_service = [], 0
        feasible = [(state, window["begin_raw_ns"], window["end_raw_ns"])
                    for state, window in windows.get((group, request), {}).items()]
        for query in queries:
            begin, end = query["begin_raw_ns"], query["end_raw_ns"]
            feasible = self.check_read((group, request, query["scope"]), query["result"], begin, end, feasible)

            # SGLang's KVPoll ABI: Failed=0, Bootstrapping=1, WaitingForInput=2, Transferring=3, Success=4.
            if query["result"] == 4:
                if end < service_end:
                    raise ValueError(
                        f"PD Success at {group}/scope {query['scope']} precedes required service return by "
                        f"{service_end - end} ns"
                    )

                successes.append(query)
            else:
                if successes:
                    raise ValueError(f"PD endpoint {group}/{request} regresses after cached Success")
                pending_after_service += begin > service_end

        first = min(successes, key=lambda query: query["begin_raw_ns"]) if successes else None
        return {
            "group": group, "request": request, "required_senders": sorted(source[0] for source in sources),
            "remote_reservation": (group, request) in self.services,
            "service_end_raw_ns": service_end, "queries": len(queries), "success_reads": len(successes),
            "pending_reads_after_service": pending_after_service,
            "reservation_update": windows.get((group, request), {}).get(2),
            "completion_update": None if update is None else {
                field: update[field] for field in ("begin_raw_ns", "end_raw_ns")
            },
            "first_success": None if first is None else {
                field: first[field] for field in ("scope", "begin_raw_ns", "end_raw_ns")
            },
        }


    def inspect(self):
        required = self.requirements()
        if self.states.keys() - required.keys():
            raise ValueError("PD state write has no observed request/transfer participants")
        if self.queries.keys() - required.keys():
            raise ValueError("PD query has no observed request/transfer participants")

        windows = {key: self.state_windows(key, records) for key, records in self.states.items()}
        updates = {key: values[4] for key, values in windows.items() if 4 in values}

        publications, published, unobserved_peers = [], {}, []
        required_publications = {
            (self.sessions[session], request, sender)
            for (sender, request), service in self.services.items() for session in service["metadata"]
        }

        for (publisher, request, peer), publication in self.publications.items():
            recipient = self.addresses[tuple(publication["target"])]
            edge = publisher, request, recipient
            if edge in published or (edge not in required_publications and recipient in self.peer_counts):
                raise ValueError(f"PD reservation does not match one actual native transfer peer: {edge}")

            # Registered peers can lack a selected source profile; their transfers remain unverified, not mismatched.
            if recipient not in self.peer_counts:
                unobserved_peers.append({"publisher": publisher, "request": request, "recipient": recipient})
            published[edge] = publication["begin_raw_ns"]
            ready = windows.get((recipient, request), {}).get(2)

            # The remote write may precede send return, but cannot precede invocation of any required send.
            if ready is not None and ready["end_raw_ns"] < publication["begin_raw_ns"]:
                raise ValueError(f"PD readiness precedes its reservation submission: {edge}")
            publications.append({"publisher": publisher, "request": request, "recipient": recipient, "peer": peer})

        receipts, received = [], set()
        for (recipient, request, session), time in self.receipts.items():
            publisher = self.sessions[session]
            edge = publisher, request, recipient
            if edge not in required_publications or (edge in published and time < published[edge]):
                raise ValueError(f"native reservation receipt has no causal send at {edge}")

            ready = windows.get((recipient, request), {}).get(2)
            if ready is not None and ready["end_raw_ns"] < time:
                raise ValueError(f"PD readiness precedes its reservation receipt: {edge}")
            receipts.append({"publisher": publisher, "request": request, "recipient": recipient, "raw_ns": time})
            received.add(edge)

        endpoints = []

        # Transfers define the endpoints to check; missing reads cannot remove a required participant.
        for (group, request), sources in sorted(required.items()):
            endpoints.append(self.inspect_endpoint(group, request, sources, windows, updates))

        def missing_receipts(actual):
            return [{"publisher": sender, "request": request, "recipient": recipient}
                    for sender, request, recipient in sorted(required_publications - actual)]

        result = {
            "status": "observed_bounds_consistent" if endpoints else "not_applicable",
            "endpoints": endpoints, "reservations": publications, "reservation_receipts": receipts,
            "unobserved_transfer_peers": unobserved_peers,
            "missing_publications": missing_receipts(published.keys()),
            "missing_reservation_receipts": missing_receipts(received),
            "completion_updates_checked": sum(row["completion_update"] is not None for row in endpoints),
            "reservation_updates_checked": sum(row["reservation_update"] is not None for row in endpoints),
            "state_writes_checked": sum(map(len, self.states.values())),
            "scope": "Recorded reservations, monotonic state-write intervals and service-return prerequisites.",
        }
        gaps = (("reservation_update", "missing_reservation_updates", None),
                ("completion_update", "missing_completion_updates", None),
                ("success_reads", "missing_success_reads", 0))

        for field, missing, absent in gaps:
            result[missing] = [{"group": row["group"], "request": row["request"]}
                               for row in endpoints if row[field] == absent]
        return result
