"""Verify request ingress, PD polling prerequisites and transferred KV and metadata."""

import json
from collections import Counter, defaultdict

from prologue.verification.core.actions import ActionIdentity
from prologue.verification.core.adapters import kv_buffers
from prologue.verification.engine import CausalIndex, FindingLedger


class IngressEvidence:
    def __init__(self, instances, submissions):
        self.sources = {instance["id"]: entry for instance, entry in zip(instances, submissions, strict=True)}
        self.origin_ns, = {entry.origin_ns for entry in submissions}
        self.expected = {}
        self.actual = defaultdict(list)
        self.input_attempts = {}
        self.native_reads = {}
        self.actual_reads = defaultdict(list)
        self.received = defaultdict(set)
        self.findings = FindingLedger()

    def native(self, instance, raw):
        if raw["request_ingress"]:
            if instance in self.expected:
                raise ValueError("instance has more than one observed external ingress")
            entry = self.sources[instance]
            self.expected[instance] = entry.receives(raw["request_ingress"])
            self.input_attempts[instance] = self.attempts(raw)
            self.native_reads[instance] = self.reads(instance, raw, entry.aliases, entry.origin_ns)

    @staticmethod
    def attempts(raw):
        records = raw.get("request_input_attempts")
        if records is None:
            return {"status": "unobserved"}
        received, previous_end = [], None
        empty, unknown_context, control_only = 0, 0, 0
        for record in records:
            begin, end, messages = record["begin_raw_ns"], record["received_raw_ns"], record["message_count"]
            if (any(type(value) is not int or value < 0 for value in (begin, end, messages))
                    or begin > end or (previous_end is not None and begin < previous_end)):
                raise ValueError("input attempts require ordered RAW intervals and nonnegative message counts")
            previous_end = end
            context = record["context"]
            if context is not None and not isinstance(context, dict):
                raise ValueError("input attempt context must be a native state snapshot or explicitly unobserved")
            unknown_context += context is None
            empty += messages == 0
            control_only += messages > 0 and not record["requests"]
            if record["requests"]:
                if messages == 0:
                    raise ValueError("empty input attempt cannot receive workload requests")
                received.append({"requests": record["requests"], "received_raw_ns": end})
        if received != raw["request_ingress"]:
            raise ValueError("input attempts disagree with the observed external receive cohorts")

        # These are read intervals, not socket availability, CPU cost, or a schedule to replay.
        return {"status": "partial" if unknown_context else "observed", "attempts": len(records),
                "empty": empty, "control_only": control_only, "unknown_context": unknown_context}

    @staticmethod
    def reads(instance, raw, aliases, origin_ns):
        result, received = [], set()
        for record in raw.get("request_input_attempts", ()):
            members = [aliases[request] for request in record["requests"]]
            time_ns = record["received_raw_ns"] - origin_ns
            if time_ns < 0:
                raise ValueError("input read return precedes the workload clock")

            # Prior deliveries distinguish otherwise equal empty queues; neither the result nor poll ordinal is a key.
            identity = None if record["context"] is None else {
                "instance": instance, "context": record["context"], "received": sorted(received),
            }
            result.append({"identity": identity, "begin_ns": record["begin_raw_ns"] - origin_ns,
                           "time_ns": time_ns, "requests": members})
            received.update(members)
        return result

    @staticmethod
    def idle(context):
        queues = ("waiting", "running", "previous", "chunked")

        # Placement fields may remain populated even after all request progress has drained.
        return (context is not None and all(name in context and not context[name] for name in queues)
                and not ActionIdentity.request_progress(context))

    def readiness(self, instance):
        submitted = self.sources[instance].by_request
        return {request: {"submitted_ns": submitted[request]["submitted_ns"], "received_ns": read["time_ns"]}
                for read in self.expected[instance] for request in read["requests"]}

    @staticmethod
    def groups(reads):
        groups = defaultdict(list)
        for row in reads:
            if row["identity"] is not None:
                groups[json.dumps(row["identity"], sort_keys=True)].append(row)
        return groups

    def observe(self, row):
        if row["kind"] == "resources":
            self.requests = [request["id"] for request in row["requests"]]
        elif row["kind"] == "input_read":
            identity = row["identity"]
            instance = identity["instance"]
            if instance not in self.sources:
                raise ValueError("ingress names an unknown serving instance")
            indices = row["request_indices"]
            if any(type(index) is not int or not 0 <= index < len(self.requests) for index in indices):
                raise ValueError("ingress has an invalid request index")
            members = [self.requests[index] for index in indices]
            previous = self.actual_reads[instance]
            in_flow = row["in_flow"]
            if (type(in_flow) is not bool or (members and not in_flow)
                    or not isinstance(identity["context"], dict)
                        or identity["received"] != sorted(self.received[instance])
                    or len(set(members)) != len(members) or self.received[instance].intersection(members)
                    or type(row["time_ns"]) is not int or row["time_ns"] < 0
                    or (previous and row["time_ns"] < previous[-1]["time_ns"])):
                raise ValueError("input read disagrees with its entering delivery history or clock")
            observed = {"requests": members, "time_ns": row["time_ns"]}
            previous.append({"identity": identity, "in_flow": in_flow,
                "poll_deadline_ns": row.get("poll_deadline_ns"), **observed})
            self.received[instance].update(members)
            if members:
                self.actual[instance].append(observed)

    def compare_reads(self, instance, pairs):
        reads = [row for row in self.actual_reads[instance] if row["in_flow"]]
        outside = len(self.actual_reads[instance]) - len(reads)
        if self.input_attempts[instance]["status"] == "unobserved":
            self.findings.add("input_read_evidence", {"instance": instance},
                {"reason": "native input attempts were not recorded"})
            return {"status": "unobserved", "predicted_reads": len(reads), "outside_flow": outside}
        if pairs is None:
            native, actual = self.groups(self.native_reads[instance]), self.groups(reads)
        else:

            # A separately checked consumed prefix can identify otherwise repeatable reads.
            native = {str(index): [row] for index, row in enumerate(self.native_reads[instance])}
            actual = {str(source): [self.actual_reads[instance][index]] for index, source in pairs.items()}
        matched, ambiguous, unmatched_native, unmatched_predicted, differences = 0, 0, 0, 0, []
        for key in sorted(native.keys() | actual.keys()):
            left, right = native.get(key, []), actual.get(key, [])
            reason = None
            if not left:
                unmatched_predicted += len(right)
                reason = "no native read with this entering context and delivery history"
            elif not right:
                unmatched_native += len(left)
                reason = "no predicted read with this entering context and delivery history"
            elif len(left) != 1 or len(right) != 1:
                ambiguous += 1
                reason = "logical read identity repeats"
            elif pairs is None and not left[0]["requests"]:
                ambiguous += 1
                reason = "read returns no workload requests and may repeat"
            elif pairs is None and not ActionIdentity.request_progress(left[0]["identity"]["context"]):
                ambiguous += 1
                reason = "entering context has no request work anchor for an exact read time"
            elif (left[0]["identity"] == right[0]["identity"] and left[0]["time_ns"] == right[0]["time_ns"]
                  and left[0]["requests"] == right[0]["requests"]):
                matched += 1
            else:
                differences.append({"identity": left[0]["identity"], "native": left[0], "predicted": right[0]})

            # Explain incomplete correspondence without treating an unmatched read as a proven control error.
            if reason is not None:
                self.findings.add("input_read_evidence", json.loads(key), {
                    "reason": reason, "native_calls": len(left), "predicted_calls": len(right),
                })
        unknown = sum(row["identity"] is None for row in self.native_reads[instance])
        if unknown:
            self.findings.add("input_read_evidence", {"instance": instance}, {
                "reason": "native entering context was not observed", "native_calls": unknown,
            })
        incomplete = ambiguous or unmatched_native or unmatched_predicted or unknown
        return {"status": "different" if differences else "partial" if incomplete else "matched", "matched": matched,
                "outside_flow": outside,
                "ambiguous": ambiguous, "unmatched_native": unmatched_native,
                    "unmatched_predicted": unmatched_predicted,
                "unknown_native_context": unknown, "differences": differences}

    def compare(self, read_pairs=None, *, complete):
        results = {}
        for instance in sorted(self.sources.keys() | self.actual.keys()):
            native, actual = self.expected.get(instance), self.actual[instance]
            if native is None:
                results[instance] = {"status": "missing_native_evidence"}
                continue

            # A stopped run has not reached later delivery rounds; compare only its observed prefix.
            expected = native if complete else native[:len(actual)]

            # Time is paired only after request identities and receive cohorts agree, never by round number alone.
            cohorts = [row["requests"] for row in expected] == [row["requests"] for row in actual]
            differences = []
            if cohorts:
                for left, right in zip(expected, actual, strict=True):
                    if left["time_ns"] != right["time_ns"]:
                        differences.append({"requests": left["requests"], "native_ns": left["time_ns"],
                                            "predicted_ns": right["time_ns"],
                                                "delta_ns": right["time_ns"] - left["time_ns"]})
            results[instance] = {"status": "matched" if cohorts and not differences else "different",
                "cohorts_match": cohorts,
                                 "comparison_scope": "complete_history" if complete else "observed_prefix",
                                 "native_rounds": len(native), "predicted_rounds": len(actual),
                                     "time_differences": differences,
                                 "native_input_attempts": self.input_attempts[instance],
                                 "input_reads": self.compare_reads(instance, None
                                     if read_pairs is None else read_pairs[instance])}
            if not cohorts:
                results[instance].update(native_cohorts=[row["requests"] for row in native],
                                         predicted_cohorts=[row["requests"] for row in actual])
        return results


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
            endpoints.append({
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
            })
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


class PdPollingEvidence:
    """Follow per-rank polling values through reductions and Decode admission, without pairing rounds."""

    def __init__(self, members):
        self.members = members
        self.reads, self.inputs, self.latest = {}, {}, {}
        self.metadata_pending = {}
        self.collectives = {}
        self.commits = 0
        self.returns = {}
        self.decisions = {}
        self.checked, self.unobserved = 0, 0
        self.operand_differences = 0
        self.unobserved_commits, self.commit_differences = 0, 0
        self.findings = FindingLedger()

    @staticmethod
    def in_gate(work):
        return any(caller["owner"] == "decode_receive" for caller in work.get("callers", []))

    @staticmethod
    def in_poll(work):
        return any(caller["owner"] == "protocol_poll" for caller in work.get("callers", []))

    @staticmethod
    def request_order(work):
        return tuple(work["context"]["poll_requests"])

    def begin(self, group, work):
        continuation = work.get("continuation", {})
        if continuation.get("entry") != "call.enter":
            return
        if continuation.get("owner") == "protocol_poll":
            self.reads[group] = {}
            self.metadata_pending[group] = set()
        if continuation.get("owner") == "decode_receive":
            self.decisions[group] = set()
            self.latest.pop(group, None)

    def read(self, group, kind, request, result, gated):
        if kind == "transport":
            self.reads[group][request] = result
            if gated and result == 4:
                self.metadata_pending[group].add(request)
        else:
            self.metadata_pending[group].remove(request)

            # Native _apply_metadata_gate downgrades Success until the metadata row has landed.
            self.reads[group][request] = 4 if result else 3

    def operands(self, group, requests):
        if self.metadata_pending[group].intersection(requests):
            raise ValueError("Decode polling operands lack required metadata reads")
        values = self.reads[group]
        return None if values is None else [values[request] for request in requests]

    def check_operands(self, group, requests, expected, values, origin, boundary):
        if values is None:
            self.reads[group] = None
            self.decisions[group] = None
            self.unobserved += 1
            return

        # A following group reduction consumes this result, not the original local poll values.
        self.reads[group] = dict(zip(requests, values["output"], strict=True))
        self.decisions[group] = {request for request, poll in self.reads[group].items() if poll == 4}
        if expected is None:
            self.unobserved += 1
            return
        self.checked += 1
        if values["input"] != expected:
            self.operand_differences += 1
            category = "native_evidence" if origin == "native" else "polling_operands"
            self.findings.add(category, {"group": group}, {
                "reason": "PD MIN operands differ from the current reads or preceding group reduction",
                "origin": origin, "boundary": boundary, "requests": requests,
                "expected": expected, "actual": values["input"],
            })

    def commit(self, group, request, origin, boundary):
        if len(self.members[group]) == 1:
            ready = {request} if self.operands(group, (request,)) == [4] else set()
            self.reads[group].pop(request)
        else:
            ready = self.decisions[group]
        if ready is None:
            self.unobserved_commits += 1
            return
        self.commits += 1
        if request not in ready:
            self.commit_differences += 1
            self.findings.add("native_evidence" if origin == "native" else "admission_decision", {"group": group}, {
                "reason": "Decode commit does not consume a current unconsumed successful gate result",
                "origin": origin, "boundary": boundary, "request": request,
            })
        else:

            # A successful result admits this request once; a later receive call must obtain a fresh decision.
            ready.remove(request)

    def native(self, raw, group):
        queries = {query["scope"]: query for query in raw["state_queries"]}
        values = {record["scope"]: record["values"]["collective"]
                  for record in raw.get("action_values", ()) if "collective" in record["values"]}

        # Scope order is local to this actual control thread; it never pairs rounds across runs.
        for scope in raw["scopes"]:
            self.begin(group, scope["work"])
            query = queries.get(scope["id"]) if self.in_poll(scope["work"]) else None
            if query is not None and query["kind"] in {"transport", "metadata"}:
                self.read(
                    group,
                    query["kind"],
                    scope["work"]["context"]["request"]["id"],
                    query["result"],
                    self.in_gate(scope["work"]),
                )
            if scope["role"] == "metadata_collective" and self.in_poll(scope["work"]):
                requests = self.request_order(scope["work"])
                self.check_operands(
                    group, requests, self.operands(group, requests), values.get(scope["id"]), "native", scope["id"]
                )
            continuation = scope["work"].get("continuation", {})
            if continuation.get("owner") == "kv_commit" and continuation["entry"] == "call.enter":
                self.commit(group, scope["work"]["context"]["request"]["id"], "native", scope["id"])

    def observe(self, record, query, groups):
        kind = record["kind"]
        if kind == "state_query" and self.in_poll(query["work"]) and query["kind"] in {"transport", "metadata"}:
            self.read(
                query["group"],
                query["kind"],
                query["work"]["context"]["request"]["id"],
                record["result"],
                self.in_gate(query["work"]),
            )
        elif kind == "action_values" and record["boundary"] in self.returns and "collective" in record["values"]:
            group, requests, expected = self.returns.pop(record["boundary"])
            self.check_operands(
                group, requests, expected, record["values"]["collective"], "predicted", record["boundary"]
            )
        elif kind == "edge" and record["relation"] == "communication" and record["target"] in self.collectives:
            self.collectives[record["target"]]["inputs"].add(record["source"])
        elif kind == "completion" and record["event"] in self.collectives:
            self.collectives[record["event"]]["completed"] = record["time_ns"]
        elif kind == "graph":
            self.graph(record, groups)

    def graph(self, record, groups):
        graph = record["graph"]
        root = graph["scopes"][0]
        entered = next(port["event"] for port in root["ports"] if port["name"] == "call.enter")
        rank = graph["resources"][graph["events"][entered]["resource"]]["rank"]
        group = groups[rank]
        work = root["work"]
        self.begin(group, work)
        if root["role"] == "metadata_collective" and self.in_poll(work):
            requests = self.request_order(work)
            if (len(requests) != root["parameters"]["elements"]
                    or work["context"]["collective"]["reduction"] != "RedOpType.MIN"):
                raise ValueError(f"PD polling reduction differs from its request vector or MIN at {group}")
            polls = self.operands(group, requests)
            returned = next(port["event"] for port in root["ports"] if port["name"] == "call.return")
            self.returns[record["event_ids"][returned]] = group, requests, polls

            if self.in_gate(work):
                communication, members = root["communication"], self.members[group]
                if tuple(communication["members"]) != members or tuple(communication["required"]) != members:
                    raise ValueError(f"Decode admission collective differs from native participants at {group}")
                ready = {request for request, poll in zip(requests, polls, strict=True) if poll == 4}
                port, = [port for port in root["ports"] if "rendezvous" in port]
                source, completed = record["event_ids"][port["rendezvous"]], record["event_ids"][port["event"]]

                # Actual edges join Decode operands; no cross-run ordinal supplies the peers.
                self.inputs[source] = rank, requests, ready
                self.collectives[completed] = {
                    "inputs": set(),
                    "completed": None,
                    "members": members,
                    "requests": requests,
                }
                self.latest[group] = completed
        continuation = work.get("continuation", {})
        if continuation.get("owner") != "kv_commit" or continuation["entry"] != "call.enter":
            return
        request = work["context"]["request"]["id"]
        if len(self.members[group]) > 1:
            collective = self.collectives[self.latest[group]]
            operands = [self.inputs[event] for event in collective["inputs"]]
            if (
                collective["completed"] is None
                or collective["completed"] > record["time_ns"]
                or len(operands) != len(collective["members"])
                or {rank for rank, _, _ in operands} != set(collective["members"])
                or any(requests != collective["requests"] or request not in ready for _, requests, ready in operands)
            ):
                raise ValueError(f"Decode commit precedes its complete successful peer gate at {group}/{request}")
        self.commit(group, request, "predicted", record["event_ids"][entered])

    def operand_counts(self):
        return {
            "checked": self.checked,
            "unobserved": self.unobserved + len(self.returns),
            "different": self.operand_differences,
        }

    def commit_counts(self):
        return {"checked": self.commits, "unobserved": self.unobserved_commits, "different": self.commit_differences}

    def finish(self):
        return {"admission_collectives_observed": len(self.collectives), "admitted_requests_checked": self.commits,
                "polling_operands": self.operand_counts(), "admission_decisions": self.commit_counts()}


class PdPredictionEvidence:
    def __init__(self, source, readiness, origin_ns):
        self.required = source.requirements()
        self.targets = {key: {source.sessions[session] for session in service["metadata"]}
                        for key, service in source.services.items()}
        self.transfers, self.signals = {}, {}
        self.reservation_targets = {(row["publisher"], row["request"], row["peer"]): row["recipient"]
                                    for row in readiness["reservations"]}
        self.reservation_events = {}
        self.notifications = {}
        self.notification_edges = set()
        self.declarations, self.timed = {}, set()
        self.bounds = {
            (row["group"], row["request"], name): (
                row[field]["begin_raw_ns"] - origin_ns,
                row[field]["end_raw_ns"] - origin_ns,
            )
            for row in readiness["endpoints"]
            for name, field in (("kv_ready", "reservation_update"), ("kv_success", "completion_update"))
            if row[field] is not None and (name != "kv_ready" or row["remote_reservation"])
        }
        for row in readiness["reservation_receipts"]:
            time = row["raw_ns"] - origin_ns
            self.bounds[row["recipient"], row["request"], "kv_reservation_received", row["publisher"]] = (time, time)
        self.reads = 0
        self.polls = 0
        self.successes = 0
        self.last_states = {}
        self.polling = PdPollingEvidence(source.admission_members)

    @staticmethod
    def notification_key(identity):
        work = identity["work"]
        key = (identity["group"], work["request"], identity["name"])
        return (*key, work["publisher"]) if "publisher" in work else key

    def observe(self, record, query, queries):
        self.polling.observe(record, query, queries.groups)
        if record["kind"] == "graph":
            self.graph(record, queries.groups)
        elif record["kind"] == "notification_declared":
            self.declarations[record["event"]] = self.notification_key(record["identity"])
        elif record["kind"] == "boundary_time" and record["event"] in self.declarations:
            key = self.declarations[record["event"]]
            if key not in self.bounds or (record["begin_ns"], record["end_ns"]) != self.bounds[key]:
                raise ValueError("PD timing input differs from the independently observed transport boundary")
            self.timed.add(record["event"])
        elif record["kind"] == "notification_ready":
            identity = record["identity"]
            notice = self.notification_key(identity)
            key = notice[:2]
            if key not in self.required or notice in self.notifications:
                raise ValueError("PD notification has an unknown or repeated endpoint identity")
            events = set(record["prerequisites"])
            if identity["name"] == "kv_reservation_received":
                publisher = identity["work"]["publisher"]
                if events != {self.reservation_events[publisher, key[1], key[0]]}:
                    raise ValueError("PD receipt lacks its actual native send")
            elif identity["name"] == "kv_ready":
                if key not in self.targets:
                    raise ValueError("remote reservation readiness belongs to a Prefill endpoint")
                expected = {
                    self.notifications[(*key, "kv_reservation_received", publisher)]["event"]
                    for publisher in self.targets[key]
                }
                if events != expected:
                    raise ValueError("PD readiness lacks the actual native reservation publishers")
            elif identity["name"] == "kv_success":
                if key in self.targets:
                    transfer = self.transfers[key]
                    if events != transfer["events"] or transfer["metadata"] != self.targets[key]:
                        raise ValueError("PD publication lacks the complete final transfer group")
                elif events != {self.notifications[(*sender, "kv_success")]["event"] for sender in self.required[key]}:
                    raise ValueError("PD receiver visibility lacks its actual native sender publications")
            else:
                raise ValueError("PD notification has an unknown protocol transition")
            if record["event"] in self.timed:
                begin, end = self.bounds[notice]
                if not begin <= record["time_ns"] <= end:
                    raise ValueError(
                        f"PD visibility at {notice} lies outside the independently observed state-write interval"
                    )
            self.notifications[notice] = {"event": record["event"], "time_ns": record["time_ns"]}
            self.notification_edges.update((event, record["event"]) for event in events)
        elif query is not None and query["kind"] == "transport":
            request = query["work"]["context"]["request"]["id"]
            key = query["group"], request
            required = self.required[key]
            if not required:
                raise ValueError(f"PD endpoint {key} has no independent native sender evidence")

            if record["kind"] == "state_query":
                PdStateEvidence.check_read(
                    (*key, record["boundary"]),
                    record["result"],
                    query["returned_ns"],
                    record["time_ns"],
                    (
                        (state, notice["time_ns"], notice["time_ns"])
                        for name, state in (("kv_ready", 2), ("kv_success", 4))
                        if (notice := self.notifications.get((*key, name))) is not None
                    ),
                )

                # These returns are ordered within one worker, even when simulated timestamps are equal.
                if record["result"] < self.last_states.get(key, 1):
                    raise ValueError(f"PD endpoint {key} regresses from its preceding poll")
                self.last_states[key] = record["result"]
                notice = None
                if key in self.targets and record["result"] >= 2:
                    ready = self.notifications.get((*key, "kv_ready"))
                    if ready is None or ready["time_ns"] > record["time_ns"]:
                        raise ValueError(f"PD sender readiness at {key} precedes its complete reservation cohort")
                    notice = ready
                if record["result"] == 4:
                    notice = self.notifications.get((*key, "kv_success"))

                # Native and predicted query returns both include the state actually consumed, including cached states.
                if notice is not None and (notice["event"], record["boundary"]) not in queries.observation_edges:
                    raise ValueError(f"PD read at {key} lacks its actual visibility cause at return")
                self.polls += 1
            if record["kind"] == "completion_read":
                producers = self.signals[record["signal"]]
                if not producers or not producers <= required:
                    raise ValueError(f"PD endpoint {key} reads completion from another request or sender: {producers}")
                self.reads += 1
            elif record["result"] == 4:
                if (*key, "kv_success") not in self.notifications or self.notifications[(*key, "kv_success")][
                    "time_ns"
                ] > record["time_ns"]:
                    raise ValueError(f"PD Success at {key} precedes its own visible completion")

                # Cached Success still requires the entire native cohort, not just the signals polled this time.
                for sender in required:
                    transfer = self.transfers.get(sender)
                    if transfer is None or transfer["metadata"] != self.targets[sender]:
                        raise ValueError(f"PD Success at {key} lacks final metadata for sender {sender}")
                    for event in transfer["events"]:
                        completed_ns = queries.signal_completions[event]
                        if completed_ns is None or completed_ns > record["time_ns"]:
                            raise ValueError(f"PD Success at {key} precedes required transfer event {event}")
                    if key != sender and (
                        (*sender, "kv_success") not in self.notifications
                        or self.notifications[(*sender, "kv_success")]["time_ns"] > record["time_ns"]
                    ):
                        raise ValueError(f"PD receiver Success at {key} precedes notification release by {sender}")
                self.successes += 1

    def graph(self, record, groups):
        graph = record["graph"]
        root = graph["scopes"][0]
        if root["role"] == "kv_reservation":
            context = root["work"]["context"]
            publisher = groups[root["work"]["rank"]]
            request = context["request"]["id"]
            recipient = self.reservation_targets[publisher, request, context["reservation_peer"]]
            key = publisher, request, recipient
            port, = [port for port in root["ports"] if port["name"] == "reservation.ready"]
            entered = next(port["event"] for port in root["ports"] if port["name"] == "call.enter")
            if (key in self.reservation_events or port["event"] != entered
                    or port["use"] != "signal" or port["direction"] != "output"):
                raise ValueError(f"PD reservation has no unique send-entry signal: {key}")
            self.reservation_events[key] = record["event_ids"][entered]
            return
        effects = defaultdict(set)
        for scope in graph["scopes"]:
            if scope["role"] != "kv_transfer":
                continue
            for access in scope["kv_accesses"]:
                if access["access"] == "write":
                    for region in access["regions"]:
                        effects[access["finish"]].add((access["pool"], region["request"], "kv"))
            for access in scope["slot_accesses"]:
                if access["access"] == "write" and access["field"] == "metadata":
                    for request in access["requests"]:
                        effects[access["finish"]].add((access["pool"], request, "metadata"))
        if not effects:
            return
        sender = groups[root["work"]["rank"]]
        dependencies = CausalIndex.observed(graph)
        exported = set()
        for port in root["ports"]:
            if port["use"] != "signal" or port["direction"] != "output":
                continue

            # One chunk completion may join distinct KV and metadata services; every payload must reach it.
            covered = {finish for finish in effects if dependencies.reaches(finish, port["event"])}
            if not covered:
                continue
            event = record["event_ids"][port["event"]]
            producers = set()
            for target, request, kind in set().union(*(effects[finish] for finish in covered)):
                key = sender, request
                if target not in self.targets[key]:
                    raise ValueError(f"PD transfer {key} writes to an unobserved destination {target}")
                transfer = self.transfers.setdefault(key, {"events": set(), "metadata": set()})
                transfer["events"].add(event)
                if kind == "metadata":
                    if root["parameters"]["last_chunk"] != 1:
                        raise ValueError("PD metadata completion is not from the final chunk")
                    transfer["metadata"].add(target)
                producers.add(key)
            self.signals[port["signal"]] = producers
            exported.update(covered)
        if exported != effects.keys():
            raise ValueError("PD transfer effects lack exported completion evidence")

    def finish(self):
        return {
            **self.polling.finish(),
            "completion_reads_checked": self.reads,
            "protocol_polls_checked": self.polls,
            "successful_polls_checked": self.successes,
            "final_notifications_checked": sum(key[2] == "kv_success" for key in self.notifications),
            "reservation_notifications_checked": sum(key[2] == "kv_ready" for key in self.notifications),
            "reservation_receipts_checked": sum(key[2] == "kv_reservation_received" for key in self.notifications),
            "reservation_submissions_checked": len(self.reservation_events),
            "notification_intervals_checked": len(self.timed),
            "successful_endpoints": sum(state == 4 for state in self.last_states.values()),
            "scope": (
                "Each reservation receipt follows its native send entry, and Prefill readiness follows all receipts "
                "for the request. Success publication follows all native-cohort transfers; each receiver visibility "
                "event depends on its actual sender publications. Transport polls follow the visibility gates and "
                "nondecreasing worker states. Success reads and Decode admission follow their own visibility and "
                "metadata gates. Supplied times are checked independently against native receipt and state-write "
                "observations; overlapping intervals only qualify the causal order they establish."
            ),
        }


class PdPayloadEvidence:
    def __init__(self, registrations, sessions, pools):
        self.registrations = registrations
        self.groups, self.pools = sessions, pools
        self.sessions = {group: session for session, group in sessions.items()}
        self.buffers = {}
        for session, recording in registrations.items():
            self.buffers[session] = kv_buffers(recording)
        self.findings, self.counts = FindingLedger(), Counter()

    def ownership(self, contents, parent, call, source, target):
        graph = contents.graph
        scope = graph["scopes"][contents.source_ids[parent["id"]]]
        entered, = [port["event"] for port in scope["ports"] if port["name"] == "call.enter"]
        begin = graph["evidence"]["events"][entered]["time_ns"]
        end = call["host_interval"]["end_raw_ns"]
        if graph["evidence"]["clock"] != "monotonic_raw_ns" or begin is None or end < begin:
            raise ValueError("native PD ownership has no common ordered host interval")
        member, = parent["work"]["members"]
        request, chunk = member["id"], parent["native_kv_chunk"]
        transfer = call["native_transfer"]

        # A table slot is required at enqueue only. Copies subsequently retain pages and metadata rows independently.
        checks = [(source, "request", chunk.get("request_slot"), (begin, begin))]
        if transfer["kind"] == "metadata":
            checks.extend((session, "metadata", transfer[f"{side}_index"], (begin, end))
                          for session, side in ((source, "source"), (target, "target")))
        for session, field, slot, window in checks:
            pool = self.pools.get(self.groups[session])
            if pool is None or slot is None or not pool.owns(field, slot, request, window):
                self.findings.add("binding_evidence", {"group": self.groups[session], "scope": parent["id"]}, {
                    "request": request, "field": field, "slot": slot,
                    "reason": "no measured ownership interval covers the enqueue or retained native transfer",
                })
            else:
                self.counts[f"{field}_ownership_checked"] += 1

        if transfer["kind"] != "kv":
            return
        for session, side in ((source, "source"), (target, "target")):
            pool = self.pools.get(self.groups[session])
            ranges = [] if pool is None else [[page - 1, 0, pool.page_size] for page in transfer[f"{side}_pages"]]
            binding = {"slot": None, "begin": 0, "ranges": ranges}
            if pool is None or pool.resolve(binding, (begin, end), graph["evidence"]["clock"]) is None:
                self.findings.add("binding_evidence", {"group": self.groups[session], "scope": parent["id"]}, {
                    "request": request, "reason": "a native transfer page has no release-free enqueue-to-copy interval",
                })
            else:
                self.counts["queued_page_intervals_checked"] += len(binding["ranges"])

    @staticmethod
    def coverage(rectangles):
        changes = defaultdict(list)
        for begin, end, first, last in rectangles:
            if not begin < end or not first < last:
                raise ValueError("PD payload contains an empty or reversed byte rectangle")
            changes[begin].append(((first, last), 1))
            changes[end].append(((first, last), -1))
        active, result, previous = Counter(), [], None
        for position, events in sorted(changes.items()):
            if active and previous < position:
                columns = []
                for first, last in sorted(active):
                    if columns and first <= columns[-1][1]:
                        columns[-1][1] = max(columns[-1][1], last)
                    else:
                        columns.append([first, last])
                if result and result[-1][1] == previous and result[-1][2] == columns:
                    result[-1][1] = position
                else:
                    result.append([previous, position, columns])
            for column, delta in events:
                active[column] += delta
                if not active[column]:
                    del active[column]
            previous = position
        return result

    def page_map(self, transfer, member, source, target, maps):
        source_pages, target_pages = transfer["source_pages"], transfer["target_pages"]
        if len(source_pages) != len(target_pages):
            raise ValueError("native PD endpoints have different page counts")
        if self.buffers[source].page_size != self.buffers[target].page_size:
            raise ValueError("native PD endpoints have different page sizes")
        if source_pages and not transfer["blocks"]:
            raise ValueError("native PD pages have no recorded copy payload")
        for session, pages in ((source, source_pages), (target, target_pages)):
            size = self.buffers[session].page_size
            begin, count = member["context_tokens"], member["new_tokens"]
            if begin % size or len(set(pages)) != len(pages) or len(pages) != (count + size - 1) // size:
                raise ValueError("native PD page list differs from its captured page-aligned chunk")
            positions = maps[session, member["id"]]
            for index, page in enumerate(pages):
                logical = begin + index * size
                value = page * size, min(size, begin + count - logical)
                if positions.setdefault(logical, value) != value:
                    raise ValueError("one native PD chunk maps a logical page to competing physical pages")

    def raw_kv(self, transfer, member, source, target, footprints):
        misrouted = Counter()
        for first, last, length in transfer["blocks"]:
            endpoints = []
            for session, mode, address, pages in ((source, "read", first, transfer["source_pages"]),
                                                   (target, "write", last, transfer["target_pages"])):
                buffers = self.buffers[session]
                layout, size = buffers.layout, buffers.page_size
                candidates = [
                    index
                    for index, (start, count) in enumerate(zip(buffers.starts, buffers.lengths, strict=True))
                    if start <= address and address + length <= start + count
                ]
                if length <= 0 or len(candidates) != 1:
                    raise ValueError("native PD bytes do not lie in one registered buffer")

                buffer, = candidates
                layers = layout.end_layer - layout.start_layer
                width = layout.key_token_bytes if buffer < layers else layout.value_token_bytes
                component = 0 if buffer < layers else layout.key_token_bytes
                layer = layout.start_layer + buffer % layers
                endpoints.append((layer, "key" if buffer < layers else "value"))
                offset, covered = address - buffers.starts[buffer], 0
                output = footprints[session, mode, layer, member["id"]]
                for index, page in enumerate(pages):
                    begin, end = max(offset, page * size * width), min(offset + length, (page + 1) * size * width)
                    covered += max(0, end - begin)

                    # Check raw byte intersections independently of the producer's address-to-region decoder.
                    valid_tokens = max(0, min(size, member["new_tokens"] - index * size))
                    end = min(end, (page * size + valid_tokens) * width)
                    if begin >= end:
                        continue

                    head, tail = begin // width, (end - 1) // width
                    if head == tail:
                        output.append((head, head + 1, component + begin % width, component + (end - 1) % width + 1))
                    else:
                        output.append((head, head + 1, component + begin % width, component + width))
                        if head + 1 < tail:
                            output.append((head + 1, tail, component, component + width))
                        output.append((tail, tail + 1, component, component + (end - 1) % width + 1))
                if covered != length:
                    raise ValueError("native PD bytes escape the chunk's actual physical pages")

            # TP changes head slices, while a model layer's K/V identity must survive the copy.
            if endpoints[0] != endpoints[1]:
                misrouted[tuple(endpoints)] += 1
            self.counts["recorded_kv_bytes"] += length
        return misrouted

    def metadata(self, transfer, parent, source, target):
        member, = parent["work"]["members"]
        chunk = parent["native_kv_chunk"]
        if (transfer["room"], transfer["source_index"]) != (chunk["room"], chunk["metadata_index"]) or parent[
            "parameters"
        ]["last_chunk"] != 1:
            raise ValueError("native metadata transfer differs from its final submitted chunk")
        rows = []
        for session, index in ((source, transfer["source_index"]), (target, transfer["target_index"])):
            metadata = self.registrations[session]["registration"]["metadata"]
            if metadata["reservations"][str(transfer["room"])] != {"request": member["id"], "index": index}:
                raise ValueError("native metadata row differs from its request reservation")
            rows.append(
                [
                    (pointer + index * width, width)
                    for pointer, width in zip(metadata["pointers"], metadata["row_bytes"], strict=True)
                ]
            )
            if any(width <= 0 or length % width or not 0 <= index < length // width
                   for length, width in zip(metadata["lengths"], metadata["row_bytes"], strict=True)):
                raise ValueError("native metadata row escapes its registered buffer")
        if not rows[0] or [width for _, width in rows[0]] != [width for _, width in rows[1]]:
            raise ValueError("native metadata endpoints have incompatible row layouts")
        copied = defaultdict(list)
        for source_address, target_address, length in transfer["blocks"]:
            matches = [
                (index, source_address - first)
                for index, ((first, width), (last, _)) in enumerate(zip(*rows, strict=True))
                if first <= source_address
                and source_address + length <= first + width
                and target_address - last == source_address - first
            ]
            if len(matches) != 1 or length <= 0:
                raise ValueError("native metadata copy changes its field or row offset")
            index, offset = matches[0]
            copied[index].append((0, 1, offset, offset + length))
            self.counts["recorded_metadata_bytes"] += length
        if any(self.coverage(copied[index]) != [[0, 1, [[0, width]]]] for index, (_, width) in enumerate(rows[0])):
            raise ValueError("native metadata bytes omit a registered request field")
        self.counts["metadata_transfers"] += 1
        return {
            (source, "read", "metadata", member["id"], None),
            (target, "write", "metadata", member["id"], "first-token"),
        }

    def inspect(self, contents, raw, group, dependencies):
        writers = defaultdict(list)
        for original in raw["scopes"]:
            for access in original.get("state_accesses", []):
                if (access["pool"], access.get("field"), access["access"]) == ("local", "metadata", "write"):
                    for request in access["requests"]:
                        writers[request].append(original["id"])

        for index, scope in enumerate(contents.graph["scopes"]):
            calls = [
                raw["scopes"][origin]
                for origin in scope["bindings"]["source_scopes"]
                if "native_transfer" in raw["scopes"][origin]
            ]
            if not calls:
                continue
            native, declared = defaultdict(list), defaultdict(list)
            maps, slots = defaultdict(dict), set()
            for call in calls:
                transfer = call["native_transfer"]
                parent = raw["scopes"][call["submitted_by"]]
                member, = parent["work"]["members"]
                source, target = raw["kv_session"], transfer["session"]
                self.ownership(contents, parent, call, source, target)
                if transfer["kind"] == "metadata":
                    slots.update(self.metadata(transfer, parent, source, target))
                    origin, = writers[member["id"]]
                    producer = contents.graph["scopes"][contents.source_ids[origin]]
                    written, = [access["finish"] for access in producer["slot_accesses"]
                                if (access["pool"], access["field"], access["access"]) == ("local", "metadata", "write")
                                and member["id"] in access["requests"]]
                    consumed, = [access["start"] for access in scope["slot_accesses"]
                                 if access["pool"] == "local" and access["access"] == "read"]

                    # The original field write and registered copy row establish content origin, independently of
                    # prediction.
                    if not dependencies.reaches(written, consumed):
                        raise ValueError("native metadata copy loses its observed first-token producer")
                    self.counts["metadata_producers_checked"] += 1
                elif transfer["kind"] == "kv":
                    if transfer["source_pages"] != parent["native_kv_chunk"]["source_pages"]:
                        raise ValueError("native KV bytes differ from their enqueued source pages")
                    self.page_map(transfer, member, source, target, maps)
                    misrouted = self.raw_kv(transfer, member, source, target, native)
                    if misrouted:
                        self.findings.add("native_pd_payload", {"group": group, "scope": index}, {
                            "reason": "native KV copy changes the model layer or K/V component",
                            "request": member["id"], "misrouted_blocks": sum(misrouted.values()),
                            "routes": [{"source": first, "target": last, "blocks": count}
                                       for (first, last), count in sorted(misrouted.items())],
                        })
                    self.counts["kv_transfers"] += 1
                else:
                    raise ValueError("native PD transfer has an unknown payload kind")

            for access in scope["kv_accesses"]:
                session = self.sessions[contents.pool(access["pool"], group)]
                layout = self.buffers[session].layout
                interval = access["slice"]
                if (not layout.start_layer <= access["layer"] < layout.end_layer
                        or interval["token_bytes"] != layout.key_token_bytes + layout.value_token_bytes
                        or not 0 <= interval["begin"] < interval["end"] <= interval["token_bytes"]):
                    raise ValueError("declared PD access changes its registered layer/byte layout")
                for region in access["regions"]:
                    covered = 0
                    for logical, (physical, count) in maps[session, region["request"]].items():
                        begin, end = max(logical, region["begin"]), min(logical + count, region["end"])
                        if begin < end:
                            declared[session, access["access"], access["layer"], region["request"]].append(
                                (
                                    physical + begin - logical,
                                    physical + end - logical,
                                    interval["begin"],
                                    interval["end"],
                                )
                            )
                            covered += end - begin
                    if covered != region["end"] - region["begin"]:
                        raise ValueError("declared PD access escapes its native request/page mapping")

            for access in scope["slot_accesses"]:
                if (
                    access["members"] != f"metadata/{access['access']}"
                    or scope["work"][access["members"]] != access["requests"]
                ):
                    raise ValueError("declared PD metadata changes its request selector")
            actual_slots = {
                (
                    self.sessions[contents.pool(access["pool"], group)],
                    access["access"],
                    access["field"],
                    request,
                    access.get("publication"),
                )
                for access in scope["slot_accesses"]
                for request in access["requests"]
            }
            differences = [
                key
                for key in native.keys() | declared.keys()
                if self.coverage(native[key]) != self.coverage(declared[key])
            ]
            if differences or slots != actual_slots:
                self.findings.add("pd_payload", {"group": group, "scope": index}, {
                    "reason": "declared PD state differs from the actual copied byte footprint or metadata rows",
                    "kv_regions_different": len(differences), "metadata_rows_missing": len(slots - actual_slots),
                    "metadata_rows_extra": len(actual_slots - slots),
                })
            self.counts["scopes_checked"] += 1
            self.counts["kv_accesses_checked"] += len(scope["kv_accesses"])
            self.counts["metadata_accesses_checked"] += len(scope["slot_accesses"])
