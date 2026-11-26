"""Compare predicted PD protocol progress with the independently observed readiness."""

from collections import defaultdict

from prologue.verification.core.pd_polling import PdPollingEvidence
from prologue.verification.core.protocol import PdStateEvidence
from prologue.verification.engine import CausalIndex


class PdPredictionEvidence:
    """Tie simulated reservation visibility and completion reads to their independently observed native transfer
    cohort.
    """

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
            self.observe_notification(record)
        elif query is not None and query["kind"] == "transport":
            self.observe_transport(record, query, queries)

    def observe_notification(self, record):
        """Check visibility against the exact native transport prerequisites."""

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

    def observe_transport(self, record, query, queries):
        """Check that reads consume a state the native transfer cohort could have published."""

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
