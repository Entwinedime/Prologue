"""Build unique action identities and preserve query result witnesses."""

import json
from collections import Counter, defaultdict


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
            self.observe_graph(record)
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
            return self.observe_result(record)

    def observe_graph(self, record):
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

    def observe_result(self, record):
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

    @staticmethod
    def mapped(label, pairs):
        key = pairs.get(label[0])
        return None if key is None else (key, *label[1:])
