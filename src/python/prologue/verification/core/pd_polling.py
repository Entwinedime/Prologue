"""Check receiver gate, poll and admission results against witnessed state."""

from prologue.verification.engine import FindingLedger


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
