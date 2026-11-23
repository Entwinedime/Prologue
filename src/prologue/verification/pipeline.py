"""Compare the dynamically issued pipeline against independent physical stage observations."""

from collections import defaultdict

from prologue.acquisition.observation.batches import action_identity
from prologue.integrations.platforms.ascend.collectives import PhysicalRank
from prologue.verification.reference import validate_reference

_PHYSICAL_CUTS = {"collective", "signal_record", "signal_wait", "transfer_post", "transfer_wait"}


class ObservedRank:
    def __init__(self, trace, origin):
        self.calls = trace.evidence["calls"]
        self.origin = origin
        self.offset = trace.offset_ns
        self.children = defaultdict(list)
        self.work_by_call = defaultdict(list)
        for call in self.calls:
            self.children[call["parent"]].append(call["id"])
        for work in trace.evidence["work"]:
            self.work_by_call[work["owner"]].append(work)

        # A receiver rejoins the schedule stream at its observed native Work.wait.
        self.schedule_lane, = {
            item["lane"]
            for call in self.calls if call["operation"] == "transfer_wait"
            if self.calls[call["parameters"]["post"]]["parameters"]["direction"] == "receive"
            for item in self.work_by_call[call["id"]] if item["kind"] == "EVENT_WAIT"
        }

    def device_time(self, timestamp):
        return timestamp + self.offset - self.origin

    def interval(self, item):
        return {"begin_ns": self.device_time(item["begin_ns"]), "end_ns": self.device_time(item["end_ns"])}

    def events(self, scope, kind):
        return [item for item in self.work_by_call[scope] if item["kind"] == kind]

    def timing(self, identity):
        call = self.calls[identity]
        op, parameters = call["operation"], call["parameters"]
        ready = 0
        if op == "signal_record":
            event, = self.events(identity, "EVENT_RECORD")
            ready = self.device_time(event["end_ns"])
        if op == "signal_wait":
            producer, = self.events(parameters["producer"], "EVENT_RECORD")
            gates = [producer, *self.events(identity, "EVENT_WAIT")]
            ready = max(self.device_time(event["end_ns"]) for event in gates)
            if parameters["host"] and call["end_ns"] - self.origin < ready:
                raise ValueError("native result wakes before its recorded device signal")
        if op == "transfer_wait":
            ready = max((self.device_time(item["end_ns"]) for item in self.events(identity, "EVENT_WAIT")),
                        default=0)
        return {"begin_ns": call["begin_ns"] - self.origin, "end_ns": call["end_ns"] - self.origin,
                "ready_ns": ready}


class ObservedCut:
    def __init__(self, rank, root):
        self.rank = rank
        self.scopes = {}
        self.collect(root, ())
        local = sorted((work for identity in self.scopes.values()
                        if rank.calls[identity]["operation"] not in _PHYSICAL_CUTS
                        for work in rank.work_by_call[identity]), key=lambda work: work["id"])
        lanes = {}
        self.work = defaultdict(list)
        for work in local:
            lanes.setdefault(work["lane"], len(lanes))
            self.work[work["owner"]].append(work)
        for entries in self.work.values():
            entries.sort(key=lambda work: (work["slot"], lanes[work["lane"]]))
        self.compute_lane = rank.schedule_lane if not lanes else next(iter(lanes))
        if len(lanes) > 1 and any(rank.calls[scope]["operation"] == "collective" for scope in self.scopes.values()):
            raise ValueError("multiple streams cross a pipeline collective's observed compute scope")

    def collect(self, scope, path):
        self.scopes[path] = scope
        for index, child in enumerate(self.rank.children[scope]):
            self.collect(child, (*path, index))

    def timing(self):
        return self.rank.timing(self.scopes[()])

    def sample(self, path, work):
        scope = self.scopes[tuple(path)]
        kind = work["kind"]
        if kind == "collective":
            record, wait = [item for item in self.rank.work_by_call[scope]
                            if item["lane"] == self.compute_lane and item["kind"] in ("EVENT_RECORD", "EVENT_WAIT")]
            if record["kind"] != "EVENT_RECORD" or wait["kind"] != "EVENT_WAIT":
                raise ValueError("pipeline collective device gates are out of order")
            return {"begin_ns": self.rank.device_time(record["end_ns"]),
                    "end_ns": self.rank.device_time(wait["end_ns"])}
        if kind == "transfer":
            event, = self.rank.events(scope, "EVENT_WAIT")
            return self.rank.interval(event)
        observed = self.work[scope][work["ordinal"]]
        if (observed["slot"], observed["kind"], observed["name"]) != (work["slot"], kind, work["name"]):
            raise ValueError(f"physical work differs at scope {scope}")
        return self.rank.interval(observed)


class PipelineReference:
    def __init__(self, recording):
        if recording.manifest["placement"] != "full" or len(recording.windows) != 1:
            raise ValueError("pipeline comparison requires a simultaneous physical recording")
        self.traces = {rank: PhysicalRank(recording, rank) for rank in recording.bindings()}
        publisher = self.traces[recording.publication_rank()]
        manifest = {**recording.manifest, "delivery": recording.delivery}
        arrivals = validate_reference(manifest, publisher.publications, "detailed")
        self.origin = recording.delivery["origin_raw_ns"]
        self.domain = recording.domain
        self.workload = {**manifest["workload"], "requests": [
            {**request, "arrival_ns": arrivals[request["id"]]}
            for request in manifest["workload"]["requests"]
        ]}
        self.ranks = {rank: ObservedRank(trace, self.origin) for rank, trace in self.traces.items()}
        self.publications = {
            (request, row["count"]): row["raw_ns"] - self.origin
            for request, entries in publisher.publications.items() for row in entries
        }
        self.expected = {}

    def prepare(self, programs, graphs):
        for program, outcome in zip(programs, graphs, strict=True):
            if outcome["status"] != "covered":
                raise ValueError(f"physical pipeline has an unresolved program: {outcome['status']}")
            graph = outcome["graph"]
            cuts = [[ObservedCut(self.ranks[scope["rank"]], scope["scope"]) for scope in step["scopes"]]
                    for step in program["steps"]]
            work = []
            for entry in graph["work"]:
                cut, member, *path = entry["path"]
                work.append(cuts[cut][member].sample(path, entry))
            for collective in graph["collectives"]:
                # Transport only begins once all contributors satisfy their real stream gate.
                readings = [work[index] for index in collective["results"]]
                ready = max(sample["begin_ns"] for sample in readings)
                for sample in readings:
                    sample["begin_ns"] = ready
                    if sample["end_ns"] < ready:
                        raise ValueError("a collective result predates the observed group readiness")

            publications = {member["request"]: self.publications[member["request"], member["output_tokens"] + 1]
                            for member in program["members"]}
            measurements = {"work": work, "steps": [[cut.timing() for cut in step] for step in cuts],
                            "publications": publications}
            key = action_identity(program)
            self.expected[key] = graph, measurements, self.resources(program)

    def resources(self, program):
        resources = {}
        for step in program["steps"]:
            name = step["name"]
            if name not in ("launch", "process"):
                continue
            state = resources.setdefault(str(step["stage"]), {})
            for scope in step["scopes"]:
                observed = self.ranks[scope["rank"]].calls[scope["scope"]]
                if name == "launch":
                    members = [{**member, "slot": member["slot"] - 1} for member in observed["members"]]
                    launch = {"members": members, "resources": observed["resources"]}
                    if state and any(state[key] != item for key, item in launch.items()):
                        raise ValueError("same-stage TP ranks disagree on native selection or resources")
                    state.update(launch)
                elif state.setdefault("after_result", observed["resources"]) != observed["resources"]:
                    raise ValueError("same-stage TP ranks disagree on resource release")
        return resources

    def conditions(self, query):
        key = action_identity(query["action"])
        reference = self.expected.get(key)
        if reference is None or query["program"] != reference[0]:
            return {"failure": {"status": "action_mismatch", "reason": f"pipeline program differs at {key}"}}
        return reference[1]

    def compare(self, result):
        report = {"status": result["status"], "reason": result["reason"], "expected_actions": len(self.expected),
                  "matched_actions": 0, "measured_work_consumed": 0, "finished_requests": result["finished_requests"],
                  "max_work_shift_ns": 0, "max_publication_shift_ns": 0, "transfers": 0,
                  "measured_wakeups": 0, "max_wakeup_shift_ns": 0,
                  "peak_inflight": result["peak_inflight"]}
        seen = set()
        for actual in result["actions"]:
            key = action_identity(actual)
            expected, measured, resources = self.expected[key]
            if key in seen or actual["graph"] != expected or actual["actors"] != resources:
                return {**report, "status": "structure_mismatch", "action": key}
            seen.add(key)

            for index, ((begin, end), sample) in enumerate(zip(actual["intervals"], measured["work"], strict=True)):
                if end - begin != sample["end_ns"] - sample["begin_ns"]:
                    return {**report, "status": "timing_mismatch", "action": key, "work": index}
                delta = abs(begin - sample["begin_ns"])
                if delta > report["max_work_shift_ns"]:
                    node = expected["work"][index]
                    cut = expected["steps"][node["path"][0]]
                    report["max_work_shift_ns"] = delta
                    report["work_shift_at"] = {
                        "action": key, "cut": cut["name"], "rank": node["rank"], "work": node["name"],
                        "measured_begin_ns": sample["begin_ns"], "actual_begin_ns": begin,
                    }
            actual_publications = {row["request"]: row["at_ns"] for row in actual["publications"]}
            if actual_publications.keys() != measured["publications"].keys():
                return {**report, "status": "structure_mismatch", "action": key, "field": "publications"}
            for request, stamp in measured["publications"].items():
                report["max_publication_shift_ns"] = max(report["max_publication_shift_ns"],
                                                        abs(actual_publications[request] - stamp))
            for receipt in actual["cuts"]:
                step = receipt["step"]
                if expected["steps"][step]["name"] != "result_wait":
                    continue
                for (_, returned, ready), sample in zip(receipt["boundaries"], measured["steps"][step], strict=True):
                    report["max_wakeup_shift_ns"] = max(report["max_wakeup_shift_ns"],
                                                        abs(returned - sample["end_ns"]),
                                                        abs(ready - sample["ready_ns"]))
                    report["measured_wakeups"] += 1
            report["matched_actions"] += 1
            report["transfers"] += len(expected["transfers"])
            report["measured_work_consumed"] += len(measured["work"])

        if result["status"] == "complete":
            if len(seen) != len(self.expected):
                report["status"] = "structure_mismatch"
            elif report["max_work_shift_ns"] or report["max_publication_shift_ns"] or report["max_wakeup_shift_ns"]:
                report["status"] = "timing_mismatch"
            else:
                report["status"] = "matched"
        if "gap" in result:
            report["gap"] = result["gap"]
        return report
