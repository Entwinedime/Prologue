"""Prepare measured time conditions and compare the result to native structure evidence."""

from prologue.acquisition.observation.batches import ObservedBatches, action_identity
from prologue.verification.reference import validate_reference


class StructureComparison:
    def __init__(self, domain, publications, manifest):
        self.arrivals = validate_reference(manifest, publications, "detailed")
        self.domain = domain
        self.workload = manifest["workload"]
        self.actions = []
        self.measurements = []
        self.indices = {}
        self.ingress = None
        self.boundaries = {(request, item["count"]): item["raw_ns"]
                           for request, rows in publications.items() for item in rows}

    def record(self, action, graph, call, work, begin_ns, end_ns, *, outputs=None):
        key = action_identity(action)
        if key in self.indices:
            raise ValueError(f"ambiguous reference timing for {key}")
        self.indices[key] = len(self.actions)

        if outputs is None:
            outputs = {member["request"]: member["output_tokens"] + 1 for member in action["members"]
                       if member.get("publishes", True)}
        self.measurements.append({
            "work": work, "prepare_ns": 0, "model_return_ns": call["end_ns"] - begin_ns,
            "result_finished": self.boundary(end_ns - begin_ns),
            "publications": {rid: self.boundary(self.boundaries[rid, count] - begin_ns)
                             for rid, count in outputs.items()},
        })
        observed = {**action, "publications": outputs, "graph": graph}
        if "resources" in call:
            observed["resources"] = call["resources"]
            observed["after_result"] = call["result_processing"]["resources"]
        if "next_read_ns" in call.get("result_processing", {}):
            self.measurements[-1]["next_read"] = self.boundary(call["result_processing"]["next_read_ns"] - begin_ns)
        self.actions.append(observed)

    @staticmethod
    def boundary(after_start_ns, after_model_ns=0):
        return {"after_start_ns": after_start_ns, "after_model_ns": after_model_ns}

    def conditions(self, query):
        key = action_identity(query["action"])
        index = self.indices.get(key)
        if index is None:
            return {"failure": {"status": "action_mismatch", "reason": f"no observed action at {key}"}}

        expected = self.actions[index]["graph"]["work"]
        if not expected:
            return {"failure": {"status": "timing_gap", "reason": f"no measured work at {key}"}}
        if query["work"] != expected:
            return {"failure": {"status": "action_mismatch", "reason": f"work differs at {key}"}}
        if query["action"].get("expert_routes") != self.actions[index].get("expert_routes"):
            return {"failure": {"status": "action_mismatch", "reason": f"expert routing differs at {key}"}}

        # The reference owns the measured graph. A selected action receives only elapsed times.
        return self.measurements[index]

    @staticmethod
    def consumed_conditions(actual, measured):
        samples = measured["work"]
        if len(samples) != len(actual["intervals"]):
            return False

        began = actual["began_ns"]
        lanes = {}
        ends = []
        graph = actual["graph"]
        for work, sample, (start, end) in zip(graph["work"], samples, actual["intervals"], strict=True):
            ready = max(lanes.get(work["lane"], began), began + sample["submitted_after_ns"])
            for predecessor in work.get("after", ()):
                ready = max(ready, ends[predecessor])
            if start != ready + sample["dispatch_ns"] or end - start != sample["duration_ns"]:
                return False
            lanes[work["lane"]] = end
            ends.append(end)

        work_ready = max(lanes.values(), default=began)
        model_work_ready = work_ready
        if "collectives" in graph:
            publisher = graph["calls"][0]["parameters"]["publication_rank"]
            model_work_ready = lanes[publisher]
        ready = max(began + measured["model_return_ns"], model_work_ready)
        if ready != actual["model_ready_ns"]:
            return False
        for publication in actual["publications"]:
            ready = max(ready, began + measured["publications"][publication["request"]]["after_start_ns"])
            if ready != publication["at_ns"]:
                return False
        finish = max(ready, work_ready, began + measured["result_finished"]["after_start_ns"])
        if "next_read" in measured:
            if actual["next_read_ns"] != max(finish, began + measured["next_read"]["after_start_ns"]):
                return False
        return actual["finished_ns"] == finish

    def compare(self, result):
        comparisons = 0
        consumed = 0
        for index, (expected, actual) in enumerate(zip(self.actions, result["actions"], strict=False)):
            for field in ("phase", "members", "resources", "after_result", "graph", "cache"):
                if field in expected and actual[field] != expected[field]:
                    return {"status": "structure_mismatch", "action": index, "field": field,
                            "matched_actions": comparisons}
            if expected["publications"] != {row["request"]: row["count"] for row in actual["publications"]}:
                return {"status": "structure_mismatch", "action": index, "field": "publications",
                        "matched_actions": comparisons}

            measured = self.measurements[index]
            if not self.consumed_conditions(actual, measured):
                return {"status": "timing_mismatch", "action": index, "matched_actions": comparisons}
            comparisons += 1
            consumed += len(measured["work"])

        status = result["status"]
        if status == "complete":
            status = "matched" if comparisons == len(self.actions) == len(result["actions"]) else "structure_mismatch"
        report = {"status": status, "reason": result["reason"], "matched_actions": comparisons,
                  "expected_actions": len(self.actions), "measured_work_consumed": consumed,
                  "finished_requests": result["finished_requests"]}
        if "gap" in result:
            report["gap"] = result["gap"]
        return report


class StructureReference(StructureComparison):
    def __init__(self, evidence, publications, manifest):
        super().__init__(evidence["domain"], publications, manifest)
        origin = manifest["delivery"]["origin_raw_ns"]
        self.starts = {}
        if "ingress" in evidence:
            observed = evidence["ingress"]
            if observed.keys() != self.arrivals.keys():
                raise ValueError("native ingress must cover the full observed workload")
            self.arrivals = {request: read - origin for request, read in observed.items()}
            self.ingress = {"arrivals": [self.arrivals]}
        self.observed = ObservedBatches(evidence).entries
        resident = {}
        for batch in self.observed:
            call = batch.call
            action = batch.action()
            result = call["result_processing"]
            self.record(action, batch.graph, call, batch.measurements(), call["begin_ns"], result["end_ns"])
            if self.ingress is not None:
                self.starts[action_identity(action)] = call["begin_ns"] - origin
                self.check_state(action, call, resident)

    def check_state(self, action, call, resident):
        phase = action["phase"]
        processing = call["result_processing"]
        cache = call["cache"]
        written = []
        for member, row in zip(action["members"], cache["rows"], strict=True):
            request = member["request"]
            extent = member["kv_committed_tokens"]
            begin = member.get("read_tokens", extent - 1 if phase == "decode" else 0)
            if len(row) != extent or not 0 <= begin < extent or row[:begin] != resident.get(request, [])[:begin]:
                raise ValueError(f"observed KV prefix changed during {phase} for {request}")
            resident[request] = row
            written.extend(row[begin:])

            count = member["output_tokens"] + 1
            publication = self.boundaries.get((request, count))
            publishes = publication is not None and processing["begin_ns"] <= publication <= processing["end_ns"]
            if publishes != member.get("publishes", True):
                raise ValueError(f"observed publication disagrees with {phase} for {request}")
        if written != cache["written"]:
            raise ValueError("native writes do not cover the batch's declared token intervals")

        page_size = self.domain["configuration"]["page_size"]
        self.actions[-1]["cache"] = {
            "rows": [[index - page_size for index in row] for row in cache["rows"]],
            "written": [index - page_size for index in written],
        }

    def conditions(self, query):
        timing = super().conditions(query)
        if self.ingress is None or "failure" in timing:
            return timing
        action = query["action"]
        start = self.starts[action_identity(action)]
        return {**timing, "prepare_ns": max(0, start - action["clock_ns"])}
