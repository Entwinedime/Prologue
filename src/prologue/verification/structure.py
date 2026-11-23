"""Prepare measured time conditions and compare the result to native structure evidence."""

from prologue.acquisition.observation.batches import ObservedBatches
from prologue.verification.reference import validate_reference


class StructureReference:
    def __init__(self, evidence, publications, manifest):
        self.arrivals = validate_reference(manifest, publications, "detailed")
        self.domain = evidence["domain"]
        self.workload = manifest["workload"]
        self.observed = ObservedBatches(evidence).entries
        self.actions = []
        self.measurements = []
        self.indices = {}
        boundaries = {(request, item["count"]): item["raw_ns"]
                      for request, rows in publications.items() for item in rows}
        for batch in self.observed:
            action = batch.action()
            call = batch.call
            outputs = {member["request"]: member["output_tokens"] + 1 for member in action["members"]}
            start = call["begin_ns"]
            result = call["result_processing"]
            key = self.action_key(action)
            if key in self.indices:
                raise ValueError(f"ambiguous reference timing for {key}")
            self.indices[key] = len(self.actions)

            self.measurements.append({
                "work": batch.measurements(), "prepare_ns": 0,
                "model_return_ns": call["end_ns"] - start,
                "result_finished": self.boundary(result["end_ns"] - start),
                "publications": {rid: self.boundary(boundaries[rid, count] - start) for rid, count in outputs.items()},
            })
            self.actions.append({
                **action, "resources": call["resources"], "after_result": result["resources"],
                "publications": outputs, "graph": batch.graph,
            })

    @staticmethod
    def boundary(after_start_ns, after_model_ns=0):
        return {"after_start_ns": after_start_ns, "after_model_ns": after_model_ns}

    @staticmethod
    def action_key(action):
        return action["phase"], tuple((member["request"], member["kv_committed_tokens"], member["output_tokens"])
                                      for member in action["members"])

    def conditions(self, query):
        key = self.action_key(query["action"])
        index = self.indices.get(key)
        if index is None:
            return {"failure": {"status": "action_mismatch", "reason": f"no observed action at {key}"}}

        expected = self.actions[index]["graph"]["work"]
        if not expected:
            return {"failure": {"status": "timing_gap", "reason": f"no measured work at {key}"}}
        if query["work"] != expected:
            return {"failure": {"status": "action_mismatch", "reason": f"work differs at {key}"}}

        # The reference owns the measured graph. A selected action receives only elapsed times.
        return self.measurements[index]

    @staticmethod
    def consumed_conditions(actual, measured):
        samples = measured["work"]
        if len(samples) != len(actual["intervals"]):
            return False

        began = actual["began_ns"]
        lanes = {}
        for work, sample, (start, end) in zip(actual["graph"]["work"], samples, actual["intervals"], strict=True):
            ready = max(lanes.get(work["lane"], began), began + sample["submitted_after_ns"])
            if start != ready + sample["dispatch_ns"] or end - start != sample["duration_ns"]:
                return False
            lanes[work["lane"]] = end

        ready = max(began + measured["model_return_ns"], max(lanes.values(), default=began))
        if ready != actual["model_ready_ns"]:
            return False
        for publication in actual["publications"]:
            ready = max(ready, began + measured["publications"][publication["request"]]["after_start_ns"])
            if ready != publication["at_ns"]:
                return False
        return actual["finished_ns"] == max(ready, began + measured["result_finished"]["after_start_ns"])

    def compare(self, result):
        comparisons = 0
        consumed = 0
        for index, (expected, actual) in enumerate(zip(self.actions, result["actions"], strict=False)):
            for field in ("phase", "members", "resources", "after_result", "graph"):
                if actual[field] != expected[field]:
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
