"""Preserve measured external readiness while replaying work on a fixed native graph."""


class FixedGraph:
    def __init__(self, reference, hardware):
        self.domain = {**reference.domain, "hardware": hardware}
        self.batches = []
        self.actual = []
        self.kinds = set()
        self.calibration_width_ns, fixed_work = reference.fixed_work()
        self.publications_in_clock_interval = 0
        previous_end = 0

        for begin, batch, measured, (work, intervals) in zip(reference.starts, reference.actions,
                                                             reference.measurements, fixed_work, strict=True):
            graph = batch["graph"]
            action = {key: batch[key] for key in ("phase", "members", "expert_routes") if key in batch}
            self.kinds.update(node["kind"] for node in graph["work"])

            device_ready = max(end for _, end in intervals)
            output_ready = max(end for node, (_, end) in zip(graph["work"], intervals, strict=True)
                               if reference.publisher is None or node["lane"] == reference.publisher)
            model_ready = max(output_ready, begin + measured["model_return_ns"])
            model_tail = model_ready - output_ready
            input_ready = max(reference.arrivals[member["request"]] for member in action["members"])
            prepare = begin - max(input_ready, previous_end)
            finish = begin + measured["result_finished"]["after_start_ns"]
            published = {rid: begin + boundary["after_start_ns"] for rid, boundary in measured["publications"].items()}
            self.publications_in_clock_interval += sum(
                output_ready <= time < output_ready + self.calibration_width_ns for time in published.values()
            )
            if min(prepare, finish - model_ready, *(time - model_ready for time in published.values())) < 0:
                raise ValueError("native action and publication boundaries do not have a causal fixed-graph order")

            # The measured result tail follows modeled device completion in replays.
            timing = {
                "work": work, "prepare_ns": prepare, "model_return_ns": 0,
                "result_finished": reference.boundary(0, finish - model_ready),
                "publications": {rid: reference.boundary(0, time - model_ready) for rid, time in published.items()},
            }
            self.batches.append({"action": action, "graph": graph, "timing": timing,
                                 "input_ready_ns": input_ready, "model_tail_ns": model_tail})
            self.actual.append({"began_ns": begin, "device_ready_ns": device_ready, "model_ready_ns": model_ready,
                                "finished_ns": finish, "intervals": intervals, "publications": published})
            previous_end = finish

    def job(self, costs=None, rules=None, *, device_only=False, restore_kind=None):
        result = {"domain": self.domain, "batches": self.batches, "restore_kind": restore_kind}
        if costs is not None:
            result.update(costs=costs, rules=rules, device_only=device_only)
        return result

    def check_baseline(self, output):
        worst = 0
        for expected, action in zip(self.actual, output["actions"], strict=True):
            if (action["intervals"] != expected["intervals"] or action["began_ns"] != expected["began_ns"]
                    or action["device_ready_ns"] != expected["device_ready_ns"]):
                return {"status": "device_baseline_mismatch"}
            if {row["request"] for row in action["publications"]} != expected["publications"].keys():
                return {"status": "publication_baseline_mismatch"}
            for row in action["publications"]:
                worst = max(worst, abs(row["at_ns"] - expected["publications"][row["request"]]))
            worst = max(worst, abs(action["finished_ns"] - expected["finished_ns"]),
                        abs(action["model_ready_ns"] - expected["model_ready_ns"]))
        return {
            "status": "matched" if worst == 0 else "boundary_baseline_mismatch",
            "max_boundary_error_ns": worst,
            "max_clock_alignment_width_ns": self.calibration_width_ns,
            "publications_in_clock_interval": self.publications_in_clock_interval,
            "actions": len(self.actual),
        }

    def compare(self, output):
        if output["status"] != "complete":
            return output
        differences = []
        for batch, expected, action in zip(self.batches, self.actual, output["actions"], strict=True):
            ready = batch["input_ready_ns"]
            differences.append({
                "model_start_error_ns": action["began_ns"] - expected["began_ns"],
                "device_ready_error_ns": action["device_ready_ns"] - expected["device_ready_ns"],
                "model_ready_error_ns": action["model_ready_ns"] - expected["model_ready_ns"],
                "finished_error_ns": action["finished_ns"] - expected["finished_ns"],
                "input_to_begin_ns": {"observed": expected["began_ns"] - ready,
                                      "estimated": action["began_ns"] - ready},
                "publication_error_ns": {row["request"]: row["at_ns"] - expected["publications"][row["request"]]
                                         for row in action["publications"]},
            })
        return {"status": "complete", "actions": differences}
