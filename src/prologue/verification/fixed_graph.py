"""Preserve measured external readiness while replaying work on a fixed native graph."""


class FixedGraph:
    def __init__(self, reference, manifest, hardware):
        self.domain = {**reference.domain, "hardware": hardware}
        self.batches = []
        self.actual = []
        self.kinds = set()
        lower = max(batch.call["end_ns"] - batch.profile_end_ns for batch in reference.observed)
        upper = min(batch.call["begin_ns"] - batch.profile_begin_ns for batch in reference.observed)
        if lower > upper:
            raise ValueError("native action markers do not share a clock alignment")
        self.calibration_width_ns = upper - lower
        self.publications_in_clock_interval = 0
        delivery = manifest["delivery"]
        origin = delivery["origin_raw_ns"]
        arrivals = reference.arrivals
        previous_end = 0

        for batch, measured in zip(reference.observed, reference.measurements, strict=True):
            call = batch.call
            action = batch.action()
            shift = lower - (call["begin_ns"] - batch.profile_begin_ns)
            self.kinds.update(work["kind"] for work in batch.graph["work"])

            begin = call["begin_ns"] - origin
            true_intervals = [[row["begin_ns"] + lower - origin, row["end_ns"] + lower - origin]
                              for row in batch.device]
            device_ready = max(end for _, end in true_intervals)
            native_finish = call["end_ns"] - origin
            model_tail = max(0, native_finish - device_ready)
            model_ready = device_ready + model_tail
            input_ready = max(arrivals[member["request"]] for member in action["members"])
            prepare = begin - max(input_ready, previous_end)
            finish = begin + measured["result_finished"]["after_start_ns"]
            published = {rid: begin + boundary["after_start_ns"] for rid, boundary in measured["publications"].items()}
            self.publications_in_clock_interval += sum(
                device_ready <= time < device_ready + self.calibration_width_ns for time in published.values()
            )
            if min(prepare, finish - model_ready, *(time - model_ready for time in published.values())) < 0:
                raise ValueError("native action and publication boundaries do not have a causal fixed-graph order")

            # The measured result tail follows modeled device completion in replays.
            work = [{**row, "submitted_after_ns": row["submitted_after_ns"] + shift} for row in measured["work"]]
            timing = {
                "work": work, "prepare_ns": prepare, "model_return_ns": 0,
                "result_finished": reference.boundary(0, finish - model_ready),
                "publications": {rid: reference.boundary(0, time - model_ready) for rid, time in published.items()},
            }
            self.batches.append({"action": action, "graph": batch.graph, "timing": timing,
                                 "input_ready_ns": input_ready, "model_tail_ns": model_tail})
            self.actual.append({"began_ns": begin, "device_ready_ns": device_ready, "finished_ns": finish,
                                "intervals": true_intervals, "publications": published})
            previous_end = finish

    def job(self, costs=None, rules=None, restore_kind=None):
        result = {"domain": self.domain, "batches": self.batches, "restore_kind": restore_kind}
        if costs is not None:
            result["costs"] = costs
            result["rules"] = rules
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
            worst = max(worst, abs(action["finished_ns"] - expected["finished_ns"]))
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
        for expected, action in zip(self.actual, output["actions"], strict=True):
            differences.append({
                "model_start_error_ns": action["began_ns"] - expected["began_ns"],
                "device_ready_error_ns": action["device_ready_ns"] - expected["device_ready_ns"],
                "publication_error_ns": {row["request"]: row["at_ns"] - expected["publications"][row["request"]]
                                         for row in action["publications"]},
            })
        return {"status": "complete", "actions": differences}
