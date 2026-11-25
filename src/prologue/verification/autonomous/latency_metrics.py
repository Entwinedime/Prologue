"""Compare independent predictions with native request submission and output times."""

import argparse
from pathlib import Path

from prologue.runtime.io import read_json, write_json
from prologue.serving.latency import RequestLatency, percentile95

from ..core.metrics import compare_request_metrics
from ..core.publications import publication_marks
from ..core.reference import RecordedSubmissions, ReferenceRecording


class NativeLatencyMetrics:
    @classmethod
    def from_source(cls, source):
        instances = read_json(source / "capture.json")["instances"]
        return cls(source, instances, ReferenceRecording.sessions(source, instances))

    def __init__(self, source, instances, sessions):
        self.source = source
        records = RecordedSubmissions.read(source, sessions)
        roles = {instance["mode"]: self.read_instance(instance, record)
                 for instance, record in zip(instances, records, strict=True)}
        if len(instances) == 1 and roles.keys() == {"null"}:
            self.ingress = self.emitter = roles["null"]
        elif len(instances) == 2 and roles.keys() == {"prefill", "decode"}:
            self.ingress, self.emitter = roles["prefill"], roles["decode"]
        else:
            raise ValueError("native latency requires ordinary serving or one Prefill/Decode pair")
        intake, output = self.ingress["delivery"], self.emitter["delivery"]
        self.origin_ns = intake.origin_ns
        self.requests = list(intake.requests.values())
        if self.origin_ns != output.origin_ns or self.requests != list(output.requests.values()):
            raise ValueError("native latency roles have different clocks or workloads")
        self.specializations = None
        if all(role["specializations"] is not None for role in roles.values()):
            self.specializations = [event for role in roles.values() for event in role["specializations"]]

    def read_instance(self, instance, delivery):
        if "latency" in instance:
            measurements = read_json(self.source / instance["latency"])
            if measurements["clock"] != delivery.clock:
                raise ValueError("native latency and delivery clocks differ")
            rows = {row["id"]: row for row in measurements["requests"]}
            if len(rows) != len(measurements["requests"]) or rows.keys() != delivery.requests.keys():
                raise ValueError("native latency records must cover each submitted request exactly once")
            specializations = measurements.get("specialization_loads")
            basis = "scheduler_send_return"
        else:
            rows, specializations = self.detailed_boundaries(instance, delivery)
            basis = "detailed_scheduler_send_bracket_end"
        return {"delivery": delivery, "rows": rows, "specializations": specializations, "basis": basis}

    def detailed_boundaries(self, instance, delivery):
        rows = {name: {} for name in delivery.requests}
        receiver = publisher = None
        loads = [] if set(instance["observed_roles"]) == set(instance["native_participants"]) else None
        for rank in instance["observed_roles"]:
            raw = read_json(self.source / f"rank-{rank}.json")
            if loads is not None:
                if raw.get("specialization_loads") is None:
                    loads = None
                else:
                    loads.extend({"rank": rank, **event} for event in raw["specialization_loads"])
            if raw["request_ingress"]:
                if receiver is not None:
                    raise ValueError("multiple workers claim the same external request ingress")
                receiver = rank
                for read in delivery.receives(raw["request_ingress"]):
                    for name in read["requests"]:
                        rows[name]["ingress_raw_ns"] = delivery.origin_ns + read["time_ns"]

            # A Prefill worker hands the request to Decode. Only the final serving role publishes client tokens.
            if instance["mode"] == "prefill":
                continue
            for identity, count, begin, finish in publication_marks(raw):
                if publisher is not None and publisher != rank:
                    raise ValueError("multiple workers claim the same token publication")
                publisher = rank
                row = rows[identity]
                if count > row.get("output_tokens", 0):
                    row.setdefault("first_publication_raw_ns", finish)
                    row.setdefault("first_publication_bracket_ns", finish - begin)
                    row.update(last_publication_raw_ns=finish, last_publication_bracket_ns=finish - begin,
                               output_tokens=count)

        if receiver is None or (publisher is None and instance["mode"] != "prefill"):
            raise ValueError("detailed serving records omit the required ingress or publication owner")
        return rows, loads

    def report(self):
        requests = []
        disaggregated = self.ingress is not self.emitter
        for expected in self.requests:
            identity = expected["id"]
            published = self.emitter["rows"][identity]
            count = published["output_tokens"]
            delivery = self.ingress["delivery"].by_request[identity]
            declared, submitted = delivery["scheduled_ns"], delivery["submitted_ns"]
            received_ns = self.ingress["rows"][identity]["ingress_raw_ns"]
            arrival = received_ns - self.origin_ns
            first = published["first_publication_raw_ns"] - self.origin_ns
            last = published["last_publication_raw_ns"] - self.origin_ns
            if count != expected["output_tokens"] or not 0 <= declared <= submitted <= arrival <= first <= last:
                raise ValueError(f"native request {identity} has inconsistent delivery, output length or time")
            row = {
                **published, "id": identity, "declared_ns": declared, "submitted_ns": submitted, "ingress_ns": arrival,
                "ingress_raw_ns": received_ns,
                "first_publication_ns": first, "last_publication_ns": last, "ttft_ns": first - submitted,
                "scheduler_ttft_ns": first - arrival,
                "tpot_ns": (last - first) / (count - 1) if count > 1 else None,
            }
            if disaggregated:
                decode = self.emitter["delivery"].by_request[identity]
                decode_ingress_ns = published["ingress_raw_ns"] - self.origin_ns
                if (
                    decode["scheduled_ns"] != declared
                    or not declared <= decode["submitted_ns"] <= decode_ingress_ns <= first
                ):
                    raise ValueError(f"native request {identity} has inconsistent Decode delivery")
                row.update(decode_submitted_ns=decode["submitted_ns"], decode_ingress_ns=decode_ingress_ns)
            requests.append(row)
        total = sum(row["output_tokens"] for row in requests)
        elapsed = max(row["last_publication_ns"] for row in requests)
        return {
            "capture": str(self.source),
            "publication_basis": self.emitter["basis"],
            "origin_raw_ns": self.origin_ns,
            "ttft_basis": "prefill_submission_to_decode_publication"
            if disaggregated
            else "request_submission_to_scheduler_publication",
            "output_tokens": total,
            "tokens_per_second": total * 1e9 / elapsed if elapsed else None,
            "p95_ttft_ns": percentile95(row["ttft_ns"] for row in requests),
            "requests": requests,
            "specialization_loads": self.specializations,
        }



class LatencyComparison:
    def __init__(self, native, prediction):
        self.native = native
        self.path = prediction
        self.result = read_json(prediction)

    def report(self):
        result = self.result
        workload = [{**truth, "arrival_ns": truth["declared_ns"]} for truth in self.native["requests"]]
        prediction = RequestLatency.predicted(result, workload).report()
        return {"result": str(self.path), **prediction, **compare_request_metrics(self.native, prediction),
                "cost_queries": result["cost_queries"], "cost_query_ns": result["cost_query_ns"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--prediction", type=Path, nargs="+", help="existing result JSONs for a publication-only comparison"
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    rows = [NativeLatencyMetrics.from_source(path).report() for path in arguments.capture]
    if arguments.prediction is not None:
        for row, prediction in zip(rows, arguments.prediction, strict=True):
            row["comparison_basis"] = "token_publications_only"
            row["prediction"] = LatencyComparison(row, prediction).report()
    write_json(arguments.output, rows, indent=2, replace=False)


if __name__ == "__main__":
    main()
