"""Compare independent predictions with native request submission and output times."""

import argparse
import json
from pathlib import Path
from statistics import mean

from prologue.artifacts import write_json
from prologue.serving.latency import RequestLatency, percentile95

from .reference import ReferenceRecording


class NativeLatencyMetrics:
    @classmethod
    def from_source(cls, source):
        instances = cls.read(source / "capture.json")["instances"]
        return cls(source, instances, ReferenceRecording.sessions(source, instances))

    def __init__(self, source, instances, sessions):
        self.source = source
        roles = {instance["mode"]: self.read_instance(index, instance, sessions[index])
                 for index, instance in enumerate(instances)}
        if len(instances) == 1 and roles.keys() == {"null"}:
            self.ingress = self.emitter = roles["null"]
        elif len(instances) == 2 and roles.keys() == {"prefill", "decode"}:
            self.ingress, self.emitter = roles["prefill"], roles["decode"]
        else:
            raise ValueError("native latency requires ordinary serving or one Prefill/Decode pair")
        self.origin_ns = self.ingress["origin_ns"]
        self.requests = self.ingress["workload"]
        if self.origin_ns != self.emitter["origin_ns"] or self.requests != self.emitter["workload"]:
            raise ValueError("native latency roles have different clocks or workloads")
        self.specializations = None
        if all(role["specializations"] is not None for role in roles.values()):
            self.specializations = [event for role in roles.values() for event in role["specializations"]]

    def read_instance(self, index, instance, session):
        directory = self.source / "instances" / str(index)
        requests = session["content"]["requests"]
        delivery, = self.read(directory / "request-delivery.json")["capture"]
        if delivery["clock"] != "monotonic_raw_ns":
            raise ValueError("native latency requires the RAW workload clock")
        submissions = {index: row for row in delivery["submissions"] for index in row["requests"]}
        specializations = None
        if "latency" in instance:
            measurements = self.read(self.source / instance["latency"])
            if measurements["clock"] != delivery["clock"]:
                raise ValueError("native latency and delivery clocks differ")
            rows = {row["id"]: row for row in measurements["requests"]}
            specializations = measurements.get("specialization_loads")
            basis = "scheduler_send_return"
        else:
            if instance["mode"] != "null":
                raise ValueError("PD latency reporting requires a latency-only capture for both instances")
            aliases = dict(zip(session["source_requests"], (request["id"] for request in requests), strict=True))
            rows = self.detailed_publications(instance["observed_roles"], aliases)
            basis = "detailed_scheduler_send_bracket_end"
        return {
            "workload": requests, "origin_ns": delivery["origin_ns"], "submissions": submissions,
            "rows": rows, "specializations": specializations, "basis": basis,
        }

    @staticmethod
    def read(path):
        return json.loads(path.read_text())

    def detailed_publications(self, ranks, aliases):
        rows = None
        for rank in ranks:
            raw = self.read(self.source / f"rank-{rank}.json")
            if not any(scope.get("publication") for scope in raw["scopes"]):
                continue
            if rows is not None:
                raise ValueError("detailed serving records have more than one observed output publisher")
            rows = self.observed_publications(raw, aliases)
        if rows is None:
            raise ValueError("detailed serving records do not include an output publisher")
        return rows

    @staticmethod
    def observed_publications(raw, aliases):
        host = {row["name"]: row for row in raw["host_observation"]["intervals"]}
        rows = {aliases[identity]: {"ingress_raw_ns": pull["received_raw_ns"]}
                for pull in raw["request_ingress"] for identity in pull["requests"]}
        for scope in raw["scopes"]:
            if not scope.get("publication"):
                continue
            marker = host[f"prologue.scope/{scope['id']}"]
            finish = marker["end_raw_ns"]
            bracket = finish - marker["begin_raw_ns"]
            for item in scope["publication"]:
                row = rows[item["id"]]
                count = item["generated_tokens"]
                if count > row.get("output_tokens", 0):
                    row.setdefault("first_publication_raw_ns", finish)
                    row.setdefault("first_publication_bracket_ns", bracket)
                    row.update(last_publication_raw_ns=finish, last_publication_bracket_ns=bracket, output_tokens=count)
        return rows

    def report(self):
        requests = []
        disaggregated = self.ingress is not self.emitter
        for index, expected in enumerate(self.requests):
            identity = expected["id"]
            published = self.emitter["rows"][identity]
            count = published["output_tokens"]
            delivery = self.ingress["submissions"][index]
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
                decode = self.emitter["submissions"][index]
                decode_ingress_ns = published["ingress_raw_ns"] - self.origin_ns
                if decode["scheduled_ns"] != declared or not declared <= decode["submitted_ns"] <= decode_ingress_ns <= first:
                    raise ValueError(f"native request {identity} has inconsistent Decode delivery")
                row.update(decode_submitted_ns=decode["submitted_ns"], decode_ingress_ns=decode_ingress_ns)
            requests.append(row)
        total = sum(row["output_tokens"] for row in requests)
        elapsed = max(row["last_publication_ns"] for row in requests)
        return {"capture": str(self.source), "publication_basis": self.emitter["basis"], "origin_raw_ns": self.origin_ns,
                "ttft_basis": "prefill_submission_to_decode_publication" if disaggregated else "request_submission_to_scheduler_publication",
                "output_tokens": total, "tokens_per_second": total * 1e9 / elapsed if elapsed else None,
                "p95_ttft_ns": percentile95(row["ttft_ns"] for row in requests), "requests": requests,
                "specialization_loads": self.specializations}


def compare_request_metrics(reference, predicted):
    actual = {row["id"]: row for row in reference["requests"]}
    estimates = {row["id"]: row for row in predicted["requests"]}
    if len(actual) != len(reference["requests"]) or len(estimates) != len(predicted["requests"]) or actual.keys() != estimates.keys():
        raise ValueError("request metrics require the same unique request identities")

    differences = {identity: {"id": identity} for identity in actual}
    accuracy = {}
    for metric in ("ttft_ns", "tpot_ns"):
        errors = []
        zero = missing = 0
        for identity, row in actual.items():
            observed, estimate = row[metric], estimates[identity][metric]
            if (observed is None) != (estimate is None):
                raise ValueError(f"request {identity} changes the applicability of {metric}")
            delta = None if observed is None else estimate - observed
            differences[identity][metric] = delta

            # A one-token request has no TPOT. A measured zero has a signed error but no relative denominator.
            if observed is None:
                missing += 1
            elif observed == 0:
                zero += 1
            else:
                errors.append(abs(delta) / observed)
        accuracy[metric] = {"mape_percent": mean(errors) * 100 if errors else None, "requests": len(errors),
                            "total_requests": len(actual), "not_applicable": missing, "zero_denominators": zero}

    observed_rate, predicted_rate = reference["tokens_per_second"], predicted["tokens_per_second"]
    return {"accuracy": accuracy, "delta_from_baseline": {
        "requests": list(differences.values()), "p95_ttft_ns": predicted["p95_ttft_ns"] - reference["p95_ttft_ns"],
        "tokens_per_second": None if observed_rate is None or predicted_rate is None else predicted_rate - observed_rate,
    }}


class LatencyComparison:
    def __init__(self, native, prediction):
        self.native = native
        self.path = prediction
        self.result = NativeLatencyMetrics.read(prediction)

    def report(self):
        result = self.result
        workload = [{**truth, "arrival_ns": truth["declared_ns"]} for truth in self.native["requests"]]
        prediction = RequestLatency.predicted(result, workload).report()
        return {"result": str(self.path), **prediction, **compare_request_metrics(self.native, prediction),
                "cost_queries": result["cost_queries"], "cost_query_ns": result["cost_query_ns"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction", type=Path, nargs="+", help="existing result JSONs for a publication-only comparison")
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
