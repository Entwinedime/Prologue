"""Score autonomous predictions against an independent lightweight reference."""

import json
from collections import defaultdict

from prologue.verification.reference import validate_reference


class AutonomousReference:
    def __init__(self, directory):
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        worker = json.loads((directory / "worker.json").read_text(encoding="utf-8"))
        origin = self.manifest["delivery"]["origin_raw_ns"]
        self.arrivals = validate_reference(self.manifest, worker["publications"], "light")
        self.publications = {
            rid: [{"count": row["count"], "at_ns": row["raw_ns"] - origin} for row in rows]
            for rid, rows in worker["publications"].items()
        }

    @staticmethod
    def request_metrics(rows, arrival, total):
        result = {"ttft_ns": None, "tpot_ns": None}
        if rows and rows[0]["count"] == 1:
            result["ttft_ns"] = rows[0]["at_ns"] - arrival
            if total > 1 and rows[-1]["count"] == total:
                result["tpot_ns"] = (rows[-1]["at_ns"] - rows[0]["at_ns"]) / (total - 1)
        return result

    @staticmethod
    def throughput(publications, arrivals):
        last_rows = [rows[-1] for rows in publications.values() if rows]
        if not last_rows:
            return None
        span = max(row["at_ns"] for row in last_rows) - min(arrivals.values())
        return sum(row["count"] for row in last_rows) * 1_000_000_000 / span if span > 0 else None

    def score(self, prediction):
        for key in ("framework_revision", "model", "configuration"):
            if prediction["domain"][key] != self.manifest[key]:
                raise ValueError(f"prediction and scoring reference disagree on {key}")
        if prediction["workload"] != self.manifest["workload"]:
            raise ValueError("prediction and scoring reference require the same request workload")
        if any(source["origin_raw_ns"] == self.manifest["delivery"]["origin_raw_ns"]
               for source in prediction["cost_sources"]):
            raise ValueError("the scoring reference was used in calibration")

        published = defaultdict(list)
        for action in prediction["actions"]:
            for row in action["publications"]:
                published[row["request"]].append(row)

        workload = self.manifest["workload"]["requests"]
        requests, errors = [], defaultdict(list)
        predicted_arrivals = {}
        for request in workload:
            rid, total = request["id"], request["output_tokens"]
            predicted_arrivals[rid] = request["arrival_ns"]
            observed = self.request_metrics(self.publications[rid], self.arrivals[rid], total)
            estimated = self.request_metrics(published[rid], request["arrival_ns"], total)
            row = {"request": rid}
            for metric in observed:
                truth, estimate = observed[metric], estimated[metric]
                percentage = None
                if truth is not None and truth > 0 and estimate is not None:
                    percentage = 100 * (estimate - truth) / truth
                    errors[metric].append(abs(percentage))
                row[metric] = {"observed": truth, "predicted": estimate, "error_percent": percentage}
            requests.append(row)

        # A single output has a first token but no interval between tokens.
        applicable = {"ttft_ns": len(workload),
                      "tpot_ns": sum(request["output_tokens"] > 1 for request in workload)}
        score = {}
        for metric, count in applicable.items():
            amounts = errors[metric]
            score[metric] = {"eligible": len(amounts), "applicable": count, "total": len(workload),
                             "mape_percent": sum(amounts) / len(amounts) if amounts else None}
        return {"status": prediction["status"], "requests": requests, "scores": score,
                "tokens_per_second": {"observed": self.throughput(self.publications, self.arrivals),
                                      "predicted": self.throughput(published, predicted_arrivals)}}
