"""Summarize independent references, blind spending and the three formal checks."""

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from holdout import digest
from prologue.common.io import content_id, read_json
from prologue.host.libraries.release import LibraryRelease, asset_path
from prologue.host.workflow.latency import RequestLatency, percentile95
from prologue.verification.core import compare_request_metrics

LATENCY_FIELDS = ("ttft_ns", "tpot_ns")
RANK_FIELDS = {"tokens_per_second", "p95_ttft_ns", "mean_tpot_ns"}


def payload(record, root):
    if "report" not in record:
        return record

    path = Path(record["report"]).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"verification output escaped its matrix: {path}")
    if "sha256" in record and digest(path) != record["sha256"]:
        raise ValueError(f"sealed prediction changed after the blind phase: {path}")
    return read_json(path)


def mape(report):
    return {field: row["mape_percent"] for field, row in report.get("accuracy", {}).items() if row["requests"]}


def origins(library):
    return set(library.templates.observations) | {
        recording for samples in read_json(library.costs)["sample_recordings"] for recording in samples
    }


def median_reference(native):
    requests = []
    for rows in zip(*(run["requests"] for run in native), strict=True):
        identity = rows[0]["id"]
        if any(row["id"] != identity for row in rows):
            raise ValueError("repeated references must retain request order")

        row = {"id": identity}
        for field in LATENCY_FIELDS:
            values = [sample[field] for sample in rows]
            if any(value is None for value in values) and any(value is not None for value in values):
                raise ValueError(f"repeated references disagree on {field} applicability for {identity}")
            row[field] = None if values[0] is None else median(values)
        requests.append(row)

    return {"requests": requests, "p95_ttft_ns": percentile95(row["ttft_ns"] for row in requests),
            "tokens_per_second": median(run["tokens_per_second"] for run in native)}


class Acceptance:
    phases = ("blind", "detailed", "light", "targets", "comparisons", "fixed", "autonomous")

    def __init__(self, holdout):
        self.root = holdout.root
        self.domains = holdout.domains
        self.reports = {phase: read_json(holdout.results / f"{phase}.json") for phase in self.phases}
        names = {case["id"] for case in holdout.coordinates()}
        if any(report.keys() != names for report in self.reports.values()):
            raise ValueError("all phases must retain one terminal outcome per target")

        recordings = list(self.reports["detailed"].values())
        recordings.extend(run for report in self.reports["light"].values() for run in report.get("references", []))
        references = [record["attempt"] for record in recordings if record["status"] == "recorded"]
        if len(references) != len(set(references)):
            raise ValueError("full-service references must be distinct physical attempts")

        self.references = set(references)

    def origins(self, domain, blind):
        baseline = LibraryRelease(asset_path(self.root, domain["libraries"]))
        release = LibraryRelease(Path(blind["libraries"]))
        if (baseline.identity, release.identity) != (blind["initial_release"], blind["final_release"]):
            raise ValueError("acceptance is not using the sealed blind release and baseline")

        sessions = Path(blind["sessions"]).resolve()
        if not sessions.is_relative_to(self.root):
            raise ValueError("blind acquisition session escaped its matrix")

        recorded, anchors = set(), Counter()
        for path in sessions.glob("*/session/result.json"):
            receipt = read_json(path)
            if receipt["status"] in {"qualified", "incomplete"}:
                recorded.add(receipt["native_recording"])
            for answer in receipt.get("answers", []):
                anchors["attention_contexts"] += len(answer.get("endpoint_contexts", []))
                anchors["copy_payloads"] += len(answer.get("endpoint_bytes", []))

        initial, final = origins(baseline), origins(release)
        if self.references & (initial | final | recorded) or final - initial - recorded:
            raise ValueError("training origins cross independent references or exceed owned blind sessions")

        return release, dict(anchors)

    @staticmethod
    def ranking(readouts):
        comparable = []
        for cohort in readouts.values():
            if len(cohort) < 2:
                continue

            groups = {}
            for field in cohort[0][1].keys() & cohort[0][2].keys():
                ranks = []
                for lane in (1, 2):
                    ties = defaultdict(list)
                    for row in cohort:
                        ties[row[lane][field]].append(row[0])
                    ranks.append([sorted(ties[x]) for x in sorted(ties, reverse=field == "tokens_per_second")])
                groups[field] = {"measured": ranks[0], "modeled": ranks[1], "consistent": ranks[0] == ranks[1]}
            comparable.append(groups)

        return comparable

    def latency(self, case, blind, reports, recordings, readouts):
        checks_by_run = reports.get("references", [])
        runs = [payload(row, self.root) for row in checks_by_run]
        checks = [row.get("training", {}) for row in runs]
        count = len(case["latency_observations"])
        result = {"status": reports["status"], "error": {}, "throughput_error": None, "training": checks}
        references = []
        for index, record in enumerate(recordings):
            run = runs[index] if index < len(runs) else {}
            entry = record | {"verification_status": run.get("status", "not-compared"),
                              "mape": mape(run.get("prediction", {}))}
            if run:
                entry["verification"] = checks_by_run[index].get("report")
            references.append(entry)
        result["references"] = references

        result["isolated"] = len(recordings) == len(checks) == count and all(
            self.separate(check, record.get("attempt")) for check, record in zip(checks, recordings, strict=True)
        )
        if (reports["status"] != "compared" or blind["status"] != "complete" or not result["isolated"]
                or any(run["libraries"] != blind["libraries"] for run in runs)):
            return result

        source = read_json(asset_path(self.root, case["prediction_input"]))
        prediction = RequestLatency.predicted(payload(blind["prediction"], self.root),
                                              source["workload"]["requests"]).report()
        if any(any(run["prediction"][name] != value for name, value in prediction.items()) for run in runs):
            return result | {"status": "blind_prediction_mismatch"}

        native = median_reference([run["native"] for run in runs])
        result["error"] = mape(compare_request_metrics(native, prediction))
        result["throughput_error"] = self.metrics(case["id"], source, native, prediction, readouts)
        return result

    @staticmethod
    def separate(report, attempt):
        return bool(attempt) and report.get("status") == "separate" and report.get("target_attempt") == attempt

    def case(self, domain, case, readouts):
        name = case["id"]
        blind, detailed, light, target = (self.reports[phase][name] for phase in self.phases[:4])
        structure, fixed = (payload(self.reports[phase][name], self.root) for phase in self.phases[4:6])
        release, anchors = self.origins(domain, blind)
        references = light.get("references", [])
        latency = self.latency(case, blind, self.reports["autonomous"][name], references, readouts)
        training = {"structure": structure.get("training", {}), "fixed": fixed.get("training", {}),
                    "autonomous": latency["training"]}
        isolation = latency["isolated"] and all(self.separate(training[phase], detailed.get("attempt"))
                                                for phase in ("structure", "fixed"))

        baseline = fixed.get("baseline", {}).get("status")
        coverage = fixed.get("service_coverage", {})
        error = latency["error"]
        fixed_error = mape(fixed.get("scenarios", {}).get("modeled", {}))
        usage = blind["usage"]
        light_cards = sum(ref.get("card_seconds", 0) for ref in references)
        reference_cards = detailed.get("card_seconds", 0) + light_cards
        cards = usage["card_ns"] / 1e9
        gates = {
            "references": detailed["status"] == light["status"] == "recorded" and target["status"] == "produced",
            "blind_exploration": blind["status"] == "complete",
            "structure": structure["status"] == "full_structure_matched"
                         and structure["libraries"] == blind["libraries"],
            "fixed": fixed["status"] == "compared" and baseline == "measured_boundaries_matched"
                     and fixed["cost_asset"] == str(release.costs) and bool(coverage)
                     and all(row["work"] == row["measured"] == row["modeled"] for row in coverage.values()),
            "autonomous": latency["status"] == "compared" and set(error) == set(LATENCY_FIELDS)
                          and all(0 <= value < 6 for value in error.values()),
            "isolation": isolation,
        }

        passed = all(gates.values())
        return {
            "case": name, "domain": domain["id"], "gates": gates, "passed": passed, "training": training,
            "structure_status": structure["status"], "structure_actions": structure.get("actions_compared"),
            "fixed_status": fixed["status"], "fixed_baseline": baseline, "fixed_mape": fixed_error,
            "fixed_coverage": coverage, "autonomous_status": latency["status"], "autonomous_mape": error,
            "latency_repeats": len(case["latency_observations"]), "latency_references": latency["references"],
            "throughput_relative_error_percent": latency["throughput_error"],
            "initial_hot": blind["initial_hot"], "pure_hot_passed": passed and not (usage["native_attempts"] or cards),
            "initial_gap": blind.get("initial_gap"), "local_attempts": usage["native_attempts"],
            "local_card_seconds": cards, "endpoint_observations": anchors,
            "reference_card_seconds": {"detailed": detailed.get("card_seconds"), "light": light_cards},
            "local_share_of_full_reference": cards / reference_cards if reference_cards else None,
        }

    @staticmethod
    def metrics(name, source, evidence, predicted, readouts):
        throughput = evidence.get("tokens_per_second")
        if not throughput or "tokens_per_second" not in predicted:
            return None

        workload = {k: v for k, v in source["workload"].items() if k != "id"}
        cohort = content_id([source["model"], workload])
        native = {key: evidence[key] for key in ("p95_ttft_ns", "tokens_per_second")}
        if tpots := [row["tpot_ns"] for row in evidence["requests"] if row.get("tpot_ns") is not None]:
            native["mean_tpot_ns"] = mean(tpots)

        readouts[cohort].append((name, native, {key: predicted[key] for key in native if key in predicted}))
        return (predicted["tokens_per_second"] - throughput) / throughput * 100

    def report(self):
        readouts = defaultdict(list)
        cases = [self.case(domain, case, readouts) for domain in self.domains for case in domain["targets"]]
        rankings = self.ranking(readouts)
        declared = defaultdict(set)
        for domain in self.domains:
            for case in domain["targets"]:
                if group := case.get("draw_group"):
                    declared[group].add(case["id"])

        # A shared draw with missing metrics or a split input cohort must fail the ranking gate.
        expected = {frozenset(members) for members in declared.values() if len(members) > 1}
        ranked = {frozenset(name for tie in group["tokens_per_second"]["measured"] for name in tie)
                  for group in rankings if RANK_FIELDS <= group.keys()}
        complete_rankings = expected <= ranked
        successful = sum(case["passed"] for case in cases)
        passing = (successful == len(cases) and complete_rankings
                   and all(row["consistent"] for group in rankings for row in group.values()))

        return {"status": "passed" if passing else "unqualified", "threshold_percent": 6,
                "cases_passed": successful, "cases_total": len(cases), "cases": cases,
                "initial_hot_coverage": sum(case["initial_hot"] for case in cases) / len(cases),
                "isolation_passed": sum(case["gates"]["isolation"] for case in cases),
                "cross_configuration_ranking_complete": complete_rankings,
                "cross_configuration_rankings": rankings}
