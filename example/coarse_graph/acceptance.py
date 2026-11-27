"""Summarize independent references, blind spending and the three formal checks."""

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from prologue.common.io import content_id, read_json
from prologue.host.libraries.release import LibraryRelease, asset_path


def payload(record, root):
    if "report" not in record:
        return record

    path = Path(record["report"]).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"verification output escaped its matrix: {path}")
    return read_json(path)


def mape(report):
    return {field: row["mape_percent"] for field, row in report.get("accuracy", {}).items() if row["requests"]}


def origins(library):
    return set(library.templates.observations) | {
        recording for samples in read_json(library.costs)["sample_recordings"] for recording in samples
    }


class Acceptance:
    phases = ("blind", "detailed", "light", "targets", "comparisons", "fixed", "autonomous")

    def __init__(self, holdout):
        self.root = holdout.root
        self.domains = holdout.domains
        self.reports = {phase: read_json(holdout.results / f"{phase}.json") for phase in self.phases}
        names = {case["id"] for case in holdout.coordinates()}
        if any(report.keys() != names for report in self.reports.values()):
            raise ValueError("all phases must retain one terminal outcome per target")

        references = [report["attempt"] for phase in ("detailed", "light")
                      for report in self.reports[phase].values() if report["status"] == "recorded"]
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

    def case(self, domain, case, readouts):
        name = case["id"]
        blind, detailed, light, target = (self.reports[phase][name] for phase in self.phases[:4])
        structure, fixed, autonomous = (payload(self.reports[phase][name], self.root) for phase in self.phases[4:])
        release, anchors = self.origins(domain, blind)
        training = {phase: report.get("training", {}) for phase, report in
                    (("structure", structure), ("fixed", fixed), ("autonomous", autonomous))}
        expected = {"structure": detailed.get("attempt"), "fixed": detailed.get("attempt"),
                    "autonomous": light.get("attempt")}
        isolation = all(row.get("status") == "separate" and row["target_attempt"] == expected[phase]
                        for phase, row in training.items())

        baseline = fixed.get("baseline", {}).get("status")
        coverage = fixed.get("service_coverage", {})
        native, modeled = autonomous.get("native", {}), autonomous.get("prediction", {})
        error = mape(modeled)
        fixed_error = mape(fixed.get("scenarios", {}).get("modeled", {}))
        comparison_ready = self.metrics(name, case, native, modeled, readouts)
        usage = blind["usage"]
        reference_cards = sum(ref.get("card_seconds", 0) for ref in (detailed, light))
        cards = usage["card_ns"] / 1e9
        gates = {
            "references": detailed["status"] == light["status"] == "recorded" and target["status"] == "produced",
            "blind_exploration": blind["status"] == "complete",
            "structure": structure["status"] == "full_structure_matched"
                         and structure["libraries"] == blind["libraries"],
            "fixed": fixed["status"] == "compared" and baseline == "measured_boundaries_matched"
                     and fixed["cost_asset"] == str(release.costs) and bool(coverage)
                     and all(row["work"] == row["measured"] == row["modeled"] for row in coverage.values()),
            "autonomous": autonomous["status"] == "compared" and autonomous["libraries"] == blind["libraries"]
                          and set(error) == {"ttft_ns", "tpot_ns"} and all(0 <= value < 6 for value in error.values()),
            "isolation": isolation,
        }

        passed = all(gates.values())
        return {
            "case": name, "domain": domain["id"], "gates": gates, "passed": passed, "training": training,
            "structure_status": structure["status"], "structure_actions": structure.get("actions_compared"),
            "fixed_status": fixed["status"], "fixed_baseline": baseline, "fixed_mape": fixed_error,
            "fixed_coverage": coverage, "autonomous_status": autonomous["status"], "autonomous_mape": error,
            "throughput_relative_error_percent": comparison_ready,
            "initial_hot": blind["initial_hot"], "pure_hot_passed": passed and not (usage["native_attempts"] or cards),
            "initial_gap": blind.get("initial_gap"), "local_attempts": usage["native_attempts"],
            "local_card_seconds": cards, "endpoint_observations": anchors,
            "reference_card_seconds": {"detailed": detailed.get("card_seconds"), "light": light.get("card_seconds")},
            "local_share_of_full_reference": cards / reference_cards if reference_cards else None,
        }

    def metrics(self, name, case, evidence, predicted, readouts):
        throughput = evidence.get("tokens_per_second")
        if not throughput or "tokens_per_second" not in predicted:
            return None

        source = read_json(self.root / case["prediction_input"])
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
        successful = sum(case["passed"] for case in cases)
        passing = successful == len(cases) and all(row["consistent"] for group in rankings for row in group.values())

        return {"status": "passed" if passing else "unqualified", "threshold_percent": 6,
                "cases_passed": successful, "cases_total": len(cases), "cases": cases,
                "initial_hot_coverage": sum(case["initial_hot"] for case in cases) / len(cases),
                "isolation_passed": sum(case["gates"]["isolation"] for case in cases),
                "cross_configuration_rankings": rankings}
