"""Use qualified native sources or bounded sampling to extend a release."""

import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid5

from prologue.workflow.budget import Acquisition, acquisition_usage


class Supplement:
    def __init__(self, deployment, training):
        self.deployment = deployment
        self.output = deployment.output
        self.spec = deployment.spec
        self.training = training
        self.attempts = self.output / "acquisition"

    def inside(self, path):
        return str(Path("/run") / path.relative_to(self.output))

    def run(self, release, gaps, directory):
        checks = []
        training = []
        for index, group in enumerate(gaps["structure"], 1):
            targets = [row for row in group["targets"] if row["status"] != "candidate_conflict"]
            if not targets:
                continue
            case = directory / f"structure-{index:04}"
            case.mkdir(exist_ok=True)
            domain = group["domain"]
            candidates = self.retained_sources(domain, targets, case, checks)
            result = self.publish_structure(release, domain, targets, candidates, case, checks)
            if result:
                return {**result, "checks": checks}
            training.append((domain, targets, candidates, case))
        if gaps["costs"]:
            result = self.costs(release, gaps["costs"], directory, checks)
            if result:
                return {**result, "checks": checks}
        for domain, targets, candidates, case in training:
            result = self.acquire_structure(release, domain, targets, candidates, case, checks)
            if result:
                return {**result, "checks": checks}
        return {"status": "no_qualified_evidence" if checks else "no_eligible_acquisition", "checks": checks}

    def retained_sources(self, domain, targets, directory, checks):
        known = [{"target": row["target"], "query": row["required"]} for row in targets if "required" in row]
        scope = directory / "scope.json"
        if known:
            Acquisition.write(scope, {"domain": domain, "targets": known})

        candidates = []
        for index, source in enumerate(self.spec["sources"], 1):
            if source["domain"] == domain:
                candidate = self.compile(Path(source["path"]), scope, directory / f"source-{index:04}.json", checks)
                if candidate:
                    candidates.append(candidate)
        return candidates

    def acquire_structure(self, release, domain, targets, candidates, directory, checks):
        training_target = self.training_target(domain)
        if training_target is None or self.spec["device"] is None:
            return None
        training = {"plan": self.training, "target": training_target}
        arguments = ["sample-structure", "/run/training-plan.json", "--configuration",
                     training_target["configuration"], "--workload", training_target["workload"]]
        capture, report = self.sample("structure", training, arguments, checks,
                                      network="host" if "disaggregation" in domain["configuration"] else "none")
        if report["status"] != "observed":
            return None
        observed = report["attempts"][-1]
        source = capture / "attempts" / observed["id"] / observed["native"]["observation"]
        candidate = self.compile(source, directory / "scope.json", directory / "captured.json", checks)
        if candidate is None:
            return None
        return self.publish_structure(release, domain, targets, [*candidates, candidate], directory, checks)

    def training_target(self, domain):
        if self.training is None:
            return None
        config = domain["configuration"]
        if config not in self.training["configurations"] or domain["model"] != self.training["model"]:
            return None
        return {"configuration": config["id"], "workload": self.training["workloads"][0]["id"]}

    def compile(self, source, scope, path, checks):
        if path.exists():
            return path
        args = ["compile", "/source/0"]
        if scope.exists():
            args.extend(("--for-queries", self.inside(scope)))
        try:
            candidate = self.deployment.app(args, sources=[source])
        except ValueError as error:
            checks.append({"status": "source_not_qualified", "source": str(source), "reason": str(error)})
            return None
        Acquisition.write(path, candidate)
        return path

    def publish_structure(self, release, domain, targets, candidates, directory, checks):
        if not candidates:
            return None
        required = directory / "required.json"
        queries = [{"target": row["target"], "query": row["query"]} for row in targets if "query" in row]
        if not queries:
            # Before the first native control turn, the requirement is a usable domain.
            for path in candidates:
                asset = Acquisition.read(path)
                if asset["domain"] != domain:
                    raise ValueError("the template observation changed the structure domain")
                queries.extend({"target": targets[0]["target"], "query": row["query"]} for row in asset["roots"])
        Acquisition.write(required, {"domain": domain, "targets": queries})
        args = ["--require-structure", self.inside(required)]
        for candidate in candidates:
            args.extend(("--add-library", self.inside(candidate)))
        return self.publish(release, args, [row["target"] for row in targets], directory, checks)

    def costs(self, release, targets, directory, checks):
        query = directory / "cost-targets.json"
        sampling = directory / "cost-plan.json"
        Acquisition.write(query, {"targets": targets})
        plan = self.deployment.app(["plan-costs", self.inside(query), "--store", "/library", "--release", release])
        Acquisition.write(sampling, plan)
        if self.spec["device"] is None or not plan["jobs"]:
            checks.append({"status": "no_eligible_cost_sampling", "demands": plan["demands"]})
            return None

        for index, job in enumerate(plan["jobs"]):
            case = directory / f"cost-{index + 1:04}"
            case.mkdir(exist_ok=True)
            args = ["sample-costs", self.inside(sampling), "--job", index]
            capture, report = self.sample("cost", job, args, checks)
            if report["status"] != "observed":
                continue
            fitted = case / "fitted.json"
            if not fitted.exists():
                args = ["fit-costs", "--for-queries", self.inside(capture / "plan.json")]
                for attempt in report["attempts"]:
                    for sample in attempt.get("native", {}).get("samples", []):
                        if sample["status"] == "observed":
                            args.extend(("--device", self.inside(capture / "attempts" / attempt["id"]
                                                               / sample["observation"])))
                try:
                    fit = self.deployment.app(args)
                except ValueError as error:
                    checks.append({"status": "cost_evidence_not_qualified", "reason": str(error)})
                    continue
                Acquisition.write(fitted, fit)
            affected = [row["target"] for row in targets if row["domain"] == job["domain"]
                        and any(coordinate in job["coordinates"] for coordinate in row["coordinates"])]
            args = ["--add-costs", self.inside(fitted), "--require-costs", self.inside(capture / "plan.json")]
            result = self.publish(release, args, affected, case, checks)
            if result:
                return result
        return None

    def sample(self, method, specification, args, checks, *, network="none"):
        identity = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:24]
        directory = self.attempts / f"{method}-{identity}"
        receipt = directory / "sampling.json"
        if receipt.exists():
            report = Acquisition.read(receipt)
        else:
            report = self.observe(directory, args, network)
        checks.append({"status": report["status"], "acquisition": str(directory.relative_to(self.output))})
        return directory, report

    def observe(self, directory, args, network):
        other = acquisition_usage(self.attempts, exclude=directory)
        limits = self.spec["limits"]
        attempts = limits["attempts"] - other["attempts"]
        seconds = min(limits["card_seconds"] - other["card_ns"] / 1_000_000_000,
                      self.deployment.clock.remaining() - 5)
        retained_bytes = limits["max_bytes"] - other["retained_bytes"]
        if min(attempts, int(seconds), retained_bytes) <= 0:
            return {"status": "acquisition_budget_exhausted"}

        arguments = [args[0], args[1], self.inside(directory), *args[2:],
                     "--attempts", attempts, "--card-seconds", int(seconds), "--max-bytes", retained_bytes]
        directory.mkdir(parents=True, exist_ok=True)
        try:
            report = self.deployment.observe(arguments, timeout=min(seconds + 5, self.deployment.clock.remaining()),
                                             network=network)
        except ValueError as error:
            report = {"status": "observer_failed", "reason": str(error)}
        Acquisition.write(directory / "sampling.json", report)
        return report

    def publish(self, release, args, affected, directory, checks):
        # Replaying a publication after interruption must identify the exact operation.
        operation = uuid5(UUID(self.spec["run_id"]), json.dumps([self.inside(directory), args]))
        report = self.deployment.app(["publish", "--store", "/library", "--base", release, "--operation",
                                      operation, *args], writable=True)
        checks.append(report)
        if report["status"] == "published":
            return {"status": "published", "release": report["release"],
                    "affected": list(dict.fromkeys(affected)), "publication": report}
        if report["status"] == "stale_release":
            return {"status": "stale_release", "publication": report}
        return None
