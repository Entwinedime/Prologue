"""Plan finite source jobs for implementation domains and declared input opportunities."""

import json
from itertools import product
from pathlib import Path

from prologue.integrations.frameworks.sglang.configuration import NativeConfiguration, ServingInput
from prologue.integrations.frameworks.sglang.acquisition.domains import SourceDomain
from prologue.integrations.frameworks.sglang.acquisition.local_sources import LocalSources
from prologue.integrations.frameworks.sglang.acquisition.sources import RepresentativeSources


class SourcePlanning:
    def __init__(self, source, *, joint_observation=True):
        if source.keys() != {"framework", "model", "configurations", "workloads"}:
            raise ValueError("source planning needs framework, model, configurations and workloads")
        if source["framework"] != "sglang":
            raise ValueError("source planning currently requires the SGLang adapter")
        self.source = source
        self.joint_observation = joint_observation
        self.configurations = self.named(source["configurations"], "configurations")
        self.workloads = self.named(source["workloads"], "workloads")
        self.workloads = {identity: ServingInput.workload_input(workload) for identity, workload in self.workloads.items()}
        self.capture_inputs = {}
        for configuration in self.configurations.values():
            if configuration.keys() != {"id", "instances"}:
                raise ValueError("an exploration configuration needs id and complete serving instances")
        self.expert_count = None
        if any(NativeConfiguration(instance["configuration"]).ep > 1
               for configuration in self.configurations.values() for instance in configuration["instances"]):
            metadata = json.loads((Path(source["model"]["path"]) / "config.json").read_text())
            counts = {metadata[name] for name in ("num_experts", "num_local_experts") if name in metadata}
            if len(counts) != 1:
                raise ValueError("expert source planning needs one native model expert count")
            self.expert_count, = counts
        self.representatives = RepresentativeSources(self.workloads, self.expert_count)

    @staticmethod
    def named(values, kind):
        if not isinstance(values, list) or not values:
            raise ValueError(f"{kind} must be a nonempty array")
        result = {}
        for value in values:
            identity = value["id"]
            if not isinstance(identity, str) or not identity or identity in result:
                raise ValueError(f"{kind} IDs must be unique nonempty strings")
            result[identity] = value
        return result

    @staticmethod
    def primary(plan, source, *, available):
        serving = ServingInput(source)
        structure = SourceDomain.requested(serving)
        opportunities = RepresentativeSources({}, plan["expert_count"])
        required = opportunities.conditions(serving.instances, serving.workload)
        candidates = [job for job in plan["sources"] if job["id"] in available
                      and job["structure"] == structure and job["configurations"]]
        if not candidates:
            return None

        # This chooses input evidence, not support: unseen targets still face the actual template contracts and guards.
        return min(candidates, key=lambda job: (
            len(required - set(job["source"]["input_conditions"])),
            job["source"]["input_cost"]["prompt_tokens"] + job["source"]["input_cost"]["decode_tokens"], job["id"],
        ))["id"]

    def document(self, *, service_evidence=False, targets=None):
        pairs = list(product(self.configurations, self.workloads)) if targets is None else list(targets)
        domains = []
        for identity, name in pairs:
            serving = ServingInput({"model": self.source["model"], "instances": self.configurations[identity]["instances"],
                                    "workload": self.workloads[name]})
            requested = SourceDomain.requested(serving)
            domain = next((item for item in domains if item["structure"] == requested), None)
            if domain is None:
                domain = {"structure": requested, "inputs": []}
                domains.append(domain)
            domain["inputs"].append((identity, name, serving))

        jobs = []
        covered_workloads = {}
        for domain in domains:
            for selected, workload, groups, covered in self.representatives.select(domain["inputs"], service_evidence=service_evidence):
                identity = len(jobs)
                covered_workloads[identity] = {name for _, name in covered}
                instances = self.configurations[selected["configuration"]]["instances"]
                required = selected.pop("required_coverage")
                selected["input"] = f"sources/{identity}.json"
                jobs.append({
                    "id": identity, "structure": domain["structure"],
                    "configurations": sorted({configuration for configuration, _ in covered}),
                    "required_coverage": required, "source": selected,
                    "native_workers": sum(instance["configuration"]["tp"] * instance["configuration"].get("pp", 1)
                                          for instance in instances),
                })
                self.capture_inputs[identity] = {
                    "model": self.source["model"], "instances": instances, "workload": workload,
                    "submission_groups": groups, "observe_all_workers": False,
                }

        LocalSources(self.source["model"], self.workloads, self.configurations).extend(jobs, self.capture_inputs, covered_workloads)
        if service_evidence:
            for job in jobs:
                capture = self.capture_inputs[job["id"]]
                capture["warmup"] = capture["workload"]

                # Only a simultaneous physical capture can measure service across the target group.
                capture["observe_all_workers"] = self.joint_observation and job["native_workers"] > 1
        configurations = list(dict.fromkeys(configuration for configuration, _ in pairs))
        workloads = list(dict.fromkeys(workload for _, workload in pairs))
        scope = {"configurations": configurations, "workloads": workloads, "count": len(pairs)}

        # Only a sparse selection needs its individual pairs recorded to reconstruct the training demand.
        if len(pairs) != len(configurations) * len(workloads):
            scope["pairs"] = pairs
        return {
            "framework": self.source["framework"], "sources": jobs, "expert_count": self.expert_count,
            "targets": scope,
            "qualification": "initial_representative_plan", "native_jobs": len(jobs),
            "limits": ["input conditions are opportunities, not observed native branches or support gates",
                       "complementary jobs are fixed before capture; one complete job supplies target worker templates",
                       "selected workloads run sequentially after all-worker native idle; cache persists between groups",
                       "redundant whole requests are removed only while declared input opportunities remain; retained arrivals and expert routes are unchanged",
                "performance sources preserve prompt and final Decode context endpoints and retain work up to native resident batch capacity",
                       "selected mixed-batch workloads preserve their arriving requests for performance sampling; actual mixing still requires observation",
                       "representative lengths preserve prefix, page/chunk remainders and route transitions; padding is unchanged",
                       "model batch sizes are checked by serving role and phase; short pairs exercise joint Decode within each primary source",
                       "cached local output1/2 pairs shorten Prefill below one page when possible; request routes shift at the Prefill/Decode boundary",
                       "fixed arrival gaps keep original lengths; other shortening must preserve declared input conditions",
                       "allocator path opportunities use a cold-input page bound, not predicted batches or branch coverage",
                       "cached KV pressure and PD page reuse retain role-local demand; actual eviction/sorting still need evidence",
                       "cached-page demand counts shared token paths once; conditional expert routes may require more pages",
                       "shared-prefix probes run consecutively after completion; actual cache hits still require native evidence",
                       "receiver-empty/positive route opportunities come from supplied routes and model expert metadata",
                       "local coverage is checked on observations; failures stop acquisition without supplementation"],
        }
