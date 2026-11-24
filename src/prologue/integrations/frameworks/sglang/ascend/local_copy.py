"""Bind a template's synchronous host/device copy to an isolated native measurement."""

import json
from collections import defaultdict

from prologue.acquisition.rules import EvidenceGap
from prologue.integrations.frameworks.sglang.ascend.launch import SGLangAscendLaunch
from prologue.integrations.frameworks.sglang.worker_templates import WorkerDomain
from prologue.integrations.platforms.ascend.copy_sampling import TensorCopySamples
from prologue.integrations.platforms.ascend.deployment import AscendDeployment
from prologue.performance.requirements import cost_identity
from prologue.performance.samples import ServiceSamples


class LocalCopies:
    name = "ascend.synchronous_tensor_copy"
    runtime_fields = ("device", "torch", "torch_npu")

    @staticmethod
    def operand(definition, interval, transport):
        work = definition["work"][interval]
        copy = work.get("copy")
        if work["kind"] != "transfer" or not copy or copy["caller"] not in TensorCopySamples.directions:
            return None
        service = TensorCopySamples.service(copy["caller"])
        resource = definition["resources"][work["resource"]]
        if (transport != service or any(copy[key] != service[key] for key in ("implementation", "direction"))
                or [resource["kind"], resource["role"]] != service["resource"]):
            return None
        return copy["caller"], copy["tensor"]["dtype"], copy["element_bytes"]

    @classmethod
    def matches(cls, query):
        return (query.get("work") == "transfer" and query["domain"]["framework"] == "sglang"
                and "torch_npu" in query["domain"]
                and cls.operand(query["definition"], query["interval"], query["features"].get("transport")) is not None)

    @classmethod
    def point(cls, query):
        features = query["features"]
        caller, dtype, width = cls.operand(query["definition"], query["interval"], features["transport"])
        size = features["axes"].get("transfer.bytes")
        if (features["phase"] or set(features["axes"]) != {"transfer.bytes"} or type(width) is not int or width <= 0
                or type(size) is not int or size <= 0 or size % width):
            raise EvidenceGap("unbound_copy_operand", caller=caller, dtype=dtype, axes=features["axes"])
        return caller, dtype, width, size

    @staticmethod
    def job(runtime, operands):
        return {
            "runtime": runtime,
            "operands": [{"caller": caller, "dtype": dtype, "element_bytes": width,
                          "points": [{"bytes": size, "ranks": sorted(ranks)} for size, ranks in sorted(sizes.items())]}
                         for (caller, dtype, width), sizes in sorted(operands.items())],
        }

    @classmethod
    def prepare(cls, queries, deployment, arguments, limit):
        session = None
        operands = defaultdict(lambda: defaultdict(set))
        for query in queries:
            domain = query["domain"]
            rank = WorkerDomain.rank(domain)
            target = AscendDeployment(deployment, max(map(int, domain["worker_roles"])) + 1, False)
            card = target.devices[0 if target.single_device else rank]
            runtime = {k: domain[k] for k in cls.runtime_fields}
            conditions = card, domain["implementation"]["framework_revision"], runtime
            if session is not None and session != conditions:
                raise EvidenceGap("different_sessions_required", conditions=["physical_device", "framework_revision", "runtime"])
            session = conditions
            caller, dtype, width, size = cls.point(query)
            operands[caller, dtype, width][size].add(rank)
        card, revision, runtime = session
        return SGLangAscendLaunch.local_attempt(arguments, limit, deployment, card, revision, cls.job(runtime, operands),
                                                "--local-copy", runtime_trace=True)

    @staticmethod
    def measurements(recording):
        manifest, job = recording.manifest, recording.inputs["job"]
        if (manifest["completion"] != "successful_synchronous_acl_return"
                or any(manifest["runtime"][key] != value for key, value in job["runtime"].items())
                or manifest["services"] != {op["caller"]: TensorCopySamples.service(op["caller"]) for op in job["operands"]}):
            raise ValueError("native copy observation has different runtime, completion or service ownership")
        measured = json.loads((recording.source / manifest["measurements"]).read_bytes())
        rows = {(row["caller"], row["dtype"], row["bytes"]): row for row in measured}
        expected = {(op["caller"], op["dtype"], p["bytes"]) for op in job["operands"] for p in op["points"]}
        if len(rows) != len(measured) or rows.keys() - expected:
            raise ValueError("native copy recording has duplicate or unrequested operands")
        return rows

    @staticmethod
    def samples(domain, coordinates, rows):
        samples = ServiceSamples(domain)
        unmatched = unverified = 0
        for key in sorted(set(coordinates)):
            row = rows.get(key)
            if row is None or not row["verified_operand"]:
                unverified += 1
                continue
            unmatched += row["unmatched_calls"]
            for duration in row["observations_ns"]:
                samples.add_external(TensorCopySamples.service(row["caller"]), "transfer",
                                     {"phase": "", "axes": {"transfer.bytes": row["bytes"]}}, duration)
        return samples.document({"unmatched_calls": unmatched, "unverified_operands": unverified})

    @classmethod
    def qualify(cls, queries, recording):
        rows = cls.measurements(recording)
        inputs = recording.inputs
        ranks = {(op["caller"], op["dtype"], op["element_bytes"], p["bytes"]): p["ranks"]
                 for op in inputs["job"]["operands"] for p in op["points"]}
        groups, products, answers = {}, {}, []
        for query in queries:
            point = cls.point(query)
            caller, dtype, _, size = point
            domain = query["domain"]
            rank = WorkerDomain.rank(domain)
            if (domain["implementation"]["framework_revision"] != inputs["framework_revision"]
                    or any(domain[field] != recording.manifest["runtime"][field] for field in cls.runtime_fields)):
                answers.append({"status": "untriggered", "missing": "target_runtime"})
                continue

            key = caller, dtype, size
            row = rows.get(key)
            if rank not in ranks.get(point, ()) or row is None or not row["verified_operand"] or not row["observations_ns"]:
                answers.append({"status": "untriggered", "missing": "matching_synchronous_copy_and_destination"})
                continue

            group = groups.setdefault(cost_identity(domain), (f"samples/{len(groups)}.json", domain, set()))
            group[2].add(key)
            answers.append({"status": "qualified", "asset": group[0], "observations": len(row["observations_ns"])})
        for name, domain, coordinates in groups.values():
            products[name] = cls.samples(domain, coordinates, rows)
        return products, answers
