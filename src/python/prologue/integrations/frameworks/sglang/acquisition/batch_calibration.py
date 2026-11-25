"""Measure the admission alternatives of a concurrent, unchunked native Prefill."""

from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.integrations.frameworks.sglang.acquisition.batch_readback import BatchReadback
from prologue.integrations.frameworks.sglang.acquisition.device_samples import CostSamples
from prologue.integrations.frameworks.sglang.acquisition.host_cost_samples import HostCostSamples
from prologue.integrations.frameworks.sglang.acquisition.worker_capture import WorkerCapture


class BatchCalibration:
    @staticmethod
    def cuts(job, sequence_capacity):
        batch = job["batch"]
        rows = batch["requests"]
        if batch["cut"] != "model" or batch["phase"] != "prefill" or len(rows) < 2:
            return {}
        sizes = [sum(n for _, n in row["prompt"]) for row in rows]
        length = max(sizes)
        longest = rows[sizes.index(length)]
        config = job["configuration"]
        page = config["page_size"]

        # The longest member may be admitted alone. Bound the measured endpoint by the original batch and native limits.
        room = sequence_capacity - longest["output_tokens"] if sequence_capacity is not None else length
        upper = min(
            (length + page - 1) // page * page,
            sum(sizes),
            config["max_prefill_tokens"],
            config["max_total_tokens"],
            room,
        )
        prompt = longest["prompt"].copy()
        token, count = prompt[-1]
        prompt[-1] = [token, count + max(0, upper - length)]
        solo = batch | {"requests": [longest | {"prompt": prompt}]}
        result = {"single-prefill": solo}

        # Two live cohorts can join on Decode. One step of progress distinguishes the cohorts without replaying long
        # requests.
        active = [row for row in rows if row["output_tokens"] > 1]
        if len(active) < 2:
            return result
        left, right = active[:-1], active[-1]
        lead = int(all(row["output_tokens"] > 2 for row in left))
        priming = [
            {"admit": [r["id"] for r in left], "decode_steps": lead},
            {"admit": [right["id"]], "decode_steps": 0},
        ]
        final = [r | {"output_tokens": 2 + lead} for r in left] + [right | {"output_tokens": 2}]
        joined = batch | {"phase": "decode", "requests": final, "priming": priming}
        return result | {"joined-forward": joined, "joined-result": joined | {"cut": "host"}}

    @staticmethod
    def products(recording, rank, model):
        batches = recording.inputs["job"]["calibration"]
        if batches.keys() != recording.manifest["calibration"].keys():
            raise ValueError("native calibration did not return each planned batch cut")
        products = {}
        for name, batch in batches.items():
            worker, timed = (WorkerCapture(recording, rank, part, cut=name) for part in ("structure", "timing"))
            with TemporaryDirectory(dir=recording.directory.parent) as scratch:
                facts, catalog = worker.catalog(model, Path(scratch))
            boundary = worker.read("boundary")
            BatchReadback.observed_asset(batch, boundary, facts)
            actual = [(m["id"], m["context_tokens"], m["new_tokens"]) for m in boundary["members"]]
            expected = []
            for row in batch["requests"]:
                length = sum(n for _, n in row["prompt"])
                expected.append((row["id"], 0, length) if batch["phase"] == "prefill"
                                else (row["id"], length + row["output_tokens"] - 2, 1))
            if actual != expected:
                raise ValueError("native calibration selected different members or token positions")
            host = HostCostSamples.local(worker, timed, catalog)
            device = CostSamples.worker_table(facts, catalog)
            required = [
                [definition, interval, features] for work, definition, interval, features in catalog if "copy" in work
            ]
            if required:

                # A serving copy can wait for its input. Its recorded operand belongs to the ready-buffer sampler.
                device["required_work"] = required
            products[f"samples/{name}-host.json"] = host | {
                "native_probe": {"cut": name, "rank": rank, "parts": ["structure", "timing"]}
            }
            products[f"samples/{name}-device.json"] = device | {
                "native_probe": {"cut": name, "rank": rank, "parts": ["structure"]}
            }
            if batch["cut"] == "host":
                merges = [scope for scope in facts["scopes"] if scope["role"] == "batch_transform"
                          and scope["work"]["context"]["operation"] == "merge"]
                if len(merges) != 1 or not all(effect["finished"] for effect in boundary["effects"]):
                    raise ValueError("joined cohorts must reach one native merge and release their request slots")
                merge, = merges
                left, right = (cohort["admit"] for cohort in batch["priming"])
                groups = [[row["id"] for row in merge["work"]["context"][key]] for key in ("left", "right", "result")]
                if groups != [left, right, left + right]:
                    raise ValueError("native merge changed the planned cohort membership or order")
                products[f"structure/{name}.json"] = worker.read("facts") | {"export_site": merge["site"]}
        return products
