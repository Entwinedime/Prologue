"""Plan and execute scoped cost collection from exact work queries."""

import sys

from prologue.acquisition.planning import cost_demands
from prologue.app.releases import cost_quotes, pinned, read_document
from prologue.integrations.frameworks.sglang.ascend.attention import DecodeAttention
from prologue.workflow.budget import Acquisition


def plan_costs(args):
    targets = read_document(args.query)["targets"]
    store, release, rules = pinned(args)
    costs = store.asset(release, "performance")
    jobs = []
    demands = []
    for group in cost_demands(targets):
        domain = group["domain"]
        coordinates = [row["coordinate"] for row in group["demands"]]
        quotes = cost_quotes(costs, rules, {"domain": domain, "coordinates": coordinates})
        runners = {}
        for demand, quote in zip(group["demands"], quotes, strict=True):
            if quote["status"] == "covered":
                continue
            coordinate = demand["coordinate"]
            length = DecodeAttention.length(domain, coordinate) if quote["status"] == "cost_gap" else None
            demands.append({**demand, "domain": domain, "status": quote["status"],
                            "sample_eligible": length is not None})
            if length is not None:
                name = coordinate["implementation"]
                job = runners.setdefault(name, {"domain": domain, "implementation": name,
                                                 "mode": DecodeAttention.mode, "sequence_lengths": set(),
                                                 "coordinates": []})
                job["sequence_lengths"].add(length)
                job["coordinates"].append(coordinate)
        for job in runners.values():
            rules.require_observation("device_attention", job["mode"])
            job["sequence_lengths"] = sorted(job["sequence_lengths"])
            jobs.append(job)
    return {"release": release["release"], "demands": demands, "jobs": jobs}


def sample_costs(args):
    jobs = read_document(args.plan)["jobs"]
    if not jobs:
        return {"status": "no_eligible_samples", "attempts": [], "charged_card_ns": 0}
    if not 0 <= args.job < len(jobs):
        raise ValueError("the requested sampling job does not exist")
    job = jobs[args.job]
    lengths = {DecodeAttention.length(job["domain"], row) for row in job["coordinates"]}
    if (job["mode"] != DecodeAttention.mode or None in lengths or sorted(lengths) != job["sequence_lengths"]
            or any(row["implementation"] != job["implementation"] for row in job["coordinates"])):
        raise ValueError("the sampling job does not match the native Decode scope")

    command = [sys.executable, "-m", "prologue.integrations.frameworks.sglang.ascend.attention"]
    return Acquisition(args.output, job).run(command, max_attempts=args.attempts,
                                            max_card_ns=args.card_seconds * 1_000_000_000, max_bytes=args.max_bytes)
