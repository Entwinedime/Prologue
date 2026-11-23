"""Group failed structural queries by the exact requirement they share."""

import json


def structure_demands(targets, decisions):
    results = []
    demands = {}
    for target, decision in zip(targets, decisions, strict=True):
        results.append({**target, **decision})
        status = decision["status"]
        if status == "covered":
            continue

        required = decision["required"]
        key = (status, json.dumps(required, sort_keys=True))
        demand = demands.setdefault(key, {"status": status, "required": required, "targets": []})
        if target["target"] not in demand["targets"]:
            demand["targets"].append(target["target"])
    return {"results": results, "demands": list(demands.values())}
