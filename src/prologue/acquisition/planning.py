"""Group shared requirements while keeping each affected target."""

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


def cost_demands(targets):
    domains = {}
    for target in targets:
        domain = target["domain"]
        group = domains.setdefault(json.dumps(domain, sort_keys=True), {"domain": domain, "demands": {}})
        for coordinate in target["coordinates"]:
            key = json.dumps(coordinate, sort_keys=True)
            demand = group["demands"].setdefault(key, {"coordinate": coordinate, "targets": []})
            if target["target"] not in demand["targets"]:
                demand["targets"].append(target["target"])
    return [{"domain": group["domain"], "demands": list(group["demands"].values())} for group in domains.values()]
