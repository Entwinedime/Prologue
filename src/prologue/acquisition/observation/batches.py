"""Project native call and device evidence into bound batch graphs."""

from collections import defaultdict
from dataclasses import dataclass


def action_identity(action):
    return action["phase"], tuple((row["request"], row["kv_committed_tokens"], row["output_tokens"])
                                  for row in action["members"])


def aligned_batches(ranks):
    for batches in zip(*ranks, strict=True):
        key = action_identity(batches[0].action())
        if any(action_identity(batch.action()) != key for batch in batches[1:]):
            raise ValueError(f"the physical ranks disagree on native action {key}")
        yield batches


def aligned_collectives(batches):
    for entries in zip(*(batch.collectives() for batch in batches), strict=True):
        first = entries[0][1]
        if first["parameters"]["participants"] != list(range(len(entries))) or any(
            any(call[name] != first[name] for name in ("parameters", "inputs", "outputs"))
            for _, call in entries[1:]
        ):
            raise ValueError("the observed collective operands disagree across ranks")
        yield entries


@dataclass(slots=True)
class ObservedBatch:
    call: dict
    graph: dict
    device: list
    profile_begin_ns: int
    profile_end_ns: int
    scopes: list

    def action(self):
        members = [{**member, "slot": member["slot"] - 1} for member in self.call["members"]]
        return {"phase": self.call["parameters"]["phase"], "members": members}

    def collectives(self):
        return [(scope, call) for scope, call in zip(self.scopes, self.graph["calls"], strict=True)
                if call["operation"] == "collective"]

    def measurements(self, device=None):
        owned = {row["id"] for row in self.device}
        result = []
        for row in self.device if device is None else device:
            api = row["api_begin_ns"]
            prior_end = row["previous_device_end_ns"]
            ready = max(api, prior_end) if prior_end is not None else api

            # A trace-exclusive predecessor enters only as measured external readiness.
            release = api if row["previous_work"] in owned else ready
            if ready > row["begin_ns"] or release < self.profile_begin_ns:
                raise ValueError(f"inconsistent measured readiness for device work {row['id']}")
            result.append({
                "submitted_after_ns": release - self.profile_begin_ns,
                "dispatch_ns": row["begin_ns"] - ready,
                "duration_ns": row["end_ns"] - row["begin_ns"],
                "submission_ns": row["api_end_ns"] - api,
            })
        return result


class ObservedBatches:
    def __init__(self, evidence):
        calls = evidence["calls"]
        parents, children = self.identities(calls)
        owned = defaultdict(list)
        for row in evidence["work"]:
            owned[row["owner"]].append(row)

        calls_by_root = defaultdict(list)
        scopes_by_root = defaultdict(list)
        work_by_root = defaultdict(list)
        device_by_root = defaultdict(list)
        coordinates = {}
        for call in calls:
            identity = call["id"]
            root, path = parents[identity]
            calls_by_root[root].append(self.call_structure(call, path, children[identity]))
            scopes_by_root[root].append(identity)
            local = sorted(owned[identity], key=lambda row: (row["slot"], row["lane"]))
            for ordinal, row in enumerate(local):
                description = {name: row[name] for name in ("slot", "lane", "kind", "name")}
                coordinates[row["id"]] = {**description, "path": path, "ordinal": ordinal}
        for row in evidence["work"]:
            root, _ = parents[row["owner"]]
            work_by_root[root].append(coordinates[row["id"]])
            device_by_root[root].append(row)

        self.entries = []
        for call in calls:
            if call["parent"] is None:
                identity = call["id"]
                graph = {"calls": calls_by_root[identity], "work": work_by_root[identity]}
                self.entries.append(ObservedBatch(
                    call, graph, device_by_root[identity], evidence["profile_scope_begins_ns"][identity],
                    evidence["profile_scope_ends_ns"][identity], scopes_by_root[identity],
                ))

    @staticmethod
    def identities(calls):
        paths, children = {}, defaultdict(list)
        for call in calls:
            parent = call["parent"]
            if parent is None:
                paths[call["id"]] = (call["id"], [])
            else:
                root, path = paths[parent]
                paths[call["id"]] = (root, [*path, len(children[parent])])
                children[parent].append(call)
        return paths, children

    @staticmethod
    def call_structure(call, path, children):
        links = []
        written = {port["tensor"]: {"child": None, "port": port["name"]} for port in call["inputs"]}
        for slot, child in enumerate(children):
            for target in child["inputs"]:
                source = written.get(target["tensor"])
                if source is not None:
                    links.append({"from": source, "to": {"child": slot, "port": target["name"]}})
            for port in child["outputs"]:
                written[port["tensor"]] = {"child": slot, "port": port["name"]}

        for target in call["outputs"]:
            source = written.get(target["tensor"])
            if source is not None:
                links.append({"from": source, "to": {"child": None, "port": target["name"]}})

        return {
            "path": path, "tensor_links": links,
            **{key: call[key] for key in ("operation", "implementation", "parameters")},
            **{key: [{k: v for k, v in port.items() if k != "tensor"} for port in call[key]]
               for key in ("inputs", "outputs")},
        }
