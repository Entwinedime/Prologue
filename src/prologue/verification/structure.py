"""Prepare measured time conditions and compare the result to native structure evidence."""

from collections import defaultdict


class StructureReference:
    def __init__(self, evidence, publications, manifest):
        if manifest["initial_state"] != {"kv": "empty", "workspace": "warmed"}:
            raise ValueError("structure comparison requires a warmed target with empty KV state")
        self.domain = evidence["domain"]
        self.workload = manifest["workload"]
        self.actions = []
        self.measurements = []
        self.indices = {}
        calls = evidence["calls"]
        buckets = defaultdict(list)
        for work in evidence["work"]:
            buckets[work["owner"]].append(work)

        parents, children = self.identities(calls)
        calls_by_batch = defaultdict(list)
        work_by_batch = defaultdict(list)
        time_by_batch = defaultdict(list)
        entries = {}
        for call in calls:
            identity = call["id"]
            root, path = parents[identity]
            calls_by_batch[root].append(self.call_structure(call, path, children[identity]))
            local = sorted(buckets[identity], key=lambda work: (work["slot"], work["lane"]))
            for ordinal, work in enumerate(local):
                description = {name: work[name] for name in ("slot", "lane", "kind", "name")}
                entries[work["id"]] = {**description, "path": path, "ordinal": ordinal}
        for work in evidence["work"]:
            root, _ = parents[work["owner"]]
            profile_start = evidence["profile_scope_begins_ns"][root]
            work_by_batch[root].append(entries[work["id"]])
            time_by_batch[root].append({
                "duration_ns": work["end_ns"] - work["begin_ns"],
                "submitted_after_ns": work["submitted_ns"] - profile_start,
            })

        boundaries = {(request, item["count"]): item["raw_ns"]
                      for request, rows in publications.items() for item in rows}
        for call in calls:
            if call["parent"] is not None:
                continue
            members = [{**member, "slot": member["slot"] - 1} for member in call["members"]]
            outputs = {member["request"]: member["output_tokens"] + 1 for member in members}
            start = call["begin_ns"]
            phase = call["parameters"]["phase"]
            result = call["result_processing"]
            key = self.action_key(phase, members)
            if key in self.indices:
                raise ValueError(f"ambiguous reference timing for {key}")
            self.indices[key] = len(self.actions)

            self.measurements.append({
                "work": time_by_batch[call["id"]],
                "model_return_ns": call["end_ns"] - start,
                "result_finished_ns": result["end_ns"] - start,
                "publications_after_ns": {rid: boundaries[rid, count] - start for rid, count in outputs.items()},
            })
            self.actions.append({
                "phase": phase, "members": members, "resources": call["resources"],
                "after_result": result["resources"], "publications": outputs,
                "graph": {"calls": calls_by_batch[call["id"]], "work": work_by_batch[call["id"]]},
            })

    @staticmethod
    def action_key(phase, members):
        return phase, tuple((member["request"], member["kv_committed_tokens"], member["output_tokens"])
                            for member in members)

    def conditions(self, query):
        action = query["action"]
        key = self.action_key(action["phase"], action["members"])
        index = self.indices.get(key)
        if index is None:
            return {"failure": {"status": "action_mismatch", "reason": f"no observed action at {key}"}}

        expected = self.actions[index]["graph"]["work"]
        if not expected:
            return {"failure": {"status": "timing_gap", "reason": f"no measured work at {key}"}}
        if query["work"] != expected:
            return {"failure": {"status": "action_mismatch", "reason": f"work differs at {key}"}}

        # The reference owns the measured graph. A selected action receives only elapsed times.
        return self.measurements[index]

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
        written = [(None, port) for port in call["inputs"]]
        for slot, target_ports, new_ports in [
            *((slot, child["inputs"], child["outputs"]) for slot, child in enumerate(children)),
            (None, call["outputs"], []),
        ]:
            for target in target_ports:
                owner = next((row for row in reversed(written) if row[1]["tensor"] == target["tensor"]), None)
                if owner is not None:
                    source, port = owner
                    links.append({"from": {"child": source, "port": port["name"]},
                                  "to": {"child": slot, "port": target["name"]}})
            written.extend((slot, port) for port in new_ports)

        return {
            "path": path, "tensor_links": links,
            **{key: call[key] for key in ("operation", "implementation", "parameters")},
            **{key: [{k: v for k, v in port.items() if k != "tensor"} for port in call[key]]
               for key in ("inputs", "outputs")},
        }

    def job(self, source_libraries, model_config):
        libraries = []
        for source in source_libraries:
            if source["domain"] != self.domain:
                raise ValueError("template source and native reference have different domains")
            libraries.append({
                "definitions": source["definitions"],
                "instances": [instance for instance in source["instances"] if instance["parent"] is None],
            })
        return {
            "workload": self.workload, "model_config": model_config,
            "domain": self.domain, "libraries": libraries,
        }

    @staticmethod
    def consumed_conditions(actual, measured):
        samples = measured["work"]
        if len(samples) != len(actual["intervals"]):
            return False

        began = actual["began_ns"]
        lanes = {}
        for work, sample, (start, end) in zip(actual["graph"]["work"], samples, actual["intervals"], strict=True):
            ready = max(lanes.get(work["lane"], began), began + sample["submitted_after_ns"])
            if start != ready or end - start != sample["duration_ns"]:
                return False
            lanes[work["lane"]] = end

        ready = max(began + measured["model_return_ns"], max(lanes.values(), default=began))
        if ready != actual["model_ready_ns"]:
            return False
        for publication in actual["publications"]:
            ready = max(ready, began + measured["publications_after_ns"][publication["request"]])
            if ready != publication["at_ns"]:
                return False
        return actual["finished_ns"] == max(ready, began + measured["result_finished_ns"])

    def compare(self, result):
        comparisons = 0
        consumed = 0
        for index, (expected, actual) in enumerate(zip(self.actions, result["actions"])):
            for field in ("phase", "members", "resources", "after_result", "graph"):
                if actual[field] != expected[field]:
                    return {"status": "structure_mismatch", "action": index, "field": field,
                            "matched_actions": comparisons}
            if expected["publications"] != {row["request"]: row["count"] for row in actual["publications"]}:
                return {"status": "structure_mismatch", "action": index, "field": "publications",
                        "matched_actions": comparisons}

            measured = self.measurements[index]
            if not self.consumed_conditions(actual, measured):
                return {"status": "timing_mismatch", "action": index, "matched_actions": comparisons}
            comparisons += 1
            consumed += len(measured["work"])

        status = result["status"]
        if status == "complete":
            status = "matched" if comparisons == len(self.actions) == len(result["actions"]) else "structure_mismatch"
        return {"status": status, "reason": result["reason"], "matched_actions": comparisons,
                "expected_actions": len(self.actions), "measured_work_consumed": consumed,
                "finished_requests": result["finished_requests"]}
