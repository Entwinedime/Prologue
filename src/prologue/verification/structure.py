"""Prepare measured time conditions and compare the result to native structure evidence."""

from collections import deque

from prologue.acquisition.observation.batches import ObservedBatches, action_identity
from prologue.verification.reference import validate_reference


class StructureComparison:
    record_cache = False

    def __init__(self, domain, publications, manifest):
        self.arrivals = validate_reference(manifest, publications, "detailed")
        self.domain = domain
        self.workload = manifest["workload"]
        self.actions = []
        self.measurements = []
        self.indices = {}
        self.ingress = None
        self.boundaries = {(request, item["count"]): item["raw_ns"]
                           for request, rows in publications.items() for item in rows}

    def record(self, action, graph, call, work, begin_ns, end_ns, *, outputs=None):
        key = action_identity(action)
        if key in self.indices:
            raise ValueError(f"ambiguous reference timing for {key}")
        self.indices[key] = len(self.actions)

        if outputs is None:
            outputs = {member["request"]: member["output_tokens"] + 1 for member in action["members"]
                       if member.get("publishes", True)}
        self.measurements.append({
            "work": work, "prepare_ns": 0, "model_return_ns": call["end_ns"] - begin_ns,
            "result_finished": self.boundary(end_ns - begin_ns),
            "publications": {rid: self.boundary(self.boundaries[rid, count] - begin_ns)
                             for rid, count in outputs.items()},
        })
        observed = {**action, "publications": outputs, "graph": graph}
        if "resources" in call:
            observed["resources"] = call["resources"]
            observed["after_result"] = call["result_processing"]["resources"]
        if "next_read_ns" in call.get("result_processing", {}):
            self.measurements[-1]["next_read"] = self.boundary(call["result_processing"]["next_read_ns"] - begin_ns)
        self.actions.append(observed)

    @staticmethod
    def boundary(after_start_ns, after_model_ns=0):
        return {"after_start_ns": after_start_ns, "after_model_ns": after_model_ns}

    def conditions(self, query):
        key = action_identity(query["action"])
        index = self.indices.get(key)
        if index is None:
            return {"failure": {"status": "action_mismatch", "reason": f"no observed action at {key}"}}

        expected = self.actions[index]["graph"]["work"]
        if not expected:
            return {"failure": {"status": "timing_gap", "reason": f"no measured work at {key}"}}
        if query["work"] != expected:
            return {"failure": {"status": "action_mismatch", "reason": f"work differs at {key}"}}
        if query["action"].get("expert_routes") != self.actions[index].get("expert_routes"):
            return {"failure": {"status": "action_mismatch", "reason": f"expert routing differs at {key}"}}

        # The reference owns the measured graph. A selected action receives only elapsed times.
        return self.measurements[index]

    @staticmethod
    def consumed_conditions(actual, measured):
        samples = measured["work"]
        if len(samples) != len(actual["intervals"]):
            return False

        began = actual["began_ns"]
        lanes = {}
        ends = []
        graph = actual["graph"]
        for work, sample, (start, end) in zip(graph["work"], samples, actual["intervals"], strict=True):
            ready = max(lanes.get(work["lane"], began), began + sample["submitted_after_ns"])
            for predecessor in work.get("after", ()):
                ready = max(ready, ends[predecessor])
            if start != ready + sample["dispatch_ns"] or end - start != sample["duration_ns"]:
                return False
            lanes[work["lane"]] = end
            ends.append(end)

        work_ready = max(lanes.values(), default=began)
        ready = began + measured["model_return_ns"]
        if ready != actual["model_ready_ns"]:
            return False
        for publication in actual["publications"]:
            ready = max(ready, began + measured["publications"][publication["request"]]["after_start_ns"])
            if ready != publication["at_ns"]:
                return False
        finish = max(ready, work_ready, began + measured["result_finished"]["after_start_ns"])
        if "next_read" in measured:
            if actual.get("next_read_ns") != max(finish, began + measured["next_read"]["after_start_ns"]):
                return False
        return actual["finished_ns"] == finish

    def compare(self, result):
        comparisons = 0
        consumed = 0
        for index, (expected, actual) in enumerate(zip(self.actions, result["actions"], strict=False)):
            for field in ("phase", "members", "resources", "after_result", "graph", "cache"):
                if field in expected and actual[field] != expected[field]:
                    return {"status": "structure_mismatch", "action": index, "field": field,
                            "matched_actions": comparisons}
            if expected["publications"] != {row["request"]: row["count"] for row in actual["publications"]}:
                return {"status": "structure_mismatch", "action": index, "field": "publications",
                        "matched_actions": comparisons}

            measured = self.measurements[index]
            if "next_read" in measured and "next_read_ns" not in actual and result["status"] != "complete":
                break
            if not self.consumed_conditions(actual, measured):
                return {"status": "timing_mismatch", "action": index, "matched_actions": comparisons}
            comparisons += 1
            consumed += len(measured["work"])

        status = result["status"]
        if status == "complete":
            status = "matched" if comparisons == len(self.actions) == len(result["actions"]) else "structure_mismatch"
        report = {"status": status, "reason": result["reason"], "matched_actions": comparisons,
                  "expected_actions": len(self.actions), "measured_work_consumed": consumed,
                  "finished_requests": result["finished_requests"]}
        if "gap" in result:
            report["gap"] = result["gap"]
        return report


class ScheduledReference(StructureComparison):
    def __init__(self, domain, publications, manifest, evidence):
        super().__init__(domain, publications, manifest)
        self.origin = manifest["delivery"]["origin_raw_ns"]
        self.starts = {}
        self.resident = {}
        configuration = domain["configuration"]
        self.page_size = configuration["page_size"]
        self.record_cache = bool(configuration.get("prefill_chunk_tokens") or configuration.get("prefix_cache"))
        self.cache_effects = evidence.get("cache_effects")
        self.pending_effects = deque(self.cache_effects or ())
        self.cached_pages = set()
        if "ingress" in evidence:
            observed = evidence["ingress"]
            if observed.keys() != self.arrivals.keys():
                raise ValueError("native ingress must cover the full observed workload")
            self.arrivals = {request: read - self.origin for request, read in observed.items()}
            self.ingress = {"arrivals": [self.arrivals]}
        if configuration.get("prefix_cache") and (self.ingress is None or self.cache_effects is None):
            raise ValueError("a cached schedule requires both measured ingress and cache effects")

    def record(self, action, graph, call, work, begin_ns, end_ns, *, outputs=None):
        super().record(action, graph, call, work, begin_ns, end_ns, outputs=outputs)
        if self.ingress is not None:
            self.starts[action_identity(action)] = begin_ns - self.origin
        if self.record_cache:
            self.apply_cache_effects(len(self.actions) - 1)
            self.check_state(action, call)

    def apply_cache_effects(self, after_action):
        while self.pending_effects and self.pending_effects[0]["after_action"] == after_action:
            event = self.pending_effects.popleft()
            pages = set(event["pages"])
            if not pages or len(pages) != len(event["pages"]) or min(pages) < 0:
                raise ValueError("a native cache effect must name distinct physical pages")
            if event["operation"] == "store" and pages.isdisjoint(self.cached_pages):
                self.cached_pages.update(pages)
            elif event["operation"] == "evict" and pages <= self.cached_pages:
                self.cached_pages.difference_update(pages)
            else:
                raise ValueError(f"native cache effect disagrees with current residency: {event}")

    def check_state(self, action, call):
        phase = action["phase"]
        processing = call["result_processing"]
        cache = call["cache"]
        prefixes = cache.get("prefix_tokens", (0,) * len(action["members"]))
        written = []
        for member, row, protected in zip(action["members"], cache["rows"], prefixes, strict=True):
            request = member["request"]
            extent = member["kv_committed_tokens"]
            begin = member.get("read_tokens", extent - 1 if phase == "decode" else 0)
            previous = self.resident.get(request, [])
            if (len(row) != extent or not 0 <= protected <= begin < extent or protected % self.page_size
                    or row[protected:begin] != previous[protected:begin]):
                raise ValueError(f"observed KV prefix changed during {phase} for {request}")
            pages = {index // self.page_size - 1 for index in row[:protected:self.page_size]}
            if not pages <= self.cached_pages or len(pages) * self.page_size != protected:
                raise ValueError(f"observed cache binding is not resident for {request}")
            self.resident[request] = row
            written.extend(row[begin:])

            count = member["output_tokens"] + 1
            publication = self.boundaries.get((request, count))
            publishes = publication is not None and processing["begin_ns"] <= publication <= processing["end_ns"]
            if publishes != member.get("publishes", True):
                raise ValueError(f"observed publication disagrees with {phase} for {request}")
        if written != cache["written"]:
            raise ValueError("native writes do not cover the batch's declared token intervals")

        if self.cache_effects is not None:
            available = call["resources"]
            if len(self.cached_pages) * self.page_size != available["evictable_tokens"] + available["protected_tokens"]:
                raise ValueError("observed cache effects disagree with native token capacity")

        self.actions[-1]["cache"] = {
            **cache,
            "rows": [[index - self.page_size for index in row] for row in cache["rows"]],
            "written": [index - self.page_size for index in written],
        }

    def conditions(self, query):
        timing = super().conditions(query)
        if self.ingress is None or "failure" in timing:
            return timing
        action = query["action"]
        start = self.starts[action_identity(action)]
        return {**timing, "prepare_ns": max(0, start - action["clock_ns"])}

    def compare(self, result):
        report = super().compare(result)
        if self.cache_effects is None or report["status"] != "matched":
            return report

        self.apply_cache_effects(len(self.actions))
        if self.pending_effects:
            raise ValueError("native cache effects extend past the recorded actions")
        actual = result["cache_effects"]
        if self.cache_effects != actual:
            matched = 0
            for expected, observed in zip(self.cache_effects, actual, strict=False):
                if expected != observed:
                    break
                matched += 1
            return {**report, "status": "structure_mismatch", "field": "cache_effects", "matched_effects": matched}
        return {**report, "matched_cache_effects": len(actual),
                "evicted_pages": sum(len(event["pages"]) for event in actual if event["operation"] == "evict")}


class StructureReference(ScheduledReference):
    def __init__(self, evidence, publications, manifest):
        super().__init__(evidence["domain"], publications, manifest, evidence)
        self.observed = ObservedBatches(evidence).entries
        for batch in self.observed:
            call = batch.call
            self.record(batch.action(), batch.graph, call, batch.measurements(), call["begin_ns"],
                        call["result_processing"]["end_ns"])
