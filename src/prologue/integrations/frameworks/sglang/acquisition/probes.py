"""Construct finite request-lifecycle probes without inventing expert choices."""

import json
from math import lcm

from prologue.contracts.workload import WorkloadContent
from prologue.contracts.workload.expert_routes import ExpertRoutes


class LifecycleProbes:
    def __init__(self, instances, topologies):
        self.page_size = instances[0].configuration.values["page_size"]
        self.chunks = {instance.configuration.values["prefill_chunk_tokens"] for instance in instances}
        self.prefix_cache = any(instance.configuration.values["prefix_cache"] for instance in instances)
        self.topologies = topologies
        self.prompt_period = lcm(self.page_size, *(chunk for chunk in self.chunks if chunk))

    def outputs_to_new_page(self, prompt):

        # Prefill emits the first output. Decode consumes the first token of the next page one output later.
        return (-prompt) % self.page_size + 2

    def normalize(self, workload):
        content = WorkloadContent(workload["requests"], workload["prefixes"])
        requests = content.requests
        limits = {}
        for identity, request in requests.items():
            prompt = request["prompt_tokens"]
            prefix = content.lengths[request.get("prefix_id")]
            minimum = max(prefix + (prompt > prefix), self.page_size + 1 if prompt > self.page_size else 1)
            minimum = max([minimum, *(chunk + 1 for chunk in self.chunks if 0 < chunk < prompt)])
            output = request["output_tokens"]
            decode_minimum = min(output, 3)

            first_new_page = self.outputs_to_new_page(prompt)
            if output >= first_new_page:
                decode_minimum = max(decode_minimum, first_new_page)
            limits[identity] = {"prompt_tokens": minimum, "output_tokens": decode_minimum}

        # Keep interval boundaries visible, including gaps or overlaps that native validation must still reject.
        for topology in self.topologies:
            for entry in workload["ep_routes"][topology]["assignments"]:
                request = requests[entry["request"]]
                prompt = request["prompt_tokens"]
                total = prompt + request["output_tokens"] - 1
                for position in ExpertRoutes.interval(entry["tokens"], total, "tokens"):
                    if position == total:
                        continue
                    field = "output_tokens" if position >= prompt else "prompt_tokens"
                    minimum = position - prompt + 2 if position >= prompt else position + 1
                    limits[entry["request"]][field] = max(limits[entry["request"]][field], minimum)

        cohorts = {"prompt_tokens": {}, "output_tokens": {}}
        for identity, request in requests.items():
            for field, lengths in cohorts.items():
                length = request[field]
                lengths[length] = max(lengths.get(length, 0), limits[identity][field])

        # Preserve page/chunk remainders and equal-length cohorts; Decode also keeps separated output-length cohorts.
        prompts = self.shorten(cohorts["prompt_tokens"], self.prompt_period, 1)
        outputs = self.shorten(cohorts["output_tokens"], self.page_size, 2)
        normalized = {
            key: {**request, "prompt_tokens": prompts[request["prompt_tokens"]],
                  "output_tokens": outputs[request["output_tokens"]]}
            for key, request in requests.items()
        }
        return self.resize(workload, normalized)

    def resize(self, workload, requests):
        original = {request["id"]: request for request in workload["requests"]}
        result = {**workload, "requests": list(requests.values())}
        result.pop("ep_routes", None)

        # Keep original expert choices for retained tokens; shorter Prefill moves Decode's origin.
        routes = {}
        for topology in self.topologies:
            source = workload["ep_routes"][topology]
            assignments = []
            for entry in source["assignments"]:
                identity = entry["request"]
                if identity not in requests:
                    continue
                before, after = original[identity], requests[identity]
                tokens = []
                for position in entry["tokens"]:
                    if position < before["prompt_tokens"]:
                        tokens.append(min(position, after["prompt_tokens"]))
                    else:
                        tokens.append(after["prompt_tokens"] + min(position - before["prompt_tokens"], after["output_tokens"] - 1))
                if tokens[0] < tokens[1]:
                    assignments.append({**entry, "tokens": tokens})
            routes[topology] = {**source, "assignments": assignments}
        if routes:
            result["ep_routes"] = routes
        return result

    @staticmethod
    def shorten(cohorts, period, separation):
        result = {}
        previous, original = 0, 0
        for length, minimum in sorted(cohorts.items()):
            minimum = max(minimum, previous + min(length - original, separation))
            previous = minimum + (length - minimum) % period
            result[length] = previous
            original = length
        return result

    @staticmethod
    def shared_prefix_requests(content, page_size):
        owners = {}
        for request in content.requests.values():
            prefix = request.get("prefix_id")
            while prefix is not None:
                if content.lengths[prefix] >= page_size:
                    owners.setdefault(prefix, []).append(request["id"])
                prefix = content.prefixes[prefix].get("parent")
        shared = {identity for identities in owners.values() if len(identities) > 1 for identity in identities}

        # A producer can cache its entire prompt; a consumer must leave its last token for model execution.
        return [request for identity, request in content.requests.items() if identity in shared and request["prompt_tokens"] > page_size]

    def extend(self, workload, groups, *, model_phases, allocation_capacity):
        coverage = self.decode_probes(workload, groups, "decode" in model_phases, allocation_capacity)
        if self.prefix_cache:
            content = WorkloadContent(workload["requests"], workload["prefixes"])
            candidates = self.shared_prefix_requests(content, self.page_size)
            if candidates:
                source = min(candidates, key=lambda request: (
                    request["output_tokens"] < 3, request["prompt_tokens"] + request["output_tokens"], request["id"],
                ))

                # Consecutive singleton groups let native completion publish the prefix before its next use.
                if groups[-1] != [source["id"]]:
                    self.append(workload, groups, source, "cache_seed")
                self.append(workload, groups, source, "cache_reuse")
        minimum_output = 2 if "decode" in model_phases else 3
        candidates = [request for request in workload["requests"] if request["output_tokens"] >= minimum_output]
        if not candidates and model_phases:
            candidates = workload["requests"]
        identities = {request["id"] for request in candidates}
        if candidates and not any(len(group) == 1 and group[0] in identities for group in groups):
            source = min(candidates, key=lambda request: (request["prompt_tokens"] + request["output_tokens"], request["id"]))

            # The same singleton supplies degenerate model geometry and, when present, Decode continuation.
            self.append(workload, groups, source, "isolated")
        return coverage

    def decode_probes(self, workload, groups, joint, capacity):
        decodable = [request for request in workload["requests"] if request["output_tokens"] > 1]
        pool_pages = capacity // self.page_size
        pair_crosses = False
        if joint:
            source = min(decodable, key=lambda r: (r["output_tokens"] == 2, r["prompt_tokens"] + r["output_tokens"], r["id"]))

            # Unit prompts exercise the compact multi-request model input even when paging needs a longer peer.
            self.append(workload, groups, source, "decode_pair", prompts=(1, 1))
            pair_crosses = pool_pages >= 4 and source["output_tokens"] >= self.outputs_to_new_page(1)

        page_cases = []
        if self.page_size > 1 and pool_pages >= 2:
            for request in decodable:
                prompt = min(request["prompt_tokens"], self.page_size)
                crossing = self.outputs_to_new_page(prompt)
                if crossing <= request["output_tokens"]:
                    page_cases.append((crossing, request["output_tokens"] == crossing, prompt, request["id"], request))

        coverage = []
        if page_cases:
            crossing, _, prompt, _, source = min(page_cases)
            output = min(crossing + 1, source["output_tokens"])
            single = {group[0] for group in groups if len(group) == 1}
            if not any(request["id"] in single and request["prompt_tokens"] <= (pool_pages - 1) * self.page_size
                       and self.outputs_to_new_page(request["prompt_tokens"]) <= request["output_tokens"] for request in decodable):
                self.append(workload, groups, source, "decode_page", prompts=(prompt,), output_tokens=output)
            coverage.append("allocation/decode_pages/single")

            # The short peer stays in Decode until the other crosses a page. Both borrow the source's real token ranges.
            peer_pages = (output + self.page_size - 1) // self.page_size
            if joint and 2 + peer_pages <= pool_pages:
                if not pair_crosses:
                    self.append(workload, groups, source, "decode_page_pair", prompts=(1, prompt), output_tokens=output)
                coverage.extend(("allocation/decode_pages/multiple", "prefill/multiple/unit_queries"))
        return coverage

    def append(self, workload, groups, source, purpose, *, prompts=(None,), output_tokens=None):
        identities = []
        for index, prompt in enumerate(prompts):
            request = dict(source)
            if prompt is not None:
                request["prompt_tokens"] = prompt
                request.pop("prefix_id", None)
            if output_tokens is not None:
                request["output_tokens"] = output_tokens
            identity = json.dumps(["probe", purpose, index, source["id"]], ensure_ascii=False, separators=(",", ":"))
            workload["requests"].append({**request, "id": identity, "arrival_ns": 0})
            identities.append(identity)
            if self.topologies:
                seed = self.resize(workload, {source["id"]: request})
                for topology, route in seed["ep_routes"].items():
                    workload["ep_routes"][topology]["assignments"].extend({**entry, "request": identity} for entry in route["assignments"])
        groups.append(identities)
