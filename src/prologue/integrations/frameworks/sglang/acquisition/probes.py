"""Construct finite request-lifecycle probes without inventing expert choices."""

import json
from collections import Counter
from math import lcm

from prologue.contracts.workload import WorkloadContent
from prologue.contracts.workload.expert_routes import ExpertRoutes
from prologue.integrations.frameworks.sglang.batches import BatchTensors
from prologue.integrations.frameworks.sglang.configuration import NativeConfiguration


class LifecycleProbes:
    def __init__(self, instances, topologies):
        self.page_size = instances[0].configuration.values["page_size"]
        self.chunks = {instance.configuration.values["prefill_chunk_tokens"] for instance in instances}
        self.prefix_cache = any(instance.configuration.values["prefix_cache"] for instance in instances)
        self.topologies = topologies
        self.prefill = next(instance for instance in instances if instance.mode != "decode")
        self.prompt_period = lcm(self.page_size, self.prefill.configuration.model_token_alignment(),
                                 *(chunk for chunk in self.chunks if chunk))
        self.ordinary_configuration = self.prefill.configuration.values if self.prefill.mode == "null" else None

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

    @staticmethod
    def prefill_shape_witness(parameters):
        shape = "prefill/single"
        if "block_table_strided" in parameters:
            shape += f"/block_table_strided/{parameters['block_table_strided']}"
        if parameters["model_tokens"] > parameters["input_tokens"]:
            shape += "/padded"
        return None if shape == "prefill/single" else shape

    @classmethod
    def prefill_shape(cls, configuration, request):
        tokens = request["prompt_tokens"]
        config = configuration.values
        page = config["page_size"]
        if (tokens + page - 1) // page > config["max_total_tokens"] // page:
            return None
        parameters = {"page_size": page, "model_members": 1, "input_tokens": tokens,
                      "model_tokens": BatchTensors.align_model_tokens(tokens, configuration.model_token_alignment())}
        parameters.update(BatchTensors.attention_layout(config["attention_backend"], parameters,
                                                       [{"context_tokens": 0, "new_tokens": tokens}]))
        return cls.prefill_shape_witness(parameters)

    def isolated_prefill(self, workload, groups, seeds, minimum_output):
        configuration = self.prefill.configuration
        alignment = configuration.model_token_alignment()
        candidates = {}
        for request in sorted(seeds, key=lambda row: (
            row["output_tokens"] < minimum_output, row["prompt_tokens"] + row["output_tokens"], row["id"],
        )):
            if shape := self.prefill_shape(configuration, request):
                candidates.setdefault(shape, (request, None))
            if alignment == 1:
                continue

            # Shorten within the available route assignments to witness the other padding branch.
            tokens = request["prompt_tokens"]
            aligned = tokens - tokens % alignment
            alternate = aligned - 1 if tokens == aligned else aligned
            if alternate > 0 and (shape := self.prefill_shape(configuration, {**request, "prompt_tokens": alternate})):
                candidates.setdefault(shape, (request, alternate))
        requests = {request["id"]: request for request in workload["requests"]}
        isolated = {self.prefill_shape(configuration, requests[group[0]]) for group in groups if len(group) == 1}

        # A native singleton must exercise its attention layout together with its padding branch.
        for shape in sorted(candidates.keys() - isolated):
            source, prompt = candidates[shape]
            self.append(workload, groups, source, f"isolated_prefill/{shape}", prompts=(prompt,))
        return sorted(candidates)

    def extend(self, workload, groups, *, model_phases, service_evidence):
        seeds = tuple(workload["requests"])
        coverage = {}
        if decode := self.decode_probes(workload, groups, "decode" in model_phases, service_evidence):
            coverage["null"] = decode
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
        if shapes := self.isolated_prefill(workload, groups, seeds, minimum_output):
            coverage.setdefault(self.prefill.mode, []).extend(shapes)
        candidates = [request for request in workload["requests"] if request["output_tokens"] >= minimum_output]
        if not candidates and model_phases:
            candidates = workload["requests"]
        identities = {request["id"] for request in candidates}
        if candidates and not any(len(group) == 1 and group[0] in identities for group in groups):
            source = min(candidates, key=lambda request: (request["prompt_tokens"] + request["output_tokens"], request["id"]))

            # The same singleton supplies degenerate model geometry and, when present, Decode continuation.
            self.append(workload, groups, source, "isolated")
        return coverage

    def decode_probes(self, workload, groups, joint, service_evidence):
        decodable = {request["id"]: request for request in workload["requests"] if request["output_tokens"] > 1}
        requested_width, unit_output = 1, 0
        if joint:
            requested_width = 2
            if service_evidence:
                for group in groups:
                    cohorts = Counter(decodable[identity]["arrival_ns"] for identity in group if identity in decodable)
                    requested_width = max(requested_width, max(cohorts.values(), default=0))
            source = min(decodable.values(), key=lambda r: (r["output_tokens"] == 2, r["prompt_tokens"] + r["output_tokens"], r["id"]))

            # Unit prompts exercise the compact multi-request model input even when paging needs a longer peer.
            unit_output = min(3, source["output_tokens"])
            self.append(workload, groups, source, "decode_pair", prompts=(1, 1), output_tokens=unit_output)

        # PD reserves transferred KV through a different lifecycle.
        if self.ordinary_configuration is None or self.page_size == 1:
            return []
        pool_pages = self.ordinary_configuration["max_total_tokens"] // self.page_size
        if pool_pages < 2:
            return []

        # The compact pair's first Decode reuses both prompt pages, independently of later page growth.
        unit_pair = joint and self.page_probe_capacity(pool_pages, unit_output) >= 2
        coverage = ["allocation/decode_tail/members/2", "prefill/multiple/unit_queries"] if unit_pair else []
        page_cases = []
        for identity, request in decodable.items():
            prompt = min(request["prompt_tokens"], self.page_size)
            crossing = self.outputs_to_new_page(prompt)
            if crossing <= request["output_tokens"]:
                page_cases.append((crossing, request["output_tokens"] == crossing, prompt, identity))
        if not page_cases:
            return coverage

        crossing, _, prompt, identity = min(page_cases)
        source = decodable[identity]
        widest = min(requested_width, self.page_probe_capacity(pool_pages, crossing))
        if widest < 1:
            return coverage

        # Keep a further Decode step when it fits the same width.
        output = crossing
        if source["output_tokens"] > crossing and self.page_probe_capacity(pool_pages, crossing + 1) >= widest:
            output += 1
        widths = {1, widest}
        if widest > 1:
            widths.add(2)

        existing = set()
        for group in groups:
            if len(group) == 1 and (request := decodable.get(group[0])) is not None:
                length = request["prompt_tokens"]
                if length <= (pool_pages - 1) * self.page_size and self.outputs_to_new_page(length) <= request["output_tokens"]:
                    existing.add(1)
                    break
        if unit_pair and unit_output >= self.outputs_to_new_page(1):
            existing.add(2)

        # All short peers remain in Decode when the final member needs its next page.
        for width in sorted(widths - existing):
            prompts = (1,) * (width - 1) + (prompt,)
            self.append(workload, groups, source, f"decode_pages_{width}", prompts=prompts, output_tokens=output)
        coverage.extend(f"allocation/decode_pages/members/{width}" for width in sorted(widths))
        return coverage

    def page_probe_capacity(self, pool_pages, output):
        configuration, page = self.ordinary_configuration, self.page_size
        slots = NativeConfiguration.batch_capacity(configuration)
        input_tokens = configuration["max_prefill_tokens"]
        chunk = configuration["prefill_chunk_tokens"]
        pool_tokens = pool_pages * page

        # PrefillAdder reserves another page for each cold one-page input.
        reservation = 2 * page + output
        if configuration["prefix_cache"]:
            input_slots = max(1, (input_tokens - 1) // page)
            pool_slots = (pool_tokens - 1) // reservation
        else:

            # Ignore-EOS admission deducts the last page reserve after admitting the request.
            input_slots = (input_tokens + page - 1) // page
            pool_slots = (pool_tokens + page - 1) // (reservation + 1)
        if chunk:
            input_slots = min(input_slots, max(1, chunk // page))
        peer_pages = (output + page - 1) // page
        return min(slots, input_slots, pool_slots, 1 + (pool_pages - 2) // peer_pages)

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
