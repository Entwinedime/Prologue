"""Resolve request content, expert assignments and routing-dependent cache identities."""

from __future__ import annotations

from array import array
from collections import defaultdict, deque
from collections.abc import Sequence
from itertools import pairwise
from typing import Any


class WorkloadContent:
    """Normalize the public workload once before request translation and simulation. Request identity survives
    preparation of the native token inputs.
    """

    def __init__(self, requests: Sequence[dict], prefixes: Sequence[dict] = ()) -> None:
        self.requests = {request["id"]: request for request in requests}
        if len(self.requests) != len(requests) or any(
            not isinstance(identity, str) or not identity for identity in self.requests
        ):
            raise ValueError("content request IDs must be nonempty strings and unique")

        self.prefixes = {}
        children = defaultdict(list)
        for prefix in prefixes:
            if prefix.keys() - {"id", "parent", "tokens"}:
                raise ValueError("prefix fields are id, parent and tokens")
            identity = prefix["id"]

            if not isinstance(identity, str) or not identity or identity in self.prefixes:
                raise ValueError("prefix IDs must be nonempty strings and unique")
            if type(prefix["tokens"]) is not int or prefix["tokens"] <= 0:
                raise ValueError("prefix tokens must be a positive integer")

            parent = prefix.get("parent")
            if parent is not None and (not isinstance(parent, str) or not parent):
                raise ValueError("prefix parent must be a nonempty ID")
            self.prefixes[identity] = prefix
            children[parent].append(identity)

        for parent in children:
            if parent is not None and parent not in self.prefixes:
                raise ValueError(f"unknown prefix parent: {parent}")

        # Assign distinct tokens only where paths diverge, rather than hashing IDs into a vocabulary.
        self.prefix_tokens = {}
        self.private_tokens = {}
        branches = defaultdict(list)
        for identity, prefix in self.prefixes.items():
            branches[prefix.get("parent")].append((0, identity))

        for identity, request in self.requests.items():
            parent = request.get("prefix_id")
            if parent is not None and (not isinstance(parent, str) or parent not in self.prefixes):
                raise ValueError(f"unknown request prefix: {parent}")

            branches[parent].append((1, identity))

        for siblings in branches.values():
            for token, (kind, identity) in enumerate(sorted(siblings)):
                mapping = self.prefix_tokens if kind == 0 else self.private_tokens
                mapping[identity] = token

        self.vocabulary = max((len(siblings) for siblings in branches.values()), default=0)

        self.lengths = {None: 0}
        pending = deque(children[None])
        while pending:
            identity = pending.popleft()
            prefix = self.prefixes[identity]
            self.lengths[identity] = self.lengths[prefix.get("parent")] + prefix["tokens"]
            pending.extend(children[identity])

        if len(self.lengths) != len(self.prefixes) + 1:
            raise ValueError("prefix parent relationships contain a cycle")
        for request in requests:
            if self.lengths[request.get("prefix_id")] > request["prompt_tokens"]:
                raise ValueError(f"shared prefix exceeds request prompt: {request['id']}")

    def check_vocabulary(self, size: int) -> None:
        if self.vocabulary > size:
            raise ValueError(f"content needs {self.vocabulary} distinct sibling tokens, model vocabulary has {size}")

    def prompt_runs(self, identity: str) -> list[tuple[int, int]]:
        request = self.requests[identity]
        path = []
        parent = request.get("prefix_id")

        while parent is not None:
            prefix = self.prefixes[parent]
            path.append((self.prefix_tokens[parent], prefix["tokens"]))
            parent = prefix.get("parent")

        path.reverse()
        private = request["prompt_tokens"] - self.lengths[request.get("prefix_id")]
        if private:
            path.append((self.output_token(identity), private))

        return path

    def prompt(self, identity: str) -> list[int]:
        return [token for token, length in self.prompt_runs(identity) for _ in range(length)]

    def output_token(self, identity: str) -> int:

        # A private continuation also separates requests whose entire prompt is shared.
        return self.private_tokens[identity]


class ExpertRoutes:
    """Validate route coordinates against each native layer's declared expert geometry."""

    def __init__(self, source, requests, layers, model_layers: int, padding_limit: int) -> None:
        self.layers = {layer["layer"]: layer for layer in layers}
        self.model_layers = model_layers
        self.padding_limit = padding_limit
        if len(self.layers) != len(layers):
            raise ValueError("conditional routes need unique native expert layers")
        for layer in layers:
            if not 0 <= layer["layer"] < model_layers or not 0 < layer["top_k"] <= layer["experts"]:
                raise ValueError("invalid native expert-layer geometry")
        self.check(source, requests)

    def check(self, source: dict[str, Any], requests: list[dict[str, Any]]) -> None:
        if not isinstance(source, dict) or source.keys() - {"assignments", "padding"}:
            raise ValueError("EP route condition needs assignments and optional padding")
        tokens = {request["id"]: request["prompt_tokens"] + request["output_tokens"] - 1 for request in requests}
        segments = {name: {} for name in tokens}
        assignments = source["assignments"]

        if not isinstance(assignments, list):
            raise ValueError("EP route assignments must be an array")
        for assignment in assignments:
            name = assignment["request"]
            if name not in tokens:
                raise ValueError(f"EP route refers to an unknown request: {name}")

            self.insert(assignment, segments[name], tokens[name], True)

        for name, length in tokens.items():
            self.coverage(segments[name], length, name)
        self.assignments = segments

        padding = self.padding_input(source)
        segments = {}
        self.padding = max((assignment["tokens"][1] for assignment in padding), default=0)
        for assignment in padding:
            self.insert(assignment, segments, self.padding, False)

        # A shared route table may cover several TP layouts; unused positions do not create padded work.
        if self.padding:
            self.coverage(segments, self.padding, "padding")
        self.padding_routes = segments

    @staticmethod
    def padding_input(source):
        padding = source.get("padding", [])
        if not isinstance(padding, list):
            raise ValueError("padding routes must be an array")
        for assignment in padding:
            ExpertRoutes.interval(assignment["tokens"], (1 << 63) - 1, "padding tokens")
        return padding

    def insert(
        self, source: dict[str, Any], segments: dict[int, list[tuple[int, int, list[int]]]], tokens: int, request: bool
    ) -> None:
        fields = {"layers", "tokens", "experts"} | ({"request"} if request else set())
        if not isinstance(source, dict) or source.keys() - fields:
            raise ValueError("unsupported EP route assignment fields")

        layers = self.interval(source["layers"], self.model_layers, "layers")
        interval = self.interval(source["tokens"], tokens, "tokens")
        experts = source["experts"]
        if (
            not isinstance(experts, list)
            or not experts
            or any(type(expert) is not int or expert < 0 for expert in experts)
            or len(set(experts)) != len(experts)
        ):
            raise ValueError("route experts must be distinct model expert IDs matching top-k")

        for layer, geometry in self.layers.items():
            if not layers[0] <= layer < layers[1]:
                continue
            if len(experts) != geometry["top_k"] or any(expert >= geometry["experts"] for expert in experts):
                raise ValueError(f"route experts do not match native layer {layer}")

            segments.setdefault(layer, []).append((*interval, experts))

    def coverage(self, segments: dict[int, list[tuple[int, int, list[int]]]], tokens: int, context: str) -> None:

        # Each worker validates the actual expert layers it owns, including sparse model and PP layouts.
        for layer in self.layers:
            covered = 0
            ranges = segments.get(layer, [])
            ranges.sort(key=lambda item: item[0])
            for begin, end, _ in ranges:
                if begin != covered:
                    raise ValueError(f"route gap or overlap for {context}, layer {layer}")
                covered = end
            if covered != tokens:
                raise ValueError(f"incomplete route for {context}, layer {layer}")

    def check_batch(self, members, padded_tokens):
        tokens = sum(member["new_tokens"] for member in members)
        if not tokens <= padded_tokens <= tokens + min(self.padding, self.padding_limit):
            raise ValueError("native token padding exceeds its TP alignment or supplied route coverage")
        return tokens

    def batch_segments(self, members, layer: int, padded_tokens: int):
        tokens = self.check_batch(members, padded_tokens)
        ranges = [
            (
                self.assignments[member["id"]][layer],
                member["context_tokens"],
                member["context_tokens"] + member["new_tokens"],
            )
            for member in members
        ]

        if padded_tokens > tokens:
            ranges.append((self.padding_routes[layer], 0, padded_tokens - tokens))
        for segments, begin, end in ranges:
            if not 0 <= begin <= end <= segments[-1][1]:
                raise ValueError("native token interval exceeds the supplied route coverage")
            for first, last, experts in segments:
                count = max(0, min(end, last) - max(begin, first))
                if count:
                    yield count, experts

    def batch_rows(self, members, layer: int, padded_tokens: int) -> list[list[int]]:
        result = []
        for count, experts in self.batch_segments(members, layer, padded_tokens):
            result.extend([experts] * count)
        return result

    @staticmethod
    def interval(source: list[int], limit: int, field: str) -> tuple[int, int]:
        if (
            not isinstance(source, list)
            or len(source) != 2
            or any(type(value) is not int for value in source)
            or not 0 <= source[0] < source[1] <= limit
        ):
            raise ValueError(f"route {field} needs an integer [begin, end) range within its logical bounds")
        return source[0], source[1]


class RoutedContent:
    def __init__(self, content: WorkloadContent, assignments: list[dict], layers: set[int]) -> None:
        changes = {identity: defaultdict(dict, {0: {}}) for identity in content.requests}
        for assignment in assignments:
            begin, _ = assignment["tokens"]
            first, last = assignment["layers"]
            for layer in layers:
                if first <= layer < last:
                    changes[assignment["request"]][begin][layer] = tuple(assignment["experts"])

        signatures = {}
        self.segments = {}
        for identity in sorted(content.requests):
            request = content.requests[identity]
            length = request["prompt_tokens"] + request["output_tokens"] - 1
            events = changes[identity]
            events[length] = {}
            positions = sorted(events)
            active = {}
            segments = []

            for begin, end in pairwise(positions):
                active.update(events[begin])

                # Equal conditions have the same identity even when input intervals are split differently.
                signature = tuple(sorted(active.items()))
                symbol = signatures.setdefault(signature, len(signatures))
                segments.append((end, symbol * content.vocabulary))

            self.segments[identity] = segments

    def key(self, identity: str, tokens) -> array:
        segments = self.segments[identity]
        if len(tokens) > segments[-1][0]:
            raise ValueError(f"cache key exceeds supplied route coverage for {identity}")

        # Negative symbols separate conditional keys from ordinary native warmup tokens.
        # A radix path retains earlier differences, so later equal routes cannot rejoin it.
        result = array("q")
        begin = 0
        for end, offset in segments:
            result.extend(-offset - token - 1 for token in tokens[begin : min(end, len(tokens))])
            if end >= len(tokens):
                break

            begin = end

        return result
