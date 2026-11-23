"""Build deterministic token inputs for the declared shared prefixes."""

from collections import defaultdict, deque


class PromptContent:
    def __init__(self, workload):
        self.requests = {request["id"]: request for request in workload["requests"]}
        entries = workload.get("prefixes", [])
        if not isinstance(entries, list):
            raise ValueError("prefixes must be an array")

        self.prefixes = {}
        children = defaultdict(list)
        branches = defaultdict(list)
        for entry in entries:
            if not isinstance(entry, dict) or entry.keys() - {"id", "parent", "tokens"}:
                raise ValueError("prefix fields are id, tokens and optional parent")
            identity, tokens, parent = entry.get("id"), entry.get("tokens"), entry.get("parent")
            if not isinstance(identity, str) or not identity or identity in self.prefixes:
                raise ValueError("prefix IDs must be nonempty and unique")
            if type(tokens) is not int or tokens < 1 or (parent is not None and not isinstance(parent, str)):
                raise ValueError(f"invalid tokens or parent for prefix {identity}")
            self.prefixes[identity] = (tokens, parent)
            children[parent].append(identity)
            branches[parent].append((0, identity))

        if any(parent is not None and parent not in self.prefixes for parent in children):
            raise ValueError("unknown prefix parent")
        lengths = {None: 0}
        pending = deque(children[None])
        while pending:
            identity = pending.popleft()
            tokens, parent = self.prefixes[identity]
            lengths[identity] = lengths[parent] + tokens
            pending.extend(children[identity])
        if len(lengths) != len(self.prefixes) + 1:
            raise ValueError("prefix parent relationships contain a cycle")

        for identity, request in self.requests.items():
            parent = request.get("prefix_id")
            if parent is not None and (not isinstance(parent, str) or parent not in self.prefixes):
                raise ValueError(f"unknown prefix for request {identity}")
            if lengths[parent] > request["prompt_tokens"]:
                raise ValueError(f"shared prefix exceeds prompt for request {identity}")
            branches[parent].append((1, identity))

        # Tokens distinguish siblings. The same values can label branches with different ancestors.
        self.vocabulary = max(map(len, branches.values()))
        self.prefix_tokens = {}
        self.outputs = {}
        for siblings in branches.values():
            for token, (private, identity) in enumerate(sorted(siblings)):
                mapping = self.outputs if private else self.prefix_tokens
                mapping[identity] = token

    @classmethod
    def declared(cls, workload, enabled=False):
        if enabled or "prefixes" in workload or any("prefix_id" in request for request in workload["requests"]):
            return cls(workload)
        return None

    def prompt(self, identity):
        request = self.requests[identity]
        path = []
        node = request.get("prefix_id")
        while node is not None:
            length, parent = self.prefixes[node]
            path.append((self.prefix_tokens[node], length))
            node = parent

        tokens = []
        for token, length in reversed(path):
            tokens.extend([token] * length)
        tokens.extend([self.outputs[identity]] * (request["prompt_tokens"] - len(tokens)))
        return tokens
