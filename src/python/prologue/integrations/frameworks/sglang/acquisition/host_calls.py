"""Pair native host calls and PD notifications before measuring their service intervals."""

from collections import defaultdict

from prologue.common.cost import cost_identity


class NativeHostCalls:
    batch_roles = {"run_batch", "launch", "output_prepare", "output_skip"}
    local_batch_axes = {"batch.members", "batch.query_tokens"}

    def __init__(self, scopes):
        self.scopes = scopes
        self.calls = {0: None}
        self.children = defaultdict(list)
        self.batch_of = {0: None}
        self.batches = defaultdict(list)
        self.dimensions = {0: {}}

        for scope in scopes[1:]:
            identity, parent = scope["id"], scope["parent"]
            if parent not in self.calls or "service" in scope:
                continue

            self.calls[identity] = self.identity(scope)
            self.children[parent].append(identity)
            self.batch_of[identity] = self.batch_of[parent]
            dimensions = self.dimensions[parent]
            self.dimensions[identity] = dimensions | scope["dimensions"] if "dimensions" in scope else dimensions

            if scope["role"] == "run_batch":
                self.batch_of[identity] = identity
                self.batches[self.batch_key(scope)].append(identity)

    @staticmethod
    def batch_key(scope):
        work = scope["work"]
        members = tuple((row["id"], row["new_tokens"], row["phase"]) for row in work["members"])
        return work["phase"], members

    @classmethod
    def identity(cls, scope):
        role, work = scope["role"], scope["work"]
        detail = None
        if role in cls.batch_roles:
            detail = [work["phase"], [{k: v for k, v in m.items() if k != "slot"} for m in work["members"]]]
        elif role == "segment":
            detail = [work["continuation"], scope["bindings"]["condition"]]
        elif role in {"invocation", "query", "host_read", "batch_transform",
                      "cache_match", "cache_insert", "kv_reservation"}:
            detail = work.get("context")
        elif role == "metadata_collective":
            detail = [scope["parameters"], scope["bindings"]["condition"], work["context"].get("poll_requests")]
        elif role in {"transfer_send", "transfer_receive"}:
            detail = [work["context"]["message"], scope["bindings"]["dtype"]]
        elif role in {"transfer_send_wait", "transfer_receive_wait", "signal_wait"}:
            context = work["context"]
            producer = context["producer"]
            detail = [context["device_wait"], context.get("wait_stream"),
                      producer.get("record_stream"), producer["message"]]
        elif role == "stream_wait":
            detail = work["context"]["streams"]
        elif role == "signal_record":
            detail = work["context"]["record_stream"]
        elif role == "gather":
            detail = scope["bindings"].get("collective")

        return role, scope["implementation"], scope["site"], cost_identity(detail)

    def index(self, paths, typed):
        routes = {0: 0}
        indexed = defaultdict(list)
        for identity, call in self.calls.items():
            if identity == 0:
                continue
            scope = self.scopes[identity]
            parent = scope["parent"]

            # Exact matches retain the enclosing batch context as part of the call route.
            component = call if scope["role"] in self.batch_roles else call[:3]
            route = routes[identity] = paths.setdefault((routes[parent], component), len(paths) + 1)
            arguments = cost_identity(scope.get("parameters")) if call[:3] in typed else self.calls[parent]
            indexed[route, call, arguments, self.child_keys(identity)].append(identity)

        return indexed

    def child_keys(self, identity):
        return tuple(self.calls[child] for child in self.children[identity])

    def exact_pairs(self, observed):
        paths = {}
        typed = {(s["role"], s["implementation"], s["site"]) for s in observed.scopes if "parameters" in s}
        measured = observed.index(paths, typed)

        # Polling may change multiplicity. Each physical call can service at most one detailed call.
        pairs = {left: right for key, source in self.index(paths, typed).items()
                 for left, right in zip(source, measured.get(key, ()), strict=False)}
        return pairs, len(self.calls) - 1 - len(pairs)

    def local_pairs(self, observed, exact):
        used_source = {self.batch_of[scope] for scope in exact}
        used_target = {observed.batch_of[scope] for scope in exact.values()}
        pairs = {}
        for key, source in self.batches.items():
            targets = observed.batches.get(key, ())
            left = (scope for scope in source if scope not in used_source)
            right = (scope for scope in targets if scope not in used_target)
            for original, timed in zip(left, right, strict=False):
                self.pair_subtree(observed, original, timed, pairs)
        return pairs

    def pair_subtree(self, observed, original, timed, pairs):
        work = [(original, timed)]
        for source, target in work:
            source_call, target_call = self.calls[source], observed.calls[target]
            if (source_call[:3] != target_call[:3] or (source != original and source_call != target_call)
                    or self.child_keys(source) != observed.child_keys(target)):
                continue

            measured = observed.scopes[target]
            if "parameters" in measured and measured["parameters"] != self.scopes[source].get("parameters"):
                continue
            pairs[source] = target
            work.extend(zip(self.children[source], observed.children[target], strict=True))

    def supports_local_work(self, scope, features, observed, aligned_scopes):
        sources = scope["bindings"]["source_scopes"]
        if len(sources) != 1 or scope["kv_accesses"] or scope["slot_accesses"] or sources[0] not in aligned_scopes:
            return False

        source, = sources
        batch = self.scopes[self.batch_of[source]]
        axes = features["axes"]
        if features["phase"] != batch["work"]["phase"] or not self.local_batch_axes <= axes.keys():
            return False

        # Native dimensions supplement the query geometry without importing parameters from the detailed run.
        measured = observed.dimensions[aligned_scopes[source]]
        return all(k.startswith("parameter.") and measured.get(k.removeprefix("parameter.")) == axes[k]
                   for k in axes.keys() - self.local_batch_axes)


class PdNotificationSamples:
    def __init__(self, observations):
        addresses = {tuple(row["endpoint"]): rank for rank, row in observations.items()}
        sessions = {row["session"]: rank for rank, row in observations.items()}
        if len(addresses) != len(observations) or len(sessions) != len(observations):
            raise ValueError("PD notification measurements repeat a native endpoint")

        self.states, self.chunks = {}, defaultdict(dict)
        self.reservations, self.receipts = defaultdict(dict), defaultdict(dict)
        self.senders = defaultdict(set)
        self.destinations, self.published_to = defaultdict(set), defaultdict(set)

        for rank, observation in observations.items():
            self.record_states(rank, observation["states"])
            for row in observation["reservations"]:
                recipient = addresses.get(tuple(row["target"]))
                key = rank, row["request"]
                self.published_to[key].add(recipient)
                observed = self.reservations[recipient, row["request"]]

                if rank in observed:
                    raise ValueError("PD notification source repeats a reservation publication")
                observed[rank] = row["begin_raw_ns"]

            for row in observation["receipts"]:
                publisher = sessions.get(row["session"])
                if publisher is not None:
                    recorded = self.receipts[rank, row["request"]]
                    if publisher in recorded:
                        raise ValueError("PD notification source repeats a reservation receipt")

                    recorded[publisher] = row["received_raw_ns"]

            for row in observation["transfers"]:
                key = rank, row["request"]
                ends = self.chunks[key]
                ends[row["chunk"]] = max(ends.get(row["chunk"], 0), row["end_raw_ns"])

                if row["kind"] == "metadata":
                    recipient = sessions.get(row["destination"])
                    self.destinations[key].add(recipient)
                    self.senders[recipient, row["request"]].add(rank)

    def record_states(self, rank, rows):
        for row in rows:
            state = row["after"]
            if state not in (2, 4) or state == row["before"]:
                continue
            key = rank, row["request"], state
            begin, end = row["begin_raw_ns"], row["end_raw_ns"]
            lower, upper = self.states.get(key, (begin, end))
            lo, hi = self.states[key] = max(lower, begin), min(upper, end)
            if hi < lo:
                raise ValueError("PD state transition observations do not overlap")

    def prerequisites(self, key, state):
        _, request = key
        if key in self.chunks:
            reservations = self.reservations[key]
            destinations = self.destinations[key]
            if not destinations or None in destinations or destinations != reservations.keys():
                return ()

            times = self.receipts[key] if state == 2 else self.chunks[key]
            if state == 2 and times.keys() != reservations.keys():
                return ()
            return [(time, time) for time in times.values()]

        if state == 4 and (senders := self.senders[key]) and senders == self.published_to[key]:
            return [self.states.get((sender, request, 4)) for sender in senders]
        return ()

    @staticmethod
    def sample(operation, request, completion, prerequisites, **identity):
        bounds = None
        if prerequisites and None not in prerequisites:

            # Both operands are observed on the same RAW clock. Subtract only the service after the latest cause.
            low = max(0, completion[0] - max(upper for _, upper in prerequisites))
            high = completion[1] - max(lower for lower, _ in prerequisites)
            if high < low:
                raise ValueError(f"PD {operation} for request {request} precedes its observed cause")

            bounds = low, high

        return {"operation": operation, "request": request, "dependencies": len(prerequisites),
                "duration_bounds_ns": bounds, **identity}

    def intervals(self, rank):
        for (recipient, request), receipts in self.receipts.items():
            if recipient != rank:
                continue
            for publisher, received in receipts.items():
                sent = self.reservations[recipient, request].get(publisher)
                prerequisites = () if sent is None else ((sent, sent),)
                yield self.sample("kv_reservation_received", request, (received, received), prerequisites,
                                  publisher=publisher)

        for (actor, request, state), completion in self.states.items():
            if actor != rank or (state == 2 and (rank, request) not in self.chunks):
                continue
            operation = "kv_ready" if state == 2 else "kv_success"
            yield self.sample(operation, request, completion, self.prerequisites((rank, request), state))
