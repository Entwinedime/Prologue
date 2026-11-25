"""Compare the fixed graph's output ports with independently recorded native sends."""

from prologue.host.workflow.latency import RequestLatency


def publication_marks(raw):
    scopes = [scope for scope in raw["scopes"] if scope.get("publication")]
    if not scopes:
        return

    intervals = {row["name"]: row for row in raw["host_observation"]["intervals"]}
    for scope in scopes:
        marker = intervals[f"prologue.scope/{scope['id']}"]
        for token in scope["publication"]:
            yield token["id"], token["generated_tokens"], marker["begin_raw_ns"], marker["end_raw_ns"]


class FixedPublications:
    def __init__(self, case, requests, origin_ns):
        self.requests = requests
        self.counts = {row["id"]: row["output_tokens"] for row in requests}
        self.positions = case.graph.positions
        self.ports, self.marks = {}, {}
        for rank, (contents, raw, _) in case.sources.items():
            if raw["environment"]["implementation"]["disaggregation_mode"] == "prefill":
                continue

            # The projected port identifies what the graph publishes. Its source scope's return is a separate witness.
            for scope in contents.graph["scopes"]:
                for item in scope["bindings"].get("publication", ()):
                    key = item["id"], item["generated_tokens"]
                    for port in scope["ports"]:
                        if port["name"] == "publication" and port["direction"] == "output":
                            address = rank, port["event"]
                            if self.ports.setdefault(key, address) != address:
                                raise ValueError(f"token {key} has multiple graph publication ports")

            for request, token, begin, end in publication_marks(raw):
                key = request, token
                mark = rank, begin - origin_ns, end - origin_ns
                if self.marks.setdefault(key, mark) != mark:
                    raise ValueError(f"token {key} has multiple native publications")

    def coverage_gaps(self):
        expected = {(name, token) for name, count in self.counts.items() for token in range(1, count + 1)}
        gaps = {}
        for source, records in (("graph", self.ports), ("native", self.marks)):
            missing, extra = expected - records.keys(), records.keys() - expected
            if missing or extra:
                gaps[source] = {"missing": len(missing), "first_missing": min(missing, default=None),
                                "unexpected": len(extra), "first_unexpected": min(extra, default=None)}
        return gaps

    def compare(self, times):
        checked = first_tokens = last_tokens = differences = max_width = 0
        first = None
        for key, (rank, begin, end) in self.marks.items():
            if key not in self.ports:
                continue
            address = self.ports[key]
            replayed = times[self.positions[address]]
            request, token = key
            checked += 1
            first_tokens += token == 1
            last_tokens += token == self.counts.get(request)
            max_width = max(max_width, end - begin)

            # Detailed capture timestamps the end of the send bracket; the actual socket send lies inside it.
            if rank != address[0] or replayed != end:
                differences += 1
                if first is None or end < first["native_bracket_ns"][1]:
                    first = {"request": request, "generated_tokens": token, "native_rank": rank, "graph_port": address,
                             "native_bracket_ns": [begin, end], "replayed_ns": replayed}
        return {"basis": "detailed_scheduler_send_bracket_end", "checked_tokens": checked,
                "checked_first_tokens": first_tokens, "checked_last_tokens": last_tokens,
                "max_send_bracket_ns": max_width, "different_tokens": differences, "first_difference": first}

    def metrics(self, times):
        outputs = [{"request": name, "generated_tokens": token, "finished": token == count,
                    "time_ns": times[self.positions[self.ports[name, token]]]}
                   for name, count in self.counts.items() for token in range(1, count + 1)]
        return RequestLatency(outputs, self.requests).report()
