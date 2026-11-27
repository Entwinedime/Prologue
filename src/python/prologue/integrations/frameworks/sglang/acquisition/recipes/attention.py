"""Qualify projected Decode attention with native KV readback."""

from collections import defaultdict
from math import prod

from prologue.common.cost import ServiceSamples, cost_features
from prologue.host.acquisition.rules import EvidenceGap
from prologue.integrations.bindings import RuleName, platform_backend
from prologue.integrations.frameworks.sglang.shared.configuration import NativeConfiguration

from ..contracts import CopyOperand, LocalCut, LocalRule, WorkerDomain
from ..endpoints import outside_edges


class LocalAttention(LocalRule):
    name = RuleName("decode_attention")
    worker = "--local-attention"
    priority = 10
    source_window = True
    runtime_trace = False
    conditions = ("worker", "attention_cut")

    @staticmethod
    def session_key(query, deployment):
        return query["domain"], query["definition"]

    @staticmethod
    def matches(query):
        if query.get("work") != "device":
            return False
        domain, features, definition = query["domain"], query["features"], query["definition"]
        return (features["phase"] == "decode" and "transport" not in features and WorkerDomain.dense_worker(domain)
                and definition.get("role") == "attention" and not definition["children"])

    def cut(self, queries, session, deployment, arguments):
        domain, definition = session
        contexts = set()
        for q in queries:
            lengths = LocalAttention.contexts(q["features"]["axes"])
            if not lengths:
                raise EvidenceGap("unbound_decode_context", features=q["features"])

            contexts.add(lengths)

        selected = None
        sampling_sources = self.context.sampling_sources
        for recording, source in sampling_sources() if sampling_sources else ():
            if source["domain"] != domain:
                continue
            settings = NativeConfiguration(source["runner"]["configuration"])

            if not all(LocalAttention.fits(settings, batch) for batch in contexts):
                continue
            for cut in source["cuts"]:
                if cut["definition"] != definition:
                    continue
                if selected and selected[2]["parts"] != cut["parts"]:
                    raise EvidenceGap("ambiguous_device_partition")

                if selected is None:
                    selected = recording, source["runner"], cut

        if selected is None:
            raise EvidenceGap("missing_qualified_attention_cut", contexts=sorted(contexts))
        recording, runner, cut = selected
        settings = NativeConfiguration(runner["configuration"])
        for query in queries:
            contexts.update(self.endpoints(query, settings))

        world, rank = settings.tp, WorkerDomain.rank(domain)
        if world != domain["parallel"]["logical_tp"]:
            raise EvidenceGap("attention_runner_topology_changed")

        card = platform_backend().deployment(deployment, world, False).card(rank)
        revision = domain["implementation"]["framework_revision"]
        job = {"runner": runner, "site": cut["site"], "rank": rank,
               "contexts": [list(batch) for batch in sorted(contexts)]}

        return LocalCut(job, [card], revision, runner["model"]["path"],
                        {"template_recording": recording, "cut": cut, "domain": domain})

    def committed(self, call, lengths):
        if "kv" not in call:
            return False
        prefill, decode = call["kv"]["prefill"], call["kv"]["decode"]
        return (all(row["matching_operands"] and row["preserved_prefix"] for row in (prefill, decode))
                and prefill["slots"] == decode["slots"] and prefill["committed"] == list(lengths)
                and decode["committed"] == [n + 1 for n in lengths]
                and prefill["locations"] == [row[:-1] for row in decode["locations"]]
                and prefill["written"] == [loc for row in prefill["locations"] for loc in row]
                and decode["written"] == [row[-1] for row in decode["locations"]])

    def qualify(self, queries, recording, timeout=None):
        domain, cut = recording.inputs["domain"], recording.inputs["cut"]
        record = self.recording_part(recording, "measurements")
        native = domain["implementation"]
        same_worker = (record["rank"] == WorkerDomain.rank(domain) and record["model"] == native["model"]
                       and recording.manifest["model_runner_starts"] == domain["parallel"]["logical_tp"]
                       and record["load_format"] == native["load_format"]
                       and CopyOperand.runtime(record["runtime"]) == CopyOperand.runtime(domain))
        calls = defaultdict(list)

        for call in record["calls"]:
            if call["site"] == cut["site"]:
                calls[tuple(call["context"])].append(call)

        definition = cut["definition"]
        planned = [tuple(batch) for batch in recording.inputs["job"]["contexts"]]
        if calls.keys() - set(planned):
            raise ValueError("native attention captured an unplanned context geometry")

        measured = {}
        for lengths in planned:
            group = calls[lengths]
            measured[lengths] = self.measured(domain, definition, cut["parts"], group, lengths) if (
                same_worker and group and all(self.committed(call, lengths) for call in group)
            ) else (None, "unobserved_worker_or_kv")

        name = "samples/0.json"
        samples = ServiceSamples(domain, [definition])
        published, answers = set(), []
        settings = NativeConfiguration(recording.inputs["job"]["runner"]["configuration"])
        for q in queries:
            lengths, interval = self.contexts(q["features"]["axes"]), q["interval"]
            costs, detail = measured[lengths]
            if (costs is None or interval not in costs or q["domain"] != domain
                    or q["definition"] != definition or not self.matches(q)):
                answers.append({"status": "untriggered", "detail": detail})
                continue

            # One measured interval may answer several original queries.
            if (lengths, interval) not in published:
                samples.add(0, interval, q["features"], costs[interval])
                published.add((lengths, interval))

            answer = {"status": "qualified", "asset": name, "observations": 1}
            for endpoint in self.endpoints(q, settings):
                endpoint_costs, _ = measured.get(endpoint, (None, None))
                if endpoint_costs is None or interval not in endpoint_costs:
                    continue

                if (endpoint, interval) not in published:
                    samples.add(0, interval, cost_features(self.axes(endpoint), phase="decode"),
                                endpoint_costs[interval])
                    published.add((endpoint, interval))

                answer.setdefault("endpoint_contexts", []).append(endpoint)
            answers.append(answer)

        detail = [{"contexts": lengths, "result": result} for lengths, (_, result) in measured.items()]
        return self.asset(name, samples.document({"site": cut["site"], "measurements": detail}), answers)

    @classmethod
    def endpoints(cls, query, settings):
        edges = outside_edges(query)
        if not edges:
            return ()

        lengths = cls.contexts(query["features"]["axes"])
        sizes = settings.values
        page, prefill = sizes["page_size"], sizes["max_prefill_tokens"]
        pages = sizes["max_total_tokens"] // page
        context = query["domain"]["implementation"]["model"]["max_position_embeddings"]
        sequence = query.get("invocation", {}).get("parameters", {}).get("sequence_capacity", context)
        count, longest = len(lengths), max(lengths)
        max_length = min(prefill // count, pages // count * page - 1, sequence - 1, context - 1)
        members = min(settings.batch_capacity(sizes), prefill // longest, pages // (longest // page + 1))

        choices = {lengths[:1], (longest,) * members, (1,) * count, (max_length,) * count}
        axes = query["features"]["axes"]
        return sorted(batch for batch in choices if cls.fits(settings, batch)
                      and any((cls.axes(batch)[axis] - axes[axis]) * edge > 0 for axis, edge in edges.items()))

    @staticmethod
    def axes(lengths):
        count, total = len(lengths), sum(lengths)
        return {"batch.members": count, "batch.query_tokens": count, "batch.max_query_tokens": 1,
                "batch.context_tokens": total, "batch.max_context_tokens": max(lengths),
                "batch.attention_pairs": total + count, "parameter.model_tokens": count}

    @staticmethod
    def contexts(axes):
        members = axes.get("batch.members", 0)
        maximum = axes.get("batch.max_context_tokens", 0)
        total = axes.get("batch.context_tokens", 0)

        if any(n < 1 or n != int(n) for n in (members, maximum, total)):
            return ()
        members, maximum, total = map(int, (members, maximum, total))

        # Only invert coordinates that determine the actual multiset of lengths for unit Decode.
        if total == maximum * members:
            lengths = (maximum,) * members
        elif members == 2 and 0 < total - maximum <= maximum:
            lengths = (total - maximum, maximum)
        else:
            return ()

        return lengths if axes == LocalAttention.axes(lengths) else ()

    @staticmethod
    def fits(settings, lengths):
        page = settings.values["page_size"]
        return (bool(lengths) and min(lengths) > 0 and len(lengths) <= settings.batch_capacity(settings.values)
                and sum(lengths) <= settings.values["max_prefill_tokens"]
                and sum((length // page + 1) * page for length in lengths) <= settings.values["max_total_tokens"])

    def same_heads(self, call, definition):
        heads = call["heads"]
        widths = {"argument.q": heads["query"] * heads["key_dim"], "argument.k": heads["key"] * heads["key_dim"],
                  "argument.v": heads["value"] * heads["value_dim"]}
        seen = set()

        for port in definition["ports"]:
            name = port["name"]
            if name not in widths or port["direction"] != "input":
                continue

            tensor, shape = port["tensor"], port["tensor"]["shape"]
            if (not shape or shape[0] != "model_tokens" or not all(type(x) is int for x in shape[1:])
                    or prod(shape[1:]) != widths[name] or call["operands"].get(name) !=
                    {"shape": [call["query_rows"], *shape[1:]], "dtype": tensor["dtype"]}):
                return False

            seen.add(name)

        return seen == widths.keys()

    def measured(self, domain, definition, parts, calls, lengths):
        if not parts:
            return None, "source_has_no_single_stream_partition"
        matching = [c for c in calls if c["query_rows"] == len(lengths)]

        if len(matching) != 2 or "profile" in matching[0] or "profile" not in matching[1]:
            return None, "native_batch_geometry_not_repeated"
        plain, measured = matching
        native = domain["implementation"]

        for call in matching:
            if (call["kv_layout"] != domain["kv_layout"] or call["backend"] != native["attention_backend"]
                    or call["implementation"] != definition["implementation"]
                    or call["page_size"] != native["page_size"] or not call["full_context"]
                    or not self.same_heads(call, definition)):
                return None, "attention_domain_or_read_extent_changed"

        if "reason" in measured:
            return None, measured["reason"]
        rows = measured["tasks"]
        if [tuple(task[:2]) for task in rows] != [tuple(task) for _, tasks in parts for task in tasks]:
            return None, "device_task_sequence_changed"

        durations = iter(row[2] for row in rows)
        costs = {interval: sum(next(durations) for _ in tasks) for interval, tasks in parts}
        return costs, {"events_only_ns": plain["envelope_ns"], "with_profiler_ns": measured["envelope_ns"]}


rules = (LocalAttention,)
