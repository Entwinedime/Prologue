"""Retain the native module and ordered device tasks that identify a source attention cut."""

import json
from collections import defaultdict
from math import prod

from prologue.integrations.platforms.ascend.scoped_tasks import ScopedTasks
from prologue.performance.samples import WorkCatalog


class AttentionCuts:
    @staticmethod
    def runner(job):
        instances = job["instances"]
        if len(instances) == 1 and instances[0]["mode"] == "null":
            settings = instances[0]["configuration"]
            if settings["tp"] == settings.get("pp", 1) == 1 and "ep" not in settings:
                return {"model": job["model"], "configuration": settings}
        return None

    @classmethod
    def extract(cls, source, work, rank, identities, runner):
        catalog = WorkCatalog(work / f"{rank}-graph.json", identities)
        facts = json.loads((work / f"{rank}-facts.json").read_bytes())
        return cls.from_catalog(source, work / f"{rank}-sdk/trace.json", rank, catalog, facts, runner)

    @staticmethod
    def from_catalog(source, trace, rank, catalog, facts, runner):
        scopes = {}
        for piece, definition, _, features in catalog:
            scope = piece["scope"]
            if (piece["kind"] == "device" and features["phase"] == "decode"
                    and catalog.definitions[definition]["role"] == "attention" and not catalog.definitions[definition]["children"]):
                scopes[scope] = definition
        observed = ScopedTasks.source(source, trace, rank, catalog, scopes, facts) if scopes else {}

        # A shared definition can bind without a layer selector only when its source instances agree on ownership.
        groups = defaultdict(list)
        for scope, definition in scopes.items():
            groups[definition].append(scope)
        cuts = []
        for index, instances in groups.items():
            first = instances[0]
            parts = observed.get(first)
            if parts and all(observed.get(scope) == parts for scope in instances):
                cuts.append({"definition": catalog.definitions[index], "site": catalog.graph["scopes"][first]["site"],
                             "parts": parts, "source_calls": len(instances)})
        return {"domain": catalog.domain, "runner": runner, "cuts": cuts}

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
        known = {"batch.members": members, "batch.query_tokens": members, "batch.max_query_tokens": 1,
                 "batch.context_tokens": total, "batch.max_context_tokens": maximum,
                 "batch.attention_pairs": total + members, "parameter.model_tokens": members}
        return lengths if axes == known else ()

    @staticmethod
    def fits(settings, lengths):
        page = settings.values["page_size"]
        return (bool(lengths) and len(lengths) <= settings.batch_capacity(settings.values)
                and sum(lengths) <= settings.values["max_prefill_tokens"]
                and sum((length // page + 1) * page for length in lengths) <= settings.values["max_total_tokens"])

    @staticmethod
    def same_heads(call, definition):
        heads = call["heads"]
        widths = {"argument.q": heads["query"] * heads["key_dim"], "argument.k": heads["key"] * heads["key_dim"],
                  "argument.v": heads["value"] * heads["value_dim"]}
        seen = set()
        for port in definition["ports"]:
            name = port["name"]
            if name not in widths or port["direction"] != "input":
                continue
            shape = port["tensor"]["shape"]
            if (not shape or shape[0] != "model_tokens" or not all(type(x) is int for x in shape[1:])
                    or prod(shape[1:]) != widths[name]
                    or call["operands"].get(name) != {"shape": [call["query_rows"], *shape[1:]], "dtype": port["tensor"]["dtype"]}):
                return False
            seen.add(name)
        return seen == widths.keys()

    @staticmethod
    def measured(domain, definition, parts, calls, lengths):
        if not parts:
            return None, "source_has_no_single_stream_partition"
        matching = [c for c in calls if tuple(sorted(c["context"])) == lengths and c["query_rows"] == len(lengths)]
        if len(matching) != 2 or "profile" in matching[0] or "profile" not in matching[1]:
            return None, "native_batch_geometry_not_repeated"
        plain, measured = matching
        native = domain["implementation"]
        for call in matching:
            if (call["kv_layout"] != domain["kv_layout"] or call["backend"] != native["attention_backend"]
                    or call["implementation"] != definition["implementation"]
                    or call["page_size"] != native["page_size"]
                    or not call["full_context"]
                    or not AttentionCuts.same_heads(call, definition)):
                return None, "attention_domain_or_read_extent_changed"
        if "reason" in measured:
            return None, measured["reason"]
        rows = measured["tasks"]
        if [tuple(task[:2]) for task in rows] != [tuple(task) for _, tasks in parts for task in tasks]:
            return None, "device_task_sequence_changed"
        durations = iter(row[2] for row in rows)
        costs = {interval: sum(next(durations) for _ in tasks) for interval, tasks in parts}
        return costs, {"events_only_ns": plain["envelope_ns"], "with_profiler_ns": measured["envelope_ns"]}
