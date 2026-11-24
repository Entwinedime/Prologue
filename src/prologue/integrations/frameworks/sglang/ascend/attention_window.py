"""Turn an observed Decode work requirement into a bounded native request measurement."""

import json
from collections import defaultdict
from dataclasses import dataclass
from math import prod

from prologue.acquisition.recording import PublishedCapture
from prologue.artifacts import write_json
from prologue.integrations.frameworks.sglang.worker_templates import WorkerDomain
from prologue.integrations.platforms.ascend.scoped_tasks import ScopedTasks
from prologue.performance.requirements import cost_identity
from prologue.performance.samples import ServiceSamples, WorkCatalog


@dataclass(frozen=True, slots=True)
class AttentionProbe:
    rank: int
    source_rank: int
    scope: int
    site: str
    contexts: tuple[int, ...]


class DecodeSampling:
    def __init__(self, owner):
        self.owner = owner
        self.cuts, self.workers = {}, {}

    def observe(self, rank, catalog):
        domain = catalog.domain
        if domain["expert_layers"] or domain["implementation"]["disaggregation_mode"] != "null":
            return
        identity = cost_identity(domain)
        self.workers[identity] = rank, identity
        if self.owner.launch.local_shards:
            for role in domain["worker_roles"].values():
                actor = role["parallel"]["rank"]
                if actor == rank or role["representative"] != rank:
                    continue
                bound = WorkerDomain.bind_worker(domain, role)

                # Compiled cost domains keep the worker layout while omitting the global rank field.
                bound["parallel"] = {key: value for key, value in role["parallel"].items() if key != "rank"}
                self.workers.setdefault(cost_identity(bound), (actor, identity))
        definitions = {}
        for work, index, _, features in catalog:
            scope = catalog.graph["scopes"][work["scope"]]
            if work["kind"] != "device" or scope["role"] != "attention" or features["phase"] != "decode":
                continue
            if index not in definitions:
                definitions[index] = cost_identity(catalog.definitions[index])
            if not catalog.definitions[index]["children"]:
                self.cuts.setdefault((identity, definitions[index]), (rank, work["scope"], scope["site"]))

    @staticmethod
    def contexts(point):
        axes = dict(point.axes)
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

    def windows(self, demand):
        if not self.cuts:
            return
        configuration = self.owner.launch.serving.instances[0].configuration
        capacity = configuration.batch_capacity(configuration.values)
        page = configuration.values["page_size"]
        selections = defaultdict(dict)
        for point in demand.points:
            worker = self.workers.get(point.domain)
            if point.phase != "decode" or worker is None:
                continue
            rank, source_domain = worker
            cut = self.cuts.get((source_domain, point.definition))
            definition = demand.definitions[point.definition]
            if cut is None or definition["work"][point.interval]["kind"] != "device":
                continue
            lengths = self.contexts(point)
            allocated = sum(((n + page) // page) * page for n in lengths)
            if lengths and len(lengths) <= capacity and allocated <= configuration.values["max_total_tokens"]:

                # Each physical card has one scoped profiler owner. Batch all requested points for that worker.
                slot = rank if self.owner.launch.local_shards else None
                selections[slot][point] = AttentionProbe(rank, *cut, lengths)

        for slot, points in selections.items():
            name = "decode-attention" if slot is None else f"decode-attention-rank-{slot}"
            yield DecodeWindow(self.owner, name, points, demand)


class DecodeWindow:
    ready = True

    def __init__(self, owner, name, points, demand):
        self.owner, self.name, self.points = owner, name, points
        self.domains, self.definitions = demand.domains, demand.definitions

    def collect(self, arguments, requested, limit):
        launch, source = self.owner.launch, self.owner.source
        launch.require_recorded_placement(source.recording.inputs["deployment"])
        self.requested = requested
        sites = defaultdict(set)
        for point in requested:
            probe = self.points[point]
            sites[str(probe.rank)].add(probe.site)
        sites = {rank: sorted(names) for rank, names in sites.items()}
        groups, requests = [], []
        for lengths in sorted({self.points[point].contexts for point in requested}):
            for _ in range(2):
                group = []
                for length in lengths:
                    identity = f"decode-{len(requests)}"
                    requests.append({"id": identity, "arrival_ns": 0, "prompt_tokens": length, "output_tokens": 2})
                    group.append(identity)
                groups.append(group)
        workload = {"id": self.name, "requests": requests, "prefixes": []}
        job = {key: value for key, value in launch.job.items() if key != "warmup"}
        job.update(workload=workload, submission_groups=groups)
        if (arguments.output / "execution.json").is_file():
            job = json.loads((arguments.output / "input.json").read_text())
            sites = job.pop("attention_sites")
        attempt = type(launch)(job, source.deployment).capture(arguments, limit, attention_sites=sites)
        if attempt.inputs["framework_revision"] != source.recording.inputs["framework_revision"]:
            raise ValueError("Decode sampling and its source require the same upstream framework revision")
        attempt.run()

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
                    or prod(shape[1:]) != widths[name]):
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
                    or call["q_dtype"] != native["dtype"] or not call["full_context"]
                    or not DecodeWindow.same_heads(call, definition)):
                return None, "attention_domain_or_read_extent_changed"
        if "reason" in measured:
            return None, measured["reason"]
        rows = measured["tasks"]
        if [tuple(task[:2]) for task in rows] != [task for _, tasks in parts for task in tasks]:
            return None, "device_task_sequence_changed"
        durations = iter(row[2] for row in rows)
        costs = {interval: sum(next(durations) for _ in tasks) for interval, tasks in parts}
        return costs, {"events_only_ns": plain["envelope_ns"], "with_profiler_ns": measured["envelope_ns"]}

    def samples(self, output, acquisition):
        source = self.owner.source
        recording = PublishedCapture(output)
        by_rank = defaultdict(list)
        for point in sorted(self.requested, key=lambda p: (p.domain, p.definition, p.interval, p.axes)):
            by_rank[self.points[point].rank].append(point)
        for instance in recording.manifest["instances"]:
            for actor, relative in instance["attention_samples"].items():
                rank = int(actor)
                if rank not in by_rank:
                    continue
                with acquisition.processing():
                    work = source.projection / "work"
                    selections = by_rank[rank]
                    source_rank = self.points[selections[0]].source_rank
                    catalog = WorkCatalog(work / f"{source_rank}-graph.json", source.identities(source_rank))
                    domain = self.domains[selections[0].domain]
                    record = json.loads((recording.source / relative).read_text())
                    same_runtime = all(record["runtime"][k] == domain[k] for k in ("device", "torch", "torch_npu"))
                    if record["rank"] != rank or not same_runtime:
                        raise ValueError("Decode recording uses a different worker or device runtime from its source")
                    scopes = {self.points[point].scope for point in selections}
                    partitions = ScopedTasks.source(source.recording.source, work, source_rank, catalog, scopes)
                    results = ServiceSamples(domain)
                    definitions, matched, receipts = {}, {}, []
                    for point in selections:
                        probe = self.points[point]
                        key = probe.scope, probe.contexts
                        if key not in matched:
                            definition = self.definitions[point.definition]
                            calls = [c for c in record["calls"] if c["site"] == probe.site]
                            parts = partitions.get(probe.scope)
                            costs, detail = self.measured(domain, definition, parts, calls, probe.contexts)
                            matched[key] = costs
                            receipts.append({"site": probe.site, "context": probe.contexts, "measurement": detail})
                        costs = matched[key]
                        if costs is not None:
                            if point.definition not in definitions:
                                definitions[point.definition] = len(results.definitions)
                                results.definitions.append(self.definitions[point.definition])
                            duration = costs[point.interval]
                            results.add(definitions[point.definition], point.interval, point.features(), duration)
                    qualification = {"unmatched_cuts": sum(value is None for value in matched.values())}
                    report = results.document(qualification)
                    report["native_recordings"] = [source.recording.attempt_id, recording.attempt_id]
                    report["observation"] = {
                        "sampling": str(output), "training_capture": str(source.recording.directory), "rank": rank,
                        "source_rank": source_rank, "native_evidence": relative,
                    }
                    report["qualification"] = receipts
                    path = source.output / f"{self.name}-{rank}.json"
                    write_json(path, report)
                yield path, report
