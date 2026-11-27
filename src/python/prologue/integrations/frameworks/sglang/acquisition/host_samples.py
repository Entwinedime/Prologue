"""Measure CPU service using paired structure and independent native timing."""

from collections import Counter, defaultdict

from prologue.common.cost import ServiceSamples, WorkCatalog
from prologue.common.io import read_json, read_jsonl, write_json
from prologue.common.observation import CapturedWorkers, ThreadService
from prologue.host.acquisition.recording import PublishedCapture
from prologue.integrations.bindings import platform_backend
from prologue.integrations.runtimes.pytorch.shared import CpuProtocolWork

from .contracts import SGLangPlacement
from .host_calls import NativeHostCalls


class HostCostSamples:
    def __init__(self):
        self.qualification = Counter()

    @classmethod
    def local(cls, original, timed, catalog):
        inputs = ("image", "physical_devices", "framework_revision")
        if (original.job != timed.job or original.rank != timed.rank or original.boundary() != timed.boundary()
                or any(original.recording.inputs[k] != timed.recording.inputs[k] for k in inputs)):
            raise ValueError("independent CPU probe changed the qualified native cut or its placement")

        probe = timed.read("host")
        rank = original.rank
        raw = read_json(original.source / f"rank-{rank}.json")
        if raw["environment"] != probe["environment"] or probe["rank"] != rank or probe["clock"] != "monotonic_raw_ns":
            raise ValueError("local CPU probe differs from the observed native worker or clock")

        trace = original.source / "sdk/trace.json"
        return cls().measure(catalog, {rank: probe}, rank, raw, original.source, trace, timed.source)

    @staticmethod
    def supports(work, resource):
        return work["kind"] == "host" or (work["kind"] == "collective" and resource["kind"] == "thread")

    @staticmethod
    def cold_scopes(probe):
        excluded = set()
        for load in probe["timing"]["specialization_loads"]:
            time = load["raw_ns"]
            active = {scope["id"]: scope["parent"] for scope in probe["scopes"][1:]
                      if scope["raw_interval"][0] <= time < scope["raw_interval"][1]}

            if not active:
                raise ValueError("post-warmup specialization lies outside the observed native calls")

            # The hook identifies the containing call, not the compiler's own end. Exclude that complete leaf window.
            excluded.update(active.keys() - active.values())

        return {s["id"]: (s["thread"], *s["raw_interval"]) for s in probe["scopes"] if s["id"] in excluded}

    def collectives(self, records, rank, cold):
        channels = defaultdict(lambda: defaultdict(list))
        by_thread = {actor: ThreadService.by_thread(windows.values()) for actor, windows in cold.items()}
        for actor, record in records.items():
            for scope in record["scopes"]:
                group = scope.get("group")
                if group is None or scope["bindings"]["condition"]["collective"]["root"] is not None:
                    continue

                channels[group["channel"], tuple(group["members"])][actor].append(scope)

        durations = {}
        for (_, members), actors in channels.items():
            if rank not in actors:
                continue
            if set(actors) != set(members) or len({len(rows) for rows in actors.values()}) != 1:
                self.qualification["unpaired_collectives"] += len(actors[rank])
                continue

            local = members.index(rank)
            for participants in zip(*(actors[actor] for actor in members), strict=True):
                if any(ThreadService.within(by_thread[actor].get(scope["thread"], ()), *scope["raw_interval"])
                       for actor, scope in zip(members, participants, strict=True)):
                    continue

                if len({NativeHostCalls.identity(scope) for scope in participants}) != 1:
                    self.qualification["collective_operands"] += 1
                    continue

                # All ranks share the host RAW clock. The last entry marks the end of peer wait.
                arrival = max(scope["raw_interval"][0] for scope in participants)
                if any(scope["raw_interval"][1] < arrival for scope in participants):
                    self.qualification["collective_clock"] += 1
                    continue

                scope = participants[local]
                durations[scope["id"]] = scope["raw_interval"][1] - arrival

        return durations

    @classmethod
    def external_work(cls, capture):
        instances = read_json(capture / "capture.json")["instances"]
        workers = CapturedWorkers(instances)
        demands = {}

        for instance in instances:
            participants = instance["native_participants"]
            for window in workers.windows[instance["id"]]:
                directory = window.protocol_directory(capture)
                records = {rank: read_json(directory / f"rank-{rank}.json") for rank in participants}
                for rank in window.observed:
                    demands[rank] = [(identity, "host", features)
                                     for identity, features, _, _ in CpuProtocolWork.local(records[rank])]

                # Both endpoints of a message must come from the very same native request execution.
                for rank, identity, features, _ in CpuProtocolWork(records).transfers():
                    if rank in window.observed:
                        demands[rank].append((identity, "transfer", features))

        return demands

    @staticmethod
    def cpu_services(samples, protocol, service):
        for identity, features, thread, bounds in CpuProtocolWork.local(protocol):
            if bounds is not None and (duration := service.duration(thread, *bounds, False)) is not None:

                # The waiting thread can sleep while the transport progresses. Count its actual CPU service only.
                samples.add_external(identity, "host", features, duration)

    @classmethod
    def export(cls, arguments):
        detailed, lightweight = arguments.detailed, arguments.host
        source_recording, host_recording = PublishedCapture(detailed), PublishedCapture(lightweight)
        source, host = source_recording.inputs, host_recording.inputs

        if source["image"] != host["image"]:
            raise ValueError("host cost capture changes the measured native image")
        inputs = ("model", "instances", "workload", "observe_all_workers")
        if any(source["job"][field] != host["job"][field] for field in inputs):
            raise ValueError("host cost capture changes the measured native workload or configuration")

        SGLangPlacement(source["job"], source["deployment"]).require_recorded_placement(host["deployment"])
        source_root, host_root = source_recording.source, host_recording.source
        catalog = WorkCatalog(arguments.graph, arguments.work_identities)
        environment = catalog.graph["environment"]
        rank = environment["parallel"]["rank"]
        instance, = (candidate for candidate in host_recording.manifest["instances"]
                     if rank in candidate["native_participants"])
        paths = {int(actor): host_root / relative for actor, relative in instance["host_samples"].items()}
        path = paths[rank]
        records = {actor: read_json(measurement) for actor, measurement in paths.items()}
        facts = read_json(arguments.facts)
        original = read_json(source_root / f"rank-{rank}.json")

        if (facts["measurements"]["window_ns"] != catalog.window
                or not facts["environment"] == original["environment"] == environment):
            raise ValueError("host measurement uses a graph from a different native capture")
        all_timed = set(records) == set(instance["native_participants"]) and all(
            (r["rank"], r["clock"]) == (actor, "monotonic_raw_ns") for actor, r in records.items())

        if not host["host_timing"] or "profile_windows" in instance or not all_timed:
            raise ValueError("host timing requires all CPU participants from one run on the common RAW clock")

        report = cls().measure(catalog, records, rank, original, source_root, arguments.trace, path.parent)
        report["observation"] = {"detailed": str(detailed.resolve()), "host": str(lightweight.resolve()), "rank": rank}
        report["native_recordings"] = [source_recording.attempt_id, host_recording.attempt_id]
        write_json(arguments.output, report, replace=False)

    def measure(self, catalog, records, rank, original, source_root, trace_path, directory):
        probe = records[rank]
        graph = catalog.graph
        cold = {actor: self.cold_scopes(record) for actor, record in records.items()}
        source_calls = NativeHostCalls(original["scopes"])
        observed_calls = NativeHostCalls(probe["scopes"])
        pairs, unmatched = source_calls.exact_pairs(observed_calls)
        self.qualification["unmatched_calls"] += unmatched
        local_pairs = source_calls.local_pairs(observed_calls, pairs)

        # A host-only window runs all CPU ranks together even when device shards share a card.
        collectives = self.collectives(records, rank, cold)
        timing = probe["timing"]
        queues = read_json(directory / timing["queues"])
        if queues["clock"] != "monotonic_raw_ns" or queues["unpaired"]:
            raise ValueError("native CPU timing requires paired queue evidence on the RAW clock")

        profile, = (source_root / "native" / f"rank-{rank}").glob(platform_backend().profile_glob)
        trace = platform_backend().trace(profile, source_root / "runtime", original, trace_path)
        runtime, activity = (list(read_jsonl(directory / timing[key])) for key in ("runtime", "activity"))
        observer_intervals = queues.get("observer_intervals", ())
        alignment = platform_backend().host_alignment(trace, original["scopes"], probe["scopes"], queues["commands"],
                                                      runtime, activity, probe["auxiliary_intervals"],
                                                      cold[rank].values(),
                                                      observer_intervals=observer_intervals)
        exact = alignment.cuts(pairs, original["state_queries"])
        local = alignment.cuts(local_pairs, original["state_queries"], strict_runtime=True) if local_pairs else None

        if local is not None:
            local_pairs = {scope: local_pairs[scope] for scope in local.scopes}

        evidence = graph["evidence"]["events"]
        samples = ServiceSamples(catalog.domain, catalog.definitions)
        measured, missing = Counter(), Counter()
        copies = Counter()
        for piece, definition, interval, features in catalog:
            resource = graph["resources"][piece["resource"]]
            thread_copy = piece["kind"] == "transfer" and resource["kind"] == "thread" and "copy" not in piece
            if not thread_copy and not self.supports(piece, resource):
                continue

            scope = graph["scopes"][piece["scope"]]
            key = f"{resource['role']}/{scope['role']}"
            direct = False
            if piece["kind"] == "collective":
                source, = scope["bindings"]["source_scopes"]
                duration = collectives.get(pairs.get(source))
            else:
                begin, end = (evidence[piece[side]]["order"] for side in ("start", "finish"))
                thread = int(resource["identity"].split("/")[1])

                if thread_copy:

                    # Copies with declared tensor operands use the ready-operand probe.
                    copy = exact.synchronous_copy(thread, begin, end)
                    if copy is None:
                        copies["missing"] += 1
                    else:
                        duration, operand = copy
                        samples.add(definition, interval, features, duration, copy=operand, direct_native_call=True)
                        copies["measured"] += 1
                    continue

                submission = resource["role"] == "submission"
                duration = exact.duration(thread, begin, end, submission)
                direct = (thread, begin, end) in exact.direct_reads

                if duration is None and local_pairs and source_calls.supports_local_work(
                    scope, features, observed_calls, local_pairs
                ):
                    duration = local.duration(thread, begin, end, submission)
                    if duration is not None:
                        self.qualification["locally_matched_work"] += 1
                        direct = (thread, begin, end) in local.direct_reads

            if duration is None:
                missing[key] += 1
                continue
            samples.add(definition, interval, features, duration, direct_native_call=direct)
            measured[key] += 1

        if "protocol" in probe:
            self.cpu_services(samples, probe["protocol"], alignment.service)
            protocols = {actor: record["protocol"] for actor, record in records.items()}
            for actor, identity, features, duration in CpuProtocolWork(protocols).transfers():
                if actor == rank and duration is not None:
                    samples.add_external(identity, "transfer", features, duration)

        report = samples.document({"host": sum(missing.values()), "transfer": copies["missing"]})
        batches = [s["id"] for s in original["scopes"] if s["role"] == "run_batch"]
        counts = {"detailed": len(batches), "host": sum(s["role"] == "run_batch" for s in probe["scopes"]),
                  "matched": sum(i in pairs for i in batches)}
        report["qualification"] = {
            "native_batches": counts,
            "excluded_cold_scopes": sorted(cold[rank]),
            "queue_correction_ns": queues["raw_correction_ns"],
            "measured": dict(measured), "missing": dict(missing),
            "alignment": dict(self.qualification + exact.misses), "lost_threads": alignment.service.loss,
            "paired_synchronous_transfers": copies["measured"],
        }
        if observer_intervals:
            report["qualification"]["measured_observer_intervals"] = len(observer_intervals)

        return report
