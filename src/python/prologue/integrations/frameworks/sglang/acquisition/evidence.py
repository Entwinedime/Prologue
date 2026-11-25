"""Read native observations and extract independently measured CPU, device and peer services."""

from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.cost import ServiceSamples, WorkCatalog, cost_features, cost_identity
from prologue.common.io import read_json, read_jsonl, write_json
from prologue.common.observation import CapturedWorkers, ThreadService
from prologue.host.acquisition.recording import PublishedCapture
from prologue.host.libraries.release import asset_path
from prologue.integrations.bindings import device_runtime, platform_backend, platform_domain
from prologue.integrations.frameworks.sglang.shared.calls import PipelineABI
from prologue.integrations.frameworks.sglang.shared.configuration import (
    CaptureInput,
    KvBufferAddresses,
    MhaKvLayout,
    ServingInput,
)
from prologue.integrations.frameworks.sglang.shared.resources import SourceLayout
from prologue.integrations.runtimes.pytorch.shared import CpuProtocolWork, CpuTransfer

from . import CopyOperand, SGLangPlacement, WorkerDomain, embedding_rule, scheduler_batch_rule

capture_input = CaptureInput
serving_input = ServingInput
cpu_protocol = CpuProtocolWork
cpu_transfer = CpuTransfer
stream_declarations = PipelineABI.stream_declarations


def observed_scopes(raw, layouts):
    return SourceLayout(raw["environment"], layouts).lower(raw["scopes"])


def kv_buffers(recording):
    native = recording["registration"]
    return KvBufferAddresses(MhaKvLayout.read(recording["layout"]), recording["page_size"],
                             native["pointers"], native["lengths"], native["page_bytes"])


class WorkerCapture:
    def __init__(self, recording, rank, part=None, *, cut=None):
        self.recording, self.rank = recording, rank
        self.job, revision = recording.inputs["job"], recording.inputs["framework_revision"]
        self.source, self.manifest = recording.source, recording.manifest
        if cut is not None:
            self.descend("calibration", cut)
            self.job = self.job.copy()
            self.job["batch"] = self.job.pop("calibration")[cut]
        if part is not None:
            self.descend("probes", part)
            if self.manifest["framework_revision"] != revision:
                raise ValueError("paired local probes must use the same native framework")
        if "workers" in self.manifest:
            self.descend("workers", str(rank))
            if (self.manifest["rank"], self.manifest["framework_revision"]) != (rank, revision):
                raise ValueError("local capture manifest belongs to a different worker")
        elif rank != 0:
            raise ValueError("a single-worker recording contains only rank zero")

    def descend(self, group, key):
        self.source = asset_path(self.source, self.manifest[group][key])
        self.manifest = read_json(self.source / "capture.json")

    @classmethod
    def observed(cls, recording):
        return [cls(recording, int(rank)) for rank in recording.manifest.get("workers", {"0": "."})]

    def read(self, part):
        return read_json(asset_path(self.source, self.manifest[part]))

    def boundary(self):
        result = self.read("boundary")
        if "output" in result:

            # Dummy weights may change token values. Compare slots, then keep the type and location for pairing.
            output = BatchReadback.output(result)
            result["output"] = {k: output[k] for k in ("dtype", "device")}
        return result

    @staticmethod
    def template_resources(facts):
        environment = facts["environment"]
        if environment["parallel"]["logical_pp"] != 1 or environment["implementation"]["disaggregation_mode"] != "null":
            return facts["resources"]

        # Synchronous cuts run in the caller's stream. Both device markers record that actual stream, including
        # around collective transport.
        streams = {facts["spans"][p["at"]["span"]]["resource"] for scope in facts["scopes"] for p in scope["ports"]
                   if p["name"] in {"device.enter", "device.complete"}}
        resources = facts["resources"]
        if len(streams) == 1:
            lane, = streams
            if resources[lane]["kind"] == "stream":
                resources = resources.copy()
                resources[lane] = resources[lane] | {"binding": "synchronous_caller"}
        return resources

    @staticmethod
    def observed_facts(source, rank, trace):
        raw = read_json(source / f"rank-{rank}.json")
        raw["scopes"] = observed_scopes(raw, device_runtime().tensor_layouts)
        return device_runtime().observation(source, rank, raw, trace, evidence=True).document()

    @staticmethod
    def project_work(facts, model, directory):
        native, graph, work = (directory / name for name in ("facts.json", "graph.json", "work.json"))
        write_json(native, facts)
        model.run_model("project", graph, native, options=("--evidence",))
        model.run_model("work-identities", work, graph)
        return WorkCatalog(graph, work)

    def catalog(self, model, directory):
        facts = self.observed_facts(self.source, self.rank, self.source / "sdk/trace.json")
        return facts, self.project_work(facts, model, directory)

    def measure(self, model, workspace):
        with TemporaryDirectory(dir=workspace) as scratch:
            return self.catalog(model, Path(scratch))


class BatchReadback:
    @classmethod
    def observed_asset(cls, batch, boundary, facts):
        if boundary["phase"] != batch["phase"]:
            raise ValueError("native batch acquisition ran in a different phase")
        if "output" in boundary:
            cls.check_model(boundary, facts)
            return "structure/model.json"
        if boundary["phase"] == "admission":
            if boundary["waiting"] != [row["id"] for row in batch["requests"]]:
                raise ValueError("native waiting queue differs from the supplied request order")
        elif boundary["phase"] not in {"selection", "scheduler"}:
            cls.check_host(batch, boundary, facts)
        return "structure/host.json"

    @staticmethod
    def geometry(members):
        return [{k: member[k] for k in ("id", "phase", "context_tokens", "new_tokens")} for member in members]

    @staticmethod
    def output(bounds):
        native = bounds["output"]
        if native["tokens"] != native["relayed"] or len(native["tokens"]) != len(bounds["members"]):
            raise ValueError("timed model output did not reach the native batch slots")
        return native

    @classmethod
    def check_model(cls, bounds, facts):
        native = cls.output(bounds)
        if native["device"] != platform_domain().device_type:
            raise ValueError("each sampled device token must reach its original native request slot")

        scopes = facts["scopes"]
        root, = (i for i, scope in enumerate(scopes) if scope["role"] == "run_batch" and scope["parent"] == 0)
        family = [s for i, s in enumerate(scopes) if i == root or s["parent"] == root]

        def tensor(role, direction, name):
            ports = [p for s in family if s["role"] == role for p in s["ports"]
                     if p["direction"] == direction and p["name"] == name]
            if len(ports) != 1:
                raise ValueError(f"one native model cut requires one {role} {direction} port for {name}")
            return ports[0]["tensor"]

        logits = tensor("model", "output", "next_token_logits")
        tokens = tensor("run_batch", "output", "next_token_ids")
        if (tensor("sample", "input", "next_token_logits")["value"] != logits["value"]
                or tensor("sample", "output", "next_token_ids")["value"] != tokens["value"]):
            raise ValueError("model, sampler and batch must hand off the observed tensor values")
        if tokens["shape"] != [len(bounds["members"])] or tokens["dtype"] != native["dtype"]:
            raise ValueError("the observed model output port must describe the returned token tensor")

    @staticmethod
    def check_host(batch, bounds, facts):
        expected = []
        for row, member in zip(batch["requests"], bounds["members"], strict=True):
            prompt = sum(n for _, n in row["prompt"])
            count = 1 if batch["phase"] == "prefill" else member["context_tokens"] - prompt + 2
            expected.append({"id": row["id"], "generated_tokens": count, "finished": count == row["output_tokens"]})
        publication = [
            effect | {"tokens": [row["next_token"]]} for effect, row in zip(expected, batch["requests"], strict=True)
        ]
        emitter = facts["environment"]["parallel"]["tp_rank"] == 0
        if bounds["effects"] != expected or bounds["publication"] != (publication if emitter else []):
            raise ValueError("native request state and receiving socket must agree on each produced token")
        scopes = facts["scopes"]
        bindings = [scope["bindings"] for scope in scopes]
        owners = {b.get("segment", {}).get("owner") for b in bindings}
        produced = [b["publication"] for b in bindings if "publication" in b]
        completed = [b["effects"] for b in bindings if "effects" in b and b["segment"]["owner"] == "result"]
        named = [{k: e[k] for k in ("id", "generated_tokens")} for e in expected]
        required = {"selection", "preparation", "result"} | ({"publication"} if emitter else set())
        if not required <= owners or produced != ([named] if emitter else []) or completed != [expected]:
            raise ValueError("observed host segments must account for the native request and publication effects")

        # The supplied token tensor is on the device. Its materialization must account for the actual batch payload.
        reads = [i for i, s in enumerate(scopes) if s["role"] == "host_read"
                 and s["bindings"]["condition"]["input"] == "next_token_ids"]
        copies = [s for s in facts["spans"] if s["scope"] in reads
                  and s.get("copy", {}).get("operand") == "next_token_ids"]
        if (len(reads) != 1 or len(copies) != 1 or copies[0]["copy"]["direction"] != "device_to_host"
                or copies[0]["operand_bytes"] != len(expected) * copies[0]["copy"]["element_bytes"]
                or not any(p["direction"] == "input" and p["name"] == "next_token_ids"
                           for p in scopes[reads[0]]["ports"])):
            raise ValueError("the host read must consume the supplied device output for each batch member")


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
        alignment = platform_backend().host_alignment(trace, original["scopes"], probe["scopes"], queues["commands"],
                                                      runtime, activity, probe["auxiliary_intervals"],
                                                      cold[rank].values())
        exact = alignment.cuts(pairs, original["state_queries"])
        local = alignment.cuts(local_pairs, original["state_queries"], strict_runtime=True) if local_pairs else None
        if local is not None:
            local_pairs = {scope: local_pairs[scope] for scope in local.scopes}

        evidence = graph["evidence"]["events"]
        samples = ServiceSamples(catalog.domain, catalog.definitions)
        measured, missing = Counter(), Counter()
        for piece, definition, interval, features in catalog:
            resource = graph["resources"][piece["resource"]]
            if not self.supports(piece, resource):
                continue
            scope = graph["scopes"][piece["scope"]]
            key = f"{resource['role']}/{scope['role']}"
            if piece["kind"] == "collective":
                source, = scope["bindings"]["source_scopes"]
                duration = collectives.get(pairs.get(source))
            else:
                begin, end = (evidence[piece[side]]["order"] for side in ("start", "finish"))
                thread = int(resource["identity"].split("/")[1])
                submission = resource["role"] == "submission"
                duration = exact.duration(thread, begin, end, submission)
                if duration is None and local_pairs and source_calls.supports_local_work(
                    scope, features, observed_calls, local_pairs
                ):
                    duration = local.duration(thread, begin, end, submission)
                    if duration is not None:
                        self.qualification["locally_matched_work"] += 1
            if duration is None:
                missing[key] += 1
                continue
            samples.add(definition, interval, features, duration)
            measured[key] += 1

        if "protocol" in probe:
            self.cpu_services(samples, probe["protocol"], alignment.service)
            protocols = {actor: record["protocol"] for actor, record in records.items()}
            for actor, identity, features, duration in CpuProtocolWork(protocols).transfers():
                if actor == rank and duration is not None:
                    samples.add_external(identity, "transfer", features, duration)
        report = samples.document({"host": sum(missing.values())})
        batches = [s["id"] for s in original["scopes"] if s["role"] == "run_batch"]
        counts = {"detailed": len(batches), "host": sum(s["role"] == "run_batch" for s in probe["scopes"]),
                  "matched": sum(i in pairs for i in batches)}
        report["qualification"] = {
            "native_batches": counts,
            "excluded_cold_scopes": sorted(cold[rank]),
            "queue_correction_ns": queues["raw_correction_ns"],
            "measured": dict(measured), "missing": dict(missing),
            "alignment": dict(self.qualification + exact.misses), "lost_threads": alignment.service.loss,
        }
        return report


class CostSamples:
    batch_rule = scheduler_batch_rule

    def __init__(self, production, projection, ranks):
        self.production = production.resolve()
        root = projection or production
        inputs = read_json(root / "production.json")["inputs"]
        self.recording = PublishedCapture.from_source(Path(inputs["capture_path"]))
        if (root != production and inputs["capture_path"]
                != read_json(production / "production.json")["inputs"]["capture_path"]):
            raise ValueError("regenerated cost evidence belongs to a different native observation")
        observed = read_json(root / "result.json")["observed_roles"]
        workers = CapturedWorkers(inputs["capture"]["instances"])
        if observed != workers.observed or set(ranks) - set(observed):
            raise ValueError("cost samples require an observed production rank")

        self.work = root / "work"
        self.joint = workers.joint_execution
        self.facts = {rank: self.read_facts(rank) for rank in ranks} if self.joint else {}
        self.collectives = self.notifications = None
        if self.joint:
            measured = [body["measurements"] for body in self.facts.values()]
            collective = any(row["collectives"] for row in measured)
            pd = any("pd_protocol" in row for row in measured)

            # A measurement between workers needs its peers from the same execution.
            if collective or pd:
                self.facts.update((rank, self.read_facts(rank)) for rank in observed if rank not in self.facts)
            self.sources = dict.fromkeys(self.facts, self.recording.source)
            if collective:
                self.collectives = platform_backend().collectives(self.facts, self.sources, ranks)
            if pd:
                peers = {rank: body["measurements"]["pd_protocol"] for rank, body in self.facts.items()}
                self.notifications = PdNotificationSamples(peers)

    def read_facts(self, rank):
        return read_json(self.work / f"{rank}-facts.json")

    @classmethod
    def export(cls, arguments):
        jobs = tuple(zip(arguments.rank, arguments.work_identities, arguments.output, strict=True))
        ranks = arguments.rank
        if len(set(ranks)) != len(jobs) or len({path.resolve() for _, _, path in jobs}) != len(jobs):
            raise ValueError("cost sample exports require distinct ranks and output paths")

        source = cls(arguments.production, arguments.projection, ranks)
        for rank, identities, path in jobs:
            catalog = WorkCatalog(source.work / f"{rank}-graph.json", identities)
            report = source.measure(rank, catalog, transport_only=arguments.transport_only)
            write_json(path, report, replace=False)

    @classmethod
    def structural_rank(cls, recording):

        # A paired CPU probe also contains one independently observed device cut at this shape.
        if "probes" in recording.manifest or recording.inputs["rule"] == cls.batch_rule:
            return recording.inputs["job"]["batch"].get("rank", 0)
        return None

    @classmethod
    def product(cls, recording, query):
        rank = WorkerDomain.rank(query["domain"])
        if query["work"] in {"device", "transfer"}:
            if cls.structural_rank(recording) == rank and not CopyOperand.matches(query):
                return "derived/device-services"
            return None
        rule = recording.inputs["rule"]
        observed = {int(actor) for actor in recording.manifest.get("workers", {"0": "."})}
        peers = query["features"].get("transport", {}).get("members", ())
        physical = recording.inputs["deployment"].get("acquisition_mode", "full") == "full"
        peer_run = rule == embedding_rule or "peer_ranks" in recording.inputs["job"].get("batch", {})
        if (query["work"] == "collective" and physical and peer_run and rank in observed
                and len(peers) > 1 and observed.issuperset(peers)):
            return f"derived/peer-service-{rank}"
        return None

    @classmethod
    def extract(cls, session, recording, queries, model, workspace):
        if queries[0]["work"] != "collective":
            return cls.batch(session, model, workspace, recording=recording)

        with TemporaryDirectory(dir=workspace) as scratch:
            report = cls.transport(recording, queries, model, Path(scratch))
        return report | {"acquisition_rule": recording.inputs["rule"], "native_recordings": [recording.attempt_id]}

    @classmethod
    def batch(cls, session, model, workspace, *, recording=None):
        recording = recording or PublishedCapture(session / "native")
        receipt = recording.qualified_receipt()

        rank = cls.structural_rank(recording)
        if rank is None:
            raise ValueError("local costs require a structural worker observation")
        worker = WorkerCapture(recording, rank, "structure" if "probes" in recording.manifest else None)
        services, asset = cls.worker_services(worker, model, workspace)
        accepted = {answer["asset"] for answer in receipt["answers"] if answer["status"] == "qualified"}
        if not accepted or ("probes" not in recording.manifest and accepted != {asset}):
            raise ValueError("the qualified session and its observed native batch boundary disagree")
        return services | {"acquisition_rule": recording.inputs["rule"], "native_recordings": [recording.attempt_id]}

    @classmethod
    def worker_services(cls, worker, model, workspace):

        # Timing and the graph have one source. A staged rank cannot measure service between peers.
        facts, catalog = worker.measure(model, workspace)
        asset = BatchReadback.observed_asset(worker.job["batch"], worker.read("boundary"), facts)
        return cls.worker_table(facts, catalog), asset

    @classmethod
    def worker_table(cls, facts, catalog):
        services = platform_backend().observed_services(catalog.graph, facts, {}, physical_peers=False)
        samples, absent = cls.local_samples(catalog, services)
        return samples.document(absent)

    @classmethod
    def transport(cls, recording, queries, model, directory):
        embedded = recording.inputs["rule"] == embedding_rule
        sources = (dict.fromkeys(map(int, recording.manifest["workers"]), recording.source) if embedded
                   else {worker.rank: worker.source for worker in WorkerCapture.observed(recording)})
        traces = {r: path / (f"sdk-{r}" if embedded else "sdk") / "trace.json" for r, path in sources.items()}
        rank = WorkerDomain.rank(queries[0]["domain"])
        if any(WorkerDomain.rank(q["domain"]) != rank or set(q["features"]["transport"]["members"]) - sources.keys()
               for q in queries):
            raise ValueError("transport service requires one worker and an observed physical peer group")

        if recording.inputs["deployment"].get("acquisition_mode", "full") != "full":
            raise ValueError("transport calibration requires the full physical deployment")
        peers = {actor: WorkerCapture.observed_facts(source, actor, traces[actor]) for actor, source in sources.items()}
        catalog = WorkerCapture.project_work(peers[rank], model, directory)
        backend = platform_backend()
        clocks = backend.collectives(peers, sources, [rank])
        services = backend.observed_services(catalog.graph, peers[rank], clocks.samples[rank], physical_peers=True)
        origin = {"native_recording": recording.attempt_id, "rank": rank}
        profile = platform_backend().transport_profile(sources, peers, traces[rank], origin,
                                         recording.inputs["image"], recording.inputs["deployment"])
        samples, absent = profile.samples(catalog, services)
        report = samples.document(absent) | {"transport_profile": profile.qualification}
        if conflicts := clocks.notification_conflicts[rank]:
            report["collective_notification_conflicts"] = conflicts
        return report

    def measure(self, rank, catalog, *, transport_only=False):
        observation = {"production": str(self.production), "rank": rank}
        facts = self.facts[rank] if self.joint else self.read_facts(rank)
        collective_services = self.collectives.samples[rank] if self.collectives is not None else {}
        services = platform_backend().observed_services(
            catalog.graph, facts, collective_services, physical_peers=self.joint
        )
        notifications = list(self.notifications.intervals(rank)) if self.notifications is not None else []
        if transport_only:
            if self.collectives is None and self.notifications is None:
                raise ValueError("a transport profile requires a joint physical observation of the peer group")
            profile = platform_backend().transport_profile(self.sources, self.facts,
                        self.work / f"{rank}-sdk/trace.json", observation,
                        self.recording.inputs["image"], self.recording.inputs["deployment"])
            if notifications and "state_transfer_protocol" not in profile.qualification:
                raise ValueError("physical notification samples require their measured transport protocol")
            samples, absent = profile.samples(catalog, services)
            provenance = {"transport_profile": profile.qualification}
        else:
            samples, absent = self.local_samples(catalog, services)
            provenance = {"observation": observation}

        for measured in notifications:
            span = measured["duration_bounds_ns"]
            if span is None:
                absent["notification"] += 1
                continue
            features = cost_features({"parameter.dependencies": measured["dependencies"]})
            samples.add_external({"operation": measured["operation"]}, "transfer", features, sum(span) // 2)

        report = {**samples.document(absent), **provenance, "native_recordings": [self.recording.attempt_id]}
        if notifications:
            report["notification_measurements"] = notifications
        if self.collectives is not None and (conflicts := self.collectives.notification_conflicts[rank]):
            report["collective_notification_conflicts"] = conflicts
        return report

    @staticmethod
    def local_samples(catalog, services):
        samples, absent = ServiceSamples(catalog.domain, catalog.definitions), Counter()
        for piece, definition, interval, features in catalog:
            duration, copy, collective = services.service(piece)
            if duration is None:
                absent[piece["kind"]] += 1
            elif collective is not None and "transport" not in features:
                absent["undeclared_peer_operand"] += 1
            elif "copy" in piece:
                if copy != [features["axes"]["transfer.bytes"], piece["copy"]["direction"]]:
                    raise ValueError("native copy size or direction differs from its bound tensor operand")

                # Operand-ready timing comes from the isolated copy probe.
                absent["requires_ready_operand_timing"] += 1
            else:
                samples.add(definition, interval, features, duration, copy, collective)
        return samples, absent
