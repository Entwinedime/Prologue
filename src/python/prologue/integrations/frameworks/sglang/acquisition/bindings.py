"""Bind SGLang native sessions, sampling windows and worker roles to host operations."""

import json
from collections import defaultdict
from dataclasses import asdict, replace
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

from prologue.common.cost import cost_identity
from prologue.common.io import content_id, read_json, write_json
from prologue.host.acquisition.capture import NativeAttempt
from prologue.host.acquisition.options import HostSampleOptions
from prologue.host.acquisition.recording import PublishedCapture
from prologue.host.acquisition.rules import EvidenceGap, LocalAcquisition
from prologue.host.libraries.release import asset_path
from prologue.host.performance.requirements import CostRequirements, MeasuredWork, WorkPoint
from prologue.host.runtime.docker import DockerImage
from prologue.host.runtime.limits import AllowanceExhausted
from prologue.integrations.bindings import RuleName, device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.abi.registry import default_revision, select
from prologue.integrations.frameworks.sglang.shared.batches import BatchOperands
from prologue.integrations.frameworks.sglang.shared.configuration import (
    AttentionCatalog,
    CaptureInput,
    InstanceConfiguration,
    ServingInput,
)

from . import LocalCut, LocalRule, SGLangPlacement, WorkerDomain
from .evidence import CostSamples, HostCostSamples, WorkerCapture
from .rules import LocalAttention, LocalCopies, LocalEmbedding, LocalPrebuilt, LocalSchedulerBatch, LocalWorkerBinding


class SGLangTask:
    @property
    def control_command(self):
        domain = json.dumps(asdict(platform_domain()), separators=(",", ":"))
        return ("python3", "-m", "prologue.workers.control", "--kernel-python",
                "/opt/prologue/sglang/sgl-kernel/python", "--platform-domain", domain)

    def execution_domains(self):
        return (WorkerDomain.execution_domain(w["environment"]) for i in self.instances for w in i["participants"])

    def require_control(self, image):
        revision = WorkerDomain.image_revision(image)
        if any(worker["domain"]["implementation"]["framework_revision"] != revision for worker in self.workers):
            raise ValueError("control image and primary source use different upstream SGLang revisions")

    def bind_gap(self, action, domain):
        return BatchOperands.bind(action, self.serving, domain)

    def __init__(self, source, catalog) -> None:
        if source.keys() != {"model", "instances", "workload"}:
            raise ValueError("prediction requires model, instances and workload")
        self.serving = ServingInput(source)
        structures = {i.mode: i.configuration.structure() for i in self.serving.instances}
        self.domain = {"model": self.serving.model, "instances": structures}
        self.binding = catalog.select(self.domain)

    @property
    def workers(self):
        return self.binding.headers[:self.binding.worker_count]

    def missing_workers(self):
        instances = []
        for instance in self.serving.instances:
            implementation = ["framework_revision", "model", "dtype", "attention_backend", "load_format",
                              "disaggregation_mode", "page_size", "allocator", "cache", "scheduling"]
            roles = ["parallel", "groups", "modules", "kv_layout", "expert_layers", "representative"]
            if "ep" in instance.configuration.values:
                implementation.append("expert_layout")
            if instance.mode != "null":
                roles.append("kv_peers")
            readback = {"environment": ["framework", "torch", "device"], "implementation": implementation,
                        "worker_roles": roles, "initial_state": ["logical_kv", "workspace"]}
            instances.append({"id": instance.identity, "mode": instance.mode, "rank_begin": instance.ranks.start,
                              "rank_end": instance.ranks.stop, "settings": instance.configuration.settings(),
                              "readback": readback})

        # Capacities belong to native initialization. The binding domain names reusable structure.
        return {"status": "missing_worker_binding", "action": "initialize_workers", "domain": self.domain,
                "instances": instances}

    def participants(self, description: InstanceConfiguration):
        representatives = {}
        for index, header in enumerate(self.workers):
            domain = header["domain"]
            if domain["implementation"]["disaggregation_mode"] != description.mode:
                continue
            rank = header["resources"][0]["rank"]
            if rank in representatives:
                raise ValueError(f"more than one library claims representative worker {rank}")
            representatives[rank] = index, domain

        domain = next(iter(representatives.values()))[1]
        common = WorkerDomain.role_plan(domain)
        if any(WorkerDomain.role_plan(source) != common for _, source in representatives.values()):
            raise ValueError("candidate libraries describe different native role plans or implementation environments")

        roles = domain["worker_roles"].values()
        participants = []
        for role in sorted(roles, key=lambda r: (r["parallel"]["pp_rank"], r["parallel"]["tp_rank"])):
            rank, representative = role["parallel"]["rank"], role["representative"]
            if representative not in representatives:
                raise ValueError(f"worker {rank} requires missing representative library {representative}")
            index, source = representatives[representative]
            participants.append({"id": f"{description.identity}/{rank}", "rank": rank, "template": index,
                                 "environment": WorkerDomain.bind_worker(source, role)})
        return participants

    @cached_property
    def instances(self):
        if WorkerDomain.observed(self.workers) != self.domain:
            raise ValueError("published workers do not match the target serving domain")
        instances = []
        for worker in self.serving.instances:
            config = worker.configuration
            instances.append({"id": worker.identity, "mode": worker.mode, "settings": config.settings(),
                              "control": config.control(), "expert_routes": config.routes(self.serving.workload),
                              "participants": self.participants(worker)})
        return instances

    def document(self):
        return {"instances": self.instances, "workload": self.serving.workload,
                "template_sources": self.binding.headers}


class LocalSchedulerWork(LocalRule):
    compiler_gap = "missing_work_compiler"

    @staticmethod
    def session_key(query, deployment):
        return query["domain"], query.get("invocation")

    @staticmethod
    def answers(queries, report, asset, model=None, directory=None):
        missing = MeasuredWork(report).uncovered(queries, model, directory)
        answers = [{"status": "untriggered", "detail": missing[i]} if i in missing
                   else {"status": "qualified", "asset": asset} for i in range(len(queries))]
        return LocalRule.asset(asset, report, answers)

    @staticmethod
    def native_invocation(query):
        return "invocation" in query and LocalSchedulerBatch.supports(query["invocation"] | {"domain": query["domain"]})


class LocalDeviceWork(LocalSchedulerWork):
    name = RuleName("scheduler_device_service")

    def matches(self, query):
        return (query.get("work") in {"device", "transfer"} and not LocalAttention.matches(query)
                and not LocalCopies.matches(query) and self.native_invocation(query))

    def cut(self, queries, session, deployment, arguments):
        return LocalSchedulerBatch.work_job(queries, deployment)

    def qualify(self, queries, recording, timeout=None):
        worker, = WorkerCapture.observed(recording)
        report, _ = CostSamples.worker_services(worker, self.compiler(timeout), recording.directory.parent)
        return self.answers(queries, report, "samples/device.json")


class LocalPeerWork(LocalSchedulerWork):
    name = RuleName("peer_service")

    def matches(self, query):
        operand = query.get("features", {}).get("transport")
        return (query.get("work") == "collective" and operand and len(operand["members"]) > 1
                and self.native_invocation(query))

    @staticmethod
    def session_key(query, deployment):
        return LocalSchedulerWork.session_key(query, deployment), query["features"]["transport"]["members"]

    def cut(self, queries, session, deployment, arguments):
        cut = LocalSchedulerBatch.work_job(queries, deployment)
        peers = session[1]
        if cut.job["single_device"] or any(q["features"]["transport"]["members"] != peers for q in queries):
            raise EvidenceGap("missing_physical_peers", required_ranks=peers, physical_devices=cut.cards)

        # Service after rendezvous requires simultaneous arrivals from every peer.
        cut.job["batch"]["peer_ranks"] = peers
        return cut

    def qualify(self, queries, recording, timeout=None):
        model = self.compiler(timeout)
        with TemporaryDirectory(dir=recording.directory.parent) as scratch:
            directory = Path(scratch)
            report = CostSamples.transport(recording, queries, model, directory)

            return self.answers(queries, report, "samples/transport.json", model, directory)


class LocalHostWork(LocalSchedulerWork):
    name = RuleName("scheduler_host_service")

    selected_catalog = None

    def matches(self, query):
        return query.get("work") == "host" and WorkerDomain.native_domain(query["domain"])

    def sources(self, deployment, rank):
        for directory in self.arguments.structure_session:
            receipt = read_json(directory / "result.json")
            if (receipt.get("rule") not in {LocalSchedulerBatch.name, LocalWorkerBinding.name}
                    or receipt["status"] not in {"qualified", "incomplete"}):
                continue
            original = PublishedCapture(directory / "native")
            original.qualified_receipt(receipt)
            if original.inputs["deployment"] == deployment:
                yield from (worker for worker in WorkerCapture.observed(original) if worker.rank == rank)

    def cut(self, queries, session, deployment, arguments):
        domain = queries[0]["domain"]
        requirements = CostRequirements()
        needed = {requirements.gap("query", q) for q in queries}
        image = DockerImage.id(arguments.source_image)
        model = self.compiler(arguments.preparation_seconds)
        for worker in self.sources(deployment, WorkerDomain.rank(domain)):
            source = worker.recording
            job = source.inputs["job"]
            if (source.inputs["image"] != image
                    or WorkerDomain.execution_domain(worker.read("facts")["environment"]) != domain):
                continue
            _, catalog = worker.measure(model, arguments.output.parent)
            points = {point for _, point in requirements.observe_graph("source", catalog)}
            if needed <= points:
                self.selected_catalog = source.attempt_id, worker.rank, catalog
                return LocalCut(job, source.inputs["physical_devices"], source.inputs["framework_revision"],
                                job["model"]["path"], {"template_recording": source.attempt_id},
                                worker="--local-host", host_timing=True)

        # Recreate a live cut when the portable release has no raw structural probe.
        return replace(LocalSchedulerBatch.work_job(queries, deployment), worker="--local-host-pair", host_timing=True)

    def qualify(self, queries, recording, timeout=None):
        inputs = recording.inputs
        rank = WorkerDomain.rank(queries[0]["domain"])
        if "probes" in recording.manifest:
            original, timed = (WorkerCapture(recording, rank, name) for name in ("structure", "timing"))
        else:
            original = next((w for w in self.sources(inputs["deployment"], rank)
                             if w.recording.attempt_id == inputs["template_recording"]), None)
            if original is None:
                raise EvidenceGap("missing_local_structure_source", required_recording=inputs["template_recording"])
            timed = WorkerCapture(recording, rank)
        if self.selected_catalog and self.selected_catalog[:2] == (original.recording.attempt_id, rank):
            catalog = self.selected_catalog[2]
        else:
            _, catalog = original.measure(self.compiler(timeout), recording.directory.parent)
        report = HostCostSamples.local(original, timed, catalog)
        return self.answers(queries, report, "samples/host.json")


class HostWindow:
    native_directory = "."

    def __init__(self, owner, name, job, points):
        self.owner, self.name, self.job = owner, name, job
        self.points = points
        self.ready = cost_identity(job) not in owner.measured_host

    def collect(self, arguments, requested, limit, processing):
        source = self.owner.source
        launch = type(self.owner.launch)(self.job, source.deployment)
        launch.require_recorded_placement(source.recording.inputs["deployment"])
        arguments = replace(arguments, source_image=source.recording.inputs["image"])
        launch.capture(arguments, limit, host_timing=True).run()

    def samples(self, output, acquisition):
        source, work = self.owner.source, self.owner.source.projection / "work"
        for rank in source.ranks:
            path = source.output / f"{self.name}-{rank}-host.json"
            options = HostSampleOptions(detailed=source.recording.directory, host=output,
                        facts=work / f"{rank}-facts.json", trace=work / f"{rank}-sdk/trace.json",
                        graph=work / f"{rank}-graph.json", work_identities=source.identities(rank), output=path)
            acquisition.sample(path, acquisition.sample_adapter.host_samples, options)
            yield path, read_json(path)


class SourceRuleWindow:
    """Pass demand from an explicitly supplied training graph to the same local rule as direct queries."""

    ready = True
    native_directory = "native"

    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self.points = {}

    def collect(self, arguments, requested, limit, processing):
        source = self.owner.source
        self.owner.launch.require_recorded_placement(source.recording.inputs["deployment"])
        queries = sorted((self.points[point] for point in requested), key=content_id)
        receipt = self.owner.sessions.run(
            arguments, queries, source.deployment, limit, processing, available_queries=self.points.values()
        )
        if receipt["status"] == "allowance_exhausted":
            raise AllowanceExhausted(receipt["detail"])
        if receipt["status"] == "native_failed":
            raise EvidenceGap("capture_failed", detail=receipt["detail"], attempt=receipt["attempt"])
        if receipt["status"] not in {"qualified", "incomplete", "untriggered"}:
            raise EvidenceGap(
                receipt["status"], **{key: receipt[key] for key in ("detail", "query", "attempt") if key in receipt}
            )
        self.assets = receipt["assets"]

    def samples(self, output, acquisition):
        source = self.owner.source
        with acquisition.processing():
            for ordinal, name in enumerate(self.assets):
                product = read_json(asset_path(output, name))
                origins = product["native_recordings"]
                if source.recording.attempt_id not in origins:
                    origins.insert(0, source.recording.attempt_id)
                path = source.output / f"{self.name}-{ordinal}.json"
                write_json(path, product)
                yield path, product


class SGLangSampling:
    def __init__(self, launch, source):
        self.launch, self.source = launch, source
        self.host_points, self.copies, self.attention = set(), {}, {}
        probes = (LocalCopies(), LocalAttention(self.sampling_sources))
        self.sessions = LocalAcquisition(probes, launch.check_local_placement, WorkerCapture.template_resources)
        self.runner_input = AttentionCatalog.runner(launch.job)
        self.single_request_host_points = set()
        self.protocol_services = set()
        self.measured_host = set()
        self.external = HostCostSamples.external_work(source.recording.source)

    @staticmethod
    def cohort_widths(scopes):
        widths = []
        for scope in scopes:
            work = scope["work"]
            context = work.get("context") or {}
            width = max(len(work.get("members", ())), len(context.get("poll_requests", ())))
            parent = scope["parent"]

            # A nested call can have local cost features and still require its enclosing native batch to be present.
            widths.append(width if parent is None else max(width, widths[parent]))
        return widths

    def observe(self, rank, target, catalog, demand):
        card = self.launch.platform.card(rank)
        widths = self.cohort_widths(catalog.graph["scopes"])
        for work, point in demand.observe_graph(target, catalog):
            definition = demand.definitions[point.definition]
            interval = definition["work"][point.interval]
            if HostCostSamples.supports(interval, definition["resources"][interval["resource"]]):
                self.host_points.add(point)
                if widths[work["scope"]] <= 1:
                    self.single_request_host_points.add(point)
            elif point.transport is not None:
                if operand := LocalCopies.operand(definition, point.interval, json.loads(point.transport)):
                    self.copies[point.domain, operand] = card
            elif work["kind"] == "device" and self.runner_input and LocalAttention.matches(demand.work_query(point)):
                self.attention.setdefault((point.domain, point.definition), len(self.attention))

        for identity, kind, features in self.external[rank]:
            point = demand.external(target, catalog.domain, identity, kind, features)
            self.host_points.add(point)
            self.protocol_services.add((point.domain, point.definition, point.interval, point.phase))

    def sampling_sources(self):
        source = self.source
        rank, = source.ranks
        stored = read_json(source.production / "result.json").get("sampling", {}).get(str(rank))
        cut = read_json(asset_path(source.production, stored)) if stored else AttentionCatalog.extract(
            source.recording.source, source.projection / "work", rank, source.identities(rank),
            self.runner_input, device_runtime().scoped_tasks.source)
        yield source.recording.attempt_id, cut

    def local_windows(self, demand):
        windows = {}
        for point in demand.points:
            if point.transport is not None:
                query = demand.work_query(point)
                operand = LocalCopies.operand(query["definition"], point.interval, query["features"]["transport"])
                if (card := self.copies.get((point.domain, operand))) is None:
                    continue
                try:
                    LocalCopies.point(query)
                except EvidenceGap:
                    continue
                name = f"local-copy-card-{card}"
            elif point.phase == "decode" and (point.domain, point.definition) in self.attention:
                lengths = LocalAttention.contexts(dict(point.axes))
                if not LocalAttention.fits(self.launch.serving.instances[0].configuration, lengths):
                    continue
                name = f"decode-attention-{self.attention[point.domain, point.definition]}"
                query = demand.work_query(point)
            else:
                continue
            if name not in windows:
                windows[name] = SourceRuleWindow(self, name)
            windows[name].points[point] = query
        return windows.values()

    def windows(self, demand, samples):
        training = str(self.source.recording.directory)
        observations = (s.get("observation", {}) for s in samples)
        receipts = {Path(o["host"]) for o in observations if o.get("detailed") == training and "host" in o}

        # An imported sample can outlive its native session.
        self.measured_host.update(cost_identity(PublishedCapture(p).inputs["job"]) for p in receipts
                                  if (p / "result.json").is_file())

        yield from self.local_windows(demand)
        job, serving = self.launch.job, self.launch.serving
        yield HostWindow(self, "observed-submission", job, self.host_points)
        if self.single_request_host_points and any(len(group) != 1 for group in serving.submission_groups):
            groups = [[request["id"]] for request in serving.workload["requests"]]
            separate = {**job, "submission_groups": groups}
            yield HostWindow(self, "individual-requests", separate, self.single_request_host_points)
        if len(serving.submission_groups) > 1:

            # A training split can hide the larger envelopes created by the normal serving ingress.
            unseen = {point for point in demand.points if point not in self.host_points
                      and (point.domain, point.definition, point.interval, point.phase) in self.protocol_services}
            if unseen:
                normal = {key: value for key, value in job.items() if key != "submission_groups"}
                yield HostWindow(self, "default-cohorts", normal, unseen)


class BatchHostReuse:
    @staticmethod
    def environment(domain):
        implementation = dict(domain["implementation"])
        implementation.pop("disaggregation_mode")
        if implementation["load_format"] == "dummy":
            implementation["model"] = dict(implementation["model"])
            implementation["model"].pop("_name_or_path", None)
        return cost_identity([domain[k] for k in ("device", "torch", platform_domain().runtime_domain_key)]
                             + [implementation])

    @staticmethod
    def local(definition, interval):
        edges = ("children", "imports", "exports", "kv_accesses", "slot_accesses")
        return (definition.get("role") == "batch_transform" and definition["work"][interval]["kind"] == "host"
                and not any(definition[field] for field in edges))

    @classmethod
    def match(cls, demand, missing, sources):
        environments = {key: cls.environment(domain) for key, domain in demand.domains.items()
                        if domain["framework"] == "sglang"}
        needed = defaultdict(list)
        for point in missing:
            if point.domain in environments and cls.local(demand.definitions[point.definition], point.interval):
                key = replace(point, domain=environments[point.domain])
                needed[key].append(point)
        if not needed:
            return []
        groups, selected = {}, set()
        for path, source in sources.items():
            domain = source["domain"]
            if domain["framework"] != "sglang" or "qualification" not in source:
                continue
            environment = cls.environment(domain)
            native_domain = cost_identity(domain)
            definitions = {index: cost_identity(definition) for index, definition in enumerate(source["definitions"])
                           if definition.get("role") == "batch_transform"}
            for index, row in enumerate(source["samples"]):
                definition = definitions.get(row["definition"])
                if definition is None or not row["observations_ns"] or "copy" in row or "collective" in row:
                    continue
                key = WorkPoint.at(environment, definition, row["interval"], row["features"])
                for point in needed.get(key, ()):
                    if point in selected or point.domain == native_domain:
                        continue

                    # Keep the observed duration on its host interval even when the same call has device work.
                    ordinals, samples = groups.setdefault(point.domain, ({}, []))
                    ordinal = ordinals.setdefault(definition, len(ordinals))
                    samples.append(row | {"definition": ordinal, "observation": {"source": str(path), "row": index}})
                    selected.add(point)
        return [{"domain": demand.domains[key], "definitions": [demand.definitions[d] for d in defs],
                 "samples": samples, "unmeasured": {}} for key, (defs, samples) in groups.items()]


class SGLangLaunch(SGLangPlacement):
    sampling_input = staticmethod(AttentionCatalog.runner)

    @staticmethod
    def export_profile(rank, output):
        return device_runtime().export.command(rank, output)

    @staticmethod
    def interpret(rank, trace, output, *, evidence):
        return ["python3", "-m", "prologue.workers.capture", "interpret", "--input", "/source", "--rank", str(rank),
                "--trace", trace, "--output", output, *(["--evidence"] if evidence else [])]

    @staticmethod
    def sampling_cuts(rank, identities, output):
        return ["python3", "-m", "prologue.workers.capture", "sampling-cuts", "--input", "/source",
                "--rank", str(rank), "--work", "/production/work", "--identities", identities,
                "--runner", "/production/inputs/sampling.json", "--output", output]

    @classmethod
    def local_acquisition(cls, release, arguments):
        attention = LocalAttention(release.templates.sampling_sources if release else None)
        rules = (LocalWorkerBinding(), LocalSchedulerBatch(arguments), LocalHostWork(arguments),
                 LocalDeviceWork(arguments), LocalPeerWork(arguments), LocalCopies(), attention,
                 LocalPrebuilt(), LocalEmbedding())
        origin = release.identity if release else None
        return LocalAcquisition(rules, cls.check_local_placement, WorkerCapture.template_resources, origin)

    def sampling(self, source):
        return SGLangSampling(self, source)

    def capture(self, arguments, limit=None, *, latency_only=False, host_timing=False, service_reference=False,
                attention_sites=None):
        if attention_sites is not None:
            ranks = {str(rank) for rank in range(self.serving.worker_count)}
            if not isinstance(attention_sites, dict) or not attention_sites or not attention_sites.keys() <= ranks:
                raise ValueError("serving comparison requires module lists keyed by participating ranks")
            for sites in attention_sites.values():
                if (not isinstance(sites, list) or not sites or not all(isinstance(s, str) and s for s in sites)
                        or len(sites) != len(set(sites))):
                    raise ValueError("each selected rank requires distinct native module names")

        destination = Path(self.serving.model["path"]).absolute()
        mounts = self.model_mount(destination, arguments.output)
        image = DockerImage.inspect(arguments.source_image)
        inputs = {
            "job": self.job, "deployment": self.platform.source, "image": image["Id"],
            "framework_revision": self.image_revision(image), "model_path": str(destination.resolve(strict=True)),
            "model_destination": str(destination), "resource_trace": arguments.resource_trace or service_reference,
        }
        latency = latency_only or bool(attention_sites)
        flags = {"resource_trace": arguments.resource_trace, "latency_only": latency_only,
                 "attention": attention_sites, "host_timing": host_timing, "service_reference": service_reference}
        if attention_sites:
            inputs["job"] = self.job | {"attention_sites": attention_sites}
        inputs.update({k: True for k in ("host_timing", "service_reference") if flags[k]})
        if latency or host_timing:
            inputs["latency_only"] = True
        runtime = self.platform.options(latency_only=latency, detailed_timing=service_reference,
                                        host_timing=host_timing)
        worker = self.worker_options(reference=service_reference or latency)
        worker.extend("--" + key.replace("_", "-") for key, active in flags.items() if active)
        return NativeAttempt(arguments, inputs, runtime, worker, mounts, self.platform_runtime(), limit)


class HostAdapter:
    prediction = SGLangTask
    capture_input = CaptureInput
    native = SGLangLaunch
    samples = CostSamples
    host_samples = staticmethod(HostCostSamples.export)
    device_samples = staticmethod(CostSamples.export)
    reuse = BatchHostReuse

    @staticmethod
    def revision(value):
        return select(value).revision

    @staticmethod
    def images(selection):
        suffix = "" if selection.revision == default_revision else "-" + selection.revision[:8]
        tag = selection.platform_module().platform.image_tag
        return {"source_image": f"prologue-{selection.framework}{suffix}:{tag}",
                "control_image": f"prologue-control{suffix}:{tag}"}
