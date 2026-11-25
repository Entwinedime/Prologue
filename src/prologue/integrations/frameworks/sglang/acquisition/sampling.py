"""Choose native measurements from work actually observed in an SGLang deployment."""

import json
from dataclasses import replace
from pathlib import Path

from prologue.acquisition.options import HostSampleOptions
from prologue.acquisition.recording import PublishedCapture
from prologue.acquisition.rules import EvidenceGap, LocalAcquisition
from prologue.integrations.frameworks.sglang.acquisition.host_cost_samples import HostCostSamples
from prologue.integrations.frameworks.sglang.acquisition.local_attention import LocalAttention
from prologue.integrations.frameworks.sglang.acquisition.local_copy import LocalCopies
from prologue.integrations.frameworks.sglang.acquisition.worker_capture import WorkerCapture
from prologue.integrations.frameworks.sglang.shared.attention import AttentionCatalog
from prologue.integrations.platforms.ascend.scoped_tasks import ScopedTasks
from prologue.libraries.release import asset_path
from prologue.performance.requirements import cost_identity
from prologue.runtime.io import content_id, read_json, write_json
from prologue.runtime.limits import AllowanceExhausted


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
            options = HostSampleOptions(
                detailed=source.recording.directory,
                host=output,
                facts=work / f"{rank}-facts.json",
                trace=work / f"{rank}-sdk/trace.json",
                graph=work / f"{rank}-graph.json",
                work_identities=source.identities(rank),
                output=path,
            )
            acquisition.sample(path, acquisition.sample_adapter.host_samples, options)
            yield path, json.loads(path.read_bytes())


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
        self.sessions = LocalAcquisition(
            (LocalCopies, LocalAttention(self.sampling_sources)),
            launch.check_local_placement,
            WorkerCapture.template_resources,
        )
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
        platform = self.launch.platform
        card = platform.card(rank)
        widths = self.cohort_widths(catalog.graph["scopes"])
        for work, point in demand.observe_graph(target, catalog):
            definition = demand.definitions[point.definition]
            interval = definition["work"][point.interval]
            if HostCostSamples.supports(interval, definition["resources"][interval["resource"]]):
                self.host_points.add(point)
                if widths[work["scope"]] <= 1:
                    self.single_request_host_points.add(point)
            elif point.transport is not None:
                operand = LocalCopies.operand(definition, point.interval, json.loads(point.transport))
                if operand:
                    self.copies[point.domain, operand] = card
            elif work["kind"] == "device" and self.runner_input is not None:
                key = point.domain, point.definition
                if key not in self.attention and LocalAttention.matches(demand.work_query(point)):
                    self.attention[key] = len(self.attention)
        for identity, kind, features in self.external[rank]:
            point = demand.external(target, catalog.domain, identity, kind, features)
            self.host_points.add(point)
            self.protocol_services.add((point.domain, point.definition, point.interval, point.phase))

    def sampling_sources(self):
        source = self.source
        rank, = source.ranks
        publication = json.loads((source.production / "result.json").read_bytes())
        path = publication.get("sampling", {}).get(str(rank))
        if path:
            cut = json.loads((source.production / path).read_bytes())
        else:
            cut = AttentionCatalog.extract(
                source.recording.source,
                source.projection / "work",
                rank,
                source.identities(rank),
                self.runner_input,
                ScopedTasks.source,
            )
        yield source.recording.attempt_id, cut

    def local_windows(self, demand):
        windows = {}
        for point in demand.points:
            definition = demand.definitions[point.definition]
            if point.transport is not None:
                operand = LocalCopies.operand(definition, point.interval, json.loads(point.transport))
                card = self.copies.get((point.domain, operand))
                if card is None:
                    continue
                name = f"local-copy-card-{card}"
            elif (point.domain, point.definition) in self.attention and point.phase == "decode":
                index = self.attention[point.domain, point.definition]
                name = f"decode-attention-{index}"
            else:
                continue
            query = demand.work_query(point)
            if point.transport is not None:
                try:
                    LocalCopies.point(query)
                except EvidenceGap:
                    continue
            elif not LocalAttention.fits(
                self.launch.serving.instances[0].configuration, LocalAttention.contexts(dict(point.axes))
            ):
                continue
            if name not in windows:
                windows[name] = SourceRuleWindow(self, name)
            windows[name].points[point] = query
        return windows.values()

    def windows(self, demand, samples):
        receipts = set()
        for sample in samples:
            observation = sample.get("observation", {})
            if observation.get("detailed") == str(self.source.recording.directory) and "host" in observation:
                receipts.add(Path(observation["host"]))
        for path in receipts:

            # Imported standalone samples remain usable even when their native session is not available for replay.
            if (path / "result.json").is_file():
                self.measured_host.add(cost_identity(PublishedCapture(path).inputs["job"]))

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
