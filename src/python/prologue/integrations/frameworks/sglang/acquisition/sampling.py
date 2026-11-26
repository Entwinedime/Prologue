"""Choose targeted sampling windows from the original training graph."""

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from prologue.common.cost import cost_identity
from prologue.common.io import content_id, read_json, write_json
from prologue.host.acquisition.capture import NativeAttempt
from prologue.host.acquisition.options import HostSampleOptions
from prologue.host.acquisition.recording import LocalSession, PublishedCapture
from prologue.host.acquisition.rules import EvidenceGap
from prologue.host.libraries.release import asset_path
from prologue.host.performance.requirements import WorkPoint
from prologue.host.runtime.limits import AllowanceExhausted
from prologue.integrations.bindings import device_runtime, platform_domain
from prologue.integrations.frameworks.sglang.shared.configuration import (
    AttentionCatalog,
)

from .host_samples import HostCostSamples
from .recipes.attention import LocalAttention
from .recipes.models import LocalCopies


class HostWindow:
    native_attempts = staticmethod(NativeAttempt.attempts)

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
    @staticmethod
    def native_attempts(directory):
        return list(LocalSession(directory).attempts())

    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self.points = {}

    def collect(self, arguments, requested, limit, processing):
        source = self.owner.source
        self.owner.launch.require_recorded_placement(source.recording.inputs["deployment"])
        queries = sorted((self.points[point] for point in requested), key=content_id)
        acquisition = self.owner.launch.local_acquisition(None, arguments, training_sources=self.owner.sampling_sources)
        receipt = acquisition.run(
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
