"""Choose native measurements from work actually observed in an SGLang deployment."""

import json
from collections import defaultdict
from pathlib import Path

from prologue.acquisition.capture import NativeAttempt
from prologue.acquisition.recording import PublishedCapture
from prologue.artifacts import write_json
from prologue.integrations.frameworks.sglang.ascend.attention_window import DecodeSampling
from prologue.integrations.frameworks.sglang.ascend.host_samples import HostCostSamples
from prologue.integrations.platforms.ascend.copy_sampling import TensorCopySamples
from prologue.integrations.platforms.ascend.deployment import AscendDeployment
from prologue.performance.requirements import cost_identity
from prologue.performance.samples import ServiceSamples


class HostWindow:
    def __init__(self, owner, name, job, points):
        self.owner, self.name, self.job = owner, name, job
        self.points = points
        self.ready = cost_identity(job) not in owner.measured_host

    def collect(self, arguments, requested, limit):
        source = self.owner.source
        launch = type(self.owner.launch)(self.job, source.deployment)
        launch.require_recorded_placement(source.recording.inputs["deployment"])
        arguments.source_image = source.recording.inputs["image"]
        launch.capture(arguments, limit, host_timing=True).run()

    def samples(self, output, acquisition):
        source, work = self.owner.source, self.owner.source.projection / "work"
        for rank in source.ranks:
            path = source.output / f"{self.name}-{rank}-host.json"
            sample = acquisition.samples_from(path, "host-cost-samples", detailed=source.recording.directory, host=output,
                                             facts=work / f"{rank}-facts.json", trace=work / f"{rank}-sdk/trace.json",
                                             graph=work / f"{rank}-graph.json", work_identities=source.identities(rank))
            yield path, sample


class LocalCopyWindow:
    ready = True
    services = {cost_identity(TensorCopySamples.service(caller)): caller for caller in TensorCopySamples.directions}

    def __init__(self, owner, card, domain):
        self.owner, self.card = owner, card
        self.name = f"local-copy-card-{card}"
        self.domains, self.roles, self.points = {}, {}, {}
        self.runtime = {key: domain[key] for key in ("device", "torch", "torch_npu")}

    @staticmethod
    def operand(point, definition):
        declared = definition["work"][point.interval].get("copy")
        if not declared or LocalCopyWindow.services.get(point.transport) != declared["caller"]:
            return None
        return declared["caller"], declared["tensor"]["dtype"], declared["element_bytes"]

    def observe(self, rank, domain, point, operand):
        if any(domain[key] != value for key, value in self.runtime.items()):
            raise ValueError("roles observed on the same physical card disagree about its native runtime")
        self.domains[rank] = domain
        self.roles[point.domain, operand] = rank

    def select(self, demand):
        for point in demand.points:
            if point.transport not in self.services:
                continue
            operand = self.operand(point, demand.definitions[point.definition])
            rank = self.roles.get((point.domain, operand))
            if rank is None:
                continue
            size = dict(point.axes)["transfer.bytes"]
            if type(size) is int and size > 0 and size % operand[2] == 0:
                self.points[point] = rank, operand, size

    def collect(self, arguments, requested, limit):
        launch, recording = self.owner.launch, self.owner.source.recording
        launch.require_recorded_placement(recording.inputs["deployment"])
        platform = AscendDeployment({"acquisition_mode": "single_device", "physical_devices": [self.card]}, 1, False)
        image = NativeAttempt.image(arguments.source_image)
        revision = launch.domain.image_revision(image)
        if revision != recording.inputs["framework_revision"]:
            raise ValueError("isolated sampler and originating templates use different framework revisions")

        operands = defaultdict(lambda: defaultdict(set))
        for point in requested:
            rank, operand, size = self.points[point]
            operands[operand][size].add(rank)
        job = {
            "runtime": self.runtime,
            "operands": [{"caller": caller, "dtype": dtype, "element_bytes": width,
                          "points": [{"bytes": size, "ranks": sorted(ranks)} for size, ranks in sorted(sizes.items())]}
                         for (caller, dtype, width), sizes in sorted(operands.items())],
        }
        if (arguments.output / "execution.json").is_file():

            # A completed or interrupted session owns its original selection even if another recipe now covers a point.
            job = json.loads((arguments.output / "input.json").read_text())
        inputs = {
            "job": job, "deployment": platform.source, "image": image["Id"], "framework_revision": revision,
            "training_capture": str(recording.directory),
        }
        NativeAttempt(arguments, inputs, platform.options(), ["--local-copy"], [], limit).run()

    def samples(self, output, acquisition):
        recording = PublishedCapture(output)
        manifest = recording.manifest
        operands = recording.inputs["job"]["operands"]
        if manifest["services"] != {op["caller"]: TensorCopySamples.service(op["caller"]) for op in operands}:
            raise ValueError("isolated sampler measured different native services")
        measured = json.loads((recording.source / manifest["measurements"]).read_text())
        rows = {(row["caller"], row["dtype"], row["bytes"]): row for row in measured}
        results, missing = {}, defaultdict(int)
        for operand in operands:
            service = manifest["services"][operand["caller"]]
            for point in operand["points"]:
                row = rows[operand["caller"], operand["dtype"], point["bytes"]]
                features = {"phase": "", "axes": {"transfer.bytes": row["bytes"]}}
                for rank in point["ranks"]:
                    if rank not in results:
                        results[rank] = ServiceSamples(self.domains[rank])
                    for duration in row["observations_ns"]:
                        results[rank].add_external(service, "transfer", features, duration)
                    missing[rank] += row["unmatched_calls"]
        for rank, samples in results.items():
            path = self.owner.source.output / f"{self.name}-{rank}.json"
            report = samples.document({"unmatched_calls": missing[rank]})
            report["native_recordings"] = [self.owner.source.recording.attempt_id, recording.attempt_id]
            report["observation"] = {
                "sampling": str(output), "training_capture": recording.inputs["training_capture"], "rank": rank,
                "physical_device": self.card, "sampling_image": recording.inputs["image"],
            }
            write_json(path, report)
            yield path, report


class SGLangSampling:
    def __init__(self, launch, source):
        self.launch, self.source = launch, source
        self.host_points, self.copies = set(), {}
        self.single_request_host_points = set()
        self.protocol_services = set()
        self.attention = DecodeSampling(self)
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
        self.attention.observe(rank, catalog)
        platform = self.launch.platform
        card = platform.devices[0 if platform.single_device else rank]
        widths = self.cohort_widths(catalog.graph["scopes"])
        for work, point in demand.observe_graph(target, catalog):
            definition = demand.definitions[point.definition]
            interval = definition["work"][point.interval]
            if HostCostSamples.supports(interval, definition["resources"][interval["resource"]]):
                self.host_points.add(point)
                if widths[work["scope"]] <= 1:
                    self.single_request_host_points.add(point)
            elif operand := LocalCopyWindow.operand(point, definition):
                if card not in self.copies:
                    self.copies[card] = LocalCopyWindow(self, card, catalog.domain)
                self.copies[card].observe(rank, catalog.domain, point, operand)
        for identity, kind, features in self.external[rank]:
            point = demand.external(target, catalog.domain, identity, kind, features)
            self.host_points.add(point)
            self.protocol_services.add((point.domain, point.definition, point.interval, point.phase))

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

        for window in self.copies.values():
            window.select(demand)
            if window.points:
                yield window
        yield from self.attention.windows(demand)
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
