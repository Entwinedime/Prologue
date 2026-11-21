"""Bind native SGLang shard geometry to the available Ascend devices."""

from prologue.integrations.frameworks.sglang.configuration import CaptureInput
from prologue.integrations.platforms.ascend.deployment import AscendDeployment


class SGLangAscendLaunch:
    @staticmethod
    def source_planner(source, deployment):
        from prologue.integrations.frameworks.sglang.acquisition.planning import SourcePlanning

        return SourcePlanning(source, joint_observation=not AscendDeployment.is_single_device(deployment))

    def __init__(self, job, deployment):
        self.job = job
        self.serving = CaptureInput(job)
        self.platform = self.placement(deployment)
        self.local_shards = self.platform.single_device and self.serving.worker_count > 1

    def placement(self, deployment):
        return AscendDeployment(deployment, self.serving.worker_count, len(self.serving.instances) > 1)

    def require_recorded_placement(self, deployment):
        if self.placement(deployment).environment != self.platform.environment:
            raise ValueError("published capture uses a different placement from the planned source")

    def worker_options(self, *, reference):
        if self.local_shards and reference:
            raise ValueError("an independent multi-worker timing reference requires the physical target topology")
        return ["--local-shards"] if self.local_shards else []
