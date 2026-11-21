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
        instances = self.serving.instances
        self.platform = AscendDeployment(deployment, self.serving.worker_count, len(instances) > 1)
        self.local_shards = self.platform.single_device and self.serving.worker_count > 1

    def worker_options(self, *, reference):
        if self.local_shards and reference:
            raise ValueError("an independent multi-worker timing reference requires the physical target topology")
        return ["--local-shards"] if self.local_shards else []
