"""Common placement, session identity and compiler contracts for native local work."""

from prologue.acquisition.options import LocalAcquisitionOptions, QualificationOptions
from prologue.acquisition.rules import EvidenceGap
from prologue.integrations.frameworks.sglang.acquisition.worker_templates import WorkerDomain
from prologue.integrations.platforms.ascend.host.deployment import AscendDeployment
from prologue.runtime.docker import ModelingProcess


class LocalRule:
    @classmethod
    def common_session(cls, queries, deployment, *conditions):
        identity = cls.session_key(queries[0], deployment)
        if any(cls.session_key(query, deployment) != identity for query in queries[1:]):
            raise EvidenceGap("different_sessions_required", conditions=list(conditions))
        return identity

    @staticmethod
    def physical_card(domain, deployment):
        workers = max(map(int, domain["worker_roles"])) + 1
        placement = AscendDeployment(deployment, workers, False)
        return placement.card(WorkerDomain.rank(domain))

    @staticmethod
    def binary(options: QualificationOptions | LocalAcquisitionOptions, missing):
        if options.binary is None:
            raise EvidenceGap(missing, required="--binary")
        return options.binary.resolve(strict=True)

    @classmethod
    def compiler(cls, options, seconds, missing):
        binary = cls.binary(options, missing)
        timeout = options.timeout if seconds is None else min(options.timeout, seconds)
        return ModelingProcess(binary, options.modeling_image, timeout)
