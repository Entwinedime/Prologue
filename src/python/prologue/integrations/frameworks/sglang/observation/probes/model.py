"""Register device and model-only observations with their native runner prerequisites."""

from prologue.integrations.bindings import device_runtime

from . import CaptureProbe


def copy(invocation):
    device_runtime().copy_sampler().sample(invocation.job, invocation.output, invocation.revision)


def attention(invocation):
    from ..model_samples import AttentionRunner

    AttentionRunner(invocation.job, invocation.output).run(invocation.revision)


def embedding(invocation):
    from ..model_samples import EmbeddingRunner

    EmbeddingRunner(invocation.job, invocation.output).run(invocation.revision)


probes = (
    CaptureProbe("local-copy", "time isolated native tensor copies on one device", copy),
    CaptureProbe("local-attention", "measure Decode on a native ModelRunner and KV pool", attention),
    CaptureProbe(
        "local-embedding", "observe a vocabulary lookup across native TP shards", embedding, runtime_queues=True
    ),
)
