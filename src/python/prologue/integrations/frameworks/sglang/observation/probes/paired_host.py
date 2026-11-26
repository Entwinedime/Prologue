"""Register the two-pass cut used to measure uninstrumented scheduler CPU service."""

from . import CaptureProbe
from .scheduler import WorkerProbes


def paired_host(invocation):
    WorkerProbes(invocation.revision).pair(invocation.source, invocation.output)


probes = (CaptureProbe("local-host-pair", "capture and independently time one local worker cut", paired_host),)
