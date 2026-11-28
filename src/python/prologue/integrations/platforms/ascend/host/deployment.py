"""Allocate physical Ascend devices for a native capture."""

import json
from dataclasses import asdict
from pathlib import Path

from ..placement import WorkerPlacement


class AscendDeployment:
    """Map visible physical devices to logical ranks for one native attempt."""

    driver_mounts = ("/usr/local/Ascend/driver", "/etc/ascend_install.info", "/etc/hccn.conf")

    @staticmethod
    def source_docker_runtime() -> str:
        return "ascend"

    @staticmethod
    def capture_entrypoint() -> str:
        return "/opt/prologue/capture.sh"

    @staticmethod
    def is_single_device(source: dict) -> bool:
        mode = source.get("acquisition_mode", "full")
        if mode not in ("full", "single_device"):
            raise ValueError("Ascend acquisition_mode must be full or single_device")
        return mode == "single_device"

    def __init__(self, source: dict, workers: int, transfer: bool) -> None:
        if source.keys() - {"acquisition_mode", "physical_devices", "store_url", "transfer_protocol", "compiler_cache"}:
            raise ValueError("Ascend deployment contains unsupported fields")
        self.single_device = self.is_single_device(source)
        self.compiler_cache = source.get("compiler_cache")
        if self.compiler_cache is not None and not (
            isinstance(self.compiler_cache, str) and Path(self.compiler_cache).is_absolute()
        ):
            raise ValueError("Ascend compiler_cache must be an absolute host directory")

        # Local shard peers share a container. Only physical groups and PD need the host fabric or store.
        self.network = "bridge" if self.single_device and not transfer else "host"
        devices = source["physical_devices"]
        if any(type(device) is not int or device < 0 for device in devices) or devices != sorted(set(devices)):
            raise ValueError("Ascend physical_devices must be distinct nonnegative integers in increasing order")
        if self.single_device and len(devices) != 1:
            raise ValueError("single-device acquisition requires exactly one physical device")
        if not self.single_device and len(devices) < workers:
            raise ValueError(f"native protocol requires {workers} physical devices, got {len(devices)}")

        self.source = source
        self.devices = devices if self.single_device else devices[:workers]

        # The container runtime mounts only the selected card; the native worker then sees device zero.
        visibility = "ASCEND_VISIBLE_DEVICES" if self.single_device else "ASCEND_RT_VISIBLE_DEVICES"
        self.environment = {visibility: ",".join(map(str, self.devices)), "OMP_NUM_THREADS": "1"}
        if transfer and not source.get("store_url"):
            raise ValueError("Ascend PD capture needs a reachable store_url")
        if "transfer_protocol" in source and "store_url" not in source:
            raise ValueError("transfer_protocol needs a store_url")

        if "store_url" in source:
            protocol = source.get("transfer_protocol", "sdma")

            # The native engine silently falls back to SDMA for an unknown protocol.
            if protocol not in ("sdma", "device_rdma"):
                raise ValueError("Ascend transfer_protocol must be sdma or device_rdma")

            if transfer:
                self.environment["ASCEND_MF_STORE_URL"] = source["store_url"]
                self.environment["ASCEND_MF_TRANSFER_PROTOCOL"] = protocol

    def card(self, logical_rank):
        return self.devices[0 if self.single_device else logical_rank]

    def options(self, *, detailed_timing=False, host_timing=False, latency_only=False) -> list[str]:
        placement = WorkerPlacement.select(self.devices)
        cpus = sorted({cpu for worker in placement for cpu in worker.cpus})
        nodes = sorted({worker.numa_node for worker in placement})
        arguments = [
            "--runtime", self.source_docker_runtime(), "--ipc", "host", "--network", self.network,
            "--cpuset-cpus", ",".join(map(str, cpus)), "--cpuset-mems", ",".join(map(str, nodes)),
            "--env", f"{WorkerPlacement.environment}={json.dumps([asdict(worker) for worker in placement])}",
        ]
        if not self.single_device:
            arguments.append("--privileged")
        else:

            # Docker's default seccomp profile allows NUMA memory policy with SYS_NICE.
            arguments.extend(["--cap-add", "SYS_NICE"])
            if detailed_timing or host_timing:
                arguments.extend(["--cap-add", "PERFMON", "--ulimit", "memlock=-1:-1"])

        # Offline placement comparison needs only the declaration. Native execution needs the driver mounts.
        for path in self.driver_mounts:
            mounted = Path(path).resolve(strict=True)
            arguments.extend(["--mount", f"type=bind,src={mounted},dst={path},readonly"])

        if self.compiler_cache:

            # Persist compiler outputs across fresh runtime processes and their normal warmup.
            destination = "/opt/prologue/compiler-cache"
            arguments.extend(["--mount", f"type=bind,src={self.compiler_cache},dst={destination}",
                              "--env", f"TRITON_CACHE_DIR={destination}"])

        if detailed_timing:

            # Kernel wake recipients use the host PID namespace even though workers keep their private namespace.
            arguments.extend([
                "--mount", "type=bind,src=/sys/kernel/tracing/events,dst=/opt/prologue/trace-events,readonly",
                "--mount", "type=bind,src=/proc,dst=/opt/prologue/host-proc,readonly",
            ])

        for name, value in self.environment.items():
            arguments.extend(["--env", f"{name}={value}"])
        if latency_only:
            arguments.extend(("--env", "LD_PRELOAD="))

        return arguments
