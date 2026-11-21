"""Bind declared physical resources to one native Ascend capture container."""

from pathlib import Path


class AscendDeployment:
    @staticmethod
    def is_single_device(source: dict) -> bool:
        mode = source.get("acquisition_mode", "full")
        if mode not in ("full", "single_device"):
            raise ValueError("Ascend acquisition_mode must be full or single_device")
        return mode == "single_device"

    def __init__(self, source: dict, workers: int, transfer: bool) -> None:
        if source.keys() - {"acquisition_mode", "physical_devices", "store_url", "transfer_protocol"}:
            raise ValueError("Ascend deployment contains unsupported fields")
        self.single_device = self.is_single_device(source)

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

            self.environment["ASCEND_MF_STORE_URL"] = source["store_url"]
            self.environment["ASCEND_MF_TRANSFER_PROTOCOL"] = protocol

    def options(self, *, detailed_timing=False, host_timing=False, latency_only=False) -> list[str]:
        arguments = ["--runtime", "ascend", "--ipc", "host", "--network", self.network]
        if not self.single_device:
            arguments.append("--privileged")
        elif detailed_timing or host_timing:

            # Perf pins one ring per worker thread. Docker's default memlock limit fails for larger groups.
            arguments.extend(["--cap-add", "PERFMON", "--ulimit", "memlock=-1:-1"])

        # Offline placement comparison needs only the declaration. Native execution needs the driver mounts.
        for path in ("/usr/local/Ascend/driver", "/etc/ascend_install.info", "/etc/hccn.conf"):
            mounted = Path(path).resolve(strict=True)
            arguments.extend(["--mount", f"type=bind,src={mounted},dst={path},readonly"])

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
