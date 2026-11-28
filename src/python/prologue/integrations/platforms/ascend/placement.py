"""Place native worker processes on the CPU and memory local to their physical devices."""

import ctypes
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def read_cpu_list(path: Path) -> tuple[int, ...]:
    cpus = []
    for interval in path.read_text().strip().split(","):
        first, _, last = interval.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return tuple(cpus)


@dataclass(frozen=True)
class WorkerPlacement:
    physical_device: int
    numa_node: int
    cpus: tuple[int, ...]

    environment = "PROLOGUE_ASCEND_WORKER_PLACEMENT"
    pci_driver = Path("/sys/bus/pci/drivers/devdrv_device_driver")
    numa_nodes = Path("/sys/devices/system/node")
    online_cpus = Path("/sys/devices/system/cpu/online")
    mpol_bind = 2

    @classmethod
    def select(cls, physical_devices):
        peers = defaultdict(list)
        known = set()
        for entry in cls.pci_driver.glob("*/dev_id"):
            device = int(entry.read_text())
            node = int((entry.parent / "numa_node").read_text())
            if node < 0 or device in known:
                raise ValueError(f"invalid physical device locality for {entry.parent.name}")
            peers[node].append(device)
            known.add(device)

        if missing := set(physical_devices) - known:
            raise ValueError(f"missing Ascend PCI/NUMA locality for physical devices {sorted(missing)}")

        online = set(read_cpu_list(cls.online_cpus))
        placement = {}
        for node, devices in peers.items():
            cpus = [cpu for cpu in read_cpu_list(cls.numa_nodes / f"node{node}" / "cpulist") if cpu in online]
            width, extra = divmod(len(cpus), len(devices))
            if not width:
                raise ValueError(f"NUMA node {node} has fewer online CPUs than physical devices")

            # Partition by all physical peers so a card keeps the same cores in every capture mode.
            start = 0
            for index, device in enumerate(sorted(devices)):
                end = start + width + (index < extra)
                placement[device] = cls(device, node, tuple(cpus[start:end]))
                start = end

        return [placement[device] for device in physical_devices]

    def bind_memory(self):
        numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
        nodes = numa.numa_num_possible_nodes()
        if not 0 <= self.numa_node < nodes:
            raise ValueError(f"NUMA node {self.numa_node} is outside the kernel memory mask")

        word_bits = 8 * ctypes.sizeof(ctypes.c_ulong)
        mask_type = ctypes.c_ulong * ((nodes + word_bits - 1) // word_bits)
        mask = mask_type()
        word, bit = divmod(self.numa_node, word_bits)
        mask[word] = 1 << bit
        limit = ctypes.c_ulong(nodes)
        if numa.set_mempolicy(self.mpol_bind, mask, limit):
            error = ctypes.get_errno()
            raise OSError(error, f"set_mempolicy(MPOL_BIND, node {self.numa_node}): {os.strerror(error)}")

        mode, actual = ctypes.c_int(), mask_type()
        if numa.get_mempolicy(ctypes.byref(mode), actual, limit, None, 0):
            error = ctypes.get_errno()
            raise OSError(error, f"get_mempolicy: {os.strerror(error)}")
        if mode.value != self.mpol_bind or list(actual) != list(mask):
            raise RuntimeError(f"kernel did not bind native worker memory to NUMA node {self.numa_node}")

    @classmethod
    def bind_worker(cls, logical_device):
        encoded = os.environ.get(cls.environment)
        if encoded is None:
            raise RuntimeError("native worker placement is missing; launch capture through the host CLI")
        worker = cls(**json.loads(encoded)[logical_device])

        # The worker must bind before the native scheduler allocates memory or creates execution threads.
        os.sched_setaffinity(0, worker.cpus)
        cpus = sorted(os.sched_getaffinity(0))
        if cpus != list(worker.cpus):
            raise RuntimeError(f"kernel restricted the CPU binding for physical device {worker.physical_device}")
        worker.bind_memory()
        print("prologue: worker placement " + json.dumps({
            "pid": os.getpid(), "logical_device": logical_device, "physical_device": worker.physical_device,
            "numa_node": worker.numa_node, "cpus": cpus, "memory_policy": "MPOL_BIND",
        }), flush=True)
