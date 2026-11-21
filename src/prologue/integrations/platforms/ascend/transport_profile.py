"""Publish portable HCCL service measurements for an explicitly selected physical placement."""

import json
from collections import Counter, defaultdict
from math import prod

from prologue.performance.samples import ServiceSamples


class AscendTransportProfile:
    # HcclDataType is the native payload ABI. Names in tensor ports come from PyTorch.
    dtypes = ("int8", "int16", "int32", "float16", "float32", "int64", "uint64", "uint8",
              "uint16", "uint32", "float64", "bfloat16")
    uniform = {"all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter", "reduce_scatter_tensor",
               "broadcast", "transfer"}

    def __init__(self, capture, peers, trace, observation, image):
        self.qualification = {"observation": observation, "source_image": image}
        placement = {}
        for actor in peers:
            device, = (capture / "native" / f"rank-{actor}").glob("*/PROF_*/device_*")
            number = int(device.name.removeprefix("device_"))
            info = json.loads((device / f"info.json.{number}").read_text())
            placement[str(actor)] = {"host": info["hostname"], "device": number, "driver": info["drvVersion"]}
        if len({(p["host"], p["device"]) for p in placement.values()}) != len(peers):
            raise ValueError("external transport calibration requires distinct physical peers")
        self.qualification["placement"] = placement
        self.qualification["logical_workers"] = {str(actor): doc["environment"]["parallel"] for actor, doc in peers.items()}

        # SDK link ranks are communicator-local. Keep the observed link types separate from the physical rank map.
        links = set()
        for event in json.loads(trace.read_text()):
            args = event.get("args", {})
            link = args.get("link type")
            if link and link not in {"ON_CHIP", "INVALID_TYPE"}:
                links.add((link, args["transport type"]))
        self.qualification["observed_links"] = sorted(links)

    def samples(self, catalog, services):
        source = ServiceSamples(catalog, external_only=True)
        absent = Counter()
        receipts = defaultdict(lambda: defaultdict(list))
        identities = {}
        for piece, _, _, features in catalog:
            if piece["kind"] != "collective":
                continue
            duration, _, payload = services.service(piece)
            if payload is None:
                absent["unmeasured_peer_service"] += 1
                continue
            identity = features.get("transport")
            if identity is None:
                absent["undeclared_operand_geometry"] += 1
                continue
            operation = identity["operation"]
            if operation not in self.uniform:
                absent["requires_peer_payload_distribution"] += 1
                continue

            # HCCL reports the receive count for reduce-scatter and the posted buffer for send/receive.
            receives = operation in {"reduce_scatter", "reduce_scatter_tensor"} or (
                operation == "transfer" and identity["rank"] != identity["members"][0])
            ports = identity["ports"].get("output" if receives else "input", {}).values()
            dtype, _, elements = payload
            operands = {(port["dtype"], prod(port["shape"])) for port in ports}
            if not (0 <= dtype < len(self.dtypes) and operands == {(f"torch.{self.dtypes[dtype]}", elements)}):
                absent["sdk_operand_mismatch"] += 1
                continue
            key = json.dumps(identity, sort_keys=True)
            identities[key] = identity
            receipts[key][tuple(payload)].append(duration)

        for key, variants in receipts.items():
            if len(variants) != 1:
                absent["ambiguous_native_implementation"] += sum(map(len, variants.values()))
                continue
            (payload, durations), = variants.items()
            for duration in durations:
                source.add_external(identities[key], "collective", {"phase": "", "axes": {}}, duration, collective=payload)

        return {**source.document(absent), "transport_profile": self.qualification}
