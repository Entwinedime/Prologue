"""Publish physical peer service measurements for explicitly selected placements."""

import json
from collections import Counter, defaultdict

from prologue.common.cost import ServiceSamples, cost_features, cost_identity


class AscendTransportProfile:
    # HcclDataType is the native payload ABI. Names in tensor ports come from PyTorch.
    dtypes = ("int8", "int16", "int32", "float16", "float32", "int64", "uint64", "uint8",
              "uint16", "uint32", "float64", "bfloat16")
    uniform = {"all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter", "reduce_scatter_tensor",
               "broadcast", "transfer"}

    def __init__(self, worker_sources, peers, trace, observation, image, deployment):
        self.qualification = {"observation": observation, "source_image": image}
        placement = {}
        for actor in peers:
            device, = (worker_sources[actor] / "native" / f"rank-{actor}").glob("*/PROF_*/device_*")
            number = int(device.name.removeprefix("device_"))
            info = json.loads((device / f"info.json.{number}").read_text())
            placement[str(actor)] = {"host": info["hostname"], "device": number, "driver": info["drvVersion"]}
        if len({(p["host"], p["device"]) for p in placement.values()}) != len(peers):
            raise ValueError("external transport calibration requires distinct physical peers")
        self.qualification["placement"] = placement
        self.qualification["logical_workers"] = {
            str(actor): doc["environment"]["parallel"] for actor, doc in peers.items()
        }

        # SDK link ranks are communicator-local. Keep the observed link types separate from the physical rank map.
        links = set()
        for event in json.loads(trace.read_text()):
            args = event.get("args", {})
            link = args.get("link type")
            if link and link not in {"ON_CHIP", "INVALID_TYPE"}:
                links.add((link, args["transport type"]))
        self.qualification["observed_links"] = sorted(links)
        if "transfer_protocol" in deployment:
            self.qualification["state_transfer_protocol"] = deployment["transfer_protocol"]

    @staticmethod
    def state_transfer(catalog, services, piece):
        scope = catalog.graph["scopes"][piece["scope"]]
        accesses = scope["kv_accesses"] + scope["slot_accesses"]
        pools = defaultdict(set)
        for access in accesses:
            if (access["start"], access["finish"]) == (piece["start"], piece["finish"]):
                pools[access["access"]].add(access["pool"])
        if pools["read"] != {"local"} or len(pools["write"]) != 1 or "local" in pools["write"]:
            raise ValueError("physical state transfer requires one remote destination")

        duration, copy, _ = services.service(piece)
        if copy is not None and copy[0] != scope["parameters"]["transfer_bytes"]:
            raise ValueError("measured state transfer changes payload bytes")
        return duration, copy

    def samples(self, catalog, services):
        source = ServiceSamples(catalog.domain)
        absent = Counter()
        receipts = defaultdict(lambda: defaultdict(list))
        coordinates, definitions = {}, {}
        for piece, definition, interval, features in catalog:
            native = catalog.definitions[definition]
            if ServiceSamples.state_transfer(native, interval):
                if "state_transfer_protocol" not in self.qualification:
                    raise ValueError("a physical state transfer profile requires its measured protocol")
                duration, copy = self.state_transfer(catalog, services, piece)
                if duration is None:
                    absent["unmeasured_state_transfer"] += 1
                    continue
                if definition not in definitions:
                    definitions[definition] = len(source.definitions)
                    source.definitions.append(native)
                source.add(definitions[definition], interval, features, duration, copy)
                continue
            if piece["kind"] != "collective":
                continue
            duration, _, payload = services.service(piece)
            if duration is None:
                reason = "unmeasured_peer_service" if payload is None else "unqualified_endpoint_clock"
                absent[reason] += 1
                continue
            identity = features.get("transport")
            if identity is None:
                absent["undeclared_operand_geometry"] += 1
                continue
            operation = identity["operation"]
            row_axes = identity.get("row_axes", ())
            if operation not in self.uniform and not row_axes:
                absent["requires_peer_payload_distribution"] += 1
                continue

            # HCCL reports the receive count for reduce-scatter and the posted buffer for send/receive.
            receives = operation in {"reduce_scatter", "reduce_scatter_tensor"} or (
                operation == "transfer" and identity["rank"] != identity["members"][0])
            ports = identity["ports"].get("output" if receives else "input", {}).values()
            dtype, _, elements = payload
            axes = features["axes"]
            operands = {(port["dtype"], axes[port["elements_axis"]]) for port in ports}
            if not 0 <= dtype < len(self.dtypes):
                absent["sdk_operand_mismatch"] += 1
                continue
            native_dtype = f"torch.{self.dtypes[dtype]}"

            # For variable exchanges the scalar SDK count cannot identify the payload split.
            compatible = (
                {name for name, _ in operands} == {native_dtype} if row_axes else operands == {(native_dtype, elements)}
            )
            if not compatible:
                absent["sdk_operand_mismatch"] += 1
                continue
            payload_axes = {port["elements_axis"]: axes[port["elements_axis"]]
                            for ports in identity["ports"].values() for port in ports.values()}
            payload_axes.update((axis, axes[axis]) for axis in row_axes)
            key = cost_identity((identity, payload_axes))
            coordinates[key] = identity, cost_features(payload_axes)
            receipts[key][tuple(payload)].append(duration)

        for key, variants in receipts.items():
            if len(variants) != 1:
                absent["ambiguous_native_implementation"] += sum(map(len, variants.values()))
                continue
            (payload, durations), = variants.items()
            identity, features = coordinates[key]
            for duration in durations:
                source.add_external(identity, "collective", features, duration, collective=payload)

        return source, absent
