"""Resolve host activity, measured transfers and collective service costs."""

import json
from bisect import bisect_left
from collections import Counter, defaultdict

from prologue.common.cost import ServiceSamples, cost_features, cost_identity
from prologue.integrations.platforms.ascend.analysis.trace import SynchronousCopy


class ObservedServices:
    """Resolve measurements from the same observation by their actual resource intervals."""

    def __init__(self, graph, facts, collectives, *, physical_peers):
        self.graph, self.collectives = graph, collectives
        self.physical_peers = physical_peers
        measurement = facts["measurements"]

        if (graph["environment"] != facts["environment"]
                or graph["evidence"]["window_ns"] != measurement["window_ns"]):
            raise ValueError("cost measurements and coarse graph come from different native observations")

        self.host_positions = {}
        for lane, order, service_ns in measurement.get("host_service_positions", ()):
            if service_ns < 0 or (lane, order) in self.host_positions:
                raise ValueError("CPU service has a negative prefix or repeated position")
            self.host_positions[lane, order] = service_ns

        self.services = {}
        for lane, begin, end, duration_ns in measurement.get("device_services", ()):
            if duration_ns < 0 or (lane, begin, end) in self.services:
                raise ValueError("physical service has a negative duration or repeated position")
            self.services[lane, begin, end] = duration_ns

        self.transfers = {}
        for lane, begin, end, duration_ns, size, direction in measurement.get("synchronous_transfers", ()):
            if duration_ns < 0 or size <= 0 or (lane, begin, end) in self.transfers:
                raise ValueError("synchronous transfer has invalid or repeated service evidence")
            self.transfers[lane, begin, end] = duration_ns, size, direction

        self.payload = [resource["payload"] for resource in facts["resources"]]
        self.positions = [[begin for begin, _ in ranges] for ranges in self.payload]

    def host_service(self, piece):
        lane = piece["resource"]
        evidence = self.graph["evidence"]["events"]
        begin, end = (self.host_positions.get((lane, evidence[piece[key]]["order"])) for key in ("start", "finish"))
        if begin is None or end is None:
            return None
        if end < begin:
            raise ValueError("CPU service prefixes reverse work order")
        return end - begin

    def service(self, piece):
        lane = piece["resource"]
        evidence = self.graph["evidence"]["events"]
        begin, end = (evidence[piece[side]]["order"] for side in ("start", "finish"))

        if piece["kind"] == "collective":
            duration, payload = self.collectives.get((lane, begin, end), (None, None))
            return duration, None, payload
        if piece["kind"] == "communication":
            return None, None, None

        resource = self.graph["resources"][lane]
        if resource["kind"] != "stream":
            if piece["kind"] != "transfer":
                return None, None, None
            if resource["kind"] == "service" and not self.physical_peers:
                scope = self.graph["scopes"][piece["scope"]]

                # On one card this interval measures staging into another logical worker's pool.
                if any(access["pool"] != "local" for field in ("kv_accesses", "slot_accesses")
                       for access in scope[field]):
                    return None, None, None

            copy = self.transfers.get((lane, begin, end))
            if copy is None:
                return None, None, None
            duration, size, direction = copy

            if "copy" in piece:
                label = piece["copy"]["direction"]
                if direction == SynchronousCopy.directions[label]:
                    direction = label

            return duration, [size, direction], None

        positions, payload = self.positions[lane], self.payload[lane]
        first = bisect_left(positions, begin)
        last = bisect_left(positions, end, lo=first)
        if first == last or (first and payload[first - 1][1] > begin) or payload[last - 1][1] > end:
            raise ValueError("device work does not enclose complete native payload")

        durations = [self.services.get((lane, start, finish)) for start, finish in payload[first:last]]
        return (None if None in durations else sum(durations)), None, None


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
