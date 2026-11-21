"""Resolve owned service intervals from Ascend measurements and graph coordinates."""

from bisect import bisect_left


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
                if any(access["pool"] != "local" for field in ("kv_accesses", "slot_accesses") for access in scope[field]):
                    return None, None, None
            copy = self.transfers.get((lane, begin, end))
            return (copy[0], [copy[1], copy[2]], None) if copy is not None else (None, None, None)

        positions, payload = self.positions[lane], self.payload[lane]
        first = bisect_left(positions, begin)
        last = bisect_left(positions, end, lo=first)
        if first == last or (first and payload[first - 1][1] > begin) or payload[last - 1][1] > end:
            raise ValueError("device work does not enclose complete native payload")
        durations = [self.services.get((lane, start, finish)) for start, finish in payload[first:last]]
        return (None if None in durations else sum(durations)), None, None
