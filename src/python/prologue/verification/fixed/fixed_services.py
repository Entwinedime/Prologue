"""Measure fixed-graph service and identify the intervals eligible for modeled replacement."""

from prologue.verification.core.adapters import framework_evidence, platform_evidence

from .fixed_timeline import CpuTransportInterval, ServiceInterval


class FixedServices:
    def __init__(self, case, clock, peers, progress_cuts, cpu_calls, gaps):
        self.case, self.clock, self.peers = case, clock, peers
        self.progress_cuts, self.cpu_calls, self.gaps = progress_cuts, cpu_calls, gaps

    def services(self):
        case = self.case
        for rank, (contents, raw, _) in case.sources.items():
            graph = contents.graph
            document = case.measurements[rank]
            measurements = document["measurements"]
            captured_collectives = {
                (lane, begin, end)
                for _, lane, begin, end, payload in measurements.get("collectives", ())
                if payload is not None
            }
            services = platform_evidence().observed_services(graph, document, {}, physical_peers=case.joint_execution)
            isolated = case.measured_service_pairs(rank, graph, raw, resources=document["resources"])

            for index, piece in enumerate(graph["work"]):
                start, finish = piece["start"], piece["finish"]
                host = piece["kind"] == "host"
                if host:
                    duration = services.host_service(piece)
                    basis = "observed_host_occupancy" if duration is not None else "missing"
                elif piece["kind"] == "collective":
                    window = self.collective_window(rank, graph, piece, captured_collectives)
                    duration = window[1] - window[0] if window is not None else None
                    resource = graph["resources"][piece["resource"]]
                    basis = "observed_host_collective" if resource["kind"] == "thread" else "observed_device_collective"

                    if duration is None:
                        basis = "missing"
                else:
                    duration, _, _ = services.service(piece)
                    payload = document["device_payload_times"].get((start, finish))
                    if duration is not None and payload:
                        duration = sum(self.clock.device_duration(rank, first, last) for first, last in payload)

                    basis = "observed_payload" if duration is not None else "missing"

                if duration is None and not host and (start, finish) in isolated:
                    duration = self.clock.timestamp(rank, finish) - self.clock.timestamp(rank, start)
                    basis = "isolated_observed_interval"

                cuts = self.progress_cuts.get((rank, start, finish), ())
                if any(cut.service_ns != duration for cut in cuts):
                    self.gaps.setdefault("unqualified_work_service_partitions", []).append((rank, start, finish))
                yield ServiceInterval(rank, index, start, finish, piece["kind"], duration, basis, cuts)

        for call in self.cpu_calls:
            rank = call.entered[0]
            for start, finish, operation in call.services():
                duration = case.measurements[rank]["fixed_cpu_calls"][start, finish]
                cuts = self.progress_cuts.get((rank, start, finish), ())
                if any(cut.service_ns != duration for cut in cuts):
                    self.gaps.setdefault("unqualified_work_service_partitions", []).append((rank, start, finish))

                basis = "observed_cpu_protocol_occupancy" if duration is not None else "missing_cpu_protocol_occupancy"
                yield ServiceInterval(rank, None, start, finish, "host", duration, basis, cuts, call, operation)

    def collective_window(self, rank, graph, piece, captured):
        start, finish = piece["start"], piece["finish"]
        destination = rank, finish
        if destination not in self.peers:
            return None

        posts = self.peers[destination] | {(rank, start)}
        lane = piece["resource"]
        if graph["resources"][lane]["kind"] == "thread":
            ports = {port["name"]: port["event"] for port in graph["scopes"][piece["scope"]]["ports"]}
            if (ports.get("call.enter"), ports.get("call.return")) != (start, finish):
                return None

            # Host participants share the RAW clock. Require real call boundaries for the full rendezvous.
            if any(type(self.clock.point(*event)) is not int for event in posts | {destination}):
                return None
        else:
            events = graph["evidence"]["events"]
            if (lane, events[start]["order"], events[finish]["order"]) not in captured:
                return None

        readiness = [self.clock.timestamp(*event) for event in posts]
        completion = self.clock.timestamp(*destination)
        return None if completion is None or None in readiness else (max(readiness), completion)

    def work_fields(self, interval):
        if isinstance(interval, CpuTransportInterval):
            descriptor = interval.post.descriptor
            return {
                "kind": interval.kind,
                "scope": "cpu_protocol",
                "site": framework_evidence().cpu_protocol.readiness_operation(descriptor),
                "phase": "",
                "direction": "receive" if descriptor["receiving"] else "send",
                "caller": None,
            }

        if call := interval.cpu_call:
            return {
                "kind": interval.kind,
                "scope": "cpu_protocol",
                "site": interval.cpu_operation,
                "phase": "",
                "direction": "receive" if call.descriptor["receiving"] else "send",
                "caller": None,
            }

        graph = self.case.sources[interval.rank][0].graph
        piece = graph["work"][interval.index]
        scope = graph["scopes"][piece["scope"]]
        copy = piece.get("copy", {})
        return {
            "kind": interval.kind,
            "scope": scope["role"],
            "site": scope["site"],
            "phase": scope["work"].get("phase"),
            "direction": copy.get("direction"),
            "caller": copy.get("caller"),
        }

    def detail(self, interval):
        if isinstance(interval, CpuTransportInterval):
            call = interval.post
            return {
                "rank": call.returned[0],
                **self.work_fields(interval),
                "message": call.key,
                "consumer": interval.consumer,
                "ready_window_ns": interval.window_ns,
            }

        return {
            "rank": interval.rank,
            "work_index": interval.index,
            **self.work_fields(interval),
            "start": interval.start,
            "finish": interval.finish,
        }

    def external_cost(self, interval):
        if isinstance(interval, CpuTransportInterval):
            call = interval.post
            operation = framework_evidence().cpu_protocol.readiness_operation(call.descriptor)
            execution = interval.execution
        elif call := interval.cpu_call:
            operation, execution = interval.cpu_operation, call.execution
        else:
            return None
        identity, features = framework_evidence().cpu_protocol.service(call.descriptor, execution, operation)
        return call.returned[0], identity, features
