"""Prepare native execution conditions and independently check their actual consumption."""

import json
from collections import Counter, defaultdict

from prologue.common.io import read_json, write_json
from prologue.verification.core.ingress import IngressEvidence
from prologue.verification.core.native_case import NativeCase
from prologue.verification.core.source_clock import ClockRealization
from prologue.verification.engine import ContextValues


class NativeInputs:
    def __init__(self, capture, graphs, output, layouts=None, *, peer_handoffs=False):
        self.capture, self.graphs, self.output = capture, graphs, output
        self.layouts, self.peer_handoffs = layouts, peer_handoffs
        self.steps = defaultdict(list)
        self.cpu_positions = {}
        self.cpu_consumers = set()
        self.origins, self.ingress = {}, {}
        self.counts = Counter()
        self.contexts = ContextValues()
        self.layout_values = ContextValues()
        self.realization = None

    @staticmethod
    def window(begin, end, origin):
        if begin > end or end < origin:
            raise ValueError("native condition has a reversed or pre-workload interval")
        return {"begin_ns": max(begin - origin, 0), "end_ns": end - origin}

    @staticmethod
    def port(name, direction, window):
        return {"at": {"port": name, "direction": direction}, "window": window}

    def actions(self, case, rank, realization, origin):
        group = case.groups[rank]
        source, raw, identities = case.sources[rank]
        graph, evidence = source.graph, source.graph["evidence"]

        if evidence["clock"] != "monotonic_raw_ns":
            raise ValueError("native input conditions require the common RAW clock")
        layouts = None if self.layouts is None else read_json(self.layouts / f"{rank}-layouts.json")
        if layouts is not None and set(layouts) != {str(index) for index in identities}:
            raise ValueError("timing layouts and independently observed action scopes differ")
        if layouts is not None and sum(len(layout["identity"][-1]) for layout in layouts.values()) != len(
            graph["work"]
        ):
            raise ValueError(f"rank {rank} action cuts omit or repeat native work")

        services = case.measured_service_pairs(rank, graph, raw) if layouts is not None else set()
        self.counts["unmeasured_work_finishes"] += sum(
            evidence["events"][work["finish"]]["time_ns"] is None for work in graph["work"]
        )

        for index, identity in identities.items():
            scope = graph["scopes"][index]
            entered = next(port["event"] for port in scope["ports"] if port["name"] == "call.enter")
            begin = evidence["events"][entered]["time_ns"]

            if begin is None:
                raise ValueError(f"rank {rank} action {index} event {entered} has no measured entry")
            if "work" not in identity:
                raise ValueError(f"rank {rank} action {index} has no complete logical identity")

            record = {
                "operation": "action",
                "identity": self.contexts.encode({name: identity[name] for name in ("action", "parameters", "work")}),
                "source": {"scope": index},
            }

            if layouts is not None:
                layout = layouts[str(index)]
                self.layout_conditions(record, layout, case, rank, index, realization, origin, services)

            if layouts is None or self.peer_handoffs:
                self.boundary_conditions(
                    record, None if layouts is None else layout, case, rank, index, realization, origin
                )

            self.steps[group].append((begin, record, None))

    def layout_conditions(self, record, layout, case, rank, index, realization, origin, services):
        clock = case.clocks[rank]
        record["layout"] = self.layout_values.intern(layout["identity"])

        # The full work signature selects the graph. Timing consumes only its two event coordinates.
        work_endpoints = (interval[2:4] for interval in layout["identity"][-1])
        record["services"] = [
            [start, finish]
            for start, finish in work_endpoints
            if (layout["events"][start], layout["events"][finish]) in services
        ]
        record["endpoints"] = []

        for event in layout["events"]:
            measured = clock.window(event)
            if measured is None:
                if event not in case.logical[rank]:
                    raise ValueError(
                        f"rank {rank} action {index} event {event} lacks measurement or a verified order witness"
                    )

                time_ns = None
                self.counts["logical_boundaries"] += 1
            else:
                selected = realization.timestamp(rank, event)
                time_ns = self.window(selected, selected, origin)["begin_ns"]
                self.counts["measured_boundaries"] += 1
                self.counts["calibrated_boundary_ranges"] += measured[0] != measured[1]

            record["endpoints"].append([event, time_ns])

        self.counts["measured_work"] += len(layout["identity"][-1])
        self.counts["isolated_observed_services"] += len(record["services"])

    def boundary_conditions(self, record, layout, case, rank, index, realization, origin):
        clock = case.clocks[rank]
        source = case.sources[rank][0]
        group = case.groups[rank]
        boundaries = []

        for boundary in case.completion_boundaries(rank, index, peer_handoffs=self.peer_handoffs):
            event = boundary["event"]
            measured = clock.window(event)
            if measured is None:
                self.counts["unmeasured_boundaries"] += 1
                continue

            selected = realization.timestamp(rank, event)
            condition = self.window(selected, selected, origin)
            if layout is None:
                self.counts["measured_boundaries"] += 1
                self.counts["calibrated_boundary_ranges"] += measured[0] != measured[1]
            else:
                self.counts["common_port_conditions"] += 1

            boundaries.append({"at": boundary["at"], "window": condition})

        bindings = source.invocation_bindings(index, group)
        if bindings:
            record["bindings"] = bindings
            self.counts["bound_invocations"] += 1

        record.update(service_ns=None, boundaries=boundaries)
        if layout is not None:
            record["host_tails"] = []
            for tail in layout["host_tails"]:
                if clock.window(tail["event"]) is None:
                    continue
                selected = realization.timestamp(rank, tail["event"])
                record["host_tails"].append({**tail, "window": self.window(selected, selected, origin)})

    def input_reads(self, instance, rank, raw, submitted):
        if not raw["request_ingress"]:
            return
        name, group = instance["id"], f"{instance['id']}/{rank}"

        if name in self.ingress:
            raise ValueError("native input has more than one external ingress per instance")
        self.ingress[name] = submitted.receives(raw["request_ingress"])
        if IngressEvidence.attempts(raw)["status"] != "observed":
            raise ValueError("native input reads lack their complete entering contexts")

        origin = submitted.origin_ns
        reads = IngressEvidence.reads(name, raw, submitted.aliases, origin)
        for number, (interval, read) in enumerate(zip(raw["request_input_attempts"], reads, strict=True)):
            self.steps[group].append(
                (
                    interval["begin_raw_ns"],
                    {
                        "operation": "input",
                        "identity": self.contexts.encode(read["identity"]),
                        "source": {"input": number},
                        "window": {"begin_ns": read["time_ns"], "end_ns": read["time_ns"]},
                    },
                    None,
                )
            )

    def cpu(self, evidence, rank, origin):
        group = evidence.groups[rank]
        for call in evidence.cpu.native_calls[rank]:
            if not call.execution["in_flow"]:
                continue
            begin, end = (evidence.cpu.native_times[point] for point in (call.entered, call.returned))
            position = (call.key, rank) if call.phase == "post" else None
            boundaries = [
                self.port("call.enter", "input", self.window(begin, begin, origin)),
                self.port("call.return", "output", self.window(end, end, origin)),
            ]

            if position is not None:
                if call.posted is None:
                    raise ValueError(f"serving CPU post {call.key} at rank {rank} lacks an actual transport boundary")
                posted = evidence.cpu.native_times[call.posted]
                boundaries.append(self.port("transport.post", "output", self.window(posted, posted, origin)))

            if call.phase == "wait" or call.execution["operation"] in {"send", "recv"}:
                self.cpu_consumers.add((call.key, rank))
            self.steps[group].append(
                (
                    begin,
                    {
                        "operation": "cpu",
                        "identity": self.contexts.encode(call.identity()),
                        "source": call.source(),
                        "service_ns": None,
                        "boundaries": boundaries,
                    },
                    position,
                )
            )

    def transfers(self, evidence):
        result = []
        for key, peers in evidence.cpu.messages["native"].items():
            positions = [self.cpu_positions[key, rank] for rank in peers if (key, rank) in self.cpu_positions]
            consumers = [rank for rank in peers if (key, rank) in self.cpu_consumers]
            if not positions and not consumers:
                continue

            origins = {self.origins[evidence.groups[rank]] for rank in peers}
            if len(origins) != 1:
                raise ValueError("native CPU message crosses unrelated workload clocks")
            (origin,) = origins
            timing = evidence.cpu.native_readiness(peers)

            if timing is None or None in timing[1].values():
                raise ValueError(f"CPU transfer {key} lacks independent readiness on both endpoints")
            _, windows = timing
            if not positions:
                if any(windows[rank][1] > origin for rank in consumers):
                    raise ValueError(f"CPU transfer {key} becomes ready during serving without a timed post")
                continue

            # The pairing uses already checked posting conditions, before a consumer can advance.
            posts = sorted(positions, key=lambda row: (row["group"], row["step"]))
            result.append(
                {
                    "posts": posts,
                    "ready": {
                        side: self.window(*windows[rank], origin)
                        for side, rank in zip(("send", "receive"), key[1:3], strict=True)
                    },
                }
            )

        return result

    def load_ingress(self, evidence):
        for instance, session, submitted in zip(
            evidence.manifest["instances"], evidence.sessions, evidence.submissions, strict=True
        ):
            if set(instance["observed_roles"]) != set(session["native_participants"]):
                raise ValueError("native input needs every actual participant")
            origin = submitted.origin_ns

            for rank in session["native_participants"]:
                group = f"{instance['id']}/{rank}"
                self.origins[group] = origin
                _, raw, _ = evidence.sources[rank]
                self.input_reads(instance, rank, raw, submitted)
                self.cpu(evidence, rank, origin)

            if instance["id"] not in self.ingress:
                raise ValueError("native case omits external ingress")

        if len(set(self.origins.values())) != 1:
            raise ValueError("native serving instances have different workload origins")

    def ordered_actors(self):
        actors = {}
        for group in self.origins:
            ordered = sorted(self.steps[group], key=lambda step: step[0])
            if len({begin for begin, _, _ in ordered}) != len(ordered):
                raise ValueError(f"native {group} has unordered operation entries at the same RAW timestamp")
            actors[group] = {"steps": [step for _, step, _ in ordered]}
            for position, (_, _, post) in enumerate(ordered):
                if post is not None:
                    self.cpu_positions[post] = {"group": group, "step": position}
        return actors

    def run(self, reference=None):
        self.output.mkdir(parents=True)
        report = {"capture": str(self.capture.resolve()), "graphs": str(self.graphs.resolve()), "status": "unverified"}
        try:
            evidence = NativeCase.read(self.capture, self.graphs) if reference is None else reference
            if self.peer_handoffs and not any(source[0] != target[0] for source, target, _ in evidence.links):
                raise ValueError("peer handoff timing requires a verified cross-worker dependency")

            self.load_ingress(evidence)

            native_report = evidence.report(False)
            report["native_evidence"] = native_report
            if evidence.graph is None or native_report["status"] == "differences":
                raise ValueError("native source has no consistent causal graph")

            self.realization = ClockRealization(evidence.clocks, evidence.graph, evidence.cpu.native_times).solve()
            for rank in evidence.sources:
                group = evidence.groups[rank]
                self.actions(evidence, rank, self.realization, self.origins[group])

            actors = self.ordered_actors()

            notifications = [
                {
                    "identity": {"group": row["group"], "name": name, "work": {"request": row["request"]}},
                    "window": self.window(
                        row[field]["begin_raw_ns"], row[field]["end_raw_ns"], self.origins[row["group"]]
                    ),
                }
                for row in evidence.readiness["endpoints"]
                for name, field in (("kv_ready", "reservation_update"), ("kv_success", "completion_update"))
                if row[field] is not None and (name != "kv_ready" or row["remote_reservation"])
            ]

            for row in evidence.readiness["reservation_receipts"]:
                group, time = row["recipient"], row["raw_ns"]
                notifications.append(
                    {
                        "identity": {
                            "group": group,
                            "name": "kv_reservation_received",
                            "work": {"request": row["request"], "publisher": row["publisher"]},
                        },
                        "window": self.window(time, time, self.origins[group]),
                    }
                )

            document = {
                "context_values": self.contexts.values,
                "actors": actors,
                "ingress": self.ingress,
                "notifications": notifications,
                "transfers": self.transfers(evidence),
            }

            if self.layouts is not None:
                document["layouts"] = self.layout_values.values
            report.update(
                status="partial"
                if self.counts["unmeasured_boundaries"]
                or self.counts["unmeasured_work_finishes"]
                or native_report["status"] != "consistent"
                else "extracted",
                clock_realization=self.realization.report(),
                counts=dict(self.counts),
                steps={group: len(actor["steps"]) for group, actor in actors.items()},
            )

            if report["status"] == "extracted":
                write_json(self.output / "execution-inputs.json", document, replace=False)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            report.update(status="unverified", error=str(error))

        report["timing_scope"] = "peer_handoffs" if self.peer_handoffs else "observed_execution"
        report["scope"] = (
            "Measured endpoints use one feasible device-clock realization rounded upward to integer "
            "nanoseconds. Independently verified elided waits supply causal witnesses without time "
            "constraints. Original calibration uncertainty and CPU wire and notification bounds remain."
        )
        (self.output / "execution-inputs-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )

        return report
