"""Compare native Prefill and Decode actions and the physical transfer between them."""

from prologue.acquisition.observation.batches import ObservedBatches, action_identity, aligned_batches
from prologue.integrations.frameworks.sglang.page_table import PageTable
from prologue.integrations.platforms.ascend.collectives import PhysicalCollectives
from prologue.verification.parallel import RecordedParallelBatch
from prologue.verification.structure import StructureComparison


class PDStructureReference(StructureComparison):
    def __init__(self, pair):
        if pair.manifest["placement"] != "full":
            raise ValueError("PD structure comparison needs a simultaneous physical reference")
        window, = pair.qualify()
        self.origin = window["origin_raw_ns"]
        self.handoffs = {row["request"]: row for row in window["handoffs"]}
        publications = {request: handoff["publications"] for request, handoff in self.handoffs.items()}
        manifest = {**pair.decode.manifest, "delivery": pair.decode.delivery}
        super().__init__(pair.domain, publications, manifest)

        self.ingress = {
            "arrivals": [{row["request"]: row["submitted_raw_ns"] - self.origin
                          for row in recording.delivery["requests"]} for recording in (pair.prefill, pair.decode)],
            "admission": [{"at_ns": row["at_ns"] - self.origin, "pause": row["kind"] == "paused"}
                          for row in window["controls"]],
        }
        self.publisher = pair.decode.publication_rank()
        self.starts = {}
        self.prebuilt_result_boundaries = 0

        for batch in ObservedBatches(pair.prefill.read(0).document()).entries:
            call = batch.call
            self.starts[action_identity(batch.action())] = call["begin_ns"] - self.origin
            self.record(batch.action(), batch.graph, call, batch.measurements(), call["begin_ns"],
                        call["result_processing"]["end_ns"], outputs={})
        self.read_decoder(pair)

    def read_decoder(self, pair):
        if len(pair.decode.bindings()) == 1:
            evidence = pair.qualify_decode(0, pair.decode.read(0).document())
            for batch in ObservedBatches(evidence).entries:
                call = batch.call
                self.record_decode(batch, batch.graph, batch.measurements(), call["begin_ns"],
                                   call.get("result_processing", call)["end_ns"])
            return

        physical = PhysicalCollectives(pair.decode)
        traces = []
        for rank, trace in enumerate(physical.ranks):
            evidence = pair.qualify_decode(rank, trace.evidence)
            traces.append(ObservedBatches(evidence).entries)
        for batches in aligned_batches(traces):
            parallel = RecordedParallelBatch(physical.ranks, batches, self.publisher)
            self.record_decode(batches[self.publisher], parallel.graph, parallel.measurements,
                               parallel.begin_ns, parallel.end_ns)

    def record_decode(self, owner, graph, work, begin, end):
        action = owner.action()
        self.starts[action_identity(action)] = begin - self.origin
        outputs = None
        if action["phase"] == PageTable.assembly_operation:
            outputs = {member["request"]: 1 for member in action["members"]}
            self.prebuilt_result_boundaries += "result_processing" in owner.call

            # Older physical recordings observe the page copy and first publication.
            end = max(end, *(self.boundaries[request, count] for request, count in outputs.items()))
        self.record(action, graph, owner.call, work, begin, end, outputs=outputs)

    def elapsed(self, start, current):
        return max(0, start - self.origin - current)

    def conditions(self, query):
        action = query["action"]
        phase = action["phase"]
        if phase in ("pd_reservation", "pd_handoff"):
            member, = action["members"]
            handoff = self.handoffs[member["request"]]
            if phase == "pd_reservation":
                return self.reservation(action["clock_ns"], handoff)
            return self.transport(action["clock_ns"], handoff, query["work"])

        timing = super().conditions(query)
        if "failure" in timing:
            return timing
        start = self.starts[action_identity(action)]
        return {**timing, "prepare_ns": max(0, start - action["clock_ns"])}

    def reservation(self, current, handoff):
        begins, ends = zip(*(port["reservation_ns"] for port in handoff["ports"]), strict=True)
        start, end = min(begins), max(ends)
        return {"work": [], "prepare_ns": self.elapsed(start, current), "model_return_ns": end - start,
                "result_finished": self.boundary(end - start), "publications": {}}

    def transport(self, current, handoff, work):
        samples = []
        observed = self.transfer_intervals(handoff)
        anchor = handoff["offer_ns"]
        for entry in work:
            key = entry["lane"], entry["name"]
            if entry["kind"] != "transport" or key not in observed:
                return {"failure": {"status": "action_mismatch", "reason": "PD transport work differs"}}
            begin, end = observed.pop(key)
            samples.append({"submitted_after_ns": begin - anchor, "dispatch_ns": 0, "duration_ns": end - begin})
        if observed:
            return {"failure": {"status": "action_mismatch", "reason": "PD transport work is missing"}}
        ready = max(port["notification_ns"] for port in handoff["ports"])
        return {"work": samples, "prepare_ns": self.elapsed(anchor, current), "model_return_ns": 0,
                "result_finished": self.boundary(ready - anchor), "publications": {}}

    @staticmethod
    def transfer_intervals(handoff):
        observed = {}
        for port in handoff["ports"]:
            observed[port["rank"], "kv_payload"] = port["kv_interval_ns"]
            observed[port["rank"], "metadata_payload"] = port["metadata_interval_ns"]
        return observed

    @staticmethod
    def endpoint(slot, pages, tokens):
        return {"slot": slot - 1, "pages": [page - 1 for page in pages], "tokens": tokens}

    @staticmethod
    def transport_ports(record):
        port_shapes = {}
        for port in record["ports"]:
            rank = port["rank"]
            port_shapes[f"kv_payload_target_{rank}"] = [2, port["layers"], len(port["pages"]), port["bytes_per_page"]]
            port_shapes[f"metadata_payload_target_{rank}"] = [port["metadata_bytes"]]
        return port_shapes

    def compare_handoff(self, expected, actual):
        tokens = expected["tokens"]
        source = self.endpoint(expected["source_slot"], expected["source_pages"], tokens)
        if any(actual["source"][key] != value for key, value in source.items()):
            return "prefill_write"
        for port in expected["ports"]:
            target = self.endpoint(port["slot"], port["pages"], tokens)
            if any(actual["target"][key] != value for key, value in target.items()):
                return "decode_reservation"
        outputs = actual["graph"]["calls"][0]["outputs"]
        shapes = {port["name"]: port["shape"] for port in outputs if port["dtype"] == "bytes"}
        if shapes != self.transport_ports(expected):
            return "transfer_ports"

        if not actual["reserved_ns"] <= actual["begin_ns"] <= actual["ready_ns"] <= min(
            actual["released_ns"], actual["committed_ns"]
        ):
            return "transfer_readiness"
        physical = self.transfer_intervals(expected)
        shift = actual["begin_ns"] - expected["offer_ns"]
        for work, actual_interval in zip(actual["graph"]["work"], actual["intervals"], strict=True):
            observed = physical[work["lane"], work["name"]]
            if actual_interval != [instant + shift for instant in observed]:
                return "transfer_intervals"
        return None

    def compare(self, result):
        # The two native schedulers have separate action orders and share only the handoff.
        actions = sorted(result["actions"], key=lambda action: action["phase"] != "prefill")
        report = super().compare({**result, "actions": actions})
        if report["status"] != "matched":
            return report
        if result["handoffs"].keys() != self.handoffs.keys():
            return {**report, "status": "structure_mismatch", "field": "handoff_members"}

        admissions = [row for row in actions if row["phase"] == PageTable.assembly_operation]
        for action in admissions:
            requests = [member["request"] for member in action["members"]]
            if action["handoff_inputs"] != requests or any(
                result["handoffs"][request]["committed_ns"] > action["began_ns"] for request in requests
            ):
                return {**report, "status": "structure_mismatch", "field": "admission_inputs"}
        ready_offsets = []
        for request, expected in self.handoffs.items():
            actual = result["handoffs"][request]
            if field := self.compare_handoff(expected, actual):
                return {**report, "status": "structure_mismatch", "request": request, "field": field}
            ready = max(port["notification_ns"] for port in expected["ports"]) - self.origin
            ready_offsets.append(abs(actual["ready_ns"] - ready))

        # Report the time evidence separately from the structural verdict.
        return {**report, "matched_handoffs": len(self.handoffs), "max_handoff_notice_offset_ns": max(ready_offsets),
                "admission_sizes": [len(row["members"]) for row in admissions],
                "measured_transfer_work": sum(len(row["intervals"]) for row in result["handoffs"].values()),
                "measured_prebuilt_result_boundaries": self.prebuilt_result_boundaries}
