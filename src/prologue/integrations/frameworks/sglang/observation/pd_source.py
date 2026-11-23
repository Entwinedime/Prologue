"""Project native Decode page-table facts onto the shared compiler input."""

import json
from decimal import Decimal

from prologue.integrations.platforms.ascend.evidence import AscendEvidence


class PDSource:
    copy_purpose = "pd_page_copy"
    assembly_purpose = "pd_page_table_assembly"

    def __init__(self, directory):
        self.directory = directory
        self.domain = json.loads((directory / "pd-domain.json").read_text(encoding="utf-8"))
        if self.domain["configuration"]["disaggregation_mode"] != "decode":
            raise ValueError("page-table assembly requires native Decode evidence")

        self.transfers = {}
        for record in self.records("pd-transfers.jsonl"):
            request = record["request"]
            if request in self.transfers:
                raise ValueError(f"repeated KV transfer for {request}")
            self.transfers[request] = record
        self.batches = list(self.records("pd-preparation.jsonl"))

    def records(self, name):
        with (self.directory / name).open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)

    def qualify(self, batch):
        admitted = batch["admitted"]
        waiting = batch["waiting_before"]
        remaining = batch["waiting_after"]
        occupied = batch["active_before"] + len(admitted)
        if (not admitted or admitted != [member["request"] for member in batch["members"]]
                or sorted(waiting) != sorted(admitted + remaining) or len(set(waiting)) != len(waiting)
                or occupied > batch["limit"] or (remaining and occupied != batch["limit"])):
            raise ValueError("native admission does not match the prepared members and remaining queue")

        transfers = [self.transfers[request] for request in waiting]
        if (batch["end_ns"] <= batch["begin_ns"] or any(row["at_ns"] > batch["begin_ns"] for row in transfers)
                or len({row["slot"] for row in transfers}) != len(waiting) or not batch["writes_output"]):
            raise ValueError("page-table inputs must be resident before assembly with distinct slots")

        cursor = 0
        for member, copy, computed, sequence in zip(batch["members"], batch["copies"], batch["computed"],
                                                   batch["sequence_lengths"], strict=True):
            transfer = self.transfers[member["request"]]
            begin, end = copy["source"]
            length = member["length"]
            if (transfer["slot"] != member["slot"] or copy["slot"] != member["slot"] or begin != member["prefix"]
                    or not 0 <= begin < end or end - begin != length or end != transfer["committed"]
                    or end != member["committed"] or end != computed or end != sequence
                    or copy["target"] != [cursor, cursor + length]):
                raise ValueError(f"page-table extent or progress differs for {member['request']}")
            cursor += length
        if cursor != batch["output_size"]:
            raise ValueError("page-table copies do not cover the native output")

    @classmethod
    def geometry(cls, copy):
        begin, end = copy["source"]
        fields = ("source_dtype", "target_dtype", "source_device", "target_device")
        return {"phase": cls.copy_purpose, "length": end - begin, **{field: copy[field] for field in fields}}

    @staticmethod
    def port(name, copy, side):
        return {"name": name, "shape": [copy["length"]], "dtype": copy[f"{side}_dtype"],
                "device": copy[f"{side}_device"]}

    def copy_calls(self, batches):
        calls = []
        for batch in batches:
            for copy in batch["copies"]:
                identity = copy["marker"]
                if identity != len(calls):
                    raise ValueError("profiled copy markers must identify each observed call once")
                query = self.geometry(copy)
                calls.append({
                    "id": identity, "parent": None, "query": query, "operation": self.copy_purpose,
                    "implementation": "sglang.prebuilt.page_table_assignment", "parameters": query,
                    "inputs": [{**self.port("source", query, "source"), "tensor": 0}],
                    "outputs": [{**self.port("destination", query, "target"), "tensor": 1}],
                })
        return calls

    def assembly_calls(self, batches):
        calls = []
        for batch in batches:
            root = len(calls)
            copies = batch["copies"]
            output = {**self.port("out_cache_loc", self.geometry(copies[0]), "target"),
                      "shape": [batch["output_size"]], "views": []}
            query = {"phase": self.assembly_purpose, "members": []}
            inputs = []
            calls.append({
                "id": root, "parent": None, "query": query, "operation": self.assembly_purpose,
                "implementation": "sglang.prebuilt.page_table_assembly",
                "parameters": {"sequence_lengths": batch["sequence_lengths"]}, "inputs": inputs, "outputs": [output],
            })

            # Copy operands carry local identities. The parent output projects their actual destination slices.
            for ordinal, copy in enumerate(copies):
                geometry = self.geometry(copy)
                source = {**self.port("source", geometry, "source"), "tensor": 2 * ordinal}
                target = {**self.port("destination", geometry, "target"), "tensor": 2 * ordinal + 1}
                inputs.append({**source, "name": f"member_{ordinal}_pages"})
                output["views"].append({"tensor": target["tensor"], "interval": copy["target"]})
                query["members"].append({"begin": copy["source"][0], "copy": geometry})
                calls.append({"id": len(calls), "parent": root, "template_query": geometry,
                              "inputs": [source], "outputs": [target]})
        return calls

    def document(self, scope):
        detail = scope == "copy"
        batches = []
        for batch in self.batches:
            marked = {copy["marker"] is not None for copy in batch["copies"]}
            if marked == {detail}:
                self.qualify(batch)
                batches.append(batch)
        if not batches:
            raise ValueError(f"no observed PD {scope} batches")

        purpose, mode = (self.copy_purpose, "detailed") if detail else (self.assembly_purpose, "boundary")
        source = {"purpose": purpose, "mode": mode, "origin_raw_ns": batches[0]["begin_ns"]}
        if detail:
            calls = self.copy_calls(batches)
            trace, = (self.directory / "profile").rglob("trace_view.json")
            rows = json.loads(trace.read_text(encoding="utf-8"), parse_float=Decimal)
            document = AscendEvidence(self.domain, {"calls": calls, "publications": {}}, rows).document()
            if {work["owner"] for work in document["work"]} != {call["id"] for call in calls}:
                raise ValueError("a sampled copy has no attributable device work")
        else:
            calls = self.assembly_calls(batches)
            document = {"domain": self.domain, "calls": calls, "work": [],
                        "evidence": {"observed_batches": len(batches)}}
        return document, source
