"""Bind observed KV writes, transport, and Decode consumption by request and peer."""

import json
from collections import defaultdict

from prologue.integrations.frameworks.sglang.page_table import PageTable
from prologue.integrations.platforms.ascend.evidence import AscendRecording


class PDWorker:
    def __init__(self, worker):
        self.document = worker
        self.events = defaultdict(list)
        for row in worker["pd_events"]:
            if "request" in row:
                self.events[row["request"], row["kind"], row.get("peer")].append(row)

        # Build the request index once. Later output iterations do not witness the first KV read.
        self.batches = defaultdict(list) if "calls" in worker else None
        for call in worker.get("calls", ()):
            if call["operation"] != "batch":
                continue
            for index, member in enumerate(call["members"]):
                if member["output_tokens"] <= 1:
                    self.batches[member["request"], member["output_tokens"]].append((call, index))

    def event(self, request, kind, peer=None):
        row, = self.events[request, kind, peer]
        return row

    def member_batch(self, request, output):
        if self.batches is None:
            return None
        match, = self.batches[request, output]
        return match


class PDWindow:
    def __init__(self, prefill, decoders, requests, page_size, publisher):
        self.prefill = PDWorker(prefill)
        self.decoders = [PDWorker(worker) for worker in decoders]
        self.requests, self.page_size = requests, page_size
        self.publisher = publisher

    @staticmethod
    def admits(worker):
        batches = []
        for event in worker["pd_events"]:
            if event["kind"] != "admit":
                continue
            end = 0
            for member in event["members"]:
                copy = member["copy"]
                if (copy["slot"] != member["slot"] or copy["source"] != [0, member["tokens"]]
                        or copy["target"] != [end, end + member["tokens"]]):
                    raise ValueError("Decode page-table assembly disagrees with the native members")
                end += member["tokens"]
            if not event["writes_output"] or end != event["extent"]:
                raise ValueError("Decode copies do not describe the produced page table")
            batches.append(event)
        return batches

    def request(self, request, admissions):
        name, tokens = request["id"], request["prompt_tokens"]
        offer = self.prefill.event(name, "offer")
        page_row = self.prefill.event(name, "source_pages")
        pages = page_row["pages"]
        page_count = (tokens + self.page_size - 1) // self.page_size
        if (not offer["last"] or offer["tokens"] != [0, tokens] or len(pages) != page_count
                or page_row["positions"] != [0, page_count]):
            raise ValueError(f"incomplete Prefill offer for {name}")

        captured = self.prefill.member_batch(name, 0)
        if captured is not None:
            call, position = captured
            cache = call["cache"]
            locations = cache["rows"][position]
            if (call["members"][position]["slot"] != offer["slot"] or len(locations) != tokens
                    or self.pages(locations) != pages or cache["written"] != sum(cache["rows"], [])
                    or call["end_ns"] > page_row["at_ns"]):
                raise ValueError(f"Prefill output does not supply the actual source pages for {name}")

        ports = []
        reads = request["output_tokens"] > 1
        for rank, decoder in enumerate(self.decoders):
            ports.append(self.receiver(name, tokens, rank, decoder, admissions[rank], offer, page_row, reads))
        return {"request": name, "tokens": tokens, "source_slot": offer["slot"], "source_pages": pages, "ports": ports,
                "offer_ns": offer["begin_ns"], "source_metadata_slot": offer["metadata_slot"]}

    def receiver(self, name, tokens, rank, decoder, admissions, offer, source_pages, reads):
        kv = self.prefill.event(name, "kv", rank)
        metadata = self.prefill.event(name, "metadata", rank)
        notice = self.prefill.event(name, "notify", rank)
        reserved = decoder.event(name, "reserve")
        ready = decoder.event(name, "ready")
        committed = decoder.event(name, "commit")
        (admitted, member), = admissions[name]

        if ({row["room"] for row in (kv, metadata, notice, reserved, ready, committed, offer, source_pages)}
                != {offer["room"]} or kv["status"] != 0 or metadata["status"] != 0
                or kv["source_pages"] != source_pages["pages"] or kv["target_pages"] != reserved["pages"]
                or metadata["source_slot"] != offer["metadata_slot"]
                or metadata["target_slot"] != committed["metadata_slot"]
                or committed["token"] != offer["token"] or committed["token"] is None
                or member["slot"] != committed["slot"] or committed["slot"] != reserved["slot"]
                or member["tokens"] != tokens or committed["tokens"] != tokens or reserved["tokens"] != tokens):
            raise ValueError(f"PD ports do not join for {name} to Decode rank {rank}")

        boundaries = [max(source_pages["at_ns"], reserved["at_ns"]), kv["begin_ns"], kv["at_ns"],
                      metadata["begin_ns"], metadata["at_ns"], notice["begin_ns"], ready["begin_ns"],
                      committed["begin_ns"], committed["at_ns"], admitted["begin_ns"], admitted["at_ns"]]
        if boundaries != sorted(boundaries):
            raise ValueError(f"native PD boundaries are out of order for {name} to rank {rank}")

        read = decoder.member_batch(name, 1) if reads else None
        if read is not None:
            call, index = read
            cache = call["cache"]
            expected = reserved["pages"]
            if (call["begin_ns"] < admitted["at_ns"] or call["members"][index]["slot"] != reserved["slot"]
                    or self.pages(cache["rows"][index][:tokens]) != expected
                    or cache["blocks"][index][:len(expected)] != expected):
                raise ValueError(f"the first Decode attention reads a different KV table for {name} to rank {rank}")

        return {"rank": rank, "slot": reserved["slot"], "pages": reserved["pages"], "layers": kv["layers"],
                "bytes_per_page": kv["bytes_per_page"], "metadata_bytes": metadata["bytes"], "sliced": kv["sliced"],
                "reservation_ns": [reserved["begin_ns"], reserved["at_ns"]], "metadata_slot": metadata["target_slot"],
                "kv_interval_ns": [kv["begin_ns"], kv["at_ns"]],
                "metadata_interval_ns": [metadata["begin_ns"], metadata["at_ns"]],
                "notification_ns": ready["begin_ns"], "commit_ns": committed["at_ns"],
                "admission_ns": admitted["at_ns"], "first_read_ns": read[0]["begin_ns"] if read else None}

    def pages(self, locations):
        return [loc // self.page_size for loc in locations[::self.page_size]]

    def qualify(self):
        admissions = [self.admits(worker.document) for worker in self.decoders]
        members = [[member["request"] for member in row["members"]] for row in admissions[0]]
        if any([[member["request"] for member in row["members"]] for row in other] != members
               for other in admissions[1:]):
            raise ValueError("Decode ranks did not agree on the observed admissions")

        controls = [[row for row in worker.document["pd_events"] if row["kind"] in ("paused", "resumed")]
                    for worker in self.decoders]
        if any([row["kind"] for row in events] != [row["kind"] for row in controls[self.publisher]]
               for events in controls):
            raise ValueError("Decode ranks did not observe the same admission commands")
        admitted_members = []
        for batches in admissions:
            indexed = defaultdict(list)
            for batch in batches:
                for member in batch["members"]:
                    indexed[member["request"]].append((batch, member))
            admitted_members.append(indexed)

        handoffs = [self.request(request, admitted_members) for request in self.requests]
        decoder = self.decoders[self.publisher].document
        publications = decoder["publications"]
        for handoff, request in zip(handoffs, self.requests, strict=True):
            outputs = publications[request["id"]]
            ready = max(port["notification_ns"] for port in handoff["ports"])
            admitted = handoff["ports"][self.publisher]["admission_ns"]
            if ([row["count"] for row in outputs] != list(range(1, request["output_tokens"] + 1))
                    or outputs[0]["raw_ns"] < max(ready, admitted)):
                raise ValueError(f"Decode publications disagree with the handoff for {request['id']}")
            handoff["publications"] = outputs
        return {"admissions": members, "handoffs": handoffs, "controls": controls[self.publisher],
                "ingress": [self.prefill.document.get("ingress"), decoder.get("ingress")]}


class PDRecording:
    def __init__(self, directory, manifest):
        self.directory, self.manifest = directory, manifest
        self.domain = {key: manifest[key] for key in ("framework_revision", "model", "configuration")}
        self.prefill = AscendRecording(directory / "prefill")
        self.decode = AscendRecording(directory / "decode")
        if len(self.prefill.bindings()) != 1:
            raise ValueError("PD handoff qualification requires one Prefill rank")
        self.model_config = self.prefill.manifest["model_config"]
        if self.model_config != self.decode.manifest["model_config"]:
            raise ValueError("PD roles loaded different model configurations")
        shared = ("framework_revision", "model", "workload", "placement")
        for role in (self.prefill, self.decode):
            if any(role.manifest[key] != manifest[key] for key in shared):
                raise ValueError("PD role manifests disagree with their capture")

    @staticmethod
    def worker(recording, window, rank):
        return json.loads((recording.directory / window["directory"] / f"rank-{rank}" / "worker.json")
                          .read_text(encoding="utf-8"))

    def qualify(self):
        windows = []
        width = len(self.decode.bindings())
        publisher = self.decode.publication_rank()
        for prefill, decode in zip(self.prefill.windows, self.decode.windows, strict=True):
            time = prefill["delivery"]["origin_raw_ns"]
            if decode["delivery"]["origin_raw_ns"] != time:
                raise ValueError("PD roles must use a common workload clock")
            decoders = [self.worker(self.decode, decode, rank) for rank in range(width)]
            evidence = PDWindow(self.worker(self.prefill, prefill, 0), decoders, self.manifest["workload"]["requests"],
                                self.domain["configuration"]["page_size"], publisher)
            windows.append({"origin_raw_ns": time, **evidence.qualify()})
        return windows

    def qualify_decode(self, rank, document):
        selected, = (window for window in self.decode.windows if str(rank) in window["traces"])
        worker = self.worker(self.decode, selected, rank)
        admits = PDWindow.admits(worker)
        calls = [row for row in document["calls"] if row["operation"] == PageTable.assembly_operation]
        for call, admitted in zip(calls, admits, strict=True):
            if not admitted["begin_ns"] <= call["begin_ns"] < call["end_ns"] <= admitted["at_ns"]:
                raise ValueError("the native page-table call is outside its admission step")
            call["parameters"]["phase"] = PageTable.assembly_operation
            call["members"] = [
                {"request": item["request"], "output_tokens": 1, "slot": item["slot"],
                 "kv_committed_tokens": item["tokens"], "kv_reserved_tokens": item["tokens"]}
                for item in admitted["members"]
            ]
        return document

    @staticmethod
    def transport_document(windows):
        # Native endpoint geometry determines shared transfer definitions. Physical times stay in the reference.
        calls, work = [], []
        for handoff in windows[0]["handoffs"]:
            root = len(calls)
            inputs, outputs = [], []
            query = {"phase": "pd_handoff", "tokens": handoff["tokens"]}
            calls.append({"id": root, "parent": None, "operation": "pd_handoff", "implementation": "pd_handoff",
                          "query": query, "parameters": query, "inputs": inputs, "outputs": outputs})
            for port in handoff["ports"]:
                rank = port["rank"]
                layout = {"pages": len(port["pages"]), "layers": port["layers"],
                          "bytes_per_page": port["bytes_per_page"], "sliced": port["sliced"]}
                transfers = (
                    ("kv_payload", [2, layout["layers"], layout["pages"], layout["bytes_per_page"]], layout),
                    ("metadata_payload", [port["metadata_bytes"]], {}),
                )
                for operation, shape, parameters in transfers:
                    call = len(calls)
                    operand = {"name": "source", "tensor": call * 2, "dtype": "bytes", "shape": shape}
                    received = {"name": "target", "tensor": call * 2 + 1, "dtype": "bytes", "shape": shape}
                    inputs.append({**operand, "name": f"{operation}_source_{rank}"})
                    outputs.append({**received, "name": f"{operation}_target_{rank}"})
                    calls.append({"id": call, "parent": root, "operation": operation, "implementation": "ascend_sdma",
                                  "parameters": parameters, "inputs": [operand], "outputs": [received]})
                    work.append({"owner": call, "slot": 0, "lane": rank, "kind": "transport", "name": operation})
        return {"calls": calls, "work": work}
