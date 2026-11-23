"""Bind observed pipeline cuts to native batch identities and tensor handoffs."""

from collections import defaultdict

from prologue.acquisition.observation.batches import action_identity


class WorkerPrograms:
    def __init__(self, document):
        self.calls = document["calls"]
        self.cuts = defaultdict(dict)
        self.dependencies = defaultdict(set)
        self.messages = defaultdict(dict)
        batches = {}
        consumers = {}
        producers = {}

        for call in self.calls:
            if call["operation"] == "batch":
                identity = action_identity({
                    "phase": call["parameters"]["phase"], "members": call["members"],
                })
                if identity in self.cuts:
                    raise ValueError(f"repeated native pipeline batch {identity}")
                batches[call["id"]] = identity
                self.cuts[identity]["launch"] = call["id"]
                consumers.update((port["tensor"], (identity, "launch")) for port in call["inputs"])
                producers.update((port["tensor"], (identity, "launch")) for port in call["outputs"])
            elif call["operation"] in ("pipeline_prepare", "pipeline_process"):
                identity = batches[call["parameters"]["batch"]]
                step = "prepare" if call["operation"] == "pipeline_prepare" else "process"
                self.cuts[identity][step] = call["id"]
                if step == "prepare":
                    consumers.update((port["tensor"], (identity, step)) for port in call["inputs"])

        self.signals(batches)
        self.transport(consumers, producers)

    def signals(self, batches):
        for wait in self.calls:
            if wait["operation"] != "signal_wait" or wait["parent"] is not None:
                continue

            record = self.calls[wait["parameters"]["producer"]]
            anchor = self.calls[record["parameters"]["producer"]]
            if anchor["operation"] == "pipeline_launch":
                identity = batches[anchor["batch"]]
                before, label = "launch", "launch"
            else:
                identity = batches[anchor["parameters"]["batch"]]
                before, label = "prepare", "result"

            recorded, waited = f"{label}_signal", f"{label}_wait"
            self.cuts[identity][recorded] = record["id"]
            self.cuts[identity][waited] = wait["id"]
            self.dependencies[identity].update(((before, recorded), (recorded, waited)))
            if wait["parameters"]["host"]:
                self.dependencies[identity].add((waited, "process"))

    def transport(self, consumers, producers):
        transfers = defaultdict(list)
        waits = {}
        receives, sends = [], []
        for call in self.calls:
            operation = call["operation"]
            if operation == "transfer_post":
                transfers[call["parent"]].append(call)
            elif operation == "transfer_wait" and call["completed"]:
                waits[call["parameters"]["post"]] = call
            elif operation == "pipeline_message":
                target = sends if call["parameters"]["direction"] == "send" else receives
                target.append(call)

        for message in receives:
            identity, consumer = self.endpoint(message["outputs"], consumers)
            label = self.message(identity, message, transfers, waits)
            self.dependencies[identity].add((label, consumer))
            producers.update((port["tensor"], (identity, label)) for port in message["outputs"])
        for message in sends:
            identity, producer = self.endpoint(message["inputs"], producers)
            label = self.message(identity, message, transfers, waits)
            ready = "launch_wait" if producer == "launch" else producer
            self.dependencies[identity].add((ready, label))

    @staticmethod
    def endpoint(ports, owners):
        binding, = {owners[port["tensor"]] for port in ports}
        return binding

    def message(self, identity, call, transfers, waits):
        parameters = call["parameters"]
        direction, kind = parameters["direction"], parameters["kind"]
        label = f"{direction}:{kind}"
        steps = self.cuts[identity]
        if label in steps:
            raise ValueError(f"repeated pipeline message {label} in {identity}")
        steps[label] = call["id"]

        outbound = direction == "send"
        wires = {}
        for post in transfers[call["id"]]:
            wait = waits[post["id"]]
            field = post["parameters"]["field"]
            port, = post["inputs"] if outbound else post["outputs"]
            wire = {key: value for key, value in port.items() if key not in ("tensor", "name")}
            wait_label = None
            if outbound:
                wait_label = f"wait:{kind}:{field}"
                steps[wait_label] = wait["id"]
                self.dependencies[identity].add((label, wait_label))
            wires[field] = {"layout": wire, "wait": wait_label}

        logical = {port["name"]: {key: value for key, value in port.items() if key not in ("tensor", "name")}
                   for port in (call["inputs"] if outbound else call["outputs"])}
        peer = parameters["target" if outbound else "source"]
        self.messages[identity][label] = (peer, logical, wires)
        return label

    def normalize(self):
        fields = {
            "pipeline_message": ("direction", "kind"),
            "transfer_wait": ("direction", "field"),
            "pipeline_prepare": (),
            "pipeline_process": (),
            "signal_record": (),
            "signal_wait": ("host",),
        }
        normalized = []
        for call in self.calls:
            operation = call["operation"]
            if operation not in fields:
                normalized.append(call)
                continue
            parameters = call["parameters"]
            if operation == "transfer_wait":
                parameters = self.calls[parameters["post"]]["parameters"]
            normalized.append({**call, "parameters": {key: parameters[key] for key in fields[operation]}})
        return normalized


class PipelinePrograms:
    def __init__(self, recording, workers, bindings):
        self.recording = recording
        self.layouts = recording.manifest["workers"]
        self.bindings = bindings
        self.observed = {worker["rank"]: WorkerPrograms(worker["document"]) for worker in workers}

    def programs(self):
        publisher = self.recording.publication_rank()
        first = self.observed[self.bindings[publisher]]
        for observed in self.observed.values():
            if observed.cuts.keys() != first.cuts.keys():
                raise ValueError("pipeline ranks observed different native microbatches")
        return [self.program(identity, publisher) for identity in first.cuts]

    def program(self, identity, publisher):
        scopes = defaultdict(list)
        dependencies = set()
        messages = {}
        for rank, representative in enumerate(self.bindings):
            role = self.layouts[str(rank)]
            stage = role["stage"]
            observed = self.observed[representative]
            for label, scope in observed.cuts[identity].items():
                scopes[stage, label].append({"rank": rank, "observed_rank": representative, "scope": scope})
            dependencies.update(((stage, source, rank), (stage, target, rank))
                                for source, target in observed.dependencies[identity])
            peers = dict(zip(self.layouts[str(representative)]["pp_members"], role["pp_members"], strict=True))
            for label, (peer, logical, wires) in observed.messages[identity].items():
                messages[rank, label] = (peers[peer], logical, wires)

        positions = {key: index for index, key in enumerate(sorted(scopes))}
        transfers = []
        paired_receives = set()
        for (rank, label), (peer, logical, wires) in sorted(messages.items()):
            if not label.startswith("send:"):
                continue
            kind = label.removeprefix("send:")
            target_label = f"receive:{kind}"
            source_stage, target_stage = (self.layouts[str(value)]["stage"] for value in (rank, peer))
            sender, receiver = (source_stage, label), (target_stage, target_label)
            reverse, output, received = messages[peer, target_label]
            if reverse != rank or logical != output or wires.keys() != received.keys():
                raise ValueError(f"pipeline ports disagree between ranks {rank} and {peer} for {identity}")
            paired_receives.add((peer, target_label))
            dependencies.add(((*sender, rank), (*receiver, peer)))
            for field, wire in wires.items():
                if wire["layout"] != received[field]["layout"]:
                    raise ValueError(f"pipeline wire {field} disagrees between ranks {rank} and {peer}")
                transfers.append({
                    "source": {"step": positions[sender], "rank": rank},
                    "target": {"step": positions[receiver], "rank": peer},
                    "sender_wait": positions[source_stage, wire["wait"]],
                    "field": field, "logical": logical[field], "wire": wire["layout"],
                })
        if paired_receives != {key for key in messages if key[1].startswith("receive:")}:
            raise ValueError(f"pipeline microbatch {identity} has unpaired receives")

        first = self.observed[self.bindings[publisher]]
        call = first.calls[first.cuts[identity]["launch"]]
        return {
            "phase": call["parameters"]["phase"], "members": call["members"],
            "steps": [{"stage": stage, "name": label, "queue": self.queue(label), "scopes": scopes[stage, label]}
                      for stage, label in positions],
            "dependencies": [
                [{"step": positions[stage, label], "rank": rank} for stage, label, rank in pair]
                for pair in sorted(dependencies)
            ],
            "transfers": transfers, "publication_rank": publisher,
        }

    @staticmethod
    def queue(label):
        if label == "launch":
            return "forward"
        if label == "prepare":
            return "result_copy"
        return "schedule"
