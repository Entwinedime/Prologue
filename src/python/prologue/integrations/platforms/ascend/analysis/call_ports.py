"""Translate native call boundaries, tensor ports and signals before lowering dependencies."""

from __future__ import annotations


class ObservedPorts:
    """Project semantic ports onto the independently located invocation and completion boundaries."""

    def __init__(self, graph, queues):
        self.graph, self.queues = graph, queues
        self.signal_inputs = {}

    def call_ports(self) -> None:
        observed = []
        for scope, raw in enumerate(self.graph.raw["scopes"]):
            if scope == 0:
                continue
            host = self.graph.host[scope]
            ports = self.graph.scopes[scope]["ports"]
            boundaries = (
                ("call.enter", host, "begin"),
                ("call.return", host, "end"),
                ("caller.return", self.graph.callers.get(scope), "end"),
                ("service.complete", self.graph.services.get(scope), "end"),
                ("device.enter", self.graph.device.get(scope), "begin"),
                ("device.complete", self.graph.device.get(scope), "end"),
            )

            for name, span, side in boundaries:
                if span is not None:
                    direction = "input" if side == "begin" else "output"
                    ports.append(
                        {
                            "name": name,
                            "direction": direction,
                            "use": "control",
                            "relation": "control",
                            "at": self.graph.endpoint(span, side),
                        }
                    )

            if "communication" in raw:
                span = self.graph.device[scope] if "device_entry" in raw else host
                completion = (
                    self.graph.work_end(scope) if raw.get("asynchronous", False) else self.graph.endpoint(span, "end")
                )
                ports.append(
                    {
                        "name": "communication.complete",
                        "direction": "output",
                        "use": "control",
                        "relation": "communication",
                        "at": completion,
                        "rendezvous": self.graph.endpoint(span, "begin"),
                    }
                )

            for port in raw["ports"]:
                if port["use"] == "signal":
                    self.signal_port(scope, raw, port, ports, host)
                    continue

                side, converted = self.tensor_port(scope, raw, port, host)
                ports.append(converted)
                if port["tensor"] is not None and port["use"] not in {"state", "metadata"}:
                    observed.append((getattr(self.graph.trace.scopes[scope], side), scope, converted))

        self.data_dependencies(observed)

    def signal_port(self, scope, raw, port, ports, host) -> None:
        if "boundary" in port:
            boundary = port["boundary"]
            span = {"call": self.graph.host, "device": self.graph.device, "service": self.graph.services}[
                boundary["interval"]
            ][boundary["scope"]]
            ports.append(
                {
                    "name": port["name"],
                    "direction": port["direction"],
                    "use": "signal",
                    "relation": "signal",
                    "signal": f"scope/{boundary['scope']}/{boundary['interval']}/{boundary['side']}",
                    "at": self.graph.endpoint(span, boundary["side"]),
                }
            )
            return

        if (producer := port.get("producer")) in self.graph.peer_services:
            completed = self.graph.work_end(producer)
            at = completed
            if port["direction"] == "input":
                entered, returned = (
                    self.graph.trace.device_points[raw[f"device_{side}"]] for side in ("entry", "return")
                )

                # The entry marker is submitted before the measured host call. Only completion joins it.
                self.graph.staged_boundary(scope, entered)
                at = self.graph.staged_boundary(scope, returned)
                self.graph.edge(self.graph.endpoint(host, "begin"), at, "control")
                self.graph.edge(completed, at, "signal")

            ports.append(
                {
                    "name": port["name"],
                    "direction": port["direction"],
                    "use": "signal",
                    "relation": "signal",
                    "signal": f"work/{producer}",
                    "at": at,
                }
            )
            return

        task = (
            self.graph.work_completions[port["producer"]]
            if "producer" in port
            else self.graph.streams.task(self.graph.trace.device_points[port["point"]].task)
        )
        connection = task.source.record["args"]["connection_id"]

        if port["direction"] == "input":
            self.signal_inputs[scope, connection] = port["name"]
        else:
            ports.append(
                {
                    "name": port["name"],
                    "direction": "output",
                    "use": "signal",
                    "relation": "signal",
                    "signal": f"event/{connection}",
                    "at": self.graph.task_anchor(task),
                }
            )
            self.queues.submission(task)
        return

    def tensor_port(self, scope, raw, port, host):
        side = "end" if port["direction"] == "output" or port.get("at_return", False) else "begin"
        span = self.graph.device[scope] if port["use"] in {"device", "state"} else host
        tensor = port["tensor"]
        use = port["use"]

        if use in {"host", "device"}:
            use = "data" if tensor is not None else "control"
        elif use == "state":

            # Persistent buffer ports describe layout; state_accesses owns the actual read/write intervals.
            use = "metadata"

        relation = "data" if use == "data" else "control"
        converted = {
            "name": port["name"],
            "direction": port["direction"],
            "use": use,
            "relation": relation,
            "at": self.graph.endpoint(span, side),
        }

        if raw.get("asynchronous", False) and port["use"] == "device" and port["direction"] == "output":
            converted["at"] = self.graph.work_end(scope)
        if tensor is not None:
            converted["tensor"] = {
                name: tensor[name]
                for name in ("value", "storage", "version", "dtype", "shape", "writer", "allocation", "buffer")
            }
            converted["tensor"].update(
                (name, tensor[name]) for name in ("shape_rule", "packing") if name in tensor
            )
        return side, converted

    def data_dependencies(self, observed: list) -> None:

        # Only an already published version can feed another call; an enclosing writer's return is in the future.
        producers = {}
        for _, scope, port in sorted(observed, key=lambda item: item[0]):
            tensor = port["tensor"]
            key = tensor["storage"], tensor["version"]
            if port["direction"] == "output":
                producer, at = producers.setdefault(key, (scope, port["at"]))

                # A parent re-exports the child's ready value, not its own later completion. Siblings keep local ports.
                owner = producer
                while owner is not None and owner != scope:
                    owner = self.graph.scopes[owner]["parent"]
                if owner == scope:
                    port["at"] = at
            elif key in producers:
                self.graph.edge(producers[key][1], port["at"], "data")

    def state_accesses(self) -> None:
        for scope, raw in enumerate(self.graph.raw["scopes"]):
            for access in raw.get("state_accesses", []):
                spans = {"device": self.graph.device, "call": self.graph.host, "service": self.graph.services}[
                    access["interval"]
                ]
                field = {"kv": "kv_accesses", "slot": "slot_accesses"}[access["space"]]
                self.graph.scopes[scope][field].append(
                    {
                        **{name: value for name, value in access.items() if name != "interval"},
                        "start": self.graph.endpoint(spans[scope], "begin"),
                        "finish": self.graph.endpoint(spans[scope], "end"),
                    }
                )
