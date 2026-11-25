"""Separate CPU transfer identity from its physical wire features."""

from collections import defaultdict

from prologue.contracts.cost import cost_features


class CpuTransfer:
    def __init__(self, world) -> None:
        self.world = world

    @staticmethod
    def identity(description):

        # Private object serialization can change byte counts without changing the native control operation.
        return {name: description[name] for name in ("group", "source", "target", "tag", "receiving", "dtype")}

    def describe(self, tensor, group, tag, endpoint, group_endpoint, receiving):
        import torch.distributed as distributed

        group = distributed.group.WORLD if group is None else group
        if group_endpoint is not None:
            endpoint = distributed.get_global_rank(group, group_endpoint)
        if endpoint is None:
            raise ValueError("Native metadata receive requires an explicit sender")
        local = distributed.get_rank()
        source, target = (endpoint, local) if receiving else (local, endpoint)
        return {
            "group": group.group_name, "source": self.world[source], "target": self.world[target],
            "tag": tag, "receiving": receiving, "dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "elements": tensor.numel(), "element_bytes": tensor.element_size(),
        }


class CpuProtocolWork:
    """Use the same native message geometry for demands and measurements."""

    def __init__(self, records):
        self.records = records

    @staticmethod
    def service(descriptor, execution, operation):
        route = execution["work"]["context"]["cpu_transfer"]
        identity = {name: descriptor[name] for name in ("source", "target", "dtype")}
        identity.update(operation=operation, channel=route["group"])
        axes = {f"parameter.{name}": descriptor[name] for name in ("elements", "element_bytes")}
        return identity, cost_features(axes)

    @staticmethod
    def readiness_operation(descriptor):
        return "cpu_receive_ready" if descriptor["receiving"] else "cpu_send_ready"

    @staticmethod
    def post_operations(operation):
        return operation + "_before_post", operation + "_after_post"

    @staticmethod
    def transport(post):
        transport = post.get("transport")
        if transport is None:
            return None
        stamp, completion = transport["post_raw_ns"], transport["completion_raw_ns"]
        interval = post["interval"]
        if (any(type(transport[name]) is not int or transport[name] < 0 for name in ("slot", "peer", "post_raw_ns"))
                or not interval["begin_raw_ns"] <= stamp <= interval["end_raw_ns"]):
            raise ValueError("Gloo transport post lies outside its native point-to-point call")
        if completion is not None:
            begin, end = completion
            thread = transport["completion_thread"]
            if (type(begin) is not int or type(end) is not int or type(thread) is not int or thread <= 0
                    or not stamp <= begin <= end
                    or post["consumed"] is not None and begin > post["consumed"]["end_raw_ns"]):
                raise ValueError("Gloo completion contradicts its native post or successful consumption")
        return transport

    @staticmethod
    def ready_window(transport, last_post, consumed):
        if (completion := transport["completion_raw_ns"]) is None:
            return None
        begin, end = completion
        begin = max(begin, last_post)
        if consumed is not None:
            end = min(end, consumed)
        return (begin, end) if begin <= end else None

    @classmethod
    def local(cls, record):
        for post in record["posts"]:
            descriptor = post["descriptor"]
            interval = post["interval"]
            execution = interval["execution"]
            if execution["in_flow"]:
                transport = cls.transport(post)
                before, after = cls.post_operations(execution["operation"])
                begin, end = interval["begin_raw_ns"], interval["end_raw_ns"]
                stamp = transport["post_raw_ns"] if transport is not None else None
                for operation, bounds in ((before, (begin, stamp)), (after, (stamp, end))):
                    identity, features = cls.service(descriptor, execution, operation)
                    yield identity, features, interval["thread"], bounds if stamp is not None else None

            consumed = post["consumed"]
            if post["operation"] in {"isend", "irecv"} and consumed is not None and consumed["execution"]["in_flow"]:
                execution = consumed["execution"]
                identity, features = cls.service(descriptor, execution, execution["operation"])
                yield identity, features, consumed["thread"], (consumed["begin_raw_ns"], consumed["end_raw_ns"])

    def transfers(self):
        messages = defaultdict(dict)
        for rank, record in self.records.items():
            if record.get("transport_clock", "monotonic_raw_ns") != "monotonic_raw_ns":
                raise ValueError("CPU transport requires a shared RAW clock")
            for post in record["posts"]:
                roles = messages[tuple(post["key"])]
                if rank in roles:
                    raise ValueError("CPU transport repeats an observed endpoint")
                roles[rank] = post
        for peers in messages.values():
            if len(peers) != 2:
                continue
            executions = (post["interval"]["execution"] for post in peers.values())
            active = [call for call in executions if call["in_flow"]]
            if not active:
                continue
            geometry = {tuple(post["descriptor"][name] for name in ("dtype", "elements", "element_bytes"))
                        for post in peers.values()}
            if len(geometry) != 1:
                raise ValueError("native CPU message endpoints disagree on transfer geometry")
            measured = {rank: self.transport(post) for rank, post in peers.items()}
            last_post = None
            if None not in measured.values():
                if len({row["slot"] for row in measured.values()}) != 1:
                    raise ValueError("CPU transport pairs different native slots")
                last_post = max(row["post_raw_ns"] for row in measured.values())
            for rank, post in peers.items():
                descriptor = post["descriptor"]
                execution = post["interval"]["execution"]
                identity, features = self.service(descriptor, execution if execution["in_flow"] else active[0],
                                                  self.readiness_operation(descriptor))
                consumed = post["consumed"]
                duration = None
                if last_post is not None:
                    end_ns = None if consumed is None else consumed["end_raw_ns"]
                    bound = self.ready_window(measured[rank], last_post, end_ns)
                    if bound is not None:
                        duration = bound[0] - last_post
                yield rank, identity, features, duration
