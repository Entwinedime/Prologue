"""Observe native device collectives, keeping their participant contract outside graph algorithms."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import Any

from prologue.acquisition.observation.parameters import ScopeParameters
from prologue.integrations.frameworks.sglang.pipeline import PipelineABI
from prologue.integrations.frameworks.sglang.observation.module_scopes import tensor_ports
from prologue.integrations.runtimes.pytorch.layouts import GatherLayout


@dataclass
class CollectiveOperand:
    coordinator: Any
    tensor: Any
    gather_dim: int | None = None

    def implementation(self):
        communicator = self.coordinator.npu_communicator
        if communicator is None or communicator.disabled or not self.tensor.is_contiguous():
            return None

        operation = {"dimension": self.gather_dim} if self.gather_dim is not None else {"reduction": "sum"}
        return {
            "backend": f"{type(communicator).__module__}.{type(communicator).__qualname__}",
            "input_layout": "contiguous", **operation,
        }


class DeviceCollectiveCuts:
    """Keep structural and CPU-only observations on the same native call boundaries."""

    def __init__(self, scopes) -> None:
        self.scopes = scopes
        self.enclosing = ContextVar("native_communication", default=False)

    def invoke(self, signature, original, group, *args, **kwargs):
        if not self.scopes.enabled or group.world_size == 1 or self.enclosing.get():
            return original(group, *args, **kwargs)
        arguments = signature.bind(group, *args, **kwargs)
        arguments.apply_defaults()
        tensors = list(tensor_ports(arguments.arguments, "arguments"))
        if not any(tensor.device.type == "npu" for _, tensor in tensors):
            return original(group, *args, **kwargs)

        root_rank = group.ranks[arguments.arguments["src"]] if original.__name__ == "broadcast" else None
        operand = None
        role = "communication"
        if original.__name__ == "all_gather" and arguments.arguments["output_tensor_list"] is None:
            tensor = arguments.arguments["input_"]
            operand = CollectiveOperand(group, tensor, arguments.arguments["dim"] % tensor.ndim)
            if self.scopes.collective_at_root:
                role = "gather"
        elif original.__name__ == "all_reduce" and not self.scopes.collective_at_root:
            operand = CollectiveOperand(group, arguments.arguments["input_"])
        with self.outer():
            return self.scopes.observe_exchange(
                original, role, group.device_group, root_rank, operand, partial(original, group, *args, **kwargs),
            )

    def distributed(self, signature, original, *args, **kwargs):
        if not self.scopes.enabled or self.enclosing.get():
            return original(*args, **kwargs)
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        tensors = tensor_ports(arguments.arguments, "arguments")
        if not any(tensor.device.type == "npu" for _, tensor in tensors):
            return original(*args, **kwargs)
        if arguments.arguments["async_op"]:
            raise ValueError("asynchronous native collectives need an explicit work-completion boundary")

        import torch.distributed as distributed

        group = arguments.arguments["group"]
        if group is None:
            group = distributed.group.WORLD
        if distributed.get_world_size(group) == 1:
            return original(*args, **kwargs)
        root_rank = arguments.arguments.get("src")
        if arguments.arguments.get("group_src") is not None:
            root_rank = distributed.get_global_rank(group, arguments.arguments["group_src"])
        with self.outer():
            return self.scopes.observe_exchange(
                original, "communication", group, root_rank, None, partial(original, *args, **kwargs),
            )

    @contextmanager
    def outer(self):

        # Public group methods may call each other; record one scope for each native exchange.
        token = self.enclosing.set(True)
        try:
            yield
        finally:
            self.enclosing.reset(token)

    def install(self, expert_exchange=None) -> None:
        import torch.distributed as distributed
        from sglang.srt.distributed.parallel_state import GroupCoordinator
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        for name in ("all_reduce", "quant_all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter", "reduce_scatter_tensor", "all_to_all_single", "broadcast"):
            target = f"sglang.srt.distributed.parallel_state.GroupCoordinator.{name}"
            HookRegistry.register(target, partial(self.invoke, inspect.signature(getattr(GroupCoordinator, name))), HookType.AROUND)
        for name in ("all_gather_into_tensor", "all_gather", "all_reduce", "broadcast"):
            target = f"torch.distributed.{name}"
            HookRegistry.register(target, partial(self.distributed, inspect.signature(getattr(distributed, name))), HookType.AROUND)
        if expert_exchange is not None:
            for name in ("dispatch", "combine"):
                target = f"sglang.srt.layers.moe.token_dispatcher.deepep.DeepEPDispatcher.{name}"
                HookRegistry.register(target, expert_exchange, HookType.AROUND)


class NativeCommunications:
    def __init__(self, observation: Any) -> None:
        self.observation = observation
        self.cuts = DeviceCollectiveCuts(self)
        self.orders = {}
        self.action = None
        self.pending_reductions = []

    @property
    def enabled(self):
        return self.observation.active and self.observation.selected

    @property
    def collective_at_root(self):
        return self.observation.capture.current.get() == 0

    @contextmanager
    def local_exchange(self, operation, source, outputs):
        owner = self.observation
        scope = owner.capture.current.get()
        if not self.enabled or "communication" not in owner.capture.scopes[scope]:
            yield
            return

        # These markers locate the target protocol service. Elapsed time between them belongs to acquisition.
        runtime = owner.capture.runtime
        begin = runtime.point(scope, "exchange_begin")
        yield
        end = runtime.point(scope, "exchange_end")
        with owner.tensors.metadata():
            owner.capture.scopes[scope].setdefault("local_exchanges", []).append({
                "operation": operation, "begin": begin, "end": end, "bytes": source.numel() * source.element_size(),
                "dtype": str(source.dtype),
            })
            for tensor in outputs:
                owner.tensors.written(tensor)

    def observe_exchange(self, original, role, group, root_rank, operand, run):
        if operand is None:
            with self.exchange(original, group, root_rank=root_rank, role=role):
                return run()
        if operand.gather_dim is None:
            return self.reduce(original, group, operand, run)

        owner = self.observation
        standalone = role == "gather"
        tensor, dim, peers = operand.tensor, operand.gather_dim, operand.coordinator.world_size
        with owner.tensors.metadata():
            parameters = PipelineABI.tensor_parameters(tensor)
            shape = list(parameters)
            parent_shape = shape if standalone else owner.tensors.shape_rule(tensor)
            relations = {} if standalone else dict(zip(shape, parent_shape, strict=True))

        suffix, use = ("", "device") if standalone else (".layout", "metadata")
        with self.exchange(original, group, root_rank=root_rank, role=role, parameters=parameters, arguments=relations) as call:
            with owner.tensors.metadata():
                implementation = operand.implementation()
                if implementation is not None:
                    call.record["bindings"]["local_implementation"] = implementation
                call.record["guards"] = [GatherLayout.guard(tuple(tensor.shape), shape, dim, peers)]
                call.port(f"input{suffix}", "input", {**owner.tensors.observe(tensor, None), "shape_rule": shape}, use=use)
                if standalone:
                    call.record["bindings"]["collective"] = PipelineABI.gather_condition(tensor, dim, call.record["bindings"]["collective_group"])
            result = run()
            output_shape = list(shape)
            output_shape[dim] = ["mul", peers, shape[dim]]
            call.port(f"output{suffix}", "output", {**owner.tensors.observe(result, None), "shape_rule": output_shape}, use=use)

            # Local axis names belong to this call; later callers see the producer's enclosing expressions.
            with owner.tensors.metadata():
                parent_output = list(parent_shape)
                parent_output[dim] = ["mul", peers, parent_shape[dim]]
                owner.tensors.declare_shape(result, parent_output, scope=call.identity if standalone else call.record["parent"])
            return result

    def reduce(self, original, group, operand, run):
        tensors = self.observation.tensors
        with tensors.metadata():
            parent_shape = tensors.declared_shape(operand.tensor)
            parameters = PipelineABI.tensor_parameters(operand.tensor)
            storage = tensors.storage(operand.tensor)
            source = tensors.observe(operand.tensor, None)
            implementation = operand.implementation()

        # The collective preserves geometry even when its backend mutates the input storage.
        with self.exchange(original, group, parameters={}, arguments={}) as call:
            result = run()
            with tensors.metadata():

                # A native reduction can write its aliased result below PyTorch dispatch. Publish that change exactly once.
                if tensors.storage(result) is storage and storage.version == source["version"]:
                    tensors.written(result)

            if parent_shape is None:
                returned = tensors.observe(result, None)
                self.pending_reductions.append((call, parameters, source, implementation, returned["value"]))
            else:
                self.bind_reduction(call, parameters, source, implementation, parent_shape)
            return result

    @staticmethod
    def bind_reduction(call, parameters, source, implementation, parent_shape):
        call.record["parameters"] = parameters
        call.record["arguments"] = dict(zip(parameters, parent_shape, strict=True))
        if implementation is not None:
            call.record["bindings"]["local_implementation"] = implementation
        call.port("input.layout", "input", {**source, "shape_rule": list(parameters)}, use="metadata")

    def row_projection(self, original, layer, input_, *args, **kwargs):
        if not self.enabled or layer.tp_size == 1:
            return original(layer, input_, *args, **kwargs)
        first = len(self.pending_reductions)
        result = original(layer, input_, *args, **kwargs)
        if first == len(self.pending_reductions):
            return result

        tensors = self.observation.tensors
        output, _ = result
        with tensors.metadata():
            input_rule = tensors.declared_shape(input_)
            if input_rule is None or tuple(output.shape) != (*input_.shape[:-1], layer.output_size):
                return result

        # The native projection preserves leading axes and owns the output feature width.
        returned = tensors.observe(output, [*input_rule[:-1], layer.output_size])
        for index in reversed(range(first, len(self.pending_reductions))):
            if self.bind_returned_reduction(self.pending_reductions[index], (returned,)):
                del self.pending_reductions[index]
        return result

    def bind_returned_reduction(self, pending, outputs):
        call, parameters, source, implementation, value = pending
        rule = None

        # Only a declaration for the exact tensor version can identify the operand's layout.
        for tensor in outputs:
            if not tensor or tensor["value"] != value or "shape_rule" not in tensor:
                continue
            if (tensor["dtype"], tensor["shape"]) != (source["dtype"], source["shape"]):
                raise ValueError("native all-reduce changed the dtype or shape of its operand")
            if rule is not None and rule != tensor["shape_rule"]:
                raise ValueError("the same returned reduction has conflicting shape declarations")
            rule = tensor["shape_rule"]
        if rule is None:
            return False
        self.bind_reduction(call, parameters, source, implementation, rule)
        return True

    def bind_returned_operands(self):
        scopes = self.observation.capture.scopes
        for pending in self.pending_reductions:
            call, *_ = pending
            parent = scopes[call.record["parent"]]
            self.bind_returned_reduction(pending, (port.get("tensor") for port in parent["ports"] if port["direction"] == "output"))
        self.pending_reductions.clear()

    @contextmanager
    def exchange(self, original, group, *, root_rank=None, role="communication", parameters=None, arguments=None):
        import torch.distributed as distributed

        owner = self.observation
        root = owner.capture.current.get()
        while owner.capture.scopes[root]["parent"] not in (None, 0):
            root = owner.capture.scopes[root]["parent"]
        if root != self.action:
            self.action = root
            self.orders.clear()
        offset = owner.description["rank_offset"]
        group_role = owner.groups.name(group)
        descriptor = owner.worker_roles[str(owner.parallel["rank"])]["groups"][group_role]
        members, channel = descriptor["members"], descriptor["channel"]
        order = self.orders.get(channel, 0)
        self.orders[channel] = order + 1

        work = owner.pipeline.messages.work() if role == "gather" and owner.description["pp_size"] > 1 else None
        with owner.call(role, original, parameters=parameters, arguments=arguments, work=work, device=True) as call:
            call.record["bindings"]["collective_group"] = group_role
            call.record["communication"] = {
                "channel": channel, "operation": original.__name__, "members": members,
                "required": members if root_rank is None else [offset + root_rank],
                "rank": offset + distributed.get_rank(), "order": order,
            }
            yield call

    def experts(self, original, dispatcher, *args, **kwargs):
        owner = self.observation
        if not owner.active or not owner.selected:
            return original(dispatcher, *args, **kwargs)

        from sglang.srt.distributed.parallel_state import get_tp_group

        # SGLang supplies this process group to DeepEP; dispatch/combine include their local preparation and completion waits.
        group = get_tp_group()
        implementation = dispatcher._get_impl()
        if implementation.group is not group.device_group:
            raise ValueError("DeepEP dispatcher uses a different native communication group")
        bound = inspect.signature(original).bind(dispatcher, *args, **kwargs)
        dispatch = original.__name__ == "dispatch"
        tensor = bound.arguments["hidden_states"] if dispatch else bound.arguments["combine_input"].hidden_states

        # The layout all-gather owns its own rendezvous inside dispatch; the enclosing call owns AllToAllV and local work.
        with self.exchange(original, implementation.group) as call:
            handle = None if dispatch else implementation.handle
            rows = "expert_input_tokens" if dispatch else "expert_received_tokens"
            owner.experts.port(call, "hidden_states", "input", tensor, rows)
            result = original(dispatcher, *args, **kwargs)
            if dispatch:
                handle = implementation.handle
            sent, received = list(handle["input_splits"]), list(handle["output_splits"])
            if not dispatch:
                sent, received = received, sent
            exchange = call.record["communication"]
            exchange["required"] = [
                rank for rank, send, receive in zip(exchange["members"], sent, received, strict=True)
                if rank == exchange["rank"] or send or receive
            ]
            parameter = ScopeParameters(owner.capture.scopes).resolve("expert_received_tokens", call.identity)
            call.record["bindings"]["required_peers"] = parameter.name
            call.record["peer_rows"] = {"send": sent, "receive": received}

            rows = "expert_received_tokens" if dispatch else "expert_input_tokens"
            owner.experts.port(call, "hidden_states", "output", result.hidden_states if dispatch else result, rows)
            if dispatch and result.hidden_states_scale is not None:
                owner.experts.port(call, "scales", "output", result.hidden_states_scale, rows)
            return result

    def install(self) -> None:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        self.cuts.install(self.experts)
        HookRegistry.register("sglang.srt.layers.linear.RowParallelLinear.forward", self.row_projection, HookType.AROUND)
