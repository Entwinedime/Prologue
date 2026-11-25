"""Record collectives and pipeline transfers at their native completion boundaries."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch.distributed as distributed

from prologue.common.observation import ScopeParameters
from prologue.integrations.bindings import platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.observation.model import enabled_hook, tensor_ports
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors
from prologue.integrations.frameworks.sglang.shared.calls import PipelineABI, PipelineMessages
from prologue.integrations.runtimes.pytorch.capture import GatherLayout, NativeSignals


class TransferWork:
    def __init__(self, transfers, native, producer, direction, context) -> None:
        self.transfers = transfers
        self.native = native
        self.producer = producer
        self.direction = direction
        self.context = context

    def wait(self, *args, **kwargs):
        from torch.distributed import Work

        transfers = self.transfers
        if not transfers.calls.enabled:
            return self.native.wait(*args, **kwargs)
        work = transfers.messages.wait_work(self.context, device=False)
        implementation = f"{Work.__module__}.{Work.__qualname__}.wait"
        with transfers.consume(f"transfer_{self.direction}_wait", implementation, work, self.producer):
            return self.native.wait(*args, **kwargs)


class TensorTransfers:
    def __init__(self, messages, current, rank_offset) -> None:
        self.messages = messages
        self.calls = messages.calls
        self.current = current
        self.rank_offset = rank_offset

    def invoke(self, signature, original, *args, **kwargs):

        if not self.calls.enabled:
            return original(*args, **kwargs)
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        tensor = values["tensor"]
        if tensor.device.type != platform_domain().device_type:
            return original(*args, **kwargs)
        if self.current.get() != 0:
            raise ValueError("pipeline tensor transfer must expose its own action boundary")
        group = distributed.group.WORLD if values["group"] is None else values["group"]
        receiving = original.__name__ == "irecv"
        endpoint = "src" if receiving else "dst"
        peer = values[endpoint]
        if values.get(f"group_{endpoint}") is not None:
            peer = distributed.get_global_rank(group, values[f"group_{endpoint}"])
        if peer is None:
            raise ValueError("pipeline tensor receive requires an explicit sender")
        rank = distributed.get_rank()
        source, target = (peer, rank) if receiving else (rank, peer)
        offset = self.rank_offset
        members = [offset + source, offset + target]
        direction = "receive" if receiving else "send"
        parameters = PipelineABI.tensor_parameters(tensor)
        work = self.messages.work()
        communication = {
            "channel": f"{group.group_name}/{members[0]}>{members[1]}/{values['tag']}",
            "operation": "transfer", "members": members, "required": members,
            "rank": offset + rank, "order": 0,
        }

        # This boundary sees the actual transmitted slice, after the native send-allgather transformation.
        with self.submit(original, direction, tensor, parameters, work, communication) as producer:
            result = original(*args, **kwargs)
        return TransferWork(self, result, producer, direction, work["context"])

    def install(self):

        for name in ("isend", "irecv"):
            HookTable.around(
                f"torch.distributed.{name}", partial(self.invoke, inspect.signature(getattr(distributed, name))),
            )


class NativeTransfers(TensorTransfers):
    def __init__(self, observation, messages) -> None:
        super().__init__(messages, observation.capture.current, observation.description["rank_offset"])
        self.observation = observation

    @contextmanager
    def submit(self, original, direction, tensor, parameters, work, communication):
        owner = self.observation
        with owner.capture.call(
            f"transfer_{direction}", original.__qualname__, f"{original.__module__}.{original.__qualname__}",
            parameters=parameters, work=work, device=True,
        ) as call:
            call.record["bindings"]["dtype"] = str(tensor.dtype)
            call.record["asynchronous"] = True
            call.record["communication"] = communication
            if direction == "send":
                call.port("payload", "input", owner.tensors.observe(tensor, list(parameters)), use="device")
            yield call.identity
            if direction == "receive":
                owner.tensors.written(tensor)
                call.port("payload", "output", owner.tensors.observe(tensor, list(parameters)), use="device")
            call.work_signal("completion", "output", call.identity)
        owner.tensors.retire()

    @contextmanager
    def consume(self, role, implementation, work, producer):
        with self.observation.capture.call(
            role, "Work.wait", implementation, parameters={}, work=work, device=True
        ) as call:
            call.work_signal("completion", "input", producer)
            yield


class HostTransfers(TensorTransfers):
    def __init__(self, host, messages) -> None:
        super().__init__(messages, host.parent, host.rank_offset)

    def submit(self, original, direction, tensor, parameters, work, communication):
        return self.calls.native_call(
            f"transfer_{direction}", original, work=work, parameters=parameters, bindings={"dtype": str(tensor.dtype)},
        )

    def consume(self, role, implementation, work, producer):
        return self.calls.scope(role, "Work.wait", implementation, work=work)


class PipelineCuts:
    def __init__(self, calls, current, runtime=None) -> None:
        self.calls = calls
        self.current = current
        self.messages = PipelineMessages(calls)
        self.signals = NativeSignals(current, self.site_scope, self.messages, runtime)

    def wait_stream(self, original, stream, other):
        if not self.calls.enabled or self.current.get() != 0:
            return original(stream, other)
        streams = PipelineABI.streams(self.calls.scheduler, stream, other)
        work = self.messages.work()
        work["context"]["streams"] = streams
        with self.site_scope(
            "stream_wait", original, work=work, bindings={"streams": streams}, device=True, stream=stream
        ):
            return original(stream, other)

    def install(self, device_api):
        self.messages.install()
        self.transfers.install()
        scheduler = symbols().scheduler + "."
        hooks = {"_pp_launch_batch": self.launch, "_pp_prep_batch_result": self.prepare,
                 "_pp_make_skip_output_result": self.skip_output}
        HookTable.group({
            scheduler + name: enabled_hook(self.observation, hook) for name, hook in hooks.items()
        }).install()
        for name in ("record", "wait", "synchronize", "query"):
            HookTable.around(f"{device_api}.Event.{name}", getattr(self.signals, name))
        HookTable.around(f"{device_api}.Stream.wait_stream", self.wait_stream)


class PipelineScopes(PipelineCuts):
    def __init__(self, observation) -> None:
        super().__init__(observation.calls, observation.capture.current, observation.capture.runtime)
        self.observation = observation
        self.transfers = NativeTransfers(observation, self.messages)

    def site_scope(self, role, original, *, work, device, stream, bindings=None):
        return self.observation.capture.call(
            role, original.__qualname__, f"{original.__module__}.{original.__qualname__}", parameters={},
            work=work, bindings=bindings, device=device, stream=stream,
        )

    def launch(self, original, scheduler, mb_id, pp_proxy_tensors, mb_metadata, last_rank_comm_queue):
        owner = self.observation

        with (
            owner.model_call("launch", original, scheduler.cur_batch, stream=scheduler.forward_stream) as call,
            self.signals.observe(),
        ):
            if pp_proxy_tensors is not None:
                for name, tensor in PipelineABI.proxy_tensors(pp_proxy_tensors).items():

                    # Launch forwards handles before its stream wait; the model body owns the actual device read.
                    call.port(
                        f"proxy.{name}",
                        "input",
                        owner.tensors.observe(tensor, owner.modules.proxy_shape(tensor)),
                        use="metadata",
                    )
            result, event = original(scheduler, mb_id, pp_proxy_tensors, mb_metadata, last_rank_comm_queue)
            self.signals.publish(call, event, "completion")
            owner.result_outputs(call, result)
            return result, event

    def prepare(self, original, scheduler, batch, mb_metadata, pp_outputs):
        owner = self.observation
        work = owner.batch_work(batch)
        parameters = BatchTensors.parameters(batch, sum(member["new_tokens"] for member in work["members"]))
        with (
            owner.request_state.members(batch),
            owner.call(
                "output_prepare", original, parameters=parameters, work=work, device=True, stream=scheduler.copy_stream
            ) as call,
        ):
            call.port(
                "next_token_ids",
                "input",
                owner.tensors.observe(pp_outputs["next_token_ids"], ["members"]),
                use="device",
            )
            result = original(scheduler, batch, mb_metadata, pp_outputs)
            owner.result_outputs(call, result)
            return result

    def skip_output(self, original, scheduler, batch, mb_metadata):
        owner = self.observation
        work = owner.batch_work(batch)
        parameters = BatchTensors.parameters(batch, sum(member["new_tokens"] for member in work["members"]))
        with (
            owner.call("output_skip", original, parameters=parameters, work=work, device=True) as call,
            self.signals.observe(),
        ):
            outputs, result, event = original(scheduler, batch, mb_metadata)
            owner.result_outputs(call, result)
            self.signals.publish(call, event, "completion")
            return outputs, result, event


class HostPipelineScopes(PipelineCuts):
    def __init__(self, host) -> None:
        super().__init__(host, host.parent)
        self.observation = host
        self.transfers = HostTransfers(host, self.messages)

    def site_scope(self, role, original, *, work, device, stream, bindings=None):
        return self.calls.native_call(role, original, work=work, bindings=bindings)

    def launch(self, original, scheduler, mb_id, pp_proxy_tensors, mb_metadata, last_rank_comm_queue):
        args = scheduler, mb_id, pp_proxy_tensors, mb_metadata, last_rank_comm_queue
        work = self.calls.batch_work(scheduler.cur_batch)
        with self.calls.native_call("launch", original, work=work) as call, self.signals.observe():
            result, event = original(*args)
            self.signals.publish(call, event, "completion")
            return result, event

    def prepare(self, original, scheduler, batch, mb_metadata, pp_outputs):
        with self.calls.native_call("output_prepare", original, work=self.calls.batch_work(batch)):
            return original(scheduler, batch, mb_metadata, pp_outputs)

    def skip_output(self, original, scheduler, batch, mb_metadata):
        work = self.calls.batch_work(batch)
        with self.calls.native_call("output_skip", original, work=work) as call, self.signals.observe():
            outputs, result, event = original(scheduler, batch, mb_metadata)
            self.signals.publish(call, event, "completion")
            return outputs, result, event


@dataclass
class CollectiveOperand:
    coordinator: Any
    tensor: Any
    gather_dim: int | None = None
    output: Any | None = None

    def implementation(self):
        communicator = getattr(self.coordinator, symbols().communicator_attribute)
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
        if not any(tensor.device.type == platform_domain().device_type for _, tensor in tensors):
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
        elif original.__name__ in {"all_gather_into_tensor", "reduce_scatter_tensor"}:
            operand = CollectiveOperand(group, arguments.arguments["input"], output=arguments.arguments["output"])
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
        if not any(tensor.device.type == platform_domain().device_type for _, tensor in tensors):
            return original(*args, **kwargs)
        if arguments.arguments["async_op"]:
            raise ValueError("asynchronous native collectives need an explicit work-completion boundary")

        group = arguments.arguments["group"]
        if group is None:
            group = distributed.group.WORLD
        if distributed.get_world_size(group) == 1:
            return original(*args, **kwargs)
        root_rank = arguments.arguments.get("src")
        if arguments.arguments.get("group_src") is not None:
            root_rank = distributed.get_global_rank(group, arguments.arguments["group_src"])
        operand = None
        if original.__name__ == "all_gather_into_tensor":
            operand = CollectiveOperand(
                None, arguments.arguments["input_tensor"], output=arguments.arguments["output_tensor"]
            )
        with self.outer():
            return self.scopes.observe_exchange(
                original, "communication", group, root_rank, operand, partial(original, *args, **kwargs),
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
        GroupCoordinator = resolve(symbols().group_coordinator)

        for name in (
            "all_reduce",
            "quant_all_reduce",
            "all_gather",
            "all_gather_into_tensor",
            "reduce_scatter",
            "reduce_scatter_tensor",
            "all_to_all_single",
            "broadcast",
        ):
            target = f"{symbols().group_coordinator}.{name}"
            HookTable.around(target, partial(self.invoke, inspect.signature(getattr(GroupCoordinator, name))))
        for name in ("all_gather_into_tensor", "all_gather", "all_reduce", "broadcast"):
            target = f"torch.distributed.{name}"
            HookTable.around(target, partial(self.distributed, inspect.signature(getattr(distributed, name))))
        if expert_exchange is not None:
            for name in ("dispatch", "combine"):
                target = f"{symbols().deep_ep_dispatcher}.{name}"
                HookTable.around(target, expert_exchange)


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
        if operand.output is not None:
            return self.buffer_exchange(original, group, operand, run)
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
        with self.exchange(
            original, group, root_rank=root_rank, role=role, parameters=parameters, arguments=relations
        ) as call:
            with owner.tensors.metadata():
                implementation = operand.implementation()
                if implementation is not None:
                    call.record["bindings"]["local_implementation"] = implementation
                call.record["guards"] = [GatherLayout.guard(tuple(tensor.shape), shape, dim, peers)]
                call.port(
                    f"input{suffix}", "input", {**owner.tensors.observe(tensor, None), "shape_rule": shape}, use=use
                )
                if standalone:
                    call.record["bindings"]["collective"] = PipelineABI.gather_condition(
                        tensor, dim, call.record["bindings"]["collective_group"]
                    )
            result = run()
            output_shape = list(shape)
            output_shape[dim] = ["mul", peers, shape[dim]]
            call.port(
                f"output{suffix}",
                "output",
                {**owner.tensors.observe(result, None), "shape_rule": output_shape},
                use=use,
            )

            # Local axis names belong to this call; later callers see the producer's enclosing expressions.
            with owner.tensors.metadata():
                parent_output = list(parent_shape)
                parent_output[dim] = ["mul", peers, parent_shape[dim]]
                owner.tensors.declare_shape(
                    result, parent_output, scope=call.identity if standalone else call.record["parent"]
                )
            return result

    def buffer_exchange(self, original, group, operand, run):
        owner = self.observation
        parent = owner.capture.scopes[owner.capture.current.get()]
        with self.exchange(original, group) as call:
            with owner.tensors.metadata():
                peers = len(call.record["communication"]["members"])
                shapes = self.buffer_shapes(parent, original.__name__, operand, peers)
                if shapes is not None:
                    source = owner.tensors.observe(operand.tensor, None)
                    call.port("input.layout", "input", {**source, "shape_rule": shapes[0]}, use="metadata")
            result = run()
            if shapes is not None:
                target = owner.tensors.observe(operand.output, None)
                call.port("output.layout", "output", {**target, "shape_rule": shapes[1]}, use="metadata")
            return result

    @staticmethod
    def buffer_shapes(parent, operation, operand, peers):
        source, target = list(operand.tensor.shape), list(operand.output.shape)
        if parent["role"] == "layer":

            # TP scatter and gather exchange the same hidden width across whole or sharded batch rows.
            full = parent["parameters"]["model_tokens"]
            local = full // peers
            scatter = operation == "reduce_scatter_tensor"
            wide, narrow = (source, target) if scatter else (target, source)
            if full % peers != 0 or wide[0] != full or narrow[0] != local or wide[1:] != narrow[1:]:
                raise ValueError("collective buffer differs from the declared TP row layout")
            wide[0], narrow[0] = "model_tokens", ["floor_div", "model_tokens", peers]
            return source, target

        # Scheduler metadata and the native expert histogram have a fixed ABI for this domain.
        if (parent["implementation"] == symbols().mlp_sync_batch_info_all_gather
                or parent.get("communication", {}).get("operation") == "dispatch"):
            return source, target
        return None

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

                # A native reduction can write its aliased result below PyTorch dispatch. Publish that change exactly
                # once.
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
            self.bind_returned_reduction(
                pending, (port.get("tensor") for port in parent["ports"] if port["direction"] == "output")
            )
        self.pending_reductions.clear()

    @contextmanager
    def exchange(self, original, group, *, root_rank=None, role="communication", parameters=None, arguments=None):

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

        # SGLang supplies this process group to DeepEP; dispatch/combine include their local preparation and
        # completion waits.
        group = resolve(symbols().get_tp_group)()
        implementation = dispatcher._get_impl()
        if implementation.group is not group.device_group:
            raise ValueError("DeepEP dispatcher uses a different native communication group")
        bound = inspect.signature(original).bind(dispatcher, *args, **kwargs)
        dispatch = original.__name__ == "dispatch"
        tensor = bound.arguments["hidden_states"] if dispatch else bound.arguments["combine_input"].hidden_states

        # The layout all-gather owns its own rendezvous inside dispatch; the enclosing call owns AllToAllV and local
        # work.
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
        self.cuts.install(self.experts)
        HookTable.around(symbols().row_parallel_linear_forward, self.row_projection)
