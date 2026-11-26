"""Identify native scheduling, allocation and batch-construction branches."""

import __future__

import ast
import inspect
import sys
import textwrap

from prologue.integrations.bindings import platform_domain
from prologue.integrations.frameworks.sglang.abi.hooks import HookTable
from prologue.integrations.frameworks.sglang.abi.registry import resolve, symbols
from prologue.integrations.frameworks.sglang.shared.batches import BatchTensors
from prologue.integrations.frameworks.sglang.shared.calls import ActionCall


class AllocationABI:
    def __init__(self, implementation: str) -> None:
        if implementation != symbols().paged_allocator:
            raise ValueError(f"allocation structure has no declared implementation: {implementation}")
        self.implementation = implementation

    def branch(self, members, page_size, decode):
        if page_size == 1:
            return "token"
        pages = sum(
            (member["context_tokens"] + member["new_tokens"] + page_size - 1) // page_size
            - (member["context_tokens"] + page_size - 1) // page_size
            for member in members
        )

        if decode:
            return "decode_pages" if pages else "decode_tail"

        return self.extend_branch(pages)

    @staticmethod
    def extend_branch(pages):
        limit = platform_domain().paged_extend_kernel_limit
        return "extend_kernel" if limit is None or pages < limit else "extend_naive"


class AllocationPrefixes:
    """Wrap each native prefix expression, leaving its branch and enclosing loop native."""

    def __init__(self, calls) -> None:
        alloc_for_extend = resolve(symbols().alloc_for_extend)

        self.calls = calls
        self.original = inspect.unwrap(alloc_for_extend)
        lines, start = inspect.getsourcelines(self.original)
        native = ast.parse(textwrap.dedent("".join(lines))).body[0]
        ast.increment_lineno(native, start - 1)
        assignment, = [node for node in ast.walk(native) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "last_loc" for target in node.targets)
                       and isinstance(node.value, ast.ListComp)]
        comprehension = assignment.value
        generator, = comprehension.generators
        item, batch = generator.target.id, native.args.args[0].arg

        # Compile the installed expression, rather than copying its index/empty-prefix algorithm into this adapter.
        value = ast.parse("def prefix_value(prefix, batch):\n    return None").body[0]
        value.args.args[0].arg, value.args.args[1].arg = item, batch
        value.body[0].value = comprehension.elt
        ast.copy_location(value, comprehension.elt)
        ast.copy_location(value.body[0], comprehension.elt)
        index = "prologue_prefix_index"
        arguments = [ast.Name(name, ast.Load()) for name in ("prefix_value", index, item, batch)]
        comprehension.elt = ast.copy_location(
            ast.Call(ast.Name("observe", ast.Load()), arguments, []), comprehension.elt
        )
        generator.iter = ast.copy_location(
            ast.Call(ast.Name("enumerate", ast.Load()), [generator.iter], []), generator.iter
        )
        generator.target = ast.copy_location(
            ast.Tuple([ast.Name(index, ast.Store()), generator.target], ast.Store()), generator.target
        )

        # The closure holds only the instrumentation callback; native globals stay live for the framework's other hooks.
        factory = ast.parse("def instrument(observe):\n    pass").body[0]
        ast.copy_location(factory, native)
        factory.body = [
            value,
            native,
            ast.copy_location(ast.Return(ast.Name(native.name, ast.Load())), native.body[-1]),
        ]
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        namespace = {}
        flags = self.original.__code__.co_flags & __future__.annotations.compiler_flag
        exec(compile(module, self.original.__code__.co_filename, "exec", flags=flags, dont_inherit=True),
             self.original.__globals__, namespace)
        self.instrumented = namespace["instrument"](self.prefix)

    def install(self):
        HookTable.replace(symbols().alloc_for_extend, self.allocate)

    def allocate(self, batch):
        if not self.calls.enabled or self.calls.current.get() is None:
            return self.original(batch)
        return self.instrumented(batch)

    def prefix(self, native, index, prefix, batch):
        parent = self.calls.current.get()
        work = {**parent.work, "context": {"member": parent.work["members"][index]}}

        # The private index value is consumed after this synchronous construction/view returns; only its causal
        # boundary escapes.
        with self.calls.call(
            "allocation_prefix", self.original, parameters={}, work=work, condition={"cached": len(prefix) > 0}
        ):
            return native(prefix, batch)


class AllocationIndices:
    def __init__(self, calls):
        self.calls = calls

    def install(self):
        alloc_extend_naive = resolve(symbols().paged_alloc_extend_naive)


        original = inspect.unwrap(alloc_extend_naive)
        lines, start = inspect.getsourcelines(original)
        definition = ast.parse(textwrap.dedent("".join(lines))).body[0]
        loop, = [node for node in definition.body if isinstance(node, ast.For)]
        self.code = original.__code__
        self.begin, self.end = start + loop.lineno - 1, start + loop.end_lineno - 1
        self.parts = {}
        names = {"num1": "existing_page", "num2": "full_pages", "num3": "last_page"}

        for branch in loop.body:
            if isinstance(branch, ast.If) and isinstance(branch.test, ast.Name):
                name = names[branch.test.id]
                for statement in branch.body:
                    for line in range(start + statement.lineno - 1, start + statement.end_lineno):
                        self.parts[line] = name

        HookTable.around(symbols().paged_alloc_extend_naive, self.invoke)

    def invoke(self, original, prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, device):
        arguments = (prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size, device)
        if not self.calls.enabled or self.calls.current.get() is None:
            return original(*arguments)

        previous = sys.gettrace()
        if previous is not None:
            raise RuntimeError("native allocation observation requires ownership of this thread's Python trace")
        iteration = AllocationIteration(self, original, out_indices)

        try:
            sys.settrace(iteration.observe)
            return original(*arguments)
        finally:
            error = sys.exc_info()
            sys.settrace(previous)
            iteration.finish(error)


class AllocationIteration:
    def __init__(self, owner, original, output):
        self.owner, self.original, self.output = owner, original, output
        self.parent = owner.calls.current.get()
        self.active = None

    def observe(self, native, event, argument):
        if native.f_code is not self.owner.code:
            return None
        if event == "line":
            line = native.f_lineno
            if line == self.owner.begin:
                self.finish((None, None, None))
            elif self.owner.begin < line <= self.owner.end:
                if self.active is None:
                    self.start(native.f_locals["i"])
                parts = self.active[1].condition["allocation_parts"]
                part = self.owner.parts.get(line)

                if part is not None and part not in parts:
                    parts.append(part)

        return self.observe

    def start(self, index):
        work = dict(self.parent.work)
        work["context"] = {"member": work["members"][index]}
        inputs = {name: value for name, value in self.parent.tensor_inputs.items() if name != "cache_locations"}
        outputs = {**inputs, "cache_locations": (self.output, ["input_tokens"])}

        # Every member borrows storage identity; only the first registers it, without reading uninitialized contents.
        metadata = {"cache_locations": outputs["cache_locations"]}
        allocations = {"cache_locations": self.output} if index == 0 else {}
        manager = self.owner.calls.call(
            "allocation_member", self.original, work=work, tensor_inputs=inputs, metadata_inputs=metadata,
            tensor_outputs=outputs, allocations=allocations,
            condition={"allocation_parts": []},
        )
        frame = manager.__enter__()
        frame.tensor_outputs = outputs
        self.active = manager, frame

    def finish(self, error):
        if self.active is not None:
            manager, self.active = self.active[0], None
            manager.__exit__(*error)


class BatchPreparation(ActionCall):
    def __init__(self, calls, batch, mode) -> None:
        if mode == "decode":
            contexts = batch.seq_lens_cpu.tolist()
            lengths = [1] * len(batch.reqs)
        else:
            contexts = [len(req.prefix_indices) for req in batch.reqs]
            lengths = [req.extend_range.length for req in batch.reqs]

        members = [
            {"id": req.rid, "context_tokens": context, "new_tokens": tokens, "phase": mode}
            for req, context, tokens in zip(batch.reqs, contexts, lengths, strict=True)
        ]
        super().__init__(
            "preparation", BatchTensors.parameters(batch, sum(lengths)),
            calls.work(batch, members, phase=mode),
        )
        self.batch = batch

    def complete(self):
        self.outputs = BatchTensors.boundaries(self.batch)


class BatchTransform(ActionCall):
    @staticmethod
    def private(operation, members):

        # Native filter_batch returns before accessing tensors when its input batch is empty.
        return operation == "filter" and members == 0

    def __init__(self, calls, operation, batch, other=None) -> None:
        super().__init__(
            "batch_transform",
            {
                **BatchTensors.capacity(batch.token_to_kv_pool_allocator, batch.req_to_token_pool),
                "left_members": len(batch.reqs), "right_members": 0 if other is None else len(other.reqs),
                "members": len(batch.reqs),
            },
            calls.work(batch),
        )
        self.calls = calls
        self.transformation = operation
        self.batch = batch
        self.work["context"] = {"operation": operation, "left": self.work["context"]["members"]}

        if other is not None:
            self.work["context"]["right"] = calls.progress(other.reqs)
        for side, value in (("left", batch), ("right", other)):
            if value is None or not value.reqs:
                continue
            self.inputs.update({
                f"{side}.{name}": (tensor, [f"{side}_members"])
                for name, (tensor, _) in BatchTensors.boundaries(value).items()
                if name in {"request_slots", "sequence_lengths"}
            })

            if operation == "mix":
                self.parameters[f"{side}_tokens"] = int(value.out_cache_loc.numel())
                self.inputs[f"{side}.cache_locations"] = value.out_cache_loc, [f"{side}_tokens"]

        if operation == "mix":
            self.parameters["input_tokens"] = self.parameters["left_tokens"] + self.parameters["right_tokens"]

    def complete(self):
        before, after = self.parameters["left_members"], len(self.batch.reqs)
        self.parameters["members"] = after
        outcome = self.filter_outcome(before, after) if self.transformation == "filter" else self.transformation
        self.condition = {"transformation": {"operation": self.transformation, "outcome": outcome}}

        # Replace the context: the caller's preceding fragment retains the original input snapshot.
        self.work["context"] = {**self.work["context"], "result": self.calls.progress(self.batch.reqs)}

        # Native early returns neither read batch tensors nor produce new ones, even when stale tensors remain.
        if outcome in {"empty", "unchanged"}:
            self.inputs.clear()
            return

        self.outputs = {
            name: item for name, item in BatchTensors.boundaries(self.batch).items()
            if self.transformation == "mix" or name in {"request_slots", "sequence_lengths"}
        }

    @staticmethod
    def filter_outcome(before, after):
        return "unchanged" if after == before else "empty" if after == 0 else "subset"


class BatchAllocation(ActionCall):
    def __init__(self, calls, batch, decode) -> None:
        parent = calls.current.get()
        super().__init__("allocation", dict(parent.parameters), dict(parent.work))
        self.inputs = {"sequence_lengths": (batch.seq_lens, ["members"])}
        if decode:
            self.inputs["request_slots"] = batch.req_pool_indices, ["members"]

        # Extend creates lengths here; Decode borrows tensors that allocation reads first.
        if not decode:
            parent.tensor_outputs.update(self.inputs)

    def complete(self, result, decode):
        self.outputs.update(self.inputs)
        self.outputs["cache_locations"] = (result if decode else result[0]), ["input_tokens"]
        if not decode:
            self.outputs["request_slots"] = result[1], ["members"]


class BatchSelection:
    def __init__(self, calls) -> None:
        self.calls = calls

    def select(self, original, scheduler, *args, **kwargs):
        if not self.calls.enabled:
            return original(scheduler, *args, **kwargs)

        work = self.calls.scheduler_work()

        # Native admission and loops choose child actions; the envelope has no second execution graph.
        with self.calls.call("selection", original, parameters={}, work=work, host_only=True):
            return original(scheduler, *args, **kwargs)

    def prepare(self, original, batch):
        if not self.calls.enabled:
            return original(batch)
        mode = {"prepare_for_extend": "prefill", "prepare_for_decode": "decode", "prepare_for_idle": "idle"}[
            original.__name__
        ]
        boundary = BatchPreparation(self.calls, batch, mode)

        with self.calls.call(
            boundary.operation, original, parameters=boundary.parameters, work=boundary.work,
            tensor_inputs=boundary.inputs,
        ) as frame:

            # Allocation and resource calls own their work; preparation resumes with their published tensors.
            result = original(batch)
            boundary.complete()
            frame.tensor_outputs = boundary.outputs

        return result

    def allocate(self, original, batch, token_per_req=None):
        decode = token_per_req is not None
        arguments = (batch, token_per_req) if decode else (batch,)
        if not self.calls.enabled or self.calls.current.get() is None:
            return original(*arguments)

        boundary = BatchAllocation(self.calls, batch, decode)
        with self.calls.call(
            boundary.operation, original, parameters=boundary.parameters, work=boundary.work,
            tensor_inputs=boundary.inputs, tensor_outputs=boundary.outputs,
        ) as frame:

            # Eviction remains an independent resource action inside the allocation lifecycle.
            result = original(*arguments)
            boundary.complete(result, decode)
            frame.tensor_outputs = boundary.outputs

        return result

    def transform(self, operation, original, batch, arguments, keywords, other=None):
        parent = self.calls.current.get()
        if (not self.calls.enabled or parent is None
                or parent.host_only and BatchTransform.private(operation, len(batch.reqs))):
            return original(batch, *arguments, **keywords)

        boundary = BatchTransform(self.calls, operation, batch, other)
        with self.calls.separate(boundary.operation, boundary.work), self.calls.boundary(original, boundary):

            # mix owns its nested merge; independently submitting it would count the same work twice.
            result = original(batch, *arguments, **keywords)
            boundary.complete()

        return result

    def filter(self, original, batch, *args, **kwargs):
        return self.transform("filter", original, batch, args, kwargs)

    def merge(self, original, batch, other):
        return self.transform("merge", original, batch, (other,), {}, other)

    def mix(self, original, batch, other):
        return self.transform("mix", original, batch, (other,), {}, other)

    def synchronize(self, original, adapter, batch):
        if not self.calls.enabled or self.calls.current.get() is None:
            return original(adapter, batch)
        work = self.calls.work(batch)

        # An idle participant has an observed empty batch, not an unknown logical context.
        if batch is None:
            work["context"] = {"members": []}

        with self.calls.call("batch_sync", original, parameters={}, work=work):
            return original(adapter, batch)

    def gather(self, original, info, device, group):
        if not self.calls.enabled or self.calls.current.get() is None:
            return original(info, device, group)
        work = self.calls.current.get().work
        observing = self.calls.request_state.observe
        inputs = None

        if observing:
            inputs = {name: getattr(info, name) for name in (
                "num_tokens", "num_tokens_for_logprob", "can_cuda_graph", "is_extend_in_batch",
                "local_can_run_tbo", "local_forward_mode", "can_run_breakable_cuda_graph",
            )}

        with self.calls.call("batch_sync_exchange", original, parameters={}, work=work) as frame:
            result = original(info, device, group)
            if observing:
                get_tp_group = resolve(symbols().get_tp_group)

                # Native all_gather already materialized these Python results; observation performs no device read.
                frame.values["batch_sync"] = {
                    "group": self.calls.groups.name(group),
                    "layout": [info.dp_size, info.tp_size, info.cp_size],
                    "active": get_tp_group().active_ranks_cpu.tolist(),
                    "input": inputs,
                    "output": {
                        "global_num_tokens": list(info.global_num_tokens),
                        "global_num_tokens_for_logprob": list(info.global_num_tokens_for_logprob),
                        **{name: getattr(info, name) for name in (
                            "can_cuda_graph", "is_extend_in_batch", "can_run_breakable_cuda_graph",
                        )},
                    },
                }

            return result

    def install(self) -> None:
        AllocationIndices(self.calls).install()
        AllocationPrefixes(self.calls).install()
        HookTable.around(symbols().mlp_sync_batch_info_all_gather, self.gather)
        owner = symbols().schedule_batch + "."

        for method, hook in (
            ("prepare_for_extend", self.prepare),
            ("prepare_for_decode", self.prepare),
            ("prepare_for_idle", self.prepare),
            ("filter_batch", self.filter),
            ("merge_batch", self.merge),
            ("mix_with_running", self.mix),
        ):
            HookTable.around(owner + method, hook)

        for method in ("alloc_for_extend", "alloc_for_decode"):
            HookTable.around(symbols().common + "." + method, self.allocate)
        HookTable.around(symbols().scheduler_get_next_batch_to_run, self.select)
        HookTable.around(symbols().scheduler_dp_attn_adapter_prepare_mlp_sync_batch, self.synchronize)
