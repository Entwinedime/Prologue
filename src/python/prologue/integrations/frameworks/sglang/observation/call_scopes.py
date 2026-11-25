"""Record one native call with its tensor boundaries and request-owned state effects."""

from contextlib import ExitStack, contextmanager
from typing import Any

from prologue.common.observation import SemanticCapture
from prologue.integrations.frameworks.sglang.shared.state_calls import RequestStateCalls
from prologue.integrations.runtimes.pytorch.capture import TensorBoundaries


class NativeCallObservation:
    def __init__(self, markers, mode, *, resource_trace=False):
        self.mode = mode
        self.capture = SemanticCapture(markers)
        self.tensors = TensorBoundaries(self.capture)
        self.request_state = RequestStateCalls(observe=resource_trace)
        self.action_values = []
        self.selected = True
        self.active = False
        self.resource_history = None

    @property
    def protocol_enabled(self):
        return self.active

    @contextmanager
    def call(self, role: str, original: Any, *, parameters=None, arguments=None, work=None, device=False, stream=None):
        outer = self.capture.current.get() == 0
        try:
            with ExitStack() as contexts:
                call = contexts.enter_context(self.capture.call(
                    role, original.__qualname__, f"{original.__module__}.{original.__qualname__}",
                    parameters=parameters, arguments=arguments, work=work, device=device, stream=stream,
                    constants=None if parameters is None else {
                        name: parameters[name] for name in ("page_size", "sequence_capacity") if name in parameters
                    },
                ))
                if outer:
                    contexts.enter_context(self.tensors)
                with self.request_state.record() as accesses:
                    yield call
                for key, requests in accesses.members.items():
                    identities = [self.aliases[request] for request in requests]
                    if not identities:
                        continue
                    field, mode = key.split("/")
                    call.record["work"][key] = identities
                    call.record.setdefault("state_accesses", []).append({
                        "space": "slot", "pool": "local", "field": field, "access": mode,
                        "members": key, "requests": identities, "interval": "device",
                    })
            if accesses.values:
                self.action_values.append({"scope": call.identity, "values": {"state": [
                    {**value, "request": self.aliases[value["request"]]} for value in accesses.values
                ]}})
        finally:
            if outer:
                self.tensors.retire()
