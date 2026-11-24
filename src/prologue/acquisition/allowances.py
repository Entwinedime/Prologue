"""Account for the sessions and library additions owned by one exploration."""

import json
import math
import subprocess
import time
from contextlib import contextmanager

from prologue.acquisition.capture import NativeAttempt
from prologue.artifacts import write_json
from prologue.workflow.limits import AllowanceExhausted, OperationLimit, Usage


class AcquisitionBudget:
    def __init__(self, root, store, arguments):
        seconds = arguments.card_budget_seconds, arguments.postprocessing_budget_seconds
        if (any(x is None or not math.isfinite(x) or x < 0 for x in seconds)
                or arguments.attempt_limit is None or arguments.attempt_limit < 0
                or arguments.storage_bytes is None or arguments.storage_bytes < 0):
            raise ValueError("on-demand acquisition requires nonnegative card, processing, attempt and storage allowances")
        self.cards, self.processing_ns = (int(x * 1e9) for x in seconds)
        self.attempts = arguments.attempt_limit
        self.root, self.store = root, store
        self.account = root / "acquisition/account.json"
        self.limit = OperationLimit(root / "acquisition", arguments.storage_bytes, self.published_bytes)

    def open(self):
        self.state = json.loads(self.account.read_bytes()) if self.account.is_file() else {"processing_ns": 0, "published": {}}
        self.deadline = None

    def published_bytes(self):
        return sum(size for name, size in self.state["published"].items() if (self.store.root / "releases" / name).is_dir())

    def before_commit(self, candidate):

        # An unchanged file has another link in a previous release. Charge only the additions owned by this writer.
        size = sum(stat.st_size for p in candidate.rglob("*") if p.is_file() and (stat := p.stat()).st_nlink == 1)
        self.state["published"][candidate.name] = size
        write_json(self.account, self.state)
        self.check()

    def usage(self):
        native, cards = [], 0
        for path in self.root.glob("acquisition/sessions/*/session/native/execution.json"):
            execution = json.loads(path.read_bytes())
            charged = NativeAttempt.charges(execution["attempts"])
            native.append(charged)
            cards += len(execution["inputs"]["physical_devices"]) * charged.native_ns
        return Usage.total(native), cards

    def remaining(self, directory):
        used, card_ns = self.usage()
        execution = directory / "native/execution.json"
        current = json.loads(execution.read_bytes()) if execution.is_file() else None
        own = NativeAttempt.charges(current["attempts"]) if current else Usage()
        own_cards = own.native_ns * len(current["inputs"]["physical_devices"]) if current else 0

        # A resumed child receives its own historical charges plus the unused task allowance.
        return own.attempts + max(0, self.attempts - used.attempts), own_cards + max(0, self.cards - card_ns)

    def check(self):
        self.limit.check()
        if self.deadline is not None and time.monotonic_ns() >= self.deadline:
            raise AllowanceExhausted("cumulative postprocessing allowance exhausted")

    @contextmanager
    def processing(self):
        spent = self.state["processing_ns"]
        remaining = self.processing_ns - spent
        if remaining <= 0:
            raise AllowanceExhausted("cumulative postprocessing allowance exhausted")
        start = time.monotonic_ns()
        self.deadline = start + remaining

        # If the coordinator dies, the reserved time remains charged on resume.
        self.state["processing_ns"] = spent + remaining
        write_json(self.account, self.state)
        try:
            self.check()
            yield remaining / 1e9
            self.check()
        except (TimeoutError, subprocess.SubprocessError):
            self.check()
            raise
        finally:
            self.deadline = None
            self.state["processing_ns"] = spent + time.monotonic_ns() - start
            write_json(self.account, self.state)

    def document(self):
        native, cards = self.usage()
        return {**native.document(), "card_ns": cards, "postprocessing_ns": self.state["processing_ns"],
                "retained_bytes": self.limit.sample(), "published_bytes": self.published_bytes()}
