"""Run one local native session and answer its original evidence queries."""

import fcntl
import json
import subprocess
from argparse import Namespace
from hashlib import sha256
from typing import Protocol

from prologue.acquisition.capture import NativeAttempt
from prologue.acquisition.recording import PublishedCapture
from prologue.artifacts import write_json
from prologue.performance.requirements import require_work_interval
from prologue.workflow.limits import AllowanceExhausted, OperationLimit


class EvidenceGap(Exception):
    def __init__(self, status, **details):
        super().__init__(status)
        self.report = {"status": status, **details}


class EvidenceRule(Protocol):
    name: str

    def matches(self, query: dict) -> bool: ...
    def prepare(self, queries: list[dict], deployment: dict, arguments: Namespace, limit: OperationLimit) -> NativeAttempt: ...
    def qualify(self, queries: list[dict], recording: PublishedCapture) -> tuple[dict, list[dict]]: ...


def missing_root(query):
    cut = query.get("cut", {})
    return (query.get("status") == "no_declared_candidates"
            or (query.get("status") == "unmatched_conditions" and cut.get("cause") == "guard" and cut["path"] == []))


class LocalAcquisition:
    """Adapters select a cut, prepare a bounded native attempt, and qualify its readback."""

    def __init__(self, rules: tuple[EvidenceRule, ...], placement_check, release_id=None):
        self.rules = rules
        self.placement_check = placement_check
        self.release_id = release_id

    def select(self, query, deployment):
        if "work" in query:
            require_work_interval(query)
        self.placement_check(query, deployment)
        matches = [rule for rule in self.rules if rule.matches(query)]
        if not matches:
            raise EvidenceGap("no_matching_rule")
        if len(matches) != 1:
            raise EvidenceGap("conflicting_rules", rules=[rule.name for rule in matches])
        return matches[0]

    def run(self, arguments, queries, deployment):
        if not queries:
            raise ValueError("local acquisition needs at least one evidence query")
        root = arguments.output.resolve()
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self.observe(arguments, queries, deployment, root)

    @staticmethod
    def identity(query):
        return sha256(json.dumps(query, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    @staticmethod
    def recording(rule, queries, keys, deployment, arguments, limit):
        directory = arguments.output
        if not (directory / "result.json").is_file():
            attempt = rule.prepare(queries, deployment, arguments, limit)
            attempt.inputs.update(rule=rule.name, queries=keys)
            attempt.run()
        recording = PublishedCapture(directory)
        if (recording.inputs["rule"] != rule.name or recording.inputs["queries"] != keys
                or recording.inputs["deployment"] != deployment):
            raise ValueError("native recording belongs to different queries, rule or deployment")
        return recording

    @staticmethod
    def products(rule, queries, recording):
        products, answers = rule.qualify(queries, recording)
        if len(answers) != len(queries):
            raise ValueError("local rule must answer each original query")
        assets = {answer["asset"] for answer in answers if answer["status"] == "qualified"}
        if assets != products.keys():
            raise ValueError("qualified queries must refer to exactly the delivered local evidence")
        sources = [recording.inputs["template_recording"]] if "template_recording" in recording.inputs else []
        for product in products.values():
            product.update(acquisition_rule=rule.name, native_recordings=[*sources, recording.attempt_id])
        return products, answers

    def observe(self, arguments, queries, deployment, root):
        keys = [self.identity(q) for q in queries]
        request = {"queries": keys, "deployment": deployment, "source_image": arguments.source_image}
        if self.release_id is not None:
            request["release"] = self.release_id
        state = root / "request.json"
        if state.exists():
            if json.loads(state.read_bytes()) != request:
                raise ValueError("local session belongs to different queries, placement or image")
        else:
            if any(path.name != ".lock" for path in root.iterdir()):
                raise ValueError("new local session requires an empty output directory")
            write_json(state, request, replace=False)

        result = root / "result.json"
        if result.exists():
            receipt = json.loads(result.read_bytes())
            assets = (root / row["asset"] for row in receipt.get("answers", ()) if "asset" in row)
            if receipt["status"] not in {"allowance_exhausted", "native_failed"} and all(path.is_file() for path in assets):
                return receipt

        native = root / "native"
        receipt = {}
        limit = OperationLimit(root, arguments.storage_bytes)
        try:
            selected = []
            for key, query in zip(keys, queries, strict=True):
                try:
                    selected.append(self.select(query, deployment))
                except EvidenceGap as gap:
                    gap.report["query"] = key
                    raise
            rule = selected[0]
            if any(candidate is not rule for candidate in selected):
                raise EvidenceGap("different_sessions_required", rules=sorted({r.name for r in selected}))
            receipt["rule"] = rule.name
            limit.check()

            # NativeAttempt owns the bounded process, recovery, physical placement receipt and teardown.
            child = Namespace(**(vars(arguments) | {"output": native}))
            recording = self.recording(rule, queries, keys, deployment, child, limit)
            products, answers = self.products(rule, queries, recording)
            for name, product in products.items():
                write_json(root / name, product)
            status = "untriggered"
            if products:
                status = "qualified" if all(answer["status"] == "qualified" for answer in answers) else "incomplete"
            receipt.update(status=status, answers=answers, native_recording=recording.attempt_id, image=recording.inputs["image"])
            limit.check()
        except EvidenceGap as gap:
            receipt.update(gap.report)
        except AllowanceExhausted as error:
            receipt.update(status="allowance_exhausted", detail=str(error))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            attempts = NativeAttempt.attempts(native)
            if not attempts or attempts[-1]["status"] != "failed":
                raise
            receipt.update(status="native_failed", detail=str(error), attempt=attempts[-1]["number"])

        state = native / "execution.json"
        execution = json.loads(state.read_bytes()) if state.is_file() else {}
        attempts = execution.get("attempts", [])
        usage = NativeAttempt.charges(attempts)
        retained = limit.sample()
        peak = limit.sampled_peak_bytes
        if attempts:
            receipt["physical_devices"] = execution["inputs"]["physical_devices"]

            # Offline qualification cannot reconstruct an unmonitored native attempt's peak.
            peaks = [attempt.get("sampled_collection_peak_bytes") for attempt in attempts]
            peak = None if None in peaks else max(peak, *peaks)
        receipt.update(queries=keys, native_attempts=usage.attempts, native_charged_ns=usage.native_ns,
                       card_ns=usage.native_ns * len(receipt.get("physical_devices", ())),
                       storage={"retained_bytes_before_receipt": retained, "sampled_peak_bytes": peak})
        write_json(result, receipt, indent=2)
        return receipt
