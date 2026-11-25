"""Run one local native session and answer its original evidence queries."""

import subprocess
from contextlib import nullcontext
from dataclasses import replace
from typing import Protocol

from prologue.common.io import content_id, read_json, write_json
from prologue.host.acquisition.capture import NativeAttempt
from prologue.host.acquisition.options import LocalAcquisitionOptions
from prologue.host.acquisition.recording import PublishedCapture
from prologue.host.performance.requirements import require_work_interval
from prologue.host.runtime.limits import AllowanceExhausted, OperationLimit
from prologue.host.runtime.workspace import TaskWorkspace, file_lock


class EvidenceGap(Exception):
    def __init__(self, status, **details):
        super().__init__(status)
        self.report = {"status": status, **details}


class EvidenceRule(Protocol):
    name: str

    def matches(self, query: dict) -> bool: ...
    def session_key(self, query: dict, deployment: dict): ...
    def prepare(
        self, queries: list[dict], deployment: dict, arguments: LocalAcquisitionOptions, limit: OperationLimit
    ) -> NativeAttempt: ...
    def qualify(self, queries: list[dict], recording: PublishedCapture, timeout=None) -> tuple[dict, list[dict]]: ...


def missing_root(query):
    cut = query.get("cut", {})
    return (query.get("status") == "no_declared_candidates"
            or (query.get("status") == "unmatched_conditions" and cut.get("cause") == "guard" and cut["path"] == []))


class LocalAcquisition:
    """Adapters select a cut, prepare a bounded native attempt, and qualify its readback."""

    def __init__(self, rules: tuple[EvidenceRule, ...], placement_check, template_resources, release_id=None):
        self.rules = rules
        self.placement_check = placement_check
        self.template_resources = template_resources
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

    def run(self, arguments, queries, deployment, limit=None, processing=nullcontext, *, available_queries=None):
        if not queries:
            raise ValueError("local acquisition needs at least one evidence query")
        root = arguments.output.resolve()
        with file_lock(root / ".lock"):
            request = root / "request.json"
            if available_queries is not None and request.is_file():

                # A fit can shrink current demand while a previously claimed native session is still recoverable.
                available = {content_id(query): query for query in available_queries}
                keys = read_json(request)["queries"]
                if not available.keys() >= set(keys):
                    raise ValueError("the training source no longer supplies the original local queries")
                queries = [available[key] for key in keys]
            return self.observe(arguments, queries, deployment, root, limit, processing)

    @staticmethod
    def recording(rule, queries, keys, deployment, arguments, limit, processing=nullcontext):
        directory = arguments.output
        if not (directory / "result.json").is_file():
            with processing() as seconds:
                attempt = rule.prepare(queries, deployment, replace(arguments, preparation_seconds=seconds), limit)
            attempt.inputs.update(rule=rule.name, queries=keys)
            if arguments.card_budget_ns is not None:
                attempt.limit_ns = arguments.card_budget_ns // len(attempt.inputs["physical_devices"])
            attempt.run()
        recording = PublishedCapture(directory)
        if (recording.inputs["rule"] != rule.name or recording.inputs["queries"] != keys
                or recording.inputs["deployment"] != deployment):
            raise ValueError("native recording belongs to different queries, rule or deployment")
        return recording

    @staticmethod
    def products(rule, queries, recording, timeout=None):
        products, answers = rule.qualify(queries, recording, timeout)
        if len(answers) != len(queries):
            raise ValueError("local rule must answer each original query")
        assets = {name for answer in answers if answer["status"] == "qualified"
                  for name in (answer["asset"], *answer.get("support", ()))}
        if assets != products.keys():
            raise ValueError("qualified queries must refer to exactly the delivered local evidence")
        sources = [recording.inputs["template_recording"]] if "template_recording" in recording.inputs else []
        for product in products.values():
            product.update(acquisition_rule=rule.name, native_recordings=[*sources, recording.attempt_id])
        return products, answers

    def requalify(self, recording, queries):
        rule, = (rule for rule in self.rules if rule.name == recording.inputs["rule"])
        return self.products(rule, queries, recording)

    def observe(self, arguments, queries, deployment, root, limit, processing):
        keys = [content_id(query) for query in queries]
        request = {"queries": keys, "deployment": deployment, "source_image": arguments.source_image}
        if self.release_id is not None:
            request["release"] = self.release_id
        state = root / "request.json"
        with TaskWorkspace(root).initialize(state.name) as draft:
            if draft is not None:
                write_json(draft / state.name, request, replace=False)
            elif read_json(state) != request:
                raise ValueError("local session belongs to different queries, placement or image")

        result = root / "result.json"
        if result.exists():
            receipt = read_json(result)
            assets = (root / row["asset"] for row in receipt.get("answers", ()) if "asset" in row)
            if receipt["status"] not in {"allowance_exhausted", "native_failed", "untriggered"} and all(
                path.is_file() for path in assets
            ):
                return receipt

        native = root / "native"
        receipt = {}
        if limit is None:
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
            child = replace(arguments, output=native)
            recording = self.recording(rule, queries, keys, deployment, child, limit, processing)
            with processing() as seconds:
                products, answers = self.products(rule, queries, recording, seconds)
                for name, product in products.items():
                    if required := product.pop("required_work", ()):
                        receipt.setdefault("required_work", {})[name] = required
                    write_json(root / name, product)
            status = "untriggered"
            if products:
                status = "qualified" if all(answer["status"] == "qualified" for answer in answers) else "incomplete"
            receipt.update(
                status=status,
                answers=answers,
                native_recording=recording.attempt_id,
                image=recording.inputs["image"],
                assets={name: content_id(product) for name, product in products.items()},
            )
            limit.check()
        except EvidenceGap as gap:
            receipt.update(gap.report)
        except AllowanceExhausted as error:
            receipt.update(status="allowance_exhausted", detail=str(error))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            attempts = NativeAttempt.attempts(native)
            if not attempts or attempts[-1]["status"] != "failed":
                raise
            detail = (
                "native attempt exceeded its deadline"
                if isinstance(error, subprocess.TimeoutExpired)
                else f"native process exited with {error.returncode}"
            )
            receipt.update(status="native_failed", detail=detail, attempt=attempts[-1]["number"])

        state = native / "execution.json"
        execution = read_json(state) if state.is_file() else {}
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
