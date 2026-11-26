"""Resolve and qualify local evidence without giving an acquisition rule ownership of its budget."""

import subprocess
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from itertools import groupby
from typing import Protocol

from prologue.common.io import content_id, read_json, write_json
from prologue.host.acquisition.capture import NativeAttempt
from prologue.host.acquisition.options import LocalAcquisitionOptions, QualificationOptions
from prologue.host.acquisition.recording import LocalSession, PublishedCapture
from prologue.host.performance.requirements import require_work_interval
from prologue.host.runtime.limits import AllowanceExhausted, OperationLimit
from prologue.host.runtime.workspace import TaskWorkspace, file_lock


class EvidenceGap(Exception):
    """A valid query that this method cannot prove. Invalid evidence is a separate error."""

    def __init__(self, status, **details):
        super().__init__(status)
        self.report = {"status": status, **details}


@dataclass(frozen=True, slots=True)
class AcquisitionContext:
    """Read-only inputs for constructing and independently qualifying a native recipe."""

    options: LocalAcquisitionOptions | QualificationOptions
    release_id: str | None = None
    sampling_sources: Callable[[], Iterable[tuple[str, dict]]] | None = None


class EvidenceRule(Protocol):
    """A recipe contributes a match, a compatible batch key, a native job, and its own evidence qualification."""

    name: str
    priority: int

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
    """Try applicable methods in priority order and publish evidence from one qualifying recording."""

    def __init__(self, rules: Iterable[EvidenceRule], context: AcquisitionContext, placement_check, template_resources):
        ordered = sorted(rules, key=lambda rule: (-rule.priority, rule.name))
        self.rules = {rule.name: rule for rule in ordered}
        if len(self.rules) != len(ordered):
            raise ValueError("local acquisition rule names must be unique")
        if any(name in {".", ".."} or "/" in name or not name for name in self.rules):
            raise ValueError("local acquisition rule names must be path components")

        self.context = context
        self.placement_check = placement_check
        self.template_resources = template_resources

    def matching(self, query, deployment):
        if "work" in query:
            require_work_interval(query)
        self.placement_check(query, deployment)

        matches = [rule for rule in self.rules.values() if rule.matches(query)]
        if not matches:
            raise EvidenceGap("no_matching_rule", query=content_id(query))
        return matches

    @staticmethod
    def ranked(matches):
        """Reject ambiguity at the next priority tier only when that tier becomes eligible."""

        for priority, rules in groupby(matches, key=lambda rule: rule.priority):
            tier = list(rules)
            if len(tier) != 1:
                raise EvidenceGap("conflicting_rules", priority=priority, rules=[rule.name for rule in tier])
            yield tier[0]

    def session_key(self, query, deployment):
        rule = next(self.ranked(self.matching(query, deployment)))
        return rule.name, content_id(rule.session_key(query, deployment))

    def run(self, arguments, queries, deployment, limit=None, processing=nullcontext, *, available_queries=None):
        if not queries:
            raise ValueError("local acquisition needs at least one evidence query")

        session = LocalSession(arguments.output.resolve())
        with file_lock(session.directory / ".lock"):
            request = session.directory / "request.json"
            if available_queries is not None and request.is_file():

                # A fit can shrink demand while a claimed native session is still recoverable.
                available = {content_id(query): query for query in available_queries}
                keys = read_json(request)["queries"]
                if not available.keys() >= set(keys):
                    raise ValueError("the training source no longer supplies the original local queries")

                queries = [available[key] for key in keys]

            return self.observe(arguments, queries, deployment, session, limit, processing)

    @staticmethod
    def recording(rule, queries, keys, deployment, arguments, session, limit, processing):
        """Charge prior candidates before preparing this attempt. A completed raw recording can be requalified
        after a process restart.
        """

        directory = arguments.output
        if not (directory / "result.json").is_file():
            used, cards = session.charges(excluding=directory)
            remaining_cards = arguments.card_budget_ns
            if remaining_cards is not None:
                remaining_cards = max(0, remaining_cards - cards)

            remaining = replace(arguments, attempt_limit=max(0, arguments.attempt_limit - used.attempts),
                                card_budget_ns=remaining_cards,
                                budget_seconds=max(0.0, arguments.budget_seconds - used.native_ns / 1e9))
            with processing() as seconds:
                attempt = rule.prepare(queries, deployment, replace(remaining, preparation_seconds=seconds), limit)

            attempt.inputs.update(rule=rule.name, queries=keys)
            if remaining_cards is not None:
                attempt.limit_ns = remaining_cards // len(attempt.inputs["physical_devices"])
            attempt.run()

        return PublishedCapture(directory, conditions={"rule": rule.name, "queries": keys, "deployment": deployment})

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
        return self.products(self.rules[recording.inputs["rule"]], queries, recording)

    def candidates(self, queries, deployment):
        choices = [self.matching(query, deployment) for query in queries]
        preferred = {next(self.ranked(group)).name for group in choices}
        if len(preferred) != 1:
            raise EvidenceGap("different_sessions_required", rules=sorted(preferred))

        common = set.intersection(*({choice.name for choice in group} for group in choices))
        return self.ranked(rule for rule in self.rules.values() if rule.name in common)

    def attempt(self, rule, queries, keys, deployment, options, session, limit, processing):
        limit.check()
        try:
            recording = self.recording(rule, queries, keys, deployment, options, session, limit, processing)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            attempts = NativeAttempt.attempts(options.output)
            if not attempts or attempts[-1]["status"] != "failed":
                raise

            detail = (
                "native attempt exceeded its deadline"
                if isinstance(error, subprocess.TimeoutExpired)
                else f"native process exited with {error.returncode}"
            )

            return {"status": "native_failed", "detail": detail, "attempt": attempts[-1]["number"]}

        receipt = {}
        with processing() as seconds:
            products, answers = self.products(rule, queries, recording, seconds)
            for name, product in products.items():
                if required := product.pop("required_work", ()):
                    receipt.setdefault("required_work", {})[name] = required
                write_json(session.directory / name, product)

        status = "untriggered"
        if products:
            status = "qualified" if all(answer["status"] == "qualified" for answer in answers) else "incomplete"
        receipt.update(status=status, answers=answers, native_recording=recording.attempt_id,
                       image=recording.inputs["image"], physical_devices=recording.inputs["physical_devices"],
                       assets={name: content_id(product) for name, product in products.items()})
        limit.check()

        return receipt

    def collect(self, queries, keys, deployment, arguments, session, limit, processing):
        misses = []
        try:
            for index, rule in enumerate(self.candidates(queries, deployment)):
                native = session.native(rule.name, primary=index == 0)
                options = replace(arguments, output=native)
                try:
                    receipt = self.attempt(rule, queries, keys, deployment, options, session, limit, processing)
                except EvidenceGap as gap:
                    misses.append({"rule": rule.name, "priority": rule.priority, **gap.report})
                    continue

                receipt["rule"] = rule.name
                if index:
                    receipt["recording"] = str(native.relative_to(session.directory))
                break
            else:

                # Preserve the highest-priority method's explanation when none could reach the requested cut.
                receipt = dict(misses[0])
        except EvidenceGap as gap:
            receipt = gap.report
        except AllowanceExhausted as error:
            receipt = {"status": "allowance_exhausted", "detail": str(error)}

        if misses:
            receipt["fallback"] = misses
        return receipt

    def observe(self, arguments, queries, deployment, session, limit, processing):
        keys = [content_id(query) for query in queries]
        request = {"queries": keys, "deployment": deployment, "source_image": arguments.source_image}
        if self.context.release_id is not None:
            request["release"] = self.context.release_id

        root = session.directory
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

        limit = limit if limit is not None else OperationLimit(root, arguments.storage_bytes)
        receipt = self.collect(queries, keys, deployment, arguments, session, limit, processing)
        usage, card_ns = session.charges()
        retained = limit.sample()
        peak = limit.sampled_peak_bytes

        for path in session.executions():
            execution = read_json(path)
            if not execution["attempts"]:
                continue

            receipt.setdefault("physical_devices", execution["inputs"]["physical_devices"])

            # A recovered attempt can lack the storage measurement made by a live supervisor.
            peaks = [attempt.get("sampled_collection_peak_bytes") for attempt in execution["attempts"]]
            peak = None if peak is None or None in peaks else max(peak, *peaks)

        receipt.update(queries=keys, native_attempts=usage.attempts, native_charged_ns=usage.native_ns,
                       card_ns=card_ns, storage={"retained_bytes_before_receipt": retained, "sampled_peak_bytes": peak})
        write_json(result, receipt, indent=2)
        return receipt
