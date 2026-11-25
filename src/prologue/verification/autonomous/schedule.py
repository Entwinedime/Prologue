"""Find the first batch decision affected by autonomous execution."""

import argparse
import json
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from prologue.runtime.io import write_json

from ..core.actions import ActionIdentity
from ..core.context_values import ContextValues
from ..core.ingress import IngressEvidence
from ..core.native_case import NativeCase
from ..core.reference import ReferenceRecording


class ScheduleComparison:
    def __init__(self, reference):
        self.reference = reference
        self.ingress = IngressEvidence(reference.manifest["instances"], reference.submissions)
        self.native, self.predicted = defaultdict(list), defaultdict(list)
        self.native_releases, self.predicted_releases = defaultdict(list), defaultdict(list)

    @staticmethod
    def decision(work, entered, returned):
        return {"phase": work["phase"], "members": ActionIdentity.work_identity(work)["members"],
                "entered_ns": entered, "returned_ns": returned}

    def observe_native(self):
        releases = {"release_request", "release_pages"}
        for rank, (contents, raw, identities) in self.reference.sources.items():
            group = self.reference.groups[rank]
            self.ingress.native(self.reference.ranks[rank], raw)
            clock = self.reference.clocks[rank]
            for index, identity in identities.items():
                if identity["action"] != "run_batch":
                    continue
                scope = contents.graph["scopes"][index]
                ports = {port["name"]: port["event"] for port in scope["ports"]}
                observed = [clock.window(ports[name]) for name in ("call.enter", "call.return")]
                if any(window is None or window[0] != window[1] for window in observed):
                    raise ValueError(f"batch lacks measured host boundaries for {group}")
                entered, returned = (window[0] - self.ingress.origin_ns for window in observed)
                self.native[group].append(self.decision(scope["work"], entered, returned))
            history = self.reference.histories[group]
            for row in history["events"]:
                operation = row["operation"]
                if operation["kind"] in releases and not operation.get("grouped"):
                    self.native_releases[group].append(
                        {
                            "operation": operation["kind"],
                            "request": operation.get("request"),
                            "time_ns": row["end_raw_ns"] - self.ingress.origin_ns,
                        }
                    )

    def observe_prediction(self, trace):
        contexts, pending = ContextValues(), {}
        requests = sorted(self.reference.submissions[0].requests.values(), key=lambda request: request["arrival_ns"])
        complete = False
        with trace.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                kind = row["kind"]
                complete = kind == "end"
                if kind == "context_value":
                    contexts.declare(row)
                elif kind == "resources":
                    if row["requests"] != requests:
                        raise ValueError("autonomous trace has different workload requests")
                    self.ingress.observe(row)
                elif kind == "input_read":
                    row["identity"] = contexts.decode(row["identity"])
                    self.ingress.observe(row)
                elif kind == "graph":
                    scope = row["graph"]["scopes"][0]
                    if scope["role"] != "run_batch":
                        continue
                    work = contexts.decode(scope["work"])
                    group = self.reference.groups[work["rank"]]
                    stamp = {**self.decision(work, None, None), "submitted_ns": row["time_ns"]}
                    self.predicted[group].append(stamp)
                    ports = {port["name"]: row["event_ids"][port["event"]] for port in scope["ports"]}
                    pending[ports["call.enter"]] = stamp, "entered_ns"
                    pending[ports["call.return"]] = stamp, "returned_ns"
                elif kind == "completion" and row["event"] in pending:
                    stamp, boundary = pending.pop(row["event"])
                    stamp[boundary] = row["time_ns"]
                elif kind == "resource" and row["capacity_increased"]:
                    op = row["operation"]
                    self.predicted_releases[row["group"]].append({"operation": op["kind"], "request": op.get("request"),
                                                                 "time_ns": row["time_ns"]})
        if complete and pending:
            raise ValueError("complete execution omitted a batch return")
        return complete

    @staticmethod
    def preceding(rows, stamp):
        return next((row for row in reversed(rows) if row["time_ns"] <= stamp), None)

    def difference(self, group, left, right, previous):
        instance = group.rsplit("/", 1)[0]
        native_reads = self.ingress.native_reads[instance]
        predicted_reads = self.ingress.actual_reads[instance]
        readiness = self.ingress.readiness(instance)
        members = {row["id"] for batch in (left, right) if batch for row in batch["members"]}
        bounds = {request: readiness[request] for request in sorted(members)}
        native_input = self.preceding(native_reads, left["entered_ns"]) if left else None
        predicted_input = self.preceding(predicted_reads, right["submitted_ns"]) if right else None
        uncertain = [] if predicted_input is None else [
            name for name, window in bounds.items()
            if window["submitted_ns"] <= predicted_input["time_ns"] < window["received_ns"]
        ]
        return {
            "native_batch": left,
            "predicted_batch": right,
            "last_common_batch": previous,
            "native_input": native_input,
            "predicted_input": predicted_input,
            "input_availability_bounds": bounds,
            "readiness_unresolved_at_predicted_read": uncertain,
            "native_prior_release": self.preceding(self.native_releases[group], left["entered_ns"]) if left else None,
            "predicted_prior_release": self.preceding(self.predicted_releases[group], right["submitted_ns"])
            if right
            else None,
        }

    def run(self, trace):
        self.observe_native()
        complete = self.observe_prediction(trace)
        reports = {}
        for group in sorted(self.native.keys() | self.predicted.keys()):
            native, predicted = self.native[group], self.predicted[group]
            matched, difference, last = 0, None, None
            for left, right in zip_longest(native, predicted):
                if right is None and not complete:
                    break
                if (
                    left is None
                    or right is None
                    or (left["phase"], left["members"]) != (right["phase"], right["members"])
                ):
                    difference = self.difference(group, left, right, last)
                    break
                matched += 1
                last = {"native": left, "predicted": right}
            reports[group] = {"native_batches": len(native), "predicted_batches": len(predicted),
                              "matched_prefix": matched, "first_difference": difference}
        inputs = {}
        for instance, reads in self.ingress.native_reads.items():
            predicted = [row for row in self.ingress.actual_reads[instance] if row["in_flow"]]
            inputs[instance] = {
                "native_attempts": self.ingress.input_attempts[instance],
                "predicted_attempts": len(predicted),
                "predicted_empty_timed_polls": [
                    row["time_ns"] for row in predicted if not row["requests"] and row["poll_deadline_ns"] is not None
                ],
                "native_nonempty": [row for row in reads if row["requests"]],
                "predicted_nonempty": [row for row in predicted if row["requests"]],
                "availability_bounds": self.ingress.readiness(instance),
            }
        return {"prediction_complete": complete, "groups": reports, "inputs": inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "graphs", "execution", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    recording = ReferenceRecording(args.capture)
    reference = NativeCase.read(recording.source, args.graphs)
    target = {field: recording.inputs["job"][field] for field in recording.target_fields}
    if mismatch := recording.input_difference(recording.inputs["job"], target) or reference.input_difference(target):
        raise ValueError(f"native batch reference differs in {mismatch}")
    if reference.report(False)["status"] != "consistent":
        raise ValueError("native batch reference has unqualified observations")
    report = ScheduleComparison(reference).run(args.execution)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report, indent=2, replace=False)


if __name__ == "__main__":
    main()
