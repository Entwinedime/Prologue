"""Compare given-action expansion with observed local contents, separately from control and execution."""

import json
from collections import Counter

from prologue.verification.core.actions import ActionBoundaries, ActionIdentity, GraphContents
from prologue.verification.core.source import NativeCase
from prologue.verification.engine import CausalIndex


class LocalActionComparison:
    def __init__(self):
        self.counts, self.findings = Counter(), []

    def finding(self, category, rank, scope, detail):
        self.counts[category] += 1
        if self.counts[category] <= 3:
            self.findings.append({"category": category, "rank": rank, "scope": scope, **detail})

    def intrinsic(self, dependencies, graph, paths, retained):
        waits = {(port["event"], port["wait"]["resource"])
                 for scope in paths for port in graph["scopes"][scope]["ports"] if port.get("wait")}
        signal = 1 << CausalIndex.relations.index("signal")
        local = CausalIndex()
        local.add_events(retained)
        for source in retained:
            for target, relations in dependencies.next[source].items():
                if target not in retained:
                    continue

                # A target invocation supplies its actual stream frontier; all other local relations remain checked.
                if relations & signal and (target, graph["events"][source]["resource"]) in waits:
                    relations &= ~signal
                    self.counts["stream_frontier_relations_deferred"] += 1
                if relations:
                    local.next[source][target] = relations

        # Serial resources retain their native order across omitted foreign cuts; no cross-resource order is inferred.
        ordered = sorted((graph["events"][event]["resource"], graph["evidence"]["events"][event]["order"], event)
                         for event in retained if graph["resources"][graph["events"][event]["resource"]]["serial"])
        previous = {}
        for resource, _, event in ordered:
            if resource in previous:
                local.connect(previous[resource], event, "order")
            previous[resource] = event
        local.prepare()
        return local

    def compare(self, case, rank, scope, predicted):
        source = case.sources[rank][0]
        group = case.groups[rank]
        target = GraphContents(predicted)
        target.pools = source.pools
        differences = list(GraphContents.differences(source.snapshot(scope, group), target.snapshot(0, group)))
        self.counts["contents_checked"] += 1
        self.counts["contents_matched"] += not differences
        for category, detail in differences:
            self.finding(category, rank, scope, detail)

        paths = source.paths(scope)
        native = ActionBoundaries(source, scope)
        generated = ActionBoundaries(target, 0)
        retained = {event for scope in paths for event in source.events[scope]}
        retained.update(native.events.values())
        retained.update(event for _, start, finish in native.work for event in (start, finish))
        local = self.intrinsic(case.dependencies[rank], source.graph, paths, retained)
        expected = local.signature(native)
        actual = CausalIndex.observed(predicted).signature(generated)
        changes = list(expected.differences(actual))
        self.counts["dependencies_checked"] += 1
        self.counts["dependencies_matched"] += not changes
        for category, detail in changes:
            self.finding(category, rank, scope, detail)


class LocalComparison(LocalActionComparison):
    def __init__(self, capture, graphs, expanded, output):
        super().__init__()
        self.capture, self.graphs, self.expanded, self.output = capture, graphs, expanded, output

    def worker(self, case, rank):
        expected = case.sources[rank][2]
        seen, built, summary = set(), 0, None
        with (self.expanded / f"rank-{rank}.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                if summary is not None:
                    raise ValueError("local graph output continues after its summary")
                if record["kind"] == "local_summary":
                    summary = record
                    continue
                scope = record["source_scope"]
                if record["kind"] != "local_graph" or type(scope) is not int or scope not in expected or scope in seen:
                    raise ValueError("local output has an unknown or duplicate action")
                seen.add(scope)
                if "error" in record:
                    self.finding("generation", rank, scope, {"reason": record["error"]})
                    continue
                built += 1
                graph = record["graph"]
                supplied = case.sources[rank][0].graph["scopes"][scope]
                group = case.groups[rank]
                if ActionIdentity.record(supplied, group) != ActionIdentity.record(graph["scopes"][0], group):
                    self.finding(
                        "action_input", rank, scope, {"reason": "generated root changes the given action metadata"}
                    )
                    continue
                self.compare(case, rank, scope, graph)
        if summary is None or summary["attempted"] != len(seen) or summary["built"] != built or seen != expected.keys():
            raise ValueError("local output is incomplete or omits actual actions")
        self.counts["given_actions"] += len(seen)
        self.counts["built_actions"] += built

    def run(self):
        self.output.mkdir(parents=True)
        report = {"status": "unverified", "capture": str(self.capture.resolve()), "graphs": str(self.graphs.resolve()),
                  "expanded": str(self.expanded.resolve())}
        try:
            case = NativeCase.read(self.capture, self.graphs)
            report["native_evidence"] = case.report(False)
            if report["native_evidence"]["status"] == "differences":
                raise ValueError("observed case has confirmed contradictions")
            for rank in sorted(case.observed_roles):
                try:
                    self.worker(case, rank)
                except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                    self.finding("output_evidence", rank, None, {"reason": str(error)})
            issues = self.counts.keys() - {
                "given_actions", "built_actions", "contents_checked", "contents_matched", "dependencies_checked",
                "dependencies_matched", "stream_frontier_relations_deferred",
            }
            report["status"] = (
                "differences" if issues - {"output_evidence"} else "incomplete" if issues else "local_actions_matched"
            )
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            report["error"] = str(error)
        report.update(
            counts=dict(self.counts),
            findings=self.findings,
            scope=(
                "Given action metadata, local contents and intrinsic dependencies only. Symbolic external "
                "values/pools/signals; no cross-graph binding, dynamic stream frontier, scheduling, execution time "
                "or held-out generalization claim."
            ),
        )
        (self.output / "local-graphs.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        text = [
            "# Local Graph Expansion Check",
            "",
            f"Result `{report['status']}`.",
            "",
            f"Expanded {self.counts['built_actions']} of {self.counts['given_actions']} supplied native actions.",
            f"Matched contents for {self.counts['contents_matched']} of {self.counts['contents_checked']} checks "
            "and local "
            f"dependencies for {self.counts['dependencies_matched']} of {self.counts['dependencies_checked']} checks.",
            f"Execution checks cover the {self.counts['stream_frontier_relations_deferred']} deferred dynamic "
            "stream dependencies.",
            "",
            "This check expands source templates for supplied native actions, with symbolic bindings for external "
            "values, pools, and signals. "
            "Controller behavior, cross-graph dependencies, timing, and held-out generalization require separate "
            "evidence.",
            "",
        ]
        (self.output / "local-graphs.md").write_text("\n".join(text))
        return 0 if report["status"] == "local_actions_matched" else 1
