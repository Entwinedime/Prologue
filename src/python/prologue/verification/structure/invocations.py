"""Pair native and simulated invocations from work identities and independently witnessed control decisions."""

import json
from collections import Counter, defaultdict

from prologue.verification.core.actions import ActionBoundaries, ActionIdentity, GraphContents
from prologue.verification.engine import ContextValues


class InvocationComparison:
    def __init__(self, case, finding, details):
        self.case, self.finding, self.details = case, finding, details
        self.expected = {}
        self.pending = []
        self.compared = self.matched = self.matched_dependencies = 0
        self.occurrences, self.full_occurrences = defaultdict(list), Counter()
        self.native_contexts = defaultdict(list)
        self.native_classes, self.predicted_classes = defaultdict(set), defaultdict(set)

        # Native and predicted work use the same local value table for comparison keys.
        self.keys = ContextValues()

        # Use the native ABI only when every declaration is host-only; predicted contents never select a source class.
        for source, _, identities in case.sources.values():
            for index, identity in identities.items():
                if "work" not in identity:
                    continue
                key = source.structure_key(index, identity, self.case.controls, self.keys)
                full_key = ActionIdentity.logical_key(identity, self.keys)
                self.occurrences[key].append(index)
                self.full_occurrences[full_key] += 1
                self.native_contexts[full_key, key].append(index)
                self.native_classes[full_key].add(key)
                self.expected.setdefault(key, (source, index, identity))
        # Compare contents independently of occurrence pairing, after native remote pools have their actual owners.
        self.contents = {
            key: source.snapshot(index, identity["group"]) for key, (source, index, identity) in self.expected.items()
        }
        self.varied = {
            key
            for key, indices in self.occurrences.items()
            if any(
                self.expected[key][0].snapshot(index, self.expected[key][2]["group"]) != self.contents[key]
                for index in indices[1:]
            )
        }
        self.predicted_counts, self.full_predicted_counts = Counter(), Counter()
        self.content_checks, self.content_matches, self.paired_counts = Counter(), Counter(), Counter()
        self.pairs, self.checks, self.signatures = {}, [], {}
        self.actions, self.repeated_groups = [], {}
        self.dependency_checks, self.dependency_matches = Counter(), Counter()

    def compare_contents(self, identity, actual, predicted):
        different = False
        for category, detail in GraphContents.differences(actual, predicted, self.details):
            self.finding(category, identity, detail)
            different = True
        return not different

    def observe(self, row, rank, port, target, conditioned_scope):
        graph, events = row["graph"], row["event_ids"]
        root = graph["scopes"][0]
        group = ActionIdentity.group(graph, self.case.ranks[rank])
        try:
            identity = ActionIdentity.record(root, group)
        except ValueError as error:
            identity = {"group": group, "event": events[port["event"]], "action": root["role"]}
            self.finding("missing_predicted_identity", identity, {"reason": str(error)})
        key = (
            target.structure_key(0, identity, self.case.controls, self.keys)
            if "work" in identity
            else ActionIdentity.structure_key(identity, self.keys)
        )
        self.predicted_counts[key] += 1
        full_key = ActionIdentity.logical_key(identity, self.keys)
        self.full_predicted_counts[full_key] += 1
        self.predicted_classes[full_key].add(key)
        target_key = group, events[port["event"]]

        # Run-local identities retain every writer and occurrence; cross-run pairing happens after multiplicity is
        # known.
        self.case.bindings.invocation(graph, identity, events, target_key)
        self.case.connections.predicted(target, identity, events, target_key)
        if key not in self.expected:
            self.finding("unmatched_prediction", identity, {})
            return

        # Equal bodies do not prove when a query ran, what it returned or which communication round it joined.
        snapshot = target.snapshot(0, group)
        content_match = None
        if key not in self.varied:
            self.content_checks[key] += 1
            content_match = self.compare_contents(identity, self.contents[key], snapshot)
            self.content_matches[key] += content_match
        boundary = ActionBoundaries(target, 0, events)
        self.pending.append((key, full_key, rank, identity, target_key, boundary, content_match,
                             snapshot if key in self.varied else None, conditioned_scope))

    def check_counts(self, ended, checked_prefix):
        for key, (_, _, identity) in self.expected.items():
            if not self.predicted_counts[key]:
                self.finding("unmatched_native" if ended else "native_tail_unexecuted", identity, {})
        for key, count in self.full_occurrences.items():
            if not self.full_predicted_counts[key] and any(
                self.predicted_counts[value] for value in self.native_classes[key]
            ):
                self.finding("unpaired_native_context", self.keys.decode(json.loads(key)), {"observed_calls": count})
            if count > 1 and not checked_prefix:
                self.finding("ambiguous_native_identity", self.keys.decode(json.loads(key)), {
                    "observed_calls": count, "predicted_calls": self.full_predicted_counts[key],
                    "reason": "complete logical context repeats; structure is checked independently",
                })
        for key, count in self.full_predicted_counts.items():
            if not self.full_occurrences[key] and self.predicted_classes[key] & self.expected.keys():
                self.finding(
                    "unpaired_predicted_context", self.keys.decode(json.loads(key)), {"predicted_calls": count}
                )
            if count > 1 and self.full_occurrences[key] <= 1 and not checked_prefix:
                self.finding(
                    "ambiguous_predicted_identity", self.keys.decode(json.loads(key)), {"predicted_calls": count}
                )

    def pair(self):
        for call in self.pending:
            key, full_key, rank, identity, target_key, predicted, content_match, snapshot, conditioned_scope = call
            source, index, _ = self.expected[key]

            # Index once by logical work and structure to resolve repeat occurrences.
            indices = self.native_contexts[full_key, key]
            unique = len(indices) == self.full_occurrences[full_key] == self.full_predicted_counts[full_key] == 1
            paired_index = conditioned_scope
            if paired_index is None and unique and ActionIdentity.occurrence_anchored(identity):
                paired_index = indices[0]
            paired = paired_index is not None
            if unique and not paired:
                self.finding(
                    "unpaired_native_context",
                    identity,
                    {
                        "observed_calls": 1,
                        "predicted_calls": 1,
                        "reason": (
                            "repeatable read or no request work anchor; common structure is checked without "
                            "occurrence pairing"
                        ),
                    },
                )

            # A shared structure is not an occurrence. Pair with its own source scope, never the class representative.
            native = ActionBoundaries(source, paired_index if paired else index).ranked(rank)
            predicted = predicted.ranked(rank)
            if paired:
                if content_match is None:
                    content_match = self.compare_contents(
                        identity, source.snapshot(paired_index, identity["group"]), snapshot
                    )
                    self.content_checks[key] += 1
                    self.content_matches[key] += content_match
            self.checks.append((key, rank, identity, native, predicted, paired))

            # Only occurrence pairs may identify external producers, query results and communication rounds.
            if not paired:
                repeated_key = rank, full_key, key
                if repeated_key not in self.repeated_groups:
                    self.repeated_groups[repeated_key] = (
                        identity,
                        [ActionBoundaries(source, index).ranked(rank) for index in indices],
                        [],
                    )
                self.repeated_groups[repeated_key][2].append(predicted)
                continue
            self.pairs[identity["group"], paired_index] = target_key
            self.paired_counts[key] += 1
            self.actions.append((identity, native, predicted))
            self.compared += 1
            self.matched += content_match
        self.pending.clear()

    def compare_boundaries(self, actual_graph, predicted_graph, ordering):
        for key, rank, identity, native, predicted, paired in self.checks:
            if key not in self.signatures:
                source, index, _ = self.expected[key]
                actual = actual_graph.signature(ActionBoundaries(source, index).ranked(rank))
                self.signatures[key] = actual

                # An alternate route through another rank can supply any local boundary dependency.
                for index in self.occurrences[key][1:]:
                    repeated = actual_graph.signature(ActionBoundaries(source, index).ranked(rank))
                    if next(actual.differences(repeated), None) is not None:
                        self.signatures[key] = None
                        break
            actual = self.signatures[key]
            if paired and actual is None:
                actual = actual_graph.signature(native)
            if actual is not None:
                self.dependency_checks[key] += 1
                different = False
                for category, detail in actual.differences(predicted_graph.signature(predicted)):
                    self.finding(category, identity, detail)
                    different = True
                self.dependency_matches[key] += not different
                self.matched_dependencies += paired and not different
        ordering.compare(actual_graph, predicted_graph, self.actions)
        ordering.compare_repeated(actual_graph, predicted_graph, self.actions, self.repeated_groups)

    def unpaired(self):
        return {"native": self.case.observed_actions - self.compared,
                "predicted": sum(self.predicted_counts.values()) - self.compared}

    def report(self):
        repeated = [key for key in self.expected if len(self.occurrences[key]) > 1 or self.predicted_counts[key] > 1]
        return {
            "native_actions_checked": self.case.observed_actions,
            "predicted_actions_checked": sum(self.predicted_counts.values()),
            "actions_compared": self.compared,
            "actions_contents_matched": self.matched,
            "actions_dependencies_matched": self.matched_dependencies,
            "action_checks": {
                "predicted_occurrences": sum(self.predicted_counts.values()),
                "contents": {
                    "checked": sum(self.content_checks.values()),
                    "matched": sum(self.content_matches.values()),
                },
                "dependencies": {
                    "checked": sum(self.dependency_checks.values()),
                    "matched": sum(self.dependency_matches.values()),
                },
            },
            "declared_control_classes": len(self.case.controls),
            "repeated_calls": [
                {
                    "work": self.expected[key][2],
                    "classification": "declared_control" if key in self.case.controls else "structural_class",
                    "paired_calls": self.paired_counts[key],
                    "observed_calls": len(self.occurrences[key]),
                    "predicted_calls": self.predicted_counts[key],
                    "source_contents_uniform": key not in self.varied,
                    "contents_checked": self.content_checks[key],
                    "contents_matched": self.content_matches[key],
                    "source_dependencies_uniform": self.signatures[key] is not None
                    if key in self.signatures
                    else None,
                    "dependencies_checked": self.dependency_checks[key],
                    "dependencies_matched": self.dependency_matches[key],
                }
                for key in repeated
            ],
        }
