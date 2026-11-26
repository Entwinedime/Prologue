"""Compare execution order only at qualified corresponding action boundaries."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from prologue.verification.core.action_identity import ActionIdentity
from prologue.verification.engine import FindingLedger


@dataclass(frozen=True, slots=True)
class RankedBoundaries:
    events: dict
    work: list
    resources: dict


class ActionBoundaries:
    def __init__(self, contents, root, event_ids=None):
        graph = contents.graph
        event_ids = range(len(graph["events"])) if event_ids is None else event_ids
        paths, resources = contents.paths(root), {}
        self.events, self.work, self.resources = {}, [], {}

        for scope, path in paths.items():
            node = graph["scopes"][scope]
            boundaries = {}
            for port in sorted(node["ports"], key=lambda value: (value["name"], value["direction"])):

                # Generated action frontiers have no source port; their edges still participate in reachability.
                if scope == root and port["name"].startswith(("resource.enter/", "resource.exit/")):
                    continue
                name = f"{path}/{port['direction']}:{port['name']}"
                boundaries[name] = event_ids[port["event"]]
                resource = graph["events"][port["event"]]["resource"]
                resources.setdefault(resource, len(resources))

                if "wait" in port:
                    boundaries[name + "/wait.begin"] = event_ids[port["wait"]["begin"]]
                    resources.setdefault(port["wait"]["resource"], len(resources))
                if "rendezvous" in port:
                    boundaries[name + "/rendezvous"] = event_ids[port["rendezvous"]]

            for field in ("kv_accesses", "slot_accesses"):
                for index, access in enumerate(node[field]):
                    for side in ("start", "finish"):
                        boundaries[f"{path}/{field}/{index}/{side}"] = event_ids[access[side]]

            self.events.update(boundaries)

        for scope, path in paths.items():
            for work in contents.work[scope]:
                resource = work["resource"]
                descriptor = contents.resource(resource)
                self.resources[resource] = (*descriptor.values(), resources.get(resource))
                identity = (path, work["kind"], resource)
                self.work.append((identity, event_ids[work["start"]], event_ids[work["finish"]]))

    def ranked(self, rank):
        return RankedBoundaries(
            events={name: (rank, event) for name, event in self.events.items()},
            work=[(identity, (rank, start), (rank, finish)) for identity, start, finish in self.work],
            resources=self.resources,
        )


class ActionOrderComparison:
    def __init__(self, contexts, details=False):
        self.contexts, self.details = contexts, details
        self.counts, self.findings = Counter(), FindingLedger()
        self.repeated_counts = Counter()
        self.qualified = False


    def compare_repeated(self, actual, predicted, actions, repeated):
        if not repeated:
            return
        names = ("root/input:call.enter", "root/output:call.return")
        anchors = [(native.events[name], target.events[name]) for _, native, target in actions for name in names]

        if not anchors:
            self.repeated_counts["unqualified_contexts"] += len(repeated)
            self.repeated_counts["unqualified_calls"] += sum(len(targets) for _, _, targets in repeated.values())
            return

        frontiers = []
        for side, graph in enumerate((actual, predicted)):
            events = set()
            for row in repeated.values():
                for call in row[side + 1]:
                    events.update(call.events[name] for name in names)

            frontiers.append(graph.frontiers([row[side] for row in anchors], events))

        for identity, native, targets in repeated.values():
            if not native:
                self.repeated_counts.update(unqualified_contexts=1, unqualified_calls=len(targets))
                continue

            positions = [{tuple(value for name in names for value in masks[call.events[name]]) for call in calls}
                         for masks, calls in zip(frontiers, (native, targets), strict=True)]
            missing, extra = positions[0] - positions[1], positions[1] - positions[0]
            self.repeated_counts.update(
                contexts_checked=1,
                native_positions=len(positions[0]),
                predicted_positions=len(positions[1]),
                matched_positions=len(positions[0] & positions[1]),
                calls_checked=len(targets),
            )

            if not missing and not extra:
                self.repeated_counts.update(contexts_matched=1, calls_matched=len(targets))
                continue

            # Compare whole position sets. No member is selected to supply a source occurrence, body or duration.
            examples = {}
            for label, differences, calls, masks in (
                ("native", missing, native, frontiers[0]),
                ("predicted", extra, targets, frontiers[1]),
            ):
                for call in calls:
                    signature = tuple(value for name in names for value in masks[call.events[name]])
                    if signature in differences:
                        examples[label] = {"events": {name: call.events[name] for name in names},
                                           "anchor_counts": [mask.bit_count() for mask in signature]}
                        break

            self.findings.add("repeated_call_order", identity, {"native_only_positions": len(missing),
                                  "predicted_only_positions": len(extra), "examples": examples})

    def compare(self, actual, predicted, actions):
        labels, identities = defaultdict(list), {}
        for identity, native, estimate in actions:
            if native.events.keys() != estimate.events.keys():
                self.findings.add("ordering_evidence", identity, {"reason": "action boundary sets differ"})
                return

            # Occurrence pairs are already established; repeated work still needs distinct order labels.
            key = ActionIdentity.logical_key(identity, self.contexts), native.events["root/input:call.enter"]
            identities[key] = identity
            for name, event in native.events.items():
                labels[event, estimate.events[name]].append((key, name))

        pairs = list(labels)
        self.counts.update(actions_compared=len(identities), boundaries_compared=sum(map(len, labels.values())),
                           paired_events=len(pairs))
        if not pairs:
            return

        # Equal ports may alias one event; their partitions must agree before comparing partial orders.
        for side, origin in enumerate(("native", "predicted")):
            aliases = defaultdict(list)
            for pair in pairs:
                aliases[pair[side]].append(pair)

            for event, entries in aliases.items():
                if len(entries) == 1:
                    continue
                self.counts["alias_conflicts"] += 1
                key, _ = labels[entries[0]][0]
                self.findings.add("action_order_alias", identities[key], {
                    "origin": origin, "event": event, "opposite_events": len(entries),
                    "reason": "named boundaries alias one event in only one run",
                })

        if self.findings:
            return

        projections = [
            graph.projected([pair[side] for pair in pairs]) for side, graph in enumerate((actual, predicted))
        ]
        covers = [
            {(source, target) for source, targets in graph.next.items() for target in targets} for graph in projections
        ]
        names = [min(labels[pair]) for pair in pairs]
        differences = []

        for side, kind in enumerate(("missing", "extra")):
            other = projections[1 - side]
            self.counts[("native" if side == 0 else "predicted") + "_relations"] = len(covers[side])
            for source, target in covers[side]:
                if other.reaches(source, target):
                    continue
                reverse = other.reaches(target, source)
                differences.append((side, source, target, reverse))
                self.counts[kind + "_relations"] += 1
                self.counts[kind + "_reversed_relations"] += reverse
                self.counts[kind + "_cross_rank_relations"] += (
                    identities[names[source][0]]["group"] != identities[names[target][0]]["group"]
                )

        self.qualified = True

        # Covers generate the whole order. Checking both directions preserves strict reachability equality.
        # The first native topological position is a representative, not a unique first cause under concurrency.
        differences.sort(key=lambda item: (actual.positions[pairs[item[2]][0]], item[0], item[1]))
        shown = set()
        for side, source, target, reverse in differences:
            source_key, source_name = names[source]
            target_key, target_name = names[target]
            kind = "missing" if side == 0 else "extra"
            detail = {"source": {"work": identities[source_key], "boundary": source_name}, "boundary": target_name,
                      "opposite_order_reversed": reverse}

            if self.details or kind not in shown:
                detail["path"] = (actual, predicted)[side].witness(pairs[source][side], pairs[target][side])
                shown.add(kind)
            self.findings.add(kind + "_order", identities[target_key], detail)
