"""Shared causal indexing, evidence collection, trace reading and verification reports."""

import json
from collections import Counter, defaultdict

from prologue.common.io import read_json, write_json


class FindingLedger:
    """Count every finding while retaining only the requested diagnostic examples."""

    def __init__(self, limit=None):
        self.counts = Counter()
        self.samples = []
        self.limit = limit

    def add(self, category, work, detail):
        self.counts[category] += 1
        if self.limit is None or self.counts[category] <= self.limit:
            self.samples.append((category, work, detail))

    def __bool__(self):
        return bool(self.counts)

    def __iter__(self):
        return iter(self.samples)

    def collect(self, *streams):
        for stream in streams:

            # An upstream ledger retains complete counts even if it has already limited the examples.
            total = self.counts + stream.counts if isinstance(stream, FindingLedger) else None
            for category, work, detail in stream:
                self.add(category, work, detail)
            if total is not None:
                self.counts = total

    def examples(self, gaps):
        return [{"category": category, "kind": "evidence_gap" if category in gaps else "difference",
                 "work": work, **detail} for category, work, detail in self.samples]

    def outcome(self, gaps, *, incomplete, success):
        issues = {name for name, count in self.counts.items() if count}
        partial = bool(issues & gaps) or bool(incomplete)

        # Missing evidence in one check cannot discard a contradiction established by another.
        status = "differences" if issues - gaps else "incomplete" if partial else success
        return {"status": status, "coverage": "partial" if partial else "checked_scope"}


class TraceReader:
    """Read raw workers separately; decode complete predicted records in their declaration order."""

    def __init__(self, path):
        self.path = path
        self.complete = False

    @staticmethod
    def worker(capture, graphs, rank):
        return read_json(graphs / f"{rank}-graph.json"), read_json(capture / f"rank-{rank}.json")

    def __iter__(self):
        contexts = ContextValues()
        self.complete = False
        with self.path.open("rb") as stream:
            for line in stream:

                # A killed writer may leave an unterminated record after the last usable prefix.
                if not line.endswith(b"\n"):
                    self.complete = False
                    return
                row = json.loads(line)
                kind = row["kind"]
                self.complete = kind == "end"
                if kind == "context_value":
                    contexts.declare(row)
                    continue
                if kind == "graph":
                    for scope in row["graph"]["scopes"]:
                        scope["work"] = contexts.decode(scope["work"])
                elif kind == "input_read":
                    row["identity"] = contexts.decode(row["identity"])
                yield row


class ReportBuilder:
    """Publish a concise human report beside its complete machine-readable evidence."""

    def __init__(self, title):
        self.lines = [f"# {title}", ""]

    def note(self, text):
        self.lines.extend((text, ""))

    def table(self, columns, rows):
        def line(values):
            return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"

        self.lines.extend((line(columns), line(["---"] * len(columns))))
        self.lines.extend(line(row) for row in rows)
        self.lines.append("")

    def publish(self, output, name, document):
        write_json(output / f"{name}.json", document, indent=2, replace=False)
        (output / f"{name}.md").write_text("\n".join(self.lines) + "\n")


class ContextValues:
    snapshots = {"context", "entry_context", "exit_context", "observation_context"}
    frames = {"work", "continuation"}

    def __init__(self):
        self.values = []
        self.indices = {}

    @classmethod
    def transform(cls, frame, convert):
        if not isinstance(frame, dict):
            return frame
        mapped = frame.copy()
        for name, value in frame.items():
            if name in cls.snapshots and isinstance(value, dict):
                mapped[name] = {key: convert(field) for key, field in value.items()}
            elif name == "callers" and isinstance(value, list):
                mapped[name] = [cls.transform(caller, convert) for caller in value]
            elif name in cls.frames:
                mapped[name] = cls.transform(value, convert)
        return mapped

    def intern(self, value):
        key = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if key not in self.indices:
            self.indices[key] = len(self.values)
            self.values.append(value)
        return self.indices[key]

    def encode(self, frame):
        return self.transform(frame, self.intern)

    def declare(self, record):
        if type(record["index"]) is not int or record["index"] != len(self.values):
            raise ValueError("execution trace context values must be declared in reference order")
        self.values.append(record["value"])

    def resolve(self, index):
        if type(index) is not int or not 0 <= index < len(self.values):
            raise ValueError("execution context refers to an undeclared value")
        return self.values[index]

    def decode(self, frame):
        return self.transform(frame, self.resolve)


class BoundarySignature:
    def __init__(self, names, predecessors, work):
        self.names, self.predecessors, self.work = names, predecessors, work

    def explain_work(self, group):
        resource, work = group
        signature, count = min(work)
        scope, kind, *masks = signature
        result = {"resource": resource, "work_count": sum(count for _, count in work),
                  "example": {"scope": scope, "kind": kind, "count": count}}
        for field, mask in zip(("before_start", "before_finish", "after_start", "after_finish"), masks, strict=True):
            result["example"][field] = {"count": mask.bit_count(), "examples": self.labels(mask, 3)}
        return result

    def labels(self, mask, limit):
        result = []
        while mask and len(result) < limit:
            bit = mask & -mask
            result.append(self.names[bit.bit_length() - 1])
            mask ^= bit
        return result

    def differences(self, predicted):
        if self.names != predicted.names:
            yield "boundary_set", {"native_only": sorted(set(self.names) - set(predicted.names)),
                                   "predicted_only": sorted(set(predicted.names) - set(self.names))}
            return
        for target in self.names:
            native, estimate = self.predecessors[target], predicted.predecessors[target]
            if native != estimate:
                yield (
                    "boundary_order",
                    {
                        "target": target,
                        "missing_predecessors": (native & ~estimate).bit_count(),
                        "extra_predecessors": (estimate & ~native).bit_count(),
                        "missing_examples": self.labels(native & ~estimate, 3),
                        "extra_examples": self.labels(estimate & ~native, 3),
                    },
                )
        if self.work != predicted.work:
            missing, extra = self.work - predicted.work, predicted.work - self.work
            yield "work_placement", {
                "native_resource_groups": sum(missing.values()), "predicted_resource_groups": sum(extra.values()),
                "native_example": self.explain_work(next(iter(missing))) if missing else None,
                "predicted_example": predicted.explain_work(next(iter(extra))) if extra else None,
            }


class GraphProjection:
    def __init__(self, graph, anchors):
        self.graph = graph
        self.indices = {event: index for index, event in enumerate(anchors)}
        self.result = CausalIndex()
        self.result.add_events(range(len(anchors)))
        self.pending, self.live = {}, Counter()

    def release(self, labels):
        for label in labels:
            self.live[label] -= 1
            if not self.live[label]:
                del self.live[label]

    def receive(self, target, maximal, closure):
        other, previous = self.pending.get(target, (set(), set()))
        current = {label for label in closure if label in self.live}
        previous = {label for label in previous if label in self.live}
        merged = (maximal - previous) | (other - current) | (maximal & other)

        # Retain incoming references before releasing the old frontier; labels cannot transiently disappear.
        self.live.update(merged)
        self.release(other)
        self.pending[target] = merged, current | previous

    def run(self):
        for event, position in self.graph.positions.items():
            previous, closure = self.pending.pop(position, (set(), set()))
            closure = {label for label in closure if label in self.live}
            anchor = self.indices.get(event)
            maximal = previous
            if anchor is not None:
                for source in sorted(previous):
                    self.result.connect(source, anchor, "order")
                maximal = {anchor}
                self.live.update(maximal)
                closure = closure | maximal

            # Popped predecessors remain live until every successor has received this frontier.
            if maximal:
                for target in self.graph.successors[position]:
                    self.receive(target, maximal, closure)
            self.release(previous)
            if anchor is not None:
                self.release(maximal)
        self.result.prepare()
        return self.result


class CausalIndex:
    relations = ("order", "submission", "data", "signal", "control", "state", "communication", "observation")

    def __init__(self):
        self.next = {}

    @classmethod
    def observed(cls, graph):
        result = cls()
        result.add_events(range(len(graph["events"])))
        for edge in graph["edges"]:
            result.connect(edge["source"], edge["target"], edge["kind"])
        for work in graph["work"]:
            result.connect(work["start"], work["finish"], "order")
        result.prepare()
        return result

    @classmethod
    def joined(cls, graphs, dependencies):
        result = cls()
        result.next = {(rank, event): {(rank, target): relations for target, relations in targets.items()}
                       for rank, graph in graphs.items() for event, targets in graph.next.items()}
        for source, target, relation in dependencies:
            result.connect(source, target, relation)
        result.prepare()
        return result

    def projected(self, anchors):
        return GraphProjection(self, anchors).run()

    def measured_dependencies(self, measured):
        pending = {}
        for event in self.positions:
            predecessors = (event,) if event in measured else pending.pop(event, ())
            if not predecessors:
                continue

            # A measured intermediate already supplies its own clock constraints.
            for target in self.next[event]:
                if target in measured:
                    for source in predecessors:
                        yield source, target
                else:
                    pending.setdefault(target, set()).update(predecessors)

    def add_events(self, events):
        for event in events:
            if event in self.next:
                raise ValueError("execution repeats an event identity")
            self.next[event] = {}

    def connect(self, source, target, relation):
        if source not in self.next or target not in self.next:
            raise ValueError("dependency refers to an undeclared event")
        edges = self.next[source]
        edges[target] = edges.get(target, 0) | (1 << self.relations.index(relation))

    def prepare(self):
        pending = dict.fromkeys(self.next, 0)
        for targets in self.next.values():
            for target in targets:
                pending[target] += 1
        ordered = [event for event, count in pending.items() if not count]
        for event in ordered:
            for target in self.next[event]:
                pending[target] -= 1
                if not pending[target]:
                    ordered.append(target)
        if len(ordered) != len(self.next):
            raise ValueError("dependency graph contains a cycle")
        self.positions = {event: index for index, event in enumerate(ordered)}
        self.successors = [[self.positions[target] for target in self.next[event]] for event in ordered]

    def signature(self, action):
        names = sorted(action.events)
        anchors = [self.positions[action.events[name]] for name in names]
        work = [(identity, self.positions[start], self.positions[finish]) for identity, start, finish in action.work]
        positions = anchors + [event for _, start, finish in work for event in (start, finish)]
        begin, end = min(positions), max(positions) + 1
        # Paths can leave the action tree. Only the topological interval bounds the search.
        before = self.propagate(anchors, begin, end)
        after = self.propagate(anchors, begin, end, reverse=True)

        boundaries = {name: before[position - begin] for name, position in zip(names, anchors, strict=True)}
        placement = defaultdict(Counter)
        for (scope, kind, resource), start, finish in work:
            placement[resource][
                (
                    scope,
                    kind,
                    before[start - begin],
                    before[finish - begin],
                    after[start - begin],
                    after[finish - begin],
                )
            ] += 1

        # Private submission threads need no public port. Compare their work as resource groups, not by thread ID.
        placement = Counter(
            (action.resources[resource], frozenset(items.items())) for resource, items in placement.items()
        )
        return BoundarySignature(names, boundaries, placement)

    def reaches(self, source, target):
        start, finish = self.positions[source], self.positions[target]
        if start >= finish:
            return start == finish
        if target in self.next[source]:
            return True

        pending, visited = [start], {start}
        while pending:
            current = pending.pop()
            if current == finish:
                return True
            for successor in self.successors[current]:
                if successor <= finish and successor not in visited:
                    visited.add(successor)
                    pending.append(successor)
        return False

    def witness(self, source, target):
        pending, parents = [source], {source: None}
        finish = self.positions[target]
        for event in pending:
            if event == target:
                break
            for successor in sorted(self.next[event]):
                if self.positions[successor] <= finish and successor not in parents:
                    parents[successor] = event
                    pending.append(successor)
        if target not in parents:
            raise ValueError("reported ordering difference has no supporting path")

        counts, path = Counter(), []
        event = target
        while (previous := parents[event]) is not None:
            relations = [name for bit, name in enumerate(self.relations) if self.next[previous][event] & (1 << bit)]
            counts.update(relations)
            path.append({"source": previous, "target": event, "relations": relations})
            event = previous
        path.reverse()

        # One actual path explains the finding; it does not prove that other paths or state effects are absent.
        return {"edges": len(path), "relations": dict(counts), "first_edges": path[:3], "last_edges": path[-3:]}

    def propagate(self, positions, begin, end, *, reverse=False):
        bits = [0] * (end - begin)
        for index, position in enumerate(positions):
            bits[position - begin] |= 1 << index

        if reverse:
            for source in reversed(range(begin, end)):
                for target in self.successors[source]:
                    if target < end:
                        bits[source - begin] |= bits[target - begin]
        else:
            for source in range(begin, end):
                mask = bits[source - begin]
                if mask:
                    for target in self.successors[source]:
                        if target < end:
                            bits[target - begin] |= mask
        return bits

    def reachable_masks(self, origins):
        return self.propagate((self.positions[event] for event in origins), 0, len(self.positions))

    def frontiers(self, origins, events):
        result = {event: [0, 0] for event in events}

        # Limit propagation memory to one block; only inspected events retain masks for all origins.
        for begin in range(0, len(origins), 1024):
            positions = [self.positions[event] for event in origins[begin:begin + 1024]]
            for side in (0, 1):
                masks = self.propagate(positions, 0, len(self.positions), reverse=bool(side))
                for event, pair in result.items():
                    pair[side] |= masks[self.positions[event]] << begin
        return result
