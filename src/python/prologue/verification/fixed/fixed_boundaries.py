"""Qualify fixed-graph completion windows and find the first drift on native boundaries."""

from collections import Counter, defaultdict
from itertools import chain


def readiness_parents(target, predecessors, joins):
    pending = list(predecessors[target])
    seen, physical = set(), set()

    # Follow the actual prerequisites of untimed joins while retaining their edges in the replay.
    for parent in pending:
        if parent in seen:
            continue
        seen.add(parent)

        if parent in joins:
            pending.extend(predecessors[parent])
        else:
            physical.add(parent)

    return physical


class FixedBoundaries:
    """Compare recomputed times against complete or bounded independent native observations."""

    def __init__(
        self, case, origin_ns, observer_joins, elided_waits, cpu_points, external_cpu_points, predecessors, gaps
    ):
        self.case, self.origin_ns = case, origin_ns
        self.observer_joins, self.elided_waits = observer_joins, elided_waits
        self.cpu_points, self.external_cpu_points = cpu_points, external_cpu_points
        self.gaps = gaps
        self.windows = self.completion_windows(predecessors)

    def completion_windows(self, predecessors):
        graph = self.case.graph
        windows = {}
        for rank, event in graph.positions:
            key = rank, event
            if event >= 0:
                observed = self.case.clocks[rank].window(event)
            else:
                stamp = self.case.cpu.native_times[key]
                observed = stamp, stamp

            if key not in self.observer_joins:
                windows[key] = observed
                continue

            # At a logical join the latest original prerequisite makes the result ready.
            parents = readiness_parents(key, predecessors, self.elided_waits)
            sources = [windows[parent] for parent in parents]
            if observed is None or not sources or None in sources:
                self.gaps.setdefault("unqualified_observer_joins", []).append(key)
                windows[key] = None
                continue

            first = max(begin for begin, _ in sources)
            last = min(observed[1], max(end for _, end in sources))
            if first > last:
                raise ValueError("observer marker precedes a verified join prerequisite")

            windows[key] = first, last

        return list(windows.values())

    def check_baseline(self, replayed):
        checked, max_clock_width = 0, 0
        evidence, differences = Counter(), Counter()
        first = None

        for (rank, event), position in self.case.graph.positions.items():
            if (rank, event) in self.external_cpu_points:
                evidence["external_protocol_condition"] += 1
                continue

            bounds = self.windows[position]
            if bounds is None:
                evidence["unmeasured"] += 1
                continue

            checked += 1
            time = replayed[position]
            begin, end = (stamp - self.origin_ns for stamp in bounds)
            if (rank, event) in self.observer_joins:
                evidence["derived_join"] += 1
            elif event < 0:
                evidence["host_protocol_raw"] += 1
            elif self.case.clocks[rank].device[event]:
                evidence["calibrated_device"] += 1
                max_clock_width = max(max_clock_width, end - begin)
            else:
                evidence["host_raw"] += 1

            if begin <= time <= end:
                continue
            differences["early" if time < begin else "late"] += 1
            if first is None or begin < first["measured_window_ns"][0]:
                call = self.cpu_points.get((rank, event))
                scope = (
                    {"cpu_protocol": call.source()}
                    if call
                    else self.case.sources[rank][0].graph["events"][event]["scope"]
                )
                first = {
                    "rank": rank,
                    "event": event,
                    "scope": scope,
                    "measured_window_ns": [begin, end],
                    "replayed_ns": time,
                }

        return {
            "checked_boundaries": checked,
            "observer_joins": len(self.observer_joins),
            "boundary_evidence": dict(evidence),
            "max_device_clock_window_ns": max_clock_width,
            "differences": dict(differences),
            "first_difference": first,
        }

    def local_unexplained_waits(self, plan, services):
        original = len(self.windows)
        latest = [None if window is None else window[1] - self.origin_ns for window in self.windows]
        latest.extend(plan["releases_ns"][original:])
        work = (
            (source, target, lag + service)
            for (source, target, lag), service in zip(plan["work"], services, strict=True)
        )
        incoming = plan["releases_ns"].copy()
        unknown = set()
        outgoing, remaining = defaultdict(list), Counter()

        for source, target, lag in chain(plan["dependencies"], work):
            outgoing[source].append((target, lag))
            if target >= original:
                remaining[target] += 1

        # Original boundaries are fixed witnesses here. Only auxiliary cuts need causal propagation.
        ready = list(range(original))
        ready.extend(index for index in range(original, len(latest)) if not remaining[index])
        for source in ready:
            stamp = latest[source]
            for target, lag in outgoing[source]:
                if target < original:
                    if stamp is None:
                        unknown.add(target)
                    else:
                        incoming[target] = max(incoming[target], stamp + lag)
                    continue

                if latest[target] is not None:
                    latest[target] = None if stamp is None else max(latest[target], stamp + lag)
                remaining[target] -= 1
                if not remaining[target]:
                    ready.append(target)

        # Use original predecessor evidence here so earlier replay drift cannot create a local gap.
        checked, groups = 0, {}
        for (rank, event), position in self.case.graph.positions.items():
            window = self.windows[position]
            if window is None or position in unknown or (rank, event) in self.external_cpu_points:
                continue

            checked += 1
            gap = window[0] - self.origin_ns - incoming[position]
            if gap <= 0:
                continue

            if event < 0:
                role = "cpu_protocol"
            else:
                graph = self.case.sources[rank][0].graph
                resource = graph["resources"][graph["events"][event]["resource"]]
                role = "observer_join" if (rank, event) in self.observer_joins else resource["role"]

            group = groups.setdefault(role, {"count": 0, "max_lower_ns": 0})
            group["count"] += 1
            if gap > group["max_lower_ns"]:
                group.update(max_lower_ns=gap, rank=rank, event=event)

        return {"checked_boundaries": checked, "by_role": groups}
