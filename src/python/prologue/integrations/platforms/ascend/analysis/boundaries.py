"""Keep projected boundaries dense while preserving anchored native order coordinates."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

from .commands import span_references


def resource_positions(resources, spans, anchors, kind):
    """Index positions that can separate adjacent observations on one native lane."""

    cuts = defaultdict(set)
    for span in spans:
        if resources[span["resource"]]["kind"] == kind:
            cuts[span["resource"]].update((span["begin"], span["end"]))

    for anchor in anchors:
        resource = spans[anchor["span"]]["resource"]
        if resource in cuts:
            cuts[resource].add(anchor["order"])

    return {resource: sorted(positions) for resource, positions in cuts.items()}


def trim_device_boundaries(result: dict, device_spans: dict[int, int]) -> None:
    """Retain actual work and referenced fences inside the observed device markers."""

    spans, resources, scopes = (result[key] for key in ("spans", "resources", "scopes"))
    anchors = list(span_references(result))
    references = Counter(anchor["span"] for anchor in anchors)
    device = set(device_spans.values())
    points = defaultdict(set)

    for identity, span in enumerate(spans):
        if identity not in device and resources[span["resource"]]["kind"] == "stream":
            points[span["resource"]].update((span["begin"], span["end"]))

    by_span = defaultdict(list)
    for anchor in {id(anchor): anchor for anchor in anchors}.values():
        identity, position = anchor["span"], anchor["order"]
        span = spans[identity]
        by_span[identity].append(anchor)

        # Interval endpoints come from observer records; interior native dependencies and explicit points remain.
        if resources[span["resource"]]["kind"] == "stream" and (
            identity not in device or span["begin"] < position < span["end"]
        ):
            points[span["resource"]].add(position)

    points = {resource: sorted(values) for resource, values in points.items()}
    for scope, span in device_spans.items():
        references[span] -= sum(
            port["name"] in {"device.enter", "device.complete"} for port in scopes[scope]["ports"]
        )

    discarded = set()
    for identity in device:
        span = spans[identity]
        begin, end = span["begin"], span["end"]
        positions = points.get(span["resource"], ())
        left, right = bisect_left(positions, begin), bisect_right(positions, end)
        bounds = [positions[left], positions[right - 1]] if left < right else []
        payload = resources[span["resource"]]["payload"]
        first = bisect_right(payload, begin, key=lambda interval: interval[1])
        last = bisect_left(payload, end, key=lambda interval: interval[0])

        if first < last:
            bounds.extend((max(begin, payload[first][0]), min(end, payload[last - 1][1])))
        if not bounds:
            if references[identity] == 0:
                discarded.add(identity)
            continue

        # Semantic device lifetime covers native work and required fences, not the observer's enclosing markers.
        span["begin"], span["end"] = min(bounds), max(bounds)
        if span["begin"] == span["end"]:
            span["kind"] = "boundary"
        for anchor in by_span[identity]:
            if anchor["order"] == begin:
                anchor["order"] = span["begin"]
            elif anchor["order"] == end:
                anchor["order"] = span["end"]

    if not discarded:
        return

    for scope in scopes:
        scope["ports"][:] = [port for port in scope["ports"] if port["at"]["span"] not in discarded]

    # Publication ends interpretation: remap every surviving reference, including shared anchor objects, once.
    positions = {
        old: new for new, old in enumerate(index for index in range(len(spans)) if index not in discarded)
    }
    anchors = {id(anchor): anchor for anchor in span_references(result)}

    for anchor in anchors.values():
        anchor["span"] = positions[anchor["span"]]
    result["spans"] = [span for index, span in enumerate(spans) if index not in discarded]
