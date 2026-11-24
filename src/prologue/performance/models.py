"""Select a fitted backend for each measured work contract and execution phase."""

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median

from .backends import AttentionPairs, BoundedCurve, LayerTokens, StartupTransfer


class ServiceMeasurements:
    def __init__(self, rows):
        durations, payloads, radii = defaultdict(list), defaultdict(set), {}
        origins = defaultdict(list)
        for location, row in rows:
            point = tuple(sorted(row["features"]["axes"].items()))
            observations = row["observations_ns"]
            if not observations or any(type(ns) is not int or ns < 0 for ns in observations):
                raise ValueError("service samples require nonnegative integer nanoseconds")
            durations[point].extend(observations)
            origins[point].append(location)
            if "copy" in row:
                payloads[point].add(("copy", row["copy"]["bytes"], row["copy"]["direction"]))
            if "collective" in row:
                op = row["collective"]
                payloads[point].add(("collective", op["dtype"], op["algorithm"], op["elements"]))
                radii[point] = max(radii.get(point, 0), op["max_clock_radius_ns"])

        # Equal apparent coordinates can conceal different payloads in an opaque native operation.
        self.ambiguous = sum(len(values) > 1 for values in payloads.values())
        accepted = [point for point in durations if len(payloads[point]) <= 1]
        self.points = [(dict(point), median(durations[point])) for point in accepted]
        self.training_rows = sorted({location for point in accepted for location in origins[point]})
        self.radius = max((radius for point, radius in radii.items() if len(payloads[point]) == 1), default=None)


def bounds(points):
    axes = points[0][0].keys()
    if any(values.keys() != axes for values, _ in points):
        raise ValueError("same service work has inconsistent cost features")
    return {axis: [min(values[axis] for values, _ in points), max(values[axis] for values, _ in points)]
            for axis in axes}


@dataclass(frozen=True)
class ModelChoice:
    default: type = BoundedCurve
    alternatives: tuple = ()
    comparison_axis: str | None = None

    def candidates(self, axes):
        return [family for family in self.alternatives if family.supports(axes)] + [self.default]

    def fit(self, points, report):
        selected = self.default.fit(points)
        *alternatives, _ = self.candidates(points[0][0])
        if not alternatives:
            return selected
        fits = {family: family.fit(points) for family in alternatives}
        axis = self.comparison_axis
        if axis not in points[0][0]:
            return selected

        # Hold out whole coordinate values. Every candidate sees the same points inside its training span.
        coordinates = sorted({axes[axis] for axes, _ in points})
        withheld = set(coordinates[1:-1:2])
        if not withheld:
            return selected
        training = [(axes, ns) for axes, ns in points if axes[axis] not in withheld]
        ranges = bounds(training)
        checks = [(axes, ns) for axes, ns in points if axes[axis] in withheld
                  and all(low <= axes[key] <= high for key, (low, high) in ranges.items())]
        if not checks:
            return selected

        fits[self.default] = selected
        scores, excluded = {}, {}
        for family, whole in fits.items():
            trial = family.fit(training)
            if trial is None or whole is None:
                excluded[family.family] = "inadmissible_training_fit" if trial is None else "inadmissible_final_fit"
                continue
            error = mean(abs(trial.predict(axes) - ns) / max(ns, 1) for axes, ns in checks)
            if not scores or error < min(scores.values()):
                selected = whole
            scores[family.family] = error

        selection = {"axis": axis, "held_out_coordinates": sorted({axes[axis] for axes, _ in checks}),
                     "held_out_points": len(checks), "mean_relative_error": scores}
        if excluded:
            selection["excluded"] = excluded
        report["selection"] = selection
        return selected


class ModelBindings:
    """Work selects candidates; domain, definition, interval and phase still identify each fitted service."""

    default = ModelChoice()
    rules = {
        ("transfer", "local_copy"): ModelChoice(alternatives=(StartupTransfer,), comparison_axis=StartupTransfer.axis),
        ("device", "attention"): ModelChoice(alternatives=(AttentionPairs,), comparison_axis=AttentionPairs.axis),
        ("device", "layer"): ModelChoice(alternatives=(LayerTokens,), comparison_axis=LayerTokens.axis),
    }

    @classmethod
    def for_interval(cls, definition, interval):
        work = definition["work"][interval]
        role = definition.get("role")
        if "copy" in work or definition.get("external", {}).get("operation") == "local_copy":
            role = "local_copy"
        return cls.rules.get((work["kind"], role), cls.default)

    @classmethod
    def fit(cls, definition, interval, rows):
        data = ServiceMeasurements(rows)
        result = {"points": len(data.points), "ambiguous_payload_points": data.ambiguous, "training_rows": data.training_rows}
        if data.points:
            if data.radius is not None:
                result["max_sample_clock_radius_ns"] = data.radius
            result["bounds"] = bounds(data.points)
            selected = cls.for_interval(definition, interval).fit(data.points, result)
            result.update(family=selected.family, **selected.parameters())
        return result
