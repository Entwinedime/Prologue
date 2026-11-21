"""Bind a measured service to a model family; only the C++ evaluator serves predictions."""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median


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


@dataclass
class BoundedCurve:
    family = "bounded_curve"
    axis: str | None
    knots: list

    @classmethod
    def fit(cls, points):
        result = cls(None, [[0, round(median(ns for _, ns in points))]])
        best_gain = 0
        for axis in points[0][0]:
            observations = defaultdict(list)
            for axes, ns in points:
                observations[axes[axis]].append(ns)
            knots = [[x, round(median(values))] for x, values in sorted(observations.items())]
            gain = cls.gain(knots, observations)
            if gain > best_gain:
                best_gain, result.axis, result.knots = gain, axis, knots
        return result

    @staticmethod
    def interpolate(left, right, value):
        x1, y1 = left
        x2, y2 = right
        return round(y1 + (y2 - y1) * (value - x1) / (x2 - x1))

    @classmethod
    def gain(cls, knots, observations):
        gain, count = 0, 0
        for index, (x, _) in enumerate(knots[1:-1], 1):
            constant = median(value for coord, values in observations.items() if coord != x for value in values)
            estimate = cls.interpolate(knots[index - 1], knots[index + 1], x)
            for actual in observations[x]:
                gain += (abs(constant - actual) - abs(estimate - actual)) / max(actual, 1)
                count += 1
        return gain / count if count else 0

    def training_estimate(self, axes):
        if self.axis is None:
            return self.knots[0][1]
        coordinate = axes[self.axis]
        right = bisect_left(self.knots, coordinate, key=lambda knot: knot[0])
        return self.interpolate(self.knots[right - 1], self.knots[right], coordinate)

    def parameters(self):
        return {"axis": self.axis, "knots": self.knots}


@dataclass
class StartupTransfer:
    family = "startup_transfer"
    startup_ns: float
    ns_per_byte: float

    @classmethod
    def fit(cls, points):
        sizes = [axes["transfer.bytes"] for axes, _ in points]
        costs = [ns for _, ns in points]
        x_mean, y_mean = mean(sizes), mean(costs)
        covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(sizes, costs, strict=True))
        slope = covariance / sum((x - x_mean) ** 2 for x in sizes)
        startup = y_mean - slope * x_mean
        return cls(startup, slope) if startup >= 0 and slope > 0 else None

    def training_estimate(self, axes):
        return round(self.startup_ns + axes["transfer.bytes"] * self.ns_per_byte)

    def parameters(self):
        return {"startup_ns": self.startup_ns, "ns_per_byte": self.ns_per_byte}


class ModelBindings:
    """Select candidates by work contract, keeping the native implementation and role in the service identity."""

    candidates = {"local_copy": (StartupTransfer, BoundedCurve)}

    @classmethod
    def for_interval(cls, definition, interval):
        work = definition["work"][interval]
        operation = "local_copy" if "copy" in work else definition.get("external", {}).get("operation")
        return cls.candidates.get(operation, (BoundedCurve,)) if work["kind"] == "transfer" else (BoundedCurve,)

    @classmethod
    def fit(cls, definition, interval, rows):
        data = ServiceMeasurements(rows)
        points = data.points
        result = {"points": len(points), "ambiguous_payload_points": data.ambiguous, "training_rows": data.training_rows}
        if not points:
            return result
        if data.radius is not None:
            result["max_sample_clock_radius_ns"] = data.radius
        axes = points[0][0].keys()
        if any(values.keys() != axes for values, _ in points):
            raise ValueError("same service work has inconsistent cost features")
        result["bounds"] = {axis: [min(values[axis] for values, _ in points), max(values[axis] for values, _ in points)]
                            for axis in axes}
        candidates = cls.for_interval(definition, interval)
        empirical = selected = BoundedCurve.fit(points)
        if len(candidates) > 1:
            if axes != {"transfer.bytes"} or any(
                type(row["transfer.bytes"]) is not int or row["transfer.bytes"] <= 0 for row, _ in points
            ):
                raise ValueError("a local copy requires a positive byte count from the target operand")
            points.sort(key=lambda point: point[0]["transfer.bytes"])
            withheld = points[1:-1:2]
            if withheld:

                # Compare on identical unseen sizes inside the training span. Independent runs are scored separately.
                training = points[::2]
                if len(points) % 2 == 0:
                    training.append(points[-1])
                scores, excluded = {}, {}
                for family in candidates:
                    candidate = family.fit(training)
                    whole = empirical if family is BoundedCurve else family.fit(points)
                    if candidate is None or whole is None:
                        excluded[family.family] = "inadmissible_training_fit" if candidate is None else "inadmissible_final_fit"
                        continue
                    error = mean(abs(candidate.training_estimate(values) - ns) / max(ns, 1) for values, ns in withheld)
                    if not scores or error < min(scores.values()):
                        selected = whole
                    scores[family.family] = error
                result["size_selection"] = {"held_out_bytes": [values["transfer.bytes"] for values, _ in withheld],
                                            "mean_relative_error": scores}
                if excluded:
                    result["size_selection"]["excluded"] = excluded
        return {**result, "family": selected.family, **selected.parameters()}
