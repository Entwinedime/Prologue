"""Fit bounded service curves to qualified measurements."""

from collections import defaultdict
from statistics import median

from prologue.performance.requirements import cost_identity


class ServiceCurve:
    def __init__(self, rows):
        groups = defaultdict(list)
        payloads = defaultdict(set)
        for row in rows:
            point = tuple(sorted(row["features"]["axes"].items()))
            groups[point].extend(row["observations_ns"])
            if "copy" in row:
                payloads[point].add(("copy", row["copy"]["bytes"], row["copy"]["direction"]))
            if "collective" in row:
                collective = row["collective"]
                payloads[point].add(("collective", collective["dtype"], collective["algorithm"], collective["elements"]))
        self.ambiguous_payloads = sum(len(values) > 1 for values in payloads.values())
        points = [(dict(point), median(values)) for point, values in groups.items() if len(payloads[point]) <= 1]
        if not points:
            raise ValueError("no usable feature points for service work")
        axes = tuple(points[0][0])
        if any(set(values) != set(axes) for values, _ in points):
            raise ValueError("same service work has inconsistent cost features")

        self.bounds = {axis: [min(values[axis] for values, _ in points), max(values[axis] for values, _ in points)]
                       for axis in axes}
        self.knots = [[0, round(median(value for _, value in points))]]
        self.axis = None
        self.points = len(points)

        # Compare interpolation and a constant on the same interior points, withheld one coordinate at a time.
        best_gain = 0
        for axis in axes:
            observed = defaultdict(list)
            for values, cost in points:
                observed[values[axis]].append(cost)
            knots = [[x, round(median(values))] for x, values in sorted(observed.items())]
            gain = self.gain(knots, observed)
            if gain > best_gain:
                best_gain = gain
                self.axis = axis
                self.knots = knots

    @staticmethod
    def interpolate(left, right, value):
        x1, y1 = left
        x2, y2 = right
        return round(y1 + (y2 - y1) * (value - x1) / (x2 - x1))

    @classmethod
    def gain(cls, knots, observations):
        constant_loss, curve_loss, count = 0, 0, 0
        for index, (x, _) in enumerate(knots[1:-1], 1):
            constant = median(value for coord, values in observations.items() if coord != x for value in values)
            estimate = cls.interpolate(knots[index - 1], knots[index + 1], x)
            for actual in observations[x]:
                scale = max(actual, 1)
                constant_loss += abs(constant - actual) / scale
                curve_loss += abs(estimate - actual) / scale
                count += 1
        return (constant_loss - curve_loss) / count if count else 0

    def document(self):
        return {"bounds": self.bounds, "axis": self.axis, "knots": self.knots, "points": self.points,
                "ambiguous_payload_points": self.ambiguous_payloads}


class DomainCosts:
    def __init__(self, sources):
        self.definitions = []
        indices_by_definition = {}
        groups = defaultdict(list)
        for source in sources:
            indices = []
            for definition in source["definitions"]:
                key = cost_identity(definition)
                if key not in indices_by_definition:
                    indices_by_definition[key] = len(self.definitions)
                    self.definitions.append(definition)
                indices.append(indices_by_definition[key])
            for row in source["samples"]:
                groups[indices[row["definition"]], row["interval"], row["features"]["phase"]].append(row)

        self.curves = {key: ServiceCurve(rows).document() for key, rows in sorted(groups.items())}

    def document(self, domain):
        return {
            "domain": domain, "definitions": self.definitions,
            "models": [{"definition": definition, "interval": interval, "phase": phase, **model}
                       for (definition, interval, phase), model in self.curves.items()],
        }


class FittedCosts:
    """Group independent measurements by implementation domain and fit each service."""

    def __init__(self, sources):
        grouped = defaultdict(list)
        for source in sources:
            grouped[cost_identity(source["domain"])].append(source)
        self.domains = [DomainCosts(rows).document(rows[0]["domain"]) for rows in grouped.values()]
