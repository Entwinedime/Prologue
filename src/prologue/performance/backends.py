"""Fit mathematical service curves using only measured coordinates."""

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, sqrt, ulp
from statistics import mean, median


def service_ns(value):
    """Round to the nearest signed nanosecond, with halfway values going to the even integer."""
    if not 0 <= value < 2**63:
        raise ValueError("service duration exceeds signed nanoseconds")
    return round(value)


class ModelBackend(ABC):
    """A fit returns evaluator parameters, or None when the data do not identify them."""

    family: str
    form: str

    def supports(self, axes):
        return True

    @abstractmethod
    def fit(self, points): ...

    @abstractmethod
    def predict(self, parameters, axes): ...


class BoundedCurve(ModelBackend):
    family = "bounded_curve"
    form = "curve"

    def fit(self, points):
        result = {"axis": None, "knots": [[0, service_ns(median(ns for _, ns in points))]]}
        best_gain = 0
        for axis in points[0][0]:
            observations = defaultdict(list)
            for axes, ns in points:
                observations[axes[axis]].append(ns)
            knots = [[x, service_ns(median(values))] for x, values in sorted(observations.items())]
            gain = self.gain(knots, observations)
            if gain > best_gain:
                best_gain, result = gain, {"axis": axis, "knots": knots}
        return result

    @staticmethod
    def interpolate(left, right, value):
        x1, y1 = left
        x2, y2 = right
        offset = (y2 - y1) * (float(value) - x1) / (float(x2) - x1)
        return service_ns(y1 + offset)

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

    def predict(self, parameters, axes):
        knots, axis = parameters["knots"], parameters["axis"]
        if axis is None:
            return knots[0][1]
        coordinate = axes[axis]
        right = bisect_left(knots, coordinate, key=lambda knot: knot[0])
        if knots[right][0] == coordinate:
            return knots[right][1]
        return self.interpolate(knots[right - 1], knots[right], coordinate)


@dataclass(frozen=True)
class LinearRegression(ModelBackend):
    """A nonnegative startup cost plus one or more measured service rates."""

    family: str
    axes: tuple[str, ...]
    exclusive_axis: bool = False
    form = "linear"

    def supports(self, axes):
        if self.exclusive_axis:
            axis, = self.axes
            if axes.keys() != {axis} or type(axes[axis]) is not int or axes[axis] <= 0:
                raise ValueError("a local copy requires a positive byte count from the target operand")
        return all(axis in axes for axis in self.axes)

    def fit(self, points):
        if not self.axes or not all(self.supports(axes) for axes, _ in points):
            return None
        columns = [[values[axis] for values, _ in points] for axis in sorted(self.axes)]
        if any(min(column) <= 0 for column in columns):
            return None
        durations = [ns for _, ns in points]
        centers = [mean(column) for column in columns]
        middle = mean(durations)
        columns = [[x - center for x in col] for col, center in zip(columns, centers, strict=True)]
        target = [ns - middle for ns in durations]
        if len(columns) == 1:
            column, = columns
            variance = sum(x * x for x in column)
            if not variance:
                return None
            rates = [self.dot(column, target) / variance]
        else:
            rates = self.solve(columns, target)
            if rates is None:
                return None

        startup = middle - self.dot(centers, rates)
        if not isfinite(startup) or startup < 0 or any(not isfinite(rate) or rate <= 0 for rate in rates):
            return None
        return {"startup_ns": startup, "coefficients": dict(zip(sorted(self.axes), rates, strict=True))}

    @staticmethod
    def dot(left, right):
        return sum(a * b for a, b in zip(left, right, strict=True))

    @classmethod
    def solve(cls, columns, target):
        # Modified Gram-Schmidt exposes collinear measured axes before the triangular solve.
        triangular = []
        response = []
        for index, column in enumerate(columns):
            norm = sqrt(cls.dot(column, column))
            if not norm:
                return None
            vector = [x / norm for x in column]
            row = []
            for prior in triangular:
                projection = cls.dot(prior[0], vector)
                row.append(projection * norm)
                vector = [x - projection * q for x, q in zip(vector, prior[0], strict=True)]
            residual = sqrt(cls.dot(vector, vector))
            if residual <= ulp(1.0) * max(len(column), index + 1):
                return None
            unit = [x / residual for x in vector]
            triangular.append((unit, [*row, residual * norm]))
            response.append(cls.dot(unit, target))

        for i in reversed(range(len(response))):
            later = sum(triangular[j][1][i] * response[j] for j in range(i + 1, len(response)))
            response[i] = (response[i] - later) / triangular[i][1][i]
        return response

    def predict(self, parameters, axes):
        value = parameters["startup_ns"]
        for axis, rate in sorted(parameters["coefficients"].items()):
            value += axes[axis] * rate
        return service_ns(value)
