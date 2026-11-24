"""Fit service models from measured work and publish their evaluator parameters."""

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median


class ModelBackend(ABC):
    """A fit returns None when these coordinates do not identify the model."""

    @classmethod
    @abstractmethod
    def fit(cls, points): ...

    @abstractmethod
    def predict(self, axes): ...

    @abstractmethod
    def parameters(self): ...


@dataclass
class BoundedCurve(ModelBackend):
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

    def predict(self, axes):
        if self.axis is None:
            return self.knots[0][1]
        coordinate = axes[self.axis]
        right = bisect_left(self.knots, coordinate, key=lambda knot: knot[0])
        if self.knots[right][0] == coordinate:
            return self.knots[right][1]
        return self.interpolate(self.knots[right - 1], self.knots[right], coordinate)

    def parameters(self):
        return {"axis": self.axis, "knots": self.knots}


@dataclass
class AxisRegression(ModelBackend):
    startup_ns: float
    rate: float

    @classmethod
    def supports(cls, axes):
        return cls.axis in axes

    @classmethod
    def fit(cls, points):
        if not all(cls.supports(axes) for axes, _ in points):
            return None
        coordinates = [values[cls.axis] for values, _ in points]
        if min(coordinates) <= 0:
            return None
        durations = [ns for _, ns in points]
        x_mean, y_mean = mean(coordinates), mean(durations)
        variance = sum((x - x_mean) ** 2 for x in coordinates)
        if not variance:
            return None

        covariance = sum((x - x_mean) * (ns - y_mean) for x, ns in zip(coordinates, durations, strict=True))
        rate = covariance / variance
        startup = y_mean - rate * x_mean
        return cls(startup, rate) if startup >= 0 and rate > 0 else None

    def predict(self, axes):
        return round(self.startup_ns + axes[self.axis] * self.rate)

    def parameters(self):
        return {"startup_ns": self.startup_ns, self.rate_field: self.rate}


class StartupTransfer(AxisRegression):
    family = "startup_transfer"
    axis = "transfer.bytes"
    rate_field = "ns_per_byte"

    @classmethod
    def supports(cls, axes):
        if axes.keys() != {cls.axis} or type(axes[cls.axis]) is not int or axes[cls.axis] <= 0:
            raise ValueError("a local copy requires a positive byte count from the target operand")
        return True


class AttentionPairs(AxisRegression):
    family = "attention_pairs"
    axis = "batch.attention_pairs"
    rate_field = "ns_per_pair"


class LayerTokens(AxisRegression):
    family = "layer_tokens"
    axis = "parameter.model_tokens"
    rate_field = "ns_per_token"
