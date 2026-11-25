"""Bind measured work to a fit and publish the mathematical form used by the simulator."""

import json
from dataclasses import dataclass
from importlib.metadata import files
from pathlib import Path

from .backends import BoundedCurve, LinearRegression
from .models import ModelChoice, ServiceMeasurements, bounds


class ModelBackendRegistry:
    def __init__(self, *backends):
        self.backends = {}
        for backend in backends:
            self.register(backend)

    def register(self, backend):
        if backend.family in self.backends:
            raise ValueError(f"duplicate service model family: {backend.family}")
        self.backends[backend.family] = backend

    def __getitem__(self, family):
        return self.backends[family]


@dataclass(frozen=True)
class BindingRule:
    family: str
    kind: str | None = None
    role: str | None = None
    operation: str | None = None
    axes: tuple[str, ...] = ()
    priority: int = 0
    comparison_axis: str | None = None

    def matches(self, kind, role, operation, axes):
        return ((self.kind is None or self.kind == kind) and (self.role is None or self.role == role)
                and (self.operation is None or self.operation == operation) and all(axis in axes for axis in self.axes))

    def contract(self):
        return {key: value for key in ("kind", "role", "operation", "axes") if (value := getattr(self, key))}


class ModelBindingRegistry:
    def __init__(self, backends, default="bounded_curve"):
        self.backends = backends
        self.default = BindingRule(default)
        self.rules = []
        self.published = {}

    def register(self, rule):
        self.backends[rule.family]
        if rule in self.rules:
            raise ValueError(f"duplicate service binding rule: {rule}")
        self.rules.append(rule)

    def select(self, definition, interval, axes, transport=None):
        work = definition["work"][interval]
        operand = definition.get("external", transport if work["kind"] == "collective" else None)
        operation = (operand or {}).get("operation", work.get("operation"))
        if "copy" in work:
            operation = "local_copy"
        matches = [rule for rule in self.rules if rule.matches(work["kind"], definition.get("role"), operation, axes)]
        if not matches:
            return self.default
        highest = max(rule.priority for rule in matches)
        winners = [rule for rule in matches if rule.priority == highest]
        if len(winners) > 1:
            raise ValueError(f"ambiguous service binding rules at priority {highest}: {winners}")
        return winners[0]

    def choice(self, rule):
        backend = self.backends[rule.family]
        if rule.comparison_axis is None:
            return ModelChoice(backend)
        return ModelChoice(self.backends[self.default.family], (backend,), rule.comparison_axis)

    def fit(self, definition, interval, rows, transport=None):
        data = ServiceMeasurements(rows)
        result = {"points": len(data.points), "ambiguous_payload_points": data.ambiguous,
                  "training_rows": data.training_rows}
        if not data.points:
            return result
        if data.radius is not None:
            result["max_sample_clock_radius_ns"] = data.radius
        result["bounds"] = bounds(data.points)
        rule = self.select(definition, interval, data.points[0][0], transport)
        backend, parameters = self.choice(rule).fit(data.points, result)
        if parameters is None:
            return {**result, "reason": "inadmissible_training_fit"}

        # Fixed releases already name these bindings. An additional rule carries its applicability and math with it.
        original = self.published.get(rule)
        if original is None:
            result.update(form=backend.form, binding_rule=rule.contract())
            if backend.exclusive_axis:
                result["exclusive_axis"] = True
        elif backend.family == rule.family and "rates" in original:
            rates = parameters["coefficients"]
            named_rates = {field: rates[key] for key, field in original["rates"].items()}
            parameters = {"startup_ns": parameters["startup_ns"], **named_rates}
        return {**result, "family": backend.family, **parameters}

    @classmethod
    def from_catalog(cls):
        catalog = Path(__file__).resolve().parents[3] / "shared/cost_families.json"
        if not catalog.is_file():

            # Wheels install the same asset under the environment's data directory.
            catalog = next(item.locate() for item in files("prologue-collection")
                           if item.parts[-3:] == ("share", "prologue", "cost_families.json"))
        families = json.loads(catalog.read_bytes())
        registry = cls(ModelBackendRegistry(BoundedCurve()))
        registry.published[registry.default] = families[registry.default.family]
        for family, spec in families.items():
            if family == registry.default.family:
                continue
            axes = tuple(spec["rates"])
            registry.backends.register(LinearRegression(family, axes, spec.get("exclusive_axis", False)))
            rule = BindingRule(family, **spec["rule"], comparison_axis=axes[0])
            registry.register(rule)
            registry.published[rule] = spec
        return registry


service_models = ModelBindingRegistry.from_catalog()
