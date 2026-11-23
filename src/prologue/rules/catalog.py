"""Read the rule types used when assembling published assets."""

import json
from importlib.resources import files


class Rules:
    def __init__(self, document):
        sections = {"templates", "performance", "acquisition"}
        if (not isinstance(document, dict) or document.keys() != sections
                or any(not isinstance(section, dict) for section in document.values())):
            raise ValueError("rules require templates, performance and acquisition sections")
        self.document = document
        self.templates = document["templates"]
        self.performance = document["performance"]
        self.acquisition = document["acquisition"]

        self.check_choice(self.templates, "selector")
        for binding in self.performance.values():
            self.check_choice(binding, "estimator")
        for recipe in self.acquisition.values():
            self.check_choice(recipe, "mode")

    @staticmethod
    def check_choice(choice, field):
        if (not isinstance(choice, dict) or choice.keys() != {field}
                or not isinstance(choice[field], str) or not choice[field]):
            raise ValueError(f"rule requires one nonempty {field}")

    @classmethod
    def read(cls, path=None):
        source = path if path is not None else files("prologue.rules").joinpath("defaults.json")
        return cls(json.loads(source.read_text(encoding="utf-8")))

    def require_observation(self, purpose, mode):
        if self.acquisition[purpose]["mode"] != mode:
            raise ValueError(f"observation mode {mode} is outside the {purpose} acquisition rule")

    def estimator(self, family):
        return self.performance[family]["estimator"]
