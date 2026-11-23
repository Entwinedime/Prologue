"""Read a task and enumerate its configuration and workload pairs."""

import json
from dataclasses import asdict
from pathlib import Path

from prologue.app.core import run_core
from prologue.integrations.frameworks.sglang.configuration import Model, configuration_from
from prologue.integrations.frameworks.sglang.content import PromptContent


class Plan:
    def __init__(self, path: Path) -> None:
        source = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(source, dict) or source.keys() != {"model", "configurations", "workloads"}:
            raise ValueError("task fields are model, configurations and workloads")

        self.model = Model(**source["model"])
        configurations = source["configurations"]
        if not isinstance(configurations, list) or not configurations:
            raise ValueError("at least one configuration is required")

        self.configurations = [configuration_from(entry) for entry in configurations]
        if len({configuration.id for configuration in self.configurations}) != len(self.configurations):
            raise ValueError("configuration IDs must be unique")

        # The core owns request validation and arrival order for both planning and simulation.
        self.workloads = run_core("workloads", source["workloads"])
        for workload in self.workloads:
            PromptContent.declared(workload)
            for configuration in self.configurations:
                configuration.route_table(workload)

    def document(self) -> dict:
        return {
            "model": asdict(self.model),
            "configurations": [configuration.document() for configuration in self.configurations],
            "workloads": self.workloads,
            "targets": [
                {"configuration": configuration.id, "workload": workload["id"]}
                for configuration in self.configurations
                for workload in self.workloads
            ],
        }
