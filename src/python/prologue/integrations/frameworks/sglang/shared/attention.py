"""Retain the native module and ordered device tasks that identify a source attention cut."""

import json
from collections import defaultdict

from prologue.common.cost import WorkCatalog


class AttentionCatalog:
    @staticmethod
    def runner(job):
        instances = job["instances"]
        if len(instances) == 1 and instances[0]["mode"] == "null":
            settings = instances[0]["configuration"]
            if settings["tp"] == settings.get("pp", 1) == 1 and "ep" not in settings:
                return {"model": job["model"], "configuration": settings}
        return None

    @classmethod
    def extract(cls, source, work, rank, identities, runner, task_reader):
        catalog = WorkCatalog(work / f"{rank}-graph.json", identities)
        facts = json.loads((work / f"{rank}-facts.json").read_bytes())
        return cls.from_catalog(source, work / f"{rank}-sdk/trace.json", rank, catalog, facts, runner, task_reader)

    @staticmethod
    def from_catalog(source, trace, rank, catalog, facts, runner, task_reader):
        scopes = {}
        for piece, definition, _, features in catalog:
            scope = piece["scope"]
            if (
                piece["kind"] == "device"
                and features["phase"] == "decode"
                and catalog.definitions[definition]["role"] == "attention"
                and not catalog.definitions[definition]["children"]
            ):
                scopes[scope] = definition
        observed = task_reader(source, trace, rank, catalog, scopes, facts) if scopes else {}

        # A shared definition can bind without a layer selector only when its source instances agree on ownership.
        groups = defaultdict(list)
        for scope, definition in scopes.items():
            groups[definition].append(scope)
        cuts = []
        for index, instances in groups.items():
            first = instances[0]
            parts = observed.get(first)
            if parts and all(observed.get(scope) == parts for scope in instances):
                cuts.append({"definition": catalog.definitions[index], "site": catalog.graph["scopes"][first]["site"],
                             "parts": parts, "source_calls": len(instances)})
        return {"domain": catalog.domain, "runner": runner, "cuts": cuts}
