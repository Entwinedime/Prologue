"""Publish selected service models in the domains that supplied their training evidence."""

from collections import defaultdict

from prologue.performance.models import ModelBindings
from prologue.performance.requirements import cost_identity


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

        self.models = {key: ModelBindings.fit(self.definitions[key[0]], key[1], rows)
                       for key, rows in sorted(groups.items())}

    def document(self, domain):
        result = {
            "domain": domain, "definitions": self.definitions,
            "models": [{"definition": definition, "interval": interval, "phase": phase, **model}
                       for (definition, interval, phase), model in self.models.items() if model["points"]],
        }
        unresolved = [{"definition": definition, "interval": interval, "phase": phase, **model}
                      for (definition, interval, phase), model in self.models.items() if not model["points"]]
        if unresolved:
            result["unmodeled"] = unresolved
        return result


class FittedCosts:
    """Group independent measurements by implementation domain and fit each service."""

    def __init__(self, sources):
        grouped = defaultdict(list)
        placements = {}
        for source in sources:
            if profile := source.get("transport_profile"):
                target = cost_identity(profile["logical_workers"])
                physical = profile["placement"], profile["source_image"]
                if placements.setdefault(target, physical) != physical:
                    raise ValueError("transport profiles select different placements or runtimes for the same workers")
            grouped[cost_identity(source["domain"])].append(source)
        self.domains = [DomainCosts(rows).document(rows[0]["domain"]) for rows in grouped.values()]
