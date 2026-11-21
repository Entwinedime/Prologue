"""Publish selected service models in the domains that supplied their training evidence."""

from collections import defaultdict
from itertools import chain

from prologue.performance.models import ModelBindings
from prologue.performance.requirements import cost_identity


class DomainCosts:
    def __init__(self, domain):
        self.domain = domain
        self.definitions, self.indices = [], {}
        self.groups = defaultdict(list)

    def add(self, definitions, rows):
        indices = []
        for definition in definitions:
            key = cost_identity(definition)
            if key not in self.indices:
                self.indices[key] = len(self.definitions)
                self.definitions.append(definition)
            indices.append(self.indices[key])

        for location, row in rows:
            key = indices[row["definition"]], row["interval"], row["features"]["phase"]
            self.groups[key].append((location, row))

    def document(self):
        accepted, unresolved = [], []
        for (definition, interval, phase), rows in sorted(self.groups.items()):
            model = ModelBindings.fit(self.definitions[definition], interval, rows)
            destination = accepted if model["points"] else unresolved
            destination.append({"definition": definition, "interval": interval, "phase": phase, **model})

        result = {"domain": self.domain, "definitions": self.definitions, "models": accepted}
        if unresolved:
            result["unmodeled"] = unresolved
        return result


class FittedCosts:
    """Keep one source table for every selected model, including work qualified across roles."""

    def __init__(self, sources, reused):
        self.sources = [str(path) for path in sources]
        self.sample_recordings = [source.get("native_recordings", []) for source in sources.values()]
        source_indices = {path: index for index, path in enumerate(self.sources)}
        self.reused = reused
        domains, placements = {}, {}
        for index, source in chain(enumerate(sources.values()), ((None, record) for record in reused)):
            if profile := source.get("transport_profile"):
                target = cost_identity(profile["logical_workers"])
                physical = profile["placement"], profile["source_image"]
                if placements.setdefault(target, physical) != physical:
                    raise ValueError("transport profiles select different placements or runtimes for the same workers")

            domain = cost_identity(source["domain"])
            if domain not in domains:
                domains[domain] = DomainCosts(source["domain"])
            domains[domain].add(source["definitions"], self.observations(index, source["samples"], source_indices))
        self.domains = [group.document() for group in domains.values()]

    @staticmethod
    def observations(source, rows, source_indices):
        for index, row in enumerate(rows):
            if source is None:

                # Cross-role qualification retains the original measured row and its physical worker.
                donor = row["observation"]
                yield (source_indices[donor["source"]], donor["row"]), row
            else:
                yield (source, index), row

    def document(self):
        return {"domains": self.domains, "sources": self.sources, "sample_recordings": self.sample_recordings,
                "reused": self.reused}
