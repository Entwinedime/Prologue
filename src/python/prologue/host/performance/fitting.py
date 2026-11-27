"""Publish selected service models in the domains that supplied their training evidence."""

import json
from collections import defaultdict
from itertools import chain

from prologue.common.cost import cost_identity
from prologue.common.io import write_json
from prologue.host.performance.registry import service_models


class DomainCosts:
    def __init__(self, domain, observed_overhead_sources):
        self.domain = domain
        self.observed_overhead_sources = observed_overhead_sources
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
            index = indices[row["definition"]]
            features = row["features"]
            operand = features.get("transport")
            external = self.definitions[index].get("external")

            if "collective" in row and operand is None and external is None:
                raise ValueError("a hardware collective sample must identify its peer group and operands")
            if external is not None and operand is not None and operand != external:
                raise ValueError("sample operand differs from its declared external service")

            # A shared graph definition may execute against different physical peers or reduction layouts.
            transport = cost_identity(operand) if operand is not None and external is None else ""
            key = index, row["interval"], features["phase"], transport
            self.groups[key].append((location, row))

    def select_observations(self, definition, interval, rows):
        kind = self.definitions[definition]["work"][interval]["kind"]
        corrected = self.observed_overhead_sources if kind == "host" else ()
        if not corrected and not any(row.get("direct_native_call") for _, row in rows):
            return rows

        # Prefer a direct native interval at the same coordinate and physical operand.
        selected, best = [], {}
        for location, row in rows:
            key = cost_identity((row["features"], row.get("copy")))
            quality = 2 if row.get("direct_native_call") else int(location[0] in corrected)
            best[key] = max(best.get(key, 0), quality)
            selected.append((location, row, key, quality))

        return [(location, row) for location, row, key, quality in selected if quality == best[key]]

    def document(self):
        accepted, unresolved = [], []
        for (definition, interval, phase, transport), rows in sorted(self.groups.items()):
            operand = json.loads(transport) if transport else None
            rows = self.select_observations(definition, interval, rows)
            model = service_models.fit(self.definitions[definition], interval, rows, operand)
            if operand is not None:
                model["transport"] = operand

            destination = accepted if "family" in model else unresolved
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
        observed_overhead_sources = {index for index, source in enumerate(sources.values())
                                     if source.get("qualification", {}).get("measured_observer_intervals")}
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
                domains[domain] = DomainCosts(source["domain"], observed_overhead_sources)
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

    @staticmethod
    def publish(directory, asset):
        if asset is None:
            write_json(directory / "costs.json", FittedCosts({}, []).document(), replace=False)
            return

        fitted = json.loads(asset.read_bytes())
        origins = fitted["sample_recordings"]
        names = {}
        (directory / "samples").mkdir(parents=True)

        for index, (name, recordings) in enumerate(zip(fitted["sources"], origins, strict=True)):
            if name in names:
                raise ValueError(f"duplicate training source in fitted cost asset: {name}")
            path = asset.parent / name
            content = path.read_bytes()

            if json.loads(content).get("native_recordings", []) != recordings:
                raise ValueError(f"fitted model and sample disagree about the native recordings: {path}")
            relative = f"samples/{index}.json"
            (directory / relative).write_bytes(content)
            names[name] = relative

        # Model training rows index this ordered table. Reused observations must refer to the same portable samples.
        for group in fitted["reused"]:
            for sample in group["samples"]:
                donor = sample["observation"]
                donor["source"] = names[donor["source"]]

        write_json(directory / "costs.json", {
            "domains": fitted["domains"], "sources": list(names.values()), "sample_recordings": origins,
            "reused": fitted["reused"],
        }, replace=False)
