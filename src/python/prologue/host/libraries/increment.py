"""Qualify one local recording for persistent structure or service-cost publication."""

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from prologue.common.io import content_id, read_json, write_json
from prologue.host.libraries.release import asset_path
from prologue.host.libraries.templates import TemplatePublication
from prologue.host.performance.fitting import FittedCosts
from prologue.host.performance.requirements import CostRequirements, MeasuredWork


class NativeRecording(Protocol):
    """The publication boundary carries the selected source and its associated capture metadata."""

    directory: Path
    session: Path
    attempt_id: str
    inputs: dict
    source: Path
    manifest: dict

    receipt: dict


class EvidenceQualifier(Protocol):
    def requalify(self, recording: NativeRecording, queries: list[dict]) -> tuple[dict, list[dict]]: ...
    def template_resources(self, facts: dict) -> list[dict]: ...


class LocalLibraryIncrement:
    """Compile and fit only the assets attributable to this qualified local session, then publish their release
    together.
    """

    def __init__(
        self,
        recording: NativeRecording,
        queries,
        qualifier: EvidenceQualifier,
        model,
        cost_source,
        *,
        measured=None,
    ):
        self.recording = recording
        self.session = recording.session
        self.receipt = receipt = recording.receipt
        self.qualifier, self.model, self.cost_source = qualifier, model, cost_source
        self.measured = measured
        acquired = dict(zip(receipt["queries"], receipt["answers"], strict=True))

        self.direct = []
        self.derived = set()
        self.questions = defaultdict(dict)
        self.support = defaultdict(set)

        for query in queries:
            key = content_id(query)
            if key in acquired:
                answer = acquired[key]
                if answer["status"] != "qualified":
                    raise ValueError(f"native session did not answer query {key}")

                name = answer["asset"]
                self.direct.append(query)
                for related in answer.get("support", ()):
                    self.support[related].add(key)
            elif "work" in query:
                name = cost_source.product(recording, query)
                if name is None:
                    raise ValueError("the local recording has no measurement for this work or its required peers")

                self.derived.add(name)
            else:
                raise ValueError(f"structure query {key} does not belong to this native session")

            self.questions[name][key] = query

        if not self.questions:
            raise ValueError("local publication requires an original query")

        self.identities = {
            name: content_id([recording.attempt_id, name]) for name in self.questions.keys() | self.support.keys()
        }

    def applied(self, receipts):
        return all(identity in receipts and self.questions.get(name, {}).keys() <= set(receipts[identity]["queries"])
                   and self.support.get(name, set()) <= set(receipts[identity].get("supports", ()))
                   for name, identity in self.identities.items())

    def products(self, scratch):
        products = {}
        if self.direct:
            if "assets" in self.receipt:
                for name in sorted(self.identities.keys() - self.derived):
                    product = read_json(asset_path(self.session, name))
                    if content_id(product) != self.receipt["assets"][name]:
                        raise ValueError(f"local product changed after qualification: {name}")

                    products[name] = product
            else:

                # Receipts without product digests can be qualified again from the owning native recording.
                products, answers = self.qualifier.requalify(self.recording, self.direct)
                if any(answer["status"] != "qualified" for answer in answers):
                    raise ValueError("original query failed native requalification")

        for name in self.derived:
            queries = list(self.questions[name].values())
            products[name] = self.measured if self.measured is not None else self.cost_source.extract(
                self.session, self.recording, queries, self.model, scratch)

        if products.keys() != self.identities.keys():
            raise ValueError("native products differ from the qualified query receipts")
        return products

    def apply(self, candidate, receipts):
        with TemporaryDirectory(dir=candidate) as temporary:
            scratch = Path(temporary)
            products = self.products(scratch)
            structure, costs = {}, {}

            for name, product in products.items():
                if (
                    self.recording.attempt_id not in product["native_recordings"]
                    or product["acquisition_rule"] != self.receipt["rule"]
                ):
                    raise ValueError("local product belongs to a different native measurement")

                destination = costs if "samples" in product else structure
                destination[name] = product

            if structure:
                native = self.recording
                update = TemplatePublication(candidate / "templates", scratch, self.model, native.attempt_id,
                                             native.inputs["deployment"], self.qualifier.template_resources)

                for name, facts in structure.items():
                    update.observe(self.identities[name], facts, self.questions[name].values())
                update.publish()

            if costs:
                self.costs(candidate / "performance", scratch, receipts, costs)

            for name, product in products.items():
                identity = self.identities[name]
                prior = receipts.get(identity, {})
                related = self.support.get(name, set()) | set(prior.get("supports", ()))
                receipts[identity] = {
                    "product": name,
                    "queries": sorted(self.questions[name].keys() | set(prior.get("queries", ()))),
                    "recordings": product["native_recordings"],
                    "rule": self.receipt["rule"],
                    "deployment": self.recording.inputs["deployment"],
                    "physical_devices": self.recording.inputs["physical_devices"],
                    "image": self.recording.inputs["image"],
                }

                if related:
                    receipts[identity]["supports"] = sorted(related)

    def costs(self, directory, scratch, receipts, products):
        previous = read_json(directory / "costs.json")
        sources = dict.fromkeys(previous["sources"])
        measured_tables = {
            content_id(read_json(directory / path))
            for path, origins in zip(sources, previous["sample_recordings"], strict=True)
            if self.recording.attempt_id in origins
        }
        demand = CostRequirements()

        for name, product in products.items():
            identity = self.identities[name]

            # Native work binds its exact interval. A peer profile binds a measured physical operand across graph forms.
            queries = self.questions[name].values()
            if name in self.derived and MeasuredWork(product).uncovered(queries, self.model, scratch):
                if "transport_profile" in product:
                    raise ValueError("the physical calibration does not cover the requested collective operand")
                raise ValueError("the native measurement does not contain the requested work and features")

            for query in queries:
                demand.gap(identity, query)

            # One physical measurement can answer an original query and later requests for the same worker.
            digest = content_id(product)
            if identity not in receipts and digest not in measured_tables:
                path = f"samples/{digest}.json"
                write_json(directory / path, product, replace=False)
                sources[path] = None
                measured_tables.add(digest)

        # Source indices stay stable across releases, including any qualified cross-role donor references.
        fitted = previous
        if len(sources) != len(previous["sources"]):
            fitted = FittedCosts({path: read_json(directory / path) for path in sources}, previous["reused"]).document()
        covered = demand.query(self.model, fitted["domains"], scratch, nullcontext, None)

        for status, _ in covered.values():
            if status != "covered":
                raise ValueError(f"fitted publication cannot answer original cost query: {status}")
        if fitted is not previous:
            write_json(directory / "costs.json", fitted)
