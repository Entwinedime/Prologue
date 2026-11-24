"""Resolve concrete prediction gaps within the caller's placement and cumulative allowances."""

import json
from collections import defaultdict
from types import SimpleNamespace

from prologue.acquisition.allowances import AcquisitionBudget
from prologue.acquisition.capture import NativeAttempt
from prologue.acquisition.library_increment import LocalLibraryIncrement
from prologue.acquisition.library_store import LibraryStore
from prologue.acquisition.recording import PublishedCapture
from prologue.acquisition.rules import EvidenceGap
from prologue.artifacts import content_id, write_json
from prologue.integrations.runtimes.docker import ModelingProcess
from prologue.libraries import LibraryRelease, asset_path
from prologue.performance.requirements import CostRequirements
from prologue.workflow.limits import AllowanceExhausted


class Replenishment:
    def __init__(self, task, args, rules, source_costs):
        self.root, self.rules, self.source_costs = task.root, rules, source_costs
        self.store = LibraryStore(task.release.root)
        if (self.store.root / "release.json").exists():
            raise ValueError("on-demand acquisition needs a library store, not an immutable release")
        self.sessions = self.root / "acquisition/sessions"
        self.deployment = json.loads(args.deployment.read_bytes())
        self.source_image = NativeAttempt.image(args.source_image)["Id"]
        self.budget = AcquisitionBudget(self.root, self.store, args)
        self.options = dict(timeout=args.timeout, storage_bytes=args.storage_bytes, source_image=self.source_image,
                            modeling_image=task.runtime["modeling_image"], binary=task.root / "inputs/prologue")
        self.failed = {}
        self.measured = {}
        self.unresolved_services = {}

    def open(self):
        release = self.store.open()
        self.budget.open()
        for path in self.sessions.glob("*/session/native/execution.json"):
            NativeAttempt.recover_pending(path.parent)
        return release

    def groups(self, queries, release, answers):
        acquisition = self.rules(release, SimpleNamespace(**self.options))
        origin = str(release.directory.relative_to(self.store.root)) if release.identity else None
        pending = {}
        for path in sorted(self.sessions.glob("*/demand.json")):
            original = json.loads(path.read_bytes())
            keys = {content_id(q) for q in original["queries"]}
            began = (path.parent / "session/native/execution.json").is_file()
            if (began or original["origin"] == origin) and keys.intersection(queries) - answers.keys():
                pending[path.parent.name] = original

        # Recover the original batch even if another publication resolved some of its siblings.
        covered = {content_id(q) for group in pending.values() for q in group["queries"]}
        compatible = defaultdict(list)
        for key, query in queries.items():
            if key in covered or key in answers:
                continue
            try:
                rule = acquisition.select(query, self.deployment)
                compatible[rule.name, content_id(rule.session_key(query, self.deployment))].append(query)
            except EvidenceGap as gap:
                answers[key] = gap.report
        for (rule, _), group in compatible.items():
            group.sort(key=content_id)
            key = content_id([rule, origin, [content_id(q) for q in group]])
            pending[key] = {"queries": group, "rule": rule, "origin": origin}
        return pending

    def measure(self, folder, demand, release):
        directory = folder / "session"
        attempts, card_ns = self.budget.remaining(directory)
        remaining = self.budget.processing_ns - self.budget.state["processing_ns"]
        native = directory / "native"
        if self.budget.limit.storage_bytes == 0:
            raise AllowanceExhausted("acquisition storage allowance exhausted")
        if remaining <= 0 or ((attempts == 0 or card_ns == 0) and not (native / "execution.json").is_file()):
            raise AllowanceExhausted("cumulative attempt, card time or postprocessing allowance exhausted")
        self.budget.check()
        path = folder / "demand.json"
        if not path.exists():
            write_json(path, demand, replace=False)

        sources = [path.parent for path in sorted(self.sessions.glob("*/session/result.json"))]
        args = SimpleNamespace(**self.options, output=directory, structure_session=sources, budget_seconds=card_ns / 1e9,
                               card_budget_ns=card_ns, attempt_limit=attempts)
        origin = LibraryRelease(asset_path(self.store.root, demand["origin"])) if demand["origin"] else None
        acquisition = self.rules(origin, args)
        queries = demand["queries"]
        receipt = acquisition.run(args, queries, self.deployment, self.budget.limit, self.budget.processing)
        if receipt["status"] not in {"qualified", "incomplete"}:
            return receipt, None

        qualified = [q for q, answer in zip(queries, receipt["answers"], strict=True) if answer["status"] == "qualified"]
        with self.budget.processing() as seconds:
            model = ModelingProcess(args.binary, args.modeling_image, min(seconds, args.timeout))
            increment = LocalLibraryIncrement(directory, qualified, acquisition, model, self.source_costs)
            published = self.store.publish(increment, self.budget.before_commit, expected=release)
        return receipt, published

    def reuse_measurements(self, queries, release):
        requirements = CostRequirements()
        points = {key: requirements.gap(key, q) for key, q in queries.items() if q.get("work") in {"device", "transfer", "collective"}}
        if not points:
            return None
        for path in sorted(self.sessions.glob("*/session/result.json")):
            receipt = json.loads(path.read_bytes())
            if receipt["status"] not in {"qualified", "incomplete"}:
                continue
            directory = path.parent
            native = PublishedCapture(directory / "native")
            products = defaultdict(list)
            for key in points:
                if product := self.source_costs.product(native, queries[key]):
                    products[product].append(key)
            for product, keys in products.items():
                source = native.attempt_id, product
                previous = self.measured.get(source)
                if previous and "transport_profile" not in previous[0] and all(points[key] not in previous[1] for key in keys):
                    continue
                with self.budget.processing() as seconds:
                    model = ModelingProcess(self.options["binary"], self.options["modeling_image"], min(seconds, self.options["timeout"]))
                    group = [queries[key] for key in keys]
                    if source not in self.measured:
                        table = self.source_costs.extract(directory, native, group, model, self.sessions)
                        sampled = set(requirements.samples(source, table, basis="native_measurement"))
                        self.measured[source] = table, sampled
                    table, sampled = self.measured[source]
                    if "transport_profile" in table:

                        # The same physical posts supply both ranks; the staged work may have a different local definition.
                        missing = self.source_costs.uncovered(group, table, model, self.sessions)
                        qualified = [q for index, q in enumerate(group) if index not in missing]
                    else:
                        qualified = [queries[key] for key in keys if points[key] in sampled]
                    if qualified:
                        increment = LocalLibraryIncrement(directory, qualified, None, model, self.source_costs, measured=table)
                        return self.store.publish(increment, self.budget.before_commit, expected=release)
        return None

    def required_services(self, release):
        demand = CostRequirements()
        for path in self.sessions.glob("*/session/result.json"):
            receipt = json.loads(path.read_bytes())
            if receipt["status"] not in {"qualified", "incomplete"}:
                continue
            for asset, points in receipt.get("required_work", {}).items():
                source = json.loads(asset_path(path.parent, asset).read_bytes())
                if content_id(source) != receipt["assets"][asset]:
                    raise ValueError("required work refers to a changed native sample table")
                for index, interval, features in points:
                    definition = source["definitions"][index]
                    demand.gap("training", {"domain": source["domain"], "definition": definition, "interval": interval,
                                            "features": features, "work": definition["work"][interval]["kind"]})
        if not demand.points:
            return {}
        with self.budget.processing() as seconds:
            model = ModelingProcess(self.options["binary"], self.options["modeling_image"], min(seconds, self.options["timeout"]))
            domains = json.loads(release.costs.read_bytes())["domains"]
            _, missing, _ = demand.check(model, domains, self.sessions)
        queries = [demand.work_query(point) for point in missing]
        return {content_id(query): query for query in queries}

    def advance(self, targets, release):
        queries, users = {}, defaultdict(list)
        for target in targets:
            if target["status"] in {"template_gap", "cost_gap"}:
                query = json.loads((self.root / target["evidence"]).read_bytes())
                key = content_id(query)
                queries[key] = query
                users[key].append(target)

        # The store may have advanced while the targets were being predicted.
        current = LibraryRelease(self.store.root)
        if current.identity != release.identity:
            return current
        self.unresolved_services = {}
        extra = {}
        if all(target["status"] == "predicted" for target in targets):
            try:
                extra = self.required_services(release)
            except AllowanceExhausted as failure:
                self.unresolved_services = {"qualification": {"status": "allowance_exhausted", "detail": str(failure)}}
                return None
            queries.update(extra)
        if not queries:
            return None
        answers = dict(self.failed)
        if current.identity:
            for product in json.loads((current.directory / "evidence.json").read_bytes()).values():
                if product["deployment"] == self.deployment:
                    for key in queries.keys() & set(product["queries"]):
                        answers[key] = {"status": "published_evidence_did_not_resolve_query", "product": product["product"]}

        if self.budget.limit.storage_bytes and self.budget.state["processing_ns"] < self.budget.processing_ns:
            try:
                published = self.reuse_measurements({key: q for key, q in queries.items() if key not in answers}, release)
                if published and published.identity != release.identity:
                    return published
            except AllowanceExhausted as failure:
                for key in queries.keys() - answers.keys():
                    answers[key] = {"status": "allowance_exhausted", "detail": str(failure)}
        pending = self.groups(queries, current, answers)
        while pending:
            key = next(iter(pending))
            demand = pending.pop(key)
            folder = self.sessions / key
            group = demand["queries"]
            if not any(content_id(q) in queries and content_id(q) not in answers for q in group):
                continue
            try:
                receipt, published = self.measure(folder, demand, release)
            except AllowanceExhausted as failure:
                receipt, published = {"status": "allowance_exhausted", "detail": str(failure)}, None

            if receipt["status"] == "different_sessions_required" and len(group) > 1:

                # A PD receiver can share its pool only while every admitted member fits simultaneously.
                middle = len(group) // 2
                for part in (group[:middle], group[middle:]):
                    pending[content_id([key, [content_id(q) for q in part]])] = {**demand, "queries": part}
                continue
            for i, query in enumerate(group):
                detail = dict(receipt["answers"][i]) if "answers" in receipt else {"status": receipt["status"]}
                if "detail" in receipt:
                    detail["detail"] = receipt["detail"]
                if folder.exists():
                    detail["session"] = str(folder.relative_to(self.root))
                answers[content_id(query)] = detail
                if receipt["status"] in {"native_failed", "untriggered"}:
                    self.failed[content_id(query)] = detail
            if published and published.identity != release.identity:
                return published
        current = LibraryRelease(self.store.root)
        if current.identity != release.identity:
            return current
        for key, rows in users.items():
            for row in rows:
                row["acquisition"] = answers[key]
        self.unresolved_services = {key: answers[key] for key in extra}
        return None
