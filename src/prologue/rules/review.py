"""Review template sharing against observed calls and the core's bindings."""

import json
from collections import defaultdict
from dataclasses import dataclass


def key(document):
    return json.dumps(document, sort_keys=True)


@dataclass(frozen=True, slots=True)
class RootBinding:
    domain: dict
    query: dict
    graph: dict
    definitions: dict


class RuleReview:
    def __init__(self, bank, document):
        self.bank = bank
        self.cases = document["cases"]
        if not isinstance(self.cases, list) or not self.cases:
            raise ValueError("a rule review requires native cases")
        names = [case["id"] for case in self.cases]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("rule cases need distinct nonempty IDs")

        domains = {}
        for source in bank["sources"]:
            domain = source["domain"]
            batch = domains.setdefault(key(domain), {"domain": domain, "queries": {}})
            for root in source["roots"]:
                query = root["query"]
                batch["queries"][key(query)] = query
        self.batches = [{**batch, "queries": list(batch["queries"].values())} for batch in domains.values()]

    def bindings(self, results):
        queries = {}
        owners = defaultdict(set)
        for batch, decisions in zip(self.batches, results, strict=True):
            domain = batch["domain"]
            scope = key(domain)
            for query, decision in zip(batch["queries"], decisions, strict=True):
                if decision["status"] != "covered":
                    raise ValueError(f"cannot review {domain['configuration']['id']} {query}: {decision['status']}")
                graph = decision["graph"]
                indices = {tuple(call["path"]): identity for call, identity in
                           zip(graph["calls"], decision["definition_ids"], strict=True)}
                for identity in set(indices.values()):
                    owners[identity].add(scope)
                queries[scope, key(query)] = RootBinding(domain, query, graph, indices)
        return queries, owners

    def witness(self, endpoint, queries):
        domain = endpoint["domain"]
        query = endpoint["query"]
        path = endpoint["path"]
        if not isinstance(path, list) or any(type(slot) is not int or slot < 0 for slot in path):
            raise ValueError("a witness path must contain nonnegative child slots")

        identity = queries[key(domain), key(query)].definitions[tuple(path)]
        definitions = self.bank["definitions"]
        if not definitions[identity]["device_work"] or definitions[identity]["children"]:
            raise ValueError("a sharing witness must be a leaf with observed device work")

        # An imported child can expand successfully without being observed by this witness.
        for source in self.bank["sources"]:
            if source["domain"] != domain or source["source"]["origin_raw_ns"] != endpoint["origin_raw_ns"]:
                continue
            for root in source["roots"]:
                if root["query"] != query:
                    continue
                observed = root["definition"]
                for slot in path:
                    children = definitions[observed]["children"]
                    if slot >= len(children):
                        observed = None
                        break
                    observed = children[slot]
                    if not isinstance(observed, int):
                        break
                if observed == identity:
                    return identity
        raise ValueError("the witness is not a local observation of the selected call")

    def compare(self, before, after, prior_rule, rule):
        old, old_owners = self.bindings(before)
        new, new_owners = self.bindings(after)
        affected = []
        for scope, binding in new.items():
            prior = old[scope]
            graph_changed = binding.graph != prior.graph
            changed = sum(old_owners.get(prior.definitions.get(path)) != new_owners[identity]
                          for path, identity in binding.definitions.items())
            if changed or graph_changed:
                affected.append({"domain": binding.domain, "query": binding.query, "sharing_changed_calls": changed,
                                 "graph_changed": graph_changed})

        cases = []
        attestations = defaultdict(set)
        withdrawn = set()
        definitions = self.bank["definitions"]
        for case in self.cases:
            left, right = case["left"], case["right"]
            expected = case["shared"]
            if type(expected) is not bool or left["domain"] == right["domain"]:
                raise ValueError("rule cases require a Boolean expectation across distinct domains")

            left_id, right_id = self.witness(left, new), self.witness(right, new)
            operation = definitions[left_id]["operation"]
            if operation != definitions[right_id]["operation"]:
                raise ValueError("both witnesses must observe the same operation")

            old_left = old[key(left["domain"]), key(left["query"])]
            old_right = old[key(right["domain"]), key(right["query"])]
            was_shared = old_left.definitions[tuple(left["path"])] == old_right.definitions[tuple(right["path"])]
            shared = left_id == right_id
            if shared == expected:
                attestations[operation].add(shared)
                if was_shared and not shared:
                    withdrawn.add(operation)
            cases.append({**case, "operation": operation, "previously_shared": was_shared,
                          "observed_shared": shared, "passed": shared == expected})

        previous = set(prior_rule["shared_operations"])
        proposed = set(rule["shared_operations"])
        missing = sorted(name for name in proposed - previous if attestations[name] != {False, True})
        missing.extend(sorted(previous - proposed - withdrawn))
        passed = (not missing and all(case["passed"] for case in cases)
                  and not any(row["graph_changed"] for row in affected))
        return {"passed": passed, "cases": cases, "affected": affected, "missing_evidence": missing}
