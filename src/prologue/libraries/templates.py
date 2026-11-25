"""Store measured template fragments and compose initialized workers in their observed domains."""


from prologue.libraries.release import TemplateCatalog
from prologue.runtime.io import content_id, read_json, write_json


class TemplatePublication:
    def __init__(self, directory, scratch, model, origin, deployment, template_resources):
        self.directory, self.scratch, self.model = directory, scratch, model
        self.template_resources = template_resources
        self.origin = origin
        self.index = read_json(directory / "index.json")
        self.catalog = TemplateCatalog(directory, self.index)
        self.observations = self.index["observations"]
        self.observation = self.observations.setdefault(origin, {"deployment": deployment, "shards": []})
        self.added, self.initialized, self.checks = [], set(), []

    def observe(self, identity, facts, queries):
        if "participants" in facts:
            for worker in facts["participants"]:
                rank = worker["environment"]["parallel"]["rank"]
                self.observe(f"{identity}-{rank}", worker, queries)
            return
        shards = self.observation["shards"]
        shard = next((row for row in shards if row["product"] == identity), None)
        if shard is None:
            source, graph, library = (
                self.scratch / f"{identity}-{part}.json" for part in ("facts", "graph", "template")
            )
            projected = {key: value for key, value in facts.items() if key not in {"sampling", "export_site"}}
            projected["resources"] = self.template_resources(facts)
            write_json(source, projected)
            self.model.run_model("project", graph, source)
            options = ("--root-site", facts["export_site"]) if "export_site" in facts else ()
            self.model.run_model("templates", library, graph, options=options)
            shard = self.catalog.write_shard(self.directory, f"observed/{identity}.json", read_json(library),
                                             sampling=facts.get("sampling"))
            shard["product"] = identity
            shards.append(shard)
            self.added.append((self.origin, shard))

        # Only a qualified initialization can introduce serving participants. A leaf can be stored before its
        # worker exists.
        for query in queries:
            if query.get("status") != "missing_worker_binding":
                self.checks.append((query, shard))
                continue
            shard["initialization"] = {key: facts[key] for key in ("initial_state", "capacity", "configured_settings")}
            if self.catalog.select(query["domain"]) is None:
                self.index["bindings"].append({"domain": query["domain"], "workers": self.origin, "imports": []})
                self.initialized.add(content_id(query["domain"]))
                self.catalog = TemplateCatalog(self.directory, self.index)

    @staticmethod
    def same_worker(first, second):
        a, b = first["header"], second["header"]
        return (a["domain"] == b["domain"] and a["local_domain"] == b["local_domain"]
                and a["resources"][0]["rank"] == b["resources"][0]["rank"])

    def assemble(self, binding):
        workers = self.observations[binding["workers"]]
        domain = content_id(binding["domain"])
        fragments = self.added
        if domain in self.initialized:
            fragments = [
                (origin, shard) for origin, observation in self.observations.items() for shard in observation["shards"]
            ]
        mode = workers["deployment"]["acquisition_mode"]
        fragments = [(origin, shard) for origin, shard in fragments if origin != binding["workers"]
                     and self.observations[origin]["deployment"]["acquisition_mode"] == mode]
        effective = list(binding.get("compiled", workers["shards"]))

        for position, primary in enumerate(effective):
            inputs = [(origin, shard) for origin, shard in fragments if self.same_worker(primary, shard)]
            if not inputs:
                continue
            current = self.export(primary, "worker")
            for origin, shard in inputs:
                observed = self.export(shard, "observed")
                joined = self.scratch / "joined.json"
                self.model.run_model("join-observation", joined, current, observed)
                joined.replace(current)
                sources = binding.setdefault("sources", [])
                if origin not in sources:
                    sources.append(origin)
            effective[position] = self.catalog.write_shard(
                self.directory, f"workers/{domain}/{position}.json", read_json(current)
            )
            binding["compiled"] = effective

    def export(self, shard, name):
        body = self.catalog.read_shard(shard)
        body["definitions"].sort(key=lambda definition: definition["id"])
        for position, definition in enumerate(body["definitions"]):
            if definition.pop("id") != position:
                raise ValueError("incremental compilation requires contiguous local definition positions")
        path = self.scratch / (name + ".json")
        write_json(path, body | shard["header"])
        return path

    def publish(self):
        for binding in self.index["bindings"]:
            self.assemble(binding)

        # Recheck both the saved fragment and any assembled worker now using it.
        for number, (query, shard) in enumerate(self.checks):
            programs = [shard]
            for binding in self.index["bindings"]:
                if self.origin in binding.get("sources", ()):
                    programs.extend(program for program in binding["compiled"] if self.same_worker(shard, program))
            request = self.scratch / "request.json"
            write_json(request, query)
            for index, program in enumerate(programs):
                library = self.export(program, "readback")
                answer = self.scratch / f"answer-{number}-{index}.json"
                self.model.run_model("query-template", answer, library, request)
                selected = read_json(answer)
                if selected["status"] != "covered" or "observation" not in selected:
                    raise ValueError(f"published program does not answer its original template query: {selected}")

        TemplateCatalog(self.directory, self.index)
        write_json(self.directory / "index.json", self.index)
