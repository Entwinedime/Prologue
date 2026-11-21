"""Resolve published source collections for prediction and cost acquisition."""

import json


class SourceProductions:
    """Read the source jobs and their published captures and libraries."""

    def __init__(self, collection):
        collection = collection.resolve(strict=True)
        self.directory = collection
        inputs = json.loads((collection / "input.json").read_text())
        self.plan, self.jobs, self.deployment = inputs["plan"], inputs["jobs"], inputs["deployment"]
        publication = json.loads((collection / "result.json").read_text())
        self.unavailable = publication.get("unavailable", {})
        planned = {str(job["id"]) for job in self.plan["sources"]}
        jobs = [job for job in self.plan["sources"] if str(job["id"]) not in self.unavailable]
        if (not len(jobs) == len(publication["productions"]) == len(publication["captures"])
                or self.jobs.keys() != planned or self.unavailable.keys() - planned):
            raise ValueError("source publication does not match the planned source jobs")
        self.paths = {job["id"]: collection / name for job, name in zip(jobs, publication["productions"], strict=True)}

        # Explicit observations keep their original owners, which may live outside this collection.
        self.captures = {job["id"]: collection / name for job, name in zip(jobs, publication["captures"], strict=True)}


class PredictionSources(SourceProductions):
    """Resolve a target using the framework selected by the prediction entrypoint."""

    def __init__(self, collection, select_primary, task_type):
        super().__init__(collection)
        self.select_primary = select_primary
        self.task_type = task_type

    def select(self, source):
        return self.select_primary(self.plan, source, available=self.paths)

    def failure(self, source):
        primary = self.select_primary(self.plan, source, available={int(key) for key in self.unavailable})
        if primary is None:
            return {"status": "source_missing", "message": "target implementation has no planned primary source"}
        return {"job": primary, **self.unavailable[str(primary)]}

    def jobs_for(self, source):
        primary = self.select(source)
        if primary is None:
            return []
        jobs = [primary]
        for job in self.plan["sources"]:
            consumers = job.get("local_consumers", [])
            if job["id"] in self.paths and any(consumer["source"] == primary for consumer in consumers):
                jobs.append(job["id"])
        return jobs

    def prepare(self, source):
        jobs = self.jobs_for(source)
        if not jobs:
            raise ValueError("target implementation has no published primary source")
        templates = []
        productions = []
        for identity in jobs:
            directory = self.paths[identity].resolve(strict=True)
            publication = json.loads((directory / "result.json").read_text())
            libraries = []
            for name in publication["templates"]:
                path = (directory / name).resolve(strict=True)
                if not path.is_relative_to(directory):
                    raise ValueError("template publication refers outside its production directory")
                templates.append(path)
                libraries.append(json.loads(path.read_text()))
            productions.append(libraries)
        document = self.task_type(productions[0], productions[1:]).document(source)
        return templates, productions[0], document
