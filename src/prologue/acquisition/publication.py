"""Read archived source collections for explicit import and offline cost preparation."""

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
