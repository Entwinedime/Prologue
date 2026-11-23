"""Construct the SGLang input cohorts shared by observation and control."""

from collections import defaultdict

from prologue.contracts.workload import WorkloadContent


class RequestSubmissions:
    observed = "capture"

    def __init__(self, workload):
        self.requests = workload["requests"]
        self.content = WorkloadContent(self.requests, workload["prefixes"])

    @staticmethod
    def native_id(phase, index):
        return f"{phase}/{index}"

    @staticmethod
    def index(identity):
        return int(identity.rsplit("/", 1)[1])

    def cohorts(self, indices):
        arrivals = defaultdict(list)
        for index in indices:
            arrivals[self.requests[index]["arrival_ns"]].append(index)
        return sorted(arrivals.items())

    def sampling(self, index):
        return {"max_new_tokens": self.requests[index]["output_tokens"], "temperature": 0, "ignore_eos": True}

    def payload(self, phase, indices, prompts):
        return {
            "input_ids": [prompts[index] for index in indices],
            "sampling_params": [self.sampling(index) for index in indices],
            "rid": [self.native_id(phase, index) for index in indices],
            "stream": True,
        }
