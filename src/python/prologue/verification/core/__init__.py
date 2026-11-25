"""Qualify recorded requests, publication histories and independently trained assets."""

from statistics import mean

from prologue.common.io import read_json
from prologue.host.acquisition.recording import PublishedCapture
from prologue.host.workflow.latency import RequestLatency
from prologue.verification.core.adapters import framework_evidence


class RecordedSubmissions:
    clock = "monotonic_raw_ns"

    @classmethod
    def read(cls, capture, sessions):
        records = []
        for index, session in enumerate(sessions):
            path = capture / "instances" / str(index) / "request-delivery.json"
            delivery, = read_json(path)["capture"]
            records.append(cls(session, delivery))
        return records

    def __init__(self, session, delivery):
        if delivery["clock"] != self.clock:
            raise ValueError("native request submissions require the shared RAW clock")
        self.origin_ns = delivery["origin_ns"]
        requests = session["content"]["requests"]
        self.requests = {request["id"]: request for request in requests}
        self.aliases = dict(zip(session["source_requests"], self.requests, strict=True))
        if len(self.aliases) != len(requests):
            raise ValueError("each submitted request needs a unique native and workload identity")

        # The driver records a single submission time for all requests in one message.
        pairs = [(index, row) for row in delivery["submissions"] for index in row["requests"]]
        if sorted(index for index, _ in pairs) != list(range(len(requests))):
            raise ValueError("native submissions must cover every request exactly once")
        self.by_request = {requests[index]["id"]: row for index, row in pairs}

    def receives(self, records):
        rounds, received = [], set()
        for record in records:
            time_ns = record["received_raw_ns"] - self.origin_ns
            members = [self.aliases[name] for name in record["requests"]]
            if not members or time_ns < 0 or (rounds and time_ns < rounds[-1]["time_ns"]):
                raise ValueError("receive rounds require nonempty members and nondecreasing times")
            for member in members:
                if member in received or time_ns < max(
                    self.requests[member]["arrival_ns"], self.by_request[member]["submitted_ns"]
                ):
                    raise ValueError(f"receive repeats {member} or precedes its submission")
                received.add(member)
            rounds.append({"time_ns": time_ns, "requests": members})
        if received != self.requests.keys():
            raise ValueError("receive rounds do not cover every request exactly once")
        return rounds


class ReferenceRecording(PublishedCapture):
    target_fields = ("model", "instances", "workload")

    @classmethod
    def input_difference(cls, job, target):
        if set(target) != set(cls.target_fields):
            return "target fields"
        for field in cls.target_fields:
            if job[field] != target[field]:
                return field

        native = framework_evidence().capture_input(job)
        order = [row["id"] for row in native.workload["requests"]]
        return None if native.submission_groups == [order] else "native submission groups"

    @staticmethod
    def sessions(capture, instances):
        return [read_json(capture / "instances" / str(index) / "session.json")
                for index, _ in enumerate(instances)]

    @staticmethod
    def session_difference(source, instances, sessions):
        target = framework_evidence().serving_input(source)
        if len(instances) != len(target.instances):
            return "instances"

        for observed, session, instance in zip(instances, sessions, target.instances, strict=True):
            captured = dict(session["serving_input"])
            configuration = captured.pop("configuration")
            if ((observed["id"], observed["mode"]) != (instance.identity, instance.mode)
                    or configuration != instance.configuration.values):
                return f"instance:{instance.identity}"

            # Native content and routing hooks consume separate fields from this recorded session.
            if session["content"] != target.workload or captured != target.workload:
                return "workload"
        return None

    def training(self, *, templates=None, costs=None):
        shared, unknown = [], []
        samples = 0
        records = templates.recordings if templates is not None else ()

        # Publication checks each template against its native attempt. The identity survives moving the release.
        if self.attempt_id in records:
            shared.append({"template_recording": self.attempt_id})

        if costs is not None:
            asset = read_json(costs)
            sources = asset.get("sources", [])
            origins = asset.get("sample_recordings", [])
            samples = len(sources)
            if not samples or not isinstance(origins, list) or len(origins) != samples:
                unknown.append({"cost_asset": str(costs), "reason": "sample recordings unavailable"})
            else:
                for index, recordings in enumerate(origins):
                    if (
                        not isinstance(recordings, list)
                        or not recordings
                        or not all(isinstance(r, str) and r for r in recordings)
                    ):
                        unknown.append({"sample": index, "source": sources[index]})
                    elif self.attempt_id in recordings:
                        shared.append({"sample": index, "source": sources[index]})

        status = "reference_used_for_training" if shared else "training_origin_unverified" if unknown else "separate"
        return {
            "status": status,
            "target_attempt": self.attempt_id,
            "template_recordings": len(records),
            "sample_files": samples,
            "shared": shared,
            "unknown": unknown,
        }


def compare_request_metrics(reference, predicted):
    actual = {row["id"]: row for row in reference["requests"]}
    estimates = {row["id"]: row for row in predicted["requests"]}
    if (
        len(actual) != len(reference["requests"])
        or len(estimates) != len(predicted["requests"])
        or actual.keys() != estimates.keys()
    ):
        raise ValueError("request metrics require the same unique request identities")

    differences = {identity: {"id": identity} for identity in actual}
    accuracy = {}
    for metric in ("ttft_ns", "tpot_ns"):
        errors = []
        zero = missing = 0
        for identity, row in actual.items():
            observed, estimate = row[metric], estimates[identity][metric]
            if (observed is None) != (estimate is None):
                raise ValueError(f"request {identity} changes the applicability of {metric}")
            delta = None if observed is None else estimate - observed
            differences[identity][metric] = delta

            # A one-token request has no TPOT. A measured zero has a signed error but no relative denominator.
            if observed is None:
                missing += 1
            elif observed == 0:
                zero += 1
            else:
                errors.append(abs(delta) / observed)
        accuracy[metric] = {"mape_percent": mean(errors) * 100 if errors else None, "requests": len(errors),
                            "total_requests": len(actual), "not_applicable": missing, "zero_denominators": zero}

    observed_rate, predicted_rate = reference["tokens_per_second"], predicted["tokens_per_second"]
    return {
        "accuracy": accuracy,
        "delta_from_baseline": {
            "requests": list(differences.values()),
            "p95_ttft_ns": predicted["p95_ttft_ns"] - reference["p95_ttft_ns"],
            "tokens_per_second": None
            if observed_rate is None or predicted_rate is None
            else predicted_rate - observed_rate,
        },
    }


def publication_marks(raw):
    scopes = [scope for scope in raw["scopes"] if scope.get("publication")]
    if not scopes:
        return

    intervals = {row["name"]: row for row in raw["host_observation"]["intervals"]}
    for scope in scopes:
        marker = intervals[f"prologue.scope/{scope['id']}"]
        for token in scope["publication"]:
            yield token["id"], token["generated_tokens"], marker["begin_raw_ns"], marker["end_raw_ns"]


class FixedPublications:
    def __init__(self, case, requests, origin_ns):
        self.requests = requests
        self.counts = {row["id"]: row["output_tokens"] for row in requests}
        self.positions = case.graph.positions
        self.ports, self.marks = {}, {}
        for rank, (contents, raw, _) in case.sources.items():
            if raw["environment"]["implementation"]["disaggregation_mode"] == "prefill":
                continue

            # The projected port identifies what the graph publishes. Its source scope's return is a separate witness.
            for scope in contents.graph["scopes"]:
                for item in scope["bindings"].get("publication", ()):
                    key = item["id"], item["generated_tokens"]
                    for port in scope["ports"]:
                        if port["name"] == "publication" and port["direction"] == "output":
                            address = rank, port["event"]
                            if self.ports.setdefault(key, address) != address:
                                raise ValueError(f"token {key} has multiple graph publication ports")

            for request, token, begin, end in publication_marks(raw):
                key = request, token
                mark = rank, begin - origin_ns, end - origin_ns
                if self.marks.setdefault(key, mark) != mark:
                    raise ValueError(f"token {key} has multiple native publications")

    def coverage_gaps(self):
        expected = {(name, token) for name, count in self.counts.items() for token in range(1, count + 1)}
        gaps = {}
        for source, records in (("graph", self.ports), ("native", self.marks)):
            missing, extra = expected - records.keys(), records.keys() - expected
            if missing or extra:
                gaps[source] = {"missing": len(missing), "first_missing": min(missing, default=None),
                                "unexpected": len(extra), "first_unexpected": min(extra, default=None)}
        return gaps

    def compare(self, times):
        checked = first_tokens = last_tokens = differences = max_width = 0
        first = None
        for key, (rank, begin, end) in self.marks.items():
            if key not in self.ports:
                continue
            address = self.ports[key]
            replayed = times[self.positions[address]]
            request, token = key
            checked += 1
            first_tokens += token == 1
            last_tokens += token == self.counts.get(request)
            max_width = max(max_width, end - begin)

            # Detailed capture timestamps the end of the send bracket; the actual socket send lies inside it.
            if rank != address[0] or replayed != end:
                differences += 1
                if first is None or end < first["native_bracket_ns"][1]:
                    first = {"request": request, "generated_tokens": token, "native_rank": rank, "graph_port": address,
                             "native_bracket_ns": [begin, end], "replayed_ns": replayed}
        return {"basis": "detailed_scheduler_send_bracket_end", "checked_tokens": checked,
                "checked_first_tokens": first_tokens, "checked_last_tokens": last_tokens,
                "max_send_bracket_ns": max_width, "different_tokens": differences, "first_difference": first}

    def metrics(self, times):
        outputs = [{"request": name, "generated_tokens": token, "finished": token == count,
                    "time_ns": times[self.positions[self.ports[name, token]]]}
                   for name, count in self.counts.items() for token in range(1, count + 1)]
        return RequestLatency(outputs, self.requests).report()
