"""Associate measured services with the normalized work intervals in a graph."""

import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RendezvousService:
    lower_ns: int
    upper_ns: int

    @classmethod
    def after_peers(cls, spans):

        # Starting at the last ready endpoint gives the shortest of the measured finish-minus-ready spans.
        lower = min(first for first, _ in spans)
        upper = min(last for _, last in spans)
        if lower > 0 or lower == upper == 0:
            return cls(lower, upper)
        return None

    @property
    def duration_ns(self):
        return (self.lower_ns + self.upper_ns) // 2


class WorkCatalog:
    def __init__(self, graph_path, identities_path):
        self.graph = json.loads(graph_path.read_text())
        identities = json.loads(identities_path.read_text())
        self.domain, self.definitions = identities["domain"], identities["definitions"]
        self.features = identities["features"]
        self.window = self.graph["evidence"]["window_ns"]
        if self.window != identities["window_ns"]:
            raise ValueError("work identities and coarse graph come from different native observations")
        self.positions = {
            (start, finish): (definition, interval, point)
            for start, finish, definition, interval, point in identities["intervals"]
        }
        expected = {(piece["start"], piece["finish"]) for piece in self.graph["work"]}
        if (len(expected) != len(self.graph["work"]) or expected != self.positions.keys()
                or len(self.positions) != len(identities["intervals"])):
            raise ValueError("work identities must cover each coarse graph interval once")

    def __iter__(self):
        for piece in self.graph["work"]:
            definition, interval, point = self.positions[piece["start"], piece["finish"]]
            if self.definitions[definition]["work"][interval]["kind"] != piece["kind"]:
                raise ValueError("normalized work identity changes the measured interval kind")
            yield piece, definition, interval, self.features[point]


class ServiceSamples:
    def __init__(self, domain, definitions=()):
        self.domain = domain
        self.definitions = list(definitions)
        self.rows = {}
        self.external = {}

    def add(self, definition, interval, features, duration_ns, copy=None, collective=None):
        key = definition, interval, json.dumps([features, copy, collective], sort_keys=True)
        row = self.rows.setdefault(key, {
            "definition": definition, "interval": interval, "features": features, "observations_ns": [],
        })
        if copy is not None:
            row["copy"] = {"bytes": copy[0], "direction": copy[1]}
        if collective is not None:
            payload = row.setdefault("collective", {"dtype": collective[0], "algorithm": collective[1],
                                                      "elements": collective[2], "max_clock_radius_ns": 0})
            radius = duration_ns.upper_ns - duration_ns.duration_ns
            payload["max_clock_radius_ns"] = max(payload["max_clock_radius_ns"], radius)
            duration_ns = duration_ns.duration_ns
        row["observations_ns"].append(duration_ns)

    def add_external(self, identity, kind, features, duration_ns, *, collective=None):
        key = json.dumps(identity, sort_keys=True)
        if key not in self.external:
            self.external[key] = len(self.definitions)
            self.definitions.append({"external": identity, "work": [{"kind": kind}]})
        self.add(self.external[key], 0, features, duration_ns, collective=collective)

    def document(self, unmeasured):
        return {
            "domain": self.domain, "definitions": self.definitions,
            "samples": list(self.rows.values()), "unmeasured": dict(unmeasured),
        }
