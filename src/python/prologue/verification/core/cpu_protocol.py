"""Build independent host message readiness and payload evidence."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import prod

from prologue.common.io import read_json
from prologue.verification.core.action_identity import ActionIdentity
from prologue.verification.core.adapters import framework_evidence
from prologue.verification.engine import CausalIndex, FindingLedger


@dataclass(frozen=True, slots=True)
class NativeCpuCall:
    key: tuple
    phase: str
    thread: int
    entered: tuple[int, int]
    posted: tuple[int, int] | None
    returned: tuple[int, int]
    counters: tuple[int, int]
    execution: dict
    descriptor: dict

    def identity(self):
        work = self.execution["work"]
        context = work["context"]
        normalized = {
            **work,
            "context": {**context, "cpu_transfer": framework_evidence().cpu_transfer.identity(context["cpu_transfer"])},
        }
        return {"operation": self.execution["operation"], "work": ActionIdentity.work_identity(normalized)}

    def source(self):
        return {"post": list(self.key), "phase": self.phase}

    def services(self):
        operation = self.execution["operation"]
        if self.posted is None:
            return ((self.entered[1], self.returned[1], operation),)
        before, after = framework_evidence().cpu_protocol.post_operations(operation)
        return ((self.entered[1], self.posted[1], before), (self.posted[1], self.returned[1], after))


class CpuProtocolEvidence:
    def __init__(self, capture, owners):
        self.capture = capture
        self.counts, self.findings = Counter(), FindingLedger()
        self.native_threads = defaultdict(lambda: defaultdict(list))
        self.native_calls, self.native_times = defaultdict(list), {}
        self.native_groups, self.predicted_groups, self.actors = owners, {}, {}
        self.messages = {origin: defaultdict(dict) for origin in ("native", "predicted")}
        self.points = {origin: set() for origin in self.messages}
        self.edges = {origin: set() for origin in self.messages}
        self.last_cpu, self.returns, self.horizons = {}, {}, {}
        self.services, self.service_returns = {}, {}
        self.posts, self.post_times = set(), {}
        self.readiness, self.ready_events = {}, set()
        self.ready_inputs = defaultdict(dict)
        self.ready_bounds = {}
        self.service_gates = set()
        self.communication_edges = set()
        self.finished_services, self.finished_ready = set(), set()
        self.terminals = []
        self.terminals_checked = False
        self.sequence = Counter()
        self.predicted_scope = False

    def missing(self, reason, **detail):
        self.counts[reason] += 1
        self.findings.add("cpu_protocol_evidence", {}, {"reason": reason, **detail})

    def point(self, origin, rank):
        self.sequence[origin, rank] -= 1
        point = rank, self.sequence[origin, rank]
        self.points[origin].add(point)
        return point

    def native_graph(self, rank, graph):
        for event, (node, evidence) in enumerate(zip(graph["events"], graph["evidence"]["events"], strict=True)):
            resource = graph["resources"][node["resource"]]
            if resource["kind"] == "thread":
                thread = int(resource["identity"].removeprefix("thread/"))
                positions = self.native_threads[rank, thread]
                positions[evidence["order"]].append((rank, event))
        self.read_native_calls(rank)

    @staticmethod
    def execution(interval, operation):
        execution = interval["execution"]
        if (
            execution["operation"] != operation
            or type(execution["in_flow"]) is not bool
            or not isinstance(execution["work"], dict)
        ):
            raise ValueError("CPU protocol interval changes its native operation or entering context")
        return execution

    def native_interval(self, rank, key, phase, interval, descriptor, transport=None):
        begin, end = interval["begin_counter"], interval["end_counter"]
        entered, returned = interval["begin_raw_ns"], interval["end_raw_ns"]
        if (
            any(type(value) is not int for value in (begin, end, entered, returned))
            or begin > end
            or entered > returned
        ):
            raise ValueError("CPU protocol has an invalid native interval")

        thread = interval["thread"]
        positions = self.native_threads[rank, thread]
        endpoints = []
        for counter, stamp in ((begin, entered), (end, returned)):
            if any(event >= 0 for _, event in positions[counter]):
                self.missing("native_protocol_counter_ties_coarse_boundary", rank=rank, thread=thread, counter=counter)
            point = self.point("native", rank)
            positions[counter].append(point)
            self.native_times[point] = stamp
            endpoints.append(point)

        first, last = endpoints
        posted = None
        if transport is None:
            self.edges["native"].add((first, last))
        else:
            posted = self.point("native", rank)
            self.native_times[posted] = transport["post_raw_ns"]
            self.edges["native"].update(((first, posted), (posted, last)))

        call = NativeCpuCall(key, phase, thread, first, posted, last, (begin, end), interval["execution"], descriptor)
        self.native_calls[rank].append(call)
        return call

    def post(self, origin, rank, key, descriptor, posted):
        group, source, target, tag, position = key
        route = group, source, target, tag
        if (
            route != tuple(descriptor[name] for name in ("group", "source", "target", "tag"))
            or source == target
            or rank != (target if descriptor["receiving"] else source)
            or type(descriptor["receiving"]) is not bool
            or type(position) is not int
            or position < 0
            or any(type(size) is not int or size < 0 for size in descriptor["shape"])
            or type(descriptor["elements"]) is not int
            or descriptor["elements"] != prod(descriptor["shape"])
            or type(descriptor["element_bytes"]) is not int
            or descriptor["element_bytes"] <= 0
        ):
            raise ValueError("CPU protocol post has inconsistent route, actor or tensor geometry")

        peers = self.messages[origin][key]
        if rank in peers:
            raise ValueError("CPU protocol repeats a local post identity")
        wire = tuple(descriptor[name] for name in ("dtype", "elements", "element_bytes"))
        peers[rank] = {"posted": posted, "consumed": None, "wire": wire}
        self.counts[origin + "_posts"] += 1

        return peers[rank]

    def native_transport(self, post):
        transport = framework_evidence().cpu_protocol.transport(post)
        if transport is None:
            self.counts["native_unobserved_transport_posts"] += 1
        else:
            self.counts["native_transport_posts"] += 1
            self.counts["native_transport_completions"] += transport["completion_raw_ns"] is not None
        return transport

    def native_readiness(self, peers):
        if len(peers) != 2 or any(peer["transport"] is None for peer in peers.values()):
            return None
        last_post = max(peer["transport"]["post_raw_ns"] for peer in peers.values())
        windows = {
            rank: framework_evidence().cpu_protocol.ready_window(
                peer["transport"], last_post, None if peer["consumed"] is None else self.native_times[peer["consumed"]]
            )
            for rank, peer in peers.items()
        }
        return last_post, windows

    def read_native_calls(self, rank):
        path = self.capture / "protocol" / f"rank-{rank}.json"
        if not path.is_file():
            self.missing("missing_native_protocol", rank=rank)
            return

        document = read_json(path)
        if document["backend"] != "gloo" or set(document["operations"]) != {"send", "recv", "isend", "irecv"}:
            raise ValueError("CPU protocol observation declares a different operation scope")
        transport_clock = document.get("transport_clock")

        if transport_clock is not None and transport_clock != "monotonic_raw_ns":
            raise ValueError("Gloo transport observation requires the shared RAW clock")
        start = {tuple(row["route"]): row["position"] for row in document["start_positions"]}
        end = {tuple(row["route"]): row["position"] for row in document["end_positions"]}
        observed = Counter()
        posting_threads = defaultdict(set)

        for route, floor in start.items():
            if not 0 <= floor <= end[route]:
                raise ValueError("CPU protocol has a reversed observation horizon")
        self.horizons[rank] = {route: (start.get(route, 0), limit) for route, limit in end.items()}

        for post in document["posts"]:
            key = tuple(post["key"])
            route, position = key[:-1], key[-1]
            posting_threads[route].add(post["interval"]["thread"])

            if not position < end[route]:
                raise ValueError("CPU protocol post lies outside its declared channel history")
            source, target = key[1:3]
            if (
                self.native_groups[source] != self.native_groups[target]
                or self.native_groups[rank] != self.native_groups[source]
            ):
                raise ValueError("CPU protocol crosses independent native worlds")

            operation = post["operation"]
            if operation not in {"send", "recv", "isend", "irecv"} or post["descriptor"]["receiving"] != (
                operation in {"recv", "irecv"}
            ):
                raise ValueError("CPU protocol operation disagrees with its receive/send descriptor")

            self.execution(post["interval"], operation)
            descriptor = post["descriptor"]
            transport = self.native_transport(post) if transport_clock is not None else None
            call = self.native_interval(rank, key, "post", post["interval"], descriptor, transport)
            endpoint = self.post("native", rank, key, descriptor, call.posted or call.entered)
            endpoint["transport"] = transport

            if post["consumed"] is not None:
                if operation in {"isend", "irecv"}:
                    self.execution(post["consumed"], operation + "_wait")
                    wait = self.native_interval(rank, key, "wait", post["consumed"], descriptor)
                    if post["consumed"]["begin_counter"] < post["interval"]["end_counter"]:
                        raise ValueError("CPU Work wait starts before its post returns")

                    self.edges["native"].add((call.returned, wait.entered))
                    consumed = wait.returned
                else:

                    # Blocking operations consume the instrumented call itself.
                    if post["consumed"] != post["interval"]:
                        raise ValueError("blocking CPU work disagrees with its recorded return")
                    consumed = call.returned

                endpoint["consumed"] = consumed

            observed[route] += position >= start.get(route, 0)

        for route, limit in end.items():
            if observed[route] != limit - start.get(route, 0):
                self.missing("native_posts_missing_from_horizon", rank=rank, route=route)
            if len(posting_threads[route]) > 1:
                self.missing("native_channel_has_multiple_posting_threads", rank=rank, route=route)

    def declare_participants(self, row):
        self.predicted_groups = {
            group["id"]: (group["rank"], instance["id"])
            for instance in row["instances"]
            for group in instance["groups"]
        }

    def observe_graph(self, row):
        graph, events = row["graph"], row["event_ids"]
        ports = {
            port["name"]: events[port["event"]]
            for port in graph["scopes"][0]["ports"]
            if port["name"] in {"call.enter", "call.return", "transport.post"}
        }
        entered = next(port for port in graph["scopes"][0]["ports"] if port["name"] == "call.enter")
        rank = graph["resources"][graph["events"][entered["event"]]["resource"]]["rank"]

        if rank in self.last_cpu:
            self.edges["predicted"].add((self.last_cpu[rank], (rank, ports["call.enter"])))
        if graph["scopes"][0]["role"] == "cpu_service":
            self.service(graph, events, rank, ports)
        else:
            self.returns[ports["call.return"]] = rank

    def observe_completion(self, row):
        event = row["event"]
        if event in self.service_returns:
            rank, _, _ = self.service_returns[event]
            self.finished_services.add(event)
            self.last_cpu[rank] = rank, event
        if event in self.posts:
            self.post_times[event] = row["time_ns"]
        if event in self.ready_events:
            self.finished_ready.add(event)

    def observe_edge(self, row):
        edge = row["source"], row["target"]
        if row["target"] in self.ready_events:
            if row["relation"] != "communication" or row["source"] in self.ready_inputs[row["target"]]:
                raise ValueError("CPU readiness has a duplicate or noncommunication prerequisite")
            self.ready_inputs[row["target"]][row["source"]] = row["delay_ns"]
            self.communication_edges.add(edge)

        if edge in self.service_gates:
            if row["relation"] != "communication" or row["delay_ns"] != 0:
                raise ValueError("CPU API return changes its local readiness dependency")
            self.service_gates.remove(edge)
            self.communication_edges.add(edge)

    def observe_window(self, row):
        if row["event"] in self.ready_events:
            self.ready_bounds[row["event"]] = row["begin_ns"], row["end_ns"]

    def observe_values(self, row):
        if "outputs" not in row["values"]:
            return

        rank, _ = self.predicted_groups[row["group"]]
        if self.returns.pop(row["boundary"]) != rank:
            raise ValueError("CPU protocol action return belongs to another participant")
        self.last_cpu[rank] = rank, row["boundary"]

    def observe_scope(self, record):
        if (
            self.predicted_scope
            or record["backend"] != "gloo"
            or set(record["operations"]) != {"send", "recv", "isend", "irecv"}
        ):
            raise ValueError("CPU broker observation has a competing or different protocol scope")

        self.actors = {actor["rank"]: actor["group"] for actor in record["participants"]}
        if (
            len(self.actors) != len(record["participants"])
            or set(self.actors.values()) != set(self.predicted_groups)
            or len(self.actors) != len(self.predicted_groups)
        ):
            raise ValueError("CPU broker observation omits or repeats a native participant")

        self.predicted_scope = True

    def observe_protocol(self, row):
        record, group = row["record"], row["group"]
        if record["phase"] == "scope":
            self.observe_scope(record)
            return
        rank, _ = self.predicted_groups[group]
        actor = record["rank"]
        if not self.predicted_scope or self.actors[actor] != group:
            raise ValueError("CPU broker actor changes its native participant")

        key = tuple(record["key"])
        phase = record["phase"]
        if phase == "readiness":
            if (key, rank) in self.readiness or type(record["event"]) is not int:
                raise ValueError("CPU protocol repeats or omits its local readiness event")
            self.readiness[key, rank] = (record["event"], False)
            self.ready_events.add(record["event"])

            return

        if phase == "ready":
            self.predicted_ready(key, rank, actor, record, row["time_ns"])
            return
        execution = record["execution"]
        operation = execution["operation"]
        self.execution(record, operation)

        if record["event"] is None:
            if execution["in_flow"]:
                raise ValueError("CPU serving operation has no execution service")
            point = self.point("predicted", rank)

            if rank in self.last_cpu:
                self.edges["predicted"].add((self.last_cpu[rank], point))
            self.last_cpu[rank] = point
        else:
            point = rank, record["event"]

        if phase == "post":
            if self.predicted_groups[self.actors[key[1]]][1] != self.predicted_groups[self.actors[key[2]]][1]:
                raise ValueError("CPU broker P2P crosses independent native worlds")
            if operation not in {"send", "recv", "isend", "irecv"} or record["descriptor"]["receiving"] != (
                operation in {"recv", "irecv"}
            ):
                raise ValueError("CPU post changes its native operation")

            posted = point
            if record["event"] is not None:
                expected_rank, expected_key, descriptor, work, boundary = self.services.pop(record["event"])
                if (expected_rank, expected_key, descriptor, work) != (rank, key, record["descriptor"], execution):
                    raise ValueError("CPU post differs from its submitted service")

                posted = rank, boundary

            endpoint = self.post("predicted", actor, key, record["descriptor"], posted)
            endpoint.update(time_ns=row["time_ns"], execution=execution)
        elif phase == "consumed":
            peers = self.messages["predicted"][key]
            if len(peers) != 2 or "ready_ns" not in peers[actor] or peers[actor]["ready_ns"] > row["time_ns"]:
                raise ValueError("CPU broker consumption precedes the matching peer post")

            endpoint = peers[actor]
            if endpoint["consumed"] is not None:
                raise ValueError("CPU Work consumption is recorded twice")
            post_operation = endpoint["execution"]["operation"]

            if operation != post_operation + ("_wait" if post_operation in {"isend", "irecv"} else ""):
                raise ValueError("CPU consumption changes its posted operation")
            if record["event"] is not None:
                if record["event"] not in self.finished_services or self.service_returns[record["event"]] != (
                    rank,
                    key,
                    execution,
                ):
                    raise ValueError("CPU consumption precedes or changes its own service return")

            endpoint["consumed"] = point
        else:
            raise ValueError("CPU protocol has an unknown observation phase")

    observations = {
        "resources": declare_participants,
        "graph": observe_graph,
        "completion": observe_completion,
        "edge": observe_edge,
        "boundary_time": observe_window,
        "action_values": observe_values,
        "native_protocol": observe_protocol,
    }

    def observe(self, row):
        if handle := self.observations.get(row["kind"]):
            handle(self, row)

    def predicted_ready(self, key, rank, actor, record, stamp):
        peers = self.messages["predicted"][key]
        if len(peers) != 2 or "ready_ns" in peers[actor]:
            raise ValueError("CPU readiness is repeated or lacks either post")

        parents = {peer["posted"][1] for peer in peers.values() if peer["posted"][1] >= 0}
        event, service = record["event"], record["service_ns"]
        if (service is not None and (type(service) is not int or service < 0)) or set(
            record["prerequisites"]
        ) != parents:
            raise ValueError("CPU readiness changes its post prerequisites or service duration")

        if event is None:
            if parents or service != 0:
                raise ValueError("Untimed CPU readiness includes modeled transfer work")
        else:
            if self.readiness[key, rank] != (event, False) or event not in self.finished_ready:
                raise ValueError("CPU readiness was published before its declared event completed")
            delay = 0 if service is None else service

            if self.ready_inputs[event] != dict.fromkeys(parents, delay):
                raise ValueError("CPU readiness differs from the executor's actual dependency edges")
            if service is None and event not in self.ready_bounds:
                raise ValueError("Unknown CPU readiness has no explicit completion condition")

            posted = [self.post_times[parent] for parent in parents]
            if posted and stamp < max(posted) + delay:
                raise ValueError("CPU readiness completed before its supplied service lower bound")
            self.readiness[key, rank] = (event, True)

        peers[actor]["ready_ns"] = stamp
        self.counts["predicted_ready_endpoints"] += 1

    def service(self, graph, events, rank, ports):
        root = graph["scopes"][0]
        binding = root["bindings"]["cpu_protocol"]
        key, descriptor, execution = tuple(binding["key"]), binding["descriptor"], binding["execution"]
        operation = execution["operation"]
        self.execution(binding, operation)
        posting = operation in {"send", "recv", "isend", "irecv"}

        if not posting and operation not in {"isend_wait", "irecv_wait"}:
            raise ValueError("CPU service declares an unobserved protocol operation")
        number = 2 if posting else 1
        expected_ports = {"call.enter": events[0], "call.return": events[number]}

        if posting:
            expected_ports["transport.post"] = events[1]
        work = [
            {"scope": 0, "resource": 0, "kind": "host", "start": first, "finish": first + 1} for first in range(number)
        ]

        if (
            len(graph["scopes"]) != 1
            or len(graph["resources"]) != 1
            or len(graph["events"]) != number + 1
            or ports != expected_ports
            or graph["resources"][0]["kind"] != "thread"
            or graph["resources"][0]["role"] != "control"
            or not execution["in_flow"]
            or root["work"] != execution["work"]
            or graph["work"] != work
            or root["kv_accesses"]
            or root["slot_accesses"]
            or graph["edges"]
        ):
            raise ValueError("CPU service changes its host work or protocol boundaries")

        if posting:
            self.services[events[0]] = rank, key, descriptor, execution, events[1]
            self.posts.add(events[1])

        if operation not in {"isend", "irecv"} and (key, rank) in self.readiness:

            # Check actual executor edges, not only the service's claimed prerequisites or observed timestamps.
            self.service_gates.add((self.readiness[key, rank][0], events[number]))
        self.service_returns[events[number]] = rank, key, execution
        self.counts["predicted_service_intervals"] += number

    def connect_messages(self, origin):
        for key, peers in self.messages[origin].items():
            self.counts[origin + "_consumptions"] += sum(peer["consumed"] is not None for peer in peers.values())
            self.counts[origin + "_unconsumed_posts"] += sum(peer["consumed"] is None for peer in peers.values())
            if len({peer["wire"] for peer in peers.values()}) != 1:
                raise ValueError("CPU protocol endpoints disagree on wire dtype or element count")

            if len(peers) == 2:
                transports = [row["transport"] for row in peers.values() if row.get("transport") is not None]
                if len(transports) == 2:
                    if transports[0]["slot"] != transports[1]["slot"]:
                        raise ValueError("Gloo transport endpoints disagree on the native message slot")
                    self.counts["native_transport_pairs"] += 1

                self.counts[origin + "_messages"] += 1
                for destination in peers.values():
                    if destination["consumed"] is not None:

                        # Both posts precede either completion; the peer need not have consumed its Work.
                        for source in peers.values():
                            if source["posted"] != destination["consumed"]:
                                self.edges[origin].add((source["posted"], destination["consumed"]))
                continue

            actor, endpoint = next(iter(peers.items()))
            if endpoint["consumed"] is None:
                self.counts[origin + "_unconsumed_tail_posts"] += 1
                continue

            peer = key[2] if actor == key[1] else key[1]
            first, last = self.horizons.get(peer, {}).get(key[:-1], (0, 0))
            if origin == "native" and peer in self.horizons and key[-1] < first:
                self.counts["native_initial_external_completions"] += 1
            elif origin == "native" and peer in self.horizons and key[-1] >= last:
                self.terminals.append((key, endpoint["consumed"]))
                self.counts["native_terminal_external_completions"] += 1
            else:
                self.missing(origin + "_missing_peer_post", key=key, actor=actor)

    def native(self, native, dependencies):
        for (rank, thread), positions in self.native_threads.items():
            if rank in native and positions and all(event < 0 for points in positions.values() for _, event in points):
                self.missing("native_protocol_thread_has_no_coarse_anchor", rank=rank, thread=thread)

            # Equal clock readings do not identify an event or order two different events.
            previous = ()
            for _, current in sorted(positions.items()):
                self.edges["native"].update((source, target) for source in previous for target in current)
                previous = current

        self.connect_messages("native")
        if self.findings:
            return None
        graph = self.joined_graph("native", native, dependencies)

        # An unobserved suffix is harmless only if it cannot lead back to any retained serving event.
        relevant = [False] * len(graph.positions)
        for rank, original in native.items():
            for event in original.next:
                relevant[graph.positions[rank, event]] = True

        for position in range(len(relevant) - 1, -1, -1):
            relevant[position] |= any(relevant[target] for target in graph.successors[position])
        for key, consumed in self.terminals:
            if relevant[graph.positions[consumed]]:
                self.missing("native_terminal_message_can_reach_serving", key=key)
            else:
                self.counts["native_terminal_outside_serving"] += 1

        self.terminals_checked = True
        return graph

    def joined_graph(self, origin, graphs, dependencies):
        graph = CausalIndex.joined(graphs, dependencies)
        graph.add_events(self.points[origin])
        for source, target in self.edges[origin]:
            if source != target:
                graph.connect(source, target, "communication" if source[0] != target[0] else "control")
        graph.prepare()
        return graph

    def prediction(self, predicted, predicted_edges, *, complete):
        if not self.predicted_scope:
            self.missing("missing_predicted_protocol_scope")
        pending = sum(not published for _, published in self.readiness.values())

        if self.services or self.service_gates or self.service_returns.keys() != self.finished_services or pending:
            if complete:
                raise ValueError("CPU protocol has an unobserved service or unfinished readiness")

            # A stopped prefix can still have live work; it cannot qualify the missing protocol paths.
            self.missing(
                "unfinished_predicted_protocol",
                services=len(self.services),
                gates=len(self.service_gates),
                returns=len(self.service_returns.keys() - self.finished_services),
                pending_readiness=pending,
            )

        self.connect_messages("predicted")

        # Keep each run's observed paths available; protocol completeness gates cross-run negative conclusions.
        return self.joined_graph("predicted", predicted, predicted_edges)

    def report(self):
        return {
            "complete": not self.findings and not self.returns and (not self.terminals or self.terminals_checked),
            "unreturned_actions": len(self.returns),
            **self.counts,
            "scope": (
                "Observed CPU posts, readiness and consumption within each run. Native call boundaries and "
                "predicted action returns anchor the protocol paths. Prefix posts and unconsumed tails have "
                "explicit counts. Cross-run payloads use checked post witnesses when available."
            ),
        }
