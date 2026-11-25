"""Verify native stream commands and their recorded CPU or device completion."""

from collections import Counter, defaultdict, deque

from prologue.integrations.platforms.ascend.analysis.sdk_rows import Interval

from .recording import SdkRecording


class StreamEffects:
    def __init__(self, recording: SdkRecording):
        self.recording = recording

    def inspect_elided_waits(self, graph):
        recording = self.recording
        returns = {point: call for point, call in recording.logical_coordinates.items() if call is not None}
        if not returns:
            return

        coordinates = graph["evidence"]["events"]
        predecessors = defaultdict(set)
        for edge in graph["edges"]:
            if edge["kind"] not in {"signal", "submission"}:
                continue
            source, target = edge["source"], edge["target"]
            position = graph["events"][target]["resource"], coordinates[target]["order"]
            if position in returns:
                resource = graph["resources"][graph["events"][source]["resource"]]["identity"]
                predecessors[position].add((edge["kind"], resource, coordinates[source]["order"]))

        _, waits = recording.event_generations
        signals = {(row["thread"], row["begin_counter"]): connection
                   for (_, connection), choices in waits.items() for row, _ in choices}
        for position, (thread, begin) in returns.items():

            # A logical return may publish readiness only after the actual submission and recorded signal.
            recording.require((thread, begin) in signals, "elided wait lacks a native event generation")
            event = recording.sdk.event(signals[thread, begin])
            expected = {
                ("submission", f"thread/{thread}", begin),
                ("signal", f"stream/{event[0]}", recording.sdk.span(event)[1]),
            }
            recording.require(
                expected <= predecessors[position], "elided wait loses its native submission or signal dependency"
            )

    def enclosing_copy_tasks(self, interval, api, owner, stream):
        recording = self.recording
        submitted = recording.queues.inside(interval, stream)
        candidates = []
        for parent in recording.api_ranges.overlapping(api) if submitted else []:
            group = recording.sdk.connections.get(parent.record["args"]["connection_id"], [])
            if (
                group
                and parent.contains(api)
                and recording.owner(parent) == owner
                and recording.queues.inside(parent, stream) == submitted
            ):
                candidates.append(group)
        recording.require(len(candidates) <= 1, "native copy has ambiguous enclosing device work")
        tasks = candidates[0] if candidates else []
        return tasks

    def copy_completion(self, row, effect, copy, tasks):
        witness = effect["completion"]
        completed = None
        if copy and not tasks:
            self.recording.require(witness is not None, "unmatched copy has no raw completion witness")
            completed = self.recording.runtime_records[witness["thread"], witness["begin_counter"]]
            self.recording.require(
                completed["api"] == witness["api"]
                and completed["api"] in {"drvMemcpy", "halMemcpy"}
                and completed["thread"] == row["thread"]
                and completed["result"] == 0
                and row["begin_counter"]
                <= completed["begin_counter"]
                <= completed["end_counter"]
                <= row["end_counter"]
                and all(completed[key] == row[key] for key in ("source", "destination", "bytes")),
                "copy completion does not witness a synchronous transfer of the actual buffers",
            )
        else:
            self.recording.require(
                witness is None, "queued work or stream wait claims an unrelated host-copy completion"
            )
        return completed

    def inspect(self, graph):
        recording = self.recording
        effects = recording.facts["native_effects"]
        recorded = {(effect["thread"], effect["begin_counter"]): effect for effect in effects}
        recording.require(len(recorded) == len(effects), "native effects repeat an API instance")
        expected, coordinates, counts = set(), {}, Counter()
        for row in recording.runtime:
            if row["api"] == "aclrtMemcpy" and row["bytes"] > 0:
                interval = recording.interval(row)
                if recording.window.begin <= interval.begin and interval.end <= recording.window.end:
                    owner = recording.owner(recording.native_api(interval))
                    if owner is not None and owner not in recording.closed_scopes:
                        self.inspect_transfer(owner, row, graph)
                        counts["synchronous_copies"] += 1
            copy = row["api"] == "aclrtMemcpyAsync" and row["bytes"] > 0
            if not copy and row["api"] not in recording.waits:
                continue
            interval = recording.interval(row)
            if interval.end <= recording.window.begin or recording.window.end <= interval.begin:
                continue
            api = recording.native_api(interval)
            owner = recording.owner(api)
            if owner is None or owner in recording.closed_scopes:
                continue
            key = row["thread"], row["begin_counter"]
            expected.add(key)
            recording.require(key in recorded, f"native stream effect is omitted: {key}, {row['api']}")
            effect = recorded[key]
            connection, stream = api.record["args"]["connection_id"], row["stream"]
            recording.require(
                row["result"] == 0 and row["stream_status"] == 0,
                "raw stream operation failed or has no stream identity",
            )
            recording.require(
                (effect["source_scope"], effect["connection"]) == (owner, connection),
                "native effect changes its API or owner",
            )
            resource = recording.facts["resources"][effect["resource"]]
            tasks = recording.sdk.connections.get(connection, [])
            if copy and not tasks:
                tasks = self.enclosing_copy_tasks(interval, api, owner, stream)
                counts["enclosing_group_copies" if tasks else "host_completed_copies"] += 1
            completed = self.copy_completion(row, effect, copy, tasks)
            if completed is not None:
                host = self.inspect_transfer(owner, completed, graph)
                recording.require(effect["resource"] == host and not effect["commands"],
                             "synchronous transfer invents a stream command or changes its actual caller")
                counts["copies"] += 1
                continue
            recording.require(
                (resource["kind"], resource["identity"]) == ("stream", f"stream/{stream}"),
                "native effect changes stream",
            )
            recording.require(all(task[0] == stream for task in tasks), "native effect's SDK tasks change stream")
            instances = defaultdict(deque)
            for task in tasks:
                instances[recording.sdk.key(recording.sdk.records[task]["args"])].append(task)
            expected_tasks = (
                Counter({key: len(values) for key, values in instances.items()}) if tasks else Counter({None: 1})
            )
            recording.require(
                Counter(
                    recording.sdk.key(command["sdk"]) if command["sdk"] is not None else None
                    for command in effect["commands"]
                )
                == expected_tasks,
                "native effect duplicates, omits or invents device work",
            )

            if not copy and not tasks:
                recording.require(
                    all(submission.record["tasks"] == 0 for submission in recording.queues.inside(interval, stream)),
                    "unmatched stream wait has actual driver submissions",
                )
                command, = effect["commands"]
                recording.logical_coordinates[effect["resource"], command["begin"]] = None
                recording.logical_coordinates[effect["resource"], command["end"]] = key
                counts["elided_stream_waits"] += 1

            for command in effect["commands"]:
                begin, end = command["begin"], command["end"]
                interval = Interval(begin, end, effect["resource"], {})
                recording.require(begin < end and recording.span_ranges.enclosing(interval) is not None,
                             "native effect has no retained stream span")
                if copy:
                    recording.require(
                        recording.payload_ranges.enclosing(interval) is not None,
                        "native copy is absent from work payload",
                    )
                else:
                    recording.require(
                        (effect["resource"], begin, end) in recording.wait_intervals, "native stream wait is absent"
                    )
                if command["sdk"] is not None:
                    identity = instances[recording.sdk.key(command["sdk"])].popleft()
                    recording.require(
                        coordinates.setdefault(identity, (begin, end)) == (begin, end),
                        "nested API views duplicate a device task",
                    )
            counts["copies" if copy else "stream_waits"] += 1
        recording.require(set(recorded) == expected, "effect evidence contains an unobserved or auxiliary operation")
        counts["device_measurements"] = recording.sdk.inspect_coordinates(recording.facts, recording.apis, coordinates)
        self.inspect_elided_waits(graph)
        return dict(counts)

    def inspect_transfer(self, source_scope, completed, graph):
        recording = self.recording
        scope = recording.source_scopes[source_scope]
        resources = recording.facts["resources"]
        host, = [index for index, resource in enumerate(resources)
                 if (resource["kind"], resource["identity"]) == ("thread", f"thread/{completed['thread']}")]
        begin, end = completed["begin_counter"], completed["end_counter"]
        recording.require(recording.spans[scope, host, begin, end, "transfer"] > 0,
                     "synchronous copy loses its caller transfer interval")
        recording.require(
            begin < end and recording.payload_ranges.enclosing(Interval(begin, end, host, {})) is not None,
            "synchronous transfer is absent from its caller payload",
        )
        raw = recording.native["scopes"][source_scope]

        # Only a semantic device owner publishes copy readiness; caller envelopes contain observer markers.
        while ("device_entry" not in raw or raw.get("envelope", False)) and raw["parent"] is not None:
            raw = recording.native["scopes"][raw["parent"]]
        if "device_entry" not in raw:
            return host

        owner = recording.source_scopes[raw["id"]]
        port, = [port for port in recording.facts["scopes"][owner]["ports"] if port["name"] == "device.complete"]
        published, = [port for port in graph["scopes"][owner]["ports"] if port["name"] == "device.complete"]
        recording.check_device_end(graph, published["event"], recording.point_connection(raw["device_return"]))
        recording.require((tuple(sorted(port["at"].items())), host, end) in recording.transfer_signals,
                     "synchronous transfer has no readiness edge to its actual enclosing return fence")

        return host
