"""Verify resource history, allocation effects and in-flight memory lifetimes."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain

from prologue.verification.core.actions import ActionIdentity
from prologue.verification.engine import FindingLedger


@dataclass(frozen=True)
class MemoryAccess:
    rows: tuple
    columns: tuple
    write: bool
    start: int
    finish: int
    invocation: int
    publication: tuple | None = None

    def conflicts(self, other):
        if self.write and other.write and self.publication is not None and self.publication == other.publication:
            return False
        return (self.write or other.write) and all(
            max(left[0], right[0]) < min(left[1], right[1])
            for left, right in ((self.rows, other.rows), (self.columns, other.columns))
        )


class StateLifetimes:
    def __init__(self):
        self.active = defaultdict(set)
        self.ends = defaultdict(set)
        self.widths = {}
        self.accesses = defaultdict(list)
        self.kv_bindings = defaultdict(list)
        self.fields = defaultdict(set)
        self.required = defaultdict(dict)
        self.identities = {}
        self.counts = Counter()
        self.findings = FindingLedger()

    def invocation(self, graph, identity, events, entered):
        self.close_bindings()
        self.entered = entered
        self.identities[entered] = identity
        exposed = {port["tensor"]["storage"] for port in graph["scopes"][0]["ports"]
                   if port["use"] == "data" and port["direction"] == "output"}
        for index, scope in enumerate(graph["scopes"]):
            for port in scope["ports"]:
                if port["use"] != "data":
                    continue
                tensor = port["tensor"]
                if 0 in tensor["shape"] or (tensor["allocation"] is not None and tensor["storage"] not in exposed):
                    continue
                resource = graph["events"][port["event"]]["resource"]
                intervals = []
                if port["direction"] == "input":
                    _, finish = self.tensor_interval(graph, index, resource)
                    intervals.append((False, port["event"], finish))
                if tensor["writer"] is not None:
                    start, finish = self.tensor_interval(graph, tensor["writer"], resource)
                    intervals.append((True, start, finish))

                # Versions identify producers; all epochs still share the same physical allocation for overwrite safety.
                for write, start, finish in intervals:
                    self.bind((identity["group"], "tensor", tensor["storage"], 0), MemoryAccess(
                        (0, 1), (0, 1), write, events[start], events[finish], entered,
                    ), {"group": identity["group"], "storage": tensor["storage"]})
                self.counts["tensor_ports_checked"] += 1
            for access in scope["slot_accesses"]:
                for request in access["requests"]:
                    publication = (request, access["publication"]) if access.get("publication") else None
                    self.fields[access["pool"], access["field"], request].add(
                        MemoryAccess(
                            (0, 1),
                            (0, 1),
                            access["access"] == "write",
                            events[access["start"]],
                            events[access["finish"]],
                            entered,
                            publication,
                        )
                    )
            for access in scope["kv_accesses"]:
                layout = access["slice"]
                key = access["pool"], access["layer"]
                if self.widths.setdefault(key, layout["token_bytes"]) != layout["token_bytes"]:
                    raise ValueError("one physical KV layer has inconsistent token widths")
                interval = (access["layer"], (layout["begin"], layout["end"]), access["access"] == "write",
                            events[access["start"]], events[access["finish"]])
                for region in access["regions"]:
                    self.accesses[access["pool"], region["request"]].append((region["begin"], region["end"], interval))

    @staticmethod
    def tensor_interval(graph, scope, resource):
        owner = graph["scopes"][scope]
        if owner.get("communication"):
            completion, = [port for port in owner["ports"] if "rendezvous" in port]
            boundaries = completion["rendezvous"], completion["event"]
            if any(graph["events"][event]["resource"] == resource for event in boundaries):
                return boundaries
        prefix = "device" if graph["resources"][resource]["kind"] == "stream" else "call"
        boundaries = {port["name"]: port["event"] for port in owner["ports"]}
        start, finish = (
            boundaries[prefix + ".enter"],
            boundaries[prefix + (".complete" if prefix == "device" else ".return")],
        )
        if any(graph["events"][event]["resource"] != resource for event in (start, finish)):
            raise ValueError("tensor access has no lifetime on its declared resource")
        return start, finish

    def close_bindings(self):
        self.counts["field_bindings_missing"] += len(self.fields)
        for pool, field, request in self.fields:
            self.findings.add(
                    "state_binding_evidence",
                    self.identities[self.entered],
                    {
                        "pool": pool,
                        "field": field,
                        "request": request,
                        "reason": "declared state access has no actual physical row binding",
                    },
                )
        self.fields.clear()

        # Safety needs every declared region, including calls that cannot be paired across runs.
        for key, accesses in self.accesses.items():
            bound = sorted(self.kv_bindings[key])
            missing = []
            for left, right in sorted({(left, right) for left, right, _ in accesses if left < right}):
                covered = left
                for begin, end in bound:
                    if begin > covered:
                        break
                    covered = max(covered, end)
                    if covered >= right:
                        break
                self.counts["kv_regions_checked"] += 1
                if covered < right:
                    missing.append([left, right])
            if missing:
                self.counts["kv_regions_unbound"] += len(missing)
                self.findings.add("binding_evidence", self.identities[self.entered], {
                    "pool": key[0], "request": key[1], "regions": missing,
                    "reason": "declared KV access lacks complete actual physical binding coverage",
                })
        self.accesses.clear()
        self.kv_bindings.clear()

    def bind(self, address, value, detail):
        if value in self.active[address]:
            return

        # Logical release never separates two accesses to the same still-active physical address.
        for previous in self.active[address]:
            if previous.invocation != self.entered and value.conflicts(previous):
                self.required[previous.finish].setdefault(value.start, {
                    **detail, "previous_invocation": previous.invocation, "invocation": self.entered,
                })
        self.active[address].add(value)
        self.ends[value.finish].add(address)
        self.counts["physical_access_intervals"] += 1
        self.counts[address[1] + "_access_intervals"] += 1

    def state_bound(self, row):
        if row["invocation"] != self.entered:
            raise ValueError("state bindings must accompany their submitted graph")
        key = row["pool"], row["field"], row["request"]
        for value in self.fields.pop(key):
            self.bind((row["pool"], "field", row["slot"], row["field"]), value, {
                "pool": row["pool"], "field": row["field"], "slot": row["slot"],
            })
        self.counts["field_bindings_checked"] += 1

    def bound(self, row):
        if row["invocation"] != self.entered:
            raise ValueError("physical bindings must accompany their submitted graph")
        begin = row["begin"]
        for page, first, last in row["ranges"]:
            end = begin + last - first
            for left, right, interval in self.accesses[row["pool"], row["request"]]:
                lo, hi = max(begin, left), min(end, right)
                if lo >= hi:
                    continue
                layer, columns, write, start, finish = interval
                value = MemoryAccess(
                    (first + lo - begin, first + hi - begin), columns, write, start, finish, self.entered
                )
                self.bind((row["pool"], "kv", page, layer), value, {"pool": row["pool"], "page": page, "layer": layer})
            begin = end
        self.kv_bindings[row["pool"], row["request"]].append((row["begin"], begin))

    def completed(self, event):
        for address in self.ends.pop(event, ()):
            remaining = {access for access in self.active[address] if access.finish != event}
            if remaining:
                self.active[address] = remaining
            else:
                del self.active[address]

    def released(self, row):
        pages = set(row["returned_pages"])
        if not pages:
            return
        pending = {
            page for pool, kind, page, _ in self.active if kind == "kv" and pool == row["group"] and page in pages
        }
        self.counts.update(release_calls=1, released_pages=len(pages), released_pages_inflight=len(pending))

    def compare(self, graph, event_ranks):
        self.close_bindings()
        self.counts["pending_conflicts"] = sum(map(len, self.required.values()))
        self.counts["pending_conflicts_ordered"] = 0
        if not self.required:
            return

        # Reuse the shared run-local graph, including paths that cross workers through CPU protocols.
        sources = list(self.required)
        for begin in range(0, len(sources), 1024):
            current = sources[begin:begin + 1024]
            masks = graph.reachable_masks([(event_ranks[source], source) for source in current])
            for bit, source in enumerate(current):
                for target, detail in self.required[source].items():
                    if source != target and masks[graph.positions[event_ranks[target], target]] & (1 << bit):
                        self.counts["pending_conflicts_ordered"] += 1
                    else:
                        self.findings.add("state_lifetime", self.identities[detail["invocation"]], {
                            **detail, "previous_finish": source, "next_start": target,
                            "reason": "overlapping pending state accesses lack an ordering path",
                        })

    def report(self):
        return {
            **self.counts,
            "scope": (
                "Prediction-only cross-invocation tensor, KV and request-field safety. Nonempty boundary-visible "
                "tensor epochs share their allocation; every nonempty KV region and field access needs an actual "
                "physical binding, independently of cross-run pairing. Completed accesses retire before later "
                "submissions; logical release preserves pending hazards. Equal publications may write concurrently; "
                "readers join pending writers. Excludes intra-invocation hazards, initialization/values, native "
                "intra-call release timing and unexercised reuse. Bindings do not prove cross-run row "
                "correspondence."
            ),
        }


class NativePoolHistory:
    def __init__(self, history):
        self.history = history
        self.page_size = history["initial"]["page_size"]
        self.capacity = history["initial"]["page_capacity"]
        self.pages, self.slots = defaultdict(list), defaultdict(list)
        self.ownership = defaultdict(list)
        for record in history["events"]:
            operation = record["operation"]
            window = record["end_raw_ns"], record["begin_raw_ns"]
            if operation["kind"] == "release_pages" and not operation["grouped"]:

                # Grouped frees only stage pages; the later ungrouped call actually returns them.
                for page in {index // self.page_size for index in operation["indices"]}:
                    self.pages[page].append(window)
            elif operation["kind"] == "release_request":
                self.slots[operation["slot"]].append(window)
                self.ownership["request", operation["slot"]][-1][2] = record["begin_raw_ns"]
            elif operation["kind"] == "slots" and record["result"] is not None:
                for request, slot in zip(operation["requests"], record["result"], strict=True):
                    self.ownership["request", slot].append([request, record["end_raw_ns"], float("inf")])
        for windows in (*self.pages.values(), *self.slots.values()):
            windows.sort()

        for record in history["state_events"]:
            slot, operation = record["slot"], record["operation"]
            if slot is None:
                continue
            claims = self.ownership[record["field"], slot]
            if operation == "allocate":
                claims.append([None, record.get("end_raw_ns"), float("inf")])
            elif operation == "bind":
                claims[-1][0] = record["request"]
            elif operation == "release":
                claims[-1][2] = record.get("begin_raw_ns")

    def owns(self, field, slot, request, window):
        begin, end = window

        # API return and release entry bound certain ownership; an overlapping or unmeasured mutation proves less.
        return sum(owner == request and first is not None and last is not None and first <= begin <= end <= last
                   for owner, first, last in self.ownership[field, slot]) == 1

    def ranges(self, binding, epoch):
        result, begin = [], binding["begin"]
        if begin < 0:
            raise ValueError("request binding starts before token zero")
        for page, first, last in binding["ranges"]:
            if not (0 <= page < self.capacity and 0 <= first < last <= self.page_size):
                raise ValueError("request binding lies outside its physical pool")
            end = begin + last - first
            result.append((begin, end, (page, epoch(page)), first))
            begin = end
        return result

    def resolve(self, binding, window, clock):
        begin, end = window
        if begin is None or end is None or end < begin or clock != self.history["clock"]:
            raise ValueError("physical binding has no common measured release clock")
        slot = binding["slot"]
        histories = [self.pages[page] for page, _, _ in binding["ranges"]]
        if slot is not None:
            histories.append(self.slots[slot])

        # An observation window must not cross a release; otherwise its physical generation is ambiguous.
        if any(start <= end and finish >= begin for windows in histories for finish, start in windows):
            return None
        ranges = self.ranges(binding, lambda page: bisect_right(self.pages[page], (begin,)))
        identity = None if slot is None else (slot, bisect_right(self.slots[slot], (begin,)))
        return identity, ranges


class BindingComparison:
    def __init__(self, histories, sessions):
        self.pools = {group: NativePoolHistory(history) for group, history in histories.items()}
        self.sessions = sessions
        self.native, self.predicted = {}, defaultdict(list)
        self.invocations = {}
        self.page_epochs, self.slot_epochs = Counter(), Counter()
        self.findings, self.counts = FindingLedger(), {}
        self.unconsumed_snapshots = 0
        self.lifetimes = StateLifetimes()

    @staticmethod
    def coverage(ranges):
        merged = []
        for begin, end in sorted(ranges):
            if begin == end:
                continue
            if merged and begin <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([begin, end])
        return merged

    def observations(self, contents, raw, group):
        graph = contents.graph
        for binding in self.pools[group].history["bindings"]:
            scope = graph["scopes"][contents.source_ids[binding["scope"]]]
            events = {port["name"]: port["event"] for port in scope["ports"]}
            window = tuple(
                graph["evidence"]["events"][events[name]]["time_ns"] for name in ("call.enter", "call.return")
            )
            yield binding, group, window

        for index, scope in enumerate(raw["scopes"]):
            transfer = scope.get("native_transfer")
            if not transfer or transfer["kind"] != "kv":
                continue
            member, = raw["scopes"][scope["submitted_by"]]["work"]["members"]
            executions = scope["service"]["executions"]
            if not executions:
                continue
            window = min(item["begin_raw_ns"] for item in executions), max(item["end_raw_ns"] for item in executions)
            for side, pool in (("source", group), ("target", self.sessions[transfer["session"]])):
                if pool not in self.pools:
                    continue
                size = self.pools[pool].page_size
                count = member["new_tokens"]
                pages = transfer[f"{side}_pages"]
                if member["context_tokens"] % size or len(pages) != (count + size - 1) // size:
                    raise ValueError("native transfer pages do not describe its aligned request chunk")

                # Transport owns frozen pages; it no longer consumes the request's table slot.
                ranges = [[page - 1, 0, min(size, count - position * size)] for position, page in enumerate(pages)]
                yield {"scope": index, "request": member["id"], "slot": None,
                       "begin": member["context_tokens"], "ranges": ranges}, pool, window

    def binding_gap(self, identity, pool, request, reason):
        self.findings.add("binding_evidence", identity, {"pool": pool, "request": request, "reason": reason})

    def observed(self, contents, raw, group, action_identities):
        graph = contents.graph
        accesses, identities, seen = defaultdict(list), {}, set()
        for owner, scope in enumerate(graph["scopes"]):
            if not ActionIdentity.selected(owner, scope):
                continue
            identity = action_identities[owner]
            key = (group, owner)
            identities[key] = identity
            for index in contents.paths(owner):
                for access in graph["scopes"][index]["kv_accesses"]:
                    pool = contents.pool(access["pool"], group)
                    for region in access["regions"]:
                        accesses[key, pool, region["request"]].append((region["begin"], region["end"]))

        for binding, pool, window in self.observations(contents, raw, group):
            owner = contents.source_ids[binding["scope"]]
            while not ActionIdentity.selected(owner, graph["scopes"][owner]):
                owner = graph["scopes"][owner]["parent"]
                if owner is None:
                    raise ValueError("native physical binding has no independently identified action")
            identity = action_identities[owner]
            key = (group, owner), pool, binding["request"]
            seen.add(key)
            resolved = self.pools[pool].resolve(binding, window, graph["evidence"]["clock"])
            if resolved is None:
                self.binding_gap(identity, pool, binding["request"],
                                 "release overlaps the observed binding window")
                continue
            slot, ranges = resolved
            end = ranges[-1][1] if ranges else binding["begin"]
            if any(left < binding["begin"] or right > end for left, right in accesses[key]):
                self.binding_gap(identity, pool, binding["request"],
                                 "observed physical binding does not cover the source graph access")
                continue
            coverage = self.coverage(accesses[key])
            if not coverage:

                # Preparing a page table does not consume KV; execution only reports bindings used by graph accesses.
                self.unconsumed_snapshots += 1
                continue
            value = identity, slot, ranges, coverage
            if key in self.native and self.native[key] != value:
                self.binding_gap(identity, pool, binding["request"],
                                 "observations disagree on an action's physical binding")
            else:
                self.native[key] = value

        # Missing evidence remains missing even when prediction also omits its binding records.
        for key, pool, request in accesses.keys() - seen:
            self.binding_gap(identities[key], pool, request,
                                 "source KV access has no observed physical binding")

    def initialized(self, row):
        for instance in row["instances"]:
            settings = instance["settings"]
            geometry = settings["page_size"], settings["max_total_tokens"] // settings["page_size"]
            for group in instance["groups"]:
                pool = self.pools.get(group["id"])
                if pool is not None and geometry != (pool.page_size, pool.capacity):
                    raise ValueError("native and predicted physical pools have different geometry")

    def released(self, row):
        self.lifetimes.released(row)
        group = row["group"]
        for field, epochs in (("returned_pages", self.page_epochs), ("returned_slots", self.slot_epochs)):
            for value in row[field]:
                epochs[group, value] += 1

    def invocation(self, graph, identity, events, key):
        entered = next(port["event"] for port in graph["scopes"][0]["ports"] if port["name"] == "call.enter")
        self.invocations[events[entered]] = key, identity
        self.lifetimes.invocation(graph, identity, events, events[entered])

    def bound(self, row):
        self.lifetimes.bound(row)
        invocation, identity = self.invocations[row["invocation"]]
        group = row["pool"]
        key = invocation, group, row["request"]
        if group not in self.pools:
            self.findings.add(
                    "binding_evidence",
                    identity,
                    {"pool": group, "request": row["request"], "reason": "no observed physical pool for this action"},
                )
            return
        pool = self.pools[group]
        ranges = pool.ranges(row, lambda page: self.page_epochs[group, page])
        slot = row["slot"]
        slot_identity = None if slot is None else (slot, self.slot_epochs[group, slot])
        self.predicted[key].append((slot_identity, ranges, row["begin"]))

    @staticmethod
    def pair(native, predicted, forward, reverse):
        if native in forward and forward[native] != predicted or predicted in reverse and reverse[predicted] != native:
            return False
        forward[native], reverse[predicted] = predicted, native
        return True

    def compare(self, pairs):
        pages, slots = defaultdict(dict), defaultdict(dict)
        matched, intervals, unpaired = 0, 0, 0
        targets = set(pairs.values())
        expected = {(pairs[key[0]], *key[1:]) for key in self.native if key[0] in pairs}
        for key in self.predicted.keys() - expected:
            if key[0] in targets:
                self.findings.add(
                        "binding_evidence",
                        {"group": key[0][0], "event": key[0][1]},
                        {"pool": key[1], "request": key[2], "reason": "no observed physical binding for this action"},
                    )

        for key, (identity, native_slot, native_ranges, expected_coverage) in self.native.items():
            group = key[1]
            if key[0] not in pairs:
                unpaired += 1
                continue
            failures, coverage = [], []
            rows = self.predicted.get((pairs[key[0]], *key[1:]), [])
            if not rows:
                failures.append("observed request binding is absent from prediction")
            for slot, ranges, begin in rows:
                if native_slot is not None and (
                    slot is None or not self.pair(native_slot, slot, slots[group, "forward"], slots[group, "reverse"])
                ):
                    failures.append("request-slot sharing or release generation differs")
                end = ranges[-1][1] if ranges else begin
                native_begin = native_ranges[0][0] if native_ranges else 0
                native_end = native_ranges[-1][1] if native_ranges else native_begin
                if begin < native_begin or end > native_end:
                    failures.append("predicted access exceeds the observed request extent")
                    continue
                coverage.append((begin, end))
                cursor = 0
                for start, stop, page, offset in ranges:
                    intervals += 1
                    while cursor < len(native_ranges) and native_ranges[cursor][1] <= start:
                        cursor += 1
                    index = cursor
                    while index < len(native_ranges) and native_ranges[index][0] < stop:
                        left, right, actual_page, actual_offset = native_ranges[index]
                        overlap = max(start, left)
                        if offset + overlap - start != actual_offset + overlap - left:
                            failures.append("page offset differs at a common request token")
                        if not self.pair(actual_page, page, pages[group, "forward"], pages[group, "reverse"]):
                            failures.append("KV page sharing or release generation differs")
                        index += 1

            # A page table may exceed the accessed region; compare the observed graph's actual read/write union.
            if self.coverage(coverage) != expected_coverage:
                failures.append("predicted intervals differ from the observed KV access coverage")
            if failures:
                self.findings.add("physical_binding", identity,
                                  {"pool": group, "request": key[2], "reasons": sorted(set(failures))})
            else:
                matched += 1
        self.counts = {
            "native_request_bindings": len(self.native),
            "request_bindings_matched": matched,
            "request_bindings_unpaired": unpaired,
            "predicted_request_bindings_unpaired": sum(key[0] not in targets for key in self.predicted),
            "snapshots_without_kv_access": self.unconsumed_snapshots,
            "page_only_bindings": sum(value[1] is None for value in self.native.values()),
            "predicted_regions": sum(map(len, self.predicted.values())),
            "page_intervals": intervals,
            "scope": (
                "Observed request extents, page offsets and page/slot alias identities across recorded release "
                "epochs. Transport consumes frozen pages, not table slots; native enqueue ownership and retained "
                "pages/metadata are checked separately. This comparison does not prove allocator or in-flight "
                "lifetime correctness."
            ),
        }


class FieldEffects:
    """Compare row ownership without substituting request slots or replaying allocator policy."""

    def __init__(self):
        self.capacities, self.free = {}, {}
        self.bindings, self.owners = {}, {}
        self.rows, self.failures = defaultdict(list), Counter()
        self.operations = 0

    def declare(self, field, capacity, free):
        slots = set(free)
        if (type(capacity) is not int or capacity <= 0 or field in self.free
                or len(slots) != len(free) or not slots <= set(range(capacity))):
            raise ValueError("invalid observed slot pool declaration")
        self.capacities[field] = capacity
        self.free[field] = slots

    def apply(self, record):
        field, operation, slot = record["field"], record["operation"], record["slot"]
        free = self.free[field]
        if operation == "allocate" and slot is None:
            if free:
                raise ValueError("slot allocation failed with available rows")
            self.failures[field] += 1
        else:
            if type(slot) is not int or not 0 <= slot < self.capacities[field]:
                raise ValueError("slot operation lies outside its physical pool")
            key = field, slot
            if operation == "allocate":
                free.remove(slot)
            elif slot in free:
                raise ValueError("slot operation accesses a free row")
            elif operation == "bind":
                request = record["request"]
                if key in self.bindings and self.bindings[key] != request:
                    raise ValueError("slot binding overwrites another live request")
                if self.owners.setdefault((field, request), slot) != slot:
                    raise ValueError("request owns multiple rows in one slot pool")
                self.bindings[key] = request
            elif operation == "release":
                if key in self.bindings:
                    del self.owners[field, self.bindings.pop(key)]
                free.add(slot)
            else:
                raise ValueError("unknown observed slot operation")

            # Group by the physical slot inside this run; comparison uses the resulting histories as a multiset.
            self.rows[key].append((operation, record.get("request")))
        self.operations += 1

    def row_histories(self):
        return Counter((field, tuple(history)) for (field, _), history in self.rows.items())

    def prefix_of(self, source):
        if not self.capacities.items() <= source.capacities.items():
            return False
        histories, available = self.row_histories(), source.row_histories()
        if not self.failures <= source.failures:
            return False

        # Match specific row histories first; a shorter prefix must not consume their only compatible native row.
        for (field, history), count in sorted(histories.items(), key=lambda item: len(item[0][1]), reverse=True):
            for key, capacity in available.items():
                if key[0] != field or key[1][:len(history)] != history:
                    continue
                used = min(count, capacity)
                count -= used
                available[key] -= used
                if not count:
                    break
            if count:
                return False
        return True


class ResourceEffects:
    def __init__(self, page_size):
        self.page_size = page_size
        self.pages, self.slots, self.nodes = {}, {}, {}
        self.operations = []
        self.fields = FieldEffects()

    @staticmethod
    def name(names, identity):
        return names.setdefault(identity, len(names))

    def page(self, page):

        # Renaming hides absolute addresses, not physical reuse across allocation lifetimes.
        return self.name(self.pages, page)

    def slot(self, slot):
        return self.name(self.slots, slot)

    def ranges(self, indices):
        ranges = []
        for index in indices:
            page, offset = divmod(index, self.page_size)
            identity = self.page(page)
            if ranges and ranges[-1][0] == identity and ranges[-1][2] == offset:
                ranges[-1][2] += 1
            else:
                ranges.append([identity, offset, offset + 1])
        return ranges

    def apply(self, operation):
        kind = operation["kind"]
        if kind == "slots":
            operation["slots"] = [self.slot(slot) for slot in operation["slots"]]
        elif kind == "allocate":
            operation["ranges"] = self.ranges(operation.pop("indices"))
        elif kind == "release_request":
            slot = operation["slot"]
            operation["slot"] = self.slot(slot)
        elif kind == "release_pages":
            pages = operation["pages"]
            if not pages:
                return

            # Release order within one native free call is immaterial; grouping and alias identity are not.
            operation["pages"] = sorted(self.page(page) for page in pages)
        elif kind == "prefix":
            if "node" in operation:
                operation["node"] = self.name(self.nodes, operation["node"])
            if "indices" in operation:
                operation["ranges"] = self.ranges(operation.pop("indices"))
            elif "pages" in operation:
                operation["ranges"] = [[self.page(page), 0, self.page_size] for page in operation.pop("pages")]
        self.operations.append(operation)

    def native(self, history, scopes, cache_disabled):
        for field, pool in history["initial"].get("state_slots", {}).items():
            self.fields.declare(field, pool["capacity"], pool["free"])
        for record in history.get("state_events", []):
            self.fields.apply(record)
        for field, free in history.get("state_free", {}).items():
            if self.fields.free[field] != set(free) or len(free) != len(set(free)):
                raise ValueError("native slot operations disagree with final free rows")

        for record in history["events"]:
            observed = record["operation"]
            kind = observed["kind"]
            if kind == "prefix" and cache_disabled:

                # Disabled native cache methods are no-ops; the control view makes no core prefix call for them.
                continue
            operation = {"kind": kind}
            if kind == "slots":
                operation.update(requests=observed["requests"], slots=record["result"])
            elif kind == "allocate":
                members = scopes[record["scope"]]["work"]["members"]
                requests = [[member["id"], member["new_tokens"]] for member in members]
                if record["indices"] is None or sum(count for _, count in requests) != len(record["indices"]):
                    raise ValueError("native allocation lacks its complete request demand or result")
                operation.update(requests=requests, indices=record["indices"])
            elif kind == "release_request":
                operation.update(request=observed["request"], slot=observed["slot"])
            elif kind == "release_pages":
                if observed["grouped"]:
                    continue
                operation["pages"] = sorted({index // self.page_size for index in observed["indices"]})
            elif kind == "prefix":
                action = observed["operation"]
                operation["operation"] = action
                if action in {"match", "insert"}:
                    tokens = observed["key_tokens"]
                    length = len(tokens) // self.page_size * self.page_size
                    operation.update(tokens=tokens[:length], node=record["result"]["node"])
                    if action == "match":
                        operation["indices"] = record["result"]["indices"]
                    else:
                        operation.update(
                            indices=observed["indices"][:length], existing_tokens=record["result"]["existing_tokens"]
                        )
                elif action == "lock":
                    operation.update(node=observed["node"], acquire=observed["acquire"], delta=record["result"])
                elif action == "evict":
                    operation.update(tokens=observed["tokens"], evicted_tokens=record["result"])
                else:
                    raise ValueError("unknown observed prefix operation")
            else:
                raise ValueError("unknown observed resource operation")
            self.apply(operation)

    def predicted(self, row, requests):
        observed, result = row["operation"], row["result"]
        kind = observed["kind"]

        # Queries have no observed mutation counterpart; request-prefix rebinding is checked at consumed page tables.
        if kind in {"declare_resources", "capacity", "free_slots", "bind_prefix"}:
            return
        if kind == "state_slots":
            field, operation = observed["field"], observed["operation"]
            if operation == "declare":
                self.fields.declare(field, observed["capacity"], list(range(observed["capacity"])))
            elif operation == "available":
                if result != len(self.fields.free[field]):
                    raise ValueError("state slot availability differs from observed ownership")
            else:
                value = {
                    "field": field,
                    "operation": operation,
                    "slot": result if operation == "allocate" else observed["slot"],
                }
                if operation == "bind":
                    value["request"] = requests[observed["request"]]
                if operation == "release" and row["returned_fields"] != [[field, observed["slot"]]]:
                    raise ValueError("state slot release differs from its invalidation record")
                self.fields.apply(value)
            return
        operation = {"kind": kind}
        if kind == "slots":
            operation.update(requests=[requests[member["request"]] for member in observed["members"]],
                             slots=[member["slot"] for member in result])
        elif kind == "allocate":
            operation.update(
                requests=[[requests[member["request"]], member["tokens"]] for member in observed["members"]],
                indices=chain.from_iterable(member["indices"] for member in result["members"]),
            )
        elif kind == "release_request":
            slot, = row["returned_slots"]
            operation.update(request=requests[observed["request"]], slot=slot)
        elif kind == "release_pages":
            if observed["pages"] != row["returned_pages"]:
                raise ValueError("resource release differs from its invalidation record")
            operation["pages"] = observed["pages"]
        elif kind == "prefix":
            action = observed["operation"]
            operation["operation"] = action
            if action in {"match", "insert"}:
                pages = result["pages"] if action == "match" else observed["pages"]
                operation.update(tokens=observed["tokens"], node=result["node"], pages=pages)
                if action == "insert":
                    operation["existing_tokens"] = result["existing_tokens"]
            elif action == "lock":
                operation.update(node=observed["node"], acquire=observed["acquire"], delta=result)
            elif action == "evict":
                operation.update(tokens=observed["tokens"], evicted_tokens=sum(map(len, result)) * self.page_size)
            else:
                raise ValueError("unknown predicted prefix operation")
        else:
            raise ValueError("unknown predicted resource operation")
        self.apply(operation)


class ResourceEffectComparison:
    def __init__(self):
        self.native, self.predicted = {}, {}
        self.initial = {}
        self.requests = []
        self.counts = {}
        self.reads = {"native": {}, "predicted": {}}

    def observed(self, history, graph, group, raw):
        initial = history["initial"]
        available = initial["available"]
        if (
            sorted(initial["free_pages"]) != list(range(initial["page_capacity"]))
            or initial["released_pages"]
            or available["pages"] != initial["page_capacity"]
            or available["evictable_tokens"]
            or available["protected_tokens"]
            or sorted(initial["free_slots"]) != list(range(available["slots"]))
            or any(sorted(pool["free"]) != list(range(pool["capacity"])) for pool in initial["state_slots"].values())
        ):
            raise ValueError("native observation does not start with empty logical KV and free slot pools")
        self.initial[group] = initial
        source = ResourceEffects(initial["page_size"])
        scopes = {origin: scope for scope in graph["scopes"] for origin in scope["bindings"]["source_scopes"]}
        source.native(history, scopes, graph["environment"]["implementation"]["cache"]["disabled"])
        self.native[group] = source
        self.predicted[group] = ResourceEffects(source.page_size)

        queries = {query["scope"]: query for query in raw["state_queries"] if query["kind"] == "resource"}
        for record in history["reads"]:
            if type(record["control"]) is not bool:
                raise ValueError("native resource getter lacks its control/resource ownership")

            # Resource implementations keep their own internal checks; only controller reads cross this interface.
            if not record["control"]:
                continue
            query = queries.pop(record["scope"], None)
            if (
                query is None
                or query["read"] != record["operation"]
                or query["result"] != record["result"]
                or not query["begin_raw_ns"] <= record["begin_raw_ns"] <= record["end_raw_ns"] <= query["end_raw_ns"]
            ):
                raise ValueError("native resource getter differs from its semantic read boundary, resource or result")
            self.capacity_read("native", group, record["operation"], record["result"])
        if queries:
            raise ValueError("semantic resource query has no original getter observation")

    def capacity_read(self, origin, group, operation, result):
        kind = operation["kind"]
        name = operation["resource"] if kind == "capacity" else operation["field"]
        if (kind == "capacity" and name not in {"slots", "pages", "evictable_tokens", "protected_tokens"}
                or kind == "state_slots" and operation["operation"] != "available"
                or kind not in {"capacity", "state_slots"} or not isinstance(name, str) or not name
                or type(result) is not int or result < 0):
            raise ValueError("capacity observation requires a declared scalar resource and nonnegative integer")

        summary = self.reads[origin].setdefault((group, kind, name), {"count": 0, "minimum": result, "maximum": result})
        summary["count"] += 1
        summary["minimum"] = min(summary["minimum"], result)
        summary["maximum"] = max(summary["maximum"], result)

    def observe(self, row):
        if row["kind"] != "resource":
            return
        operation, group = row["operation"], row["group"]
        kind = operation["kind"]
        if kind == "declare_resources" and group in self.initial:
            if operation["slots"] != len(self.initial[group]["free_slots"]):
                raise ValueError("predicted request pool differs from the observed empty native pool")
        if kind == "capacity":
            self.capacity_read("predicted", group, operation, row["result"])
        elif kind == "state_slots" and operation["operation"] == "available":
            self.capacity_read("predicted", group, operation, row["result"])
        if group in self.predicted:
            self.predicted[group].predicted(row, self.requests)

    @staticmethod
    def summary(operation):
        return {name: {"count": len(value), "first": value[:4]} if isinstance(value, list) else value
                for name, value in operation.items()}

    def compare(self, *, complete):
        matched, state_matched = 0, 0
        totals, state_unobserved = Counter(), []
        for group, source in self.native.items():
            target = self.predicted[group]
            expected, actual = source.operations, target.operations
            totals.update(op["kind"] for op in expected)

            # A partial trace establishes only its independently checked native prefix.
            if not complete:
                expected = expected[:len(actual)]
            if expected == actual:
                matched += 1
            else:
                first = next((i for i, (a, b) in enumerate(zip(expected, actual, strict=False)) if a != b),
                             min(len(expected), len(actual)))
                yield "resource_effects", {"group": group}, {
                    "first_difference": first, "native_operations": len(source.operations),
                    "predicted_operations": len(actual),
                    "native": self.summary(source.operations[first]) if first < len(source.operations) else None,
                    "predicted": self.summary(actual[first]) if first < len(actual) else None,
                }

            recorded, generated = source.fields, target.fields
            if generated.capacities and not recorded.capacities:
                state_unobserved.append(group)
                continue
            if complete:
                same = (recorded.capacities == generated.capacities and recorded.failures == generated.failures
                        and recorded.row_histories() == generated.row_histories())
            else:
                same = generated.prefix_of(recorded)
            if same:
                state_matched += 1
            else:
                yield "state_slot_effects", {"group": group}, {
                    "native_capacity": recorded.capacities, "predicted_capacity": generated.capacities,
                    "native_operations": recorded.operations, "predicted_operations": generated.operations,
                }

        reads = {origin: [{"group": group, "kind": kind, "resource": resource, **value}
                          for (group, kind, resource), value in sorted(records.items())]
                 for origin, records in self.reads.items()}
        self.counts = {
            "groups": len(self.native), "groups_matched": matched, "operations": totals,
            "empty_initial_states_checked": len(self.initial),
            "comparison_scope": "complete_history" if complete else "observed_prefix",
            "reads": reads | {"scope": "Controller scalar getters in each run."},
            "state_slots": {"groups_matched": state_matched, "unobserved_native": state_unobserved,
                            "predicted_operations": sum(item.fields.operations for item in self.predicted.values())},
            "scope": "Recorded per-worker KV effects and row ownership histories, including physical row reuse.",
        }
