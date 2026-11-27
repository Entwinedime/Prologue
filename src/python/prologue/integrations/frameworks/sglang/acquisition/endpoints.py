"""Locate active bounds without changing the engine's cost coverage rule."""

from math import prod

from prologue.integrations.frameworks.sglang.shared.configuration import NativeConfiguration


def outside_edges(query):
    if query.get("status") != "outside_sample_ranges":
        return {}

    axes = query["features"]["axes"]
    return {
        axis: int(axes[axis] > upper) - int(axes[axis] < lower)
        for axis, (lower, upper) in query["sample_bounds"].items() if axis in axes and not lower <= axes[axis] <= upper
    }


def copy_sizes(query, width, size):
    edge = outside_edges(query).get("transfer.bytes")
    if not edge:
        return (size,)

    invocation = query.get("invocation", {})
    configuration = invocation.get("native_inputs", {}).get("configuration")
    if not configuration:
        return (size,)

    copy = query["definition"]["work"][query["interval"]]["copy"]
    shape = copy["tensor"]["shape"]
    symbols = [axis for axis in shape if isinstance(axis, str)]
    if len(symbols) != 1:
        return (size,)

    capacity = configuration["max_total_tokens"]
    limits = {"members": NativeConfiguration.batch_capacity(configuration),
              "release_tokens": capacity, "released_pages": capacity // configuration["page_size"]}
    axis = symbols[0]
    stride = width * prod(n for n in shape if type(n) is int)
    if axis not in limits or stride * invocation["parameters"].get(axis, 0) != size:
        return (size,)

    extreme = stride * (1 if edge < 0 else limits[axis])
    return (size, extreme) if (extreme - size) * edge > 0 else (size,)
