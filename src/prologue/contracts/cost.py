"""Wire coordinates shared by sampling, training, and simulation queries."""

import json


def cost_identity(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def cost_features(axes, *, phase="", transport=None):
    result = {"phase": phase, "axes": axes}
    if transport is not None:
        result["transport"] = transport
    return result
