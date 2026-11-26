"""Reconstruct expected scalar collective results from each run's own operands."""

from itertools import chain

from prologue.verification.core.pd_polling import PdPollingEvidence


def broadcast(collective, operands):
    return operands[collective["root"]]


def gather(collective, operands):
    return list(chain.from_iterable(operands))


def reduce_minimum(collective, operands):
    return [min(values) for values in zip(*operands, strict=True)]


operations = {
    ("broadcast", None): broadcast,
    ("all_gather_into_tensor", None): gather,
    ("all_reduce", "RedOpType.MIN"): reduce_minimum,
}


def compare_metadata(endpoints, origin, counts, finding):
    if not any(endpoint["metadata"] for endpoint in endpoints):
        return
    if not all(endpoint["metadata"] for endpoint in endpoints):
        finding("metadata_protocol", endpoints[0]["work"],
            reason="participants disagree on metadata ownership")
        return

    # A position-wise polling MIN must refer to the same ordered requests on every peer.
    queues = [PdPollingEvidence.request_order(endpoint["work"]["work"])
              if PdPollingEvidence.in_poll(endpoint["work"]["work"]) else None for endpoint in endpoints]
    if any(queue != queues[0] for queue in queues):
        finding("metadata_protocol", endpoints[0]["work"], origin=origin,
                     reason="PD polling operands refer to different peer request queues")
        return

    field = f"{origin}_metadata_exchanges"
    counts[field] = counts.get(field, 0) + 1
    if any(endpoint["values"] is None for endpoint in endpoints):
        field = f"{origin}_metadata_unobserved"
        counts[field] = counts.get(field, 0) + 1
        return

    collective = endpoints[0]["work"]["work"]["context"]["collective"]
    root = collective["root"]
    if root is not None and (type(root) is not int or not 0 <= root < len(endpoints)):
        raise ValueError("metadata broadcast root is outside its native group")

    for position, endpoint in enumerate(endpoints):
        values = endpoint["values"]
        size = endpoint["elements"] if root is None or position == root else 0
        if (endpoint["work"]["work"]["context"]["collective"] != collective
                or not isinstance(values["input"], list) or not isinstance(values["output"], list)
                or endpoint["elements"] != endpoints[0]["elements"] or len(values["input"]) != size
                or any(type(value) not in (bool, int, float) for row in values.values() for value in row)):
            raise ValueError("metadata values disagree with their declared scalar collective")

    operands = [endpoint["values"]["input"] for endpoint in endpoints]
    operation = collective["operation"]
    key = operation, collective["reduction"] if operation == "all_reduce" else None
    if key not in operations:
        raise ValueError("metadata evidence uses an unsupported scalar collective")

    expected = operations[key](collective, operands)

    # Compute from each run's actual operands; no core/broker reduction or cross-run polling index is reused.
    for endpoint in endpoints:
        if endpoint["values"]["output"] != expected:
            finding("metadata_values", endpoint["work"], expected=expected,
                actual=endpoint["values"]["output"], origin=origin)
            return

    field = f"{origin}_metadata_matched"
    counts[field] = counts.get(field, 0) + 1
