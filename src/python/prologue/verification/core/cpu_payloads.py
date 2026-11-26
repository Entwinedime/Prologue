"""Compare the native and predicted wire geometry after post identity has been proven."""

from collections import Counter, defaultdict


def compare_wire_payloads(native_messages, predicted_messages, pairs):
    """Use checked post witnesses to compare each run's actual wire descriptor."""

    if len(set(pairs.values())) != len(pairs):
        raise ValueError("multiple predicted CPU messages use the same native post witness")
    routes, differences = defaultdict(Counter), []

    for predicted_key, native_key in pairs.items():
        native = native_messages[native_key]
        predicted = predicted_messages.get(predicted_key, {})
        if len(native) != 2 or len(predicted) != 2:
            continue

        # The witness identifies the message. Each run supplies its own wire descriptor.
        _, source, target, _, _ = native_key
        execution = next(peer["execution"] for peer in predicted.values() if peer["execution"]["in_flow"])
        channel = execution["work"]["context"]["cpu_transfer"]["group"]
        counts = routes[channel, source, target]
        expected, actual = native[source]["wire"], predicted[source]["wire"]
        counts["checked"] += 1
        counts["matched"] += expected == actual

        if expected != actual:
            differences.append(
                {
                    "channel": channel,
                    "source": source,
                    "target": target,
                    "native_post": native_key,
                    "predicted_post": predicted_key,
                    "native": describe_payload(expected),
                    "predicted": describe_payload(actual),
                }
            )

    checked = sum(counts["checked"] for counts in routes.values())
    return {
        "witnessed": len(pairs),
        "checked": checked,
        "matched": checked - len(differences),
        "unresolved": len(pairs) - checked,
        "routes": [
            {"channel": channel, "source": source, "target": target, **counts}
            for (channel, source, target), counts in sorted(routes.items())
        ],
    }, differences

def describe_payload(wire):
    dtype, elements, element_bytes = wire
    return {"dtype": dtype, "elements": elements, "element_bytes": element_bytes, "bytes": elements * element_bytes}
