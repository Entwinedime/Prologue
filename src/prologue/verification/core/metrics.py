"""Compare request metrics on a shared publication basis."""

from statistics import mean


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
