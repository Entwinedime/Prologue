"""Keep an absent native PD witness visible in structural qualification."""

from . import ActionCheck


def missing_readiness(case, trace):
    for name in (
        "missing_publications",
        "missing_reservation_updates",
        "missing_completion_updates",
        "missing_success_reads",
        "unobserved_transfer_peers",
    ):
        if case.readiness[name]:
            yield "native_evidence", {}, {name: case.readiness[name]}


checks = (ActionCheck("native_pd_readiness", missing_readiness),)
