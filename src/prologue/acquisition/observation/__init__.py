"""Framework independent observation lifecycle."""


def jointly_observed(instance):
    """Whether peer clocks describe one execution on the physical topology."""
    return instance.get("placement") != "single_device"
