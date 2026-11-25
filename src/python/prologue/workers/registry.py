"""Resolve a native entry in the framework and device domain declared by its image."""

from prologue.integrations.bindings import route


def entry(component, job=None):
    selected = route()
    if job is not None:
        selected.require_job(job)
    return selected.framework_module(component)
