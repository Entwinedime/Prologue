"""Resolve a native entry in the framework and device domain declared by its image."""

from prologue.integrations.bindings import route


def entry(component):
    return route().framework_module(component)
