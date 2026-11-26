"""Reject a trace containing more than one independently framed execution."""

from . import ActionCheck


def repeated_terminator(case, trace):
    if trace.terminal_records > 1:
        yield "duplicate_end_record", {}, {"records": trace.terminal_records}


checks = (ActionCheck("single_execution_terminator", repeated_terminator),)
