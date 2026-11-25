"""Convert consecutive rows of the Azure 2024 inference trace to a Prologue workload."""

import argparse
import csv
import re
from datetime import datetime, timedelta
from itertools import islice
from pathlib import Path

from prologue.integrations.frameworks.sglang.shared.configuration import ServingInput
from prologue.runtime.io import write_json


def trace_time(value: str) -> tuple[datetime, int]:
    seconds, dot, tail = value.partition(".")
    if not dot:
        return datetime.fromisoformat(seconds), 0

    # Keep Azure's seventh fractional digit without rounding through datetime.
    fraction = re.match(r"[0-9]{1,9}(?![0-9])", tail)
    if fraction is None:
        raise ValueError("trace timestamp requires one to nine fractional digits")
    instant = datetime.fromisoformat(seconds + tail[fraction.end():])
    return instant, int(fraction[0].ljust(9, "0"))


def convert(path: Path, first_row: int, count: int) -> dict:
    requests = []
    origin = None
    previous_ns = 0
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]:
            raise ValueError("expected the Azure 2024 inference trace columns")

        for index, row in enumerate(islice(rows, first_row - 1, first_row - 1 + count), first_row):
            arrival, fraction_ns = trace_time(row["TIMESTAMP"])
            if origin is None:
                origin = arrival, fraction_ns
            start, start_fraction_ns = origin
            if (arrival.utcoffset() is None) != (start.utcoffset() is None):
                raise ValueError(f"Azure row {index} changes timestamp timezone convention")

            # A timezone-free trace shares one wall-clock basis. Replay uses elapsed nanoseconds.
            elapsed_ns = ((arrival - start) // timedelta(microseconds=1)) * 1000 + fraction_ns - start_fraction_ns
            if elapsed_ns < previous_ns:
                raise ValueError(f"Azure row {index} has an unordered timestamp")
            requests.append({
                "id": f"azure-{index}",
                "arrival_ns": elapsed_ns,
                "prompt_tokens": int(row["ContextTokens"]),
                "output_tokens": int(row["GeneratedTokens"]),
            })
            previous_ns = elapsed_ns

    if len(requests) != count:
        raise ValueError(f"the trace has fewer than {count} rows starting at {first_row}")

    return ServingInput.workload_input({"id": f"azure-{first_row}-{count}", "requests": requests})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first-row", type=int, required=True, help="one-based data row, excluding the header")
    parser.add_argument("--requests", type=int, required=True)
    args = parser.parse_args()
    if args.first_row <= 0 or args.requests <= 0:
        parser.error("the first data row and request count must be positive")

    write_json(args.output, convert(args.trace, args.first_row, args.requests), replace=False)


if __name__ == "__main__":
    main()
