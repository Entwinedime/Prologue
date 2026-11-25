"""Predict serving behavior, maintain evidence libraries, and verify results."""

import argparse

from prologue.host.cli.arguments import routing
from prologue.host.cli.commands import capture, cost, library, predict, verify
from prologue.host.registry import AdapterRegistry


def main(argv=None):
    parser = argparse.ArgumentParser(prog="prologue", description=__doc__, allow_abbrev=False)
    routing(parser)
    groups = parser.add_subparsers(required=True, metavar="DOMAIN")
    for name, module in (("predict", predict), ("library", library), ("capture", capture),
                         ("cost", cost), ("verify", verify)):
        group = groups.add_parser(name, help=module.__doc__, description=module.__doc__)
        routing(group)
        module.register(group.add_subparsers(required=True, metavar="COMMAND"))

    args, remaining = parser.parse_known_args(argv)
    handler = args.run
    if hasattr(args, "configure"):

        # Parse only the selected mode's operands. A flag for a different mode is an error.
        handler = args.configure(args.parser, args)
        args.parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
        args = args.parser.parse_args(remaining, namespace=args)
    elif remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    with AdapterRegistry.command(args) as (selection, args.adapters):
        for field, image in args.adapters.images(selection).items():
            if hasattr(args, field) and getattr(args, field) is None:
                setattr(args, field, image)
        return handler(args)
