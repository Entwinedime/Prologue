"""Common CLI argument contracts used by host commands."""

from argparse import SUPPRESS
from dataclasses import fields
from pathlib import Path


def routing(parser):
    for name, help_text in (("framework", "installed inference framework"), ("platform", "installed device platform"),
                            ("framework-version", "native revision or ABI alias")):
        parser.add_argument("--" + name, default=SUPPRESS, help=help_text)


def command(group, name, description, run=None, *, configure=None):
    parser = group.add_parser(name, help=description, description=description, add_help=configure is None,
                              allow_abbrev=False)
    routing(parser)
    parser.set_defaults(parser=parser, run=run)
    if configure is not None:
        parser.set_defaults(configure=configure)
    return parser


def paths(parser, *names):
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)


def modeling(parser, *, required=True):
    parser.add_argument("--binary", type=Path, required=required, help="Docker-built C++ simulation engine")
    parser.add_argument("--modeling-image", default="prologue-modeling:ubuntu24.04")


def control(parser):
    parser.add_argument("--control-image", help="native control image for the selected adapter")


def timeout(parser):
    parser.add_argument("--timeout", type=float, required=True, help="time allowance in seconds")


def native(parser):
    paths(parser, "deployment", "output")
    timeout(parser)
    parser.add_argument("--budget-seconds", type=float, required=True, help="cumulative native allowance")
    parser.add_argument("--attempt-limit", type=int, required=True, help="cumulative number of native attempts")
    source_image(parser)


def source_image(parser):
    parser.add_argument("--source-image", help="capture image for the selected adapter")


def options(contract, arguments):
    return contract(**{f.name: getattr(arguments, f.name) for f in fields(contract) if hasattr(arguments, f.name)})
