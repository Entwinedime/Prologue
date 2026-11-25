"""Inspect, acquire and publish template and performance evidence."""

import json
from pathlib import Path

from prologue.common.io import read_json
from prologue.host.acquisition.options import LocalAcquisitionOptions, QualificationOptions
from prologue.host.acquisition.recording import PublishedCapture, SourceProductions
from prologue.host.cli.arguments import command, modeling, native, options, paths, timeout
from prologue.host.libraries.increment import LocalLibraryIncrement
from prologue.host.libraries.store import LibraryStore
from prologue.host.runtime.docker import ModelingProcess


def register(group):
    inspect = command(group, "inspect", "print a read-only summary of a release or store", inspect_library)
    paths(inspect, "libraries")
    acquire = command(group, "acquire", "collect compatible queries in one bounded native session", acquire_local)
    native(acquire)
    modeling(acquire, required=False)
    acquire.add_argument("--libraries", type=Path, help="fixed release supplying native cuts")
    acquire.add_argument("--structure-session", type=Path, action="append", default=[])
    acquire.add_argument("--storage-bytes", type=int)
    query_inputs(acquire)
    publish = command(group, "publish", "publish from a qualified session or an archived collection",
                      configure=publication)
    source = publish.add_mutually_exclusive_group()
    source.add_argument("--from-session", type=Path, help="publish a qualified increment to a store")
    source.add_argument("--from-collection", type=Path, help="import compiled sources into a new fixed release")


def query_inputs(parser):
    parser.add_argument("--query", type=Path, nargs="+", action="extend", required=True,
                        help="original structure or cost queries")


def publication(parser, args):
    paths(parser, "libraries")
    if args.from_session:
        query_inputs(parser)
        timeout(parser)
        modeling(parser)
        return publish_session
    elif args.from_collection:
        parser.add_argument("--costs", type=Path, help="optional fitted costs for the release")
        return publish_archived
    else:
        return missing_source


def missing_source(args):
    args.parser.error("choose --from-session or --from-collection")


def inspect_library(args):
    print(json.dumps(args.release.summarize(), indent=2))


def acquire_local(args):
    release = args.release
    configuration = options(LocalAcquisitionOptions, args)
    acquisition = args.adapters.native.local_acquisition(release, configuration)
    result = acquisition.run(configuration, [read_json(q) for q in args.query], read_json(args.deployment))
    return 0 if result["status"] == "qualified" else 1


def publish_session(args):
    adapters = args.adapters
    model = ModelingProcess(args.binary, args.modeling_image, args.timeout)
    acquisition = adapters.native.local_acquisition(None, options(QualificationOptions, args))
    increment = LocalLibraryIncrement(
        PublishedCapture(args.from_session / "native"),
        [read_json(q) for q in args.query],
        acquisition,
        model,
        adapters.samples,
    )
    published = LibraryStore(args.libraries).publish(increment)
    print(json.dumps({"release": published.identity, "directory": str(published.directory)}))


def publish_archived(args):
    published = SourceProductions(args.from_collection).publish(args.libraries, args.costs)
    print(json.dumps({"release": published.identity, "directory": str(published.directory)}))
