"""Extract measured services, fit models, and acquire missing coverage."""

from pathlib import Path

from prologue.acquisition.costs import CostAcquisition
from prologue.acquisition.fitting import FitCosts, TransportSamples
from prologue.acquisition.options import CostAcquisitionOptions, DeviceSampleOptions, FitCostsOptions, HostSampleOptions
from prologue.app.arguments import command, modeling, options, paths, source_image, timeout
from prologue.app.registry import AdapterRegistry
from prologue.runtime.docker import ModelingProcess
from prologue.runtime.io import write_json


def register(group):
    extract = command(group, "extract", "extract measured services for the selected source kind", configure=extraction)
    extract.add_argument("--kind", choices=("device", "transport", "batch", "host"), help="measurement source")
    fit = command(group, "fit", "fit registered models and optionally check their coverage", fit_costs)
    acquire = command(
        group, "acquire", "collect and fit missing services within the supplied allowances", acquire_costs
    )
    for parser in (fit, acquire):
        paths(parser, "output")
        parser.add_argument("--gap", type=Path, action="append", default=[])
        parser.add_argument("--demand", type=Path, nargs=2, action="append", default=[], metavar=("GRAPH", "WORK"))
    modeling(fit, required=False)
    fit.add_argument("--samples", type=Path, nargs="+", required=True)
    fit.add_argument("--check", type=Path, action="append", default=[])
    fit.add_argument("--timeout", type=float, default=300, help="cumulative modeling allowance")
    modeling(acquire)
    timeout(acquire)
    source_image(acquire)
    acquire.add_argument("--production", type=Path, action="append", required=True)
    acquire.add_argument("--samples", type=Path, nargs="+", default=[])
    acquire.add_argument("--transport-samples", type=Path, nargs="+", default=[])
    acquire.add_argument("--keep-intermediates", action="store_true")
    acquire.add_argument("--native-budget-seconds", type=float, required=True)
    acquire.add_argument("--postprocessing-budget-seconds", type=float, required=True)
    acquire.add_argument("--attempt-limit", type=int, required=True)
    acquire.add_argument("--storage-bytes", type=int, required=True)


def extraction(parser, args):
    if args.kind in ("device", "transport"):
        paths(parser, "production")
        parser.add_argument("--projection", type=Path)
        for name, value in (("rank", int), ("work-identities", Path), ("output", Path)):
            parser.add_argument("--" + name, type=value, action="append", required=True)
        parser.set_defaults(transport_only=args.kind == "transport")
        return extract_device
    elif args.kind == "batch":
        paths(parser, "session", "output")
        modeling(parser)
        timeout(parser)
        return extract_batch
    elif args.kind == "host":
        paths(parser, "detailed", "host", "facts", "trace", "graph", "work-identities", "output")
        return extract_host
    else:
        return missing_kind


def missing_kind(args):
    args.parser.error("--kind is required; select a kind before requesting its operand help")


def extract_device(args):
    AdapterRegistry.select().samples.export(options(DeviceSampleOptions, args))


def extract_batch(args):
    model = ModelingProcess(args.binary, args.modeling_image, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = AdapterRegistry.select().samples.batch(args.session, model, args.output.parent)
    write_json(args.output, samples, replace=False)


def extract_host(args):
    AdapterRegistry.select().host_samples(options(HostSampleOptions, args)).run()


def fit_costs(args):
    FitCosts(options(FitCostsOptions, args), AdapterRegistry.select().reuse.match).run()


def acquire_costs(args):
    adapters = AdapterRegistry.select()
    profiles = TransportSamples(args.transport_samples)
    CostAcquisition(options(CostAcquisitionOptions, args), adapters.native, adapters.reuse.match,
                    transport_samples=profiles, sample_adapter=adapters.native).run()
