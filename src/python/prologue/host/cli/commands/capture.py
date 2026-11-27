"""Collect native observations and project retained captures."""

from pathlib import Path

from prologue.common.io import read_json
from prologue.host.acquisition.options import CaptureOptions, ProductionOptions
from prologue.host.acquisition.production import CaptureProduction
from prologue.host.acquisition.recording import PublishedCapture
from prologue.host.cli.arguments import command, modeling, native, options, paths, source_image


def register(group):
    capture = command(group, "run", "run or resume a bounded native capture", run)
    paths(capture, "input")
    native(capture)
    observation = capture.add_mutually_exclusive_group()

    for flag, description in (("resource-trace", "record resource and input-read evidence"),
                              ("service-reference", "record structure and same-run CPU activity"),
                              ("latency-only", "record scheduler latency without profiling"),
                              ("batch-timing", "record native scheduler call boundaries for timing diagnosis"),
                              ("host-timing", "measure CPU work and queue cuts")):
        observation.add_argument("--" + flag, action="store_true", help=description)

    observation.add_argument("--attention-sites", type=Path, help="measure named decode modules")
    project = command(group, "project", "project a completed capture into graphs or templates", project_capture)
    paths(project, "capture", "output")
    modeling(project)
    source_image(project)
    project.add_argument("--product", choices=("graphs", "templates"), default="templates")
    project.add_argument("--keep-intermediates", action="store_true")
    project.add_argument("--budget-seconds", type=float, required=True, help="cumulative postprocessing allowance")


def run(args):
    launch = args.adapters.native(args.source, read_json(args.deployment))
    sites = read_json(args.attention_sites) if args.attention_sites else None
    launch.capture(options(CaptureOptions, args), attention_sites=sites, latency_only=args.latency_only,
                   batch_timing=args.batch_timing, host_timing=args.host_timing,
                   service_reference=args.service_reference).run()


def project_capture(args):
    CaptureProduction(options(ProductionOptions, args), PublishedCapture(args.capture),
                      args.adapters.native).run()
