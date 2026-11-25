"""Collect native observations and project retained captures."""

from pathlib import Path

from prologue.acquisition.options import CaptureOptions, ProductionOptions
from prologue.acquisition.production import CaptureProduction
from prologue.acquisition.recording import PublishedCapture
from prologue.app.arguments import command, modeling, native, options, paths, source_image
from prologue.app.registry import AdapterRegistry
from prologue.runtime.io import read_json


def register(group):
    capture = command(group, "run", "run or resume a bounded native capture", run)
    paths(capture, "input")
    native(capture)
    observation = capture.add_mutually_exclusive_group()
    for flag, description in (("resource-trace", "record resource and input-read evidence"),
                              ("service-reference", "record structure and same-run CPU activity"),
                              ("latency-only", "record scheduler latency without profiling"),
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
    launch = AdapterRegistry.select().native(read_json(args.input), read_json(args.deployment))
    sites = read_json(args.attention_sites) if args.attention_sites else None
    launch.capture(options(CaptureOptions, args), attention_sites=sites, latency_only=args.latency_only,
                   host_timing=args.host_timing, service_reference=args.service_reference).run()


def project_capture(args):
    CaptureProduction(options(ProductionOptions, args), PublishedCapture(args.capture),
                      AdapterRegistry.select().native).run()
