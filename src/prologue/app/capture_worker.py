"""Capture a native job, export its SDK records, or interpret one recorded worker."""

import argparse
import json
from pathlib import Path

from prologue.artifacts import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="run one normalized native observation job")
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--local-shards", action="store_true", help="execute logical shards on the single mounted device")
    observation = capture.add_mutually_exclusive_group()
    observation.add_argument("--resource-trace", action="store_true", help="observe resources and input reads for independent comparison")
    observation.add_argument("--service-reference", action="store_true", help="observe structure, resources and same-run CPU activity")
    observation.add_argument("--latency-only", action="store_true", help="record scheduler ingress and token publication only")
    observation.add_argument("--host-timing", action="store_true", help="sample native call cuts and CPU-only queue activity")
    observation.add_argument("--local-copy", action="store_true", help="time isolated native tensor copies on one device")
    observation.add_argument("--attention", action="store_true", help="measure selected native Decode attention calls")
    interpret = commands.add_parser("interpret", help="interpret an SDK-exported worker recording")
    interpret.add_argument("--trace", type=Path, required=True)
    interpret.add_argument("--evidence", action="store_true", help="retain measured boundary clocks for source auditing")
    export = commands.add_parser("export", help="export one worker through an isolated SDK workspace")
    for command in (interpret, export):
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--rank", type=int, required=True)
        command.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "capture" and arguments.local_copy:
        import os

        from prologue.integrations.platforms.ascend.copy_sampling import TensorCopySamples

        TensorCopySamples().sample(json.loads(arguments.input.read_text()), arguments.output, os.environ["PROLOGUE_SGLANG_REVISION"])
    elif arguments.command == "capture":
        from prologue.integrations.frameworks.sglang.observation.session import NativeServingSession

        source = json.loads(arguments.input.read_text())
        attention = source.pop("attention_sites") if arguments.attention else None
        NativeServingSession(
            source, arguments.output, attention_sites=attention,
            resource_trace=arguments.resource_trace, latency_only=arguments.latency_only,
            host_timing=arguments.host_timing, service_reference=arguments.service_reference, local_shards=arguments.local_shards,
        ).run()
        if attention:
            from prologue.integrations.platforms.ascend.scoped_tasks import ScopedTasks

            # The workers have exited. Exporting SDK evidence can no longer disturb the observed serving calls.
            ScopedTasks.publish(arguments.output)
    elif arguments.command == "export":
        from prologue.integrations.platforms.ascend.export import AscendExport

        AscendExport.run(arguments.input, arguments.rank, arguments.output)
    else:
        from prologue.integrations.frameworks.sglang.observation.kv_facts import NativeKvFacts
        from prologue.integrations.frameworks.sglang.observation.source_scopes import SGLangSourceScopes
        from prologue.integrations.frameworks.sglang.pipeline import PipelineABI
        from prologue.integrations.platforms.ascend.observation import AscendObservation

        native = NativeKvFacts(arguments.input, arguments.rank)
        recording = native.document()
        recording["scopes"] = SGLangSourceScopes(recording["environment"]).lower(recording["scopes"])

        # Framework closure rules stay separate from runtime interpretation and the common graph projection.
        observation = AscendObservation(arguments.input, arguments.rank, recording, arguments.trace, evidence=arguments.evidence)
        observation.name_streams(*PipelineABI.stream_declarations(recording))
        facts = observation.document()
        native.connect_metadata(facts)
        if arguments.evidence and (timing := native.protocol_timings()) is not None:
            facts["measurements"]["pd_protocol"] = timing
        write_json(arguments.output, facts, replace=False)


if __name__ == "__main__":
    main()
