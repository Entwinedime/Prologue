"""Capture a native job, export its SDK records, or interpret one recorded worker."""

import argparse
import json
from pathlib import Path

from prologue.runtime.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="run one normalized native observation job")
    capture.add_argument("--input", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument(
        "--local-shards", action="store_true", help="execute logical shards on the single mounted device"
    )
    observation = capture.add_mutually_exclusive_group()
    observation.add_argument(
        "--resource-trace", action="store_true", help="observe resources and input reads for independent comparison"
    )
    observation.add_argument(
        "--service-reference", action="store_true", help="observe structure, resources and same-run CPU activity"
    )
    observation.add_argument(
        "--latency-only", action="store_true", help="record scheduler ingress and token publication only"
    )
    observation.add_argument(
        "--host-timing", action="store_true", help="sample native call cuts and CPU-only queue activity"
    )
    observation.add_argument(
        "--local-copy", action="store_true", help="time isolated native tensor copies on one device"
    )
    observation.add_argument(
        "--local-attention", action="store_true", help="measure Decode on a native ModelRunner and KV pool"
    )
    observation.add_argument(
        "--local-prebuilt", action="store_true", help="observe the receiver assembling a native PD batch"
    )
    observation.add_argument(
        "--local-embedding", action="store_true", help="observe a vocabulary lookup across native TP shards"
    )
    observation.add_argument(
        "--local-worker", action="store_true", help="read the initialized scheduler, model and pools"
    )
    observation.add_argument(
        "--local-host", action="store_true", help="measure local scheduler work with the independent CPU probe"
    )
    observation.add_argument(
        "--local-host-pair", action="store_true", help="capture and independently time one local worker cut"
    )
    observation.add_argument("--attention", action="store_true", help="measure selected native Decode attention calls")
    interpret = commands.add_parser("interpret", help="interpret an SDK-exported worker recording")
    interpret.add_argument("--trace", type=Path, required=True)
    interpret.add_argument(
        "--evidence", action="store_true", help="retain measured boundary clocks for source auditing"
    )
    export = commands.add_parser("export", help="export one worker through an isolated SDK workspace")
    cuts = commands.add_parser("sampling-cuts", help="bind source device work to portable native sampling cuts")
    for option in ("work", "identities", "runner"):
        cuts.add_argument("--" + option, type=Path, required=True)
    for command in (interpret, export, cuts):
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--rank", type=int, required=True)
        command.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "capture" and arguments.local_host_pair:
        import os

        from prologue.integrations.frameworks.sglang.observation.local.scheduler_runner import WorkerProbes

        WorkerProbes(os.environ["PROLOGUE_SGLANG_REVISION"]).pair(arguments.input, arguments.output)
    elif arguments.command == "capture" and arguments.local_copy:
        import os

        from prologue.integrations.platforms.ascend.runtime.copy_sampling import TensorCopySamples

        TensorCopySamples().sample(
            json.loads(arguments.input.read_text()), arguments.output, os.environ["PROLOGUE_SGLANG_REVISION"]
        )
    elif arguments.command == "capture" and arguments.local_attention:
        import os

        from prologue.integrations.frameworks.sglang.observation.local.attention_runner import AttentionRunner

        AttentionRunner(json.loads(arguments.input.read_bytes()), arguments.output).run(
            os.environ["PROLOGUE_SGLANG_REVISION"]
        )
    elif arguments.command == "capture" and (
        arguments.local_prebuilt or arguments.local_embedding or arguments.local_worker or arguments.local_host
    ):
        import os

        job = json.loads(arguments.input.read_bytes())
        if arguments.local_worker and "calibration" in job:
            from prologue.integrations.frameworks.sglang.observation.local.scheduler_runner import WorkerProbes

            WorkerProbes(os.environ["PROLOGUE_SGLANG_REVISION"]).calibration(job, arguments.output)
            return

        # Queue allocation can precede the measured calls, including during native module initialization.
        runtime = arguments.output / "runtime"
        runtime.mkdir(parents=True)
        os.environ["PROLOGUE_NATIVE_RUNTIME_DIR"] = str(runtime)
        if arguments.local_embedding:
            from prologue.integrations.frameworks.sglang.observation.local.embedding_runner import (
                EmbeddingRunner as Runner,
            )
        elif arguments.local_worker or arguments.local_host:
            from prologue.integrations.frameworks.sglang.observation.local.scheduler_runner import (
                SchedulerRunner as Runner,
            )
        else:
            from prologue.integrations.frameworks.sglang.observation.local.prebuilt_runner import (
                PrebuiltRunner as Runner,
            )

        runner = Runner(job, arguments.output)
        options = {"host_timing": True} if arguments.local_host else {}
        runner.run(os.environ["PROLOGUE_SGLANG_REVISION"], **options)
    elif arguments.command == "capture":
        from prologue.integrations.frameworks.sglang.observation.session import NativeServingSession

        source = json.loads(arguments.input.read_text())
        attention = source.pop("attention_sites") if arguments.attention else None
        NativeServingSession(
            source,
            arguments.output,
            attention_sites=attention,
            resource_trace=arguments.resource_trace,
            latency_only=arguments.latency_only,
            host_timing=arguments.host_timing,
            service_reference=arguments.service_reference,
            local_shards=arguments.local_shards,
        ).run()
        if attention:
            from prologue.integrations.platforms.ascend.analysis.scoped_tasks import ScopedTasks

            # The workers have exited. Exporting SDK evidence can no longer disturb the observed serving calls.
            ScopedTasks.publish(arguments.output)
    elif arguments.command == "sampling-cuts":
        from prologue.integrations.frameworks.sglang.observation.local.attention_cuts import publish_cuts

        publish_cuts(arguments.input, arguments.work, arguments.rank, arguments.identities, arguments.runner,
                     arguments.output)
    elif arguments.command == "export":
        from prologue.integrations.platforms.ascend.runtime.export import AscendExport

        AscendExport.run(arguments.input, arguments.rank, arguments.output)
    else:
        from prologue.integrations.frameworks.sglang.observation.source_scopes import serving_facts

        facts = serving_facts(arguments.input, arguments.rank, arguments.trace, evidence=arguments.evidence)
        write_json(arguments.output, facts, replace=False)


if __name__ == "__main__":
    main()
