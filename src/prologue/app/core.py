"""Exchange a complete command input with the installed C++ core."""

import json
import subprocess


def run_core(command, document, *arguments):
    result = subprocess.run(
        ["prologue-core", command, *arguments], input=json.dumps(document), encoding="utf-8", capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"{command} exited with {result.returncode}")
    return json.loads(result.stdout)


def run_simulation(document, socket, timing_source):
    command = ["prologue-core", "simulate", str(socket)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8") as process:
        print(json.dumps(document), file=process.stdin, flush=True)
        for line in process.stdout:
            message = json.loads(line)
            if "result" in message:
                return message["result"]
            answer = timing_source.conditions(message["timing_query"])
            print(json.dumps(answer), file=process.stdin, flush=True)

        raise ValueError(f"simulation exited with {process.wait()} before publishing a result")
