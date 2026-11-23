# Prologue

Prologue models serving for one model across multiple configurations and request workloads. The current commands plan tasks, observe native SGLang serving on one Ascend card, build hierarchical execution templates, and verify their expansion under native scheduling.

## Build and plan a task

Build the software-only toolchain and then install the C++23 core and Python CLI into the application image.

```bash
docker build -f docker/images/modeling/Dockerfile -t prologue-modeling:ubuntu24.04 .
docker build -f docker/images/application/Dockerfile -t prologue-app:ubuntu24.04 .
docker run --rm --runtime runc --network none \
  --mount type=bind,src="$PWD/example/task.json",dst=/task.json,readonly \
  prologue-app:ubuntu24.04 plan /task.json
```

The [example task](example/task.json) declares one model, two configurations and two workloads. The command prints four targets and the ordered request inputs. Planning can run on a CPU host with just this input file.

## Observe and compile a TP1 target

Use the pinned SGLang checkout and a compatible Ascend device. Build the software dependency image first if it is absent. Set `PROLOGUE_DEVICE` to the physical device allocated to the run.

```bash
git submodule update --init third_party/sglang
test -z "$(git -C third_party/sglang status --porcelain)"
docker build -t prologue-sglang-env:ubuntu22.04 docker/images/base/sglang
docker build -f docker/images/framework/sglang/Dockerfile -t prologue-sglang-runtime:ubuntu22.04 \
  --build-arg "SGLANG_REVISION=$(git -C third_party/sglang rev-parse HEAD)" .
docker build -f docker/images/observer/Dockerfile -t prologue-observer:ubuntu22.04 .

run="$PWD/data/$(date -u +%Y%m%dT%H%M%SZ)_observation"
mkdir -p "$run"
docker run --rm --runtime runc --network none \
  --mount type=bind,src="$PWD/example/task.json",dst=/task.json,readonly \
  prologue-app:ubuntu24.04 plan /task.json > "$run/plan.json"
docker run --rm --init --runtime ascend --ipc host --network bridge \
  --env "ASCEND_VISIBLE_DEVICES=${PROLOGUE_DEVICE:?Select an available Ascend device}" \
  --env OMP_NUM_THREADS=1 --env SGLANG_PLUGINS=prologue \
  --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
  --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \
  --mount type=bind,src=/etc/hccn.conf,dst=/etc/hccn.conf,readonly \
  --mount type=bind,src=/models/Qwen3-8B,dst=/models/Qwen3-8B,readonly \
  --mount type=bind,src="$run",dst=/out \
  prologue-observer:ubuntu22.04 /out/plan.json /out/detailed \
  --configuration tp1-page16 --workload short --mode detailed
docker run --rm --runtime runc --network none \
  --mount type=bind,src="$run/detailed",dst=/source,readonly \
  prologue-app:ubuntu24.04 compile /source > "$run/templates.json"
```

Use `--mode light` and a separate output directory for scheduler publication times without a profiler. Each observation records the inputs and the native revision. The template library stores definitions and concrete call bindings. Device records with unresolved ownership are counted separately.

## Verify execution structure on CPU

Capture a separate detailed target under the source configuration and save it as `reference` in the run directory. The verifier uses that observation to answer timing queries and compare the resulting graph. The native controller reads the workload and simulated resources to make its own decisions. Both processes run without an accelerator.

```bash
docker build -f docker/images/control/Dockerfile -t prologue-scheduler:ubuntu22.04 .
cp /models/Qwen3-8B/config.json "$run/model-config.json"
mkdir "$run/control"

# Start the verifier and wait for its socket before starting the controller.
docker run --rm --runtime runc --network none \
  --mount type=bind,src="$run",dst=/case,readonly \
  --mount type=bind,src="$run/control",dst=/control \
  prologue-app:ubuntu24.04 verify-structure /case/reference \
  --library /case/templates.json --model-config /case/model-config.json \
  --socket /control/control.sock
```

In a second terminal, with `run` set to the same directory, start the native control worker.

```bash
docker run --rm --runtime runc --network none \
  --mount type=bind,src="$run/control",dst=/control \
  prologue-scheduler:ubuntu22.04 /control/control.sock
```

This verification path currently covers synchronous TP1 actions whose batch shapes have source templates. Repeat `--library` to load additional observations collected in the same domain. The report includes the first missing condition or disagreement and the matched action count.

## License

Prologue is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies retain their own licenses.
