# Prologue

Prologue models serving for one model across multiple configurations and request workloads. The current commands plan tasks, observe native SGLang serving on one Ascend card, build hierarchical execution templates, and predict or verify covered TP1 targets.

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

Use `--mode light` and a separate output directory for scheduler publication times without a profiler. Each observation records inputs, concrete call bindings and the native revision. Compiled sources carry reusable definitions and their exact root queries. Device records with unresolved ownership are counted separately.

## Fit and publish shared assets

Capture a separate training observation as `host-training` with `--mode host-timing`. Use detailed training observations for device tasks, and reserve separate observations for evaluation.

```bash
store="$PWD/data/libraries"
mkdir -p "$store"
cp /models/Qwen3-8B/config.json "$run/model-config.json"

docker run --rm --runtime runc --network none \
  --mount type=bind,src="$run",dst=/case,readonly \
  prologue-app:ubuntu24.04 fit-costs --device /case/detailed --host /case/host-training > "$run/costs.json"
docker run --rm --runtime runc --network none \
  --mount type=bind,src="$run",dst=/case,readonly \
  --mount type=bind,src="$store",dst=/library \
  prologue-app:ubuntu24.04 publish --store /library --library /case/templates.json \
  --costs /case/costs.json --model-config /case/model-config.json
```

The release records the template, performance and rule objects together. `show-release` displays their identifiers. For a template update, repeat `--library` with the full set of desired sources. Omitted asset families use the base release. Pass `--base <release UUID>` when an update is prepared against a specific release.

`query-structure` uses the same selection and expansion as prediction. Pass a JSON document with the published `domain` and `targets`, an array of `{ "target": "case-id", "query": { "phase": "decode", "sequence_lengths": [9] } }` entries. `--candidate` previews an additional source and `--graph` includes the expanded graph. The report distinguishes a missing parent, a missing child and conflicting candidates. Use `publish --add-library ... --require-structure queries.json` to keep the base sources and require the original queries to resolve before changing the current release.

### Compile a native Decode page-table cut

For a native SGLang Prefill/Decode pair, enable the Prologue plugin on Decode and set `PROLOGUE_PD_CUT` to a JSON file in a fresh output directory. Set `configuration_id`, `detail_requests` and `parent_requests`. Use native request IDs for both lists. Run the profiler around `detail_requests` with output under `profile/` in the same directory. Submit `parent_requests` with the profiler stopped. The observer records the effective domain, KV handover and admitted page-table slices.

```bash
prologue compile copy-observation --pd-scope copy > page-copy.json
prologue compile assembly-observation --pd-scope assembly --for-queries queries.json > page-assembly.json
prologue publish --store data/pd-library --add-library page-assembly.json --require-structure queries.json
```

Publish the detailed copy source first in a store with the matching PD Decode domain. The assembly source refers to its copy definitions by exact query and requires the same ports on resolution. This path covers the Decode page-table assembly boundary. Complete PD execution and communication timing are separate from this structural check.

## Predict from a release on CPU

Declare the target deployment in `hardware.json` using the fields in `domain.hardware` from the cost asset. Use the training descriptor for the same deployment. The [reuse task](example/reuse-task.json) changes request identities and arrivals while using the covered 8-token geometry.

```bash
python3 -c 'import json, sys; print(json.dumps(json.load(sys.stdin)["domain"]["hardware"]))' \
  < "$run/costs.json" > "$run/hardware.json"
docker build -f docker/images/control/Dockerfile -t prologue-scheduler:ubuntu22.04 .
mkdir "$run/control"

# Wait for the socket before starting the control process.
docker run --rm --runtime runc --network none \
  --mount type=bind,src="$store",dst=/library,readonly \
  --mount type=bind,src="$PWD/example/reuse-task.json",dst=/task.json,readonly \
  --mount type=bind,src="$run/hardware.json",dst=/hardware.json,readonly \
  --mount type=bind,src="$run/control",dst=/control \
  prologue-app:ubuntu24.04 predict /task.json --configuration tp1-page16 --workload reuse \
  --store /library --release current --hardware /hardware.json \
  --socket /control/control.sock > "$run/prediction.json"
```

Start the native control worker in a second terminal with `run` set to the same directory.

```bash
docker run --rm --runtime runc --network none \
  --mount type=bind,src="$run/control",dst=/control \
  prologue-scheduler:ubuntu22.04 /control/control.sock
```

The prediction reports the resolved release UUID. Pass that UUID to `--release` to replay the same asset selection later. The current template and cost rules cover observed synchronous TP1 geometries.

## Verify from a release

Use a separate detailed recording as `reference`. The structure verifier answers timing queries while the controller chooses actions from the workload and simulated resources. Run the verifier with the same two-process socket procedure. The application container reads the shared store and the detailed reference.

```bash
prologue verify-structure reference --store /library --release current --socket /control/control.sock
prologue verify-performance reference --store /library --release current
prologue score prediction.json light-reference
```

Fixed-graph verification checks recorded endpoints before replacing device service durations. Autonomous scoring uses a separate lightweight recording of the prediction workload for TTFT, TPOT and throughput. `quote-costs` accepts a JSON document with `domain` and `coordinates` and uses the same released estimator.

## License

Prologue is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies retain their own licenses.
