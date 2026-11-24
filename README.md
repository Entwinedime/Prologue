# Prologue

Prologue models serving for one model across multiple configurations and request workloads. The current commands plan tasks, observe native SGLang serving, build hierarchical execution templates, and predict or verify targets covered by their released assets. Tensor-parallel templates can be built using one Ascend card.

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

Use `--mode light` and a separate output directory for native submission and token publication times without a profiler. The observer calls SGLang's native GC freeze after warmup. Garbage collection remains active for objects allocated during serving. For parallel or disaggregated targets, use `--placement full` with the devices required by the topology. PD reference time starts at Prefill submission and ends at Decode publication. Detailed observations carry concrete call bindings and device work for compiling templates or checking structure. Supply `compile --model-config` with the original model configuration when a source lacks it. Device records with unresolved ownership are counted separately.

### Observe a tensor-parallel target

Set `tp` in the task configuration to the logical width. With one exposed card, run the observer with `--mode detailed --placement single_device`. SGLang loads each native shard on that card and exchanges operands through the host. After warmup, the observer profiles one representative at a time. The manifest records the logical ranks, their representatives and the actual profiles. Use `compile` on the whole observation directory to produce one template source.

For a physical reference, expose the devices for the target topology and use `--placement full`. The full recording captures all ranks in the same window. Use a separate observation from the template source when running `verify-structure`.

The current ranked path covers synchronous dense TP with one observed compute stream per rank. It preserves the native publication owner's timing and the other ranks' result dependencies.

### Observe a pipeline target

Set `pp` to the number of pipeline stages and optionally set `microbatch_requests` to the native microbatch limit. Tensor parallelism within each stage still uses `tp`. With `--placement single_device`, the observer places the native shards on one visible device and collects the necessary stage and rank roles in consecutive windows. Use `--placement full` for a simultaneous physical reference.

Compile the single-device observation and use the separate full recording with `verify-structure`. The native control image runs one scheduler loop per pipeline stage. For the qualified PP3 and TP2 × PP3 targets, each stage's original control rank makes the shared batching decisions while the core tracks device work and transfer readiness for each physical rank. Autonomous pipeline prediction reports a cost gap pending qualified cut and transport timing models.

## Fit and publish shared assets

Capture a separate training observation as `host-training` with `--mode host-timing`. Use detailed training observations for device tasks, and reserve separate observations for evaluation.

```bash
store="$PWD/data/libraries"
mkdir -p "$store"

docker run --rm --runtime runc --network none \
  --mount type=bind,src="$run",dst=/case,readonly \
  prologue-app:ubuntu24.04 fit-costs --device /case/detailed --host /case/host-training > "$run/costs.json"
docker run --rm --runtime runc --network none \
  --mount type=bind,src="$run",dst=/case,readonly \
  --mount type=bind,src="$store",dst=/library \
  prologue-app:ubuntu24.04 publish --store /library --library /case/templates.json \
  --costs /case/costs.json
```

The release records the template, performance and rule objects together. `show-release` displays their identifiers. For a template update, repeat `--library` with the full set of desired sources. Omitted asset families use the base release. Pass `--base <release UUID>` when an update is prepared against a specific release.

To combine local shard costs with physical communication costs, use the current default rules with the `collective` family. Each result latency comes from the measured time between the last participant becoming ready and that rank's result becoming available. The cost asset includes the measured directed transport links.

```bash
prologue fit-costs --device single-device-observation --collective full-observation > costs.json
```

`query-structure` uses the same selection and expansion as prediction. Pass a JSON document with the target `domain` and `targets`, an array of `{ "target": "case-id", "query": { "phase": "decode", "sequence_lengths": [9] } }` entries. `--candidate` previews an additional source and `--graph` includes the expanded graph and a definition ID for each call in graph order. The report distinguishes an unobserved domain, a missing parent, a missing child and conflicting candidates. Use `publish --add-library ... --require-structure queries.json` to add sources for the original query domain and resolve those queries before changing the release.

### Review a template sharing rule

A release can contain native roots from several models and configurations. Each root needs evidence from its own domain. Selected leaf definitions with observed device work can share storage across those domains when their complete bodies and native framework revision agree. Costs retain separate qualification for each physical domain.

Copy the [default rules](src/prologue/rules/defaults.json) and set `templates.shared_operations` to the operations supported by your evidence. In `cases.json`, provide a `cases` array. Each case has an `id`, `left` and `right` witnesses, and the expected `shared` Boolean. Each witness supplies the compiled source's `domain`, an observed root `query`, the relative child-slot `path`, and the source's `origin_raw_ns`. For each added operation, provide both a sharing case and a distinct-body case.

```bash
prologue publish --store /library --base "$base_release" --rules proposed.json --rule-cases cases.json --preview
prologue publish --store /library --base "$base_release" --rules proposed.json --rule-cases cases.json
```

Set `base_release` to the current release UUID. Preview reports the case results and affected root queries. A successful publication keeps the report on the new immutable release. Full source replacement with `--library` checks any proposed sharing against an isolated build of the same sources. To rebuild an older asset, compile its original observations with the current command and publish the full source set with a current rules document.

### Compile a native Decode page-table cut

For a native SGLang Prefill/Decode pair, enable the Prologue plugin on Decode and set `PROLOGUE_PD_CUT` to a JSON file in a fresh output directory. Set `configuration_id`, `detail_requests` and `parent_requests`. Use native request IDs for both lists. Run the profiler around `detail_requests` with output under `profile/` in the same directory. Submit `parent_requests` with the profiler stopped. The observer records the effective domain, KV handover and admitted page-table slices.

```bash
prologue compile copy-observation --pd-scope copy --model-config model-config.json > page-copy.json
prologue compile assembly-observation --pd-scope assembly --model-config model-config.json \
  --for-queries queries.json > page-assembly.json
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

The prediction reports the resolved release UUID. Pass that UUID to `--release` to replay the same asset selection later. Autonomous prediction requires released cost coverage for host preparation, forward and result handling, local device work and any collective results.

Each fitted cost document covers one physical domain. Pass `--costs` once per fitted document to publish a complete catalog. Use `--add-costs` with the original demand file to add only missing coordinates to its domain while preserving the other domains.

## Explore a task

Run exploration on the Docker host. It starts the application and native control containers for each target, pins the release and image IDs, and writes a short `result.json` beside the per-target predictions. The default path reads the library without publishing or accessing a physical device.

```bash
PYTHONPATH=src python3 -m prologue.app.explore \
  example/reuse-task.json "$run/explore" --hardware "$run/hardware.json" --store "$store"
```

An installed Prologue CLI exposes the same command as `prologue explore`. Use a new output directory for a new exploration. Run the same command with the original output directory to resume from the saved release, target results and charges.

Add `--acquire` to allow publication. Repeat `--source` for independent, detailed training recordings in the relevant domains. For new physical acquisitions, pass an allocated `--device` and explicit `--attempts`, `--card-seconds` and `--max-bytes` limits. The initial release must be current when publishing. The `--processing-seconds` limit covers active elapsed time, including acquisition. Keep evaluation recordings separate from all acquisition inputs.

A retained source is compiled for the named structure gaps. The eligible TP1 Decode attention case uses a scoped ModelRunner sample. With `--training`, structure acquisition can run an independently declared workload on one card. Each accepted addition must cover its original query before publication. Only affected targets run again. Other gaps and exhausted budgets remain visible in the run report.

## Verify from a release

Use a separate detailed recording as `reference`. The structure verifier answers timing queries while the controller chooses actions from the workload and simulated resources. Run the verifier with the same two-process socket procedure. The application container reads the shared store and the detailed reference.

```bash
prologue verify-structure reference --store /library --release current --socket /control/control.sock
prologue verify-performance reference --store /library --release current
prologue score prediction.json light-reference
```

Fixed-graph verification first checks that measured inputs reproduce the recorded endpoints. The `estimate` replay keeps the native actions and external arrivals fixed and uses the released host and device timing models. The `device_only` replay replaces device durations while retaining measured host and dispatch timing. Per-kind replays restore measured device timing for one kind while preserving the modeled host schedule. Autonomous scoring uses a separate lightweight recording of the prediction workload for TTFT, TPOT and throughput. `quote-costs` accepts a JSON document with `domain` and `coordinates` and uses the same released estimator.

## Collect missing Decode attention costs

`plan-costs` accepts a `targets` array. Each target has a name in `target` and the `domain` and `coordinates` from a reported cost gap. It checks a pinned release and groups uncovered coordinates by actual sampling conditions. The native sampler supports single-member Ascend TP1 Decode and uses SGLang to build the real KV context. Other gaps remain in the plan with their coordinates and affected targets.

```bash
prologue plan-costs gaps.json --store /library > sampling.json
prologue sample-costs sampling.json collection --attempts 1 --card-seconds 180 --max-bytes 2147483648
prologue fit-costs --device collection/attempts/0001/9 --device collection/attempts/0001/17 \
  --for-queries collection/plan.json > added-costs.json
prologue publish --store /library --add-costs added-costs.json --require-costs collection/plan.json
```

Run `sample-costs` in the Ascend observer image with one physical card and the declared model mounted. Run planning, readback and publication in the application image. The sampling receipt lists the actual observation directories for `--device`. Reopening the same collection directory preserves prior attempts and reuses completed observations. Publication adds only requested, previously uncovered coordinates after checking the original query.

## License

Prologue is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies retain their own licenses.
