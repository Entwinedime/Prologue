# Prologue

Prologue predicts LLM serving performance for one model across a set of configurations and request workloads. It expands serving actions from a published template library and assigns duration from a performance library. Results include prediction status, TTFT, TPOT, throughput, and configuration comparisons. Each workload supplies output lengths and any expert routes required by the configurations.

## Build

Run these commands from the checkout on an Ascend host with Docker Compose, Docker BuildKit enabled, the Ascend container runtime, access to `/var/run/docker.sock`, and Python 3.9 or newer. Native observation needs the model files and the physical devices specified by its deployment. The offline example uses the control and modeling images, the compiled executable, and a published Qwen3-8B library release.

```bash
git submodule update --init third_party/sglang
docker build -t prologue-sglang-env:ubuntu22.04 docker/images/base/sglang
./docker/build_inference.sh
docker compose -f docker/compose/inference.yml build modeling
mkdir -p build/coarse-graph
docker run --rm --runtime runc --network none \
  -v "$PWD:/repo:ro" -v "$PWD/build/coarse-graph:/build" \
  prologue-modeling:ubuntu24.04 bash -c \
  'cmake -S /repo -B /build -DCMAKE_BUILD_TYPE=Release && cmake --build /build --parallel 8'
```

You can reuse the base image if it is already available under the specified tag. For a build proxy listening on the host loopback interface, run `PROLOGUE_BUILD_NETWORK=host ./docker/build_inference.sh`.

## Run the example

The example evaluates two Qwen3-8B TP1 configurations against two workloads. Supply a published release directory containing `release.json`. You can copy it to another location before running the command.

```bash
python3 example/coarse_graph/run.py --libraries /path/to/qwen-release --timeout 180
```

The command creates `data/<UTC-timestamp>_exploration/`. Open `summary.md` for the comparison and `result.json` for each target's result or first missing requirement. Use `--input` for a different exploration, `--output` to choose the task directory, and `--details` to retain the expanded graphs and event timeline. Predictions use the published model identity and worker state on CPU. An empty library reports the missing worker binding for each target.

To publish from an existing compatible training collection and fitted cost asset, use the host CLI in a container with the repository mounted. Mount any external recording directories at their original absolute paths. The resulting release holds the worker templates, model bounds, samples, and native recording identities needed for reuse.

```bash
docker run --rm --runtime runc --network none -v "$PWD:$PWD" -w "$PWD" \
  -e PYTHONPATH=src --entrypoint python3 prologue-control:ubuntu22.04 \
  -m prologue library publish --from-collection data/training/collection \
  --costs data/training/performance/costs.json --libraries data/libraries/qwen
```

## Command line

Run `PYTHONPATH=src python3 -m prologue --help` to see the five command groups. Installing the host package also provides `prologue-cli`. Each command has its own help.

| Group | Commands |
| --- | --- |
| `predict` | `run`, `explore` |
| `library` | `inspect`, `acquire`, `publish` |
| `capture` | `run`, `project` |
| `cost` | `extract`, `fit`, `acquire` |
| `verify` | `source`, `inputs`, `local-graphs`, `structure`, `fixed-graph`, `autonomous` |

## Validated configurations

The Ascend delivery used SGLang, dummy weights, and workloads with fixed output lengths. Each entry describes a measured configuration and workload mechanism. The listed evidence comes from earlier native measurements. The four-target example above exercises the offline reader on TP1.

| Model | Dtype | Configuration | Checked cases |
| --- | --- | --- | --- |
| Qwen3-8B | BF16 | TP1 with 16-token pages | Page boundaries and normal and burst request traffic |
| Qwen3-8B | BF16 | TP2 with 16-token pages | Dual-card serving, chunked prefill with mixed batches, and first-level prefix-cache hits and eviction |
| Qwen3-8B | BF16 | TP8 with 32-token pages | Two staggered requests on eight cards |
| Qwen3-8B | BF16 | Prefill TP1 and Decode TP2 with 16-token pages | Asymmetric disaggregation with KV and metadata handoffs |
| Mistral 7B | FP16 | PP3 with 32-token pages and TP2 × PP3 with 16-token pages | Pipeline batches and combined tensor and pipeline parallelism. Host transfer timing is checked at the endpoints. |
| Qwen3-30B-A3B | BF16 | TP4 × EP4 with 128-token pages | Expert parallelism with explicit routes in the workload |

See the [development and workflow guide](docs/design/development.md) for input, output, and verification details.

## License

Prologue is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies retain their own licenses.
