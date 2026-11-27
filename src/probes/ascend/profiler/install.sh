#!/usr/bin/env bash
set -euo pipefail

PROLOGUE_PROFILER_INSTALL=$(cd "$(dirname "$0")" && pwd)
PROLOGUE_TORCH_NPU_ROOT=$(python3 -c "import importlib.util; print(importlib.util.find_spec('torch_npu').submodule_search_locations[0])")
PROLOGUE_TORCH_NPU_REVISION=$(python3 -c "import runpy, sys; print(runpy.run_path(sys.argv[1])['git_version'])" "$PROLOGUE_TORCH_NPU_ROOT/version.py")
PROLOGUE_PROFILER_SOURCE=$(mktemp -d)
trap 'rm -r "$PROLOGUE_PROFILER_SOURCE"' EXIT

# Build the installed SDK revision with ordered startup and queue recorder timing.
curl --fail --location --max-time 120 \
    "https://codeload.github.com/Ascend/pytorch/tar.gz/$PROLOGUE_TORCH_NPU_REVISION" \
    | tar --extract --gzip --directory "$PROLOGUE_PROFILER_SOURCE" --strip-components=1 \
        --wildcards '*/torch_npu/csrc/toolkit/profiler/*'
patch --batch --fuzz=0 --directory "$PROLOGUE_PROFILER_SOURCE" --strip=1 < "$PROLOGUE_PROFILER_INSTALL/dumper.patch"

# Building this CPU-only library does not load or initialize the device backend.
TORCH_DEVICE_BACKEND_AUTOLOAD=0 cmake -S "$PROLOGUE_PROFILER_INSTALL" -B "$PROLOGUE_PROFILER_SOURCE/build" \
    -DCMAKE_BUILD_TYPE=Release -DPROLOGUE_TORCH_NPU_SOURCE_DIR="$PROLOGUE_PROFILER_SOURCE"
cmake --build "$PROLOGUE_PROFILER_SOURCE/build" --target npu_profiler --parallel 2
install -m 755 "$PROLOGUE_PROFILER_SOURCE/build/profiler/libnpu_profiler.so" \
    "$PROLOGUE_TORCH_NPU_ROOT/lib/libnpu_profiler.so"
