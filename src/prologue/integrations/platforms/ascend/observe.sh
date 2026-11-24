#!/usr/bin/env bash
set -eo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# The capture needs the real driver before toolkit analysis libraries.
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
case "${1:-}" in
    sample-costs|sample-structure) exec prologue "$@" ;;
esac
exec python3 -m prologue.app.observe "$@"
