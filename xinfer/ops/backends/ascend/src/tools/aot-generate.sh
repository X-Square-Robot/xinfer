#!/bin/bash
# Compile one TileLang family in the selected CANN toolkit environment.
#
#   aot-generate.sh <out-dir> <module> <kernel>...
set -euo pipefail

OUT=$1; shift
MODULE=$1; shift

CANN_RUNTIME=${XINFER_CANN_RUNTIME:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann}}
BISHENG=${XINFER_BISHENG_HOME:-$CANN_RUNTIME}
PY=${XINFER_TILELANG_PYTHON:-python}
SRC_ROOT=$(cd "$(dirname "$0")/.." && pwd)

set +u; source "$CANN_RUNTIME/set_env.sh" >/dev/null 2>&1 || true; set -u
export BISHENG_HOME=$BISHENG ASCEND_HOME_PATH=$BISHENG ASCEND_TOOLKIT_HOME=$BISHENG
export PATH="$BISHENG/bin:$PATH"
export MAX_JOBS=1
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$OUT/../tl-cache}"
mkdir -p "$TILELANG_CACHE_DIR"

cd "$SRC_ROOT/tools"
PYTHONPATH="$SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" generate.py \
    --output "$OUT" --module "$MODULE" --kernels "$@"
