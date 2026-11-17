#!/bin/bash
# aot-generate.sh — one TileLang family module through the AOT generator, with the environment order
# the validated pairing fixed (source the CANN runtime's set_env.sh first, then point the bisheng trio
# at the compiler CANN) and the build disciplines (MAX_JOBS=1, a private TILELANG_CACHE_DIR).
# Called by src/CMakeLists.txt; the same contract build-libraries.sh drives inline.
#
#   aot-generate.sh <out-dir> <module> <kernel>...
set -euo pipefail

OUT=$1; shift
MODULE=$1; shift

CANN_RUNTIME=${XINFER_CANN_RUNTIME:-/usr/local/Ascend/cann-9.1.1}
BISHENG=${XINFER_BISHENG_HOME:-/usr/local/Ascend/cann-9.3.0}
PY=${XINFER_TILELANG_PYTHON:-/root/tl-fork-store/venv-tl-src/bin/python}
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
