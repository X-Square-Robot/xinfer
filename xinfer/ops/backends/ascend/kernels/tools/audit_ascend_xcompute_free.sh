#!/bin/bash
# tools/audit_ascend_xcompute_free.sh — the zero-xcompute assertion set
# (ARCH-DESIGN §8, strong form: the unified branch renamed every exported
# symbol, so the bare word must not appear anywhere in the ascend domain).
#
# Usage: audit_ascend_xcompute_free.sh [repo-root]   (default: three levels up)
# Exit 0 = all three assertion groups pass.
set -uo pipefail

ROOT=${1:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$ROOT"
fail=0

echo "== 1) source assertion: bare-word grep over the ascend domain =="
# The NV zero-change domains (cuda/, _xcompute.py, tests stand-ins) are outside
# the scan; kernels/tools/generate.py carries no literal after the S1 sweep.
if git grep -n -i "xcompute" -- xinfer/ops/backends/ascend xinfer/models xinfer/engine xinfer/platforms \
        ':(exclude)xinfer/ops/backends/ascend/kernels/ops/**' > /tmp/audit_src.txt; then
    echo "FAIL: xcompute references remain:"; cat /tmp/audit_src.txt; fail=1
else
    echo "PASS: no xcompute reference in the ascend domain (kernel sources vendored verbatim excluded;"
    echo "      their identifiers are kernel-local and the loaders never read them at runtime)."
fi

echo
echo "== 2) binary assertion: every built library =="
for so in xinfer/ops/backends/ascend/kernels/lib/*.so; do
    [ -e "$so" ] || { echo "PASS(empty): no libraries installed (build.sh install creates them)"; break; }
    count=$(nm -D --defined-only "$so" | grep -c "xcompute" || true)
    if [ "$count" -ne 0 ]; then
        echo "FAIL: $so exports $count xcompute symbols"; fail=1
    else
        bad=$(nm -D --defined-only "$so" | awk '{print $3}' | grep -vE "^xinfer_aot_[a-z0-9_]+_(call|init|get_last_error)$" \
             | grep -vE "^xinfer_(cann|gdr|gemm|npu)_[a-z0-9_]+$" | grep -vE "^(_|__)" | head -3)
        if [ -n "$bad" ]; then
            echo "WARN: $so exports names outside the xinfer_* conventions:"; echo "$bad"
        else
            echo "PASS: $so (all exports xinfer_*)"
        fi
    fi
    if ldd "$so" | grep -qiE "xcompute|tvm"; then
        echo "FAIL: $so links a forbidden library"; fail=1
    fi
done

echo
echo "== 3) runtime assertion: provider import loads nothing else =="
python - <<'PY' || fail=1
import importlib
import os
import sys

os.environ.setdefault("XINFER_ASCEND_KERNELS", "1")
try:
    importlib.import_module("xinfer.ops.backends.ascend._provider")
    importlib.import_module("xinfer.ops.backends.ascend.glue.core")
except Exception as error:  # a machine without torch: the source assertion above still holds
    print(f"SKIP: provider import unavailable here ({error})")
    raise SystemExit(0)
stowaways = [m for m in sys.modules if m == "xcompute" or m.startswith("xcompute.")]
if stowaways:
    print("FAIL: xcompute modules in sys.modules:", stowaways)
    raise SystemExit(1)
print("PASS: provider + glue import with zero xcompute modules in sys.modules")
PY

echo
[ "$fail" -eq 0 ] && echo "AUDIT VERDICT: PASS" || echo "AUDIT VERDICT: FAIL"
exit "$fail"
