# kernels/ — the unified Ascend operator sources (feat/ascend-kernels-unified)

Sources only: **no binary product enters the repository** (`lib/` is empty and
gitignored; `build.sh install` fills it on the target machine). This is the same
policy the repo already uses for `xinfer/_preparefast.c`.

## Toolchain pin (BUILD-SPIKE 2026-11-16, byte-for-byte contract)

| dependency | pin |
|---|---|
| python | `/root/tl-fork-store/venv-tl-src/bin/python` (3.11.6) |
| tilelang | `0.1.15+cpu.git1e8448fc` (source-installed fork, `USE_ASCEND=ON USE_CUDA=OFF`) |
| bisheng (device compiler) | `/usr/local/Ascend/cann-9.3.0` (clang 15.0.5, `BISHENG_HOME` locates it) |
| CANN runtime (link+load) | `/usr/local/Ascend/cann-9.1.1` (`-lascendcl -lopapi -lruntime -lnnopbase`) |
| g++ | 12.3.1 (openEuler), `-shared` host link |
| NPU arch | `dav-3510` |
| build discipline | `MAX_JOBS=1` |

`build.sh` asserts the tilelang version and fails fast on drift. No xcompute
CMake, no tvm_ffi runtime, no torch linkage: the emitted `.so` files load through
pure ctypes and depend only on the CANN runtime four (`ldd` assertion in
`tools/audit_ascend_xcompute_free.sh`).

## Build

```bash
cd xinfer/ops/backends/ascend/kernels
./build.sh            # products under .build/ (out/, lib*.so, SHA256SUMS.*)
./build.sh install    # copies lib*.so into lib/ (the loader's default search path)
```

Recipes and pitfalls are the BUILD-SPIKE contract, verbatim:

1. `source /usr/local/Ascend/cann-9.1.1/set_env.sh` **before** python starts
   (torch_npu device backend import crashes otherwise); then point
   `BISHENG_HOME/ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME` and `PATH` at 9.3.0.
2. bisheng 9.3.0 compiles, CANN 9.1.1 links and loads — the validated pairing.
3. `generate.py` runs with cwd `tools/` (`from abi import ...` and the
   `--module ops...` package roots resolve from the script directory).
4. `TILELANG_CACHE_DIR` is per-invocation under `.build/tl-cache` (cache hits do
   not change product bytes — the spike proved this — but keep runs isolated
   anyway).
5. Object order in the `g++ -shared` link is the manifest order (the .so bytes
   are order-sensitive).

## Symbol prefix

Every exported launcher uses `xinfer_aot_<kernel>_{call,init,get_last_error}`
(plus `xinfer_aot_<kernel>_device` as the device symbol and the static bridges'
`xinfer_*` C ABI). The v22 `xcompute_aot_` names are intentionally **not** kept:
the rename makes the zero-xcompute grep assertion unconditional and prevents the
two branches' libraries from loading interchangeably. `generate.py
--symbol-prefix xcompute_aot_` reproduces the v22 delivery objects byte for byte
(anchor builds; see the S2 report in the UNIFY-PORT run directory).

## Byte anchors (S2 verification contract)

Each family's delivered `.o`/`.asc`/`.tir` must equal the v22 delivery
(`$R/<knife>/SHA256SUMS` or the knife `aot-core/manifest.json` object hashes) when
built with the vendored, unmodified generator at prefix `xcompute_aot_`. Mismatch
downgrades to the bit-level witness (same inputs through the v22 library and the
new library, outputs compared bitwise — `spike-smoke.py` template extended to all
44 kernels). Symbol renames and the computation-type directory layout change only
host-side names; kernel lowering is untouched.
