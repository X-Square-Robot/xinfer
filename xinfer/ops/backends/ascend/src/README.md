# The kernel sources of the Ascend backend

This tree holds the Ascend (NPU) kernels of `xinfer/ops/backends/ascend/`, as `cuda/src` holds the CUDA ones
since "Build the CUDA kernels from this repository" (#27): one backend, one self-contained source tree, one
library — `xinfer_kernels_ascend.so`, built by `src/CMakeLists.txt` and loaded by `../_native.py`. The backend
needs no other repository; the external xcompute path (`backends/_xcompute.py`) keeps serving deployments that
prefer a prebuilt external build until this tree covers them.

**Status: R1a.** The kernel sources have moved in — the family directories carry the 44 modules of the
porting audit (see "R1a" below) — and the CMake build, the loader, the per-SoC platform keys and the GEMM
shape envelopes are in place; the family-library transition build (`build-libraries.sh`) reproduces the
delivered library set. The cuda-paradigm wrappers (`takes`/`kernel_name` per family) and the TVM-FFI glue
faces are R1b; until they land, the backend registers through its transition loader and runs its existing
routes. Nothing here registers an operator until a library is built.

## Families

The directories follow the compute families of the two production models (Wall-OSS and Wall-X 2B), which align
with `cuda/src` where the hardware does:

| Directory | Kernels | Models | `cuda/src` counterpart |
|---|---|---|---|
| `gemm/` | 7 | wall2b + walloss | `gemm/` (the generic path is aclnn, not cuBLASLt) |
| `attn/` | 7 | wall2b + walloss | `attention/` |
| `gdn/` | 6 | wall2b | `gated_delta_rule/` |
| `norm/` | 9 | wall2b + walloss | `norm/` |
| `denoise/` | 3 | wall2b | `denoise_output/` |
| `elem/` | 1 | wall2b | `activation/`, `quant/` |
| `resample/` | 4 | wall2b | — (Ascend vision preprocessing) |
| `finish/` | 7 | wall2b + walloss | — (GEMM-fused epilogues live in `cuda/src/gemm/sm120` there) |
| `common/` | — | — | `common/` (FFI bridge, build info) |

44 kernel names in total: Wall-X 2B serves 29, Wall-OSS 15 + 2 one-per-request remnants, with the two `smallm`
GEMMs shared. Every kernel's shape envelope, dose and gate evidence is in the family README tables; the machine
readable source is the porting inventory CSV (2026-11-16 audit). The kernels a family does not take stay on the
aclnn vendor operators or the torch references — the audit fixed that split, and this tree does not widen it.

Each family README states which model calls which kernel at which dose, so a model owner can find every kernel
their model depends on from this tree alone.

## File naming

- A kernel module is named after its operator, snake_case, as in `cuda/src`: `gemm/smallm_r3.py`,
  `norm/adanorm_r2.py`. A kernel compiled for one SoC's instructions carries the SoC as a suffix and lives in a
  `<soc>/` subdirectory, as `sm120` does on the CUDA side.
- TileLang AOT kernel modules (the DSL sources the generator compiles) sit beside any hand-written host glue;
  the CMake story below explains how both become objects in the one library.

## Build

The build is explicit CMake, never pip (`pip install` only installs the `apache-tvm-ffi` runtime; see
`pyproject.toml`), and runs on the aarch64 build host (openEuler 24.03). From the repository root:

```bash
pip install -e ".[kernels]"
cmake -S xinfer/ops/backends/ascend/src -B build/kernels-ascend
cmake --build build/kernels-ascend -j1
```

This writes `xinfer/ops/backends/ascend/xinfer_kernels_ascend.so` next to the Python modules that call it.

### Toolchain (the validated pairing)

| Piece | Version / path | Role |
|---|---|---|
| bisheng | `/usr/local/Ascend/cann-9.3.0/bin/bisheng` (clang 15.0.5) | device compiler (`BISHENG_HOME`, `ASCEND_HOME_PATH`, `ASCEND_TOOLKIT_HOME`) |
| CANN runtime | `/usr/local/Ascend/cann-9.1.1` (`lib64`: ascendcl, opapi, runtime, nnopbase) | link and load; `set_env.sh` before Python starts |
| Python | the TileLang fork venv (`tilelang 0.1.15+cpu` fork `git1e8448fc`, source install) | runs the AOT generator |
| `apache-tvm-ffi` | 0.1.12, in the venv and in `[kernels]` | the FFI ABI of the library |
| SoC compile target | `dav-3510` | the 950DT device target the generator passes to bisheng |
| Discipline | `MAX_JOBS=1`, private `TILELANG_CACHE_DIR` | memory and cache hygiene on shared hosts |

Environment order matters and is a validated trap: source the **9.1.1** `set_env.sh` first (TorchNPU import
needs the runtime libraries), then override the three bisheng variables to **9.3.0**. The generator needs no
NPU device at build time; a device only witnesses the result. Two independent from-scratch builds of the soup2
family reproduced the delivered objects byte for byte under this recipe.

### From DSL source to the one library

The kernels are TileLang AOT: each family module builds TileLang programs, the generator (`tools/generate.py`, landed R1a)
lowers them, renames the host launcher symbols and hands the host C to bisheng for the `dav-3510` target,
producing `.o` files with the device image embedded (`aclrtBinaryLoadFromData` loads them at run time).
The CMake aggregates those objects with per-family host glue — thin `.cc` files exporting
`TVM_FFI_DLL_EXPORT_TYPED_FUNC` entries that marshal `TensorView` arguments and launch the AOT call on the
current NPU stream — into the single `xinfer_kernels_ascend.so`, linked against the four CANN runtime
libraries. `common/build_info.cc` reports the SoCs the library was built for; `_native.py` reads it and
registers the operators on exactly those platforms.

Whether the AOT launcher symbols keep their historical `xcompute_aot_` prefix or become `xinfer_aot_` during
the move is R1's call (the rename is mechanical in the generator's renaming pass); either way they are an
implementation detail behind the TVM-FFI exports, unlike the ctypes knife libraries of the v22 tree. R1a's
answer: the delivered libraries spell `xinfer_aot_` (the zero-xcompute grep assertion is unconditional); the
generator's `--symbol-prefix` reproduces either spelling for anchor builds.

## R1a: the kernels have moved in

The 44 kernel modules of the porting audit live in the family directories (`gemm/smallm_r3.py`,
`norm/adanorm_r2.py`, …), moved in from the v22 knife tree (IN-TREE-R1a, 2026-11-17): the TileLang DSL
modules beside their family READMEs, the generator pair in `tools/` (`generate.py` + `abi.py`, the
build-spike contract), the C-ABI static bridges in `common/` beside `build_info.cc`, and the rosters'
library map in `FAMILIES.md`. `finish_br_c1024` and `decodeattn_r4` arrived family-ized; the two walloss
remnant kernels ride along. The modules keep their Apache-2.0 headers (they come from the xcompute
project, as the CUDA sources did). Module roots resolved by the generator are `<family>.<module>` from
this directory (e.g. `--module gemm.smallm_r3`).

Each family README's table is the checklist: kernel name, envelope, dose, v22 source path, gate evidence.
The SHAPES envelope of the GEMM family is declared in `../gemm_950dt.py`; the other families get theirs
next to their glue as R1b rewrites the wrappers in the CUDA paradigm (`Call` normalization, `takes`,
`kernel_name`). Keep the honest boundary: the vendor faces the audit locked (FusedInferAttentionScore
residual families, aclnnChunkGatedDeltaRule, the large-shape aclnn Matmul floor, the elementwise soup)
are runtime dependencies of this library, not ports.

## The transition build: the 17 family libraries

`build-libraries.sh` (this directory) is the validated byte-anchor recipe the 20.62/38.82 stacks run on:
one core library plus one knife library per routed family (the map in `FAMILIES.md`), linked against
the CANN runtime four, loaded through the backend's transition loader with the deployment's
`XINFER_<FAMILY>_LIB` names preserved verbatim. `build-libraries.sh install` fills `../lib/`, the
loader's default search path. It is the transition form of `CMakeLists.txt` (which aggregates the same
objects into the one `xinfer_kernels_ascend.so`); R1b retires it when the TVM-FFI glue faces land.

### Symbol prefix and byte anchors

Every exported launcher uses `xinfer_aot_<kernel>_{call,init,get_last_error}` (plus
`xinfer_aot_<kernel>_device` as the device symbol and the static bridges' `xinfer_*` C ABI). The v22
`xcompute_aot_` names are intentionally not kept: the rename makes the zero-xcompute grep assertion
unconditional and prevents the two branches' libraries from loading interchangeably. `tools/generate.py
--symbol-prefix xcompute_aot_` reproduces the v22 delivery objects byte for byte (anchor builds).

Each family's delivered `.o`/`.asc`/`.tir` must equal the delivered library set (`SHA256SUMS.objects` of
the serving build) when built with the vendored, unmodified generator at the same prefix; the
`build-libraries.sh` recipe (toolchain pin below) is the byte-for-byte contract. Mismatch downgrades to
the bit-level witness (same inputs through the delivered library and the new library, outputs compared
bitwise). Symbol renames and the family-directory layout change only host-side names; kernel lowering is
untouched. The AA-resample real6 objects are assets, not regenerable sources (the auto-scheduler
deadlock documented in `build-libraries.sh`); they are renamed and relinked, never regenerated.

