# The kernel sources of the Ascend backend

This tree holds the Ascend (NPU) kernels of `xinfer/ops/backends/ascend/`, as `cuda/src` holds the CUDA ones
since "Build the CUDA kernels from this repository" (#27): one backend, one self-contained source tree, one
library — `xinfer_kernels_ascend.so`, built by `src/CMakeLists.txt` and loaded by `../_native.py`. The backend
needs no other repository; the external xcompute path (`backends/_xcompute.py`) keeps serving deployments that
prefer a prebuilt external build until this tree covers them.

**Status: scaffold.** The framework is in place — family directories, the CMake build, the loader, the per-SoC
platform keys and the GEMM shape envelopes — while the kernel sources themselves are TODO placeholders that
IN-TREE-R1 moves in from the v22 knife tree (see "R1" below). Nothing here registers an operator until the
library is built; until then the backend runs its existing routes.

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

The kernels are TileLang AOT: each family module builds TileLang programs, the generator (`tools/`, TODO R1)
lowers them, renames the host launcher symbols and hands the host C to bisheng for the `dav-3510` target,
producing `.o` files with the device image embedded (`aclrtBinaryLoadFromData` loads them at run time).
The CMake aggregates those objects with per-family host glue — thin `.cc` files exporting
`TVM_FFI_DLL_EXPORT_TYPED_FUNC` entries that marshal `TensorView` arguments and launch the AOT call on the
current NPU stream — into the single `xinfer_kernels_ascend.so`, linked against the four CANN runtime
libraries. `common/build_info.cc` reports the SoCs the library was built for; `_native.py` reads it and
registers the operators on exactly those platforms.

Whether the AOT launcher symbols keep their historical `xcompute_aot_` prefix or become `xinfer_aot_` during
the move is R1's call (the rename is mechanical in the generator's renaming pass); either way they are an
implementation detail behind the TVM-FFI exports, unlike the ctypes knife libraries of the v22 tree.

## R1: moving the kernels in

- Sources come from the v22 knife tree: the `src/<family>/npu/*.py` TileLang modules and the generator pair
  (`aot/generate.py`, `aot/abi.py`) per the build-spike contract; `finish_br_c1024` needs family-ization first
  (it lives outside the v22 `src/` families), and the two walloss remnant kernels ride along.
- Each family README's table is the checklist: kernel name, envelope, dose, v22 source path, gate evidence.
- The SHAPES envelope of the GEMM family is already declared in `../gemm_950dt.py`; the other families get
  theirs next to their glue as R1 rewrites the wrappers in the CUDA paradigm (`Call` normalization, `takes`,
  `kernel_name`).
- Keep the honest boundary: the vendor faces the audit locked (FusedInferAttentionScore residual families,
  aclnnChunkGatedDeltaRule, the large-shape aclnn Matmul floor, the elementwise soup) are runtime dependencies
  of this library, not ports.
