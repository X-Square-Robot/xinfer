# Ascend kernel library

The Wall-OSS and Wall-X 2B AOT entries build into `xinfer_kernels_ascend.so`, next to
the Python backend. TVM-FFI and the C-ABI launch adapters use the same library.

The in-tree JIT entries compile during warmup.

## Build

Build on aarch64 with TorchNPU, the TileLang Ascend compiler and `apache-tvm-ffi`.
Use one CANN 9.3 Toolkit installation with the matching 950DT operator package.

```bash
CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann}
source "$CANN_ROOT/set_env.sh"
python -m pip install -e '.[kernels]'
cmake -S xinfer/ops/backends/ascend/src -B build/kernels-ascend \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DXINFER_TILELANG_PYTHON="$(command -v python)" \
  -DXINFER_ASCEND_SOC=ascend950dt \
  -DXINFER_CANN_RUNTIME="$CANN_ROOT" \
  -DXINFER_BISHENG_HOME="$CANN_ROOT"
cmake --build build/kernels-ascend --parallel "${BUILD_JOBS:-4}"
```

All AOT objects are generated from the checked-in TileLang source with the
included template headers. No precompiled objects or external operator checkout
are needed. The TileLang compiler and standard Ascend SDK remain toolchain dependencies.

## Source layout

| Directory | Operations |
|---|---|
| `gemm` | projections and fused projection finishes |
| `attn` | attention and gated activation |
| `gdn` | Gated DeltaNet preparation and convolution |
| `norm` | normalization and rotary/split layouts |
| `denoise` | flow input and output |
| `elem`, `finish` | elementwise and GEMM epilogues |
| `common` | CANN bridges and build information |

The build links the entries listed in `CMakeLists.txt`. The backend adapters
define the launch ABIs and supported layouts. Kernel names describe their
operation and, where needed, their specialization: `m16` means 16 rows,
`h1024` a hidden width of 1024, and `rows8` an eight-row tile.
FIA V5 names the version of the CANN API.
See [kernel integration](../../../../../docs/kernels.md) for registration,
graph lifetime and validation requirements.
