# Ascend kernel library

The Wall-OSS and Wall-X 2B AOT entries build into `xinfer_kernels_ascend.so`, next to
the Python backend. `_native.py` loads it through TVM-FFI; existing C-ABI
adapters use the same file. No external xcompute package or family libraries
are required at runtime.

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

All objects are built from this tree. The AA downsample programs in
`resample/generated` are the original generated device and host source, compiled
directly to preserve their validated schedule. Their required template headers
and license are included; no precompiled objects or external operator checkout
are needed. Other AOT objects are generated from the checked-in TileLang source.
Both compilation paths use the included TileLang template headers. The
TileLang compiler and standard Ascend SDK remain toolchain dependencies.

## Source layout

| Directory | Operations |
|---|---|
| `gemm` | projections and fused projection finishes |
| `attn` | attention and gated activation |
| `gdn` | Gated DeltaNet preparation and convolution |
| `norm` | normalization and rotary/split layouts |
| `denoise` | flow input and output |
| `elem`, `finish` | elementwise and GEMM epilogues |
| `resample` | image resizing |
| `common` | CANN bridges and build information |

Kernel implementations, launch ABIs and numerical envelopes are carried from
the existing Ascend implementation. The build links the entries listed in
`CMakeLists.txt`. Unsupported layouts use the backend's existing fallback.
See [kernel integration](../../../../../docs/kernels.md) for registration,
graph lifetime and validation requirements.
