# Ascend kernel library

The Wall-OSS and Wall-X 2B kernels build into `xinfer_kernels_ascend.so`, next to
the Python backend. `_native.py` loads it through TVM-FFI; existing C-ABI
adapters use the same file. No external xcompute package or family libraries
are required at runtime.

## Build

Build on aarch64 with TorchNPU, the TileLang Ascend compiler and `apache-tvm-ffi`.
The supported target is `ascend950dt9582` (`dav-3510`). The tested toolchain uses
CANN 9.1.1 at runtime and bisheng from CANN 9.3.0 for device compilation.
Initialize the runtime environment before importing TorchNPU.

```bash
python -m pip install -e '.[kernels]'
cmake -S xinfer/ops/backends/ascend/src -B build/kernels-ascend \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DXINFER_TILELANG_PYTHON="$(command -v python)" \
  -DXINFER_ASCEND_SOC=ascend950dt9582 \
  -DXINFER_CANN_RUNTIME=/path/to/cann-runtime \
  -DXINFER_BISHENG_HOME=/path/to/cann-compiler \
  -DXINFER_AA_REAL6_DIR=/path/to/validated-aa-objects
cmake --build build/kernels-ascend -j1
```

The four AA-resampling objects are validated build inputs, preserved without
regeneration. The original compiler cannot reliably regenerate that schedule;
their absence is a build error. `XINFER_AOT_OBJECT_DIR` optionally selects
previously generated objects for the other kernels, using the same family
directory layout as the CMake generation targets.

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
