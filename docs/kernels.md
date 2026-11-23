# Kernels

English | [简体中文](kernels.zh-CN.md)

The default compositions for WALL-OSS-0.5 and WALL-OSS-1.0-Nano use fused operators on CUDA. Their kernels are in this repository (`xinfer/ops/backends/cuda/src`) and build into one library, `xinfer_kernels.so`. XInfer runs the same compositions on PyTorch reference implementations when the library is not built. Install the optional `flash-attn` package for vision and prompt attention.

The built-in `cuda` backend modules (`xinfer/ops/backends/cuda/`) call the kernel library and flash-attn and load the same way as extensions in the `xinfer.ops` entry point group ([registration](design.md#142-backend-registration-and-selection)). Operators whose library does not load run on their torch references, and a warning lists the slow ones; `XINFER_OPS_STRICT=1` makes that warning an error. For example, set `XINFER_FLASH_ATTN=0` to skip flash-attn at load time.

## GPU support

The kernel library is built for one GPU architecture and registers its operators on that architecture only. This repository has kernels for SM120 (RTX 50 series) and SM110 (Jetson Thor).

A GEMM runs on a kernel written for its architecture when the call is one that kernel was tuned on, and otherwise on cuBLASLt with the epilogue as a separate kernel. Each kernel lists the shapes it takes next to the code that launches it (`xinfer/ops/backends/cuda/gemm_sm120.py`, `gemm_sm110.py`): on an RTX 5090 the ranges of rows over which it was measured not slower than the cuBLASLt path; on a Jetson AGX Thor the ranges over which it was measured at least 5% faster at three or more consecutive scan points, matching the reference SM110 kernel build's selection and its bitwise outputs on the reference requests. On SM110 the library covers the BF16 compositions of WALL-OSS-0.5 and WALL-OSS-1.0-Nano. The attention and flow kernels list their shapes the same way (`attn.py`, `flow.py`), and the calls outside them run on the torch reference.

## Build

Building needs the CUDA 13 toolkit, CMake 3.24 or later (3.28 for SM110), the `apache-tvm-ffi` Python package and the CUTLASS headers (a submodule):

```bash
git submodule update --init third_party/cutlass
pip install -e ".[kernels]"
cmake -S xinfer/ops/backends/cuda/src -B build/kernels
cmake --build build/kernels -j
```

This writes `xinfer/ops/backends/cuda/xinfer_kernels.so`. The target architecture is the first GPU of the build machine; `-DXINFER_CUDA_ARCH=120` (or `110`) sets it on a machine without that GPU.

## Run

```bash
xinfer-serve --model wall-oss-0.5 --checkpoint /path/to/checkpoint --device cuda \
  --warmup observation.msgpack --port 33985
```

XInfer loads `xinfer_kernels.so` from its package directory. `XINFER_KERNELS_LIBRARY=/path/to/xinfer_kernels.so` loads a library built elsewhere from the same sources. `XINFER_KERNELS=0` turns the backend off. The same variables apply to `Policy`, the tests and the benchmarks.

## Check

`xinfer-serve` logs each operator's backend when the model is loaded:

```
xinfer.serve INFO model loaded in 7.2s on cuda-sm120; ops {..., 'gemm_epilogue': 'kernels', 'gemm_glu_epilogue': 'kernels', ...}
```

`--resolved-config FILE` writes the same list to `FILE`, and in Python `Policy.describe()["ops"]` lists it. With the library loaded the fused operators (`gemm_epilogue`, `gemm_glu_epilogue`, `qkv_rope_kvcache` and the other operators of `xinfer/ops/reference/` that the model uses) report `kernels`. `reference` for them means XInfer did not load it:

- `xinfer/ops/backends/cuda/xinfer_kernels.so` is not built, or `XINFER_KERNELS_LIBRARY` points to a missing file;
- the `apache-tvm-ffi` package is not installed;
- the library was built for another GPU architecture.

`Policy` and `xinfer-serve` then log a warning when they load WALL-OSS-0.5 on a GPU:

```
wall-oss-0.5: 7 operators have no fast kernel loaded and fall back to slow torch code (add_rms_norm, ...). Build the
CUDA kernel library for this GPU: docs/kernels.md
```

`pytest tests/kernels` runs every kernel on the shapes it takes against FP32 torch; it needs the GPU and the built library.

## Ascend

The `ascend` backend binds an external xcompute build through `xinfer/ops/backends/_xcompute.py`, with the
`XINFER_XCOMPUTE_LIBRARY` / `XINFER_XCOMPUTE=0` settings. Available implementations are selected automatically;
unsupported calls use their torch fallback. See the [Ascend guide](ascend950dt.md) for runtime, image processing
and NPU graphs.

## Tuning generic GEMMs

The generic GEMM uses cuBLASLt's first heuristic by default. With a library built from this revision, set `XINFER_GEMM_TUNING=1` before startup to time eligible algorithms on the first eager call for each exact shape and alignment. A candidate must match the heuristic output on the warmup input and beat its median time by 5%. Trials write to private output storage. In-place residual updates keep the heuristic. Set `XINFER_GEMM_TUNING_FILE` to a writable JSON path to share choices across processes. For reproducible replay set `XINFER_GEMM_TUNING=frozen` and point it at that file. Warm each shape eagerly before graph capture. Frozen mode reports a missing or incompatible choice; a cold capture in ordinary tuning mode uses the heuristic.
