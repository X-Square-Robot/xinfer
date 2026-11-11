# Kernels

English | [简体中文](kernels.zh-CN.md)

The default BF16 compositions for Wall-OSS and Wall-X 2B use xcompute's fused operators on CUDA. XInfer can run the same composition using PyTorch reference implementations when a compatible xcompute build is unavailable. Install the optional `flash-attn` package for vision and prompt attention.

The built-in `cuda` backend modules (`xinfer/ops/backends/cuda/`) call the xcompute and flash-attn libraries and load the same way as extensions in the `xinfer.ops` entry point group ([registration](design.md#142-backend-registration-and-selection)). Operators whose library does not import run on their torch references, and a warning lists the slow ones; `XINFER_OPS_STRICT=1` makes that warning an error. For example, set `XINFER_FLASH_ATTN=0` to skip flash-attn at load time.

## GPU support

The xcompute build must target the architecture of the serving GPU. Check `xcompute.build_target_archs()`. The [Wall-OSS performance measurements](performance.md) used an RTX 5090 and xcompute `7336b0d`.

## Install and select a build

Install a matching xcompute Python package and shared library from your provider. Neither is bundled with XInfer. For BF16, use a build with the unified GEMM, attention and norm entries. A build that supports `prepack_weight(layout="kn")` uses transposed weights without copying. Older builds copy these weights to a contiguous tensor. Wall-X 2B selects its additional CUDA operators individually when the build provides their entry points. If using a CUDA 13 source build, install the `apache-tvm-ffi` Python package in the same environment and compile for the deployment GPU.

## Run

Put the Python package of the checkout on the path; it finds the library the checkout built:

```bash
export PYTHONPATH=/path/to/xcompute/python:$PYTHONPATH
xinfer-serve --model wall_oss --checkpoint /path/to/checkpoint --device cuda \
  --warmup observation.msgpack --port 33985
```

XInfer loads the library of the importable `xcompute` package and uses its unified GEMM, attention and norm entries,
with the package's rebuild-on-import turned off. `XINFER_XCOMPUTE_LIBRARY=/path/to/xcompute.so` loads another
library instead, such as another BF16 build; the package must then come from the same build. `XINFER_XCOMPUTE=0`
turns the backend off. The same variables apply to `Policy`, the tests and the benchmarks.

## Check

`xinfer-serve` logs each operator's backend when the model is loaded:

```
xinfer.serve INFO model loaded in 7.2s on cuda-sm120; ops {..., 'gemm_epilogue': 'xcompute', 'gemm_glu_epilogue': 'xcompute', ...}
```

`--resolved-config FILE` writes the same list to `FILE`, and in Python `Policy.describe()["ops"]` lists it. With a
working build the fused operators (`gemm_epilogue`, `gemm_glu_epilogue`, `qkv_rope_kvcache` and the other
operators of `xinfer/ops/reference/fused.py` that the model uses) report `xcompute`. `reference` for them means XInfer found no
usable build:

- no `xcompute` package is importable and `XINFER_XCOMPUTE_LIBRARY` is unset, or it points to a missing file;
- the build does not target this GPU's architecture (`python -c "import xcompute; print(xcompute.build_target_archs())"`);
- the `xcompute` package lacks the unified entries;
- the library lacks the functions those entries call: it comes from a different build than the package.

`Policy` and `xinfer-serve` then log a warning when they load Wall-OSS on a GPU:

```
wall_oss: 7 operators have no fast kernel loaded and fall back to slow torch code (add_rms_norm, ...). Install
xcompute built for this GPU, or set XINFER_XCOMPUTE_LIBRARY to such a build: docs/kernels.md
```

xcompute may select its generic cuBLASLt GEMM when a build has no shape-specific kernel for a weight.
