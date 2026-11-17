# Ascend 950DT

English | [简体中文](ascend950dt.zh-CN.md)

Wall-OSS and Wall-X 2B use the shared operator registry and engine on Ascend 950DT. The in-tree AOT kernels build into `xinfer_kernels_ascend.so`; no external xcompute package is needed. Wall-OSS also retains in-tree TileLang JIT kernels, compiled during warmup. Unavailable implementations use their registered fallback.

## Build

Build on aarch64 with matching PyTorch/TorchNPU, the Ascend TileLang compiler environment, CMake 3.24+ and TVM-FFI. The toolchain uses CANN 9.1.1 for runtime/linking and bisheng from CANN 9.3.0 for compilation. Source the runtime environment before importing TorchNPU. AOT compilation uses the operator code and template headers in this tree; no precompiled objects or external operator checkout are needed. AA-resample compiles its original generated device and host source directly.

```bash
source /path/to/cann-9.1.1/set_env.sh
export BISHENG_HOME=/path/to/cann-9.3.0
export ASCEND_HOME_PATH="$BISHENG_HOME" ASCEND_TOOLKIT_HOME="$BISHENG_HOME"
export TILELANG_CACHE_DIR="$PWD/artifacts/tilelang-cache"
python -m pip install -e ".[kernels,models,serve]"
cmake -S xinfer/ops/backends/ascend/src -B build/kernels-ascend \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DXINFER_ASCEND_SOC=ascend950dt9582 \
  -DXINFER_CANN_RUNTIME=/path/to/cann-9.1.1 \
  -DXINFER_BISHENG_HOME=/path/to/cann-9.3.0 \
  -DXINFER_TILELANG_PYTHON=/path/to/tilelang/bin/python
cmake --build build/kernels-ascend -j1
```

The library is written beside the Ascend Python backend. Build the optional host helpers with the same Python and Torch used by the service:

```bash
cmake -S xinfer/src -B build/host \
  -DPython_EXECUTABLE="$(command -v python)" -DXINFER_HOST_ASCEND=ON
cmake --build build/host -j1
```

The Wall-OSS recipe needs the Ascend-capable TileLang fork and bisheng at runtime, not only when building AOT kernels. Its original G2112/SAMU factories live in `src/gemm/g2112_tilelang.py`; the backend re-exports them unchanged. The tested fork reports `0.1.15+cpu.git1e8448fc`. The `kernels` extra installs TVM-FFI, not that compiler. Keep the runtime environment from CANN 9.1.1 and select the compiler with the exports above; do not replace it by sourcing the compiler toolkit's runtime environment.

## Run

The backend name is `ascend`; the PyTorch device is `npu:N`. With `ASCEND_RT_VISIBLE_DEVICES`, the index is relative to the visible set. Both models keep their existing checkpoint and wire-request formats.

```bash
source examples/ascend/wall_oss.env
xinfer-serve --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 --reference-ops \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --warmup flow.msgpack rtc.msgpack --port 33985
```

For Wall-X 2B, source [wallx2b.env](../examples/ascend/wallx2b.env), then use `--model wallx2b` with its checkpoint and requests. These recipes select the measured BF16 configuration, explicit native pins and image path; they do not change package defaults. Use a clean shell without other XInfer overrides. `Policy(device="npu:0")` uses the same backend and engine.

XInfer loads the package-local library. `XINFER_ASCEND_KERNELS_LIBRARY=/path/to/xinfer_kernels_ascend.so` selects another build of the same sources; `XINFER_ASCEND_KERNELS=0` disables native registrations. A missing default build leaves the fallback available; a missing explicit path or a broken selected library is an error.

## Images and graphs

The service processes images on the CPU by default. The top-level `image_preprocess_device="gpu"` option selects DVPP when available; `Policy` uses it by default on a capable platform. Unsupported JPEG profiles use host decoding with device resize. Record the image path when comparing outputs or performance.

The platform owns JPEG resources, stream ordering and post-decode image graphs. The selected `image_resize` operator supplies the resize contract. In-process preparation can prefetch views into its own engine; frontend processes remain device-free.

Stage graphs are enabled when the runtime supports them. Warmup covers representative flow/RTC shapes before serving; each shape runs eagerly before capture. `--enforce-eager` or `Policy(cudagraphs=False)` disables graphs. With `--reference-ops` and no native attention pins, use `--enforce-eager` on NPU because reference SDPA may not support capture. The recipes pin capture-safe attention; unsupported native layouts use the registered graph-safe Ascend torch composition.

Stream priorities use the common runtime interface. A runtime without a verifiable priority range uses priority 0 and rejects nonzero requests.

## Check

`Policy.describe()["ops"]`, the startup log and `--resolved-config` report the selected backend, AOT library fingerprint and contract. The in-tree library reports `ascend:kernels`. Benchmark environment metadata separately records TileLang availability/version and the JIT source SHA; these identities do not prove per-call kernel coverage or fingerprint every generated JIT binary. Unsupported calls may use a generic or reference route. Native execution errors propagate after a call is admitted.

Run the CPU contract suite, then the NPU runtime/image and model tests on an exclusively reserved card. Compare fixed prepared inputs and initial noise in eager and graph modes, including RTC, a new shape and repeated requests. Use the existing [precision gates](precision.md) and [benchmark scripts](benchmark.md); keep raw evidence outside Git.

```bash
source examples/ascend/wall_oss.env
python benchmarks/performance.py --name "Wall-OSS · BF16 · Ascend 950DT" --reference-ops \
  --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --latency-requests flow.msgpack rtc.msgpack --warmup flow.msgpack rtc.msgpack \
  --robots 1 2 3 4 5 6 --out artifacts/perf-wall-oss-ascend
```

For Wall-X 2B, substitute its model, checkpoint and representative request set. Keep the same inputs, precision, image path and graph settings when comparing builds. The [performance table](performance.md) records validated results with their measurement conditions.
