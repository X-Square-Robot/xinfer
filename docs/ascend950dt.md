# Ascend 950DT

English | [简体中文](ascend950dt.zh-CN.md)

Wall-OSS and Wall-X 2B use the shared operator registry and engine on Ascend 950DT. The in-tree AOT kernels build into `xinfer_kernels_ascend.so` without an external xcompute package. In-tree JIT kernels compile during warmup.

## Build

Build on aarch64 with the CANN 9.3 Toolkit and matching 950DT operator package, compatible PyTorch/TorchNPU, Ascend TileLang, CMake 3.24+ and TVM-FFI. Set `CANN_ROOT` to the Toolkit directory containing `set_env.sh`. The AOT library uses the operator source and template headers in this repository.

```bash
CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann}
source "$CANN_ROOT/set_env.sh"
export BISHENG_HOME="$CANN_ROOT"
export TILELANG_CACHE_DIR="$PWD/artifacts/tilelang-cache"
python -m pip install -e ".[kernels,models,serve]"
cmake -S xinfer/ops/backends/ascend/src -B build/kernels-ascend \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DXINFER_ASCEND_SOC=ascend950dt \
  -DXINFER_CANN_RUNTIME="$CANN_ROOT" \
  -DXINFER_BISHENG_HOME="$CANN_ROOT" \
  -DXINFER_TILELANG_PYTHON=/path/to/tilelang/bin/python
cmake --build build/kernels-ascend --parallel "${BUILD_JOBS:-4}"
```

The library is written beside the Ascend Python backend. Build the optional host helpers with the same Python and Torch used by the service:

```bash
cmake -S xinfer/src -B build/host \
  -DPython_EXECUTABLE="$(command -v python)" -DXINFER_HOST_ASCEND=ON
cmake --build build/host --parallel "${BUILD_JOBS:-4}"
```

Set `BUILD_JOBS` to select the build parallelism.

## Run

The backend name is `ascend`; the PyTorch device is `npu:N`. With `ASCEND_RT_VISIBLE_DEVICES`, the index is relative to the visible set. Both models keep their existing checkpoint and wire-request formats.

```bash
source examples/ascend/wall_oss.env
xinfer-serve --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --warmup flow.msgpack rtc.msgpack --port 33985
```

For Wall-X 2B, source [wallx2b.env](../examples/ascend/wallx2b.env), then use `--model wallx2b` with its checkpoint and requests. These recipes select the measured BF16 configuration, explicit native pins and image path; they do not change package defaults. Use a clean shell without other XInfer overrides. `Policy(device="npu:0")` uses the same backend and engine.

XInfer loads `xinfer_kernels_ascend.so` from the Ascend backend directory.

## Images and graphs

The service processes images on the CPU by default. The top-level `image_preprocess_device="gpu"` option selects DVPP when available; `Policy` uses it by default on a capable platform. Unsupported JPEG profiles use host decoding with device resize. Record the image path when comparing outputs or performance.

The platform owns JPEG resources, stream ordering and post-decode image graphs. The selected `image_resize` operator supplies the resize contract. In-process preparation can prefetch views into its own engine; frontend processes remain device-free.

Stage graphs are enabled when the runtime supports them. Warmup covers representative flow and RTC shapes before serving. `--enforce-eager` or `Policy(cudagraphs=False)` disables graphs. The recipes pin capture-safe attention.

Stream priorities use the common runtime interface. A runtime without a verifiable priority range uses priority 0 and rejects nonzero requests.

## Check

Use `--resolved-config` to check the selected operators and service settings. The in-tree library reports `ascend:kernels`.

Use the shared [precision gates](precision.md) and [benchmark scripts](benchmark.md) to compare model outputs and latency.

```bash
source examples/ascend/wall_oss.env
benchmark_options=$(python -c 'import json, sys; opts = json.loads(sys.argv[1]); opts.pop("image_preprocess_device", None); print(json.dumps(opts))' "$ASCEND_MODEL_OPTIONS")
python benchmarks/performance.py --name "Wall-OSS · BF16 · Ascend 950DT" \
  --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$benchmark_options" --op-backends "$ASCEND_OP_BACKENDS" \
  --latency-requests flow.msgpack rtc.msgpack --warmup flow.msgpack rtc.msgpack \
  --robots 3 4 5 --out artifacts/perf-wall-oss-ascend
```

For Wall-OSS, omitting `image_preprocess_device` uses device image processing for the in-process benchmark and CPU image processing for the load service.

For Wall-X 2B, use its recipe and set `image_preprocess_device` to `"cpu"` in the model options for both tests. Bind the benchmark processes to the device's local CPU NUMA node. Measure in-process latency with `benchmarks/request_latency.py`. For each robot count in the [performance table](performance.md#ascend-950dt), start a fresh service with the same two warmup requests and `--lanes vision=1,prefill=3,action=2 --max-batch action=4`. Run `benchmarks/robot_load.py` with that count and `--requests 80 --warmup-requests 20 --rounds 2`.
