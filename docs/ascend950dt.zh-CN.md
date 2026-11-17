# 昇腾 950DT

[English](ascend950dt.md) | 简体中文

Wall-OSS 和 Wall-X 2B 在昇腾 950DT 上复用同一算子注册表和引擎。仓内 AOT 算子编译为 `xinfer_kernels_ascend.so`，无需外部 xcompute 包。仓内 JIT 算子在预热期间编译。

## 编译

在 aarch64 上使用 CANN 9.3 Toolkit、配套的 950DT 算子包、兼容的 PyTorch/TorchNPU、昇腾 TileLang、CMake 3.24+ 和 TVM-FFI。将 `CANN_ROOT` 指向包含 `set_env.sh` 的 Toolkit 目录。AOT 库使用仓内算子源码和模板头文件构建。

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

库生成在昇腾 Python 后端旁。可选的主机辅助模块使用与服务相同的 Python 和 Torch 编译：

```bash
cmake -S xinfer/src -B build/host \
  -DPython_EXECUTABLE="$(command -v python)" -DXINFER_HOST_ASCEND=ON
cmake --build build/host --parallel "${BUILD_JOBS:-4}"
```

可用 `BUILD_JOBS` 设置构建并发度。

## 运行

后端名为 `ascend`，PyTorch 设备名为 `npu:N`。设置 `ASCEND_RT_VISIBLE_DEVICES` 后，索引相对于可见设备集。两个模型继续使用原有 checkpoint 和线上请求格式。

```bash
source examples/ascend/wall_oss.env
xinfer-serve --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --warmup flow.msgpack rtc.msgpack --port 33985
```

Wall-X 2B 先加载 [wallx2b.env](../examples/ascend/wallx2b.env)，再使用 `--model wallx2b` 及其 checkpoint 和请求。这些配方选择实测的 BF16 配置、显式原生算子与图像路径，不改变包默认值。使用没有其他 XInfer 覆盖设置的干净 shell。`Policy(device="npu:0")` 使用同一后端和引擎。

XInfer 从昇腾后端目录加载 `xinfer_kernels_ascend.so`。

## 图像和图执行

服务默认在 CPU 上处理图像。顶层选项 `image_preprocess_device="gpu"` 在可用时选择 DVPP；`Policy` 在支持的平台上默认使用此路径。不支持的 JPEG 类型在主机解码后到设备缩放。比较输出和性能时需记录图像路径。

平台负责 JPEG 资源、流顺序和解码后的图像图。已选中的 `image_resize` 算子提供缩放契约。进程内预处理可以把视图预取到其所属引擎，前端进程不初始化设备。

运行时支持时默认启用阶段图。服务启动前使用代表性的 flow 和 RTC 形状预热。`--enforce-eager` 或 `Policy(cudagraphs=False)` 禁用图。配方固定了支持图捕获的注意力实现。

流优先级使用公共运行时接口。无法核实优先级范围的运行时使用优先级 0，并拒绝非零请求。

## 检查

用 `--resolved-config` 检查选中的算子和服务设置。仓内库显示为 `ascend:kernels`。

使用共享的[精度门禁](precision.zh-CN.md)和[基准脚本](benchmark.zh-CN.md)比较模型输出和时延。

```bash
source examples/ascend/wall_oss.env
benchmark_options=$(python -c 'import json, sys; opts = json.loads(sys.argv[1]); opts.pop("image_preprocess_device", None); print(json.dumps(opts))' "$ASCEND_MODEL_OPTIONS")
python benchmarks/performance.py --name "Wall-OSS · BF16 · Ascend 950DT" \
  --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$benchmark_options" --op-backends "$ASCEND_OP_BACKENDS" \
  --latency-requests flow.msgpack rtc.msgpack --warmup flow.msgpack rtc.msgpack \
  --robots 3 4 5 --out artifacts/perf-wall-oss-ascend
```

Wall-OSS 省略 `image_preprocess_device` 后，进程内基准在设备上处理图像，压测服务在 CPU 上处理图像。

测量 Wall-X 2B 时使用对应配方，两项测试均在模型选项中将 `image_preprocess_device` 设为 `"cpu"`。将测试进程绑定到设备所在的 CPU NUMA 节点，用 `benchmarks/request_latency.py` 测量进程内时延。按[性能表](performance.zh-CN.md#昇腾-950dt)中的人数逐档启动新服务，使用相同的两个预热请求，并指定 `--lanes vision=1,prefill=3,action=2 --max-batch action=4`。用 `benchmarks/robot_load.py` 按该档人数压测，指定 `--requests 80 --warmup-requests 20 --rounds 2`。
