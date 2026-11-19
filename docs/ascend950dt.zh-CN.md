# 昇腾 950DT

[English](ascend950dt.md) | 简体中文

Wall-OSS 和 Wall-X 2B 在昇腾 950DT 上复用共享的算子注册表和引擎。仓内 AOT 算子构建为 `xinfer_kernels_ascend.so`。Wall-OSS 还会在预热期间编译仓内 JIT 算子。

## 编译

在 aarch64 上使用 CANN 9.3 Toolkit 和配套的昇腾 950DT 算子包，以及兼容的 PyTorch、TorchNPU、昇腾 TileLang、CMake 3.24 或更高版本和 TVM-FFI。将 `CANN_ROOT` 指向 Toolkit 安装目录。

```bash
CANN_ROOT=/path/to/cann-9.3
source "$CANN_ROOT/set_env.sh"
export BISHENG_HOME="$CANN_ROOT"
python -m pip install -e ".[kernels,models,serve]"
cmake -S xinfer/ops/backends/ascend/src -B build/kernels-ascend \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DXINFER_ASCEND_SOC=ascend950dt \
  -DXINFER_CANN_RUNTIME="$CANN_ROOT" \
  -DXINFER_BISHENG_HOME="$CANN_ROOT"
cmake --build build/kernels-ascend --parallel "${BUILD_JOBS:-4}"
```

`xinfer_kernels_ascend.so` 生成在 Python 后端目录中。

## 运行

```bash
source examples/ascend/wall_oss.env
xinfer-serve --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --warmup flow.msgpack rtc.msgpack --port 33985
```

对于 Wall-X 2B，加载 [wallx2b.env](../examples/ascend/wallx2b.env)，使用 `--model wallx2b` 及其 checkpoint 和请求。两个配方均使用 BF16 和 CPU 图像处理。Wall-X 2B 使用共享的 RTC 夹爪限值 5.0。阶段图在预热期间捕获。用 `--resolved-config FILE` 检查选中的算子和服务设置。仓内算子显示为 `ascend:kernels`。

## 基准

使用共享的[性能测试口径](benchmark.zh-CN.md#性能表)，按同一配方测量时延和负载。

```bash
source examples/ascend/wall_oss.env
python benchmarks/performance.py --name "Wall-OSS · BF16 · Ascend 950DT" \
  --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --latency-requests flow.msgpack rtc.msgpack --warmup flow.msgpack rtc.msgpack \
  --robots 3 4 5 --out artifacts/perf-wall-oss-ascend
```

测量 Wall-X 2B 时，用对应配方和抓包请求集测[单请求时延](benchmark.zh-CN.md#单请求)，将测试进程绑定到设备所在的 CPU NUMA 节点。
