# 昇腾 950DT

[English](ascend950dt.md) | 简体中文

WALL-OSS-0.5 和 WALL-OSS-1.0-Nano 在昇腾 950DT 上复用共享的算子注册表和引擎。仓内 AOT 算子构建为 `xinfer_kernels_ascend.so`。WALL-OSS-0.5 还会在预热期间编译仓内 JIT 算子。

## 运行环境

安装相互匹配的 PyTorch 和 TorchNPU，并在导入 TorchNPU 前加载 CANN 环境。使用 `--device npu:0` 或
`Policy(device="npu:0")` 选择设备。设置 `ASCEND_RT_VISIBLE_DEVICES` 后，编号相对于可见卡集合。
平台后端名称是 `ascend`，torch 设备串是 `npu:N`。

WALL-OSS-0.5 沿用原有 checkpoint 和请求格式。服务默认在 CPU 上处理图像。PyACL DVPP V2 可用时，用顶层模型
选项 `--model-options '{"image_preprocess_device":"gpu"}'` 或 `Policy(image_preprocess_device="gpu")`
选择设备图像处理。平台支持时，`Policy` 默认使用设备路径。不要把该选项放入 `processing`。
不支持的 JPEG 格式（包括渐进式 JPEG）会记录告警，再由主机解码、设备缩放。精度和性能结果应记录图像路径。

JPEG 解码器位于各平台模块。`platforms.devices.jpeg_decoder_for(device)` 按需选择解码器，接口为
`device`、`__call__(jpegs)` 和 `close()`。平台负责厂商资源与流依赖；引擎保留输入映射、上传原始帧，
并对原始帧和解码后的图像应用相同的 resize 步骤。

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

`xinfer_kernels_ascend.so` 生成在 Python 后端目录中。加载器在算子选择元数据中记录动态库的 SHA-256 和支持的 SoC。

| 环境变量 | 含义 |
|---|---|
| `XINFER_ASCEND_KERNELS_LIBRARY` | 可选的 `xinfer_kernels_ascend.so` 路径；未设置时，由包定位动态库 |
| `XINFER_ASCEND_KERNELS` | `0` 关闭仓内原生实现 |

默认位置的动态库不存在时，原生算子不注册，使用下一个可用实现。显式指定的动态库缺失或损坏会报错。不支持的调用布局在原生下发前使用算子的 torch 回退；已准入的原生调用发生错误时直接向调用方传播，不会用 torch 重试。

## 运行

```bash
source examples/ascend/wall_oss_0_5.env
xinfer-serve --model wall-oss-0.5 --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --warmup flow.msgpack rtc.msgpack --port 33985
```

对于 WALL-OSS-1.0-Nano，加载 [wall_oss_1_0_nano.env](../examples/ascend/wall_oss_1_0_nano.env)，使用 `--model wall-oss-1.0-nano` 及其 checkpoint 和请求。两个配方均使用 BF16 和 CPU 图像处理。WALL-OSS-1.0-Nano 使用共享的 RTC 夹爪限值 5.0。阶段图在预热期间捕获。用 `--resolved-config FILE` 检查选中的算子和服务设置。仓内算子显示为 `ascend:kernels`；选择结果不代表每种调用形状都使用原生算子。

TorchNPU 官方实现仍需显式选择。例如，`--op-backends gemm_epilogue=ascend:npu_linear` 为该算子选择官方线性实现。

## 图执行

安装的 TorchNPU 提供图 API 时，默认开启图捕获，与 CUDA 相同。
请求首次遇到新形状时按需捕获；也可通过代表性的 `--warmup` 请求，在服务前捕获可达的形状分桶、通道和批量组合。
每次捕获前，引擎先在捕获流上对该形状运行一次 eager，再捕获固定缓冲区。预热后，新形状遵循
`CUDAGraphConfig.on_miss_after_warmup`：`"capture"`（默认）捕获一次，后续同形状请求回放；`"eager"`
不捕获，直接执行；`"error"` 拒绝该请求。`--enforce-eager` 或 `Policy(cudagraphs=False)` 关闭图执行。

没有原生实现支持某次调用时，`ascend:torch` 注意力在 eager 和捕获中使用相同的 FP32 matmul、softmax 和 matmul
组合，避开可能导致捕获无法结束的 SDPA 旁路流。CPU/CUDA 和显式参考算子表保留原始 SDPA 参考实现。
不指定原生算子绑定时，`--reference-ops` 选择原始参考实现；在 NPU 上应配合
`--enforce-eager`，或使用 `Policy(reference_ops=True, cudagraphs=False)`，因为参考注意力可能不支持捕获。

## 流优先级

引擎在 CUDA 和 NPU 上均通过同一 `Stream(device=..., priority=...)` 接口传入 `priority`，
依据当前运行时的流优先级能力及其报告的数值范围处理设置。范围已知时，将最终阶段偏好的优先级 −1
限制在该范围内：范围为 0–7 的运行时保持最终阶段为 0。范围未知时，所有默认值保持为 0。显式请求不支持
或越界的值会报错；声明支持优先级的运行时修改了请求的优先级，或无法读回实际优先级，也会导致通道创建失败。

TorchNPU 2.10.0.post4 和 2.10.0.post7 均未实现优先级范围查询及读回，且其托管流池创建的原生流优先级均为 0。
因此，这些运行时不声明流优先级能力，并拒绝非零请求。CANN 支持原生流优先级，并不能证明 TorchNPU
遵守其构造参数；参见 [post4 流绑定](https://github.com/Ascend/pytorch/blob/5dd8ef3f9b375b5ae4a83538d5785754148c3302/torch_npu/csrc/npu/Stream.cpp#L90-L100)、
[post7 流绑定](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/csrc/npu/Stream.cpp#L103-L113)
和 [post7 流池创建](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/csrc/core/npu/NPUStream.cpp#L254-L268)。
post7 的 `ExternalStream` 包装不能替代通道流：其文档明确限制事件记录/等待、分配器流跟踪和图捕获；
参见[上游契约](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/npu/streams.py#L112-L177)。

## 基准

使用共享的[性能测试口径](benchmark.zh-CN.md#性能表)，按同一配方测量时延和负载。

```bash
source examples/ascend/wall_oss_0_5.env
python benchmarks/performance.py --name "WALL-OSS-0.5 · BF16 · Ascend 950DT" \
  --model wall-oss-0.5 --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --latency-requests flow.msgpack rtc.msgpack --warmup flow.msgpack rtc.msgpack \
  --robots 3 4 5 --out artifacts/perf-wall-oss-0-5-ascend
```

测量 WALL-OSS-1.0-Nano 时，用对应配方和抓包请求集测[单请求时延](benchmark.zh-CN.md#单请求)，将测试进程绑定到设备所在的 CPU NUMA 节点。

## 验证

昇腾 host 测试不需要 NPU、CANN 或原生动态库：

```bash
XINFER_ASCEND_KERNELS=0 pytest -q tests/ascend
```

这些测试检查加载器、布局、workspace、生命周期和配方契约，不执行昇腾算子，也不能证明设备数值正确。在独占预留的 950DT 上，构建原生动态库并运行两个配方。使用相同的准备后输入和显式噪声比较 eager 与图执行，再通过 WebSocket 服务测试真实请求。检查无预热的冷请求和预热后的新形状能按需捕获，之后同形状请求能够回放。

运行[精度门禁](precision.zh-CN.md)和[基准测试](benchmark.zh-CN.md)，记录源码和动态库
身份、计算组合、算子选择、图像路径、checkpoint 身份与图计数。加速器 trace 和实验保留在推理包之外。
测量请求延迟和机器人容量时关闭 profiler。通过 `benchmarks/performance.py --device npu:0` 选择 NPU；
图像设置放入原有的 `--model-options`。单独比较算子后端或 eager 执行时，使用 `Policy` 或
`xinfer-serve` 的现有选项。
