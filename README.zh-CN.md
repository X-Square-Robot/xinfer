<p align="center">
  <img src="docs/assets/xinfer-logo.webp" alt="XInfer — 面向机器人的推理" width="560">
</p>

<h3 align="center">快速、稳定、准确的 VLA 推理，从单台机器人到整个机队。</h3>

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#模型与硬件">模型</a> ·
  <a href="docs/benchmark.zh-CN.md">基准测试</a> ·
  <a href="#文档">文档</a> ·
  <a href="CONTRIBUTING.md">参与贡献</a>
</p>

**XInfer 是面向视觉-语言-动作（VLA）模型的推理与服务框架。** 我们的目标是为机器人提供快速、稳定、准确的模型推理服务，覆盖从机载推理到机队共享服务的各种场景。

## 为什么选择 XInfer

- **快速。** 单请求时延低，每个加速器能服务更多机器人。异步阶段发射、设备事件以及（在支持的平台上）图回放降低了主机开销。阶段队列和零等待组批把排队中的兼容请求合并执行，共享权重，并让不同阶段的工作相互重叠。
- **稳定。** 时延可预测，服务在机队负载下持续运行：服务前先预热，有准入限制，并对逐请求的故障进行隔离。
- **准确。** 输出忠实于原始模型。默认情况下，组批不会改变单个请求的结果，量化预设均通过各模型的[精度门禁](docs/precision.zh-CN.md)。
- **可扩展。** 可复用的层和阶段接口简化了模型接入。平台适配器和算子适配器把模型代码接到新芯片和优化过的 kernel 上。

## 快速开始

使用 Python 3.10+ 和与加速器匹配的 PyTorch 版本。下面的示例使用 NVIDIA CUDA。在当前检出的仓库中安装：

```bash
python -m pip install -e ".[models,serve]"
```

请准备兼容的 checkpoint 及其 tokenizer、配置和归一化文件，并将示例中的 MODEL_ID 替换为已注册的模型 ID。

### 启动模型服务

```bash
xinfer-serve \
  --model MODEL_ID \
  --checkpoint /path/to/checkpoint \
  --device cuda \
  --warmup observation.msgpack \
  --port 33985
```

`observation.msgpack` 是一个符合[请求格式](docs/serving.zh-CN.md#协议)的代表性请求。预热在服务开始接受连接之前运行。如有 RTC 流量，请在 `--warmup` 中同时提供一个普通请求和一个 RTC 请求。

通过 WebSocket + msgpack 连接 `ws://127.0.0.1:33985`。协议、健康检查端点和部署选项见[服务指南](docs/serving.zh-CN.md)。

### 在 Python 中运行

```python
from pathlib import Path

from xinfer import Policy
from xinfer.frontend.protocol import unpackb

observation = unpackb(Path("observation.msgpack").read_bytes())
policy = Policy("MODEL_ID", "/path/to/checkpoint", device="cuda")

try:
    policy.warmup([observation])
    response = policy.infer(observation)
finally:
    policy.engine.shutdown()
```

响应中包含模型输出的动作块（action chunk）。相机名称、状态字段和动作表示见各模型的指南。

## 模型与硬件

各配置的单请求时延和单卡机器人数：[性能](docs/performance.zh-CN.md)。

| 硬件 | 设备 | 运行时 |
|---|---|---|
| NVIDIA GPU · RTX 5090 (SM120) | `cuda` | PyTorch + CUDA |
| Huawei Ascend 950DT NPU | `npu` | PyTorch + `torch_npu` + CANN |
| CPU | `cpu` | 参考实现与测试 |

每个加速器运行各自的引擎（Engine）和权重。

优化的 CUDA 执行路径使用 Triton 和可选的 FlashAttention 后端。MXFP8 还需要兼容的 **xcompute** 构建；见[量化与 kernel 配置](docs/quantization.zh-CN.md#后端)。Ascend 通过 `torch_npu` 走 BF16 参考算子路径。

## 工作原理

<p align="center">
  <img src="docs/assets/xinfer-architecture.webp" alt="机器人客户端经负责输入处理和输出处理的前端（Frontend）连接到 XInfer 引擎（Engine），引擎运行 Vision、Prefill 和 Action。平台适配器分别连到以品牌标志标识的 NVIDIA GPU 和 Ascend 950DT NPU，以及 CPU 参考执行。虚线的 More chips 框表示更多硬件后端。" width="100%">
</p>

前端（Frontend）负责输入处理和输出处理。模型定义各阶段的计算和张量交接；引擎管理准入、队列、组批和资源生命周期。平台适配器提供设备执行，算子后端提供优化的 kernel。Python API 与网络服务共用同一个引擎和同一套配置规则。执行模型见[引擎指南](docs/engine.zh-CN.md)，默认值与覆盖方式见[配置](docs/configuration.zh-CN.md)。

## 基准测试

评估单请求时延，以及在给定响应时延目标内能服务的机器人数：

| 工具 | 测量内容 |
|---|---|
| [单请求时延](benchmarks/request_latency.py) | 进程内的输入处理、引擎执行、输出处理耗时及总耗时 |
| [机器人负载](benchmarks/robot_load.py) | 多台机器人并发流量下的客户端与服务端时延、吞吐、错误及动作队列下溢（underflow） |

[基准测试指南](docs/benchmark.zh-CN.md)给出了命令和测量口径。

## 文档

| 主题 | 从这里开始 |
|---|---|
| 部署服务 | [服务](docs/serving.zh-CN.md) |
| 了解调度与 CUDA graph | [引擎](docs/engine.zh-CN.md) |
| 配置部署 | [配置](docs/configuration.zh-CN.md) |
| 选择数值格式与 kernel | [量化](docs/quantization.zh-CN.md) |
| 验证模型输出 | [精度](docs/precision.zh-CN.md) |
| 分析算子性能 | [性能剖析](docs/profiling.zh-CN.md) |
| 扩展框架 | [模型接口](xinfer/models/interface.py) · [架构与设计](docs/design.zh-CN.md) |

## 参与贡献

代码规范、测试和性能证据要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

XInfer 目前处于非公开开发阶段，许可与公开发布尚待确定。
