<p align="center">
  <img src="docs/assets/xinfer-logo.webp" alt="XInfer" width="560">
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#模型与硬件">模型与硬件</a> ·
  <a href="#基准测试">基准测试</a> ·
  <a href="#文档">文档</a>
</p>

**XInfer 为机器人运行视觉-语言-动作（VLA）模型。** 输入相机图像、机器人状态和指令，即可得到一段动作序列。你可以在 Python 中为单台机器人调用策略，也可以用一张加速卡通过 WebSocket 为多台机器人提供服务。两种方式共用同一个推理引擎。

## 快速开始

准备 Python 3.10+ 和适配加速卡的 PyTorch，然后获取源码并安装：

```bash
git clone https://github.com/X-Square-Robot/xinfer.git
cd xinfer
python -m pip install -e ".[models,serve]"
python -c "from xinfer.models import available_models; print(available_models())"
```

全新安装会列出 `('wall_oss', 'wallx2b')`。这一步不需要模型权重。

### 启动模型服务

以下示例在 CUDA 上运行 Wall-X 2B。准备一份[兼容的 checkpoint](docs/models/wallx2b.zh-CN.md#checkpoint-要求) 和一条名为 `flow.msgpack` 的观测记录。记录按[请求格式](docs/models/wallx2b.zh-CN.md#请求与响应)包含三个相机视角、双臂位姿和指令。

```bash
xinfer-serve --model wallx2b --checkpoint /path/to/wallx2b-checkpoint \
  --device cuda --warmup flow.msgpack --port 33985
```

预热完成后，在另一个终端检查服务是否就绪：

```bash
curl -fsS http://127.0.0.1:33985/healthz
```

服务会返回 `ok`。机器人可通过 WebSocket + msgpack 向 `ws://127.0.0.1:33985` 发送观测。使用实时分块（RTC）时，在 `--warmup` 中同时传入一条普通请求和一条有代表性的 RTC 请求。网络协议和部署选项见[服务指南](docs/serving.zh-CN.md)。

### 在 Python 中运行

也可以用同一份 checkpoint 和观测，在进程内查看一次预测。下面的代码在 CPU 上解码相机图像、在 CUDA 上运行模型，与上面服务命令的默认行为一致。

```python
from pathlib import Path

from xinfer import Policy
from xinfer.frontend.protocol import unpackb

observation = unpackb(Path("flow.msgpack").read_bytes())
policy = Policy("wallx2b", "/path/to/wallx2b-checkpoint",
                device="cuda", image_preprocess_device="cpu")
try:
    policy.warmup([observation])
    response = policy.infer(observation)
    print(response["action_pred"].shape)
finally:
    policy.engine.shutdown()
```

对于 Wall-X 2B 的桌面双臂格式，输出形状为 `(16, 14)`，即 16 个动作步，每步每条手臂各有 7 个值。

## 模型与硬件

两个模型都支持 BF16 推理和 RTC。权重需单独准备。

| 模型 | 模型 ID | Checkpoint 和输入 |
|---|---|---|
| Wall-X 2B | `wallx2b` | [Checkpoint 文件与桌面双臂观测格式](docs/models/wallx2b.zh-CN.md) |
| Wall-OSS | `wall_oss` | [所需的内部 checkpoint 布局、tokenizer 和归一化文件](docs/models/wall_oss.zh-CN.md#环境要求)。公开的 Wall-OSS-0.5 和 WALL-OSS-FLOW 权重布局不同，目前无法直接加载。 |

| 设备 | 运行时 | 配置 |
|---|---|---|
| NVIDIA GPU（`cuda`） | PyTorch + CUDA | [CUDA 算子与参考实现](docs/kernels.zh-CN.md)。优化的 BF16 算子需要与设备匹配的 **xcompute** 构建。 |
| 昇腾 950DT（`npu`） | PyTorch + TorchNPU + CANN | [昇腾部署指南](docs/ascend950dt.zh-CN.md)，包括 NPU 图、xcompute 和 DVPP 图像预处理。 |
| CPU（`cpu`） | PyTorch | 参考实现与测试。 |

## 基准测试

下表给出预热后单条 BF16 请求的中位时延，涵盖从输入处理到返回动作的全过程。

| 模型 | NVIDIA RTX 5090 | 昇腾 950DT |
|---|---:|---:|
| Wall-X 2B | 22.3 ms | 79.7 ms |
| Wall-OSS | 49.2 ms | 80.8 ms |

数据对应特定构建和输入。Wall-X 2B 在 RTX 5090 上的测量使用了上游 xcompute 中可省去一次拷贝的权重布局。[完整结果与测量条件](docs/performance.zh-CN.md)还列出了每张卡能服务的机器人数和并发负载下的时延。复现步骤见[基准测试指南](docs/benchmark.zh-CN.md)。

## 工作原理

<p align="center">
  <img src="docs/assets/xinfer-architecture.webp" alt="机器人经前端提交观测，引擎依次运行视觉、预填充和动作阶段。" width="100%">
</p>

前端处理观测，并将模型输出转换为机器人动作。引擎按阶段将请求排队，把兼容的任务组批，在支持的平台上回放已捕获的图。每张加速卡运行一个引擎并持有一份模型权重。调度方式见[引擎指南](docs/engine.zh-CN.md)，部署参数见[配置指南](docs/configuration.zh-CN.md)。

## 文档

| 任务 | 指南 |
|---|---|
| 部署和监控服务 | [服务](docs/serving.zh-CN.md) · [指标与仪表盘](docs/metrics.zh-CN.md) |
| 检查精度与数值格式 | [精度](docs/precision.zh-CN.md) · [量化](docs/quantization.zh-CN.md) · [Wall-X 2B 验证](docs/models/wallx2b_validation.zh-CN.md) · [Wall-OSS 验证](docs/models/wall_oss_validation.zh-CN.md) |
| 分析时延 | [性能剖析](docs/profiling.zh-CN.md) · [基准测试方法](docs/benchmark.zh-CN.md) |
| 接入模型或贡献算子 | [模型接口](xinfer/models/interface.py) · [架构设计](docs/design.zh-CN.md) · [参与贡献](CONTRIBUTING.md) |

此发行版尚未附带许可证。
