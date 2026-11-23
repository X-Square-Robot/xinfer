<p align="center">
  <img src="docs/assets/xinfer-logo.webp" alt="XInfer — 面向机器人的推理" width="560">
</p>

<h3 align="center">快速、稳定、准确的 VLA 推理，从单台机器人到整个机队。</h3>

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<p align="center">
  <a href="#特性一览">特性</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#模型与硬件">模型</a> ·
  <a href="docs/benchmark.zh-CN.md">基准测试</a> ·
  <a href="#文档">文档</a> ·
  <a href="https://arxiv.org/abs/2611.14969">论文</a> ·
  <a href="CONTRIBUTING.md">参与贡献</a>
</p>

**XInfer** 是面向视觉-语言-动作（VLA）模型的推理与服务框架。

## 特性一览

- **实时分块（RTC）**　机器人执行当前动作段时，即可根据新观测预测下一段。
- **单机器人极低时延**　WALL-OSS-1.0-Nano BF16 在 RTX 5090 上的单请求 P50 为 20.1 ms。
- **单卡服务多机器人**　通过细粒度的分阶段调度，不同请求的推理阶段可在多个 CUDA stream 上异步并发执行。
- **低精度量化**　支持按层组对线性层使用 MXFP8 或 NVFP4。
- **多硬件后端**　NVIDIA RTX 5090、Jetson Thor、昇腾 950DT、Huixi Rhino RPU（R1，适配中）。
- **高性能算子库**　使用优化融合算子，可选 FlashAttention，也可接入其他算子库。

## 快速开始

以下 CUDA 示例需要 Python 3.10+、支持 CUDA 的 PyTorch 和[兼容的 BF16 WALL-OSS-1.0-Nano checkpoint](docs/models/wall_oss_1_0_nano.zh-CN.md#checkpoint-要求)。在仓库根目录安装 XInfer。

```bash
python -m pip install -e ".[models,serve]"
```

要启用优化路径，请[编译 CUDA 算子](docs/kernels.zh-CN.md)。XInfer 也可以使用 PyTorch 参考算子运行。

### 启动模型服务

```bash
xinfer-serve \
  --model wall-oss-1.0-nano \
  --checkpoint /path/to/wall-oss-1.0-nano-checkpoint \
  --device cuda \
  --warmup observation.msgpack \
  --port 33985
```

将一条[模型请求](docs/models/wall_oss_1_0_nano.zh-CN.md#请求与响应)保存为 `observation.msgpack`。使用 RTC 时，请在 `--warmup` 中同时提供普通请求和 RTC 请求。

通过 WebSocket + msgpack 连接 `ws://127.0.0.1:33985`。部署选项见[服务指南](docs/serving.zh-CN.md)。

### 在 Python 中运行

```python
from pathlib import Path

from xinfer import Policy
from xinfer.frontend.protocol import unpackb

observation = unpackb(Path("observation.msgpack").read_bytes())
policy = Policy("wall-oss-1.0-nano", "/path/to/wall-oss-1.0-nano-checkpoint", device="cuda")

try:
    policy.warmup([observation])
    response = policy.infer(observation)
finally:
    policy.engine.shutdown()
```

## 模型与硬件

| 模型 | 模型 ID | 精度 | 实时分块（real-time chunking） | 指南 |
|---|---|---|---|---|
| WALL-OSS-1.0-Nano | `wall-oss-1.0-nano` | BF16；视觉和预填充可选 MXFP8 或 NVFP4，动作阶段保留 BF16 | 支持 | [安装与输入](docs/models/wall_oss_1_0_nano.zh-CN.md) |
| WALL-OSS-0.5 | `wall-oss-0.5` | BF16 | 支持 | [安装与输入](docs/models/wall_oss_0_5.zh-CN.md) |

WALL-OSS-0.5 需要使用[内部 checkpoint 布局](docs/models/wall_oss_0_5.zh-CN.md#环境要求)。

| 硬件 | 设备 | 运行时 |
|---|---|---|
| NVIDIA GPU · RTX 5090 (SM120) | `cuda` | PyTorch + CUDA |
| NVIDIA Jetson Thor (SM110) | `cuda` | PyTorch + CUDA |
| Huawei Ascend 950DT NPU | `npu` | PyTorch + `torch_npu` + CANN · [配置](docs/ascend950dt.zh-CN.md) |
| Rhino RPU（R1，实验性） | `rpu` | Rhino Launch SDK AOT · [配置](docs/models/wall_oss_1_0_nano_rpu.zh-CN.md) |
| CPU | `cpu` | 参考实现与测试 |

## 性能

以下为 `benchmarks/performance.py` 采用生产环境的机器人请求格式（每条请求含三路相机图像）测得的单加速器结果。单请求 P50 和并发指标来自各自的测试。测量条件和复现进度见[性能指南](docs/performance.zh-CN.md)。

<table>
  <thead>
    <tr>
      <th scope="col" rowspan="2">模型</th>
      <th scope="col" rowspan="2">精度与阶段</th>
      <th scope="col" rowspan="2">加速器</th>
      <th scope="col">单请求时延</th>
      <th scope="col" colspan="3">闭环并发服务</th>
    </tr>
    <tr>
      <th scope="col">P50</th>
      <th scope="col">容量（台）</th>
      <th scope="col">容量点服务端 P50 / P95</th>
      <th scope="col">峰值吞吐</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>BF16</td>
      <td>RTX 5090</td>
      <td>20.1 ms</td>
      <td>14</td>
      <td>85 / 107 ms</td>
      <td>81.3 req/s</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>视觉和预填充 MXFP8</td>
      <td>RTX 5090</td>
      <td>15.1 ms</td>
      <td>25</td>
      <td>67 / 88 ms</td>
      <td>141.0 req/s</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>BF16</td>
      <td><a href="https://github.com/X-Square-Robot/xinfer/pull/33">昇腾 950DT</a></td>
      <td>20.6 ms</td>
      <td>15</td>
      <td>129 / 163 ms</td>
      <td>83.9 req/s</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>BF16</td>
      <td>Jetson AGX Thor</td>
      <td>TBD</td>
      <td colspan="3">TBD</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>视觉和预填充 NVFP4</td>
      <td>RTX 5090</td>
      <td>13.8 ms</td>
      <td>28</td>
      <td>72 / 116 ms</td>
      <td>154.6 req/s</td>
    </tr>
  </tbody>
</table>

服务端 P50 / P95 在表中所列的机器人数下测得。峰值吞吐取各并发档位和轮次中的最高稳态速率。

## WALL-OSS-1.0-Nano 生产应用：包裹分拣

| 硬件 | 包裹处理量（件/小时） |
|---|---:|
| NVIDIA RTX 5090 | 2,231 |
| Huixi | 2,190 |

## 工作原理

<p align="center">
  <img src="docs/assets/xinfer-architecture.svg" alt="模型阶段调用加载时选定的算子实现。算子注册表支持优化融合算子、可选的 CUDA FlashAttention、PyTorch 参考算子和第三方插件。引擎支持 RTX 5090、Jetson AGX Thor、昇腾 950DT 和 CPU。Huixi R1 走独立的实验路径。" width="100%">
</p>

引擎负责调度模型各阶段。算子注册表在加载时为目标设备选择算子。虚线表示 Huixi R1 的实验路径。调度细节见[引擎指南](docs/engine.zh-CN.md)。

## 基准测试

测试命令见[基准测试指南](docs/benchmark.zh-CN.md)，各阶段耗时和精度见 [Nano 验证结果](docs/models/wall_oss_1_0_nano_validation.zh-CN.md)。

## 文档

| 任务 | 指南 |
|---|---|
| 部署 | [服务](docs/serving.zh-CN.md) · [配置](docs/configuration.zh-CN.md) · [监控指标](docs/metrics.zh-CN.md) |
| 优化 | [算子](docs/kernels.zh-CN.md) · [量化](docs/quantization.zh-CN.md) · [精度](docs/precision.zh-CN.md) · [性能剖析](docs/profiling.zh-CN.md) |
| 扩展 | [架构与设计](docs/design.zh-CN.md) · [模型接口](xinfer/models/interface.py) |

## 引用

```bibtex
@misc{zhang2026scalingvlaservingrobotic,
      title={Scaling VLA Serving Under Robotic Execution Constraints},
      author={Yue Zhang and Zicong Ye and Lufang Chen and Dance Yang and Mingyang Li and Zhenwei Duan and Michael Zhao and Ziye Deng and Starrick Liu and Ruizhi Chengze and YongHua Chen and SongAo Han and Yu Sun and Ran Yu and Xiaofan Li and Ruyi Gan and Hao Wang},
      year={2026},
      eprint={2611.14969},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.14969},
}
```

## 参与贡献

参见[贡献指南](CONTRIBUTING.md)。

## 许可证

XInfer 源代码采用 [Apache License 2.0](LICENSE)。
