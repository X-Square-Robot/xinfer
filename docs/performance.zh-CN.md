# 性能

[English](performance.md) | 简体中文

下表列出 `benchmarks/performance.py` 采用生产环境的机器人请求格式（每条请求含三路相机图像）测得的 WALL-OSS-1.0-Nano 单加速器结果。单请求 P50 和并发指标来自各自的测试。

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

容量和服务端 P50 / P95 对应同一个并发档位。峰值吞吐取所有实测并发档位和轮次中的最高稳态请求完成速率。

## 测试口径

- **单请求 P50** 是进程内从传输格式请求到响应的时延中位数，包括图像解码与缩放、模型推理和输出处理。
- **单卡机器人数**、**容量点服务端 P50 / P95** 和**峰值吞吐**来自闭环并发压测。容量判定和汇总方式见[基准测试口径](benchmark.zh-CN.md#性能表)。

## RTX 5090

2026 年 11 月 23 日，NVFP4 在公开 SM120 构建 `192b9fd`（内核库 SHA-256 前缀 `c0b3afbebebb`）上使用 GPU 图像预处理和 PyTorch 2.10.0+cu130。单请求 P50 来自 50 次计时。闭环压测从 12 台机器人开始，以 2 台为步长，每个档位运行 8 轮。服务端 P50 中位数从 28 台时的 72.0 ms 升至 30 台时的 186.7 ms，超过容量阈值。所有轮次均无错误或缺失请求。

BF16 和 MXFP8 的单请求 P50 均测于 2026 年 11 月 21 日。测试独占一张 RTX 5090，使用 SM120 算子版本 `4a62de1`、PyTorch 2.10.0+cu130 和 FlashAttention 2.8.3.post1。相机图像在 GPU 上解码并缩放。单请求输入集包含 216 条录制请求，每条请求有三路 JPEG 图像。

BF16 并发数据来自 11 月 13 日的测试，使用 SM120 算子版本 `7336b0d`。MXFP8 并发数据来自 11 月 17 日的测试，使用版本 `f4cdb84`。两组测试均使用带实时分块的合成机器人观测。[MXFP8 预设](models/wall_oss_1_0_nano_design.zh-CN.md#mxfp8)量化视觉和预填充阶段的部分线性层，动作阶段保留 BF16。

## 昇腾 950DT

这组 BF16 测量的公开构建产物和复现步骤待补充（**TBD**）。设备与算子配置见[昇腾指南](ascend950dt.zh-CN.md)。仓内昇腾算子见 [PR #33](https://github.com/X-Square-Robot/xinfer/pull/33)。
