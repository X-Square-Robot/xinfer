# 性能

[English](performance.md) | 简体中文

表中是 Wall-OSS 在单个 GPU 上的单请求时延和服务容量，使用 `benchmarks/performance.py` 按[性能表测试口径](benchmark.zh-CN.md#性能表)测得。分阶段时延和精度结果见 [Wall-OSS 验证](models/wall_oss_validation.zh-CN.md)。

| # | 模型 | 精度、组合 | GPU | 单请求 P50 | 单卡机器人数 | 容量点服务端 P50 | 容量点服务端 P95 | 饱和吞吐 | 精度门禁 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wall-OSS | BF16，优化组合 | RTX 5090 | 49.2 ms | 4 | 117 ms | 125 ms | 26.3 req/s | quick100 B1 −6.58 %：PASS |

- **单请求：** 进程内（`Policy`）从传输格式的请求到响应，视图在设备上解码和 resize。
- **单卡机器人数：** 在 `xinfer serve` 默认设置下（视图在 CPU 上处理），8 轮（机器人起步相位随机）中没有错误、且各轮服务端 P50 的中位数低于 150 ms 的最大机器人数。P50 和 P95 两列给出该机器人数各轮的中位数。
- **饱和吞吐：** 测得的最高完成速率，单位为每秒请求数。

## 条件

所有行：RTX 5090（SM120），torch 2.10.0+cu130，flash-attn 2.8.3.post1，xcompute `7336b0d`（`xcompute.so` sha256 `d061051e…`），于 2026-11-09 测量，GPU 上没有其他任务。

| # | 模型选项 | 单请求的请求集 | 压测负载 | 预热 | 验证 |
|---|---|---|---|---|---|
| 1 | 默认 | x2robot dummy 客户端的一个普通请求和一个 RTC 请求（一路 1280×720 原始前视图、两路 640×480 JPEG 腕部视图） | 带 RTC 的合成桌面观测 | 同样的两个请求 | [Wall-OSS](models/wall_oss_validation.zh-CN.md) |

另见 [Wall-OSS 验证](models/wall_oss_validation.zh-CN.md)，其中记录了延迟、精度结果及测量条件。
