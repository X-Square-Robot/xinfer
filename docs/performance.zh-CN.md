# 性能

[English](performance.md) | 简体中文

2026 年 11 月 13 日，使用 `benchmarks/performance.py`，按[同一测试口径](benchmark.zh-CN.md#性能表)在单张 RTX 5090 上测量了 Wall-OSS 和 Wall-X 2B、在单台 Jetson AGX Thor 上测量了 Wall-OSS 的 BF16 优化组合，并在单张 RTX 5090 上测量了视觉和预填充使用 NVFP4 的 Wall-X 2B。

| # | 模型 | 精度、组合 | GPU | 单请求 P50 | 单卡机器人数 | 容量点服务端 P50 | 容量点服务端 P95 | 饱和吞吐 | 数值一致性 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wall-OSS | BF16，优化组合 | RTX 5090 | 49.0 ms | 4 | 128 ms | 142 ms | 25.6 req/s | 在参考组合的逐阶段容差内（`tests/models/test_wall_oss_optimized.py`） |
| 2 | Wall-X 2B | BF16，优化组合 | RTX 5090 | 22.3 ms | 14 | 85 ms | 107 ms | 81.3 req/s | 动作与 checkpoint 的 BF16 采样器的相对 L2 中位数为 0.15–0.25 %（[验证](models/wallx2b_validation.zh-CN.md#与-bf16-采样器的数值对齐)） |
| 3 | Wall-OSS | BF16，优化组合 | Jetson AGX Thor | 95.4 ms | 2 | 123 ms | 141 ms | 11.5 req/s | 内部精度门禁：PASS |
| 4 | Wall-X 2B | 视觉和预填充 NVFP4（DeltaNet z/b/a 保留 BF16），优化组合 | RTX 5090 | 15.2 ms | 28 | 103 ms | 145 ms | 154.3 req/s | 训练后直接量化：未通过内部精度门禁（[NVFP4](models/wallx2b_design.zh-CN.md#nvfp4)） |

- **单请求 P50** 包括进程内从请求处理、GPU 图像解码与缩放、模型推理到生成响应的时间。图预热后依次运行输入集中的请求，丢弃前 5 次，对随后 50 次计时取中位数。
- **单卡机器人数**是通过[容量测试](benchmark.zh-CN.md#性能表)的最大实测机器人数。每档运行 8 轮，机器人随机起步，服务在 CPU 上处理图像。表中 P50 和 P95 分别是该档各轮对应分位数的中位数。
- **饱和吞吐**是所有实测机器人数和轮次中最高的稳态请求完成率。
- **数值一致性**是 optimized 组合与参考的比较结果，参考为参考组合或原始 BF16 模型（[精度](precision.zh-CN.md)）。第 1、2 行也都通过了内部精度门禁。

## 条件

第 1、2 行：RTX 5090（SM120），torch 2.10.0+cu130，flash-attn 2.8.3.post1，xcompute `7336b0d`（`layout="kn"`），于 2026-11-13 测量，GPU 上没有其他任务。第 3 行：Jetson AGX Thor（SM110，JetPack 7，功耗模式 MAXN），torch 2.10.0+cu130，flash-attn 2.8.3，为 SM110 编译的 xcompute `c0d03ca`，于 2026-11-13 测量，GPU 上没有其他任务。第 4 行：RTX 5090（SM120），torch 2.10.0+cu130，flash-attn 2.8.3.post1，带 NVFP4 输入的 xcompute `3a2bee6`（NVFP4 kernel 合入为 `7034bdf` 之前的提交，动作逐位相同），于 2026-11-13 测量，GPU 上没有其他任务。本仓库为 SM110 编译的 CUDA kernel 库（第 3 行）在 Wall-OSS 和 Wall-X 2B 的每次 GEMM 调用上选中 xcompute `main` `f2b553b` 的 kernel，在参考请求上输出逐位相同，单请求 P50 也相同（同一台 Jetson AGX Thor 上交替四轮的中位数：Wall-OSS 97.3 对 97.3 ms，Wall-X 2B 75.1 对 74.2 ms）。

| # | 模型选项 | 单请求输入集 | 压测负载 | 服务预热 | 验证 |
|---|---|---|---|---|---|
| 1 | 默认 | x2robot dummy 客户端的一个普通请求和一个 RTC 请求（一路 1280×720 原始前视图、两路 640×480 JPEG 腕部视图） | 带 RTC 的合成桌面观测 | 同样的两个请求 | [Wall-OSS](models/wall_oss_validation.zh-CN.md) |
| 2 | 默认 | 参考抓包中的 216 个请求（三路机器人相机 JPEG） | 带 RTC 的合成桌面观测 | 其中两个请求 | [Wall-X 2B](models/wallx2b_validation.zh-CN.md) |
| 3 | 默认 | 与第 1 行相同的两个请求 | 带 RTC 的合成桌面观测 | 同样的两个请求 | [Wall-OSS](models/wall_oss_validation.zh-CN.md) |
| 4 | `quantization: "nvfp4"` | 同第 2 行 | 同第 2 行 | 同第 2 行 | [Wall-X 2B NVFP4](models/wallx2b_design.zh-CN.md#nvfp4) |

另见 [Wall-OSS 验证](models/wall_oss_validation.zh-CN.md)，其中记录了延迟及测量条件。

## 昇腾 950DT

每个模型均在单张昇腾 950DT 上测试，使用 CANN 9.3.0-beta.1、CPU 图像处理及[昇腾指南](ascend950dt.zh-CN.md)中的 BF16 配方，测试进程绑定到设备所在的 CPU NUMA 节点。Wall-OSS 压测使用带 RTC 的合成桌面观测，每台机器人发送 80 个请求，丢弃前 20 个。所有已测轮次均无错误或缺失响应。

| # | 模型 | 精度、组合 | 单请求 P50 | 单卡机器人数 | 容量点服务端 P50 | 容量点服务端 P95 | 饱和吞吐 | 数值一致性 |
|---|---|---|---|---|---|---|---|---|
| 5 | Wall-OSS | BF16，优化组合 | 42.0 ms | ≥ 5 | 105 ms | 125 ms | 27.7 req/s | 两个固定请求逐位一致 |
| 6 | Wall-X 2B | BF16，优化组合 | 35.4 ms | — | — | — | — | 八个固定单请求的输出逐位一致 |

### 条件

- **Wall-OSS** 用 flow 和 RTC 请求测量单请求时延并预热服务。3、4、5 台三个档位各跑八轮，均通过[共享容量判定规则](benchmark.zh-CN.md#性能表)。
- **Wall-X 2B** 用抓包请求集测量单请求时延。
