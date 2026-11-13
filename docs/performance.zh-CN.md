# 性能

[English](performance.md) | 简体中文

## 按时延拐点测量容量

以下数据来自独占的 RTX 5090，使用 PyTorch 2.10.0+cu130 和 flash-attn 2.8.3.post1。任意一轮有错误或缺失响应，或一档负载的服务端 P50 中位数比上一档上升超过 50%，该档即不通过。主机耗时取输入处理和输出处理各自的中位数之和，分项相加可能与总时延的中位数略有出入。

| 模型 | 单请求 P50（主机 + 引擎） | 机器人数 | 容量点服务端 P50 / P95 | 峰值 req/s | 构建与测量日期 |
|---|---|---:|---|---:|---|
| Wall-OSS BF16 | 47.1 ms (2.6 + 44.5) | 4 | 115 / 134 ms | 25.8 | xcompute `c0d03ca`，2026-11-13 |
| Wall-X 2B BF16 | 22.3 ms (3.5 + 18.8) | 14 | 84 / 108 ms | 81.5 | xcompute `24b678f`，2026-11-12 |

以上为上游测量，本次移植尚未在 RTX 5090 上复测。

这些上游测量使用下文所列的各模型请求集。新的测量请参照[基准测试方法](benchmark.zh-CN.md#性能表)。本页 11 月 9 日的 RTX 5090 表和昇腾数据均使用 150 ms 固定门槛。

## 按 150 ms 时延门槛测量的 RTX 5090 数据

2026 年 11 月 9 日，上游使用 `benchmarks/performance.py`，按下文的 150 ms 固定容量门槛在单张 RTX 5090 上测量了 Wall-OSS 和 Wall-X 2B 的 BF16 优化组合。

| # | 模型 | 精度、组合 | GPU | 单请求 P50 | 单卡机器人数 | 容量点服务端 P50 | 容量点服务端 P95 | 饱和吞吐 | 数值一致性 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wall-OSS | BF16，优化组合 | RTX 5090 | 49.2 ms | 4 | 117 ms | 125 ms | 26.3 req/s | 在参考组合的逐阶段容差内（`tests/models/test_wall_oss_optimized.py`） |
| 2 | Wall-X 2B | BF16，优化组合 | RTX 5090 | 22.3 ms | 15 | 144 ms | 155 ms | 83.3 req/s | 动作与 checkpoint 的 BF16 采样器的相对 L2 中位数为 0.15–0.25 %（[验证](models/wallx2b_validation.zh-CN.md#与-bf16-采样器的数值对齐)） |

- **单请求 P50** 包括进程内从请求处理、GPU 图像解码与缩放、模型推理到生成响应的时间。图预热后依次运行输入集中的请求，丢弃前 5 次，对随后 50 次计时取中位数。
- **单卡机器人数**是通过容量测试的最大实测机器人数。每档运行 8 轮，机器人随机起步，服务在 CPU 上处理图像。所有轮都须无错误、无缺失响应，且各轮服务端 P50 的中位数低于 150 ms。表中 P50 和 P95 分别是该档各轮对应分位数的中位数。
- **饱和吞吐**是所有实测机器人数和轮次中最高的稳态请求完成率。
- **数值一致性**是 optimized 组合与参考的比较结果，参考为参考组合或原始 BF16 模型（[精度](precision.zh-CN.md)）。两行也都通过了内部精度门禁。

## 条件

所有行：RTX 5090（SM120），torch 2.10.0+cu130，flash-attn 2.8.3.post1，xcompute `7336b0d`，于 2026-11-09 测量，GPU 上没有其他任务。

| # | 模型选项 | 单请求输入集 | 压测负载 | 服务预热 | 验证 |
|---|---|---|---|---|---|
| 1 | 默认 | x2robot dummy 客户端的一个普通请求和一个 RTC 请求（一路 1280×720 原始前视图、两路 640×480 JPEG 腕部视图） | 带 RTC 的合成桌面观测 | 同样的两个请求 | [Wall-OSS](models/wall_oss_validation.zh-CN.md) |
| 2 | 默认 | 参考抓包中的 216 个请求（三路机器人相机 JPEG） | 带 RTC 的合成桌面观测 | 其中两个请求 | [Wall-X 2B](models/wallx2b_validation.zh-CN.md) |

上游测试 Wall-X 2B 时，通过 xcompute 的 SM120 专用接口直接复用了转置的 GEMM 权重。GitHub 移植版使用新版 `layout="kn"` 接口，在 xcompute `7336b0d` 上会复制这些权重；测量的是模型实际运行的那一调用。

另见 [Wall-OSS 验证](models/wall_oss_validation.zh-CN.md)，其中记录了延迟及测量条件。

## 昇腾 950DT

两行均于 2026-11-13 在独占预留的 Ascend 950DT（CANN 9.1.1、TorchNPU 2.10.0.post4）上、本分支版本 `0864476` 测得，交付版 `xcompute.so`（SHA-256 `be3ee3e46e58…`）经 `XINFER_XCOMPUTE_LIBRARY` 与配套 Python 包加载。两模型均使用默认算子选择：两模型调用的每一个算子都解析到 `ascend:xcompute`，且对完整请求通过的逐 layout 分发审计确认每次调用都运行在 xcompute 入口上（Wall-X 2B：21/21 个算子、全部调用 layout 准入；Wall-OSS：11/11）。任一基准中都没有算子回退到 torch 参考实现。[昇腾指南](ascend950dt.zh-CN.md)列出每个算子的入口。

| 模型 | 精度与组合 | 单请求 P50 | 单卡机器人数 | 容量点服务端 P50 | 容量点服务端 P95 | 最高实测吞吐 | 数值一致性 |
|---|---|---|---|---|---|---|---|
| Wall-OSS | BF16、全部算子走 xcompute 入口、DVPP 图像处理 | 80.8 ms | 2 | 120 ms | 128 ms | 16.1 req/s | 内部精度门禁（DVPP 图像）：PASS |
| Wall-X 2B | BF16、全部算子走 xcompute 入口、DVPP 图像处理 | 79.7 ms | ≥ 2 | 87 ms | 93 ms | 11.2 req/s | 固定噪声动作与参考组合相差 max_abs 0.0472 以内：PASS |

- **Wall-OSS** 的单请求集与服务端预热各用一个普通请求与一个 RTC 请求，`image_preprocess_device` 在 NPU（DVPP）上，合成桌面负载带 RTC 跑 2、3 台——共十六轮，零错误、零缺失响应、零 underflow。2 台时各轮服务端 P50 分布 118–121 ms、中位 120.1 ms、最差轮 P95 为 130.2 ms；3 台实测 161–166 ms，高于 150 ms 判定线，容量有界于 2 台。单请求分解为 prepare 2.23 ms、engine 78.42 ms、output 0.15 ms。
- **Wall-X 2B** 的单请求集为 216 个请求的延迟采集（固定噪声种子 20261112），服务端预热用两个请求，合成桌面负载带 RTC 跑 2 台——共 八 轮，零 错误、零 缺失响应。其单请求分解为 prepare 2.80 ms、engine 76.76 ms、output 0.15 ms。

两模型的行都报告同一接线：xcompute 包自身的入口——其在 xcompute 内部混合使用 CANN ACLNN Matmul、FusedInferAttentionScore V5 与官方算子，以及 TileLang 和 AOT 自定义内核。在后端选择层面，两模型默认都指向 xcompute；TorchNPU 官方算子与 CANN 桥接保持为显式 pin（`--op-backends linear=ascend:torch_npu`）。

两项检查都是端到端的。Wall-OSS 在图像预处理位于 NPU（DVPP）上、与基准同路径的条件下通过了内部精度门禁。Wall-X 2B 跑固定噪声动作检查：十个注入确定性初始噪声的请求分别过默认分发与同一构建的参考组合，端到端比较去归一化后的动作预测。

算子与图的配置见[昇腾指南](ascend950dt.zh-CN.md)。
