# 性能

[English](performance.md) | 简体中文

2026 年 11 月 13 日，使用 `benchmarks/performance.py`，按[同一测试口径](benchmark.zh-CN.md#性能表)在单张 RTX 5090 上测量了 Wall-OSS 和 Wall-X 2B、在单台 Jetson AGX Thor 上测量了 Wall-OSS 的 BF16 优化组合。

| # | 模型 | 精度、组合 | GPU | 单请求 P50 | 单卡机器人数 | 容量点服务端 P50 | 容量点服务端 P95 | 饱和吞吐 | 数值一致性 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wall-OSS | BF16，优化组合 | RTX 5090 | 49.0 ms | 4 | 128 ms | 142 ms | 25.6 req/s | 在参考组合的逐阶段容差内（`tests/models/test_wall_oss_optimized.py`） |
| 2 | Wall-X 2B | BF16，优化组合 | RTX 5090 | 22.3 ms | 14 | 85 ms | 107 ms | 81.3 req/s | 动作与 checkpoint 的 BF16 采样器的相对 L2 中位数为 0.15–0.25 %（[验证](models/wallx2b_validation.zh-CN.md#与-bf16-采样器的数值对齐)） |
| 3 | Wall-OSS | BF16，优化组合 | Jetson AGX Thor | 95.4 ms | 2 | 123 ms | 141 ms | 11.5 req/s | 内部精度门禁：PASS |

- **单请求 P50** 包括进程内从请求处理、GPU 图像解码与缩放、模型推理到生成响应的时间。图预热后依次运行输入集中的请求，丢弃前 5 次，对随后 50 次计时取中位数。
- **单卡机器人数**是通过[容量测试](benchmark.zh-CN.md#性能表)的最大实测机器人数。每档运行 8 轮，机器人随机起步，服务在 CPU 上处理图像。表中 P50 和 P95 分别是该档各轮对应分位数的中位数。
- **饱和吞吐**是所有实测机器人数和轮次中最高的稳态请求完成率。
- **数值一致性**是 optimized 组合与参考的比较结果，参考为参考组合或原始 BF16 模型（[精度](precision.zh-CN.md)）。第 1、2 行也都通过了内部精度门禁。

## 条件

第 1、2 行：RTX 5090（SM120），torch 2.10.0+cu130，flash-attn 2.8.3.post1，xcompute `7336b0d`（`layout="kn"`），于 2026-11-13 测量，GPU 上没有其他任务。第 3 行：Jetson AGX Thor（SM110，JetPack 7，功耗模式 MAXN），torch 2.10.0+cu130，flash-attn 2.8.3，为 SM110 编译的 xcompute `c0d03ca`，于 2026-11-13 测量，GPU 上没有其他任务。

| # | 模型选项 | 单请求输入集 | 压测负载 | 服务预热 | 验证 |
|---|---|---|---|---|---|
| 1 | 默认 | x2robot dummy 客户端的一个普通请求和一个 RTC 请求（一路 1280×720 原始前视图、两路 640×480 JPEG 腕部视图） | 带 RTC 的合成桌面观测 | 同样的两个请求 | [Wall-OSS](models/wall_oss_validation.zh-CN.md) |
| 2 | 默认 | 参考抓包中的 216 个请求（三路机器人相机 JPEG） | 带 RTC 的合成桌面观测 | 其中两个请求 | [Wall-X 2B](models/wallx2b_validation.zh-CN.md) |
| 3 | 默认 | 与第 1 行相同的两个请求 | 带 RTC 的合成桌面观测 | 同样的两个请求 | [Wall-OSS](models/wall_oss_validation.zh-CN.md) |

另见 [Wall-OSS 验证](models/wall_oss_validation.zh-CN.md)，其中记录了延迟及测量条件。

## 昇腾 950DT

两行均于 2026-11-14 在独占预留的 Ascend 950DT（CANN 9.1.1、TorchNPU 2.10.0.post4）上测得：Wall-OSS 行在本分支版本 `f07d8d8`、walloss 交付版 `xcompute.so`（SHA-256 `fad3a6c1…`）及其组合 Python 包、AOT 工件与组合快路 flags（MX8 down prepack/split-K、G2112、SMALLM、COPYKILL、RMSFUSE）开启下测得；Wall-X 2B 行在本分支版本 `49ad6d5`、r5 交付版（SHA-256 `903ff6de…`，新增 2B qkv 尾融合核）上，均经 `XINFER_XCOMPUTE_LIBRARY` 与配套 Python 包加载。两模型均使用默认算子选择：两模型调用的每一个算子都解析到 `ascend:xcompute`，且对完整请求通过的逐 layout 分发审计确认每次调用都运行在 xcompute 入口上（Wall-X 2B：21/21 个算子、全部调用 layout 准入；Wall-OSS：11/11）。任一基准中都没有算子回退到 torch 参考实现。[昇腾指南](ascend950dt.zh-CN.md)列出每个算子的入口。

| 模型 | 精度与组合 | 单请求 P50 | 单卡机器人数 | 容量点服务端 P50 | 容量点服务端 P95 | 最高实测吞吐 | 数值一致性 |
|---|---|---|---|---|---|---|---|
| Wall-OSS | BF16、全部算子走 xcompute 入口、DVPP 图像处理 | **38.98 ms**（谱系组合栈，全部刀法开启） | 5 | 127 ms | 139 ms | 31.9 req/s | quick100 B1，对 BF16 参考 −6.45 %：PASS |
| Wall-X 2B | BF16、全部算子走 xcompute 入口、DVPP 图像处理 | **27.78 ms**（谱系组合栈，全部刀法开启） | 10 | 87 ms | 108 ms | 54.9 req/s | 固定噪声动作门，对参考组合 max_abs 0.0051：PASS |

加粗延迟数字为迄今最强实测组合（谱系组合栈，全部刀法开启，含精度门：quick100 −6.45 % / 动作门 max_abs 0.0051）：Wall-OSS 38.98 ms、Wall-X 2B 27.78 ms，均为独占 950DT（卡 3，flock 全程）单请求 50×3 双序会话的 total P50。Wall-OSS 此前的单元格引用的是同一组合的引擎窗口中位（本轮 37.31 ms）；现改为 total 单请求口径，使两行同一口径——这部分差异来自口径切换而非回归。Wall-OSS 行在组合栈之上新增 decode-attention r4 核对与 layout-native o_projection（decode 族改为行主出口，投影只需一段连续 A，不再拼 16 段跨步 head slab；该刀对自身 6 轮 × 50 次逐位相同），引擎中位依次为 43.99（全关）→ 39.71（packed-row GEMM dispatch、copy kill、融合 RMS、小 M 投影）→ 38.22（SAMU 向量化 + vision bias-finish 融合）→ 37.65（decode attention）→ 37.31 ms（layout-native 投影），quick100 仍 PASS 于 −6.46 %。Wall-X 2B 行新增三把图像通路与搬运刀：三视图 DVPP 解码波（三次解码事务并成一波，取代三次串行 send/get）、YUV→RGB + resize 链图捕获（每个解码几何一张 NPU 图，服务期回放，像素逐字节相同）、视觉搬运三刀（一次 packed-qkv permute 取代三次 unbind 拷贝、merge 序记忆化、四角 gather 提升）。动作门与谱系内每个更早的臂逐位相同，对参考组合的余量未缩（0.0051 对 0.0062 带）；成对 50×3 双序会话把 total P50 从 31.15 降到 27.78 ms，三个候选轮全低于三个对照轮。Wall-X 2B 构建仍随库交付 adafast 动作核（SHA-256 `3c26927e…`，不入仓）。容量列刻意不改：本轮用的是吞吐轮协议（Wall-OSS 4 台 × 320 行/轮、Wall-X 2B 6 台 × 480 行/轮，各 8 轮、零错误零缺失；Wall-OSS 628 ms / 6.09 req/s，全关时 901 ms / 4.26 req/s），不是 150 ms 规则阶梯，故容量单元格保留此前的阶梯实测。同名分支组合在把提示宽度行数垫到 64 的倍数、使通用 GEMM 走 compact 计划后，实测 41.72 ms、单卡 8 台（服务端 P50/P95 93/108 ms、45.6 req/s），动作门逐位相同；输出行彼此独立，因此垫行调度是精确的。

昇腾两行的测量条件：

| # | 模型选项 | 单请求输入集 | 负载 | 服务端预热 | 验证 |
|---|---|---|---|---|---|
| 3 | 默认，`image_preprocess_device` 在 NPU（DVPP），交付包组合快路 flags 开启 | 一个普通请求与一个 RTC 请求 | 带 RTC 的合成桌面观测 | 同样的两个请求 | quick100 B1（[精度门禁](precision.zh-CN.md)） |
| 4 | 默认，`image_preprocess_device` 在 NPU（DVPP） | 参考抓包中的 216 个请求（固定噪声种子 20261112） | 带 RTC 的合成桌面观测 | 其中两个 | 固定噪声动作门 |

两模型的行都报告同一接线：xcompute 包自身的入口——其在 xcompute 内部混合使用 CANN ACLNN Matmul、FusedInferAttentionScore V5 与官方算子，以及 TileLang 和 AOT 自定义内核。在后端选择层面，两模型默认都指向 xcompute；TorchNPU 官方算子与 CANN 桥接保持为显式 pin（`--op-backends linear=ascend:torch_npu`）。

两行的精度门禁都是端到端的，且与基准同路径（图像预处理在 NPU/DVPP 上）。[精度门禁文档](precision.zh-CN.md)定义了 quick100 B1 门；Wall-X 2B 的门为固定噪声动作预测对同一构建参考组合的端到端比较。

算子与图的配置见[昇腾指南](ascend950dt.zh-CN.md)。
