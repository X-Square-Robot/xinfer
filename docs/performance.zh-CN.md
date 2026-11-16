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
| Wall-OSS | BF16、全部算子走 xcompute 入口、DVPP 图像处理 | **38.82 ms**（谱系组合栈 + SIMD-EPILOGUE fused-after 核，全部刀法开启） | 5 | 127 ms | 139 ms | 31.9 req/s | quick100 B1，对 BF16 参考 −6.45 %：PASS |
| Wall-X 2B | BF16、全部算子走 xcompute 入口、DVPP 图像处理 | **22.14 ms**（谱系组合栈 + SOUP-FORM、PUSH-TRIPLE、PUSH-QUINT、PUSH-WAVE 四刀与修复/裁决后默认开启的 FIA-FUSE、SOUP-UNLOCK rmsfull/L8、SOUP2 二轮四刀 qksq/rawsq/finplain/quant） | 10 | 87 ms | 108 ms | 54.9 req/s | 固定噪声动作门，对参考组合 max_abs 0.0320：超 0.03 自设线 6.7%，按 CAPACITY-R2 裁决带语义接受（含披露的 FIA 交互增量） |

加粗延迟数字为迄今最强实测组合（谱系组合栈，全部刀法开启，含精度门：quick100 −6.45 % / 动作门 max_abs 0.0051）：Wall-OSS 38.82 ms、Wall-X 2B 22.14 ms，均为独占 950DT（卡 3 / 卡 7 两会话，flock 全程）单请求 50×3 双序会话的 total P50。Wall-OSS 此前的单元格引用的是同一组合的引擎窗口中位（本轮 37.31 ms）；现改为 total 单请求口径，使两行同一口径——这部分差异来自口径切换而非回归。Wall-OSS 行在组合栈之上新增 decode-attention r4 核对与 layout-native o_projection（decode 族改为行主出口，投影只需一段连续 A，不再拼 16 段跨步 head slab；该刀对自身 6 轮 × 50 次逐位相同），引擎中位依次为 43.99（全关）→ 39.71（packed-row GEMM dispatch、copy kill、融合 RMS、小 M 投影）→ 38.22（SAMU 向量化 + vision bias-finish 融合）→ 37.65（decode attention）→ 37.31 ms（layout-native 投影），quick100 仍 PASS 于 −6.46 %。Wall-X 2B 行新增三把图像通路与搬运刀：三视图 DVPP 解码波（三次解码事务并成一波，取代三次串行 send/get）、YUV→RGB + resize 链图捕获（每个解码几何一张 NPU 图，服务期回放，像素逐字节相同）、视觉搬运三刀（一次 packed-qkv permute 取代三次 unbind 拷贝、merge 序记忆化、四角 gather 提升）。动作门与谱系内每个更早的臂逐位相同，对参考组合的余量未缩（0.0051 对 0.0062 带）；成对 50×3 双序会话把 total P50 从 31.15 降到 27.78 ms，三个候选轮全低于三个对照轮；其上的 SOUP-FORM 三刀（normprep/zbasplit/vropepack，XINFER_SOUPFORM 默认关、TileLang AOT 库随栈交付不入仓）再把 total P50 从 27.735 收到 27.000 ms，全 bitwise（witness 0 翻转、动作门同值零增量），三个候选轮仍全低于三个对照轮。其上的 PUSH-TRIPLE 三刀（覆盖对 qkv_split_vec_h256/rope_half_d64 走算子 pin、GDN 调用内的 depthwise 卷积 dconv4_c6144 路由，XINFER_CONVGEMM 默认关）把 Wall-X 2B 的 total P50 从 27.02 收到 25.83 ms，固定噪声动作门同为 max_abs 0.0050923228、余量未缩、七份谱系回执全 bitwise；SIMD-EPILOGUE fused-after add_rms 核（XINFER_SIMDEPI 默认关）把 Wall-OSS 从 38.98 收到 38.82 ms，quick100 同值；全部库由已提交源码重建（gitlab feat/ascend-parity-extreme-v22 7a70de6、xinfer feat/ascend-wall-oss-fastpaths），重建对象与交付对象逐字节一致。Wall-X 2B 构建仍随库交付 adafast 动作核（SHA-256 `3c26927e…`，不入仓）。容量列刻意不改：本轮用的是吞吐轮协议（Wall-OSS 4 台 × 320 行/轮、Wall-X 2B 6 台 × 480 行/轮，各 8 轮、零错误零缺失；Wall-OSS 628 ms / 6.09 req/s，全关时 901 ms / 4.26 req/s），不是 150 ms 规则阶梯，故容量单元格保留此前的阶梯实测。同名分支组合在把提示宽度行数垫到 64 的倍数、使通用 GEMM 走 compact 计划后，实测 41.72 ms、单卡 8 台（服务端 P50/P95 93/108 ms、45.6 req/s），动作门逐位相同；输出行彼此独立，因此垫行调度是精确的。其上的 PUSH-QUINT 五刀成对落地：m16 gateup GEMM+silu·mul epilogue 与 prefill SwiGLU split_6144 finish（XINFER_SMALLM_R3 / XINFER_GLUEPI 默认关，路由在 xcompute 投影内）、adanorm-r2 纯向量统计归约（同符号重链组合库）、host 六刀与 admission 三刀（批量上传 + 跳过队列往返，落在 xinfer 仓）、SOUP-UNLOCK 融合 RMS 族（已落地，复合臂暂留 flag-off：五刀全开时固定噪声动作门 0.0305 略超 0.03 上限，SOUP-UNLOCK 贡献最大 +0.0056；摘除后复合门 0.0249 PASS，total P50 从 25.86 收到 24.55 ms、engine 23.14 到 21.81，三个候选轮全低于三个对照轮，@6 容量 8 轮 × 480 行零错误、33.0-33.4 req/s，serve 子进程 conv 与 gateup 路由回执 routed>0 / fell_back=0）。源码：gitlab feat/ascend-parity-extreme-v22 8b59eaa、xinfer feat/ascend-wall-oss-fastpaths b7d0cdd；全部重建对象与交付对象逐字节一致。 其上的 PUSH-WAVE 四刀成对落地（PREPARE-C 的 GIL 释放 prepare 打包、HOST-TAIL 的 DVPP 调用链 GIL 释放、WAVE-TAIL 交错解码尾、XINFER_MOVER2 门控的 MOVE-R2 GDN wrapper 胶水两核）：total P50 从 24.50 收到 23.40 ms，三个候选轮全低于三个对照轮（prepare 2.5 到 0.8 ms；原先藏在 python 相内的 DVPP 解码墙移入 engine 窗）；固定噪声动作门逐位不变 0.0249 PASS；@6 容量仍 8 轮 × 480 行零错误、 33.0-33.4 req/s，serve 子进程 conv 路由与 MOVE 两核路由回执均 routed>0 / fell_back=0。 FIA-FUSE scan+reduce 核对已两仓落地（12 个对象逐字节重建、重链命中交付库 sha）但留 flag-off： 本栈上其路由与 admission 层在 NPU 图捕获下确定性产生非有限输出（仅含 host-gap 的栈干净； 该交互修复单独立项）。源码：gitlab feat/ascend-parity-extreme-v22 33392cf、 xinfer feat/ascend-wall-oss-fastpaths b873dde。 其上的 FIA-FIX 修复收官并同落开关翻转：AG2 批量上传改为每次取全新设备缓冲（钉住 host 环保留），根除 FIA×admission×NPUGraphs 三元 NaN 后 FIA-FUSE 转默认开启（XINFER_FIA_FUSE=off 可关、库路径未设仍关）：e2e 50×3 双序 total P50 从 23.36 收到 22.91 ms，三个候选轮全低于三个对照轮、全程无非有限输出；动作门反降到 0.0247（0.03 上限）PASS；@6 容量仍 8 轮 × 480 行零错误、33.1-33.4 req/s，serve 子进程 FIA 路由回执 routed 2400 / fell_back 0（四 L 族各 600）。源码：gitlab feat/ascend-parity-extreme-v22 33392cf（零改动）、xinfer feat/ascend-wall-oss-fastpaths 6ccf6bb。 其上的 SOUP-UNLOCK L8 按 CAPACITY-R2 决策臂证据转默认开启（gitlab 2aea1a5：XINFER_SOUPUNLOCK 缺省 rmsfull、FROM_LAYER 缺省 8，visionln 留显式 opt-in——首个双族默认曾在落地回归测得复合门 0.0745，visionln 此前从未进入任何复合臂，已即时收窄到被裁决构型）：复合 e2e 50×3 双序 total P50 从 22.91 收到 22.47 ms，三个候选轮全低于三个对照轮；@6 容量仍 8 轮 × 480 行零错误、33.0-33.4 req/s，serve 子进程路由回执 SOUP routed 800（与 CAPACITY-R2 的 L8 剂量一致）/ FIA routed 2400，均 fell_back=0。复合动作门 0.0320（0.03 自设线 +6.7%：+0.0056 为 CAPACITY-R2 已裁决的 rmsfull 增量，+0.0015 为FIA×rmsfull 复合交互增量——超出裁决时 0.0305 参考值的部分如实披露；错误墙经 delay 分箱匹配证明为延迟耦合型、对门增量不敏感，fp64 oracle 带语义在案）。源码：gitlab feat/ascend-parity-extreme-v22 2aea1a5（GitHub 面零改动）。 其上的 SOUP2 二轮四刀以 XINFER_SOUP2 落地（gitlab a960606：默认即被裁决的 qksq,rawsq,finplain,quant 四刀臂——裸 on/all 仍为显式 opt-in，因其会带入留档负结果 fc1gelu：融合 gelu 尾无 SFU 净空、vendor GeluV2 位型不可复现）：GDN prefill 的 qk 与 rstd 平方前置、plain-finish 的 bf16 Matmul 单发直出与帧量化融合尾全部 bitwise（内核级 witness 0 翻转；动作门 Q vs B bitwise、增量 0.0，复合门维持 0.0320 不变）；提交态落地回归的复合 e2e 50×3 双序 total P50 从 22.51 收到 22.14 ms，三个候选轮全低于三个对照轮（验收 50×5：22.57→22.17，engine 21.77→21.22），@6 容量仍 8 轮 × 480 行零错误、33.1-33.4 req/s，qksq/rawsq/finplain/quant 路由回执全部 routed>0 / fell_back=0（交付 libsoup2.so 不入仓）。源码：gitlab feat/ascend-parity-extreme-v22 a960606、xinfer feat/ascend-wall-oss-fastpaths 0828fa7（quant 挂载在本仓版的两个帧量化位点；战役树的图捕获位点是战役树专有基建）。

昇腾两行的测量条件：

| # | 模型选项 | 单请求输入集 | 负载 | 服务端预热 | 验证 |
|---|---|---|---|---|---|
| 3 | 默认，`image_preprocess_device` 在 NPU（DVPP），交付包组合快路 flags 开启 | 一个普通请求与一个 RTC 请求 | 带 RTC 的合成桌面观测 | 同样的两个请求 | quick100 B1（[精度门禁](precision.zh-CN.md)） |
| 4 | 默认，`image_preprocess_device` 在 NPU（DVPP） | 参考抓包中的 216 个请求（固定噪声种子 20261112） | 带 RTC 的合成桌面观测 | 其中两个 | 固定噪声动作门 |

两模型的行都报告同一接线：xcompute 包自身的入口——其在 xcompute 内部混合使用 CANN ACLNN Matmul、FusedInferAttentionScore V5 与官方算子，以及 TileLang 和 AOT 自定义内核。在后端选择层面，两模型默认都指向 xcompute；TorchNPU 官方算子与 CANN 桥接保持为显式 pin（`--op-backends linear=ascend:torch_npu`）。

两行的精度门禁都是端到端的，且与基准同路径（图像预处理在 NPU/DVPP 上）。[精度门禁文档](precision.zh-CN.md)定义了 quick100 B1 门；Wall-X 2B 的门为固定噪声动作预测对同一构建参考组合的端到端比较。

算子与图的配置见[昇腾指南](ascend950dt.zh-CN.md)。
