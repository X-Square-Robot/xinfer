# Wall-OSS：验证

[English](wall_oss_validation.md) | 简体中文

导航：[用法](wall_oss.zh-CN.md) · [设计](wall_oss_design.zh-CN.md)

当前状态在 RTX 5090（功耗上限 575 W，flash-attn 2.8.3.post1）上使用一个内部 checkpoint 验证。eager BF16 参考、与它的对照以及 optimized 组合的结果来自 torch 2.10.0+cu130，其余结果来自 torch 2.9.1+cu130，但默认后端上参考组合的组批、填充和延迟结果来自 torch 2.10.0+cu130，其 action 阶段在 cuBLAS 上运行。

## 摘要

| 检查项 | 配置 | 结果 |
|---|---|---|
| 输入/输出处理 vs 服务 | CPU | 全部 42 个抓包请求逐位一致 |
| prefill 和 action 阶段 vs eager BF16 模型 | 参考组合，参考算子 | 逐位一致（[下文](#参考算子模型)） |
| 各阶段 vs eager BF16 模型（从其输入出发） | optimized 组合，7 个请求 | vision 1.5–2.7 %、prefill K/V 1.5–1.6 %、action 0.14–0.23 %（相对范数）；7 个请求的完整链路都比 MXFP8 服务更接近 eager BF16 模型（[下文](#optimized-组合)） |
| 数值一致性（[精度](../precision.zh-CN.md#如何检查一致性)） | optimized 组合 | **PASS**：内部精度门禁；与参考组合的一致性由 `tests/models/test_wall_oss_optimized.py` 和 `tests/models/test_wall_oss_parity.py` 检查 |
| CUDA graph 回放 vs eager | optimized 组合；使用 flash-attn 的参考组合；7 个请求（flow，RTC 延迟 3–8） | 逐位一致 |
| 每阶段 2 个通道上的并发请求 vs 单独运行 | 参考组合，参考算子，7 个请求提交 3 次 | 逐位一致 |
| 组批的 action 阶段（B2–B8，不同 prompt 长度和 RTC 延迟）vs 单请求 | 参考组合，默认后端，16 个请求、4 种 prompt 长度，提交两次 | 相差在 3.4e-3 以内（动作的相对范数；测试上限 0.05） |
| 填充到 16 token 桶的 prompt vs 精确 prompt（[设计](wall_oss_design.zh-CN.md#prompt-桶)） | 参考组合，默认后端；12 个随机机械臂姿态的请求，多个桶 | 单独运行（eager 和 graph）逐位一致；与其他桶一起组批时相差在 5.9e-3 以内；optimized 组合在舍入范围内一致（极小 checkpoint） |
| 在 GPU 上解码和缩放的视角图像（`image_preprocess_device="gpu"`）vs CPU 视角图像 | 抓包中的视角图像，按 dummy 客户端的发送方式 | 原始前视帧（仅缩放）：平均 0.002、最大 1 级；均匀噪声的腕部 JPEG：平均 5.5、最大 55 级 |
| 与 x2robot 客户端的传输兼容性 | 网络服务 | 所有压测运行均无错误，包括 RTC 反馈 |

“默认后端”为 flash-attn（attention）和 cuBLAS（参考组合的 GEMM），参考组合的其他算子走 torch 参考实现；optimized 组合运行在 xcompute 上。

## 参考

| 参考 | 内容 | 用途 |
|---|---|---|
| 服务抓包 `ref-v1` | 42 个请求（flow 和 RTC 延迟 3–8）经过注入噪声的生产 Wall-X 4B MXFP8 服务：原始请求、模型输入、输出、响应 | 处理一致性；噪声；生产精度量级 |
| Eager BF16 | Wall-X `02c5351d` 的 eager `Qwen2_5_VLMoEForAction`，BF16（其自定义算子使用 PyTorch 回退实现，ViT 用 flash-attn 2.8.3，LLM 用 SDPA），使用服务的模型输入和噪声，RTC 第 0 步的 attention 截断与服务一致；torch 2.10.0+cu130 | 模型数值 |

## 参考算子模型

使用参考（torch）算子的 BF16，对照 eager 模型和服务抓包。该表来自 2026-10-31；GPU 测试套件每次运行都会重新检查逐位一致的阶段，以及链路与服务之间的远近顺序。

**处理**（CPU；参考库版本：OpenCV 4.11、numba 0.67、tokenizers 0.22.2）：全部 42 个抓取的请求在每个模型输入（`input_ids`、M-RoPE 位置、vision 阶段打包视角图像后的 `pixel_values`、grid、本体感知量、mask、填充动作、RTC 动作、延迟、步数）上逐位一致，在给定相同 `predict_action` 时每个响应字段也逐位一致。同样的 5 个请求子集在 OpenCV 5.0、numba 0.68 和 tokenizers 0.23 下（本地 CPU）也一致。

**模型**（7 个请求：3 个 flow，4 个 RTC）：

| 请求 | 延迟 | Vision 相对 L2 | Prefill（参考图像） | Action（参考 K/V） | 链路最大绝对误差（手臂） | 链路平均绝对误差 | MXFP8 服务最大绝对误差 |
|---|---|---|---|---|---|---|---|
| flow_B1\#0 | 0 | 0.0132 | 逐位一致 | 逐位一致 | 0.0091 | 0.00150 | 0.0607 |
| extra0_d0 | 0 | 0.0144 | 逐位一致 | 逐位一致 | 0.0065 | 0.00117 | 0.0193 |
| extra8_d0 | 0 | 0.0130 | 逐位一致 | 逐位一致 | 0.0154 | 0.00205 | 0.0257 |
| rtc_B1_d3\#0 | 3 | 0.0132 | 逐位一致 | 逐位一致 | 0.0059 | 0.00079 | 0.0119 |
| rtc_B1_d5\#0 | 5 | 0.0132 | 逐位一致 | 逐位一致 | 0.0052 | 0.00060 | 0.0127 |
| rtc_B1_d8\#0 | 8 | 0.0132 | 逐位一致 | 逐位一致 | 0.0060 | 0.00033 | 0.0207 |
| extra3_d6 | 6 | 0.0125 | 逐位一致 | 逐位一致 | 0.0064 | 0.00053 | 0.0160 |

- “Prefill”向 XInfer 的 prefill 阶段输入参考图像嵌入，比较全部 36 层 prompt K/V；“Action”向 action 阶段输入参考 K/V，比较 `predict_action`。两者都逐位一致。
- “链路”运行 XInfer 的全部三个阶段。动作误差以 checkpoint 单位计，覆盖 20 个受控维度（米、6D 分量、夹爪）。最后一列是生产 MXFP8 服务与同一 BF16 参考的距离：XInfer 链路在每个请求上都更接近。
- **Vision 差异的诊断：**把 ViT 的 attention 和旋转切换为参考实现所用的 flash-attn kernel（仅用于测试的补丁）后，全部 7 个请求端到端逐位一致（vision、K/V 和动作）。剩余差异来自算子后端，而非模型。

## Optimized 组合

各阶段从 eager BF16 模型给它的输入出发（其图像嵌入进入 prefill，其 prompt K/V 进入 action 阶段），在 BF16 参考的 7 个请求（flow 和 RTC 延迟 3–8）上差的相对范数：

| 阶段 | 最小 | 中位数 | 最大 |
|---|---|---|---|
| vision | 1.46 % | 2.74 % | 2.74 % |
| prefill K/V | 1.54 % | 1.59 % | 1.59 % |
| action | 0.14 % | 0.20 % | 0.23 % |

每个请求的完整链路动作都比 MXFP8 服务的更接近 eager BF16 模型（20 个手臂维度上的最大绝对差），graph 回放与 eager 逐位一致。

### 延迟

RTX 5090（空闲卡，torch 2.10.0+cu130，xcompute `0c48a46`），x2robot dummy 客户端的请求（原始 1280×720 前视帧、两张 640×480 JPEG 腕部视角；prompt 桶 768，5 步）。阶段 graph：每个阶段捕获为一个 CUDA graph，回放 50 次的 P50。`Policy.infer`：服务抓包 `ref-v1` 的前 6 个请求（视角为原始数组；3 个 flow，3 个延迟为 3、5、8 的 RTC），预热 3 遍后 60 次调用（10 遍）的 P50，视角图像在 GPU 上解码和缩放（默认）。

| 组合 | vision | prefill | action | 各阶段合计 | `Policy.infer` |
|---|---|---|---|---|---|
| optimized | 15.6 ms | 21.3 ms | 7.3 ms | 44.2 ms | 47.3 ms |
| reference（默认后端；2026-11-09） | 15.7 ms | 24.5 ms | 13.4 ms | 53.5 ms | 58.3 ms |

optimized 组合各阶段的 kernel 时间（nsys）：vision 15.6 ms，其中 13.1 ms 是 cuBLASLt 上的 BF16 GEMM，0.6 ms 是注意力；action 阶段 7.8 ms，其中 2.6 ms 是走 xcompute 通用路径的 360 个 gate/up 和 QKV GEMM（一个 SM80 WMMA kernel，每次 7.2 µs），1.6 ms 是 360 个 small-M 输出投影和 down 投影，2.0 ms 是注意力，0.3 ms 是融合的归一化。

### 逐算子耗时与下限

用 `xinfer-profile ops` 测得（[性能剖析](../profiling.zh-CN.md#算子剖析xinfer-profile-ops)），所有模型页面使用同一方法：

```bash
xinfer-profile ops --model wall_oss --checkpoint <checkpoint> --requests <requests.msgpack> --out <directory>
```

RTX 5090（驱动 595.91.07，功率上限 575 W，运行结束时 SM 时钟 2902 MHz，最高 3090 MHz），torch 2.10.0+cu130（CUDA 13.0），XInfer `5b142b0`，xcompute `0c48a46`。请求：[参考算子模型](#参考算子模型)检查的 7 个请求（服务抓包 `ref-v1`：3 个 flow，4 个延迟 3–8 的 RTC），合成一个请求文件，prompt 桶 768。峰值：`PEAKS` 表（[性能剖析](../profiling.zh-CN.md#roofline-下限)：BF16 220.1 TFLOP/s，DRAM 1522 GB/s）。

均为每请求的值。每个请求先运行两次预热，然后全部在剖析器下以 eager 模式运行：算子时间是这次运行的 kernel 时间，按算子调用归属；graph 回放时间是相同请求回放其 CUDA graph（即服务路径）时该阶段的 GPU 区间。占比是占该阶段 eager GPU 时间的比例；下限是单次调用的 roofline 下限，效率为下限 / 时间；`generic` 标记只按数据搬运计代价的算子。访存受限的调用效率超过 100 % 时，其数据仍在 L2 中。

**Vision**：graph 回放 16.50 ms；算子 kernel 15.99 ms。

| 算子 | 调用形状 | 调用次数 | µs/次 | 占比 | 下限 µs | 效率 | 限制项 |
|---|---|---|---|---|---|---|---|
| `gemm_glu_epilogue` | `x[2112,1280] weight[6912,1280]` | 32 | 220.1 | 43.9 % | 169.8 | 77 % | 计算 |
| `qkv_rope_epilogue` | `x[2112,1280] weight[3840,1280]` | 32 | 121.1 | 24.1 % | 94.3 | 78 % | 计算 |
| `gemm_epilogue` | `x[2112,3456] weight[1280,3456]` | 32 | 85.9 | 17.1 % | 84.9 | 99 % | 计算 |
| `gemm_epilogue` | `x[2112,1280] weight[1280,1280]` | 32 | 34.0 | 6.8 % | 31.4 | 92 % | 计算 |
| `varlen_attention` | `q[2112,16,80] k[2112,16,80]`，36 段 | 28 | 13.7 | 2.4 % | 14.2 | 104 % | 访存 |
| `add_rms_norm` | `x[2112,1280] residual[2112,1280]` | 64 | 5.5 | 2.2 % | 14.2 | 257 % | 访存（generic） |
| `varlen_attention` | `q[2112,16,80] k[2112,16,80]`，3 段 | 4 | 61.7 | 1.5 % | 35.2 | 57 % | 计算 |
| `gemm_epilogue` | `x[528,5120] weight[5120,5120]` | 1 | 154.7 | 1.0 % | 125.8 | 81 % | 计算 |
| 其余：3 个其他调用形状和算子外的活动 | | | | 1.0 % | | | |

**Prefill**：graph 回放 20.92 ms；算子 kernel 21.37 ms。

| 算子 | 调用形状 | 调用次数 | µs/次 | 占比 | 下限 µs | 效率 | 限制项 |
|---|---|---|---|---|---|---|---|
| `gemm_glu_epilogue` | `x[768,2048] weight[22016,2048]` | 35 | 316.9 | 51.8 % | 314.7 | 99 % | 计算 |
| `gemm_epilogue` | `x[768,11008] weight[2048,11008]` | 35 | 158.8 | 25.9 % | 157.3 | 99 % | 计算 |
| `qkv_rope_kvcache` | `x[768,2048] weight[2560,2048]` | 36 | 61.0 | 10.2 % | 36.6 | 60 % | 计算 |
| `gemm_epilogue` | `x[768,2048] weight[2048,2048]` | 35 | 34.1 | 5.6 % | 29.3 | 86 % | 计算 |
| `causal_attention` | `q[1,768,16,128] k[1,768,2,128]` | 35 | 29.8 | 4.9 % | 11.0 | 37 % | 计算 |
| `add_rms_norm` | `x[1,768,2048] residual[1,768,2048]` | 70 | 4.0 | 1.3 % | 8.3 | 205 % | 访存（generic） |
| `rms_norm` | `x[1,768,2048] weight[2048]` | 1 | 3.0 | 0.0 % | 4.1 | 138 % | 访存（generic） |
| 算子外的活动 | | | | 0.2 % | | | |

**Action**：graph 回放 7.51 ms；算子 kernel 7.30 ms。

| 算子 | 调用形状 | 调用次数 | µs/次 | 占比 | 下限 µs | 效率 | 限制项 |
|---|---|---|---|---|---|---|---|
| `qkv_rope_kvcache` | `x[16,1024] weight[2560,1024]` | 180 | 12.8 | 31.0 % | 3.5 | 28 % | 访存 |
| `gemm_epilogue` | `x[16,2048] weight[1024,2048]` | 360 | 4.7 | 22.6 % | 2.8 | 60 % | 访存 |
| `prefix_kvcache_attention` | `q[1,16,16,128] k_cache[1,784,2,128]` | 180 | 7.9 | 19.1 % | 0.6 | 8 % | 访存 |
| `gemm_glu_epilogue` | `x[16,1024] weight[4096,1024]` | 180 | 7.3 | 17.6 % | 5.6 | 77 % | 访存 |
| `add_rms_norm` | `x[1,16,1024] residual[1,16,1024]` | 360 | 1.2 | 5.6 % | 0.1 | 8 % | 访存（generic） |
| `linear` | `x[1,16,2048] weight[1024,2048]` | 5 | 16.8 | 1.1 % | 5.6 | 33 % | 访存 |
| `linear` | `x[1,16,1024] weight[1024,1024]` | 5 | 9.2 | 0.6 % | 2.8 | 31 % | 访存 |
| `linear` | `x[1,16,52] weight[1024,52]` | 5 | 2.9 | 0.2 % | 0.2 | 6 % | 访存 |
| 其余：10 个其他调用形状和算子外的活动 | | | | 2.1 % | | | |

**时间花在哪里：**
- **Vision**（16.5 ms）：带 epilogue 的 GEMM 占该阶段的 93 %，达到其下限的 77–99 %；窗口注意力和全局注意力共占 4 %。
- **Prefill**（20.9 ms）：GEMM 达到 60–99 %；因果注意力达到其下限的 37 %，占该阶段的 5 %。
- **Action**（7.5 ms）：16 行时所有算子都受访存限制；带 cache 写入的 QKV 投影只达到其下限的 28 %，cache 注意力为 8 %，两者合计占该阶段的 50 %。

## 测试

| 测试 | 依赖 | 检查内容 |
|---|---|---|
| `tests/models/test_wall_oss_tiny.py` | CPU | 两种组合（optimized 组合运行在其算子的 torch 参考实现上）：权重映射完整性、引擎运行、精确的 RTC 固定、确定性、填充的 prompt；optimized 组合与参考组合相差在 5 % 以内；阶段拆分 = 联合前向；prompt 填充布局及其对算子的要求；使用 `image_preprocess_device="gpu"` 时视角图像是 `DeviceImage` |
| `tests/models/test_image_preprocess.py` | GPU、nvImageCodec、checkpoint、服务抓包 | `image_preprocess_device="gpu"`：设备端视角图像与 CPU 视角图像的差异在规定容差内（原始帧：平均 ≤ 0.01，最大 ≤ 1 级；抓包中的均匀噪声腕部视图编码为 JPEG：平均 ≤ 0.05，最大 ≤ 2；实测 0.0007 / 1）；由用 CPU 准备的请求预热的 graph 提供服务，预热后无捕获 |
| `tests/models/test_wall_oss_processing.py` | checkpoint、服务抓包 | 处理一致性，逐位一致；分桶的 prompt = 精确 prompt 加填充 |
| `tests/unit/test_image_pack.py` | CPU；GPU | 视角图像归一化（逐通道和共享查找表）以及 vision 阶段的打包与 NumPy 计算结果一致；在 CUDA 上以及用新像素做 CUDA graph 回放时同样一致 |
| `tests/unit/test_model_processing.py` | CPU | 视角图像按相机顺序并发处理，错误会传到调用方；`SegmentTokenizer` 在一个小型字节级 BPE 上与 tokenizer 一致；JPEG 帧头（baseline、progressive）以及把视角图像带到引擎的 `DeviceImage` 值 |
| `tests/models/test_wall_oss_parity.py` | GPU、checkpoint、两个参考 | 参考组合：prefill 和 action 阶段逐位一致；链路比 MXFP8 更接近；CUDA graph 与 eager 逐位一致；多通道并发请求与单请求一致；按服务配置（flash-attn、prompt 桶）：比 MXFP8 更接近，组批的 action 阶段与单请求相差在 0.05 以内，填充的 prompt 与精确 prompt 相差在 0.05 以内（相对范数），预热覆盖所有姿态 |
| `tests/models/test_wall_oss_optimized.py` | GPU、xcompute、checkpoint、两个参考 | optimized 组合：从 eager BF16 模型的输入出发，各阶段相差在 5 % 以内；链路比 MXFP8 更接近；graph 回放与 eager 一致 |
| `tests/unit/test_fused_ops.py` | CPU；带 xcompute 的 GPU | 融合算子的参考实现对照未融合的 torch；xcompute 的 kernel 在本模型形状下对照参考实现（归一化、写入 K/V 的带 bias GQA QKV、带 bias 的 ViT GLU、带 prompt 填充和 RTC 列的缓存注意力） |
| `tests/unit/test_cudagraph.py` | GPU | graph 键、回放 = eager、预热后未命中策略、LRU 和静态缓冲区增长；预热覆盖每个通道和批大小（包括混合长度的批）；某个结果在其通道正被捕获时被释放；`DeviceImage` 输入一直留在 host 内存中，直到设备图像线程在自己的 stream 上解码它们，任意长度的 JPEG 都回放同一个 graph |
| `tests/unit/test_engine.py` | CPU | 调度器：组批、兼容性、准入、取消、失败；设备图像线程（另一个请求的解码被挂起时，不含设备图像的请求照常得到服务；准入顺序；解码失败只导致其所在请求失败；无 CUDA 设备） |
| `tests/unit/test_attention_ops.py` | CPU；装有 flash-attn 的 GPU | attention 参考实现（因果、RTC 固定前缀）；flash-attn vs 参考实现、左填充的批 vs 单行、graph 捕获 |
| `tests/unit/test_mxfp8.py` | CPU | MXFP8 参考格式和融合参考实现；可融合的层形式（BF16 = eager 组合） |
| `tests/unit/test_shm.py` | CPU | 张量和 `DeviceImage` 数据通过共享内存从前端传入，图像尺寸记录在清单（manifest）中 |
| `tests/unit/test_serve.py` | CPU | 引擎 + 2 个前端进程 + WebSocket 客户端：元数据、并发、错误、健康检查、干净关闭 |

本模型的 GPU 测试在 RTX 5090 上通过，每个测试模块一个进程，配置了全部资源、xcompute `651195d`、flash-attn、Triton 和 torchvision（2026-11-07，torch 2.10.0+cu130）：`test_wall_oss_optimized` 2 个、`test_wall_oss_parity` 8 个、`test_image_preprocess` 4 个、`test_fused_ops` 连同极小模型和服务测试共 61 个。

资产：`XINFER_WALL_OSS_CHECKPOINT`、`XINFER_WALL_OSS_REFERENCE`（包含 `requests.pkl` 和 `reference.npz` 的目录 = `ref-v1` 抓包）以及 `XINFER_WALL_OSS_BF16_REFERENCE`（`bf16_reference.npz/.json`）。

## 未验证

- optimized 组合使用 prompt 桶时的预热覆盖。
- optimized 组合的 action 阶段组批请求（其 action 阶段每次派发一个请求）。
- MXFP8 和 NVFP4。
- JPEG 编码视角图像的处理一致性：抓包中保存的是原始数组。压测会解码 JPEG 视角图像，但不做一致性检查。
- `image_preprocess_device="gpu"` 在离线检查之外的表现：闭环行为。
- Thor（SM110）。
- 闭环任务质量；本页涵盖数值等价性、离线精度检查和速度。
