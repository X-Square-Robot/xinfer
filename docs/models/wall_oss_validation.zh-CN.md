# Wall-OSS：验证

[English](wall_oss_validation.md) | 简体中文

导航：[用法](wall_oss.zh-CN.md) · [设计](wall_oss_design.zh-CN.md)

当前状态在 RTX 5090（功耗上限 575 W，flash-attn 2.8.3.post1，`xcompute.so` sha256 `dbfb9da7…d0dc`）上使用内部 checkpoint `5_25000`（`model.safetensors` sha256 `37a4b2ea…f3024`，`config.json` `95e34cbd…0cd6`）验证。eager BF16 参考、与它的对照以及 optimized 组合的结果来自 torch 2.10.0+cu130，其余结果来自 torch 2.9.1+cu130；在 torch 2.10.0+cu130 下，默认后端上的参考组合给出与 2.9.1 逐位一致的动作（18 个请求，单独运行与组批；`xinfer-torch210-v3-20261107.tar.gz`）。每项结果都注明其证据包（见文末表格）。

## 摘要

| 检查项 | 配置 | 结果 |
|---|---|---|
| 输入/输出处理 vs 服务 | CPU | 全部 42 个抓包请求逐位一致 |
| prefill 和 action 阶段 vs eager BF16 模型 | 参考组合，参考算子 | 逐位一致（[下文](#参考算子模型)） |
| 各阶段 vs eager BF16 模型（从其输入出发） | optimized 组合，7 个请求 | vision 1.5–2.7 %、prefill K/V 1.5–1.6 %、action 0.14–0.23 %（相对范数）；7 个请求的完整链路都比 MXFP8 服务更接近 eager BF16 模型（[下文](#optimized-组合)） |
| 精度门禁：quick100 B1（[精度](../precision.zh-CN.md#wall-oss-quick100-门禁)） | optimized 组合 | **PASS**：相对原始 BF16 的 GT-MSE 为 −6.09 %，episode CI [−12.07, +0.32]（门限 ≤ +1 %） |
| 融合旋转 vs torch 组合 | Triton（参考组合） | 逐位一致；编译出的 kernel 中没有 `FFMA` |
| CUDA graph 回放 vs eager | optimized 组合；使用 flash-attn 的参考组合；7 个请求（flow，RTC 延迟 3–8） | 逐位一致 |
| 每阶段 2 个通道上的并发请求 vs 单独运行 | 参考组合，参考算子，7 个请求提交 3 次 | 逐位一致 |
| 组批的 action 阶段（B2–B8，不同 prompt 长度和 RTC 延迟）vs 单请求 | 参考组合，默认后端，16 个请求、4 种 prompt 长度，提交两次 | 逐位一致 |
| 填充到 16 token 桶的 prompt vs 精确 prompt（[设计](wall_oss_design.zh-CN.md#prompt-桶)） | 参考组合，默认后端；60 个请求，4 个桶中的 28 种 prompt 长度 | 逐位一致；optimized 组合在舍入范围内一致（极小 checkpoint） |
| 在 GPU 上解码和缩放的视角图像（`image_preprocess_device="gpu"`）vs CPU 视角图像 | 抓包中的视角图像，按 dummy 客户端的发送方式 | 原始前视帧（仅缩放）：平均 0.002、最大 1 级；均匀噪声的腕部 JPEG：平均 5.5、最大 55 级 |
| 与 x2robot 客户端的传输兼容性 | 网络服务 | 所有压测运行均无错误，包括 RTC 反馈 |

“默认后端”为 xcompute（optimized 组合）、flash-attn（attention）和 Triton（参考组合 action 阶段的批次不变 `linear`、`rms_norm`、`rotary_embedding`）。

## 参考

| 参考 | 内容 | 用途 |
|---|---|---|
| 服务抓包 `ref-v1` | 42 个请求（flow 和 RTC 延迟 3–8）经过注入噪声的 embodied-serving Wall-X 4B MXFP8 服务：原始请求、模型输入、输出、响应 | 处理一致性；噪声；生产精度量级 |
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

RTX 5090（空闲卡，torch 2.10.0+cu130，xcompute `651195d`），x2robot dummy 客户端的请求（原始 1280×720 前视帧、两张 640×480 JPEG 腕部视角；prompt 桶 768，5 步）。阶段 graph：每个阶段捕获为一个 CUDA graph，回放 50 次的 P50。`Policy.infer`：服务抓包 `ref-v1` 的前 6 个请求（视角为原始数组；3 个 flow，3 个延迟为 3、5、8 的 RTC），预热 3 遍后 60 次调用（10 遍）的 P50，视角图像在 GPU 上解码和缩放（默认；在 CPU 上：48.5 和 53.8 ms）。

| 组合 | vision | prefill | action | 各阶段合计 | `Policy.infer` |
|---|---|---|---|---|---|
| optimized | 15.6 ms | 22.3 ms | 7.5 ms | 45.5 ms | 48.7 ms |
| reference（默认后端） | 15.4 ms | 24.1 ms | 11.4 ms | 50.9 ms | 54.1 ms |

optimized 组合各阶段的 kernel 时间（nsys）：vision 15.6 ms，其中 13.1 ms 是 cuBLASLt 上的 BF16 GEMM，0.6 ms 是注意力；action 阶段 7.8 ms，其中 2.6 ms 是走 xcompute 通用路径的 360 个 gate/up 和 QKV GEMM（一个 SM80 WMMA kernel，每次 7.2 µs），1.6 ms 是 360 个 small-M 输出投影和 down 投影，2.0 ms 是注意力，0.3 ms 是融合的归一化。参考组合中，action 阶段的 740 个批次不变 GEMM 耗时 4.6 ms。

### 逐算子耗时与下限

用 `xinfer-profile ops` 测得（[性能剖析](../profiling.zh-CN.md#算子剖析xinfer-profile-ops)），所有模型页面使用同一方法：

```bash
xinfer-profile ops --model wall_oss --checkpoint <checkpoint> --requests <requests.msgpack> --out <directory>
```

RTX 5090（驱动 595.91.07，功率上限 575 W，运行结束时 SM 时钟 2902 MHz，最高 3090 MHz），torch 2.10.0+cu130（CUDA 13.0），XInfer `20358f4`，xcompute `651195d`。请求：[参考算子模型](#参考算子模型)检查的 7 个请求（服务抓包 `ref-v1`：3 个 flow，4 个延迟 3–8 的 RTC），合成一个请求文件（sha256 `0eb0db3e…`），prompt 桶 768。峰值：`PEAKS` 表（[性能剖析](../profiling.zh-CN.md#roofline-下限)：BF16 220.1 TFLOP/s，DRAM 1522 GB/s）。证据包：`xinfer-fa2-causal-20261108-r3.tar.gz`（sha256 `4e520a26c98b8e5b6434593db6b95f231202b36a55a2e7b2d5803d185b6307eb`）。

均为每请求的值。每个请求先运行两次预热，然后全部在剖析器下以 eager 模式运行：算子时间是这次运行的 kernel 时间，按算子调用归属；graph 回放时间是相同请求回放其 CUDA graph（即服务路径）时该阶段的 GPU 区间。占比是占该阶段 eager GPU 时间的比例；下限是单次调用的 roofline 下限，效率为下限 / 时间；`generic` 标记只按数据搬运计代价的算子。访存受限的调用效率超过 100 % 时，其数据仍在 L2 中。

**Vision**：graph 回放 16.34 ms；算子 kernel 16.03 ms。

| 算子 | 调用形状 | 调用次数 | µs/次 | 占比 | 下限 µs | 效率 | 限制项 |
|---|---|---|---|---|---|---|---|
| `gemm_glu_epilogue` | `x[2112,1280] weight[6912,1280]` | 32 | 219.9 | 43.8 % | 169.8 | 77 % | 计算 |
| `qkv_rope_epilogue` | `x[2112,1280] weight[3840,1280]` | 32 | 121.1 | 24.1 % | 94.3 | 78 % | 计算 |
| `gemm_epilogue` | `x[2112,3456] weight[1280,3456]` | 32 | 85.8 | 17.1 % | 84.9 | 99 % | 计算 |
| `gemm_epilogue` | `x[2112,1280] weight[1280,1280]` | 32 | 34.0 | 6.8 % | 31.4 | 92 % | 计算 |
| `varlen_attention` | `q[2112,16,80] k[2112,16,80]`，36 段 | 28 | 13.7 | 2.4 % | 14.2 | 104 % | 访存 |
| `add_rms_norm` | `x[2112,1280] residual[2112,1280]` | 64 | 5.5 | 2.2 % | 14.2 | 257 % | 访存（generic） |
| `varlen_attention` | `q[2112,16,80] k[2112,16,80]`，3 段 | 4 | 61.8 | 1.5 % | 35.2 | 57 % | 计算 |
| `gemm_epilogue` | `x[528,5120] weight[5120,5120]` | 1 | 191.5 | 1.2 % | 125.8 | 66 % | 计算 |
| 其余：3 个其他调用形状和算子外的活动 | | | | 1.0 % | | | |

**Prefill**：graph 回放 22.30 ms；算子 kernel 22.69 ms。

| 算子 | 调用形状 | 调用次数 | µs/次 | 占比 | 下限 µs | 效率 | 限制项 |
|---|---|---|---|---|---|---|---|
| `gemm_glu_epilogue` | `x[768,2048] weight[22016,2048]` | 35 | 316.9 | 48.8 % | 314.7 | 99 % | 计算 |
| `gemm_epilogue` | `x[768,11008] weight[2048,11008]` | 35 | 182.0 | 28.0 % | 157.3 | 86 % | 计算 |
| `qkv_rope_kvcache` | `x[768,2048] weight[2560,2048]` | 36 | 60.8 | 9.6 % | 36.6 | 60 % | 计算 |
| `gemm_epilogue` | `x[768,2048] weight[2048,2048]` | 35 | 49.4 | 7.6 % | 29.3 | 59 % | 计算 |
| `causal_attention` | `q[1,768,16,128] k[1,768,2,128]` | 35 | 29.6 | 4.6 % | 11.0 | 37 % | 计算 |
| `add_rms_norm` | `x[1,768,2048] residual[1,768,2048]` | 70 | 4.0 | 1.2 % | 8.3 | 208 % | 访存（generic） |
| `rms_norm_cast` | `x[1,768,2048] weight[2048]` | 1 | 3.0 | 0.0 % | 4.1 | 140 % | 访存（generic） |
| 算子外的活动 | | | | 0.2 % | | | |

**Action**：graph 回放 7.78 ms；算子 kernel 7.57 ms。

| 算子 | 调用形状 | 调用次数 | µs/次 | 占比 | 下限 µs | 效率 | 限制项 |
|---|---|---|---|---|---|---|---|
| `qkv_rope_kvcache` | `x[16,1024] weight[2560,1024]` | 180 | 12.9 | 30.1 % | 3.5 | 27 % | 访存 |
| `gemm_epilogue` | `x[16,2048] weight[1024,2048]` | 360 | 4.7 | 21.8 % | 2.8 | 60 % | 访存 |
| `gemm_glu_epilogue` | `x[16,1024] weight[4096,1024]` | 180 | 8.7 | 20.4 % | 5.6 | 64 % | 访存 |
| `prefix_kvcache_attention` | `q[1,16,16,128] k_cache[1,784,2,128]` | 180 | 7.8 | 18.3 % | 0.6 | 8 % | 访存 |
| `add_rms_norm` | `x[1,16,1024] residual[1,16,1024]` | 360 | 1.2 | 5.4 % | 0.1 | 8 % | 访存（generic） |
| `linear` | `x[1,16,2048] weight[1024,2048]` | 5 | 17.6 | 1.1 % | 5.6 | 32 % | 访存 |
| `linear` | `x[1,16,1024] weight[1024,1024]` | 5 | 9.4 | 0.6 % | 2.8 | 30 % | 访存 |
| `linear` | `x[1,16,52] weight[1024,52]` | 5 | 2.9 | 0.2 % | 0.2 | 6 % | 访存 |
| 其余：10 个其他调用形状和算子外的活动 | | | | 2.1 % | | | |

**时间花在哪里：**
- **Vision**（16.3 ms）：带 epilogue 的 GEMM 占该阶段的 93 %，达到其下限的 66–99 %；窗口注意力和全局注意力共占 4 %。
- **Prefill**（22.3 ms）：GEMM 达到 59–99 %；因果注意力达到其下限的 37 %，占该阶段的 5 %。
- **Action**（7.8 ms）：16 行时所有算子都受访存限制；带 cache 写入的 QKV 投影只达到其下限的 27 %，cache 注意力为 8 %，两者合计占该阶段的 48 %。

## 测试

| 测试 | 依赖 | 检查内容 |
|---|---|---|
| `tests/models/test_wall_oss_tiny.py` | CPU | 两种组合（optimized 组合运行在其算子的 torch 参考实现上）：权重映射完整性、引擎运行、精确的 RTC 固定、确定性、填充的 prompt；optimized 组合与参考组合相差在 5 % 以内；阶段拆分 = 联合前向；prompt 填充布局及其对算子的要求；使用 `image_preprocess_device="gpu"` 时视角图像是 `DeviceImage` |
| `tests/models/test_image_preprocess.py` | GPU、torchvision、checkpoint、服务抓包 | `image_preprocess_device="gpu"`：设备端视角图像与 CPU 视角图像的差异在规定容差内（原始帧：平均 ≤ 0.01，最大 ≤ 1 级；抓包中的均匀噪声腕部视图编码为 JPEG：平均 ≤ 8，最大 ≤ 96）；由用 CPU 准备的请求预热的 graph 提供服务，预热后无捕获 |
| `tests/models/test_wall_oss_processing.py` | checkpoint、服务抓包 | 处理一致性，逐位一致；分桶的 prompt = 精确 prompt 加填充 |
| `tests/unit/test_image_pack.py` | CPU；GPU | 视角图像归一化（逐通道和共享查找表）以及 vision 阶段的打包与 NumPy 计算结果一致；在 CUDA 上以及用新像素做 CUDA graph 回放时同样一致 |
| `tests/unit/test_model_processing.py` | CPU | 视角图像按相机顺序并发处理，错误会传到调用方；`SegmentTokenizer` 在一个小型字节级 BPE 上与 tokenizer 一致；JPEG 帧头（baseline、progressive）以及把视角图像带到引擎的 `DeviceImage` 值 |
| `tests/models/test_wall_oss_parity.py` | GPU、checkpoint、两个参考 | 参考组合：prefill 和 action 阶段逐位一致；链路比 MXFP8 更接近；CUDA graph 与 eager 逐位一致；多通道并发请求与单请求一致；按服务配置（flash-attn、批次不变的 action 阶段、prompt 桶）：比 MXFP8 更接近，组批的 action 阶段与单请求一致，填充的 prompt 与精确 prompt 一致，预热覆盖所有姿态 |
| `tests/models/test_wall_oss_optimized.py` | GPU、xcompute、checkpoint、两个参考 | optimized 组合：从 eager BF16 模型的输入出发，各阶段相差在 5 % 以内；链路比 MXFP8 更接近；graph 回放与 eager 一致 |
| `tests/unit/test_fused_ops.py` | CPU；带 xcompute 的 GPU | 融合算子的参考实现对照未融合的 torch；xcompute 的 kernel 在本模型形状下对照参考实现（归一化、写入 K/V 的带 bias GQA QKV、带 bias 的 ViT GLU、带 prompt 填充和 RTC 列的缓存注意力） |
| `tests/unit/test_cudagraph.py` | GPU | graph 键、回放 = eager、预热后未命中策略、LRU 和静态缓冲区增长；预热覆盖每个通道和批大小（包括混合长度的批）；某个结果在其通道正被捕获时被释放；`DeviceImage` 输入一直留在 host 内存中，直到设备图像线程在自己的 stream 上解码它们，任意长度的 JPEG 都回放同一个 graph |
| `tests/unit/test_engine.py` | CPU | 调度器：组批、兼容性、不变性门控、准入、取消、失败；设备图像线程（另一个请求的解码被挂起时，不含设备图像的请求照常得到服务；准入顺序；解码失败只导致其所在请求失败；无 CUDA 设备） |
| `tests/unit/test_attention_ops.py` | CPU；装有 flash-attn 的 GPU | attention 参考实现（因果、RTC 固定前缀）；flash-attn vs 参考实现、批/行不变性、graph 捕获 |
| `tests/unit/test_batch_invariant.py` | 装有 Triton 的 GPU | 批次不变 linear / RMSNorm vs 参考实现、行不变性、graph 捕获；融合旋转与参考实现逐位一致，其 SASS 中没有 FFMA |
| `tests/unit/test_mxfp8.py` | CPU；设置了 `XINFER_XCOMPUTE_LIBRARY` 的 GPU | MXFP8 参考格式和融合参考实现；可融合的层形式（BF16 = eager 组合）；xcompute 算子 vs 参考实现、CUTLASS 行不变性、graph 捕获 |
| `tests/unit/test_shm.py` | CPU | 张量和 `DeviceImage` 数据通过共享内存从前端传入，图像尺寸记录在清单（manifest）中 |
| `tests/unit/test_serve.py` | CPU | 引擎 + 2 个前端进程 + WebSocket 客户端：元数据、并发、错误、健康检查、干净关闭 |

本模型的 GPU 测试在 RTX 5090 上通过，每个测试模块一个进程，配置了全部资源、xcompute `651195d`、flash-attn、Triton 和 torchvision（2026-11-07，torch 2.10.0+cu130；`xinfer-wall-oss-optimized-20261107.tar.gz`）：`test_wall_oss_optimized` 2 个、`test_wall_oss_parity` 8 个、`test_image_preprocess` 4 个、`test_fused_ops` 连同极小模型和服务测试共 61 个。

资产：`XINFER_WALL_OSS_CHECKPOINT`、`XINFER_WALL_OSS_REFERENCE`（包含 `requests.pkl` 和 `reference.npz` 的目录 = `ref-v1` 抓包）以及 `XINFER_WALL_OSS_BF16_REFERENCE`（`bf16_reference.npz/.json`）。

## 证据

每个证据包都包含被测源码和生成其结果的脚本；添加某项结果的 merge request 记录了对应证据包的存放位置。

| 证据包（sha256） | 本页中的结果 |
|---|---|
| `xinfer-per-op-tables-20261107.tar.gz`（`1409a222310efad9328ef3c50ec5feacb805e9731c37a117225fc511a98e55ad`） | 逐算子表 |
| `xinfer-wall-oss-optimized-20261107.tar.gz`（`aebb58f9ffd571d4947a6953cb2a534c2c58776d5ca5060064c125b6122a7a38`） | optimized 组合：阶段误差、延迟和 kernel 剖析、quick100 B1、GPU 测试 |
| `xinfer-torch210-v3-20261107.tar.gz`（`79579b493a0b1c89bfaeaf8985d4f6e737d4b1104afea58783a8c6fc71f242a0`） | 同一棵树上 torch 2.10.0+cu130 与 2.9.1+cu130 的对比：重新生成的 eager BF16 参考及其差异、GPU 测试套件、18 个请求的动作、单请求时延、一次 8 台压测 |
| `xinfer-gpu-image-preprocess-v3-20261106.tar.gz`（`85d80bc1c7db8b3ec1ad98c983d435c8e470a4a80e43e4d700a8722bf6e94852`） | 使用 `DeviceImage` 输入的 `image_preprocess_device="gpu"`：42 个请求的动作和视角图像与下一个证据包中所测代码的结果逐位一致，默认路径与 main 逐位一致，每个模型一次压测，GPU 测试套件 |
| `xinfer-gpu-image-preprocess-v2-20261105.tar.gz`（`9ec8364a0d1eb0e7673e5c96c104bf88cb26cfe31f9f98df8ba331b43bebae67`） | `image_preprocess_device="gpu"`：视角图像 vs CPU 路径 |
| `xinfer-prompt-buckets-20261104.tar.gz`（`2364bedfc2e2eca3715c86b4a4d230fc47beb14261c54bff882d0738a5c5e61f`） | 填充 vs 精确 prompt、prefill 长度扫描、GPU 测试套件、使用填充 prompt 的 quick100 |
| `xinfer-engine-fixes-v3-20261103.tar.gz`（`bfd95a27eed09d4e373f1af2ad6f81f4dedd47878bb2db27954b1b4cf4785726`） | GPU 测试套件和被测源码的源码归档、无捕获的服务检查 |
| `xinfer-config-defaults-20261103.tar.gz`（`6f85d5eb17df2d16cf7d4d583c0a06b10e2eee255f6b544eb639b42c698f43d2`） | 配置默认值的 GPU 测试和容量压测（早先的 MXFP8 组合；docs/configuration.md 引用） |
| `xinfer-consolidated-20261102.tar.gz`（`5723c59bc59582d463d4c553557c2f16513787e0db3323069fd5eab23acd35e5`） | 2026-11-02 各代码树的 GPU 测试套件和源码归档 |
| `xinfer-rotary-20261102.tar.gz`（`adf681a9059e532e912493e698b00b6c95c76d2e08a735cfb08e331bbc601959`） | 旋转检查、42 个请求上的 graph 回放 |
| `xinfer-action-batching-20261101.tar.gz`（`73d40300f6b1fd994506f8184dcd513b053c51b4d83cbd125a9fca666ff4475f`） | 组批 vs 单请求、MSE 完全相同的较早 quick100 运行 |
| `xinfer-flash-attn-20261101.tar.gz`（`859386445082b71360f25e4501ed9609f3b6d2ee24c254319a360efb72b03eb7`） | attention 后端对比和 action 通道压测（早先的 MXFP8 组合；docs/configuration.md 引用） |
| `xinfer-scheduler-20261101.tar.gz`（`24b2fabb8afd789dbf41165393e72f259cec057f05b7b9448d3d034f05149d29`） | 多通道并发请求 |
| `xinfer-frontend-20261101.tar.gz`（`e6edc1da59652e55615cdfcec5b44852568acd4e0ca1661db69659e0558bb976`） | 传输协议兼容性 |
| `xinfer-wallx4b-20261101.tar.gz`（`4266abc3332908604af082c361872baabe77674e8596c013086279c21054291c`） | 处理一致性、参考算子模型检查、eager BF16 参考输出 |

服务抓包 `ref-v1`（重建版）与其他测试资产一起保存（`XINFER_WALL_OSS_REFERENCE`）。

## 未验证

- optimized 组合使用 prompt 桶时的预热覆盖。
- optimized 组合的 quick100 组批场景（其 action 阶段每次派发一个请求）。
- MXFP8 和 NVFP4。
- JPEG 编码视角图像的处理一致性：抓包中保存的是原始数组。压测会解码 JPEG 视角图像，但不做一致性检查。
- `image_preprocess_device="gpu"` 在离线门禁之外的表现：闭环行为。
- Thor（SM110）。
- 闭环任务质量；本页涵盖数值等价性、quick100 离线门禁和速度。
