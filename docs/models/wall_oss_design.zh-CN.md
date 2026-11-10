# Wall-OSS：设计

[English](wall_oss_design.md) | 简体中文

导航：[用法](wall_oss.zh-CN.md) · [验证](wall_oss_validation.zh-CN.md)

## 语义来源

参考实现是 Wall-X `02c5351d`（加上 embodied-serving 独立补丁 `1e24f619`）中 eager `Qwen2_5_VLMoEForAction` 的 flow 路径；请求处理部分以 embodied-serving Wall-X 4B 服务（es `9633ffd`）为参考。HarriX 对该服务的逐位移植（`qwen2_5`、`mxfp8-sm120`）是第二个来源。文件：`xinfer/models/wall_oss/`（`config.py`、`model.py`、`processing.py`、`geometry.py`）以及 `xinfer/layers/` 中的共享层。

## 阶段划分

该 checkpoint 是一个双专家的 mixture of transformers：expert 0（语言模型，宽度 2048）处理 prompt，expert 1（宽度 1024）处理 16 个 action token；每层有一个同时覆盖两者的 attention。prompt token 从不 attend 到 action token，因此 prompt 的逐层 K/V 不依赖于动作或去噪时间。所以模型可以被精确拆分：

| 阶段 | 输入 | 输出（交接） |
|---|---|---|
| `vision` | 缩放后的视角图像 `image_<i>` uint8，在阶段内打包为 `[N_patches, 1176]` FP32 patch 行；在 host 上计算的窗口布局 | `image_embeds` `[N_tokens, 2048]` BF16 |
| `prefill` | `input_ids[:, :P]`、M-RoPE `position_ids`、`image_embeds` | `prompt_kv` `[36, 2, 1, 2, P, 128]` BF16（RoPE 之后的 K、V；`P` 为填充后的 prompt 长度） |
| `action` | `prompt_kv`、噪声、`dof_mask`、填充动作、RTC 动作和延迟 | `predict_action` `[1, 16, 26]` FP32，checkpoint 单位 |

最后一个 prompt 层只计算其 K/V（它的 attention 输出和 MLP 没有使用者）。各阶段不在 host 上读取设备值：RTC 延迟和动作位置基址是张量，图像 token 按 host 上计算的索引放置，初始噪声来自输入处理器。host 值（prompt 长度、步数、RTC 动作行数、布局列表）用于选择 graph。prefill 和 action 阶段与联合前向逐位一致（见验证）。

## 数值

参考组合（`numerics="reference"`）复现 Wall-X eager 模型；[optimized 组合](wall_oss_design.zh-CN.md#optimized-组合)将其步骤融合。

- BF16 权重和激活；语言模型的 RMSNorm 权重以及 action head、时间嵌入和 ODE 状态为 FP32（与 Wall-X 的加载方式相同）。
- RMSNorm 在 FP32 下归一化和缩放，只舍入一次。
- LLM 旋转表：FP32 角度，cos/sin 舍入到 BF16；M-RoPE 频带 16/24/24。ViT 旋转表：FP32，不舍入。
- 旋转本身（`rotary_embedding`）在 FP32 下计算，并一次性舍入到激活 dtype。
- Flow：`t = linspace(0, 1, N+1) · 0.999`，Euler `x ← x + Δt·v`，速度预测，逐 token 时间（固定（pinned）的 RTC token 取 `τ = 1`），未下发指令的维度的填充速度在第 0 步固定，用按步的 `[16, 26]` 统计量反归一化。
- ViT MLP 保留 checkpoint 的 128 对齐宽度（3456），使 GEMM 形状与来源一致。

## Optimized 组合

`numerics="optimized"`（默认，`xinfer/models/wall_oss/optimized.py`）在 `xinfer/ops/fused.py` 的算子上运行参考组合的步骤：xcompute 为该形状列出 kernel 时，每个算子是一次 xcompute 调用（xcompute `7336b0d`），否则用算子的 torch 参考实现。与参考组合一样全部为 BF16。

| 步骤 | 算子 | 本模型形状下的 kernel |
|---|---|---|
| ViT 和语言模型的残差加法 + RMSNorm | `add_rms_norm`（第一个归一化用 `rms_norm`） | xcompute `add_rmsnorm` / `rmsnorm` |
| ViT QKV + bias + 2-D RoPE | `qkv_rope_epilogue` | `qkv_proj_rope` → cuBLASLt 加一个 RoPE kernel |
| ViT 注意力（窗口、全注意力块） | `varlen_attention` | flash-attention 2 |
| ViT 投影、MLP down、merger（+ bias；merger fc1 + GELU） | `gemm_epilogue` | `gemm` → cuBLASLt |
| ViT gate/up + bias + SiLU-mul | `gemm_glu_epilogue`（`bias`） | `gemm_glu` → cuBLASLt 加 SiLU-mul kernel |
| expert 0 的 QKV + bias + M-RoPE，K/V 写入 prompt K/V | `qkv_rope_kvcache` | `qkv_proj_rope`（`kv_cache`）→ cuBLASLt 加一个 RoPE kernel |
| expert 0 的因果注意力 | `causal_attention` | flash-attention 2 |
| expert 0 的输出投影和 down 投影；gate/up + SiLU-mul | `gemm_epilogue`；`gemm_glu_epilogue` | cuBLASLt；`tma_streamk_bf16_sm120` |
| 动作嵌入（`ActionHead`，FP32） | `linear`、`silu` | cuBLAS（FP32） |
| expert 1 的 QKV + bias + M-RoPE + 追加 K/V | `qkv_rope_kvcache` | `qkv_proj_rope`（`kv_cache`）→ cuBLASLt 加一个 RoPE kernel |
| expert 1 对 prompt K/V 和动作 key 的注意力 | `prefix_kvcache_attention` | `attention_kvcache` → `splitkv` |
| expert 1 的输出投影和 down 投影（`[K, N]` 权重）；gate/up + SiLU-mul | `gemm_epilogue`；`gemm_glu_epilogue` | `gemm` → `smallm_bf16_sm120`；cuBLASLt 加 SiLU-mul kernel |
| 读出、未受控维度的速度、Euler 步、固定行 | `flow_velocity_euler` | `denoise_output_euler` → `fused_tail` |

在本模型的 BF16 形状下，除 expert 0 的 gate/up 外，xcompute 没有为 ViT 和 expert 0 的 GEMM 列出比 cuBLAS 更快的 kernel；这些 GEMM 受算力限制（约占 vision 和 prefill 阶段的 85 %）。收益来自融合的残差加法与归一化、expert 0 的 gate/up kernel，以及 action 阶段的 small-M kernel 和缓存注意力。

与参考组合不同的算术：

- **融合的尾处理**：QKV 的 bias 和 RoPE、GLU 的 SiLU-mul 以及 merger 的 GELU 作用在 FP32 累加器上，只舍入一次；参考组合先把 GEMM 输出舍入。残差和照旧舍入为 BF16。
- **action 阶段**：每次派发一个请求。prompt K/V 和动作 key 共用一个头优先的缓存 `[layers, 2, 1, kv_heads, P + 16, head_dim]`；prompt 填充和 RTC 可见性（固定行能看到 prompt 和彼此）以逐列的行区间（FlashMask `LTS`/`UTE`）传入。速度保持 FP32。
- **Prompt 桶**：填充在 action 注意力中被掩码，因此填充后 prompt 的动作与精确 prompt 的动作在舍入范围内一致，但不逐位一致。

## Action 阶段组批

参考组合的 action 阶段接受 1、2、4 或 8 个请求，各请求的 prompt 长度和 RTC 延迟可以不同；optimized 组合只接受一个。组批的请求与其单独运行的结果相差在舍入级以内（设计 §6.4）：

- **算子。** action 栈与其他阶段一样使用默认选择：cuBLAS `linear`、xcompute `rms_norm` 和 flash-attn `action_attention`。cuBLAS 可能为不同行数选择不同的 kernel，因此组批的请求可能与其单独运行有舍入级差异（实测最大 3.4e-3，动作的相对范数）。
- **不同 prompt 长度。** `prompt_kv` 声明了其序列轴。组批时对它做左填充并传入 `prompt_kv_lengths`；`action_attention` 跳过每行的填充（`cache_leftpad`），与服务的做法相同。
- **填充过的 prompt** 把有效 token 放在前面（见下一节）；action 阶段在 attention 之前把每行的填充移入左填充。
- **引擎行为。** action 通道空闲时，引擎把当时排队的请求组成一批（零等待组批）。因此在负载下 action 阶段无需任何设置即会以 B2–B8 运行；`--max-batch action=N` 可限制批大小。

## Prompt 桶

状态以文本形式写入 prompt（512 个分箱，每个数字一个 token），因此每个手臂位姿都会改变 prompt 长度；200 个随机位姿产生了 18 种长度。以精确长度为键的 graph 会随流量变化，因此输入处理器把 prompt 填充到 `ProcessingConfig.prompt_bucket`（16）个 token 的整数倍：

- **位置。** 填充 token（`<|endoftext|>`，M-RoPE 位置为 0）放在 prompt 末尾、action token 之前（`processing.pad_prompt`）。`prompt_len` 变为填充后的长度 `W`，作为 prefill graph 的键；`prompt_valid`（`[1]` int32）保存有效 token 数 `P`；`action_position_base` 仍是未填充 prompt 末尾的位置。
- **Prefill。** 在因果 attention 下，没有任何有效 token 会 attend 到其后的填充 token，而 norm 和投影对每个 token 的行独立计算，因此前 `P` 列的 K/V 与未填充时的 prefill 相同。MXFP8 CUTLASS GEMM 保证这一点（固定 tile、无 split-K）。对于 cuBLAS（BF16），这是实测结果而非保证：它的 kernel 选择可能取决于行数。flash-attn 的因果调用（`num_splits=1`）对有效行在两种长度下访问相同的 key 块。
- **Action 阶段。** `prompt_kv` 到达时宽度为 `W`，前 `P` 列有效，并在引擎合并时左填充到批内最宽的行（`prompt_kv_lengths = W`）。`denoise` 每次派发时把每行循环右移 `W − P` 一次（`left_align_prompts`，对 prompt K/V 做一次 gather），使其变成 `prompt_kv_lengths = P` 的普通左填充行：这正是合并对不同长度本来就会产生的布局，`action_attention` 用 `cache_leftpad` 跳过填充。另一种做法是像 Wall-X 服务的 action 计算那样，把 action K/V 紧接在每行的有效 prompt 之后写入，这会以每层每步一次 scatter（每个请求 360 次）取代这一次 gather；还会改变 attention 算子。
- **舍入。** flash-attn 的 action attention 跳过左填充，因此填充后的请求单独运行时得到其精确 prompt 的动作。torch 参考 attention 的 softmax 和乘积覆盖所有列，组批又会改变 GEMM 的行数：两者都可能在舍入级改变比特。

在 RTX 5090 上以默认后端实测（见验证）：12 个随机机械臂姿态请求的填充 prompt，单独运行时（eager 和 graph）与其精确 prompt 的动作逐位一致，与其他桶的请求一起组批时相差在 5.9e-3 以内（相对范数）。

## Attention

模型调用三个结构化的 attention 算子，而不是一个通用的带 mask 的算子，这样后端可以为每种情况选用合适的 kernel：

| 阶段 | 算子 | 语义 |
|---|---|---|
| `vision` | `varlen_attention` | 在打包 patch 的每个窗口 / 完整段内双向 |
| `prefill` | `causal_attention` | 对 prompt 做因果 self-attention |
| `action` | `action_attention` | 每个 action token 看到整个 prompt；固定的 RTC 行（`< delay`）只看到固定的 action token，其余行看到全部 |

torch 参考实现精确复现 eager 模型的 SDPA 调用（相同的 mask 和布局）。安装 flash-attn 时，默认后端使用：
- ViT 中用 `varlen_fwd`，prompt 用 `flash_attn_with_kvcache`（`num_splits=1`），即服务所用的调用；
- action token 每个 query 一行，采用 decode 风格并固定 split 数（见后端 docstring）。服务的两次调用形式在这里比参考实现更慢。

## RTC 语义（决策）

对于 RTC 请求，固定的 action token（`k < delay`）在**每一**步都不 attend 到正在去噪的 token。这是 embodied-serving 和 HarriX 服务实际运行的方式。Wall-X 的 eager 代码只在第 0 步不做这一截断（其第 0 步的联合 mask 在 action token 之间是对称的），因此对于 RTC 请求，eager 代码与服务所用模型在语义上就不同，而不只是精度不同。XInfer 遵循服务所用的模型；用于验证的 eager 参考也做了同样的修补。flow 请求（`delay = 0`）在两种方式下完全相同。

## 处理

`WallOSSInputProcessor` 复现参考服务：OpenCV INTER_CUBIC 缩放到长边 448，然后 `smart_resize(28)`；embodied system prompt；512 分箱的状态字符串；checkpoint 的 tokenizer 加上 `<|propri|>`=151665 和 `<|action|>`=151666；在 host 上计算的 M-RoPE 位置；float64 的本体感知量和 mask；以及从绝对末端位姿转换为相对 6D 模型布局的 RTC 动作。`WallOSSOutputProcessor` 把相对动作与请求状态组合为每条手臂 17 行绝对位姿。旋转辅助函数是参考实现中 numba 函数的原样移植（`geometry.py`）：它们的 float32 特化和 libm 调用都是服务数值的一部分。

各视角图像并发解码和缩放，prompt 用缓存的固定片段做 tokenize（`xinfer.models.processing` 中的 `map_views` 和 `SegmentTokenizer`）。缩放后的视角图像以 uint8 传给引擎。vision 阶段对其打包（`xinfer.layers.image_pack.ImagePack`）：先用每个通道取值的 float32 查找表，再按 Qwen2.5-VL 14/2/2 patch 布局排列，并在两个时间槽上重复同一图像。它只做索引和拷贝，因此在 CPU 和 GPU 上、在 CUDA graph 内外，生成的行都逐位一致，而请求只需携带打包后 float32 数据八分之一的字节。

使用 `image_preprocess_device="gpu"` 时，处理器不解码也不缩放：它把每个 JPEG 视角的字节（其他编码在 CPU 上解码）或原始帧作为 `DeviceImage` 发送，附带 CPU 路径会缩放到的尺寸（从 JPEG 头读取）和处理方案 `DEVICE_RESIZE`，因此 prompt、grid 和 vision 布局不变。在第一个阶段之前，引擎用一次 torchvision `decode_jpeg` 调用解码该请求的全部 JPEG，并用 FP32 bicubic 插值（a = −0.75，无抗锯齿，与 INTER_CUBIC 相同）缩放每个视角，再按四舍五入（half up）取整为 uint8（[引擎](../engine.zh-CN.md#设备图像)）。vision 阶段读取视角图像的方式与 CPU 预处理时相同，形状也与 CPU 路径一致，因此 vision graph 及其键不变。原始帧的缩放结果与 OpenCV 相差不超过一个灰度级；解码后的 JPEG 在锐利的色度边缘附近与 libjpeg-turbo 不同，因此动作与 CPU 路径不同，该选项默认关闭。

参考部署中的选择（固定指令、相机传输键名、embodiment 文本、准入的 RTC 延迟、客户端步数覆盖）都是显式的 `ProcessingConfig` 字段。

## 已知不足

- vision 和 prefill 每次派发只处理一个请求；optimized 组合的 action 阶段也是。
- 不支持 MXFP8 和 NVFP4。在 BF16 下 ViT 和 expert 0 的 GEMM 以 cuBLAS 的速度运行（xcompute 没有为这些形状列出更快的 kernel）；action 阶段的 gate/up 和 QKV GEMM 走 xcompute 的通用路径。
- 使用参考算子时 prompt 不填充，因此这些模型每种 prompt 长度都需要一个 graph。
