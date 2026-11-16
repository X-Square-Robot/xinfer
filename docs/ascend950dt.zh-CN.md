# 昇腾 950DT

[English](ascend950dt.md) | 简体中文

## 运行环境

安装相互匹配的 PyTorch 和 TorchNPU，并在导入 TorchNPU 前加载 CANN 环境。使用 `--device npu:0` 或
`Policy(device="npu:0")` 选择设备。设置 `ASCEND_RT_VISIBLE_DEVICES` 后，编号相对于可见卡集合。
平台后端名称是 `ascend`，torch 设备串是 `npu:N`。

Wall-OSS 沿用原有 checkpoint 和请求格式。服务默认在 CPU 上处理图像。PyACL DVPP V2 可用时，用顶层模型
选项 `--model-options '{"image_preprocess_device":"gpu"}'` 或 `Policy(image_preprocess_device="gpu")`
选择设备图像处理。平台支持时，`Policy` 默认使用设备路径。不要把该选项放入 `processing`。
不支持的 JPEG 格式（包括渐进式 JPEG）会记录告警，再由主机解码、设备缩放。精度和性能结果应记录图像路径。

JPEG 解码器位于各平台模块。`platforms.devices.jpeg_decoder_for(device)` 按需选择解码器，接口为
`device`、`__call__(jpegs)` 和 `close()`。平台负责厂商资源与流依赖；引擎保留输入映射、上传原始帧，
并对原始帧和解码后的图像应用相同的 resize 步骤。

## 外部 xcompute 后端

安装预构建的昇腾 xcompute 共享库和同一构建的 Python 包。加载器 `xinfer/ops/backends/_xcompute.py`
关闭导入时重建，只服务昇腾后端的外部构建：CUDA 后端的内核库从本仓库构建（`cuda/_native.py`，见
[内核库](kernels.zh-CN.md)），不经此路径。昇腾外部构建的配置：

| 环境变量 | 含义 |
|---|---|
| `XINFER_XCOMPUTE_LIBRARY` | 可选的 `xcompute.so` 路径；未设置时，由包定位动态库 |
| `XINFER_XCOMPUTE` | `0` 关闭 xcompute 实现 |

```bash
export PYTHONPATH=/path/to/xcompute-release/python:$PYTHONPATH
export XINFER_XCOMPUTE_LIBRARY=/path/to/xcompute-release/xcompute.so
```

支持的实现会自动选中。动态库或入口缺失时，算子使用下一个可用实现。`--op-backends gemm_epilogue=ascend`
固定实现；没有匹配注册时，该设置报错。只支持部分调用的实现按调用键缓存能力判断，不支持的调用走 torch 回退。
原生执行异常会继续向调用方传播。服务启动选择表或 `Policy.describe()["ops"]` 记录选中的后端、库和版本。
选择结果不代表每种调用形状都使用原生算子。

xcompute 使用 `-DBUILD_ASCEND_CANN_GDR=ON -DBUILD_ASCEND_CANN_INFERENCE=ON` 构建可选的 CANN 桥接。
workspace 和 FP32 epilogue 由外部包负责。已验证环境使用 CANN 9.1.1 和 TorchNPU 2.10.0.post4。

### 遵守算子契约的导出

配套的 Ascend xcompute 包可以在模块命名空间中直接导出下表八个算子名中的任意一项，提供对应的 Python
可调用对象。XInfer 优先绑定这些导出，再考虑既有 CANN/旧版路由。例如，`xcompute.gemm_epilogue` 实现
XInfer 的 `gemm_epilogue`，既有 `xcompute.gemm` 则保留自身接口。不提供这些导出的包保持既有路由。
提供全部八项导出时，XInfer 不再导入 CANN 桥接；部分提供时，其他算子继续使用桥接或 torch 回退。

导出算子名即声明遵守以下契约：

- **签名。** 参数及位置/关键字种类须与算子在
  [`gemm.py`](../xinfer/ops/reference/gemm.py)、[`attn.py`](../xinfer/ops/reference/attn.py) 或
  [`norm.py`](../xinfer/ops/reference/norm.py) 中的参考实现相同。加载时检查参数名与种类。需要 output/workspace
  参数的裸 FFI 函数应由 xcompute 提供 Python 适配层；发现算子导出时不使用模块动态属性查找。
  XInfer 按自身默认值显式传入每个可选参数，因此 provider 的默认值不会影响这些调用。
- **数值。** 保留参考实现的累加类型及 BF16 舍入位置，包括在类型转换前执行 FP32 GEMM epilogue 和 RoPE、
  GLU 使用 FP32 AdaRMS 输入，以及先得到 BF16 残差和再计算 FP32 RMSNorm。实现必须通过算子一致性测试与模型
  精度门禁；存在导出本身不代表已通过数值验证。
- **返回与修改。** 保留输出形状、类型及元组顺序。`prefix_kvcache_attention` 返回 `[B, M, H*D]`；
  `add_rms_norm` 返回 `(sum, normalized)`。`qkv_rope_kvcache` 只就地更新契约规定的 K/V cache 切片并返回 Q。
  其他输入保持不变。上述参考实现规定完整的形状和所有权契约。
- **执行。** 在当前流上异步执行，同时支持 eager 执行及图捕获/回放。可复用 workspace 应在首次 eager 调用时
  准备；输出张量允许使用捕获安全的分配器。捕获期间不得进行主机同步，或未跟踪、不安全的分配。
  缓冲区须存活到排队工作完成，可变 workspace 须在并发流/通道之间隔离。
- **部分支持。** 对不支持的布局或选项，必须在下发工作或修改输入前抛出 `ValueError`。支持判断只能依赖缓存的
  调用键：张量元数据、标量选项，以及 varlen 注意力的分段边界。XInfer 缓存首次 eager 判断，对拒绝的键使用该
  算子的 torch 回退。执行故障向调用方传播；已准入键后续失败时不会改用 torch 重试。

保持该契约的内核替换或支持形状扩展，只需要发布配套的 xcompute 包与动态库。通过既有共享加载器绑定、重启，
并重跑 eager/图执行的算子检查、通道冒烟、精度与时延基准。新增算子语义或签名须单独修改算子契约。

### Wall-OSS 算子清单

默认接线下，除不做任何事的 `prefetch_weights` 和 `join_prefetch` 外，Wall-OSS 调用的每一个算子都选择
`ascend:xcompute`：共 11 项算子绑定，另有 2 个算子依次调用它们，且对完整请求通过的
逐 layout 分发审计确认全部调用 layout 准入（没有任何算子运行在 torch 参考上）。这不代表 11 个独立的
自定义内核；下表列出实测构建中的实际路由与计算实现。可用性取决于配套的包和动态库。实现须同时支持
eager 执行和图捕获。旧的仅支持 eager 的裸 GEMM、GLU 和 varlen 入口不注册。
缺少遵守算子契约的导出或 CANN 桥接时，这些算子在两种模式下均使用对应的 torch 实现。

| 算子 | 阶段 | 实测构建中的路由 | 计算实现 |
|---|---|---|---|
| `gemm_epilogue` | vision、prefill、action | `CannProjection.gemm_epilogue` | CANN Mm + Torch epilogue |
| `gemm_glu_epilogue` | vision、prefill、action | `CannProjection.gemm_glu_epilogue` | CANN Mm + Torch FP32 SiLU-GLU |
| `qkv_rope_epilogue` | vision | `CannProjection.qkv_rope_epilogue` | CANN Mm + Torch bias/RoPE |
| `qkv_rope_kvcache` | prefill、action | `xcompute.qkv_proj_rope` | 统一投影入口配 K/V cache epilogue；CANN action 桥是无该入口构建的回退 |
| `varlen_attention` | vision | `CannAttention.varlen_attention` | CANN FIA V5（TND） |
| `causal_attention` | prefill | `CannAttention.causal_attention` | CANN FIA V5（BNSD）+ Torch mask/布局处理 |
| `prefix_kvcache_attention` | action | `CannAttention.prefix_kvcache_attention` | CANN FIA V5（BNSD）+ Torch mask/布局处理 |
| `add_rms_norm` | vision、prefill、action | `xcompute.add_rmsnorm` | 自定义 AOT 内核；包入口服务任意行数，并直接接受 vision 存储的 BF16 权重 |
| `gemm_add_rms_norm`、`mlp_add_rms_norm` | vision；prefill | 依次调用上面的 `gemm_epilogue`、`gemm_glu_epilogue` 和 `add_rms_norm` 绑定 | 与这些算子的调用相同 |
| `rms_norm` | vision、prefill、action | `xcompute.rmsnorm` | 自定义 AOT 内核；三个流的输入归一化 |
| `linear` | action | `xcompute.gemm` | 统一 GEMM 入口（这些形状走 CANN Mm）；其算子要求秩匹配，因此压平前导维 |
| `flow_velocity_euler` | action | `xcompute.denoise_output_euler` | 自定义 AOT 内核；auto 表不收的包络用命名 `ascend_tilelang` 域重试 |

此处 CANN Mm 指 ACLNN Matmul，FIA V5 指 CANN FusedInferAttentionScore V5。自定义 QKV bias/RoPE/cache
epilogue 仅支持 action 的 16 token、hidden width 1024、16 个 query head、2 个 KV head、head width 128、
cache 写入位置 768，以及符合条件的连续缓冲区。其他 prompt 宽度与 prefill 使用 Torch/TorchNPU epilogue，
两种情况的矩阵乘均由 CANN 执行。时延记录未包含条件内核的命中次数，也未给出 CANN、自定义内核和参考算子
各自的耗时占比。动态库导出了其他入口，不代表实测模型选择了这些入口。

`prefetch_weights` 和 `join_prefetch`（action）在 NPU 上不做任何事（使用参考实现）。名为 `qkv_rope_epilogue` 的 xcompute 导出须接受算子的 `qk_order` 关键字参数；标准顺序对两个取值都适用。

Wall-OSS action 使用普通 RMSNorm、带 bias 的 QKV 和无调制的 GLU。
切换动态库后，应使用相同请求和图像路径重跑[精度](precision.zh-CN.md)与
[性能](performance.zh-CN.md#昇腾-950dt)验证。

### Wall-X 2B 算子清单

Wall-X 2B 在三个阶段共声明 21 个不同算子，默认接线下全部 21 个都选择 `ascend:xcompute`：对完整请求通过的
逐 layout 分发审计确认 21/21、全部调用 layout 准入，没有任何算子运行在 torch 参考上。这些绑定都不含本仓库
手写的内核。下表只列出其中 18 个，另外三个 `gemm_epilogue`、`gemm_glu_epilogue`、`qkv_rope_epilogue` 与
上方 Wall-OSS 表所描述的是同一批入口，重复列出只会重复路由、而不是重复算子。

| 算子 | 阶段 | 入口 | 说明 |
|---|---|---|---|
| `layer_norm_cast` | vision | `xcompute.layernorm_inplace` | ViT 的 layer norm；该入口会覆写首参，因此绑定层先 clone——调用方还要把输入继续当残差用 |
| `segment_attention` | vision | `xcompute.attention_varlen` | 打包分段、无门；宿主分段布局配 `max_seqlen`，按网格列表规划并缓存 |
| `rot_pos` | vision | `rot_pos` / `get_token_counts` 库函数 | 2 维旋转角度；按 (grids, merge, device) 缓存宿主规划的网格张量，与 CUDA 后端的绑定方式一致 |
| `pos_embed_interp` | vision | `pos_embed_interp` 库函数 | 位置表的双线性插值，走同一批缓存网格张量 |
| `rope_full` | prefill | `xcompute.Rope.full` | 半切分局部 RoPE（`interleave=False`） |
| `zero_centered_rms_norm` | prefill | `xcompute.Rmsnorm(zero_centered=True)` | 按 `1 + w` 缩放，即 checkpoint 的原始权重 |
| `add_zero_centered_rms_norm` | prefill | `xcompute.AddRmsnorm(zero_centered=True)` | 返回 `(sum, normalized)`，沿用入口自身的顺序 |
| `linear` | prefill | `xcompute.gemm` | 统一 GEMM 入口；其算子要求秩匹配，因此压平前导维 |
| `qkv_norm_split` | prefill | `qwen35_vla_qkv_norm_replace_forward` 库函数 | 每一行都是文本行；切分即内核输出，零中心归一化由内核自身完成 |
| `gated_causal_attention` | prefill | `xcompute.attention_varlen(causal=True, out_gate=…)`，否则 CANN 注意力桥自带的门控组合 | 分组的 key/value 先展开到 query 头数，与参考实现的 helper 一致；桥路径承接打包入口白名单没有的头型 |
| `gated_delta_net` | prefill | `xcompute.CannGatedDeltaNet` | 参数与顺序完全一致；入口接受 batch 1、16 个 K head、head 宽 128 |
| `qkv_norm_rope_kvcache` | action | `xcompute.qkv_proj_rope` | 统一投影入口配 `QkNorm(zero_centered=True)` 与 K/V cache epilogue；本构建的 workspace 按流自带 |
| `modulated_rms_norm` | action | `xcompute.Adarmsnorm` | 调制表是 FP32 的 `[B, 3K]` |
| `gated_residual_modulated_rms_norm` | action | `xcompute.Adarmsnorm(residual=…, branch_gate=…)` | 同一入口在这种形态下返回 `(output, gate, residual_output)` |
| `identity_layer_norms_` | action | `xcompute.double_adarmsnorm_gated_residual_inplace` | 原地覆写 `residual` 与 `branch`，返回 post norm 的 gate |
| `gated_kvcache_attention` | action | `prefix_kvcache_attention` + BF16 门 | 参考实现的两步，其中 cache attention 走原生 |
| `flow_input_embed` | action | `xcompute.denoise_input_embed` | 流头的 FP32 激活与 3 维 weight；auto 表不收的包络用命名 `ascend_tilelang` 域重试 |
| `flow_velocity_euler` | action | `xcompute.denoise_output_euler` | 该入口的 `dt` 是仅关键字参数；A-26 动作宽度在 auto 表实测包络之外，由命名 `ascend_tilelang` 域覆盖 |

官方算子与 CANN 桥的注册位于 `backends/ascend/official.py` 与 `backends/ascend/cann.py`，是后端仅有的
直接调用 TorchNPU 自身算子或 CANN 桥（而非 xcompute 入口）的地方；两者的优先级都低于 xcompute 入口，
不显式 pin 就不会被选中。

四个注意力算子另有 graph-safe 的 torch 组合（`backends/ascend/_torch_attn.py`），白名单之外的布局在图捕获下
仍然可用，而不会退到无法捕获的 SDPA。两侧遵循与 Wall-OSS 相同的规则：先查选择器再调用，入口拒绝的布局留在
分部支持路径上，而不是变成运行时错误。

### CANN 官方算子（可选）

`xinfer/ops/backends/ascend/official.py` 与 `xinfer/ops/backends/ascend/cann.py` 让 Wall-X 2B 的算子改由
TorchNPU 自带的 CANN 入口承接。两个模块的注册优先级都低于 xcompute 入口，因此**默认一个都不会被选中**：
部署跑的是 xcompute 入口，只有显式指定时才会用到它们。

| 算子 | CANN 入口 | 说明 |
|---|---|---|
| `linear` | `torch_npu.npu_linear` | 展平前导维度 |
| `qkv_norm_split` | `torch_npu.npu_rms_norm` | 切分是视图；零中心归一化即官方 RMSNorm 配 `1 + w` |
| `gemm_epilogue` | `torch_npu.npu_linear` + FP32 epilogue | epilogue 保持在 FP32，与参考的舍入一致 |
| `gemm_glu_epilogue` | `torch_npu.npu_linear` + `silu·mul` | 输入的 AdaRMS 归一化沿用参考实现 |
| `qkv_norm_rope_kvcache` | `torch_npu.npu_linear` + 参考的 norm/rope 步骤 | 直接吃 BF16 权重，不必每次转换 |
| `zero_centered_rms_norm` | `torch_npu.npu_rms_norm` 配 `1 + w` | checkpoint 存的是原始权重 |
| `layer_norm_cast` | `F.layer_norm` | 一次 ACLNN 调用 |

用 pin 指定，例如：

```bash
xinfer-serve --model wallx2b --checkpoint /path --device npu:0 \
    --op-backends gemm_epilogue=ascend:npu_linear
```

[性能表](performance.zh-CN.md)报告的是默认 xcompute 接线；这些 pin 是偏好官方算子的部署的替代选择。

## 图执行

安装的 TorchNPU 提供图 API 时，默认开启图捕获，与 CUDA 相同。
请求首次遇到新形状时按需捕获；也可通过代表性的 `--warmup` 请求，在服务前捕获可达的形状分桶、通道和批量组合。
每次捕获前，引擎先在捕获流上对该形状运行一次 eager，再捕获固定缓冲区。预热后，新形状遵循
`CUDAGraphConfig.on_miss_after_warmup`：`"capture"`（默认）捕获一次，后续同形状请求回放；`"eager"`
不捕获，直接执行；`"error"` 拒绝该请求。`--enforce-eager` 或 `Policy(cudagraphs=False)` 关闭图执行。

没有原生实现支持某次调用时，`ascend:torch` 注意力在 eager 和捕获中使用相同的 FP32 matmul、softmax 和 matmul
组合，避开可能导致捕获无法结束的 SDPA 旁路流。CPU/CUDA 和显式参考算子表保留原始 SDPA 参考实现。
不指定原生算子绑定时，`--reference-ops` 选择原始参考实现；在 NPU 上应配合
`--enforce-eager`，或使用 `Policy(reference_ops=True, cudagraphs=False)`，因为参考注意力可能不支持捕获。

## 流优先级

引擎在 CUDA 和 NPU 上均通过同一 `Stream(device=..., priority=...)` 接口传入 `priority`，
依据当前运行时的流优先级能力及其报告的数值范围处理设置。范围已知时，将最终阶段偏好的优先级 −1
限制在该范围内：范围为 0–7 的运行时保持最终阶段为 0。范围未知时，所有默认值保持为 0。显式请求不支持
或越界的值会报错；声明支持优先级的运行时修改了请求的优先级，或无法读回实际优先级，也会导致通道创建失败。

TorchNPU 2.10.0.post4 和 2.10.0.post7 均未实现优先级范围查询及读回，且其托管流池创建的原生流优先级均为 0。
因此，这些运行时不声明流优先级能力，并拒绝非零请求。CANN 支持原生流优先级，并不能证明 TorchNPU
遵守其构造参数；参见 [post4 流绑定](https://github.com/Ascend/pytorch/blob/5dd8ef3f9b375b5ae4a83538d5785754148c3302/torch_npu/csrc/npu/Stream.cpp#L90-L100)、
[post7 流绑定](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/csrc/npu/Stream.cpp#L103-L113)
和 [post7 流池创建](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/csrc/core/npu/NPUStream.cpp#L254-L268)。
post7 的 `ExternalStream` 包装不能替代通道流：其文档明确限制事件记录/等待、分配器流跟踪和图捕获；
参见[上游契约](https://github.com/Ascend/pytorch/blob/3363ea59d49f3e08cfd8d4a1f532b756ccda0682/torch_npu/npu/streams.py#L112-L177)。

## 验证

测试进程只暴露一张已独占预留的卡。使用相同的准备后输入和显式噪声比较 eager 与图执行，再通过 WebSocket
服务测试真实请求。检查无预热的冷请求和预热后的新形状能按需捕获，之后同形状请求能够回放。运行[精度门禁](precision.zh-CN.md)和[基准测试](benchmark.zh-CN.md)，记录源码和动态库
身份、计算组合、算子选择、图像路径、checkpoint 身份与图计数。加速器 trace 和实验保留在推理包之外。
测量请求延迟和机器人容量时关闭 profiler。

通过 `benchmarks/performance.py --device npu:0` 选择 NPU；图像设置放入原有的 `--model-options`。
单独比较算子后端或 eager 执行时，使用 `Policy` 或 `xinfer-serve` 的现有选项。

经验证的构建还通过了 24 次并发请求测试：8 组固定输入按 3 种顺序提交，输出与顺序执行逐位一致，
每个阶段的两条通道均发生了图回放。
