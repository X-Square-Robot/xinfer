# Wall-OSS：用法

[English](wall_oss.md) | 简体中文

导航：[设计](wall_oss_design.zh-CN.md) · [验证](wall_oss_validation.zh-CN.md)

Wall-OSS 是一个 Qwen2.5-VL（3B 语言模型、32 个 block 的 ViT）checkpoint，带一个宽度 1024 的 flow matching 动作专家。XInfer 将其拆成三个阶段（`vision`、`prefill`、`action`）提供服务，使用 5 步 Euler、horizon 16 和 26 个动作维度，并支持实时分块（real-time chunking，RTC）。

Wall-OSS 是这一架构的开源名称（[X-Square-Robot/wall-x](https://github.com/X-Square-Robot/wall-x)）。这几页中的结果使用该架构的一个内部 checkpoint 测得，对照的是当时为它提供服务的生产 Wall-X 4B 服务。开源版本（Wall-OSS-0.5、WALL-OSS-FLOW）的 checkpoint 布局和归一化统计量不同，目前还无法加载。

## 状态

| 项目 | 状态 |
|---|---|
| 模型数值 | 两种组合，BF16。`numerics="optimized"`（默认）：ViT、expert 0 和动作专家跑在 xcompute 的算子上（残差加法融合进 RMSNorm、带 bias 和 RoPE 并写入 K/V 缓存的 QKV、GEMM 尾处理里的 SiLU-mul 和 GELU、action 阶段的 small-M GEMM、缓存注意力和读出-Euler 尾部）（[设计](wall_oss_design.zh-CN.md#optimized-组合)）；从相同输入出发，各阶段与 Wall-X eager 模型相差在 2.8 % 以内（[验证](wall_oss_validation.zh-CN.md#optimized-组合)）。`numerics="reference"`：使用参考（torch）算子时，prefill 和 action 阶段与 Wall-X eager BF16 模型逐位一致，ViT 仅因 attention kernel 不同而有差异（[验证](wall_oss_validation.zh-CN.md#参考算子模型)） |
| 输入/输出处理 | 在生产 Wall-X 4B 服务的 42 个请求的服务抓包上与其逐位一致；可选在 GPU 上解码和缩放各视角图像（`image_preprocess_device="gpu"`，非逐位一致；见[下文](#部署选项processingconfig)） |
| 执行 | 共置（colocated）的 `Engine`（[docs/engine.md](../engine.zh-CN.md)）：每阶段多个通道（lane）、链式派发、CUDA graph（与 eager 逐位一致）。optimized 组合每次派发一个请求；参考组合在负载下 action 阶段跨不同 prompt 长度和 RTC 延迟以 B1–B8 组批，每个请求与其单独运行相差在舍入级以内（[验证](wall_oss_validation.zh-CN.md)） |
| 服务 | `xinfer.entrypoints.serve`，基于 WebSocket + msgpack，兼容 x2robot 桌面客户端（[服务](../serving.zh-CN.md)） |
| 精度 | 仅 BF16 |
| Attention | 安装 flash-attn 时，ViT 和 prompt 的 attention 使用 flash-attn 后端；action 的 attention 使用 xcompute 的 `attention_kvcache`（optimized 组合）或 flash-attn（参考组合） |
| 数值一致性 | 与参考组合一致，由 `tests/models/test_wall_oss_optimized.py` 和 `tests/models/test_wall_oss_parity.py` 检查；optimized 组合还通过了内部精度门禁（[精度](../precision.zh-CN.md#如何检查一致性)） |
| 速度 | RTX 5090，optimized 组合：阶段 graph vision 15.6、prefill 21.3、action 7.3 ms（768 token 的桶、5 步，共 44.2 ms）；`Policy.infer` 47.3 ms（P50）。参考组合：50.9 ms 和 54.1 ms（[验证](wall_oss_validation.zh-CN.md#延迟)）。视图在设备上处理时的单请求时延和单卡机器人数见[性能](../performance.zh-CN.md)。Jetson Thor（xcompute `c0d03ca` 为 SM110 构建，默认时钟，视图在设备上处理）：三轮单请求 P50 为 94.6–96.0 ms（引擎 89.1–91.1 ms），不用 SM110 算子时为 132–145 ms（引擎 127–142 ms）（[设计](wall_oss_design.zh-CN.md#optimized-组合)） |
| 尚未支持 | optimized 组合 action 阶段的组批、静态 prompt 的 KV 复用（实测收益约 0.2 ms，不计划实现）、MXFP8、NVFP4 |

## 环境要求

- 内部布局的 checkpoint 目录，包含 `config.json`、`config.yml`、`model.safetensors`、`tokenizer.json`、`normalizer_action.pth` 和 `normalizer_propri.pth`。
- `pip install -e '.[models]'`（tokenizers、OpenCV、PyYAML）以及 `numba`。旋转换算在 numba 下运行，以便与参考服务逐位一致（见[设计](wall_oss_design.zh-CN.md#处理)）。
- 一块 CUDA GPU，BF16 权重需要约 10 GB 空闲显存（测试可在 CPU 上运行）。
- 使用 `image_preprocess_device="gpu"` 时：`pip install -e '.[gpu-image]'`（NVIDIA 的 nvImageCodec，`nvidia-nvimgcodec-cu13`）。
- optimized 组合（默认）：针对该 GPU 架构编译的 CUDA kernel 库，编译与检查方法见 [kernel](../kernels.zh-CN.md)。本文的数据在 RTX 5090（SM120）上测得。缺少时它运行在 torch 参考实现上（很慢）。

## 运行

```python
from xinfer import Policy
from xinfer.models.wall_oss.processing import ProcessingConfig

policy = Policy("wall_oss", "/path/to/checkpoint", device="cuda",
                processing=ProcessingConfig())
response = policy.infer(request)   # request: the desktop robot msgpack dict
```

`request` 遵循桌面机器人协议：`state.follow1_pos` / `state.follow2_pos`（各 7 个浮点数）、`views.camera_front|camera_left|camera_right`（HWC uint8 RGB 或 base64 JPEG）、`instruction`，RTC 请求还需 `use_rtc`、`prev_action_chunk` `[N, 14]` 和 `inference_delay`。响应包含 `follow1_pos`、`follow2_pos`（17 × 7）、`predict_relative_chunk`（17 × 20）和 `action_pred`（17 × 14）。

通过网络为机器人提供服务见[服务](../serving.zh-CN.md)；处理选项通过 `--model-options '{"processing": {...}}'` 传入。

参考组合：传入 `numerics="reference"`（服务：`--model-options '{"numerics": "reference"}'`）。

## 部署选项（`ProcessingConfig`）

| 选项 | 默认值 | 说明 |
|---|---|---|
| `cameras` | `face_view←camera_front`、`left_wrist_view←camera_left`、`right_wrist_view←camera_right` | 顺序决定 prompt |
| `instruction_override` | `None`（客户端指令） | 参考服务把每条指令都替换为一个固定的任务字符串 |
| `embodiment`、`frequency_hz`、`action_space` | `X2 ex001_desktop`、32、`Rel EEF` | 参考部署的 system prompt |
| `num_steps` / `allow_client_num_steps` | 5 / true | RTC 请求可以发送 `num_flow_steps` |
| `rtc_wire_delay_range` | (3, 8) | 对原始 `inference_delay` 做准入 |
| `prompt_bucket` | 16 | prompt 在末尾（action token 之前）填充到该 token 数的整数倍，因此 CUDA graph 以桶（bucket）为键；两种组合都对填充做掩码，填充最多在舍入级改变动作（[设计](wall_oss_design.zh-CN.md#prompt-桶)） |

`image_preprocess_device` 是顶层模型选项（不放在 `processing` 中）：`xinfer serve` 下为 `"cpu"`；`Policy` 下取平台默认值（CUDA 上为 `"gpu"`）。`"gpu"`：前端发送 JPEG 字节和原始帧及其目标尺寸，引擎在第一个阶段之前用 nvImageCodec 解码 JPEG，并在设备上缩放各视角图像（bicubic，与 OpenCV INTER_CUBIC 相同）。可省去前端的解码和缩放时间；视角图像以及由此得到的动作与 CPU 路径不逐位一致（[配置](../configuration.zh-CN.md#在-gpu-上做图像预处理)）

初始 flow 噪声由输入处理器在 CPU 上逐请求采样（`ProcessingConfig.noise_seed` 可将其固定）；测试会替换准备好的 `noise` 张量。

## CUDA graph

```python
from xinfer.engine.cudagraph import CUDAGraphConfig

policy = Policy("wall_oss", ckpt, device="cuda", cudagraphs=CUDAGraphConfig())
policy.warmup([request])        # captures the graphs these requests need on every lane and batch size
```

Graph 以各阶段读取的输入为键。Wall-OSS 需要：每种相机布局一个 vision graph，每个 prompt 桶一个 prefill graph，每个（批内最长的 prompt 桶、flow/RTC、RTC 动作行数、批大小）组合一个 action graph；不同 RTC 延迟共用一个 graph。状态以文本形式写入 prompt，因此手臂位姿会改变 prompt 长度：200 个随机位姿产生了 18 种长度（739–756 个 token），落在 2 个桶中。预热会把每个请求扩展到其指令和图像在任意手臂状态下可能落入的每个 prompt 桶（`Engine.warmup`；20 个状态值各占 1 到 3 个数字 token，因此范围为 40 个 token，即 3 或 4 个桶），所以只需一个 flow 请求和一个 RTC 请求（使用实际流量的相机、指令和 RTC 行数）即可覆盖所有位姿。预热未覆盖的形状（其他指令长度或相机布局）会在首次使用时捕获（`CUDAGraphConfig.on_miss_after_warmup`，默认 `capture`，每个 graph 约 0.1–0.4 s），之后复用。x2robot dummy 客户端发送 772 个 token 的 prompt（桶 784），以及带 16 行 RTC 的 RTC 请求。
