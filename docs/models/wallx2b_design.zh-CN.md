# Wall-X 2B 设计

[English](wallx2b_design.md) | 简体中文

Wall-X 2B 包含 Qwen3.5 主干和基于流匹配的动作专家。每条请求包含三个相机视角、机器人状态和指令。Checkpoint 文件、输入和输出格式见[模型指南](wallx2b.zh-CN.md)。

## 阶段划分

| 阶段 | 计算 | 交接数据 |
|---|---|---|
| 视觉 | 视觉 Transformer 对图像块编码，每个视角独立执行注意力计算；随后按空间分组归并图像嵌入。 | 图像 token。 |
| 预填充 | 用图像 token 替换提示词里的图像占位符，然后执行主干中的 DeltaNet 层和全注意力层。 | 各全注意力层经旋转编码的键和对应的值。 |
| 动作 | 动作专家读取缓存的提示词键值，反复预测动作 token 的速度。每个 Euler 步根据速度更新当前预测。 | 归一化动作块，由输出处理器转换为机器人位姿。 |

提示词计算独立于采样中的动作，因此动作阶段可以在各去噪步复用同一份提示词缓存。对实时分块（RTC）请求，每步都将请求提供的动作前缀固定，再以该前缀作为条件预测剩余各行。

## MXFP8

模型默认使用 BF16。在优化计算组合下，`quantization="mxfp8"` 在加载权重后一次性转换选定的投影层。每行最多 128 个连续列共享一个缩放因子。归一化和 GEMM 末端计算可以直接为下一次投影生成量化输入。中间残差和输出动作保留原有精度。

| 阶段 | MXFP8 预设生效的位置 |
|---|---|
| 视觉 | 注意力的 QKV 投影、输出投影和 MLP 的两个投影使用 MXFP8 操作数。图像块嵌入和空间归并层仍使用 BF16 路径。 |
| 预填充 | 提示词全注意力、DeltaNet 和 MLP 的投影使用 MXFP8 操作数。提示词 KV 缓存保留 BF16。 |
| 动作 | 动作专家读取提示词 KV 缓存，保留原有计算方式。 |

只量化预填充时，向 `Policy` 传入 `quantization={"groups":{"prefill":"mxfp8"}}`。只量化视觉时使用 `vision` 分组。该预设要求 `numerics="optimized"`，原生路径需要 SM120 和 CUDA kernel 库（见 [kernel 指南](../kernels.zh-CN.md)），它按各投影的尺寸选择专用或通用的块缩放 kernel。其他设备使用 PyTorch MXFP8 算子。可用 `Policy.describe()["ops"]` 查看 `mxfp8_gemm_epilogue`、`mxfp8_gemm_glu_quantize` 等算子的实际选择。E4M3 数值和缩放布局见[量化指南](../quantization.zh-CN.md)。

## NVFP4

`quantization="nvfp4"` 在加载权重后应用 Wall-X 的 NVFP4 方案：E2M1 数值，每 16 列一个 E4M3 缩放因子。视觉权重使用
amax 块缩放（`nvfp4`）。提示词权重的块缩放在每个 checkpoint 权重自己的全局缩放 `amax / (448 · 6)` 下通过 MSE 搜索选出
（`nvfp4_mse`）；全注意力层的 Q·gate、K、V 权重各自保留全局缩放，所以在同一个量化输入上作为三个 GEMM 运行。每个
GEMM 的输入都是 BF16 计算组合给它的 BF16 张量，再用 amax 块缩放量化：归一化和 GEMM 末端计算直接写出量化输入。

| 阶段 | NVFP4 预设生效的位置 |
|---|---|
| 视觉 | 注意力的 QKV 投影、输出投影、MLP 的两个投影和空间归并层的两个投影使用 NVFP4 操作数。图像块嵌入仍使用 BF16 路径。 |
| 预填充 | DeltaNet 的 `qkv` 投影和输出投影、全注意力的 Q·gate、K、V 投影和输出投影，以及 MLP 的投影使用 NVFP4 操作数。DeltaNet 的 z/b/a 投影和提示词 KV 缓存保留 BF16。 |
| 动作 | 动作专家读取提示词 KV 缓存，保留原有计算方式。 |

各分组也可以分别使用 `nvfp4` 和 `nvfp4_mse`，例如 `quantization={"groups":{"prefill":"nvfp4_mse"}}`。该预设要求
`numerics="optimized"`，原生路径需要 SM120 和 CUDA kernel 库：带残差的投影、提示词路径的投影和视觉 fc1（量化输出）
走它的 CUTLASS NVFP4 kernel，带 RoPE 的 QKV 和 gate/up 投影走它的通用块缩放路径，结果相同。其他设备使用 PyTorch
NVFP4 算子。可用 `Policy.describe()["ops"]` 查看 `nvfp4_gemm_epilogue` 等
`nvfp4_*` 算子的实际选择。E2M1 数值、缩放因子和算子形式见[量化指南](../quantization.zh-CN.md#nvfp4-格式)。

## 验证

[小型 checkpoint 测试](../../tests/models/test_wallx2b_tiny.py)使用相同的阶段输入，对照 BF16 参考计算逐阶段检查精度。测试覆盖普通 flow 请求和 RTC 请求，并检查固定的动作前缀逐值一致。[算子测试](../../tests/unit/test_mxfp8.py)覆盖舍入、提示词填充掩码、融合归一化和旋转投影；在配有兼容 xcompute 构建的 CUDA 设备上，还会按 Wall-X 2B 的投影尺寸对比 kernel 与 PyTorch 参考算子。

在 RTX 5090 和 xcompute `536ea8f`（带 MXFP8 GEMM 系列）上，MXFP8 从序列化请求到进程内返回响应的单请求 P50 为 17.0 ms（`benchmarks/request_latency.py`，图像在 GPU 上解码和缩放）：主机输入输出处理 3.4 ms，引擎 13.6 ms。同一构建上 BF16 为 22.3 ms（3.5 + 18.9）。机器人负载下（`benchmarks/performance.py`，每档 8 轮，14–24 台、步长 2），MXFP8 支撑 22 台，服务端 P50 / P95 中位数为 76 / 116 ms，24 台时为 151 ms。BF16 在 14 台时为 86 / 103 ms；14 台是它在 xcompute `7336b0d` 上的容量，那里 15 台时为 170 ms。MXFP8 对动作的改变超出舍入量级。启用 MXFP8 前，请在目标机器人任务上评估精度。

NVFP4 的[算子测试](../../tests/unit/test_nvfp4.py)对照定义检查各形式；在配有 kernel 库的 GPU 上，还会按 Wall-X 2B
的投影尺寸对比它的 NVFP4 算子与 PyTorch 算子；小型 checkpoint 测试用 `quantization="nvfp4"` 运行三个阶段。
在 RTX 5090 和 xcompute `3a2bee6`（NVFP4 kernel 合入为 `7034bdf` 之前的提交，动作逐位相同）上，各阶段 graph
耗时为视觉 1.5、预填充 3.1、动作 5.2 ms（共 9.8 ms；BF16 为 16.6 ms），单请求 P50 为 15.2 ms（3.5 + 11.7）。机器人负载下
NVFP4 支撑 28 台，服务端 P50 / P95 中位数为 103 / 145 ms，30 台时为 167 ms；饱和吞吐 154.3 req/s（同一 GPU 上 MXFP8 为
131.2，BF16 为 81.5）。加载后权重占 3.0 GB（BF16 为 4.9 GB）。对训练后的 checkpoint 直接应用 NVFP4，动作的变化比 MXFP8
大得多，在这个 checkpoint 上没有通过内部精度门禁；该方案面向用它的伪量化训练的 checkpoint。启用 NVFP4 前，请在目标
机器人任务上评估精度。
