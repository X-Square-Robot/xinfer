# Wall-X 2B

[English](wallx2b.md) | 简体中文

在 CUDA GPU 或昇腾 950DT NPU 上使用模型 ID `wallx2b` 加载 BF16 Wall-X 2B checkpoint。后者所需的算子清单与环境配置见[昇腾指南](../ascend950dt.zh-CN.md)，下文的命令以 `--device cuda` 为例。模型接收三路相机图像、桌面双臂机器人的状态和任务指令，每次预测双臂接下来 16 步的动作。实时分块（RTC）可固定上一轮预测中的部分动作，使相邻两轮预测在时间上重叠。

RTX 5090 上的延迟、单卡服务容量和开环精度见[验证结果](wallx2b_validation.zh-CN.md)。

## Checkpoint 要求

请自行准备兼容的 checkpoint 目录。仓库提供推理代码。下列文件应来自同一次训练导出：

| 文件 | 加载器读取的内容 |
|---|---|
| `model.safetensors`、`config.json` | Qwen3.5 视觉、语言和动作组件的权重与对应维度。 |
| `config.yml` | `qwen3_5` 模型类型、任务的动作与状态布局，以及嵌入 `data.dataset_config_effective` 的 Dataset V2 配置。 |
| `tokenizer.json`、`tokenizer_config.json`、`preprocessor_config.json` | checkpoint 配套的 tokenizer、附加 token 和图像处理器设置。 |
| `normalizer_action.pth`、`normalizer_propri.pth` | 与该 checkpoint 的动作与机器人状态匹配的归一化统计量。 |

## 运行

安装与 CUDA 环境匹配的 PyTorch，然后在仓库根目录执行下列命令。将一条符合下文格式的观测保存为 `flow.msgpack`。服务需要接收 RTC 请求时，另准备一条 `rtc.msgpack` 用于预热。每个文件包含一条请求。示例中的服务完成预热后在 33985 端口监听。

```bash
python -m pip install -e '.[models,serve]'
xinfer-serve --model wallx2b --checkpoint /path/to/wallx2b-checkpoint \
  --device cuda --warmup flow.msgpack rtc.msgpack --port 33985
```

默认的优化 BF16 组合可使用 xcompute。安装与 GPU 匹配的构建并查看实际选择的算子后端，见 [kernel 指南](../kernels.zh-CN.md)。如需使用参考组合并让视觉阶段使用 FlashAttention 2，安装兼容的 FlashAttention 2 构建，并在服务命令中追加 `--model-options '{"numerics":"reference"}' --reference-ops --op-backends varlen_attention=flash_attn`。

同一条 flow 观测也可以直接在进程内运行。示例选择 CPU 图像预处理，与服务默认设置一致。

```python
from pathlib import Path

from xinfer import Policy
from xinfer.frontend.protocol import unpackb

observation = unpackb(Path("flow.msgpack").read_bytes())
policy = Policy("wallx2b", "/path/to/wallx2b-checkpoint", device="cuda",
                image_preprocess_device="cpu")
try:
    policy.warmup([observation])
    response = policy.infer(observation)
    print(response["action_pred"].shape)
finally:
    policy.engine.shutdown()
```

## 请求与响应

| 请求字段 | 桌面双臂格式 |
|---|---|
| `views` | 包含 `camera_front`、`camera_left` 和 `camera_right` 的映射。每路图像可以是 JPEG 字节、base64 编码的 JPEG 字符串或 HWC 格式的 `uint8` RGB 数组。 |
| `state` | `follow1_pos` 对应左臂，`follow2_pos` 对应右臂。两者均为 `[x, y, z, roll, pitch, yaw, gripper]`，位置单位为米，角度单位为弧度。 |
| `instruction` | 任务指令字符串。 |
| RTC 字段 | 设置 `use_rtc=true`，提供整数 `s_model`（已执行步数）和 `inference_delay`。将上一动作块已执行的行移出并在末尾补齐后，作为 `prev_action_chunk` 传入。形状为 `[16, 14]`；若首行为锚定位姿，则为 `[17, 14]`。 |

响应中的 `follow1_pos` 和 `follow2_pos` 分别包含左右臂 16 步的绝对位姿。`action_pred` 将两者拼接为形状 `[16, 14]` 的数组。`predict_action_norm` 是形状为 `[1, 16, 26]` 的归一化模型输出。
