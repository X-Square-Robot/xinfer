# Wall-X 2B

English | [简体中文](wallx2b.zh-CN.md)

Use the model ID `wallx2b` to serve a BF16 Wall-X 2B checkpoint on a CUDA GPU or on an Ascend 950DT NPU; the [Ascend guide](../ascend950dt.md) covers the build and device setup for the latter, and the commands below use `--device cuda`. The model takes three camera views, the state of a desktop dual-arm robot and a task instruction. Each call predicts 16 steps for both arms. Real-time chunking (RTC) can pin part of the previous action so consecutive predictions overlap.

See the [validation results](wallx2b_validation.md) for latency, single-GPU capacity and open-loop accuracy on the RTX 5090.

For lower-precision inference on CUDA, add `--model-options '{"quantization":"mxfp8"}'` (or `"nvfp4"`) to the serving command below, or pass `quantization="mxfp8"` (or `"nvfp4"`) to `Policy`. This uses the same checkpoint and quantizes the vision blocks and prompt projections at load time. The action stage retains its BF16 composition. See the [model design](wallx2b_design.md) ([MXFP8](wallx2b_design.md#mxfp8), [NVFP4](wallx2b_design.md#nvfp4)) for stage selection, operator requirements and validation.

## Checkpoint requirements

Provide a compatible checkpoint directory obtained separately. This repository ships the inference code. Start with the following files from the same training export:

| Files | What the loader uses |
|---|---|
| `model.safetensors`, `config.json` | Weights for the Qwen3.5 vision, language and action components, and their matching dimensions. |
| `config.yml` | The `qwen3_5` model type, the task's action and state layouts, and the embedded Dataset V2 configuration at `data.dataset_config_effective`. |
| `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json` | The checkpoint tokenizer and its added tokens, and the image processor settings. |
| `normalizer_action.pth`, `normalizer_propri.pth` | Normalization statistics for the checkpoint's actions and robot state. |

## Run

Install a CUDA-compatible PyTorch build, then run the following from the repository root. Supply `flow.msgpack` as one recorded observation in the format below. Add `rtc.msgpack` when the service will receive RTC requests. Each file contains one request. The example listens on port 33985 after warmup.

```bash
python -m pip install -e '.[models,serve]'
xinfer-serve --model wallx2b --checkpoint /path/to/wallx2b-checkpoint \
  --device cuda --warmup flow.msgpack rtc.msgpack --port 33985
```

The default optimized BF16 composition uses the CUDA kernel library when it is built. See the [kernel guide](../kernels.md) to build it and inspect the selected operators. To run the reference composition with FlashAttention 2 for vision, install a compatible FlashAttention 2 build and add `--model-options '{"numerics":"reference"}' --reference-ops --op-backends varlen_attention=flash_attn` to the service command.

The same flow observation can be run in process. The example selects CPU image preprocessing to match the service default.

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

## Request and response

| Request field | Desktop dual-arm format |
|---|---|
| `views` | A mapping with `camera_front`, `camera_left` and `camera_right`. Each value is JPEG bytes, a base64 JPEG string or an HWC `uint8` RGB array. |
| `state` | `follow1_pos` for the left arm and `follow2_pos` for the right arm. Each contains `[x, y, z, roll, pitch, yaw, gripper]`, with position in metres and angles in radians. |
| `instruction` | The task as a string. |
| RTC fields | Set `use_rtc=true`. Supply integer `s_model` (steps executed) and `inference_delay`, plus `prev_action_chunk` with the executed rows already shifted out and the tail padded. The chunk has shape `[16, 14]`, or `[17, 14]` with a leading anchor row. |

The response contains `follow1_pos` and `follow2_pos`, each with 16 rows of absolute arm poses. `action_pred` concatenates those poses into a `[16, 14]` array. `predict_action_norm` is the normalized model output with shape `[1, 16, 26]`.
