# Wall-OSS: usage

English | [简体中文](wall_oss.zh-CN.md)

Navigation: [design](wall_oss_design.md) · [validation](wall_oss_validation.md)

Wall-OSS is a Qwen2.5-VL (3B language model, 32-block ViT) checkpoint with a
1024-wide flow-matching action expert. XInfer serves it as three stages
(`vision`, `prefill`, `action`) with 5 Euler steps, horizon 16 and 26 action
dimensions, and supports real-time chunking (RTC).

Wall-OSS is the open-source name of this architecture
([X-Square-Robot/wall-x](https://github.com/X-Square-Robot/wall-x)). The
results on these pages are measured with an internal checkpoint of the
architecture against the production Wall-X 4B service that served it. The open-source releases (Wall-OSS-0.5, WALL-OSS-FLOW) use a
different checkpoint layout and normalization statistics and cannot be loaded
yet.

## Status

| Item | State |
|---|---|
| Model numerics | Two compositions, BF16. `numerics="optimized"` (the default): the ViT, expert 0 and the action expert on xcompute's operators (residual adds fused into the RMSNorms, QKV with bias and RoPE writing the K/V cache, SiLU-mul and GELU in the GEMM epilogues, small-M GEMMs, cache attention and the readout-Euler tail in the action stage) ([design](wall_oss_design.md#optimized-composition)); each stage is within 2.8 % of the Wall-X eager model from the same inputs ([validation](wall_oss_validation.md#optimized-composition)). `numerics="reference"`: with reference (torch) operators the prefill and action stages are bitwise equal to the Wall-X eager BF16 model and the ViT differs only through the attention kernel ([validation](wall_oss_validation.md#reference-operator-model)) |
| Input/output processing | Bitwise equal to the production Wall-X 4B service on a service capture of 42 requests; optionally the views are decoded and resized on the GPU (`image_preprocess_device="gpu"`, not bitwise equal; [below](#deployment-options-processingconfig)) |
| Execution | Colocated `Engine` ([docs/engine.md](../engine.md)): lanes per stage, chained dispatch, CUDA graphs (bitwise equal to eager). The optimized composition runs one request per dispatch; the reference composition batches the action stage B1–B8 under load, across prompt lengths and RTC delays, each request within rounding of its single run ([validation](wall_oss_validation.md)) |
| Serving | `xinfer.entrypoints.serve` over WebSocket + msgpack, compatible with the x2robot desktop client ([serving](../serving.md)) |
| Precision | BF16 only |
| Attention | flash-attn backend for the ViT and prompt attention when installed; the action attention on xcompute's `attention_kvcache` (optimized composition) or flash-attn (reference composition) |
| Numerical agreement | Agreement with the reference composition, checked by `tests/models/test_wall_oss_optimized.py` and `tests/models/test_wall_oss_parity.py`; the optimized composition also passes the internal precision gate ([precision](../precision.md#how-agreement-is-checked)) |
| Speed | RTX 5090, optimized composition: stage graphs vision 15.6, prefill 21.3, action 7.3 ms (44.2 ms at the 768-token bucket, 5 steps); `Policy.infer` 47.3 ms (P50). Reference composition: 50.9 ms and 54.1 ms ([validation](wall_oss_validation.md#latency)). Single request with the views on the device and robots per GPU: [performance](../performance.md) |
| Not yet | batching in the optimized action stage, static-prompt KV reuse (measured gain about 0.2 ms, not planned), MXFP8, NVFP4, Thor |

## Requirements

- A checkpoint directory in the internal layout, with `config.json`, `config.yml`, `model.safetensors`,
  `tokenizer.json`, `normalizer_action.pth` and `normalizer_propri.pth`.
- `pip install -e '.[models]'` (tokenizers, OpenCV, PyYAML) and `numba`.
  Rotation conversions run under numba for bitwise parity with the reference
  service (see [design](wall_oss_design.md#processing)).
- A CUDA GPU with about 10 GB free for BF16 weights (CPU works for tests).
- For `image_preprocess_device="gpu"`: `pip install -e '.[gpu-image]'` (torchvision ≥ 0.19).
- Optimized composition (the default): the Python package of an xcompute build for the GPU's architecture
  importable. See [kernels](../kernels.md) for setup and checks. The reported measurements use an RTX 5090 (SM120).
  Without it the composition runs on the torch references (slowly).

## Run

```python
from xinfer import Policy
from xinfer.models.wall_oss.processing import ProcessingConfig

policy = Policy("wall_oss", "/path/to/checkpoint", device="cuda",
                processing=ProcessingConfig())
response = policy.infer(request)   # request: the desktop robot msgpack dict
```

`request` follows the desktop robot protocol: `state.follow1_pos` /
`state.follow2_pos` (7 floats each), `views.camera_front|camera_left|camera_right`
(HWC uint8 RGB or base64 JPEG), `instruction`, and for RTC `use_rtc`,
`prev_action_chunk` `[N, 14]` and `inference_delay`. The response has
`follow1_pos`, `follow2_pos` (17 × 7), `predict_relative_chunk` (17 × 20) and
`action_pred` (17 × 14).

To serve robots over the network, see [serving](../serving.md); processing
options go in `--model-options '{"processing": {...}}'`.

The reference composition: pass `numerics="reference"` (serving: `--model-options '{"numerics": "reference"}'`).

## Deployment options (`ProcessingConfig`)

| Option | Default | Notes |
|---|---|---|
| `cameras` | `face_view←camera_front`, `left_wrist_view←camera_left`, `right_wrist_view←camera_right` | order defines the prompt |
| `instruction_override` | `None` (client instruction) | the reference service replaced every instruction with a fixed task string |
| `embodiment`, `frequency_hz`, `action_space` | `X2 ex001_desktop`, 32, `Rel EEF` | system prompt of the reference deployment |
| `num_steps` / `allow_client_num_steps` | 5 / true | RTC requests may send `num_flow_steps` |
| `rtc_wire_delay_range` | (3, 8) | admission on the raw `inference_delay` |
| `prompt_bucket` | 16 | prompts are padded at their end, before the action tokens, to a multiple of this many tokens, so CUDA graphs are keyed by the bucket; both compositions mask the padding, which changes the actions at rounding level at most ([design](wall_oss_design.md#prompt-buckets)) |

`image_preprocess_device`, a top-level model option (not in `processing`): `"cpu"` under `xinfer serve`; under `Policy` the platform's (`"gpu"` on CUDA). `"gpu"`: the frontend sends the JPEG bytes and raw frames with their target sizes, and the engine decodes the JPEGs with nvJPEG and resizes the views on the device (bicubic like OpenCV INTER_CUBIC) before the first stage. Saves the frontend's decode and resize time; the views, and so the actions, are not bitwise equal to the CPU path ([configuration](../configuration.md#image-preprocessing-on-the-gpu))

Initial flow noise is drawn per request on the CPU by the input processor
(`ProcessingConfig.noise_seed` fixes it); tests replace the prepared `noise`
tensor.

## CUDA graphs

```python
from xinfer.engine.cudagraph import CUDAGraphConfig

policy = Policy("wall_oss", ckpt, device="cuda", cudagraphs=CUDAGraphConfig())
policy.warmup([request])        # captures the graphs these requests need on every lane and batch size
```

Graphs are keyed by the inputs each stage reads. Wall-OSS needs one vision
graph per camera layout, one prefill graph per prompt bucket and one action
graph per (longest prompt bucket in the batch, flow/RTC, RTC action rows, batch
size); RTC delays share a graph. The state is written into the prompt as text,
so arm poses change the prompt length: 200 random poses gave 18 lengths
(739–756 tokens) in 2 buckets. The warmup expands each request into every
prompt bucket any arm state can give with its instruction and images
(`Engine.warmup`; each of the 20 state values takes 1 to 3 digit
tokens, so a 40-token range, 3 or 4 buckets), so one flow and one RTC request
with the traffic's cameras, instruction and RTC rows cover every pose. Shapes
that warmup did not cover (another instruction length or camera layout) are
captured on first use (`CUDAGraphConfig.on_miss_after_warmup`, default
`capture`, about 0.1–0.4 s per graph) and reused afterwards. The x2robot dummy
client sends 772-token prompts (bucket 784) and RTC requests with 16 RTC rows.
