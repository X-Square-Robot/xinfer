# Wall-OSS: design

English | [简体中文](wall_oss_design.zh-CN.md)

Navigation: [usage](wall_oss.md) · [validation](wall_oss_validation.md)

## Source of semantics

The reference is the eager `Qwen2_5_VLMoEForAction` flow path of Wall-X
`02c5351d` (with the production service's standalone patch) and, for request
processing, the production Wall-X 4B service. Files: `xinfer/models/wall_oss/` (`config.py`, `model.py`,
`processing.py`, `geometry.py`) and the shared layers in `xinfer/models/layers/`.

## Stage split

The checkpoint is a two-expert mixture of transformers: expert 0 (the
language model, width 2048) handles the prompt, expert 1 (width 1024) the 16
action tokens; each layer has one attention over both. Prompt tokens never
attend to action tokens, so the prompt's per-layer K/V does not depend on the
actions or the denoising time. The model is therefore split exactly:

| Stage | Inputs | Output (handoff) |
|---|---|---|
| `vision` | resized views `image_<i>` uint8, packed in the stage into `[N_patches, 1176]` FP32 patch rows; host-computed window layout | `image_embeds` `[N_tokens, 2048]` BF16 |
| `prefill` | `input_ids[:, :P]`, M-RoPE `position_ids`, `image_embeds` | `prompt_kv` `[36, 2, 1, 2, P, 128]` BF16 (post-RoPE K, V; `P` the padded prompt length) |
| `action` | `prompt_kv`, noise, `dof_mask`, padding action, RTC actions and delay | `predict_action` `[1, 16, 26]` FP32, checkpoint units |

The last prompt layer only computes its K/V (its attention output and MLP have
no consumer). Stages read no device values on the host: the RTC delay and the
action position base are tensors, image tokens are placed by a host-computed
index, and the initial noise comes from the input processor. Host values
(prompt length, step count, RTC action rows, layout lists) select the graph. The prefill and action stages reproduce the joint forward bit
for bit (see validation).

## Numerics

The reference composition (`numerics="reference"`) reproduces the Wall-X eager model; the
[optimized composition](#optimized-composition) fuses its steps.

- BF16 weights and activations; FP32 language-model RMSNorm weights and
  FP32 action head, time embedding and ODE state (as Wall-X loads them).
- RMSNorm normalizes and scales in FP32 and rounds once.
- LLM rotary tables: FP32 angles, cos/sin rounded to BF16; M-RoPE bands
  16/24/24. ViT rotary: FP32, not rounded.
- The rotation itself (`rotary_embedding`) runs in FP32 and rounds once to
  the activation dtype.
- Flow: `t = linspace(0, 1, N+1) · 0.999`, Euler `x ← x + Δt·v`, velocity
  prediction, per-token time (`τ = 1` for pinned RTC tokens), padding velocity
  for the uncommanded dims fixed at step 0, unnormalization with per-step
  `[16, 26]` statistics.
- The ViT MLP keeps the checkpoint's 128-aligned width (3456) so GEMM shapes
  match the source.

## Optimized composition

`numerics="optimized"` (the default, `xinfer/models/wall_oss/optimized.py`) runs the reference composition's steps
on operators of `xinfer/ops/reference/`, each one xcompute call (xcompute `7336b0d`) where xcompute lists a kernel for
the shape, else the operator's torch reference. All BF16, like the reference.

| Step | Operator | Kernel at this model's shapes |
|---|---|---|
| ViT and LM residual add + RMSNorm | `add_rms_norm` (`rms_norm` for the first norm) | xcompute `add_rmsnorm` / `rmsnorm` |
| ViT QKV + bias + 2-D RoPE | `qkv_rope_epilogue` (`qk_order="any"`: Q and K only feed the attention) | `qkv_proj_rope` → cuBLASLt and a RoPE kernel; on SM110 `tcgen05_bf16_sm110` (Q and K in rotary-pair order) |
| ViT attention (windows, full blocks) | `varlen_attention` | flash-attention 2 |
| ViT projection and MLP down (+ bias) + residual add + the next RMSNorm | `gemm_add_rms_norm` | `gemm` → cuBLASLt, `add_rmsnorm`; on integrated GPUs (Jetson Thor) `gemm` with the residual → cuBLASLt (β·C), `rmsnorm` |
| Merger (+ bias; fc1 + GELU) | `gemm_epilogue` | `gemm` → cuBLASLt |
| ViT gate/up + bias + SiLU-mul | `gemm_glu_epilogue` (`bias`) | `gemm_glu` → cuBLASLt and the SiLU-mul kernel |
| Expert-0 QKV + bias + M-RoPE, K/V written to the prompt K/V | `qkv_rope_kvcache` | `qkv_proj_rope` (`kv_cache`) → cuBLASLt and a RoPE kernel |
| Expert-0 causal attention | `causal_attention` | flash-attention 2 |
| Expert-0 output projection | `gemm_epilogue` | cuBLASLt |
| Expert-0 MLP (gate/up + SiLU-mul, down) + residual add + the next RMSNorm | `mlp_add_rms_norm` | `gemm_glu` → `tma_streamk_bf16_sm120`, `gemm` → cuBLASLt, `add_rmsnorm`; on SM110 `fused_mlp` → `mcast_bf16_sm110` and `rmsnorm` for M 513–768, other M `gemm_glu` → `mcast_bf16_sm110`, `gemm` with the residual → `mcast_bf16_sm110` and `rmsnorm` |
| Action embedding (`ActionHead`, FP32) | `linear`, `silu` | cuBLAS (FP32) |
| Expert-1 QKV + bias + M-RoPE + K/V append | `qkv_rope_kvcache` | `qkv_proj_rope` (`kv_cache`) → cuBLASLt and a RoPE kernel |
| Expert-1 attention over the prompt K/V and the action keys | `prefix_kvcache_attention` | `attention_kvcache` → `splitkv` |
| Expert-1 output and down projections (`[K, N]` weights); gate/up + SiLU-mul | `gemm_epilogue`; `gemm_glu_epilogue` | `gemm` → `smallm_bf16_sm120`; cuBLASLt and the SiLU-mul kernel |
| Expert-1 weights read ahead: each layer's output, gate/up and down projections and the next layer's QKV, during its QKV and attention | `prefetch_weights`, `join_prefetch` (after the steps) | `l2_prefetch` on a side stream, on integrated GPUs (Jetson Thor); nothing on discrete ones, where it costs time |
| Readout, velocity of the uncommanded dimensions, Euler step, pinned rows | `flow_velocity_euler` | `denoise_output_euler` → `fused_tail` |

At this model's BF16 shapes xcompute lists no kernel faster than cuBLAS for the ViT and expert-0 GEMMs except the
expert-0 gate/up; they are compute-bound (about 85 % of the vision and prefill stages). The gains are the fused
residual adds and norms, the expert-0 gate/up kernel and the action stage's small-M kernels and cache attention.

Arithmetic that differs from the reference composition:

- **Fused epilogues**: the QKV bias and RoPE, the GLU's SiLU-mul and the merger's GELU are applied to the FP32
  accumulator and rounded once; the reference rounds the GEMM output first. Residual sums are rounded to BF16 as
  before.
- **Action stage**: one request per dispatch. The prompt K/V and the action keys share a head-major cache
  `[layers, 2, 1, kv_heads, P + 16, head_dim]`; prompt padding and the RTC visibility (pinned rows see the prompt
  and each other) are passed as per-column row ranges (FlashMask `LTS`/`UTE`). The velocity stays FP32.
- **Prompt buckets**: the padding is masked in the action attention, so the actions of a padded prompt equal those
  of the exact prompt within rounding, not bit for bit.
- **Jetson Thor**: on integrated GPUs the ViT projections add the residual to the FP32 product and round once, and
  the fused expert-0 MLP does not fix its reduction order, so two runs can differ at rounding level. On discrete GPUs
  `gemm_add_rms_norm` and `mlp_add_rms_norm` give the bits of the operators they join.

## Action-stage batching

The reference composition's action stage takes 1, 2, 4 or 8 requests, with different prompt lengths
and RTC delays; the optimized composition takes one. A batched request gets its single-request result within
rounding (design §6.4):

- **Operators.** The action stack uses the default selection, as the other
  stages do: cuBLAS `linear`, the xcompute `rms_norm` and flash-attn
  `action_attention`. cuBLAS may choose another kernel for another row count,
  so a batched request can differ from its single run at rounding level
  (measured up to 3.4e-3, relative norm of the actions).
- **Different prompt lengths.** `prompt_kv` declares its sequence axis.
  Batches left-pad it and pass `prompt_kv_lengths`; `action_attention` skips
  each row's padding (`cache_leftpad`), as the service did.
- **Padded prompts** keep their valid tokens first (next section); the action
  stage moves each row's padding into the left padding before attention.
- **Engine behavior.** The engine batches whatever is queued when the action
  lane frees up (zero-wait batching). Under load the action stage therefore
  runs B2–B8 without any setting; `--max-batch action=N` caps the batch size.

## Prompt buckets

The state is written into the prompt as text (512 bins, one token per digit),
so each arm pose changes the prompt length; 200 random poses gave 18 lengths.
Graphs keyed by the exact length would follow the traffic, so the input
processor pads the prompt to a multiple of `ProcessingConfig.prompt_bucket`
(16) tokens:

- **Where.** Pad tokens (`<|endoftext|>`, M-RoPE positions 0) go at the end of
  the prompt, before the action tokens (`processing.pad_prompt`).
  `prompt_len` becomes the padded length `W`, which keys the prefill graph;
  `prompt_valid` (`[1]` int32) holds the number of valid tokens `P`; and
  `action_position_base` stays the position of the unpadded prompt's end.
- **Prefill.** Under causal attention no valid token attends to a later pad
  token, and the norms and projections compute each token's row on its own,
  so the K/V of the first `P` columns equal the unpadded prefill's. The MXFP8
  CUTLASS GEMMs guarantee this (fixed tiles, no split-K). For cuBLAS (BF16) it
  is measured, not guaranteed: its kernel choice may depend on the row count.
  flash-attn's causal call (`num_splits=1`) visits the same key blocks for a
  valid row at both lengths.
- **Action stage.** `prompt_kv` arrives `W` wide with its first `P` columns
  valid, left-padded to the batch's widest row by the engine merge
  (`prompt_kv_lengths = W`). `denoise` rotates each row right by `W − P` once
  per dispatch (`left_align_prompts`, a gather over the prompt K/V), which
  makes it a plain left-padded row with `prompt_kv_lengths = P`: the layout the
  merge already produces for different lengths, which `action_attention` skips
  with `cache_leftpad`. The alternative, writing the
  action K/V right after each row's valid prompt as the Wall-X service's action
  pass does, would add a scatter per layer and step (360 per request) in place
  of the one gather; it would also change the attention operator.
- **Rounding.** flash-attn's action attention skips the left padding, so a
  padded request alone gets its exact prompt's actions. The torch reference
  attention runs its softmax and products over all columns, and a batch
  changes the GEMM row counts: both can change the bits at rounding level.

Measured on an RTX 5090 with the default backends (validation): the padded
prompts of 12 requests with random arm poses give their exact prompts' actions
bit for bit alone (eager and graphs), and within 5.9e-3 (relative norm) batched
with requests of other buckets.

## Attention

The model calls three structured attention operators rather than a general
masked one, so that a backend can use the kernel that fits each case:

| Stage | Operator | Semantics |
|---|---|---|
| `vision` | `varlen_attention` | bidirectional within each window / full segment of packed patches |
| `prefill` | `causal_attention` | causal self-attention over the prompt |
| `action` | `action_attention` | every action token sees the whole prompt; pinned RTC rows (`< delay`) see only the pinned action tokens, the others see all |

The torch references reproduce the eager model's SDPA calls exactly (same
masks and layouts). With flash-attn installed the default backend uses:
- `varlen_fwd` in the ViT and `flash_attn_with_kvcache` (`num_splits=1`) for the prompt, the service's calls;
- for the action tokens, one decode-style row per query with a fixed split count (backend docstring). The service's two-call form was slower than the reference here.

## RTC semantics (decision)

For RTC requests the pinned action tokens (`k < delay`) do not attend to the
denoising tokens at **every** step. This is what the production services
run. The Wall-X eager code skips that cut at step 0 only
(its step-0 joint mask is symmetric among action tokens), so for RTC requests
the eager code and the served model differ in semantics, not just precision.
XInfer follows the served model; the eager reference used for validation is
patched the same way. Flow requests (`delay = 0`) are identical either way.

## Processing

`WallOSSInputProcessor` reproduces the reference service: OpenCV INTER_CUBIC
resize to long side 448 then `smart_resize(28)`, the embodied system prompt, a 512-bin state
string, the checkpoint tokenizer plus `<|propri|>`=151665 and
`<|action|>`=151666, host M-RoPE positions, float64 proprioception and masks,
and the RTC actions converted from absolute EE poses to the relative 6D model
layout. `WallOSSOutputProcessor` composes relative actions with the request
state into 17 absolute rows per arm. Rotation helpers are the reference's
numba functions ported verbatim (`geometry.py`): their float32
specializations and libm calls are part of the served numerics.

The views are decoded and resized concurrently and the prompt is tokenized with
the cached fixed pieces (`map_views` and `SegmentTokenizer` in
`xinfer.models.processing`). The resized views cross to the engine as uint8. The vision stage packs them
(`xinfer.models.layers.image_pack.ImagePack`): a float32 lookup table of each channel
level, then the Qwen2.5-VL 14/2/2 patch layout with the image repeated over the
two temporal slots. It only indexes and copies, so the rows are bitwise equal on
the CPU and on the GPU, inside or outside a CUDA graph, and the request carries
an eighth of the packed float32 bytes.

With `image_preprocess_device="gpu"` the processor does not decode or resize:
it sends each JPEG view's bytes (other encodings are decoded on the CPU) or raw
frame as a `DeviceImage` with the size the CPU path would resize it to, read
from the JPEG header, and the recipe `DEVICE_RESIZE`, so the prompt, grids and
vision layout are unchanged. Before the first stage the engine decodes the
request's JPEGs in one call of the platform's JPEG decoder (nvImageCodec) and resizes each view
with an FP32 bicubic interpolation (a = −0.75, no antialiasing, as
INTER_CUBIC), rounded half up to uint8 ([engine](../engine.md#device-images)).
The vision stage reads the views as with CPU preprocessing and they have the
CPU path's shapes, so the vision graph and its key are the same. The resize of
a raw frame is within one level of OpenCV's and a decoded JPEG within one
level of libjpeg-turbo's; the views are still not bitwise equal, so the
actions differ from the CPU path's and the option is off by default.

Deployment choices of the reference (fixed instruction, camera wire keys,
embodiment text, admitted RTC delays, client step override) are explicit
`ProcessingConfig` fields.

## Known gaps

- Vision and prefill take one request per dispatch; the optimized composition's action stage too.
- No MXFP8 or NVFP4. In BF16 the ViT and expert-0 GEMMs run at cuBLAS speed (no faster xcompute kernel is listed
  for these shapes); the action stage's gate/up and QKV GEMMs run xcompute's generic path.
- With reference operators prompts are not padded, so these models need one
  graph per prompt length.
