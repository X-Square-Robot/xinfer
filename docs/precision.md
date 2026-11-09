# Precision gate

English | [简体中文](precision.zh-CN.md)

A change that can alter Wall-OSS outputs must pass the embodied-serving
quick100 gate, the gate the Wall-X 4B service was admitted with.

**Rule.** The GT-MSE of each required scenario relative to the frozen original
BF16 model must stay within +1 %: `MSE_candidate / MSE_BF16 − 1 ≤ 0.01`.

**Scope** (embodied-serving `docs/PRECISION_REGRESSION.md`):
- **Inputs**: 5 real episodes, 100 windows, seed 42, RTC delay 5.
- **Metric**: GT-MSE in normalized space over the 18 non-gripper dimensions, excluding the first `delay` steps of each window.
- **Reference**: the original full BF16 Wall-X model, B1, frozen.

XInfer runs the gate through an adapter for the embodied-serving harness
(`scripts/check_precision.py`, profile `quick100-batch32-v1`, scope
`model_batch`).

## Tools

| File | Role |
|---|---|
| `tools/precision/wall_oss_quick100_adapter.py` | `PRECISION_ADAPTER` for the harness (details below) |
| `tools/precision/configs/optimized.example.json` | candidate configuration: the optimized composition (the default) |
| `tools/precision/quick100_b1_score.py` | scores the B1 scenario of saved harness runs with the harness's functions |

What the adapter does for each request group:
- runs XInfer's input processing on the wire observation, with the request's saved FP32 noise (prompts padded to
  `processing.prompt_bucket`, 16 tokens by default, as served); with `processing.image_preprocess_device="gpu"` it
  decodes and resizes the views on the GPU before the vision stage, as the engine does (`xinfer.engine.device_images`);
- runs vision and prefill per request;
- runs one action-stage call over the whole group (B1 or the scenario's BN). Different prompt lengths, for example 760 and 761 tokens, are merged left-padded as the engine does (`xinfer.engine.batching`). The optimized composition takes one request per action call: the adapter raises for a batched group, so the harness stops after B1 and the run is scored with `quick100_b1_score.py`;
- reports that call's actual batch and delays.

## Running

The harness, the assets and the BF16 reference live outside this repository:

| Input | What it is |
|---|---|
| harness | an embodied-serving checkout at `fa2e8af` (clean) |
| assets | the quick100 import: manifest SHA `d2ec98c8…`, checkpoint `5_25000` weights SHA `37a4b2ea…` |
| BF16 reference | the frozen original BF16 outputs (`reference1` in the import) |
| target | the old Wall-X 4B MXFP8 service's run, embodied-serving `54691d0` |

```bash
PYTHONPATH=/path/to/xinfer/tools/precision python scripts/check_precision.py \
  --assets $ASSETS --reference $ASSETS/reference1 \
  --target $TARGET_RUN --target-ref 54691d0e20b5745eae8ebb6c1f25cdf039067dff \
  --require-scope model_batch --adapter wall_oss_quick100_adapter:Adapter \
  --config optimized.json --device 0 --output /path/to/run
python /path/to/xinfer/tools/precision/quick100_b1_score.py . $ASSETS $ASSETS/reference1 $TARGET_RUN /path/to/run
```

Run it from the harness checkout. The checkpoint is checked against the
manifest hashes, so use the original checkpoint directory, not a copy with
edited `config.yml`.

## Wall-OSS quick100 gate

Result (2026-11-07, RTX 5090, torch 2.10.0+cu130): `numerics="optimized"` with the default backends (xcompute `651195d`,
flash-attn), B1:

| Scenario | GT-MSE vs BF16 | Episode 95 % CI | vs old service (target) |
|---|---|---|---|
| quick100 B1, delay 5 (100 windows) | **−6.09 %: PASS** | [−12.07, +0.32] | +0.57 % |

- Scored the same way: the old MXFP8 service −6.62 %, XInfer's reference composition with the eager model's kernels
  −6.28 % (torch 2.9.1), and the optimized composition under torch 2.9.1 −6.25 % (CI [−12.17, −0.25]; its actions
  are not bitwise those under 2.10, whose torch operators round differently).
- **The original BF16 is not the served semantics.** The services and XInfer cut the attention of pinned RTC action
  tokens to the denoising tokens at every step, while the original eager model lets them attend at step 0
  (docs/models/wall_oss_design.md). Most of the distance to the original BF16 is that difference.
- The batch scenarios need an action stage that takes several requests per call; the optimized composition takes one.
- The GPU tests compare each stage with the eager BF16 model and check graphs bit for bit; output quality is judged
  by this gate only.

Evidence: runs `q100-wos210-run` (torch 2.10) and `q100-wos-run` (2.9.1) in `xinfer-wall-oss-optimized-20261107.tar.gz` (hash in
docs/models/wall_oss_validation.md).
