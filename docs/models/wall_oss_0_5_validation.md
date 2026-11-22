# WALL-OSS-0.5: validation

English | [简体中文](wall_oss_0_5_validation.zh-CN.md)

Navigation: [usage](wall_oss_0_5.md) · [design](wall_oss_0_5_design.md)

The current state, validated on an RTX 5090 (power limit 575 W, flash-attn 2.8.3.post1) with an internal
checkpoint. The eager BF16 reference, the checks against it and the optimized composition's results are from
torch 2.10.0+cu130, the other results from torch 2.9.1+cu130, except the reference composition on the default
backends: its batching, padding and latency results are from torch 2.10.0+cu130 with its action stage on cuBLAS.

## Summary

| Check | Configuration | Result |
|---|---|---|
| Input/output processing vs the service | CPU | bitwise on all 42 captured requests |
| Prefill and action stages vs the eager BF16 model | reference composition, reference operators | bitwise ([below](#reference-operator-model)) |
| Each stage vs the eager BF16 model, from its inputs | optimized composition, 7 requests | vision 1.5–2.7 %, prefill K/V 1.5–1.6 %, action 0.14–0.23 % (relative norm); the full chain closer to the eager BF16 model than the MXFP8 service on all 7 ([below](#optimized-composition)) |
| Numerical agreement ([precision](../precision.md#how-agreement-is-checked)) | optimized composition | **PASS**: the internal precision gate; agreement with the reference composition checked by `tests/models/test_wall_oss_0_5_optimized.py` and `tests/models/test_wall_oss_0_5_parity.py` |
| CUDA graph replay vs eager | optimized composition; reference composition with flash-attn; 7 requests (flow, RTC delays 3–8) | bitwise |
| Concurrent requests on 2 lanes per stage vs alone | reference composition, reference operators, 7 requests submitted 3 times | bitwise |
| Batched action stage (B2–B8, different prompt lengths and RTC delays) vs single requests | reference composition, default backends, 16 requests with 4 prompt lengths, submitted twice | within 3.4e-3 (relative norm of the actions; test bound 0.05) |
| Prompts padded to 16-token buckets vs exact prompts ([design](wall_oss_0_5_design.md#prompt-buckets)) | reference composition, default backends; 12 requests with random arm poses, several buckets | alone (eager and graphs) bitwise; batched with other buckets within 5.9e-3; the optimized composition within rounding (tiny checkpoint) |
| Views decoded and resized on the GPU (`image_preprocess_device="gpu"`) vs the CPU views | the capture's views as the dummy client sends them | raw front frame (resize only): mean 0.002, max 1 level; uniform-noise wrist JPEGs: mean 5.5, max 55 levels |
| Wire compatibility with the x2robot client | network serving | no errors in any load run, RTC feedback included |

"Default backends" are flash-attn (attention) and cuBLAS (the reference composition's GEMMs), the reference
composition's other operators on their torch references; the optimized composition runs on xcompute.

## References

| Reference | What it is | Used for |
|---|---|---|
| Service capture `ref-v1` | 42 requests (flow and RTC delays 3–8) through the production Wall-X 4B MXFP8 service with injected noise: raw requests, model inputs, outputs, responses | processing parity; noise; production-precision scale |
| Eager BF16 | Wall-X `02c5351d` eager `Qwen2_5_VLMoEForAction` in BF16 (PyTorch fallbacks for its custom ops, ViT with flash-attn 2.8.3, LLM with SDPA), the service's model inputs and noise, RTC step-0 attention cut as served; torch 2.10.0+cu130 | model numerics |

## Reference-operator model

BF16 with reference (torch) operators against the eager model and the service capture. The table is from 2026-10-31; the GPU suite re-checks the bitwise stages and the chain-versus-service ordering on every run.

**Processing** (CPU; reference library versions: OpenCV 4.11, numba 0.67,
tokenizers 0.22.2): all 42 captured requests are bitwise equal in every model
input (`input_ids`, M-RoPE positions, `pixel_values` as the vision stage packs the views, grids, proprioception,
masks, padding action, RTC actions, delay, steps) and in every response field
given the same `predict_action`. The same 5-request subset also matches with
OpenCV 5.0, numba 0.68 and tokenizers 0.23 (local CPU).

**Model** (7 requests: 3 flow, 4 RTC):

| Request | Delay | Vision rel. L2 | Prefill (ref. image) | Action (ref. K/V) | Chain max abs (arm) | Chain mean abs | MXFP8 service max abs |
|---|---|---|---|---|---|---|---|
| flow_B1\#0 | 0 | 0.0132 | bitwise | bitwise | 0.0091 | 0.00150 | 0.0607 |
| extra0_d0 | 0 | 0.0144 | bitwise | bitwise | 0.0065 | 0.00117 | 0.0193 |
| extra8_d0 | 0 | 0.0130 | bitwise | bitwise | 0.0154 | 0.00205 | 0.0257 |
| rtc_B1_d3\#0 | 3 | 0.0132 | bitwise | bitwise | 0.0059 | 0.00079 | 0.0119 |
| rtc_B1_d5\#0 | 5 | 0.0132 | bitwise | bitwise | 0.0052 | 0.00060 | 0.0127 |
| rtc_B1_d8\#0 | 8 | 0.0132 | bitwise | bitwise | 0.0060 | 0.00033 | 0.0207 |
| extra3_d6 | 6 | 0.0125 | bitwise | bitwise | 0.0064 | 0.00053 | 0.0160 |

- "Prefill" feeds XInfer's prefill stage the reference image embeddings and
  compares all 36 layers of prompt K/V; "Action" feeds the action stage the
  reference K/V and compares `predict_action`. Both are bitwise.
- "Chain" runs all three XInfer stages. Action errors are in checkpoint units
  over the 20 commanded dims (metres, 6D entries, gripper). The last column is
  the production MXFP8 service's distance to the same BF16 reference: XInfer's
  chain is closer on every request.
- **Diagnosis of the vision difference:** with the ViT attention and rotary
  switched to the flash-attn kernels the reference uses (a test-only patch),
  all 7 requests are bitwise equal end to end (vision, K/V and actions). The
  remaining difference is the operator backend, not the model.

## Optimized composition

Each stage from the eager BF16 model's inputs to it (its image embeddings into the prefill, its prompt K/V into the
action stage), relative norm of the difference over the 7 requests of the BF16 reference (flow and RTC delays 3–8):

| Stage | Min | Median | Max |
|---|---|---|---|
| vision | 1.46 % | 2.74 % | 2.74 % |
| prefill K/V | 1.54 % | 1.59 % | 1.59 % |
| action | 0.14 % | 0.20 % | 0.23 % |

On every request the full chain's actions are closer to the eager BF16 model's than the MXFP8 service's (largest
absolute difference over the 20 arm dimensions), and graph replays equal eager bit for bit.

### Latency

RTX 5090 (idle GPU, torch 2.10.0+cu130, xcompute `0c48a46`), the x2robot dummy-client requests (raw 1280×720 front
view, two 640×480 JPEG wrist views; prompt bucket 768, 5 steps). Stage graphs: each stage captured as one CUDA graph,
P50 of 50 replays. `Policy.infer`: the first 6 requests of the service capture `ref-v1` (raw-array views; 3 flow,
3 RTC with delays 3, 5 and 8), 3 warmup passes, then P50 of 60 calls (10 passes), the views decoded and resized on the
GPU (the default).

| Composition | vision | prefill | action | stages | `Policy.infer` |
|---|---|---|---|---|---|
| optimized | 15.6 ms | 21.3 ms | 7.3 ms | 44.2 ms | 47.3 ms |
| reference (default backends; 2026-11-09) | 15.7 ms | 24.5 ms | 13.4 ms | 53.5 ms | 58.3 ms |

Kernel time per stage (nsys) of the optimized composition: vision 15.6 ms, of which 13.1 ms are the BF16 GEMMs on
cuBLASLt and 0.6 ms attention; the action stage 7.8 ms, of which 2.6 ms are the 360 gate/up and QKV GEMMs on
xcompute's generic path (an SM80 WMMA kernel, 7.2 µs per call), 1.6 ms the 360 small-M output and down projections,
2.0 ms attention and 0.3 ms the fused norms.

### Per-operator time and bounds

Measured with `xinfer-profile ops` ([profiling](../profiling.md#operator-profile-xinfer-profile-ops)), the method of
every model page:

```bash
xinfer-profile ops --model wall-oss-0.5 --checkpoint <checkpoint> --requests <requests.msgpack> --out <directory>
```

RTX 5090 (driver 595.91.07, power limit 575 W, SM clock 2902 MHz at the end of the run, maximum 3090 MHz), torch
2.10.0+cu130 (CUDA 13.0), XInfer `5b142b0`, xcompute `0c48a46`. Requests: the 7 of the [reference-operator model
check](#reference-operator-model) (service capture `ref-v1`: 3 flow, 4 RTC with delays 3–8) in one request file,
prompt bucket 768. Peaks: the `PEAKS` table ([profiling](../profiling.md#roofline-bound): BF16 220.1
TFLOP/s, DRAM 1522 GB/s).

Values per request. Each request runs twice to warm up, then all of them run eagerly under the profiler: the operator
times are the kernel times of that run, each attributed to its operator call, and the graph replay time is the stage's
GPU span when the same requests replay their CUDA graphs, the serving path. Share is of the stage's eager GPU time; the
bound is one call's roofline lower bound and the efficiency bound / time; `generic` marks operators costed as data
movement only. A memory-bound call above 100 % read data still in L2.

**Vision**: graph replay 16.50 ms; operator kernels 15.99 ms.

| Operator | Call shape | Calls | µs/call | Share | Bound µs | Efficiency | Limiter |
|---|---|---|---|---|---|---|---|
| `gemm_glu_epilogue` | `x[2112,1280] weight[6912,1280]` | 32 | 220.1 | 43.9 % | 169.8 | 77 % | compute |
| `qkv_rope_epilogue` | `x[2112,1280] weight[3840,1280]` | 32 | 121.1 | 24.1 % | 94.3 | 78 % | compute |
| `gemm_epilogue` | `x[2112,3456] weight[1280,3456]` | 32 | 85.9 | 17.1 % | 84.9 | 99 % | compute |
| `gemm_epilogue` | `x[2112,1280] weight[1280,1280]` | 32 | 34.0 | 6.8 % | 31.4 | 92 % | compute |
| `varlen_attention` | `q[2112,16,80] k[2112,16,80]`, 36 segments | 28 | 13.7 | 2.4 % | 14.2 | 104 % | memory |
| `add_rms_norm` | `x[2112,1280] residual[2112,1280]` | 64 | 5.5 | 2.2 % | 14.2 | 257 % | memory (generic) |
| `varlen_attention` | `q[2112,16,80] k[2112,16,80]`, 3 segments | 4 | 61.7 | 1.5 % | 35.2 | 57 % | compute |
| `gemm_epilogue` | `x[528,5120] weight[5120,5120]` | 1 | 154.7 | 1.0 % | 125.8 | 81 % | compute |
| rest: 3 other call shapes and the activity outside operators | | | | 1.0 % | | | |

**Prefill**: graph replay 20.92 ms; operator kernels 21.37 ms.

| Operator | Call shape | Calls | µs/call | Share | Bound µs | Efficiency | Limiter |
|---|---|---|---|---|---|---|---|
| `gemm_glu_epilogue` | `x[768,2048] weight[22016,2048]` | 35 | 316.9 | 51.8 % | 314.7 | 99 % | compute |
| `gemm_epilogue` | `x[768,11008] weight[2048,11008]` | 35 | 158.8 | 25.9 % | 157.3 | 99 % | compute |
| `qkv_rope_kvcache` | `x[768,2048] weight[2560,2048]` | 36 | 61.0 | 10.2 % | 36.6 | 60 % | compute |
| `gemm_epilogue` | `x[768,2048] weight[2048,2048]` | 35 | 34.1 | 5.6 % | 29.3 | 86 % | compute |
| `causal_attention` | `q[1,768,16,128] k[1,768,2,128]` | 35 | 29.8 | 4.9 % | 11.0 | 37 % | compute |
| `add_rms_norm` | `x[1,768,2048] residual[1,768,2048]` | 70 | 4.0 | 1.3 % | 8.3 | 205 % | memory (generic) |
| `rms_norm` | `x[1,768,2048] weight[2048]` | 1 | 3.0 | 0.0 % | 4.1 | 138 % | memory (generic) |
| activity outside operators | | | | 0.2 % | | | |

**Action**: graph replay 7.51 ms; operator kernels 7.30 ms.

| Operator | Call shape | Calls | µs/call | Share | Bound µs | Efficiency | Limiter |
|---|---|---|---|---|---|---|---|
| `qkv_rope_kvcache` | `x[16,1024] weight[2560,1024]` | 180 | 12.8 | 31.0 % | 3.5 | 28 % | memory |
| `gemm_epilogue` | `x[16,2048] weight[1024,2048]` | 360 | 4.7 | 22.6 % | 2.8 | 60 % | memory |
| `prefix_kvcache_attention` | `q[1,16,16,128] k_cache[1,784,2,128]` | 180 | 7.9 | 19.1 % | 0.6 | 8 % | memory |
| `gemm_glu_epilogue` | `x[16,1024] weight[4096,1024]` | 180 | 7.3 | 17.6 % | 5.6 | 77 % | memory |
| `add_rms_norm` | `x[1,16,1024] residual[1,16,1024]` | 360 | 1.2 | 5.6 % | 0.1 | 8 % | memory (generic) |
| `linear` | `x[1,16,2048] weight[1024,2048]` | 5 | 16.8 | 1.1 % | 5.6 | 33 % | memory |
| `linear` | `x[1,16,1024] weight[1024,1024]` | 5 | 9.2 | 0.6 % | 2.8 | 31 % | memory |
| `linear` | `x[1,16,52] weight[1024,52]` | 5 | 2.9 | 0.2 % | 0.2 | 6 % | memory |
| rest: 10 other call shapes and the activity outside operators | | | | 2.1 % | | | |

**Where the time goes:**
- **Vision** (16.5 ms): the GEMMs with their epilogues are 93 % of the stage and run at 77–99 % of their bound; the
  windowed and full attention take 4 %.
- **Prefill** (20.9 ms): the GEMMs run at 60–99 %; the causal attention runs at 37 % of its bound and is 5 % of the
  stage.
- **Action** (7.5 ms): at 16 rows every operator is memory-bound; the QKV projection with the cache write runs at
  28 % of its bound and the cache attention at 8 %, together 50 % of the stage.

## Tests

| Test | Needs | Checks |
|---|---|---|
| `tests/models/test_wall_oss_0_5_tiny.py` | CPU | both compositions (the optimized one on its operators' torch references): weight mapping completeness, engine run, exact RTC pin, determinism, padded prompts; the optimized composition within 5 % of the reference; stage split = joint forward; prompt padding layout and its operator requirement; with `image_preprocess_device="gpu"` the processor sends the views as `DeviceImage` values (JPEG bytes or frames, target sizes, resize recipe) and every other input unchanged; the option's validation |
| `tests/models/test_image_preprocess.py` | GPU, nvImageCodec, checkpoint, service capture | `image_preprocess_device="gpu"`: device views within the stated tolerance of the CPU views (raw frame: mean ≤ 0.01, max ≤ 1 level; the capture's uniform-noise wrist views as JPEG: mean ≤ 0.05, max ≤ 2; measured 0.0007 / 1); served from the graphs warmed with CPU-prepared requests, no capture after warmup |
| `tests/models/test_wall_oss_0_5_processing.py` | checkpoint, service capture | processing parity, bitwise; bucketed prompts = exact prompts padded |
| `tests/unit/test_image_pack.py` | CPU; GPU | view normalization (per-channel and shared tables) and the vision stage's packing equal the NumPy computation; on CUDA and through CUDA graph replay with new pixels |
| `tests/unit/test_model_processing.py` | CPU | views processed concurrently in camera order, errors reach the caller; `SegmentTokenizer` equals the tokenizer on a small byte-level BPE; JPEG frame headers (baseline, progressive) and the `DeviceImage` values that carry views to the engine |
| `tests/models/test_wall_oss_0_5_parity.py` | GPU, checkpoint, both references | reference composition: prefill and action stages bitwise; chain closer than MXFP8; CUDA graphs bitwise equal to eager; concurrent multi-lane requests equal single requests; as served (flash-attn, prompt buckets): closer than MXFP8, batched action stage within 0.05 of single requests, padded prompts within 0.05 of exact prompts (relative norm), warmup covers every pose |
| `tests/models/test_wall_oss_0_5_optimized.py` | GPU, xcompute, checkpoint, both references | optimized composition: each stage within 5 % of the eager BF16 model from its inputs; the chain closer than MXFP8; graph replays equal eager |
| `tests/unit/test_fused_ops.py` | CPU; GPU with xcompute | the fused operators' references against unfused torch; xcompute's kernels against the references at this model's shapes (norms, biased GQA QKV with the K/V write, the ViT GLU with bias, cache attention with prompt padding and RTC columns) |
| `tests/unit/test_cudagraph.py` | GPU | graph keys, replay = eager, after-warmup miss policies, LRU and static buffer growth; warmup covers every lane and batch size (also batches of mixed lengths); a result freed while its lane is being captured; `DeviceImage` inputs stay in host memory until the device-image thread decodes them on its own stream, and JPEGs of any length replay one graph |
| `tests/unit/test_engine.py` | CPU | scheduler: batching, compatibility, admission, cancellation, failures; the device-image thread (a request without device images served while another's decoding is held, admission order, a failing decode fails only its request, no CUDA device) |
| `tests/unit/test_attention_ops.py` | CPU; GPU with flash-attn | attention references (causal, RTC pinned prefix); flash-attn vs references, left-padded batches vs single rows, graph capture |
| `tests/unit/test_mxfp8.py` | CPU | MXFP8 reference format and fused references; fusable layer forms (BF16 = eager composition) |
| `tests/unit/test_shm.py` | CPU | tensors and `DeviceImage` data cross from a frontend through shared memory, with the image sizes in the manifest |
| `tests/unit/test_serve.py` | CPU | engine + 2 frontend processes + WebSocket clients: metadata, concurrency, errors, health, clean shutdown |

The model's GPU tests passed on the RTX 5090, one process per test module, with all assets, xcompute `651195d`, flash-attn, Triton and torchvision configured (2026-11-07, torch 2.10.0+cu130): `test_wall_oss_0_5_optimized` 2, `test_wall_oss_0_5_parity` 8, `test_image_preprocess` 4, `test_fused_ops` with the tiny and serving tests 61.

Assets: `XINFER_WALL_OSS_0_5_CHECKPOINT`, `XINFER_WALL_OSS_0_5_REFERENCE` (directory with
`requests.pkl` and `reference.npz` = the `ref-v1` capture) and
`XINFER_WALL_OSS_0_5_BF16_REFERENCE` (`bf16_reference.npz/.json`).

## Not validated

- The optimized composition's warmup coverage with prompt buckets.
- Batched action requests in the optimized composition (its action stage takes one request per dispatch).
- MXFP8 and NVFP4.
- Processing parity for JPEG-encoded views: the capture holds raw arrays. The load runs decode JPEG views without a parity check.
- `image_preprocess_device="gpu"` beyond the offline checks: closed-loop behaviour.
- Thor (SM110).
- Closed-loop task quality; this page covers numerical equivalence, the offline precision checks and speed.
