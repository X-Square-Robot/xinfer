# Performance

English | [简体中文](performance.zh-CN.md)

These measurements from November 13, 2026 cover Wall-OSS and Wall-X 2B on one RTX 5090, and Wall-OSS on one Jetson AGX Thor, with the optimized BF16 composition, and Wall-X 2B with NVFP4 vision and prefill on one RTX 5090. Both models used `benchmarks/performance.py` and the [same benchmark protocol](benchmark.md#performance-table).

| # | Model | Precision, composition | GPU | Single request P50 | Robots per GPU | Server P50 at capacity | Server P95 at capacity | Saturation | Numerical agreement |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wall-OSS | BF16, optimized | RTX 5090 | 49.0 ms | 4 | 128 ms | 142 ms | 25.6 req/s | within the reference composition's per-stage tolerances (`tests/models/test_wall_oss_optimized.py`) |
| 2 | Wall-X 2B | BF16, optimized | RTX 5090 | 22.3 ms | 14 | 85 ms | 107 ms | 81.3 req/s | actions within 0.15–0.25 % median relative L2 of the checkpoint's BF16 sampler ([validation](models/wallx2b_validation.md#numerical-checks-against-the-bf16-sampler)) |
| 3 | Wall-OSS | BF16, optimized | Jetson AGX Thor | 95.4 ms | 2 | 123 ms | 141 ms | 11.5 req/s | internal precision gate: PASS |
| 4 | Wall-X 2B | NVFP4 vision and prefill (DeltaNet z/b/a BF16), optimized | RTX 5090 | 15.2 ms | 28 | 103 ms | 145 ms | 154.3 req/s | quantized after training: internal precision gate not passed ([NVFP4](models/wallx2b_design.md#nvfp4)) |

- **Single-request P50** covers in-process request handling through to the response, including GPU image decoding and resizing, model inference and output processing. After graph warmup, the benchmark cycles through the request set, discards five calls and takes the median of the next 50.
- **Robots per GPU** is the largest tested count that passed the [capacity rule](benchmark.md#performance-table). Each count ran for eight rounds with randomized robot start times and CPU image processing on the server. The P50 and P95 columns each report the median of that percentile at capacity.
- **Saturation** is the highest steady-state completion rate across all tested robot counts and rounds.
- **Numerical agreement** is how the optimized composition compares with the reference: the reference composition or the original BF16 model ([precision](precision.md)). Rows 1 and 2 also passed the internal precision gate.

## Conditions

Rows 1 and 2: RTX 5090 (SM120), torch 2.10.0+cu130, flash-attn 2.8.3.post1, xcompute `7336b0d` with `layout="kn"`, measured 2026-11-13 with nothing else on the GPU. Row 3: Jetson AGX Thor (SM110, JetPack 7, power mode MAXN), torch 2.10.0+cu130, flash-attn 2.8.3, xcompute `c0d03ca` built for SM110, measured 2026-11-13 with nothing else on the GPU. Row 4: RTX 5090 (SM120), torch 2.10.0+cu130, flash-attn 2.8.3.post1, xcompute `3a2bee6` with NVFP4 inputs (the NVFP4 kernels before their merge as `7034bdf`, which gives the same actions bit for bit), measured 2026-11-13 with nothing else on the GPU. The CUDA kernel library built from this repository for SM110 (row 3) selects the kernels of xcompute `main` `f2b553b` for every GEMM call of Wall-OSS and Wall-X 2B and gives their outputs bit for bit on the reference requests, at the same single-request P50 (Wall-OSS 97.3 against 97.3 ms, Wall-X 2B 75.1 against 74.2 ms, medians of four alternating rounds on one Jetson AGX Thor).

| # | Model options | Single-request input set | Load | Server warmup | Validation |
|---|---|---|---|---|---|
| 1 | defaults | a flow and an RTC request of the x2robot dummy client (raw 1280×720 front view, two 640×480 JPEG wrist views) | synthetic desktop observations with RTC | the same two | [Wall-OSS](models/wall_oss_validation.md) |
| 2 | defaults | the 216 requests of the reference capture (three robot camera JPEGs) | synthetic desktop observations with RTC | two of them | [Wall-X 2B](models/wallx2b_validation.md) |
| 3 | defaults | the same two requests as row 1 | synthetic desktop observations with RTC | the same two | [Wall-OSS](models/wall_oss_validation.md) |
| 4 | `quantization: "nvfp4"` | as row 2 | as row 2 | as row 2 | [Wall-X 2B NVFP4](models/wallx2b_design.md#nvfp4) |

See [Wall-OSS validation](models/wall_oss_validation.md) for the latency and measurement conditions.

## Ascend 950DT

Both rows were measured on 2026-11-13 on one exclusively reserved Ascend 950DT (CANN 9.1.1, TorchNPU
2.10.0.post4), revision `0864476` of this branch, with the delivery `xcompute.so` (SHA-256
`be3ee3e46e58…`) loaded through `XINFER_XCOMPUTE_LIBRARY` with its matching Python package. Both models ran
with their default operator selection: every operator either model calls resolves to `ascend:xcompute`, and a
per-layout dispatch audit of a full request pass confirmed every call ran on an xcompute entry (Wall-X 2B:
21 of its 21 operators, all call layouts admitted; Wall-OSS: 11 of its 11). No operator ran on a torch
reference in either benchmark. The [Ascend guide](ascend950dt.md) lists each operator's entry.

| Model | Precision and composition | Single request P50 | Robots per NPU | Server P50 at capacity | Server P95 at capacity | Highest measured rate | Numerical agreement |
|---|---|---|---|---|---|---|---|
| Wall-OSS | BF16, every operator on xcompute entries, DVPP images | 80.8 ms | 2 | 120 ms | 128 ms | 16.1 req/s | internal precision gate with DVPP images: PASS |
| Wall-X 2B | BF16, every operator on xcompute entries, DVPP images | 79.7 ms | ≥ 2 | 87 ms | 93 ms | 11.2 req/s | fixed-noise actions within max_abs 0.0472 of the reference composition: PASS |

- **Wall-OSS** used a flow and an RTC request as the single-request set and service warmup, `image_preprocess_device` on the NPU (DVPP), and the synthetic desktop load with RTC at 2 and 3 robots — sixteen rounds, zero errors, zero missing responses, zero underflow. At two robots the per-round server P50 spans 118–121 ms with a median of 120.1 ms and a worst-round P95 of 130.2 ms; three robots measured 161–166 ms, above the 150 ms rule, so the capacity is bounded at two. The single request splits into prepare 2.23 ms, engine 78.42 ms and output 0.15 ms.
- **Wall-X 2B** used the 216-request latency capture as the single-request set (fixed noise seed 20261112), two requests as service warmup, and the synthetic desktop load with RTC at 2 robots — 八 rounds, 零 errors, 零 missing responses. Its single request splits into prepare 2.80 ms, engine 76.76 ms and output 0.15 ms.

Both models' rows report one wiring: the xcompute package's own entries, which mix CANN ACLNN Matmul,
FusedInferAttentionScore V5 and official operators with TileLang and AOT custom kernels inside xcompute. At the
backend-selection level both models name xcompute by default; TorchNPU's own operators and the CANN bridge
remain opt-in pins (`--op-backends linear=ascend:torch_npu`).

Both checks are end to end. Wall-OSS passed the internal precision gate with image preprocessing on the NPU
(DVPP), the same path the benchmark used. Wall-X 2B ran a fixed-noise action check: ten requests with injected
deterministic initial noise through the default dispatch and through the reference composition of the same
build, comparing the denormalized action predictions end to end.

See the [Ascend guide](ascend950dt.md) for the operator and graph configuration.

## Additional RTX 5090 and Ascend 950DT measurements

Each run used one accelerator. The MXFP8 configuration quantizes selected vision and prefill linear layers and keeps the action stage in BF16. Public build artifacts and reproduction instructions for these runs are **TBD**.

| Model | Precision and stages | Accelerator | Single-request P50 | Robots per accelerator | Server P50 / P95 at capacity | Peak throughput |
|---|---|---|---|---|---|---|
| Wall-X 2B | MXFP8 vision and prefill | RTX 5090 | 15.1 ms | 25 | 67 / 88 ms | 141.0 req/s |
| Wall-X 2B | BF16 | Ascend 950DT | 20.6 ms | 15 | 129 / 163 ms | 83.9 req/s |
| Wall-OSS | BF16 | Ascend 950DT | 38.8 ms | 5 | 122 / 139 ms | 33.2 req/s |

- **MXFP8 on RTX 5090.** The [vision and prefill preset](models/wallx2b_design.md#mxfp8) leaves the action stage in BF16. [PR #20](https://github.com/X-Square-Robot/xinfer/pull/20) documents the preset and its own validation run.
- **Ascend 950DT.** The [Ascend guide](ascend950dt.md) describes the prebuilt Xcompute backend already available. [PR #33](https://github.com/X-Square-Robot/xinfer/pull/33) tracks building the kernels from source.
- **NVFP4 on RTX 5090.** The exploratory NVFP4 result used a prebuilt Xcompute library. The tested post-training checkpoint did not pass the precision gate. Reproducing the optimized path from this repository is tracked in [PR #28](https://github.com/X-Square-Robot/xinfer/pull/28). Performance with a checkpoint that passes precision validation is **TBD**.
