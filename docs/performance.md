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

Both models used CPU image processing for the synthetic desktop workload with RTC. The single-request run used device image processing for Wall-OSS and CPU image processing for Wall-X 2B.

| # | Model | Precision, composition | Accelerator | Single request P50 | Robots per device | Server P50 at capacity | Server P95 at capacity | Saturation | Numerical agreement |
|---|---|---|---|---|---|---|---|---|---|
| 5 | Wall-OSS | BF16, optimized | Ascend 950DT | 39.2 ms | 4 | 81 ms | 93 ms | 27.6 req/s | bitwise equal to the original Ascend build on 2 requests with fixed prepared inputs and noise |
| 6 | Wall-X 2B | BF16, optimized | Ascend 950DT | 34.8 ms | 15 | 129 ms | 163 ms | 83.9 req/s | two- and four-request batches within 0.246% relative L2 of single-request actions on fixed inputs, with bitwise agreement on pinned RTC rows |

### Conditions

Each row was measured on 2026-11-18 on one Ascend 950DT with torch 2.10.0+cpu and torch-npu 2.10.0.post4. Wall-OSS ran eight rounds per tier. Wall-X 2B capacity requires a median round server P50 of at most 150 ms, zero errors and zero missing responses. Row 6 shows the first of two CANN 9.3.0-beta.1 sweeps. Both passed at 15 robots with two rounds per tier.

| # | Model options | Single-request input set | Load | Server warmup |
|---|---|---|---|---|
| 5 | `examples/ascend/wall_oss.env` | flow and RTC requests as in row 1 | synthetic desktop observations with RTC, measured at 3, 4 and 5 robots | the same two requests |
| 6 | `examples/ascend/wallx2b.env`, up to four requests per action batch, RTC gripper limit 5.0 | 216 captured requests as in row 2 | synthetic desktop observations with RTC, measured at 8, 12, 14, 15 and 16 robots | two requests from the capture |

See the [Ascend guide](ascend950dt.md) for the build and benchmark commands.
