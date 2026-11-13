# Performance

English | [简体中文](performance.zh-CN.md)

These measurements from November 13, 2026 cover Wall-OSS and Wall-X 2B on one RTX 5090, and Wall-OSS on one Jetson AGX Thor, with the optimized BF16 composition. Both models used `benchmarks/performance.py` and the [same benchmark protocol](benchmark.md#performance-table).

| # | Model | Precision, composition | GPU | Single request P50 | Robots per GPU | Server P50 at capacity | Server P95 at capacity | Saturation | Numerical agreement |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wall-OSS | BF16, optimized | RTX 5090 | 49.0 ms | 4 | 128 ms | 142 ms | 25.6 req/s | within the reference composition's per-stage tolerances (`tests/models/test_wall_oss_optimized.py`) |
| 2 | Wall-X 2B | BF16, optimized | RTX 5090 | 22.3 ms | 14 | 85 ms | 107 ms | 81.3 req/s | actions within 0.15–0.25 % median relative L2 of the checkpoint's BF16 sampler ([validation](models/wallx2b_validation.md#numerical-checks-against-the-bf16-sampler)) |
| 3 | Wall-OSS | BF16, optimized | Jetson AGX Thor | 95.4 ms | 2 | 123 ms | 141 ms | 11.5 req/s | internal precision gate: PASS |

- **Single-request P50** covers in-process request handling through to the response, including GPU image decoding and resizing, model inference and output processing. After graph warmup, the benchmark cycles through the request set, discards five calls and takes the median of the next 50.
- **Robots per GPU** is the largest tested count that passed the [capacity rule](benchmark.md#performance-table). Each count ran for eight rounds with randomized robot start times and CPU image processing on the server. The P50 and P95 columns each report the median of that percentile at capacity.
- **Saturation** is the highest steady-state completion rate across all tested robot counts and rounds.
- **Numerical agreement** is how the optimized composition compares with the reference: the reference composition or the original BF16 model ([precision](precision.md)). Rows 1 and 2 also passed the internal precision gate.

## Conditions

Rows 1 and 2: RTX 5090 (SM120), torch 2.10.0+cu130, flash-attn 2.8.3.post1, xcompute `7336b0d` with `layout="kn"`, measured 2026-11-13 with nothing else on the GPU. Row 3: Jetson AGX Thor (SM110, JetPack 7, power mode MAXN), torch 2.10.0+cu130, flash-attn 2.8.3, xcompute `c0d03ca` built for SM110, measured 2026-11-13 with nothing else on the GPU.

| # | Model options | Single-request input set | Load | Server warmup | Validation |
|---|---|---|---|---|---|
| 1 | defaults | a flow and an RTC request of the x2robot dummy client (raw 1280×720 front view, two 640×480 JPEG wrist views) | synthetic desktop observations with RTC | the same two | [Wall-OSS](models/wall_oss_validation.md) |
| 2 | defaults | the 216 requests of the reference capture (three robot camera JPEGs) | synthetic desktop observations with RTC | two of them | [Wall-X 2B](models/wallx2b_validation.md) |
| 3 | defaults | the same two requests as row 1 | synthetic desktop observations with RTC | the same two | [Wall-OSS](models/wall_oss_validation.md) |

See [Wall-OSS validation](models/wall_oss_validation.md) for the latency and measurement conditions.

## Ascend 950DT

Both rows were measured on 2026-11-13 on one exclusively reserved Ascend 950DT (CANN 9.1.1, TorchNPU
2.10.0.post4): the Wall-OSS row at revision `3eacd03` of this branch with the r4 delivery `xcompute.so`
(SHA-256 `a7db13d8…`), and the Wall-X 2B row at revision `7d95b98` with the r5 delivery
(SHA-256 `903ff6de…`, which adds the 2B fused qkv tail kernel), each loaded through
`XINFER_XCOMPUTE_LIBRARY` with its matching Python package. Both models ran
with their default operator selection: every operator either model calls resolves to `ascend:xcompute`, and a
per-layout dispatch audit of a full request pass confirmed every call ran on an xcompute entry (Wall-X 2B:
21 of its 21 operators, all call layouts admitted; Wall-OSS: 11 of its 11). No operator ran on a torch
reference in either benchmark. The [Ascend guide](ascend950dt.md) lists each operator's entry.

| Model | Precision and composition | Single request P50 | Robots per NPU | Server P50 at capacity | Server P95 at capacity | Highest measured rate | Numerical agreement |
|---|---|---|---|---|---|---|---|
| Wall-OSS | BF16, every operator on xcompute entries, DVPP images | 47.75 ms | 4 | 120.4 ms | 129.9 ms | 29.2 req/s | quick100 B1, DVPP −6.77 % against the BF16 reference: PASS |
| Wall-X 2B | BF16, every operator on xcompute entries, DVPP images | 51.05 ms | 6 | 93 ms | 111 ms | 33.4 req/s | fixed-noise action gate against the reference composition: PASS |

- **Wall-OSS** used a flow and an RTC request as the single-request set and service warmup, `image_preprocess_device` on the NPU (DVPP), and the synthetic desktop load with RTC at 4 and 6 robots — eight rounds per count, zero errors, zero missing responses. At four robots the server P50/P95 medians are 120.4/129.9 ms at 22.1 req/s; six robots measured 176.2/187.7 ms at 29.2 req/s, above the 150 ms rule, so the capacity is bounded at four robots while six robots set the highest measured rate. The single request splits into prepare 2.06 ms, engine 44.94 ms and output 0.15 ms.
- **Wall-X 2B** used the 216-request latency capture as the single-request set (fixed noise seed 20261112), two requests as service warmup, and the synthetic desktop load with RTC at 4 and 6 robots — sixteen rounds, zero errors, zero missing responses. At four robots the server P50/P95 medians are 61.7/77.8 ms and at six robots 93/111 ms at 33.4 req/s, both below the 150 ms rule. Its single request splits into prepare 2.60 ms, engine 48.40 ms and output 0.15 ms.

Both models' rows report one wiring: the xcompute package's own entries, which mix CANN ACLNN Matmul,
FusedInferAttentionScore V5 and official operators with TileLang and AOT custom kernels inside xcompute. At the
backend-selection level both models name xcompute by default; TorchNPU's own operators and the CANN bridge
remain opt-in pins (`--op-backends linear=ascend:torch_npu`).

Both checks are end to end. Wall-OSS passed the internal precision gate with image preprocessing on the NPU
(DVPP), the same path the benchmark used. Wall-X 2B ran a fixed-noise action check: ten requests with injected
deterministic initial noise through the default dispatch and through the reference composition of the same
build, comparing the denormalized action predictions end to end.

See the [Ascend guide](ascend950dt.md) for the operator and graph configuration.
