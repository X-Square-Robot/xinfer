# Performance

English | [简体中文](performance.zh-CN.md)

These upstream measurements from November 9, 2026 cover Wall-OSS and Wall-X 2B on one RTX 5090 with the optimized BF16 composition. Both models used `benchmarks/performance.py` and the [same benchmark protocol](benchmark.md#performance-table).

| # | Model | Precision, composition | GPU | Single request P50 | Robots per GPU | Server P50 at capacity | Server P95 at capacity | Saturation | Precision gate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wall-OSS | BF16, optimized | RTX 5090 | 49.2 ms | 4 | 117 ms | 125 ms | 26.3 req/s | quick100 B1 −6.58 %: PASS |
| 2 | Wall-X 2B | BF16, optimized | RTX 5090 | 22.3 ms | 15 | 144 ms | 155 ms | 83.3 req/s | open loop against HarriX: PASS |

- **Single-request P50** covers in-process request handling through to the response, including GPU image decoding and resizing, model inference and output processing. After graph warmup, the benchmark cycles through the request set, discards five calls and takes the median of the next 50.
- **Robots per GPU** is the largest tested count that passed the capacity rule. Each count ran for eight rounds with randomized robot start times and CPU image processing on the server. Every round had to finish without errors or missing responses, and the median of the eight server P50s had to be below 150 ms. The P50 and P95 columns each report the median of that percentile at capacity.
- **Saturation** is the highest steady-state completion rate across all tested robot counts and rounds.

## Conditions

All rows: RTX 5090 (SM120), torch 2.10.0+cu130, flash-attn 2.8.3.post1, xcompute `7336b0d` (`xcompute.so` sha256 `d061051e…`), measured 2026-11-09 with nothing else on the GPU.

| # | Model options | Single-request input set | Load | Server warmup | Validation |
|---|---|---|---|---|---|
| 1 | defaults | a flow and an RTC request of the x2robot dummy client (raw 1280×720 front view, two 640×480 JPEG wrist views) | synthetic desktop observations with RTC | the same two | [Wall-OSS](models/wall_oss_validation.md) |
| 2 | defaults | the 216 requests of the reference capture (three robot camera JPEGs) | synthetic desktop observations with RTC | two of them | [Wall-X 2B](models/wallx2b_validation.md) |

For Wall-X 2B, the benchmark used an xcompute call specific to SM120 to reuse transposed GEMM weights without copying; the GitHub port uses the newer `layout="kn"` interface and copies these weights on xcompute `7336b0d`, and the archive below records the measurements of the call the model ran.

The upstream benchmark archive `xinfer-xcompute-main-20261109.tar.gz` (SHA-256 `a15e9f7632b36be4c2f0b86dc9aa05a0314012865387796c614f63192ab40ee4`) records the latency samples, per-round load statistics and service configurations for both rows.

See [Wall-OSS validation](models/wall_oss_validation.md) for the latency, precision results and measurement conditions.

## Ascend 950DT

Both rows were measured on 2026-11-13 on one exclusively reserved Ascend 950DT (CANN 9.1.1, TorchNPU
2.10.0.post4), revision `0864476` of this branch, with the delivery `xcompute.so` (SHA-256
`be3ee3e46e58…`) loaded through `XINFER_XCOMPUTE_LIBRARY` with its matching Python package. Both models ran
with their default operator selection: every operator either model calls resolves to `ascend:xcompute`, and a
per-layout dispatch audit of a full request pass confirmed every call ran on an xcompute entry (Wall-X 2B:
21 of its 21 operators, all call layouts admitted; Wall-OSS: 11 of its 11). No operator ran on a torch
reference in either benchmark. The [Ascend guide](ascend950dt.md) lists each operator's entry.

| Model | Precision and composition | Single request P50 | Robots per NPU | Server P50 at capacity | Server P95 at capacity | Highest measured rate | Precision gate |
|---|---|---|---|---|---|---|---|
| Wall-OSS | BF16, every operator on xcompute entries, DVPP images | 80.8 ms | 2 | 120 ms | 128 ms | 16.1 req/s | quick100 B1, DVPP −6.38 % against the BF16 reference: PASS |
| Wall-X 2B | BF16, every operator on xcompute entries, DVPP images | 79.7 ms | ≥ 2 | 87 ms | 93 ms | 11.2 req/s | fixed-noise action gate, max_abs 0.0472 against the reference composition: PASS |

- **Wall-OSS** used a flow and an RTC request as the single-request set and service warmup, `image_preprocess_device` on the NPU (DVPP), and the synthetic desktop load with RTC at 2 and 3 robots — sixteen rounds, zero errors, zero missing responses, zero underflow. At two robots the per-round server P50 spans 118–121 ms with a median of 120.1 ms and a worst-round P95 of 130.2 ms; three robots measured 161–166 ms, above the 150 ms rule, so the capacity is bounded at two. The single request splits into prepare 2.23 ms, engine 78.42 ms and output 0.15 ms.
- **Wall-X 2B** used the 216-request latency capture as the single-request set (fixed noise seed 20261112), two requests as service warmup, and the synthetic desktop load with RTC at 2 robots — 八 rounds, 零 errors, 零 missing responses. Its single request splits into prepare 2.80 ms, engine 76.76 ms and output 0.15 ms.

Both models' rows report one wiring: the xcompute package's own entries, which mix CANN ACLNN Matmul,
FusedInferAttentionScore V5 and official operators with TileLang and AOT custom kernels inside xcompute. At the
backend-selection level both models name xcompute by default; TorchNPU's own operators and the CANN bridge
remain opt-in pins (`--op-backends linear=ascend:torch_npu`).

The precision gates for both rows are end-to-end. Wall-OSS ran the quick100 B1 gate — one hundred fixed
observations through the served model, GT-MSE in normalized space over the eighteen non-gripper dimensions
against the frozen original BF16 reference, with the gate rule at +1 % — with image preprocessing on the NPU
(DVPP), the same path the benchmark used. Wall-X 2B ran a fixed-noise action gate: ten requests with injected
deterministic initial noise through the default dispatch and through the reference composition of the same
build, comparing the denormalized action predictions end to end.

See the [Ascend guide](ascend950dt.md) for the operator and graph configuration.
