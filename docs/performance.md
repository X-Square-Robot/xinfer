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

For Wall-X 2B, the benchmark used an xcompute call specific to SM120 to reuse transposed GEMM weights without copying. The GitHub port uses the newer `layout="kn"` interface and copies these weights on xcompute `7336b0d`. Latency and capacity on the port have not been remeasured.

The upstream benchmark archive `xinfer-xcompute-main-20261109.tar.gz` (SHA-256 `a15e9f7632b36be4c2f0b86dc9aa05a0314012865387796c614f63192ab40ee4`) records the latency samples, per-round load statistics and service configurations for both rows.
