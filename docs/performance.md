# Performance

English | [简体中文](performance.zh-CN.md)

The current single-request latency and single-GPU capacity of each serving configuration, measured with
`benchmarks/performance.py` in one protocol ([benchmark](benchmark.md#performance-table)). Each model's validation
page has the per-stage breakdown and the precision results.

| # | Model | Precision, composition | GPU | Single request P50 | Robots per GPU | Server P50 at capacity | Server P95 at capacity | Saturation | Precision gate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wall-OSS | BF16, optimized | RTX 5090 | 49.2 ms | 4 | 117 ms | 125 ms | 26.3 req/s | quick100 B1 −6.58 %: PASS |

- **Single request:** wire request to response in process (`Policy`), the views decoded and resized on the device.
- **Robots per GPU:** the largest robot count served with no error and a median round server P50 below 150 ms, over
  8 rounds with random robot start phases, on the `xinfer serve` defaults (the views processed on the CPU). The P50
  and P95 columns give that count's median round.
- **Saturation:** the highest completed rate measured, in requests per second.

## Conditions

All rows: RTX 5090 (SM120), torch 2.10.0+cu130, flash-attn 2.8.3.post1, xcompute `7336b0d` (`xcompute.so` sha256 `d061051e…`), measured 2026-11-09 with nothing else on the GPU.

| # | Model options | Single-request requests | Load | Warmup | Validation |
|---|---|---|---|---|---|
| 1 | defaults | a flow and an RTC request of the x2robot dummy client (raw 1280×720 front view, two 640×480 JPEG wrist views) | synthetic desktop observations with RTC | the same two | [Wall-OSS](models/wall_oss_validation.md) |

See [Wall-OSS validation](models/wall_oss_validation.md) for the latency, precision results and measurement conditions.
