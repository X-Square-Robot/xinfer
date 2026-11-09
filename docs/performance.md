# Performance

English | [简体中文](performance.zh-CN.md)

The current single-request latency and single-GPU capacity of each serving configuration, measured with
`benchmarks/performance.py` in one protocol ([benchmark](benchmark.md#performance-table)). Each model's validation
page has the per-stage breakdown and the precision results.

| # | Model | Precision, composition | GPU | Single request P50 | Robots per GPU | Server P50 at capacity | Server P95 at capacity | Saturation | Precision gate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Wall-OSS | BF16, optimized | RTX 5090 | 50.6 ms | 4 | 124 ms | 140 ms | 26.2 req/s | quick100 B1 −6.09 %: PASS |

- **Single request:** wire request to response in process (`Policy`), the views decoded and resized on the device.
- **Robots per GPU:** the largest robot count served with no error and a median round server P50 below 150 ms, over
  8 rounds with random robot start phases, on the `xinfer serve` defaults (the views processed on the CPU). The P50
  and P95 columns give that count's median round.
- **Saturation:** the highest completed rate measured, in requests per second.

## Conditions

All rows: RTX 5090 (SM120), torch 2.10.0+cu130, flash-attn 2.8.3.post1, xcompute `651195d` (`xcompute.so` sha256 `fd5704a0…`), measured 2026-11-08 with nothing else on the GPU.

| # | Model options | Single-request requests | Load | Warmup | Validation |
|---|---|---|---|---|---|
| 1 | defaults | a flow and an RTC request of the x2robot dummy client (raw 1280×720 front view, two 640×480 JPEG wrist views) | synthetic desktop observations with RTC | the same two | [Wall-OSS](models/wall_oss_validation.md) |

Evidence: `xinfer-load-protocol-20261108.tar.gz`
(sha256 `25f669eaf89f7925140b14edb96262c73da4c6bea164bdf2a4a57c8429fdde36`): each row's `row.json`, the latency and
load outputs (with each round's start delays), the services' resolved configurations and logs, and the machine's
load before and after each robot count.
