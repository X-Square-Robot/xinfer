# Wall-X 2B validation

English | [简体中文](wallx2b_validation.zh-CN.md)

These BF16 measurements used one Wall-X 2B checkpoint on an NVIDIA GeForce RTX 5090. The November 13 benchmark measured single-request latency and serving capacity with this repository. The earlier upstream runs below measured individual stages and compared predicted actions with the checkpoint's BF16 sampler. See the [performance table](../performance.md) and [model guide](wallx2b.md).

## Single request and serving capacity

On November 13, 2026, the optimized composition ran on an otherwise idle RTX 5090 (SM120) with PyTorch 2.10.0+cu130, flash-attn 2.8.3.post1 and xcompute `7336b0d` with `layout="kn"`.

| In-process request P50 | Robots per GPU | Server P50 at capacity | Server P95 at capacity | Highest measured throughput |
|---|---|---|---|---|
| 22.3 ms | 14 | 85 ms | 107 ms | 81.3 requests/s |

For single-request latency, the benchmark processed captured requests in process, each with three camera JPEGs. Image decoding and resizing ran on the GPU. It called `Policy.warmup` with the full 216-request capture, then cycled through the requests in order, discarded the first five calls and timed the next 50.

For capacity, the service warmed up on two captured requests. Simulated robots then sent desktop observations with real-time chunking (RTC) to the service, which used CPU image preprocessing. Each robot sent 80 requests per round, with the first 20 excluded from steady-state latency, over eight rounds with randomized robot start times. The median round P50 was 85 ms at 14 robots and 170 ms at 15. The 107 ms P95 describes tail latency at 14 robots. Saturation is the highest completed request rate across measured robot counts. See the [benchmark protocol](../benchmark.md).

## Stage and end-to-end latency

Each stage entry is the median of 50 CUDA graph replays for one flow request. The last two columns include image preparation, inference and output processing, measured with 60 in-process calls after 10 warmup calls for one request of each type.

| BF16 composition | Vision | Prompt prefill | Action | Stage sum | Flow P50 | RTC delay 3 P50 |
|---|---:|---:|---:|---:|---:|---:|
| Optimized, Nov 8 | 3.55 ms | 7.81 ms | 5.55 ms | 16.9 ms | 21.8 ms | 22.0 ms |
| Reference, Nov 6 | 5.32 ms | 28.17 ms | 25.95 ms | 59.4 ms | 76.6 ms | 77.4 ms |

The optimized run used PyTorch 2.10.0+cu130, xcompute `0c48a46`, GPU image preprocessing and a 455-token flow prompt padded to 464. The reference run used PyTorch 2.9.1+cu130, reference operators, FlashAttention 2 for vision, CPU image preprocessing and exact prompt widths.

## Numerical checks against the BF16 sampler

A separate check used the checkpoint's BF16 sampler as the baseline and exact prompt widths. It exercised 72 observations in three modes (flow, RTC delay 3 and RTC delay 6). Each optimized stage received the corresponding baseline inputs, which isolates its numerical differences from those of preceding stages.

| Optimized output, Nov 7 (`651195d`) | Median relative L2 difference | Maximum |
|---|---:|---:|
| Vision embeddings | 0.90% | 1.14% |
| Prompt K/V cache | 1.65% | 2.27% |
| Action, flow | 0.15% | 0.24% |
| Action, RTC delay 3 | 0.25% | 1.66% |
| Action, RTC delay 6 | 0.25% | 0.64% |

The optimized composition preserved the pinned RTC prefix exactly. For the reference composition, prompt prefill matched the sampler's prompt-only pass bit for bit on four checked requests. Given the sampler's K/V cache, the action stage also matched bit for bit on four requests covering flow and RTC. End to end, the reference composition's mean absolute normalized-action differences were 0.0016 (flow), 0.0014 (RTC 3) and 0.0010 (RTC 6), each over 72 requests. A separate check comparing xcompute `7336b0d` with `0c48a46` found bitwise-identical optimized actions on all 216 requests.

The full-checkpoint runs used separately held weights and captured observations. The repository includes [tiny-checkpoint tests](../../tests/models/test_wallx2b_tiny.py) that exercise both compositions locally. Both compositions also pass the internal precision gate on recorded robot data.
