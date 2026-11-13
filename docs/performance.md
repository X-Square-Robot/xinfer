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
2.10.0.post4): the Wall-OSS row at revision `f07d8d8` of this branch with the walloss delivery
`xcompute.so` (SHA-256 `fad3a6c1…`), its combined python package and AOT artifacts and the combined
fast-path flags (MX8 down prepack/split-K, G2112, SMALLM, COPYKILL, RMSFUSE) enabled, and the
Wall-X 2B row at revision `49ad6d5` with the r5 delivery
(SHA-256 `903ff6de…`, which adds the 2B fused qkv tail kernel), each loaded through
`XINFER_XCOMPUTE_LIBRARY` with its matching Python package. Both models ran
with their default operator selection: every operator either model calls resolves to `ascend:xcompute`, and a
per-layout dispatch audit of a full request pass confirmed every call ran on an xcompute entry (Wall-X 2B:
21 of its 21 operators, all call layouts admitted; Wall-OSS: 11 of its 11). No operator ran on a torch
reference in either benchmark. The [Ascend guide](ascend950dt.md) lists each operator's entry.

| Model | Precision and composition | Single request P50 | Robots per NPU | Server P50 at capacity | Server P95 at capacity | Highest measured rate | Numerical agreement |
|---|---|---|---|---|---|---|---|
| Wall-OSS | BF16, every operator on xcompute entries, DVPP images | **41.20 ms** (lineage combined stack) | 5 | 127 ms | 139 ms | 31.9 req/s | quick100 B1, −6.48 % against the BF16 reference: PASS |
| Wall-X 2B | BF16, every operator on xcompute entries, DVPP images | **33.29 ms** (lineage combined stack plus the adafast action kernels) | 10 | 87 ms | 108 ms | 54.9 req/s | fixed-noise action gate, max_abs 0.0051 against the reference composition: PASS |

The bold latency numbers are the strongest measured configurations to date (the lineage combined stacks, gates included: quick100 -6.48 % / action max_abs 0.0051); the Wall-OSS capacity columns still cite the integrated-branch measurement. Porting the lineage stacks onto this branch is in progress, and the rows will be re-measured on the branch composition once that lands. The bold Wall-X 2B row's latest measurement includes the xcompute-side adafast action-kernel library knife (relinked library SHA-256 `3c26927e…`, delivered with the build and not committed to the repository). The same branch composition, with the prompt-width rows padded to the 64-row multiple so the generic GEMM takes the compact plan, measures 41.72 ms at 8 robots per NPU (server P50/P95 93/108 ms, 45.6 req/s) with a bitwise-identical action gate; output rows are independent, so the padded dispatch is exact. That measurement is taken on the tree of the padded-row commit, not on the later documentation commits.

Conditions for the Ascend rows:

| # | Model options | Single-request input set | Load | Server warmup | Validation |
|---|---|---|---|---|---|
| 3 | defaults, `image_preprocess_device` on the NPU (DVPP), the delivered package's combined fast-path flags on | a flow and an RTC request | synthetic desktop observations with RTC | the same two | quick100 B1 ([precision gate](precision.md)) |
| 4 | defaults, `image_preprocess_device` on the NPU (DVPP) | the 216 requests of the reference capture (fixed noise seed 20261112) | synthetic desktop observations with RTC | two of them | fixed-noise action gate |

Both models' rows report one wiring: the xcompute package's own entries, which mix CANN ACLNN Matmul,
FusedInferAttentionScore V5 and official operators with TileLang and AOT custom kernels inside xcompute. At the
backend-selection level both models name xcompute by default; TorchNPU's own operators and the CANN bridge
remain opt-in pins (`--op-backends linear=ascend:torch_npu`).

The precision gates for both rows are end-to-end and ran through the served model on the same path the
benchmark used, with image preprocessing on the NPU (DVPP). The [precision gate doc](precision.md)
defines the quick100 B1 gate; the Wall-X 2B gate compares fixed-noise action predictions against the
reference composition of the same build.

See the [Ascend guide](ascend950dt.md) for the operator and graph configuration.
