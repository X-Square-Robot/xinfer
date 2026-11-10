# Configuration resolution

English | [简体中文](configuration.zh-CN.md)

`xinfer-serve` and `Policy` use no named profiles (docs/design.md §10) and
take no load parameter. How many robots an instance serves changes over time,
and the engine adapts to the requests that arrive:
- batches form only from requests already queued (zero-wait batching);
- a lane without work stays idle;
- CUDA graphs are captured for the keys and batch sizes that occur.

So each setting is a fixed choice or an upper bound that suits any load. It is
resolved once at startup and reported with its source. Precedence, highest
first:

1. **explicit**: a flag or argument the operator set. It is checked against the platform and the model and never rewritten. A conflict is an error: a batch size the model does not declare, CUDA graphs on a platform without capture (API only), or an unknown stage.
2. **tuned table**: measured values for (model, architecture, quantization), from `xinfer/tuned/<model>/<arch>-<quantization>.json` or `--tuned <path>`. Each table records the commit, date, method and evidence; the source reads `tuned:<table>@<commit>`. Tables can set `lanes`. No table ships at present: a table is added only when a measured value differs from the default.
3. **model / platform rules**: what the model declares (batch sizes) and what the platform supports (graph capture, CPU quota); sources `model:<rule>` and `platform:<rule>`.
4. **default**: a fixed value.

Code: `xinfer/config/` (`resolve.py`, `tuned.py`).

## Settings and rules

| Setting | Without an explicit value | Notes |
|---|---|---|
| `per_robot_inflight` | 1 | unfinished requests per robot |
| `max_running_requests` | 32 | admitted requests in flight. An upper bound: slots hold no preallocated memory, and with one request per robot it is reached only by 32 robots |
| `max_queued_requests` | 256 | submitted requests waiting for admission |
| `lanes.<stage>` | tuned value, else 2 | a lane is a stream with its own graph pool; a second lane lets two requests run the stage at once |
| `stream_priorities.<stage>` | −1 for the final stage on CUDA (Wall-OSS: `action`), else 0 | CUDA stream priority of the stage's lanes, lower is higher (`--stream-priorities action=0`, `Policy(stream_priorities={...})`) |
| `batch_sizes.<stage>` | every size the model declares | a larger allowed size never delays a request, because batches form only from queued requests; a model declares sizes above 1 only for a stage whose batched results pass its precision gate |
| `cudagraphs` | on where the platform supports capture | |
| `frontends` | 2, at most half the CPUs of the container quota | each frontend adds a bridge thread in the engine process |
| `frontend_workers` | 4, with `frontends × frontend_workers` at most half the CPUs of the quota | processing threads per frontend; requests of different connections are processed at the same time |
| `shm_buffers` | `ceil(max_running_requests / frontends)` per frontend | one buffer per running request; a frontend whose buffers are all leased waits. Default: 2 × 16 buffers of `--shm-buffer-mib` (32 MiB), pinned |

A stream priority is a hint for ready GPU work: it does not preempt a running
kernel or bypass event waits, so it moves delay between stages rather than
removing it. Unknown stages, non-integers, values the device cannot
represent, and nonzero values on a platform without stream priorities are
errors (`--dry-run` resolves the values without creating lanes, so it does not
check the device range).

The MXFP8 measurements below are historical results from an earlier Wall-OSS configuration. The current [Wall-OSS composition](models/wall_oss.md) supports BF16.

Why these values:
- **Lanes 2.** On Wall-OSS MXFP8 (RTX 5090), two action lanes gave 7 robots of capacity against 5 with one (batched action stage), and 2/2/2 beat 1/1/1 without batching (31.2 vs 27.2 req/s saturated); three or more lanes were not measured. Evidence: `xinfer-action-batching-20261101.tar.gz` and `xinfer-flash-attn-20261101.tar.gz` (hashes in docs/models/wall_oss_validation.md).
- **Two frontends.** Wall-X input processing takes 7.1 ms per request from the x2robot dummy client (one raw 720×1280 view and two 640×480 JPEG views; 1.7 ms with three raw views), so 8 robots at 5.6 Hz keep one frontend busy about a third of the time (`xinfer-wallx-input-processing-v3-20261105.tar.gz`, sha256 `04df60bd…`). The comparisons below were measured when it took 22.2 ms per request, with one process busy about 100 % of the time at 8 robots. More did not help at low load: at 4 robots (2026-11-01, before action-stage batching; one run each) the server P50 / P99 was 94.8 / 123.8 ms with 1 frontend, 95.9 / 133.0 ms with 2 and 119.9 / 144.0 ms with 4. With action batching, 4 robots had the same end-to-end latency with 1 and 2 frontends (pooled P50 85–87 and 86–88 ms); with one frontend requests queued for input processing (P50 28 ms against 21 ms), with two they waited in the engine instead (`xinfer-config-defaults-20261103.tar.gz`, sha256 `6f85d5eb…`).
- **Four processing workers per frontend.** Image decode and resize release the GIL, so one process prepares several requests at once.
- **Final stage first.** With equal priorities, ready work of the final stage waits behind vision and prefill kernels of later requests. On an RTX 5090 (repository load convention, paired instances on one tree), action priority −1 against 0:
  - Wall-OSS MXFP8, 8 robots: pooled server P50 120.9 / 123.4 against 137.1 / 142.4 ms, P99 167.5 / 162.3 against 182.2 / 203.4 ms;
  - no measurable difference at 1 and 4 robots (Wall-OSS MXFP8).

  At or beyond saturation it can cost:
  - Wall-OSS BF16, 5 robots (overloaded): a 42 ms lower median and 15 % more throughput, but a 24–37 ms higher P99;
  - Wall-OSS MXFP8, 10 robots (overloaded): a 13 ms lower median, a lower P95 and 10 % more throughput, but fewer requests under 150 ms (2–11 % against 28–32 %).

  `--stream-priorities action=0` restores equal priorities. Evidence: `xinfer-stream-priorities-20261104.tar.gz` (sha256 `b642823fb11b337e6dd5f49ec35108c1e7fafb529c6f96a283704a7de5ff2ec7`).

## Prompt shape buckets

Wall-OSS uses `BucketSpec` in `xinfer/models/interface.py`. The startup report shows its effective policy in
`shape_buckets.prompt`. The default is `processing.prompt_bucket=16`.

| Model option | Effective policy |
|---|---|
| Positive `prompt_bucket` | Round up to that granularity, capped at the model's prompt limit |
| `prompt_bucket=0` | Exact valid length, without bucket padding |

Wall-OSS masks padded positions during action attention. Changing the prompt width can change floating-point rounding.

The engine expands `meta["prompt_len_range"]` in `xinfer/engine/cudagraph.py`
and calls the model's `pad` hook. It warms the reachable widths from largest to smallest. Logs report the widths and per-stage graph counts.
An exact-length policy warms only the supplied prompt width; new lengths use
the configured post-warmup graph-miss behavior.

## Image preprocessing on the GPU

`image_preprocess_device`, an option of every bundled model, sits at the top level of the model options in every
entry point, so one set of options serves both (`benchmarks/performance.py` passes the same to each):

- `xinfer serve` (and `xinfer-profile`): `--model-options '{"image_preprocess_device": "gpu"}'`; the default is
  `"cpu"`.
- `Policy(..., image_preprocess_device=...)`: `None`, the default, takes the platform's: `"gpu"` where the views can be
  decoded and resized on the device (a CUDA device with torchvision, the platform capability `device_images`), else
  `"cpu"`.

It is not a `processing` field: there it raises `ConfigError`, as does `"gpu"` on a platform without device images.
The entry point resolves it (`xinfer.config.apply_image_preprocess_device`) and hands it to the model's processors.

| Value | Where the views are decoded and resized | Numerics |
|---|---|---|
| `"cpu"` (`xinfer serve` default) | in the frontend: OpenCV decode and the model's OpenCV / Pillow resize, the views of a request in parallel | bitwise equal to the reference services |
| `"gpu"` (`Policy` default on CUDA) | on the device, by the engine. The frontend sends each JPEG view's bytes (other encodings decoded on the CPU) or raw frame as a `DeviceImage`, with its target sizes, read from the JPEG header, and the model's resize recipe; before the first stage the engine decodes the request's JPEGs in one torchvision `decode_jpeg` call (nvJPEG) and resizes the views in FP32 ([engine](engine.md#device-images)) | not bitwise equal: decoded JPEGs differ from libjpeg-turbo's around sharp chroma edges (on robot camera views a mean of 0.1–0.2 levels, a few pixels by up to tens of levels), and the FP32 resizes differ from Pillow's (torch's bicubic uses a = −0.75, Pillow's −0.5), so the actions differ |

`"gpu"` needs a CUDA device and torchvision ≥ 0.19 (`pip install -e '.[gpu-image]'`); otherwise every request
fails, and `xinfer serve --warmup` fails at startup. The decoder is created for the first request, a warmup
request under `--warmup`. The views keep their shapes, so the CUDA graphs are the same in both modes.

Single-request P50 for Wall-OSS MXFP8 on RTX 5090 with CUDA graphs, measured with `benchmarks/request_latency.py` over 50 runs (ms):

| Model, requests | `"cpu"`: total (prepare / engine) | `"gpu"`: total (prepare / engine) |
|---|---|---|
| Wall-OSS, x2robot dummy client (raw 1280×720 front view, two 640×480 noise JPEG wrist views) | 40.4 (7.4 / 32.8) | 37.4 (3.1 / 34.2) |
| Wall-OSS, three robot camera JPEG views (1280×720 and 640×480) | 39.0 (6.1 / 32.8) | 35.7 (1.9 / 33.6) |

On the GPU path, the device-image thread spends 1.1–1.4 ms per request on decoding and resizing on its own stream ([engine](engine.md#device-images)). With Wall-OSS MXFP8 at 8 robots, both modes served 44.0–44.3 req/s. Round P50 was 83–93 ms with GPU preprocessing and 83–88 ms with CPU preprocessing. The quick100 B1 GT-MSE was 6.13% below the BF16 reference with GPU preprocessing and 5.01% below it with CPU preprocessing. See [Wall-OSS validation](models/wall_oss_validation.md) for the evidence.

Not resolved yet:
- the memory planner (design §8.2);
- `xinfer tune` (tuned tables would be written by hand from benchmark runs);
- in-process frontends (design §9.2).

## Example: Wall-OSS on RTX 5090

`xinfer-serve --model wall_oss ... --dry-run` (the optimized composition):

```text
shape_buckets.prompt          {"granularity": 16, "max_length": null, "sizes": []} model:bucket policy
per_robot_inflight            1          default
max_running_requests          32         default
max_queued_requests           256        default
lanes.vision                  2          default
lanes.prefill                 2          default
lanes.action                  2          default
stream_priorities.vision      0          default
stream_priorities.prefill     0          default
stream_priorities.action      -1         default:final stage
batch_sizes.vision            [1]        model:declares B1 only
batch_sizes.prefill           [1]        model:declares B1 only
batch_sizes.action            [1]        model:declares B1 only
cudagraphs                    True       platform:supports graph capture
frontends                     2          default, ≤ cpus/2 = 12
frontend_workers              4          default, frontends × workers ≤ cpus/2 = 12
shm_buffers                   16         default:one per running request
```

The reference composition (`--model-options '{"numerics": "reference"}'`) declares action batches
`[1, 2, 4, 8]` (`model:declared sizes`).

## Measured with the defaults

Single-GPU capacity with these settings, for every model: [performance](performance.md).
