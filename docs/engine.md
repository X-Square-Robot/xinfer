# Engine

English | [简体中文](engine.zh-CN.md)

`xinfer.engine.Engine` runs one model on one GPU with all stages in one
process (docs/design.md §6–§9). This page covers its behavior and settings.

## Threads

| Thread | Work |
|---|---|
| engine thread | admission (up to `max_running_requests`), per-stage FIFO queues, batch formation, the stages' CUDA submissions (uploads, graph replays, events) |
| completion thread | waits for each final-stage event, copies nothing, resolves the request's future |
| device-image thread | decodes and resizes the `DeviceImage` inputs of the admitted requests that have them, on its own CUDA stream ([below](#device-images)) |

Callers use `submit(prepared, robot=...) -> Future` (thread-safe) or
`run(prepared)`. `StageGraph` validates that the model has exactly one terminal
stage; completion releases each request's slot once, after that stage.

The request path in `xinfer/engine/core.py` is:

```text
submit() -> waiting queue
engine thread (_engine_thread):
  _run_engine_loop() -> _step()
    _admit_requests()
    _select_batch(stage) -> _execute_stage_batch(stage, lane, batch)
completion thread:
  _completion_loop() -> wait for final event -> resolve Future
```

`_select_batch` removes the selected requests from the stage queue.
`_execute_stage_batch` submits asynchronous CUDA work and records its events;
returning from it does not mean GPU execution has completed.

If waiting on a final-stage event raises, that batch's requests fail and the
completion thread goes on. `Engine.health()` returns why the engine can no
longer serve (its engine or completion thread has stopped), else None;
`xinfer-serve` polls it ([serving](serving.md)). With `Engine.stats_sink` set,
the completion thread reports each completed request's admission wait and
stage waits and executions to it as `RequestStats`
([metrics](metrics.md#collection)).

## Scheduling rules

1. **Chained dispatch.** A request enters a stage's queue as soon as its
   upstream stages are *dispatched*; the lane waits on their events on the
   GPU. With free lanes a request runs every stage without a host round
   trip, which is the single-request path. Under load the queues fill and
   stages pipeline.
2. **Zero-wait batching.** A dispatch takes the compatible requests already
   queued, up to the stage limit, choosing the largest allowed batch size and
   never padding rows. Compatible means equal shapes (except the batch axis)
   and equal host values for every input the stage reads. A stage batches up to
   the largest size the model declares in `Capabilities.batch_sizes`; a model
   declares sizes above 1 only for a stage whose batched results pass its
   precision gate. The first dispatch of each stage runs alone because it
   learns which inputs the stage reads.
   - **Different lengths.** A handoff tensor whose `TensorSpec` declares a
     `seq_axis` (WALL-OSS-0.5: the prompt K/V) may differ in length between batched
     requests. The batch left-pads every row to the longest, and the stage
     receives `<name>_lengths` (`int32 [B]`, also at B1). See
     `xinfer/engine/batching.py`. On CUDA the lengths are uploaded
     asynchronously from pinned memory: the lane may already wait for
     upstream events, and a blocking upload would stall the issuer.
   - **Lazy merge.** Inputs are merged when the stage reads them. Inputs the
     stage does not read, or cannot share (different shapes or host values),
     are left out of the batch.
   - **Counters.** `stats["dispatch_<stage>_b<N>"]` counts dispatches per batch
     size, and `stats["padded_dispatch_<stage>"]` counts batches that padded
     different lengths.
3. **Downstream first.** Each pass of the engine thread admits waiting
   requests while slots are free, then dispatches the stages from the last
   to the first, so downstream work is issued before upstream stages add more.

Lane streams are created ordered after the work already queued on the
current stream, such as the weight loading and conversion, so the first
dispatch never reads weights that are still being written. A lane runs one dispatch at a time; it is free again when its event
completes. Outputs that a later stage reads are copied out of the lane's
buffers into request-owned tensors before the lane's next dispatch.

A final stage's results are copied to pinned host buffers on a second stream
of the lane (`Lane.copy_stream`) that is never captured: a freed pinned buffer
records an event on the streams it was used on, and an event recorded during a
capture makes the next pinned allocation fail.

If a dispatch fails after enqueuing work, part of the copy to the host
included, the lane's stream waits for its copy stream and records an event;
the lane and the requests' slots (holding their tensors) are released after
that event.

## Device images

A prepared input may be a `DeviceImage` (`xinfer.models.interface`) instead
of a tensor: a camera view as it arrived, JPEG bytes or an RGB frame, with
its size after each resize step and each step's `interpolate` mode. The input
processors send them with `image_preprocess_device="gpu"` (every bundled model; [configuration](configuration.md#image-preprocessing-on-the-gpu)).
Before the request's first stage, outside the CUDA graphs, the engine turns
each one into the `[h, w, 3]` uint8 view that the stages read under the same
name, as with CPU preprocessing. The views have the CPU path's shapes, so the
graphs and their keys are the same, and the variable-length JPEG bytes never
key a graph.

At admission the request's tensors are uploaded as usual and its
`DeviceImage` values stay in host memory. A request with device images then
goes to the **device-image thread** instead of the queues of its first
stages. In admission order, the thread makes its own CUDA stream wait for the
request's upload, decodes all the request's JPEGs in one batched call of the
platform's JPEG decoder, uploads its frames, resizes the views in FP32 (rounded
half up to uint8) and records an event, which becomes the request's
`inputs_ready` event. The engine thread then queues the request for its first
stages, and their lanes wait for that event on the GPU: the engine thread
never waits for the decoding, and a request's decoding overlaps the stages of
the requests before it. Requests without device images do not go through the
thread. The views belong to the request until it finishes, after its final
stage, so no stage reads freed memory. If decoding raises, only that request
fails (its tensors held until the thread's event); `shutdown` drains the
thread and stops it. Warmup requests take the same path.

The decoder (`xinfer.engine.device_images`) is created on the thread for the
first request with device images, by `xinfer.platforms.devices.jpeg_decoder_for`:
on CUDA, NVIDIA's nvImageCodec (`xinfer.platforms.cuda.CudaJpegDecoder`).
Without a CUDA device or without nvImageCodec, every request with device
images fails with that reason; under `xinfer serve --warmup` the warmup
requests are prepared by the same processors, so the server then fails at
startup.

nvImageCodec decodes the batch on the decoder's own stream with nvJPEG's
hardware decoder, which runs on the GPU's JPEG engine (NVJPG, which the RTX
5090 and Jetson Thor have): the entropy decoding and the IDCT run there, and a
CUDA kernel upsamples the chroma as libjpeg-turbo does and converts to RGB, so
the decoded views are within one level of the CPU path's. The decoded images
are nvImageCodec's memory. The decoder copies them into tensors on its stream
and makes the thread's stream wait for that copy; it keeps the images until an
event after the copy has completed and then decodes into them again, so a call
neither allocates new images nor waits for the GPU. On an RTX 5090
(nvImageCodec 0.10) a call takes 0.4–0.7 ms of the thread for a request's
camera views. The decoder has one CPU thread: nvImageCodec's default, one per
visible core, spins while it waits and slowed the frontends and the engine
under load in a container with a CPU quota. One device-image thread is enough,
and the decoder is called from it only. The JPEG bytes stay in host memory,
where nvImageCodec reads the bitstream.

Whether a CUDA device has device images is checked by creating the decoder,
without decoding. A decode at that point, at platform resolution, left every
later CUDA graph replay of the process slower: on an RTX 5090 the gaps between
their kernels grew from under 0.5 µs to 0.5–1 µs, about 0.6 ms per WALL-OSS-1.0-Nano
request and up to 1 ms per WALL-OSS-0.5 request, with or without JPEGs. Without it,
single requests take as long as with torchvision's decoder, which XInfer used
before.

## Warmup

`Engine.warmup(requests)` (`Policy.warmup`, `xinfer-serve --warmup`) first
applies any model-specific `warmup_requests`, then
expands `model.shape_buckets()` through the model's `pad` hook: one variant per
shape bucket that requests like it can take (every prompt width the state
text can give with the request's instruction, in 16-token buckets by default for WALL-OSS-0.5). It runs each
variant alone, which captures its graphs at batch size 1 on the first lane;
each lane keeps the inputs of those captures. `capture_batch_sizes`
(`xinfer/engine/cudagraph.py`) then runs them on every lane of their stage at
every allowed batch size, merged as serving merges a batch, so serving finds
the graphs it needs for these shapes. Largest inputs go first, since growing a
lane's static buffer drops the graphs that use the old one. A batch of mixed
ragged lengths is padded to its longest member, so warming each length covers
the mixes. One request of each kind (WALL-OSS-0.5: a flow and an RTC request; one
per instruction) is therefore enough; shapes the model does not derive
(another instruction, camera layout or RTC row count) still need their own
warmup request, and are otherwise captured on first use and reused. The warmup logs its selected bucket widths,
its time, the graphs kept per stage,
their capture time and the device memory reserved afterwards.

Each lane keeps at most `CUDAGraphConfig.max_graphs` (64) graphs. The action
lanes hold the most: per instruction, WALL-OSS-0.5 needs 3–4 prompt buckets ×
flow/RTC × 4 batch sizes = 24–32 graphs per lane.

`Engine.warmup` returns each warmup request's outputs (from its first
variant), which `xinfer-serve` passes to the frontends for their own warmup
([serving.md](serving.md)).

## Settings (`EngineConfig`)

| Field | Default | Meaning |
|---|---|---|
| `lanes` | 1 per stage | lanes (stream + graph pool) per stage, e.g. `{"prefill": 2, "action": 2}` |
| `stream_priorities` | 0 per stage | accelerator stream priority of every lane of a stage, within the platform's supported range; lower integers have higher priority |
| `max_batch` | 1 per stage | batch limit per stage; capped by `Capabilities.batch_sizes` |
| `max_running_requests` | 8 | admitted requests in flight |
| `max_queued_requests` | 256 | submitted requests waiting for admission (`AdmitError("queue_full")` beyond) |
| `per_robot_inflight` | 1 | unfinished requests per robot id (`AdmitError("per_robot_limit")`) |
| `cudagraphs` | `None` | `CUDAGraphConfig()` replays CUDA graphs (see `xinfer.engine.cudagraph`) |

A lane's stream priority is fixed when the lane is created; the upload and
copy streams keep priority 0. Priorities change only the order in which the accelerator
runs ready work: admission, batching and event waits are unchanged, and graphs
are captured and replayed on the same lane stream.

`xinfer-serve` and `Policy` resolve these from tuned tables (none ships yet), model and
platform rules and fixed defaults that do not depend on the load
([configuration](configuration.md)); explicit values still win.

## Guarantees and tests

- `test_wall_oss_0_5_parity.py::test_action_batches_match_single_requests` compares action batches B2–B8 against requests run alone across different prompt lengths and RTC delays. The outputs agree within rounding.
- Results of a request are independent of concurrency when batch size is 1:
  `tests/models/test_wall_oss_0_5_parity.py` runs 21 concurrent requests on 2 lanes
  per stage with graphs and compares them bitwise with the same requests run
  alone.
- `tests/unit/test_engine.py` covers concurrent submission, batching and row
  splitting, compatibility keys, admission limits and
  backpressure, cancellation and per-request failures.
- `tests/unit/test_cudagraph.py::test_result_freed_during_a_capture` frees a
  result on another thread while its lane is being captured.
- `tests/unit/test_host_copy.py` holds a failed dispatch's partial copy behind
  a GPU gate and checks that its lane and slot are not reused meanwhile.
- `tests/unit/test_batching_async.py` keeps an upstream producer pending and
  checks that merging a ragged batch does not block the issuer.
- `tests/unit/test_engine.py` covers the device-image thread: a request
  without device images completes while the decoding of an earlier one is
  held, only requests with device images reach the decoder, which is created
  once on that thread, decoding follows admission order, a raising decode
  fails only its request, and without a CUDA device requests with device
  images fail while the others are served.
  `test_device_images_are_decoded_off_the_lanes_and_replay_one_graph`
  (`tests/unit/test_cudagraph.py`) checks that the images stay in host memory
  until the thread decodes them on its stream, not a lane's, and that JPEGs
  of different lengths and a frame replay one graph with the eager engine's
  outputs. `tests/unit/test_shm.py` checks that `DeviceImage` data crosses
  from a frontend through shared memory, not the pipe.
- `test_warmup_captures_every_lane_and_batch_size`,
  `test_warmed_lengths_cover_batches_of_mixed_lengths` and
  `test_warmup_requests_cover_every_bucket` in `tests/unit/test_cudagraph.py`
  serve concurrent requests after warmup with `on_miss_after_warmup="error"`.
