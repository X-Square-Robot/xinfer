# Serving

English | [简体中文](serving.zh-CN.md)

`python -m xinfer.entrypoints.serve` (or `xinfer-serve`) serves one model on
one GPU over WebSocket + msgpack, compatible with the x2robot desktop client.

## Process layout

```text
robot ──WebSocket + msgpack──▶ frontend process ×N (CPU; SO_REUSEPORT on one port), a processing thread:
                                 decode msgpack → InputProcessor
                                   ("gpu": only the JPEG headers are read; each view is a DeviceImage)
                                 lease a shared-memory buffer; write the tensors and the DeviceImage data
                                 │ manifest over a pipe
                                 ▼
engine process (GPU)
  bridge thread     one per frontend: map the buffer (no copy) → Engine.submit → waiting queue
  engine thread     admission: FIFO, a free slot; upload the tensors, record inputs_ready
                      │ no DeviceImage           │ DeviceImage values
                      │                          ▼
                      │                    device-image thread: decode and resize on its own
                      │                    stream; its event becomes inputs_ready
                      ▼                          │
                    first-stage queues ◀─────────┘ (handed back through the engine thread)
  lanes             wait for inputs_ready on the GPU → vision → … → final stage
  completer thread  final event complete → future resolved → outputs over the pipe
                                 │
                                 ▼
robot ◀──WebSocket── frontend: release the buffer → OutputProcessor → response
```

Admission, batching and the stages are described in [engine.md](engine.md),
the device-image thread in [Device images](engine.md#device-images).

- **Frontends** never initialize CUDA. Each owns a ring of shared-memory
  buffers; a buffer is reused after its response arrives. Each processes
  requests on `--frontend-workers` threads (default 4), so requests of
  different connections are processed at the same time; a connection still
  sends one request at a time.
- **Engine process.** Loads the model, runs the engine (see
  [engine.md](engine.md)), maps the buffers and registers them as pinned host
  memory, so uploads read them directly. With
  `image_preprocess_device="gpu"` the frontends only read the JPEG
  headers; the JPEG bytes and raw frames of the `DeviceImage` values cross in
  the buffers, with their sizes in the manifest, and the engine decodes the
  JPEGs from there ([configuration](configuration.md#image-preprocessing-on-the-gpu)).
- **Startup.** The frontends start listening only after the engine has
  finished warmup, so an open port means ready. With `--warmup`, each frontend
  first runs the warmup requests through the model's InputProcessor and, with
  the engine's outputs for them, its OutputProcessor, so the first robots do not
  pay first-call costs such as numba compilation. These calls use processors
  built only for warmup; the serving processors carry no state from them.

## Protocol

- On connect the server sends the model's metadata frame. For Wall-OSS:
  `{"batch_enabled": true, "model": "wall-x", "action_horizon": 16, "action_dim": 26, "rtc_ee_chunk_dim": 14, "env": "x2robot"}`.
- Each request is one msgpack message, with arrays in the msgpack-numpy layout. A connection's requests are answered one at a time, by its own input processor: a model with per-connection state keeps it for the connection's lifetime. The response carries the model's fields plus `status` (`"ok"`) and `server_timing`:
  - `infer_ms`: request decoded → response encoded;
  - `prepare_ms`, `engine_ms`, `output_ms`;
  - `frontend`: which frontend process served it.
- A failed request is answered as `{"status": ..., "error": "<message>"}`, without `server_timing`; the connection stays open. A protocol error the robot acts on also carries `error_info`:

  | `status` | Meaning |
  |---|---|
  | `invalid_request` | the message could not be decoded, or the model's input processing rejected it |
  | `queue_full` | the engine's admission queue is full (`--max-queued-requests`); retry later |
  | `per_robot_limit` | the robot already has `--per-robot-inflight` requests in flight |
  | `unavailable` | the server is shutting down |
  | `internal_error` | the server failed while serving the request (engine or output processing) |

  Count every response whose `status` is not `"ok"` as failed.
- `GET /healthz` (or `/ready`) on the same port returns 200.
- `GET /metrics` on the same port returns the Prometheus metrics of the engine and every frontend ([metrics](metrics.md)).

## Run

```bash
python -m xinfer.entrypoints.serve --model wall_oss --checkpoint /path/to/checkpoint \
  --port 33985 --warmup flow.msgpack rtc.msgpack \
  --model-options '{"quantization": "bf16", "processing": {"instruction_override": null}}'
```

| Option | Meaning |
|---|---|
| `--model-options` | JSON passed to `from_checkpoint` and `build_processors` (Wall-OSS: `processing` = `ProcessingConfig` fields) |
| `--frontends`, `--frontend-workers`, `--shm-buffers`, `--shm-buffer-mib` | frontend processes (default 2), processing threads each (default 4) and shared-memory buffers each; buffer size |
| `--lanes`, `--max-batch`, `--max-running-requests`, `--max-queued-requests`, `--per-robot-inflight` | engine settings ([engine.md](engine.md)); checked, never rewritten |
| `--enforce-eager` | run every stage eagerly, without CUDA graphs (default: CUDA graphs where the platform supports capture); `--warmup` request files (JSON or msgpack, each one request or a list): their graphs, at every shape bucket the model derives from them (one flow and one RTC request cover Wall-OSS), are captured on every lane and allowed batch size, and every frontend runs them through the model's processors, before serving |
| `--tuned` | tuned-table selection: `auto`, `none` or a path |
| `--resolved-config`, `--dry-run` | write the resolved settings with their sources; print them and exit |
| `--profile-dir` | record the engine's spans and write each request's engine segments to this directory at shutdown; for profiling runs (`xinfer-profile load --serve`, [profiling](profiling.md#through-xinfer-serve---serve)), the records grow with the requests |
| `--disable-metrics` | record no Prometheus metrics; `GET /metrics` answers 404 ([metrics](metrics.md)) |
| `--reference-ops`, `--op-backends` | use only torch reference operators; pin an implementation per operator by backend, library or both (`op=cuda`, `op=kernels`, `op=cuda:kernels`, `op=reference`), pins win over `--reference-ops` |

SIGINT/SIGTERM stops the frontends, releases graphs and pinned buffers, and
exits 0. The server checks `Engine.health()` every 0.5 s; when the engine can
no longer serve (its engine or completion thread has stopped), it shuts down
the same way and exits 1, so that its supervisor can restart it.

## Capacity knobs (Wall-X 2B fleets)

`--max-batch` with `--allow-non-invariant-batching` trades per-request latency
for fleet throughput on the action stage. The measured recipe on one Ascend
950DT (CAPACITY-R3, 2026-11-16, the 150 ms rule ladder):

```bash
python -m xinfer.entrypoints.serve ... \
  --max-batch action=4 --allow-non-invariant-batching
```

- The action stage pairs up to four single-robot requests per dispatch; the
  zero-wait batcher only merges requests already waiting, never delays one.
- `--allow-non-invariant-batching` is required because the stage's batched
  gemm/attention contracts are tolerance-level (not bitwise) batch-invariant;
  the single-request path is unchanged.
- Throughput: 70.9 req/s unbatched → **77.5 req/s** with `action=4` at 14-15
  robots (the pairing rate crosses over between 12 and 14 robots; below that
  the knob queues without pairing). At 12 robots it still passes the 150 ms
  rule at a +48 ms P50 cost -- deploy by target.
- **Do not combine with action-stage `--lanes` ≥ 2**: the lanes4 arm measures
  75.0 req/s (down from 77.5) -- parallel lanes drain the queue and starve the
  batcher's pairing window. Lanes remain the better choice at 12-13 robots
  when the vision/prefill stages queue deeply; batching wins from ~14 robots
  on. Pick one per deployment, not both.

## Load testing

`benchmarks/robot_load.py` runs virtual robots against the service in closed
loop. See [benchmark.md](benchmark.md).
