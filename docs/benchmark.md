# Benchmarks

English | [简体中文](benchmark.zh-CN.md)

| Script | Measures |
|---|---|
| `benchmarks/request_latency.py` | single-request latency of a model served in process, without network |
| `benchmarks/robot_load.py` | latency with virtual robots over the network, against a running service |
| `benchmarks/performance.py` | one row of the [performance table](performance.md): both of the above in the table's protocol |

Results stay outside Git.

## Single request

`request_latency.py` loads the model with `xinfer.Policy`, so it needs XInfer
installed (`pip install -e '.[models]'`) and the model's assets, as for
`xinfer-serve`:

```bash
python benchmarks/request_latency.py --model wall_oss --checkpoint /path/to/checkpoint \
  --model-options '{"quantization": "bf16"}' --requests-file flow.msgpack rtc.msgpack \
  --runs 50 --warmup-runs 5 --out artifacts/wall-oss-latency.json
```

- **Request files** are wire requests, as for `xinfer-serve --warmup`: JSON (`.json`) or msgpack, each holding one
  request map or a list of them.
- **Warmup.** The script runs each request once, capturing its CUDA graphs on every lane and batch size, then
  discards `--warmup-runs` timed runs.
- **Timed runs.** Requests are taken in turn, one at a time, and each is timed in three steps:
  - `prepare_ms`: the model's input processing on the CPU (decode, resize, tokenize, normalize);
  - `engine_ms`: from the prepared CPU tensors to the final outputs on the CPU (upload, every stage, copy back);
  - `output_ms`: the model's output processing into the response;
  - `total_ms`: the whole request, from the wire request to the response, which is the sum of the three.
- **Results.** The JSON holds P50/P95/P99, mean, min and max of each, every sample, the device, the resolved
  configuration and operator selection, and the CUDA graphs captured per stage after the warmup (a nonzero count
  means a timed run met a shape the warmup did not cover).

Single-robot latency over the network, including the frontend, is
`robot_load.py --robots 1`.

## Robot load

`robot_load.py` loads a running WebSocket + msgpack service with
virtual robots and reports latency per round. It needs only the client
libraries, including OpenCV for the synthetic observations:

```bash
python -m pip install numpy 'websockets>=12' msgpack msgpack-numpy opencv-python-headless
```

Each robot runs in its own process with one connection and at most one
request in flight. The script does not start or stop the service.

### Workload

The robots run in closed loop. Each robot sends its first request at a random
delay within one request period (180 ms, uniform; seeded by `--start-seed` and
the round, recorded as `start_delays_ms` in each round's summary) after a start
time shared by the round's robots, 2 s after they are launched. Each later request is sent when the robot's action queue calls for it,
or when the previous response arrives if that is later
(`benchmarks/virtual_robot.py`, with the desktop client's parameters):

- the queue is consumed at 200 Hz, six control points per model step;
- the next request is due six model steps (180 ms) after the previous chunk's
  capture, or three after the first response.

While responses are faster than the period, the robots keep their start phases
for the whole round, so one round measures one arrangement of the robots' send
times: robots that start together send together at every period. The random
start delays and several rounds sample the arrangements instead of always
measuring the robots in lockstep, and the same seed gives the same
arrangements, so two builds are compared round by round.

The requests depend on the model:

- **Wall-OSS and Wall-X 2B** use the default synthetic desktop observations, with two 640×480 JPEG wrist views, a raw 1280×720 head view and
  fresh random arm poses for each request. Camera views are generated once per robot. Pose ranges are xyz within 0.5 m, roll/pitch/yaw within π, gripper 0..1. Wall-OSS writes the state into its
  prompt as text, so the prompt length varies, as with real robots (739–756 tokens for 200 random poses). After the
  first response each request carries RTC fields:
  - the previous chunk, shifted by the executed steps;
  - `inference_delay`: the 75th percentile of the last five delays (initially 5), clamped to
    1..`--max-inference-delay` model steps, plus two;
  - `num_flow_steps`: the flow steps the service's metadata frame declares (`flow_steps`, or `rtc.num_flow_steps`),
    else 5.
- **Recorded requests (`--requests-file`).** Robot *r* sends request (*r* + *k*) mod *n* of the
  given files as its *k*-th request. It replays each request as supplied. A request file is JSON (`.json`) or msgpack and holds one
  request map or a list of them, as for `xinfer-serve --warmup`.

The repository's load convention: 80 requests per robot, the first 20 as
warmup, and 8 rounds:

```bash
python benchmarks/robot_load.py --url ws://127.0.0.1:33985 --robots 8 \
  --requests 80 --warmup-requests 20 --rounds 8 --out artifacts/wall-oss-n8
python benchmarks/robot_load.py --url ws://127.0.0.1:33985 --robots 7 --requests-file requests.json \
  --requests 80 --warmup-requests 20 --rounds 8 --out artifacts/no-rtc-n7
```

With `--requests-file`, pass the same file to `xinfer-serve --warmup`, so that
warmup covers the requests' shapes.

### Results

- **`summary.json`**, one entry per round:
  - rows, errors, missing rows and steady rows;
  - P50/P95/P99 of `server_infer_ms` and of `client_ms`;
  - successful steady responses per second over the steady window;
  - the histogram of sent `inference_delay` values;
  - the robots' total underflow: time with an empty action queue while waiting for steady responses.
- **`round_<k>.csv`**: one row per request.
- **`run.json`**: the command, the source revision, the inputs' sha256 and the service's metadata frame.

Statistics use the steady rows: each robot's requests after its first
`--warmup-requests`. Errors are counted over all rows. A response is an error
when it has an `error` or a `status` other than `"ok"`, or when the default synthetic workload receives no valid absolute arm poses.
A robot whose connection fails stops, and its remaining rows count as missing.

- **`client_ms`**: from just before the request is serialized until the
  response has been received. It excludes building the observation
  (`payload_ms`).
- **`server_infer_ms`**: the service's own `server_timing.infer_ms`.

Each round prints its P50/P95/P99 latencies, its throughput and the robots'
underflow. A robot sends at most one request per 180 ms, so a throughput below
5.6 req/s per robot means that the service no longer keeps up. The exit status
is 0 when no round has errors or missing rows.

`robot_load.py` reports each round separately. Use `performance.py` to start a fresh service for each robot count and apply the [capacity rule](#performance-table).

## Performance table

The [performance table](performance.md) reports Wall-OSS and Wall-X 2B on an RTX 5090. Run `benchmarks/performance.py` on an idle GPU with a compatible checkpoint and representative wire requests.

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/performance.py --name "Wall-OSS" --model wall_oss \
  --checkpoint /path/to/checkpoint --latency-requests flow.msgpack rtc.msgpack \
  --warmup flow.msgpack rtc.msgpack --robots 3 4 5 --out artifacts/perf-wall-oss
```

For Wall-X 2B, supply your own `requests.msgpack` in the [desktop request format](models/wallx2b.md#request-and-response) and one flow and one RTC request for service warmup. The published row used a set of 216 requests for the single-request benchmark. The load client generates its own desktop traffic and sends a shifted action chunk after the first response.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/performance.py --name "Wall-X 2B" --model wallx2b \
  --checkpoint /path/to/wallx2b-checkpoint --latency-requests requests.msgpack \
  --warmup flow.msgpack rtc.msgpack --robots 8 12 14 15 16 --out artifacts/perf-wallx2b
```

- **Single request:** `request_latency.py` on the model's request set (50 runs after 5 warmup runs), with the views
  decoded and resized on the device (`image_preprocess_device="gpu"`, `Policy`'s default). An
  `image_preprocess_device` in `--model-options` applies to this part and to the load.
- **Load:** for each robot count, in increasing order, a newly started `xinfer-serve` with the default settings and
  the model's warmup files, then `robot_load.py` with the repository convention (80 requests per robot, the first 20
  dropped, 8 rounds with random start phases). The default workload sends synthetic desktop observations with RTC for both
  Wall-OSS and Wall-X 2B. Passing `--load-requests` replays the supplied requests. The script stops at the first
  robot count that fails.
- **Capacity:** start below saturation and increase the number of robots. Stop at the first count with any errors or missing responses, or when the median of the eight round server P50s exceeds 1.5 times the previous count's median. The first count provides a baseline. Report the last passing count, or a lower bound if every tested count passes. Report the median round P50 and P95 at capacity and the highest measured throughput across all counts.
- **Pinned-pose rail tolerance (Wall-X 2B):** an "error" in the ladder includes a rejected request whose pinned `prev_action_chunk` rows leave the pose rails (`RTC_POSE_LIMITS` in the model's request processing; the gripper columns (6, 13) rail at 5.0 by default). The Ascend 950DT Wall-X 2B rows apply a gripper-column tolerance of 5.3 instead (user adjudication of 2026-11-16): the model's gripper outputs sit naturally near the rail (per-response rail P50 4.51, 90.2% of the 5.0 rail), and every archived violation is the first free row after the teacher-forced prefix boundary overshooting rather than an optimization regression -- the archived overshoot spectrum tops at 5.2724 on the batch-knob arms of the current stack (5.2523 at the 15-robot arms), inside the adjudicated 5.3. The RTX 5090 and Jetson rows keep their own original-protocol measured values; when comparing rows across hardware, note this tolerance difference.
- **Single-request breakdown:** report the median end-to-end time, the sum of the median preparation and output times on the host, and the median engine time.
- **Output:** `row.json` with every number, the configuration, the environment and the machine's load before and
  after each count, and the table row on standard output.

A merge request that changes a configuration's performance updates its row, with the earlier values in its
description. After a PyTorch, xcompute or protocol change, every affected row is measured again.

