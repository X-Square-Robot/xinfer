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
python benchmarks/request_latency.py --model wall_oss --checkpoint /path/to/5_25000 \
  --model-options '{"quantization": "mxfp8"}' --requests-file flow.msgpack rtc.msgpack \
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
libraries (OpenCV for the synthetic Wall-OSS observations), not XInfer itself:

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

- **Wall-OSS (default):** synthetic observations, with two 640×480 JPEG wrist views, a raw 1280×720 head view and
  random arm poses (xyz within 0.5 m, roll/pitch/yaw within π, gripper 0..1). Wall-OSS writes the state into its
  prompt as text, so the prompt length varies, as with real robots (739–756 tokens for 200 random poses). After the
  first response each request carries RTC fields:
  - the previous chunk, shifted by the executed steps;
  - `inference_delay`: the 75th percentile of the last five delays (initially 5), clamped to
    1..`--max-inference-delay` model steps, plus two;
  - `num_flow_steps`: the flow steps the service's metadata frame declares (`flow_steps`, or `rtc.num_flow_steps`),
    else 5.
- **Models without RTC (`--requests-file`):** robot *r* sends request (*r* + *k*) mod *n* of the
  given files as its *k*-th request, without RTC fields. A request file is JSON (`.json`) or msgpack and holds one
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
when it has an `error` or a `status` other than `"ok"`, or, for Wall-OSS, when
it carries no absolute poses. A robot whose connection fails stops, and its
remaining rows count as missing.

- **`client_ms`**: from just before the request is serialized until the
  response has been received. It excludes building the observation
  (`payload_ms`).
- **`server_infer_ms`**: the service's own `server_timing.infer_ms`.

Each round prints its P50/P95/P99 latencies, its throughput and the robots'
underflow. A robot sends at most one request per 180 ms, so a throughput below
5.6 req/s per robot means that the service no longer keeps up. The exit status
is 0 when no round has errors or missing rows.

**Single-GPU capacity** is the largest robot count whose rounds all have no
errors or missing rows and a steady `server_infer_ms` P50 below 150 ms, with
every smaller count meeting the same rule. Measure each count on a newly
started service, and report each round separately.

## Performance table

[docs/performance.md](performance.md) lists each serving configuration's current single-request latency and
single-GPU capacity. `benchmarks/performance.py` measures one row on the visible GPU, which must run nothing else:

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/performance.py --name "Wall-OSS" --model wall_oss \
  --checkpoint /path/to/5_25000 --latency-requests flow.msgpack rtc.msgpack \
  --warmup flow.msgpack rtc.msgpack --robots 3 4 5 --out artifacts/perf-wall-oss
```

- **Single request:** `request_latency.py` on the model's request set (50 runs after 5 warmup runs), with the views
  decoded and resized on the device (`image_preprocess_device="gpu"`, `Policy`'s default). An
  `image_preprocess_device` in `--model-options` applies to this part and to the load.
- **Load:** for each robot count, in increasing order, a newly started `xinfer-serve` with the default settings and
  the model's warmup files, then `robot_load.py` with the repository convention (80 requests per robot, the first 20
  dropped, 8 rounds with random start phases). A model with real-time chunking gets the synthetic desktop observations with RTC (the default
  workload); a model without gets its request set (`--load-requests`). The script stops after the first count that
  fails.
- **Capacity:** the largest count with no error or missing row in any round and a median round server P50 (the
  median over the 8 rounds of each round's P50) below 150 ms, with every smaller measured count passing too. The
  table gives that count's median round P50 and P95; `row.json` also records the worst round. The saturation
  throughput is the highest completed rate measured.
- **Output:** `row.json` with every number, the configuration, the environment and the machine's load before and
  after each count, and the table row on standard output.

A merge request that changes a configuration's performance updates its row, with the earlier values in its
description. After a PyTorch, xcompute or protocol change, every affected row is measured again.

