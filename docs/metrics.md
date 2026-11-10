# Metrics

English | [简体中文](metrics.zh-CN.md)

`xinfer-serve` exports Prometheus metrics at `GET /metrics` on its serving port, next to `/healthz`. Prometheus
scrapes `http://<host>:<port>/metrics`; [examples/monitoring](../examples/monitoring) runs Prometheus and a Grafana
dashboard against a local service.

## Collection

- **Processes.** The engine process and every frontend process record their own metrics (`prometheus_client` in
  its multiprocess mode). They write them to files in one directory, and the frontend that receives a scrape
  reads them all: every scrape returns the sum over the whole service, whichever frontend answers it.
- **Directory.** `PROMETHEUS_MULTIPROC_DIR` if it is set, else a new temporary directory that the server removes
  at exit; the log names it at startup (`recording Prometheus metrics in ...`). A directory you set must be empty
  when the server starts: files left by an earlier run would be added to the totals.
- **Engine.** The engine has no metrics of its own. With `Engine.stats_sink` set it reports each completed
  request's times and batch sizes as plain data (`RequestStats`, `xinfer/observability/stats.py`); `xinfer-serve`
  records them, and samples the engine's queue lengths, health and CUDA graph captures every 0.5 s.
- **Warmup.** The engine's metrics start after the warmup, so its requests, dispatches and CUDA graph captures are
  not counted.
- **Off.** `--disable-metrics` records nothing; `GET /metrics` then answers 404. Without `prometheus_client`
  (`pip install -e '.[serve]'`) the server logs a warning and runs the same way.
- **Cost.** On CUDA the engine records two timing events per stage dispatch and one when inputs are ready.
  The completion thread and frontends update the histograms.

## Metrics

| Name | Type | Labels | Meaning |
|---|---|---|---|
| `xinfer_requests_total` | counter | `status` | responses sent, by response status (`ok`, `invalid_request`, `queue_full`, `per_robot_limit`, `unavailable`, `internal_error`; [serving](serving.md#protocol)) |
| `xinfer_request_duration_seconds` | histogram | | successful request, message received to response built (`server_timing.infer_ms`) |
| `xinfer_request_prepare_seconds` | histogram | | decoding and the model's InputProcessor (`server_timing.prepare_ms`) |
| `xinfer_request_engine_seconds` | histogram | | shared-memory packing and the engine round trip (`server_timing.engine_ms`) |
| `xinfer_request_output_seconds` | histogram | | the model's OutputProcessor (`server_timing.output_ms`) |
| `xinfer_connections` | gauge | | open WebSocket connections (robots) |
| `xinfer_requests_waiting` | gauge | | requests waiting for admission (`--max-queued-requests` bounds them) |
| `xinfer_requests_running` | gauge | | admitted requests not finished yet (`--max-running-requests` bounds them) |
| `xinfer_engine_healthy` | gauge | | 1 while the engine's threads run, else 0 (the server then exits) |
| `xinfer_admission_wait_seconds` | histogram | | submission to admission in the engine |
| `xinfer_stage_wait_seconds` | histogram | `stage` | per request: from when the stage can start (the request's inputs on the device, or its upstream stages finished) to the start of its dispatch on the device |
| `xinfer_stage_execution_seconds` | histogram | `stage` | per request: the device time of the stage dispatch that served it (CUDA timing events after the dispatch's waits and after the stage; host times on the CPU) |
| `xinfer_stage_dispatches_total` | counter | `stage`, `batch_size` | stage dispatches of the completed requests, by batch size (each request adds 1 / batch size) |
| `xinfer_cudagraph_captures_after_warmup_total` | counter | `stage` | CUDA graphs captured while serving, for shapes the warmup did not cover; each capture delays its request (stages with CUDA graphs only) |
| `xinfer_model_info` | gauge | `model`, `quantization`, `numerics`, `platform`, `version` | 1, labeled with the served model and build |

A request's server time is its prepare, engine and output times. Its engine time is its admission wait, each
stage's wait and execution, and the rest: uploading the inputs, copying the outputs back and the transfer to the
frontend. A request's stage waits and executions are measured on one clock (CUDA timing events), so they add up
along the request, and so do their means; percentiles do not add up.

A stage's execution time is its time on its lane's stream. Under load several lanes run at once and share the
device, so it is longer than the stage alone.

Histogram buckets, in seconds: requests, admission and stage waits 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
0.15, 0.2, 0.3, 0.5, 0.75, 1, 2 and 5; stage execution 0.0005, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05,
0.075, 0.1, 0.2, 0.5 and 1.

GPU utilization, memory, clocks and power come from the NVIDIA DCGM exporter, not from XInfer.

## Queries

| Question | PromQL |
|---|---|
| Requests per second, by status | `sum by (status) (rate(xinfer_requests_total[1m]))` |
| Failed share | `sum(rate(xinfer_requests_total{status!="ok"}[5m])) / sum(rate(xinfer_requests_total[5m]))` |
| P95 server time | `histogram_quantile(0.95, sum by (le) (rate(xinfer_request_duration_seconds_bucket[1m])))` |
| P95 device time per stage | `histogram_quantile(0.95, sum by (le, stage) (rate(xinfer_stage_execution_seconds_bucket[1m])))` |
| P95 wait per stage | `histogram_quantile(0.95, sum by (le, stage) (rate(xinfer_stage_wait_seconds_bucket[1m])))` |
| Mean engine time by part | `sum by (stage) (rate(xinfer_stage_wait_seconds_sum[1m])) / sum by (stage) (rate(xinfer_stage_wait_seconds_count[1m]))`, the same for `xinfer_stage_execution_seconds` and `xinfer_admission_wait_seconds` |
| Mean batch size per stage | `sum by (stage) (rate(xinfer_stage_execution_seconds_count[5m])) / sum by (stage) (rate(xinfer_stage_dispatches_total[5m]))` |
| Graph captures while serving | `sum by (stage) (increase(xinfer_cudagraph_captures_after_warmup_total[1h]))` |

## Prometheus and Grafana

[examples/monitoring](../examples/monitoring) holds a Prometheus scrape configuration, Grafana provisioning and a
dashboard, and a Docker Compose file that runs both:

```bash
python -m xinfer.entrypoints.serve --model wall_oss --checkpoint /path/to/checkpoint --port 33985
cd examples/monitoring
docker compose up -d
```

- **Prometheus** (`http://localhost:9090`) scrapes `host.docker.internal:33985` every 5 s
  (`prometheus.yaml`; change the target to your servers, one per GPU).
- **Grafana** (`http://localhost:3000`, user `admin`, password `admin`) opens with the Prometheus data source and
  the XInfer dashboard: requests per second by status; server time P50, P95 and P99; the mean server time and the
  mean engine time stacked by part (they add up); each engine part's P95; mean batch size; waiting and running
  requests and connections; engine health, graph captures while serving and the served model.

To use your own Prometheus, add the scrape job of `prometheus.yaml` to it and import `grafana/xinfer.json` in
Grafana (Dashboards → New → Import).
