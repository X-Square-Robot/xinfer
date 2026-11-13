# 指标

[English](metrics.md) | 简体中文

`xinfer-serve` 在服务端口上的 `GET /metrics` 导出 Prometheus 指标，与 `/healthz` 同端口。Prometheus 抓取 `http://<host>:<port>/metrics`；[examples/monitoring](../examples/monitoring) 针对本机服务运行 Prometheus 和一个 Grafana 看板。

## 采集方式

- **进程。** 引擎进程和每个前端进程各自记录指标（`prometheus_client` 的多进程模式），写到同一个目录下的文件里；收到抓取请求的前端读取全部文件：无论哪个前端应答，每次抓取返回的都是整个服务的总和。
- **目录。** 设置了 `PROMETHEUS_MULTIPROC_DIR` 时用它，否则新建一个临时目录，服务退出时删除；启动日志会写出目录（`recording Prometheus metrics in ...`）。自行设置的目录在服务启动时必须为空：上一次运行留下的文件会被加进总数。
- **引擎。** 引擎本身没有指标。设置了 `Engine.stats_sink` 时，它把每个完成请求的时间和批大小作为普通数据（`RequestStats`，`xinfer/observability/stats.py`）报告出去；`xinfer-serve` 记录这些数据，并每 0.5 s 采样一次引擎的队列长度、健康状态和 CUDA graph 捕获。
- **预热。** 引擎的指标从预热结束后开始记录，预热中的请求、发射和 CUDA graph 捕获不计入。
- **关闭。** `--disable-metrics` 不记录任何指标，`GET /metrics` 返回 404。没有安装 `prometheus_client`（`pip install -e '.[serve]'`）时，服务记录一条警告，其余照常运行。
- **开销。** 在 CUDA 上，引擎线程每次阶段发射多记两个计时事件，输入就绪事件改为计时事件；直方图在完成线程和前端中写入。

## 指标

| 名称 | 类型 | 标签 | 含义 |
|---|---|---|---|
| `xinfer_requests_total` | counter | `status` | 已发出的响应数，按响应状态分（`ok`、`invalid_request`、`queue_full`、`per_robot_limit`、`unavailable`、`internal_error`；见[服务](serving.zh-CN.md#协议)） |
| `xinfer_request_duration_seconds` | histogram | | 成功请求从收到消息到构造好响应的时间（`server_timing.infer_ms`） |
| `xinfer_request_prepare_seconds` | histogram | | 解码和模型的 InputProcessor（`server_timing.prepare_ms`） |
| `xinfer_request_engine_seconds` | histogram | | 共享内存打包和引擎往返（`server_timing.engine_ms`） |
| `xinfer_request_output_seconds` | histogram | | 模型的 OutputProcessor（`server_timing.output_ms`） |
| `xinfer_connections` | gauge | | 打开的 WebSocket 连接数（机器人数） |
| `xinfer_requests_waiting` | gauge | | 等待准入的请求数（上限为 `--max-queued-requests`） |
| `xinfer_requests_running` | gauge | | 已准入、尚未完成的请求数（上限为 `--max-running-requests`） |
| `xinfer_engine_healthy` | gauge | | 引擎各线程都在运行且设备没有错误时为 1，否则为 0（随后服务退出） |
| `xinfer_admission_wait_seconds` | histogram | | 引擎中从提交到准入的时间 |
| `xinfer_stage_wait_seconds` | histogram | `stage` | 每个请求：从该阶段可以开始（请求的输入已在设备上，或其上游阶段都已执行完）到该阶段的发射在设备上开始 |
| `xinfer_stage_execution_seconds` | histogram | `stage` | 每个请求：为它服务的那次阶段发射在设备上的时间（发射的等待之后和阶段之后的 CUDA 计时事件之差；CPU 上为主机时间） |
| `xinfer_stage_dispatches_total` | counter | `stage`、`batch_size` | 已完成请求的阶段发射次数，按批大小分（每个请求计 1 / 批大小） |
| `xinfer_cudagraph_captures_after_warmup_total` | counter | `stage` | 服务期间为预热未覆盖的形状捕获的 CUDA graph 数；每次捕获都会拖慢它所在的请求（只有使用 CUDA graph 的阶段） |
| `xinfer_model_info` | gauge | `model`、`quantization`、`numerics`、`platform`、`version` | 值为 1，标签是所服务的模型和版本 |

请求的服务端时间由前处理、引擎和后处理三段组成。引擎时间由准入等待、各阶段的等待和执行，以及其余部分组成：其余部分是上传输入、拷回输出和传给前端。一个请求的各阶段等待和执行时间在同一个时钟上测量（CUDA 计时事件），沿着请求可以相加，它们的均值也可以相加；分位数不能相加。

阶段的执行时间是它在所在通道（lane）的 stream 上的时间。有负载时多个通道同时运行、共享设备，因此比该阶段单独运行时长。

直方图分桶（秒）：请求、准入和阶段等待为 0.005、0.01、0.02、0.03、0.05、0.075、0.1、0.15、0.2、0.3、0.5、0.75、1、2 和 5；阶段执行为 0.0005、0.001、0.002、0.005、0.01、0.015、0.02、0.03、0.05、0.075、0.1、0.2、0.5 和 1。

GPU 利用率、显存、时钟和功耗由 NVIDIA DCGM exporter 提供，不由 XInfer 导出。

## 查询

| 问题 | PromQL |
|---|---|
| 按状态分的每秒请求数 | `sum by (status) (rate(xinfer_requests_total[1m]))` |
| 失败比例 | `sum(rate(xinfer_requests_total{status!="ok"}[5m])) / sum(rate(xinfer_requests_total[5m]))` |
| 服务端时间 P95 | `histogram_quantile(0.95, sum by (le) (rate(xinfer_request_duration_seconds_bucket[1m])))` |
| 各阶段设备时间 P95 | `histogram_quantile(0.95, sum by (le, stage) (rate(xinfer_stage_execution_seconds_bucket[1m])))` |
| 各阶段等待时间 P95 | `histogram_quantile(0.95, sum by (le, stage) (rate(xinfer_stage_wait_seconds_bucket[1m])))` |
| 引擎时间各部分的均值 | `sum by (stage) (rate(xinfer_stage_wait_seconds_sum[1m])) / sum by (stage) (rate(xinfer_stage_wait_seconds_count[1m]))`，`xinfer_stage_execution_seconds` 和 `xinfer_admission_wait_seconds` 同理 |
| 各阶段平均批大小 | `sum by (stage) (rate(xinfer_stage_execution_seconds_count[5m])) / sum by (stage) (rate(xinfer_stage_dispatches_total[5m]))` |
| 服务期间的 graph 捕获 | `sum by (stage) (increase(xinfer_cudagraph_captures_after_warmup_total[1h]))` |

## Prometheus 和 Grafana

[examples/monitoring](../examples/monitoring) 中有 Prometheus 抓取配置、Grafana 的预配置和看板，以及同时运行两者的 Docker Compose 文件：

```bash
python -m xinfer.entrypoints.serve --model wall_oss --checkpoint /path/to/checkpoint --port 33985
cd examples/monitoring
docker compose up -d
```

- **Prometheus**（`http://localhost:9090`）每 5 s 抓取一次 `host.docker.internal:33985`（`prometheus.yaml`；把目标改成你的服务，每块 GPU 一个）。
- **Grafana**（`http://localhost:3000`，用户 `admin`，密码 `admin`）启动后已有 Prometheus 数据源和 XInfer 看板：按状态分的每秒请求数；服务端时间的 P50、P95、P99；按部分堆叠的平均服务端时间和平均引擎时间（可以相加）；引擎各部分的 P95；平均批大小；等待和运行中的请求数及连接数；引擎健康、服务期间的 graph 捕获和所服务的模型。

使用已有的 Prometheus 时，把 `prometheus.yaml` 中的抓取任务加进去，并在 Grafana 中导入 `grafana/xinfer.json`（Dashboards → New → Import）。
