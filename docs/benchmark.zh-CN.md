# 基准测试

[English](benchmark.md) | 简体中文

| 脚本 | 测量内容 |
|---|---|
| `benchmarks/request_latency.py` | 进程内服务的模型的单请求时延，不经过网络 |
| `benchmarks/robot_load.py` | 虚拟机器人经网络对运行中的服务测得的时延 |
| `benchmarks/performance.py` | [性能表](performance.zh-CN.md)中的一行：按该表的口径运行上面两项 |

结果不纳入 Git。

## 单请求

`request_latency.py` 用 `xinfer.Policy` 加载模型，因此与 `xinfer-serve` 一样，需要已安装 XInfer（`pip install -e '.[models]'`）以及模型的资产：

```bash
python benchmarks/request_latency.py --model wall_oss --checkpoint /path/to/5_25000 \
  --model-options '{"quantization": "mxfp8"}' --requests-file flow.msgpack rtc.msgpack \
  --runs 50 --warmup-runs 5 --out artifacts/wall-oss-latency.json
```

- **请求文件**是传输格式（wire）的请求，与 `xinfer-serve --warmup` 相同：JSON（`.json`）或 msgpack，每个文件包含一个请求映射或其列表。
- **预热。** 脚本把每个请求运行一次，在每个通道（lane）和批大小上捕获其 CUDA graph，然后丢弃 `--warmup-runs` 次计时运行。
- **计时运行。** 请求轮流执行，一次一个，每个请求分三步计时：
  - `prepare_ms`：模型在 CPU 上的输入处理（解码、resize、分词、归一化）；
  - `engine_ms`：从准备好的 CPU 张量到 CPU 上的最终输出（上传、各阶段、拷回）；
  - `output_ms`：模型把输出处理成响应；
  - `total_ms`：整个请求，从网络传输格式的请求到响应，即以上三者之和。
- **结果。** JSON 中包含每一项的 P50/P95/P99、均值、最小值和最大值，所有样本，设备，解析后的配置和算子选择，以及预热之后每个阶段捕获的 CUDA graph（计数非零说明有计时运行遇到了预热未覆盖的形状）。

经网络、包含前端在内的单机器人时延，用 `robot_load.py --robots 1` 测量。

## 机器人负载

`robot_load.py` 用虚拟机器人对运行中的 WebSocket + msgpack 服务施加负载，并按轮报告时延。它只需要客户端库（合成 Wall-OSS 观测需要 OpenCV），不需要 XInfer 本身：

```bash
python -m pip install numpy 'websockets>=12' msgpack msgpack-numpy opencv-python-headless
```

每台机器人在各自的进程中运行，使用一个连接，最多只有一个请求在途。脚本不负责启动或停止服务。

### 负载

机器人以闭环方式运行。每台机器人在本轮共享的起始时刻（机器人启动后 2 s）之后，再随机延迟一个请求周期（180 ms）内的时间发出第一个请求（均匀分布；由 `--start-seed` 和轮次决定种子，记录在每轮汇总的 `start_delays_ms` 中）。之后的每个请求在机器人的动作队列需要时发送；若上一个响应到达得更晚，则在其到达时发送（`benchmarks/virtual_robot.py`，使用桌面客户端的参数）：

- 队列以 200 Hz 消耗，每个模型步 6 个控制点；
- 下一个请求在上一个动作块（chunk）的采集时刻之后 6 个模型步（180 ms）到期，第一个响应之后则为 3 个模型步。

只要响应快于这个周期，机器人在整轮中都保持起步时的相位，所以一轮只测到机器人发送时刻的一种排布：同时起步的机器人每个周期都同时发送。随机的起步延迟加上多轮，是在各种排布中取样，而不是总在机器人同步发送的状态下测量；同一种子给出同样的排布，所以两个版本可以逐轮对比。

请求内容取决于模型：

- **Wall-OSS（默认）：** 合成观测，包含两路 640×480 JPEG 腕部视图、一路 1280×720 原始头部视图和随机的手臂位姿（xyz 在 0.5 m 内，roll/pitch/yaw 在 π 内，夹爪 0..1）。Wall-OSS 把状态以文本形式写入 prompt，因此 prompt 长度会变化，与真实机器人一样（200 个随机位姿为 739–756 token）。第一个响应之后，每个请求携带 RTC 字段：
  - 上一个动作块，按已执行的步数平移；
  - `inference_delay`：最近五次延迟的第 75 百分位数（初始为 5），限制在 1..`--max-inference-delay` 个模型步之间，再加 2；
  - `num_flow_steps`：服务元数据帧声明的 flow 步数（`flow_steps` 或 `rtc.num_flow_steps`），未声明时为 5。
- **不支持 RTC 的模型（`--requests-file`）：** 机器人 *r* 的第 *k* 个请求为给定文件中的第 (*r* + *k*) mod *n* 个请求，不带 RTC 字段。请求文件为 JSON（`.json`）或 msgpack，包含一个请求映射或其列表，与 `xinfer-serve --warmup` 相同。

仓库的压测口径：每台机器人 80 个请求，前 20 个作为预热，共 8 轮：

```bash
python benchmarks/robot_load.py --url ws://127.0.0.1:33985 --robots 8 \
  --requests 80 --warmup-requests 20 --rounds 8 --out artifacts/wall-oss-n8
python benchmarks/robot_load.py --url ws://127.0.0.1:33985 --robots 7 --requests-file requests.json \
  --requests 80 --warmup-requests 20 --rounds 8 --out artifacts/no-rtc-n7
```

使用 `--requests-file` 时，把同一文件也传给 `xinfer-serve --warmup`，使预热覆盖这些请求的形状。

### 结果

- **`summary.json`**，每轮一条记录：
  - 行数、错误数、缺失行数和稳态行数；
  - `server_infer_ms` 和 `client_ms` 的 P50/P95/P99；
  - 稳态窗口内每秒成功的稳态响应数；
  - 已发送的 `inference_delay` 值的直方图；
  - 机器人的总下溢（underflow）：等待稳态响应期间动作队列为空的时间。
- **`round_<k>.csv`**：每个请求一行。
- **`run.json`**：命令、源码版本、输入的 sha256 以及服务的元数据帧。

统计使用稳态行：每台机器人在前 `--warmup-requests` 个请求之后的请求。错误按全部行统计。响应带有 `error`、`status` 不是 `"ok"`，或（对 Wall-OSS）不含绝对位姿时，记为错误。连接失败的机器人会停止，其剩余的行计为缺失。

- **`client_ms`**：从请求即将序列化之前到响应接收完毕。不包括构建观测的时间（`payload_ms`）。
- **`server_infer_ms`**：服务自己报告的 `server_timing.infer_ms`。

每轮打印其 P50/P95/P99 时延、吞吐和机器人的下溢。一台机器人每 180 ms 最多发送一个请求，因此每台机器人的吞吐低于 5.6 req/s 意味着服务已跟不上。所有轮次都没有错误和缺失行时，退出状态为 0。

**单卡容量**是满足以下条件的最大机器人数：所有轮次都没有错误或缺失行，且稳态 `server_infer_ms` 的 P50 低于 150 ms，同时所有更小的机器人数也满足同样的规则。每个机器人数都要在新启动的服务上测量，并分别报告每一轮。

## 性能表

[docs/performance.md](performance.zh-CN.md) 列出每个服务配置当前的单请求时延和单卡容量。`benchmarks/performance.py` 在可见的 GPU 上测量一行，该 GPU 上不能运行其他任务：

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/performance.py --name "Wall-OSS" --model wall_oss \
  --checkpoint /path/to/5_25000 --latency-requests flow.msgpack rtc.msgpack \
  --warmup flow.msgpack rtc.msgpack --robots 3 4 5 --out artifacts/perf-wall-oss
```

- **单请求：** 对模型的请求集运行 `request_latency.py`（5 次预热后 50 次），视图在设备上解码和 resize（`image_preprocess_device="gpu"`，即 `Policy` 的默认值）。`--model-options` 中的 `image_preprocess_device` 同时作用于这一部分和负载部分。
- **压测：** 按机器人数从小到大，每个数都新启动一个使用默认设置和模型预热文件的 `xinfer-serve`，然后按仓库口径运行 `robot_load.py`（每台机器人 80 个请求，丢弃前 20 个，8 轮，起步相位随机）。支持实时分块的模型使用带 RTC 的合成桌面观测（默认负载）；不支持的模型使用其请求集（`--load-requests`）。脚本在第一个未通过的机器人数后停止。
- **容量：** 任何一轮都没有错误或缺失行、各轮服务端 P50 的中位数（8 轮各自 P50 的中位数）低于 150 ms，且所有更小的已测机器人数也都通过的最大机器人数。表中给出该机器人数各轮 P50 和 P95 的中位数；`row.json` 另外记录最差的一轮。饱和吞吐是测得的最高完成速率。
- **输出：** `row.json` 记录全部数字、配置、环境以及每个机器人数前后的机器负载；表格行打印到标准输出。

改变某个配置性能的合并请求要同时更新它的那一行，并在描述中写明之前的数值。PyTorch、xcompute 或口径变化后，所有受影响的行都要重新测量。

