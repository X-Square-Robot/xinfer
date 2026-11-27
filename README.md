<p align="center">
  <img src="docs/assets/xinfer-logo.webp" alt="XInfer — inference for robotics" width="560">
</p>

<h3 align="center">Fast, stable, and accurate VLA inference. From one robot to a fleet.</h3>

<p align="center">
  <b>English</b> | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#features-at-a-glance">Features</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#models-and-hardware">Models</a> ·
  <a href="docs/benchmark.md">Benchmarks</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="https://arxiv.org/abs/2611.14969">Paper</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

**XInfer** is an inference and serving framework for vision-language-action (VLA) models.

## Features at a glance

- **Real-time chunking (RTC).** Predict the next action chunk from a new observation while the robot executes the current one.
- **Ultra-low single-robot latency.** WALL-OSS-1.0-Nano in BF16 measured 20.1 ms P50 per request on an RTX 5090.
- **Multi-robot serving on one accelerator.** Fine-grained stage scheduling lets inference stages from different requests overlap asynchronously on multiple CUDA streams.
- **Low-precision quantization.** Apply MXFP8 or NVFP4 to selected groups of linear layers.
- **Hardware backends.** NVIDIA RTX 5090, Jetson Thor, Ascend 950DT, and HUIXI-AI Rhino RPU (R1) support in development.
- **High-performance kernels.** Use optimized fused kernels and optional FlashAttention, or plug in another kernel library.

## Quick start

The CUDA examples require Python 3.10+, CUDA-enabled PyTorch, and a [compatible BF16 WALL-OSS-1.0-Nano checkpoint](docs/models/wall_oss_1_0_nano.md#checkpoint-requirements). Install from the repository root.

```bash
python -m pip install -e ".[models,serve]"
```

For the optimized path, [build the CUDA kernels](docs/kernels.md). XInfer can also run with PyTorch reference kernels.

### Serve a model

```bash
xinfer-serve \
  --model wall-oss-1.0-nano \
  --checkpoint /path/to/wall-oss-1.0-nano-checkpoint \
  --device cuda \
  --warmup observation.msgpack \
  --port 33985
```

Use a [model request](docs/models/wall_oss_1_0_nano.md#request-and-response) as `observation.msgpack`. For RTC, pass both a standard request and an RTC request to `--warmup`.

Connect to `ws://127.0.0.1:33985` with WebSocket + msgpack. See the [serving guide](docs/serving.md) for deployment options.

### Run in Python

```python
from pathlib import Path

from xinfer import Policy
from xinfer.frontend.protocol import unpackb

observation = unpackb(Path("observation.msgpack").read_bytes())
policy = Policy("wall-oss-1.0-nano", "/path/to/wall-oss-1.0-nano-checkpoint", device="cuda")

try:
    policy.warmup([observation])
    response = policy.infer(observation)
finally:
    policy.engine.shutdown()
```

## Models and hardware

| Model | Model ID | Precision | Real-time chunking | Guide |
|---|---|---|---|---|
| WALL-OSS-1.0-Nano | `wall-oss-1.0-nano` | BF16; optional MXFP8 or NVFP4 for vision and prefill (action stays BF16) | Supported | [Setup and inputs](docs/models/wall_oss_1_0_nano.md) |
| WALL-OSS-0.5 | `wall-oss-0.5` | BF16 | Supported | [Setup and inputs](docs/models/wall_oss_0_5.md) |

WALL-OSS-0.5 requires the [internal checkpoint layout](docs/models/wall_oss_0_5.md#requirements).

| Hardware | Device | Runtime |
|---|---|---|
| NVIDIA GPUs · RTX 5090 (SM120) | `cuda` | PyTorch + CUDA |
| NVIDIA Jetson Thor (SM110) | `cuda` | PyTorch + CUDA |
| Huawei Ascend 950DT NPU | `npu` | PyTorch + `torch_npu` + CANN · [Setup](docs/ascend950dt.md) |
| Rhino RPU (R1, experimental) | `rpu` | Rhino Launch SDK AOT · [Operators](docs/rpu_ops.md) · [Profile](docs/models/wall_oss_1_0_nano_rpu.md) |
| CPU | `cpu` | Reference implementations and tests |

## Performance

Single-accelerator measurements from `benchmarks/performance.py` using the production robot request format (three camera views per request). The single-request P50 and fleet-load figures come from separate runs. See the [performance guide](docs/performance.md) for benchmark conditions and reproduction status.

<table>
  <thead>
    <tr>
      <th scope="col" rowspan="2">Model</th>
      <th scope="col" rowspan="2">Precision and stages</th>
      <th scope="col" rowspan="2">Accelerator</th>
      <th scope="col">Single-request latency</th>
      <th scope="col" colspan="3">Closed-loop fleet serving</th>
    </tr>
    <tr>
      <th scope="col">P50</th>
      <th scope="col">Capacity (robots)</th>
      <th scope="col">Server P50 / P95 at capacity</th>
      <th scope="col">Peak throughput</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>BF16</td>
      <td>RTX 5090</td>
      <td>20.1 ms</td>
      <td>14</td>
      <td>85 / 107 ms</td>
      <td>81.3 req/s</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>MXFP8 vision and prefill</td>
      <td>RTX 5090</td>
      <td>15.1 ms</td>
      <td>25</td>
      <td>67 / 88 ms</td>
      <td>141.0 req/s</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>BF16</td>
      <td><a href="https://github.com/X-Square-Robot/xinfer/pull/33">Ascend 950DT</a></td>
      <td>20.6 ms</td>
      <td>15</td>
      <td>129 / 163 ms</td>
      <td>83.9 req/s</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>BF16</td>
      <td>Jetson AGX Thor</td>
      <td>TBD</td>
      <td colspan="3">TBD</td>
    </tr>
    <tr>
      <td>WALL-OSS-1.0-Nano</td>
      <td>NVFP4 vision and prefill</td>
      <td>RTX 5090</td>
      <td>TBD</td>
      <td colspan="3">TBD</td>
    </tr>
  </tbody>
</table>

Server P50 / P95 is measured at the listed robot capacity. Peak throughput is the highest steady-state rate across tested loads and rounds.

## WALL-OSS-1.0-Nano in Production: Parcel Sorting

| Hardware | Parcel throughput (parcels/h) |
|---|---:|
| NVIDIA RTX 5090 | 2,231 |
| HUIXI-AI Rhino RPU (R1) | 2,190 |

## How it works

<p align="center">
  <img src="docs/assets/xinfer-architecture.svg" alt="Model stages call operators bound at load time. The operator registry supports optimized fused kernels, optional CUDA FlashAttention, PyTorch references and third-party plugins. The stage engine serves RTX 5090, Jetson AGX Thor, Ascend 950DT and CPU. HUIXI-AI Rhino RPU (R1) uses a separate experimental native path from Python Policy." width="100%">
</p>

The engine schedules model stages. The operator registry selects device kernels at load time. The dashed line marks the experimental HUIXI-AI Rhino RPU (R1) path. See the [engine guide](docs/engine.md) for scheduling details.

## Benchmarks

See the [benchmark guide](docs/benchmark.md) for commands and the [Nano validation results](docs/models/wall_oss_1_0_nano_validation.md) for stage timings and accuracy.

## Documentation

| Task | Guides |
|---|---|
| Deploy | [Serving](docs/serving.md) · [Configuration](docs/configuration.md) · [Metrics](docs/metrics.md) |
| Optimize | [Kernels](docs/kernels.md) · [RPU operators](docs/rpu_ops.md) · [Quantization](docs/quantization.md) · [Precision](docs/precision.md) · [Profiling](docs/profiling.md) |
| Extend | [Architecture and design](docs/design.md) · [Model interface](xinfer/models/interface.py) |

## Citation

```bibtex
@misc{zhang2026scalingvlaservingrobotic,
      title={Scaling VLA Serving Under Robotic Execution Constraints},
      author={Yue Zhang and Zicong Ye and Lufang Chen and Dance Yang and Mingyang Li and Zhenwei Duan and Michael Zhao and Ziye Deng and Starrick Liu and Ruizhi Chengze and YongHua Chen and SongAo Han and Yu Sun and Ran Yu and Xiaofan Li and Ruyi Gan and Hao Wang},
      year={2026},
      eprint={2611.14969},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.14969},
}
```

## Contributing

See the [contribution guide](CONTRIBUTING.md).

## License

The XInfer source code is licensed under the [Apache License 2.0](LICENSE).
