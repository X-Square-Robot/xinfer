<p align="center">
  <img src="docs/assets/xinfer-logo.webp" alt="XInfer — inference for robotics" width="560">
</p>

<h3 align="center">Fast, stable, and accurate VLA inference. From one robot to a fleet.</h3>

<p align="center">
  <b>English</b> | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#models-and-hardware">Models</a> ·
  <a href="docs/benchmark.md">Benchmarks</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

**XInfer is an inference and serving framework for vision-language-action
(VLA) models.** Our goal is fast, stable, and accurate model inference serving
for robots, from onboard inference to shared fleet serving.

## Why XInfer

- **Fast.** Low single-request latency and more robots per accelerator.
  Asynchronous stage dispatch, device events, and graph replay where supported
  reduce host overhead. Stage queues and zero-wait batching combine compatible
  queued requests, sharing weights and overlapping work across stages.
- **Stable.** Predictable latency and a service that keeps running under fleet
  load, with warmup before serving, admission limits, and per-request failure
  isolation.
- **Accurate.** Outputs follow the original model. By default, batching does
  not change a request's result, and quantized presets pass each model's
  [precision gate](docs/precision.md).
- **Extensible.** Reusable layers and stage interfaces
  simplify model integration. Platform and operator adapters connect model
  code to new chips and optimized kernels.

## Quick start

Use Python 3.10+ and a PyTorch build matched to your accelerator. These examples use NVIDIA CUDA. Install from this checkout:

```bash
python -m pip install -e ".[models,serve]"
```

Bring a checkpoint and its tokenizer, configuration, and normalization assets
in a supported format, for example [Wall-OSS](docs/models/wall_oss.md#requirements).
The examples below use Wall-OSS with the default BF16 precision.

### Serve a model

```bash
xinfer-serve \
  --model wall_oss \
  --checkpoint /path/to/checkpoint \
  --device cuda \
  --warmup observation.msgpack \
  --port 33985
```

`observation.msgpack` is one representative request in the
[model's input format](docs/models/wall_oss.md#run). Warmup runs before the
server accepts connections. For RTC traffic, include both a standard request and
an RTC request in `--warmup`.

Connect at `ws://127.0.0.1:33985` using WebSocket + msgpack. See the
[serving guide](docs/serving.md) for the protocol, health endpoint, and
deployment options.

### Run in Python

```python
from pathlib import Path

from xinfer import Policy
from xinfer.frontend.protocol import unpackb

observation = unpackb(Path("observation.msgpack").read_bytes())
policy = Policy("wall_oss", "/path/to/checkpoint", device="cuda")

try:
    policy.warmup([observation])
    response = policy.infer(observation)
finally:
    policy.engine.shutdown()
```

The response contains the model's action chunk. Camera names, state fields,
and action representations are documented in each model guide.

## Models and hardware

| Model | Model ID | Precision | Real-time chunking | Guide |
|---|---|---|---|---|
| Wall-OSS · internal checkpoints | `wall_oss` | BF16 (xcompute; reference operators) | Supported | [Setup and inputs](docs/models/wall_oss.md) |

Each configuration's single-request latency and robots per GPU: [performance](docs/performance.md).

| Hardware | Device | Runtime |
|---|---|---|
| NVIDIA GPUs · RTX 5090 (SM120) | `cuda` | PyTorch + CUDA |
| Huawei Ascend 950DT NPU | `npu` | PyTorch + `torch_npu` + CANN |
| CPU | `cpu` | Reference implementations and tests |

Each accelerator runs its own Engine and weights.

Optimized CUDA execution uses Triton and optional FlashAttention backends.
MXFP8 additionally uses a compatible **xcompute** build; see
[quantization and kernel setup](docs/quantization.md#backends).
Ascend uses the BF16 reference-operator path through `torch_npu`.

## How it works

<p align="center">
  <img src="docs/assets/xinfer-architecture.webp" alt="Robot clients connect through a Frontend for input and output processing to the XInfer Engine, which runs Vision, Prefill, and Action. Platform adapters branch to NVIDIA GPUs and Ascend 950DT NPUs, identified by their brand logos, plus CPU reference execution. A dashed More chips box represents additional hardware backends." width="100%">
</p>

The Frontend handles input and output processing. Models define stage
computation and tensor handoffs; the Engine manages admission, queues,
batching, and resource lifetimes. Platform adapters provide device execution,
while operator backends supply optimized kernels. The Python API and network
service share the same Engine and configuration rules. See the
[Engine guide](docs/engine.md) for the execution model and
[configuration](docs/configuration.md) for defaults and overrides.

## Benchmarks

Evaluate single-request latency and the number of robots served within a
defined response-latency target:

| Tool | Measures |
|---|---|
| [Single-request latency](benchmarks/request_latency.py) | Input processing, Engine execution, output processing, and total time in process |
| [Robot load](benchmarks/robot_load.py) | Client and server latency, throughput, errors, and action-queue underflow under concurrent robot traffic |

The [benchmark guide](docs/benchmark.md) includes commands and measurement
definitions. Model-specific results and numerical checks are recorded in the
[Wall-OSS validation](docs/models/wall_oss_validation.md) page.

## Documentation

| Topic | Start here |
|---|---|
| Deploy a service | [Serving](docs/serving.md) |
| Understand scheduling and CUDA graphs | [Engine](docs/engine.md) |
| Configure a deployment | [Configuration](docs/configuration.md) |
| Select numerical formats and kernels | [Quantization](docs/quantization.md) |
| Validate model outputs | [Precision](docs/precision.md) |
| Profile operators | [Profiling](docs/profiling.md) |
| Extend the framework | [Model interface](xinfer/models/interface.py) · [Architecture and design](docs/design.md) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for code conventions, tests, and
performance evidence requirements.

---

XInfer is in private development. Licensing and public release are pending.
