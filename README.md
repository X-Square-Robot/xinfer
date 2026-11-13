<p align="center">
  <img src="docs/assets/xinfer-logo.webp" alt="XInfer" width="560">
</p>

<p align="center">
  <b>English</b> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#models-and-hardware">Models &amp; hardware</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#documentation">Documentation</a>
</p>

**XInfer runs vision-language-action (VLA) models for robots.** Given camera images, robot state and an instruction, it returns a sequence of actions. Run a policy in Python for one robot, or serve multiple robots over WebSocket from one accelerator. Both interfaces use the same inference engine.

## Quick start

Install Python 3.10+ and a PyTorch build for your accelerator. From a fresh checkout:

```bash
git clone https://github.com/X-Square-Robot/xinfer.git
cd xinfer
python -m pip install -e ".[models,serve]"
python -c "from xinfer.models import available_models; print(available_models())"
```

A clean installation lists `('wall_oss', 'wallx2b')`. You can run this check before obtaining weights.

### Serve a model

This example serves Wall-X 2B on CUDA. Supply a [compatible checkpoint](docs/models/wallx2b.md#checkpoint-requirements) and one recorded observation named `flow.msgpack` containing the three camera views, arm poses and instruction in the [request format](docs/models/wallx2b.md#request-and-response).

```bash
xinfer-serve --model wallx2b --checkpoint /path/to/wallx2b-checkpoint \
  --device cuda --warmup flow.msgpack --port 33985
```

Once warmup finishes, check readiness from a second terminal:

```bash
curl -fsS http://127.0.0.1:33985/healthz
```

The service replies `ok`. Send observations to `ws://127.0.0.1:33985` using WebSocket + msgpack. For real-time chunking (RTC), include a representative RTC observation alongside the standard request in `--warmup`. See the [serving guide](docs/serving.md) for the wire protocol and deployment options.

### Run in Python

To inspect one prediction in process, use the same checkpoint and observation. This example decodes the camera images on the CPU and runs the model on CUDA, matching the service defaults for this command.

```python
from pathlib import Path

from xinfer import Policy
from xinfer.frontend.protocol import unpackb

observation = unpackb(Path("flow.msgpack").read_bytes())
policy = Policy("wallx2b", "/path/to/wallx2b-checkpoint",
                device="cuda", image_preprocess_device="cpu")
try:
    policy.warmup([observation])
    response = policy.infer(observation)
    print(response["action_pred"].shape)
finally:
    policy.engine.shutdown()
```

For Wall-X 2B's desktop dual-arm format, the shape is `(16, 14)`: 16 action steps, each with seven values per arm.

## Models and hardware

Both models support BF16 inference and RTC. Supply the weights separately.

| Model | Model ID | Checkpoint and inputs |
|---|---|---|
| Wall-X 2B | `wallx2b` | [Checkpoint files and desktop dual-arm observations](docs/models/wallx2b.md) |
| Wall-OSS | `wall_oss` | [Required internal checkpoint layout, tokenizer and normalization files](docs/models/wall_oss.md#requirements). The published Wall-OSS-0.5 and WALL-OSS-FLOW checkpoints use a different layout and cannot load directly. |

| Device | Runtime | Setup |
|---|---|---|
| NVIDIA GPU (`cuda`) | PyTorch + CUDA | [CUDA kernels and reference fallbacks](docs/kernels.md). Use a matching **xcompute** build for optimized BF16 operators. |
| Ascend 950DT (`npu`) | PyTorch + TorchNPU + CANN | [Ascend setup](docs/ascend950dt.md), including NPU graphs, xcompute and DVPP image preprocessing. |
| CPU (`cpu`) | PyTorch | Reference execution and tests. |

## Benchmarks

Median latency for one warmed-up BF16 request, from input processing to the returned action:

| Model | NVIDIA RTX 5090 | Huawei Ascend 950DT |
|---|---:|---:|
| Wall-X 2B | 22.3 ms | 79.7 ms |
| Wall-OSS | 49.2 ms | 80.8 ms |

These are recorded results for specific builds and inputs. The RTX 5090 Wall-X 2B run used an upstream xcompute weight layout that can avoid a copy. The [full results and conditions](docs/performance.md) also report robots per accelerator and latency under concurrent load. To reproduce the measurements, follow the [benchmark guide](docs/benchmark.md).

## How it works

<p align="center">
  <img src="docs/assets/xinfer-architecture.webp" alt="Robot clients send observations through the frontend to the vision, prefill and action stages of the engine." width="100%">
</p>

The frontend prepares observations and converts predictions into robot actions. The engine queues each request by stage, batches compatible work and replays captured graphs where supported. Each accelerator holds one engine and its model weights. See the [engine guide](docs/engine.md) for scheduling and [configuration](docs/configuration.md) for deployment settings.

## Documentation

| Task | Guide |
|---|---|
| Deploy and monitor a service | [Serving](docs/serving.md) · [Metrics and dashboards](docs/metrics.md) |
| Check accuracy and numerical formats | [Precision](docs/precision.md) · [Quantization](docs/quantization.md) · [Wall-X 2B validation](docs/models/wallx2b_validation.md) · [Wall-OSS validation](docs/models/wall_oss_validation.md) |
| Investigate latency | [Profiling](docs/profiling.md) · [Benchmark protocol](docs/benchmark.md) |
| Add a model or contribute an operator | [Model interface](xinfer/models/interface.py) · [Architecture](docs/design.md) · [Contributing](CONTRIBUTING.md) |

This distribution does not currently include a license.
