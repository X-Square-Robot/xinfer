"""WALL-OSS-1.0-Nano optimized stages on CUDA, using a tiny random checkpoint."""

import pytest
import torch
from safetensors.torch import save_file

from xinfer import ops
from xinfer.models.interface import StageBatch
from xinfer.platforms import resolve_platform

from wall_oss_1_0_nano_checkpoint import D, H, write_checkpoint
from test_wall_oss_1_0_nano_tiny import _inputs

pytestmark = [pytest.mark.gpu]
ROWS = 300


def _checkpoint(root, device):
    from xinfer.models.wall_oss_1_0_nano.config import WallOSS10NanoConfig
    from xinfer.models.wall_oss_1_0_nano.model import WallOSS10Nano, weight_mapping

    write_checkpoint(root)
    torch.manual_seed(0)
    # The tiny rotary width is outside the native RoPE kernel's alignment envelope.
    pins = {"rope_full": "reference"} if device == "cuda" else None
    table = ops.registry.build_table(resolve_platform(device), backends=pins)
    with ops.use_ops(table):
        shell = WallOSS10Nano(WallOSS10NanoConfig.load(root), embedding_rows=ROWS, numerics="reference")
    tensors = {}
    for target, source in weight_mapping(shell).items():
        shape = dict(shell.named_parameters())[target].shape
        for name in source.keys:
            tensors.setdefault(name, torch.randn(shape) * 0.02)
    save_file(tensors, str(root / "model.safetensors"))
    numerics = "optimized" if device == "cuda" else "reference"
    with ops.use_ops(table):
        model = WallOSS10Nano.from_checkpoint(root, device=device, numerics=numerics)
    return model, table


def _run(model, stage, inputs):
    device = next(model.parameters()).device
    batch = StageBatch(stage, 1, inputs)
    out = {
        name: torch.empty(spec.shape, dtype=spec.dtype, device=device)
        for name, spec in model.stage_outputs(stage, batch).items()
    }
    getattr(model, model.stage_graph[stage].method)(StageBatch(stage, 1, inputs, out))
    return out


def _sample(model, rtc_delay):
    inputs = _inputs(model, rtc_delay=rtc_delay)
    device = next(model.parameters()).device
    inputs = {name: value.to(device) if torch.is_tensor(value) else value for name, value in inputs.items()}
    with torch.no_grad():
        inputs["image_embeds"] = _run(model, "vision", inputs)["image_embeds"]
        inputs["prompt_kv"] = _run(model, "prefill", inputs)["prompt_kv"]
        return _run(model, "action", inputs)["predict_action_norm"]


@pytest.fixture(scope="module")
def tiny_on_cuda(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    ops.registry.load_plugins()
    root = tmp_path_factory.mktemp("wall-oss-1.0-nano-cuda")
    model, table = _checkpoint(root, "cuda")
    if table.selection("gemm_epilogue").library != "kernels":
        pytest.skip("needs the CUDA kernel library (docs/kernels.md)")
    assert table.selection("qkv_rope_split").library == "kernels"
    assert table.selection("gemm_split").library == "kernels"
    return model


@pytest.fixture(scope="module")
def tiny_on_cpu(tmp_path_factory):
    root = tmp_path_factory.mktemp("wall-oss-1.0-nano-cpu")
    model, _ = _checkpoint(root, "cpu")
    return model


@pytest.mark.parametrize("rtc_delay", [0, 3])
def test_the_cuda_stages_match_the_reference_composition(tiny_on_cuda, tiny_on_cpu, rtc_delay):
    cuda = _sample(tiny_on_cuda, rtc_delay)
    cpu = _sample(tiny_on_cpu, rtc_delay)
    assert cuda.shape == cpu.shape == (1, H, D)
    assert torch.isfinite(cuda).all()
    relative = (cuda.float().cpu() - cpu.float()).norm() / cpu.float().norm().clamp_min(1e-12)
    assert relative < 2e-2
    if rtc_delay:
        torch.testing.assert_close(cuda[:, :rtc_delay].cpu(), cpu[:, :rtc_delay], rtol=0, atol=0)
