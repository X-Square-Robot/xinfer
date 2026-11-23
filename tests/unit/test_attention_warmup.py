"""CUDA attention capability dispatch and portable warmup records."""

from contextlib import nullcontext

import pytest
import torch

from xinfer.ops.backends.cuda import _op_tuning as tuning
from xinfer.ops.backends.cuda import attn
from xinfer.ops.reference import attn as ref
from xinfer.ops.registry import observe_reference_calls


def test_library_choices_survive_restart_and_readonly_misses(tmp_path, monkeypatch):
    path = tmp_path / "tuning.json"
    monkeypatch.setenv("XINFER_GEMM_TUNING_FILE", str(path))
    monkeypatch.setattr(tuning, "_versions", lambda *args: ("board", "native revision", "library version"))
    monkeypatch.setattr(torch.cuda, "device", lambda d: nullcontext())
    monkeypatch.setattr(tuning, "capturing", lambda d: False)
    monkeypatch.setattr(tuning, "_compare", lambda *args: "flash_attn")
    monkeypatch.setattr(tuning, "_winners", {})
    args = (torch.ones(3),)
    assert tuning.choose("segment_attention", args, None, None, "flash_attn") == "flash_attn"
    before = path.read_bytes()
    tuning._winners.clear()
    monkeypatch.setenv("XINFER_GEMM_TUNING_READONLY", "1")
    monkeypatch.setattr(tuning, "_compare", lambda *args: pytest.fail("measured during replay"))
    assert tuning.choose("segment_attention", args, None, None, "flash_attn") == "flash_attn"
    assert tuning.choose("segment_attention", (torch.ones(2),), None, None, "flash_attn") is None
    assert path.read_bytes() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op", ["segment_attention", "gated_causal_attention", "gated_delta_net"])
def test_unmeasured_attention_routes_and_outputs(op, monkeypatch):
    d = torch.device("cuda", torch.cuda.current_device())
    if attn.arch(d) not in attn.BUILD_ARCHS or (op == "segment_attention" and attn.arch(d) != 120):
        pytest.skip("the in-tree kernel must support this device")
    generator = torch.Generator(device=d).manual_seed(151)
    def rand(*shape, dtype=torch.bfloat16):
        return (torch.randn(*shape, device=d, generator=generator) / 8).to(dtype)
    if op == "segment_attention":
        args = (rand(35, 8, 64), rand(35, 8, 64), rand(35, 8, 64), (0, 11, 35))
    elif op == "gated_causal_attention":
        args = (rand(65, 16, 128), rand(65, 2, 128), rand(65, 2, 128), rand(65, 2048))
    else:
        args = (rand(1, 35, 768), rand(1, 35, 260), rand(768, 4), rand(2, dtype=torch.float32),
                rand(2, dtype=torch.float32), rand(128), 2, 2, 128, 128, 1e-6)
    monkeypatch.setattr(tuning, "_winners", {})
    expected = getattr(ref, op)(*args)
    for setting in ("0", "1"):
        monkeypatch.setenv("XINFER_OP_TUNING", setting)
        routes = []
        with observe_reference_calls(lambda name: pytest.fail(name), routes.append):
            result = getattr(attn, op)(*args)
        assert len(routes) == 1 and routes[0] in ("kernels", "flash_attn", "fla")
        if setting == "0":
            assert routes == ["kernels"]
        relative = (result.float() - expected.float()).norm() / expected.float().norm()
        assert relative < (0.02 if op == "gated_delta_net" else 0.01)
