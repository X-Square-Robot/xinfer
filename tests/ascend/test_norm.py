"""Ascend normalization selection and row-private output ownership."""
from types import SimpleNamespace

import pytest
import torch

from xinfer.ops.backends.ascend import _native, core


@pytest.fixture
def stream_api(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        current_stream=lambda device: SimpleNamespace(npu_stream=1)), raising=False)


def test_gated_norm_launches_each_robot_into_its_own_output(monkeypatch, stream_api):
    monkeypatch.setattr(_native, "core_library", lambda: object())
    calls = []
    monkeypatch.setattr(core, "_launch", lambda library, name, tensors, scalars: calls.append((name, tensors)))
    source = torch.ones(4, 16, 1024, dtype=torch.bfloat16)
    output, gate, residual_out = (torch.empty_like(source) for _ in range(3))
    modulation = torch.ones(4, 3072)
    pinned = torch.arange(4, dtype=torch.int32)
    core.gated_residual_adarmsnorm(output, gate, residual_out, source, source, source,
                                  modulation, modulation, pinned)
    assert len(calls) == 4
    for index, (name, tensors) in enumerate(calls):
        assert name == "adarmsnorm_action_gated_bf16_f32_pinned"
        assert tensors[3].data_ptr() == residual_out[index].data_ptr()
        assert tensors[4].data_ptr() == output[index].data_ptr()
        assert tensors[5].data_ptr() == gate[index].data_ptr()
        assert tensors[6].data_ptr() == modulation[index].data_ptr()
        assert tensors[-1].data_ptr() == pinned[index].data_ptr()
        assert tensors[4].shape == (1, 16, 1024)


def test_identity_norm_keeps_inplace_rows_and_shared_scalar_options(monkeypatch, stream_api):
    monkeypatch.setattr(_native, "core_library", lambda: object())
    calls = []
    monkeypatch.setattr(core, "_launch", lambda library, name, tensors, scalars: calls.append((name, tensors)))
    residual, branch, gate, previous = (torch.ones(2, 16, 1024, dtype=torch.bfloat16) for _ in range(4))
    modulation = torch.ones(2, 3072)
    core.double_adarmsnorm_gated_residual_inplace(residual, branch, gate, previous,
                                                modulation, modulation, 1e-6)
    assert len(calls) == 2
    for index, (name, tensors) in enumerate(calls):
        assert name == "adarmsnorm_action_double_bf16_f32"
        assert tensors[0].data_ptr() == tensors[3].data_ptr() == residual[index].data_ptr()
        assert tensors[1].data_ptr() == tensors[4].data_ptr() == branch[index].data_ptr()
        assert tensors[5].data_ptr() == gate[index].data_ptr()


def test_ascend_norm_binding_preserves_the_layer_threshold(monkeypatch):
    from xinfer.ops.backends.ascend import norm, soupunlock

    monkeypatch.setattr(soupunlock, "enabled", lambda family: family == "rmsfull")
    monkeypatch.setattr(soupunlock, "from_layer", lambda: 8)
    inputs = torch.ones(1, 3, 4), torch.ones(1, 3, 4), torch.zeros(4), 1e-6
    marker = torch.full((1, 3, 4), 7.0)
    monkeypatch.setattr(soupunlock, "rmsnorm_full", lambda *args: (marker, marker))
    lower = norm.add_zero_centered_rms_norm.bind(layer_index=7)
    upper = norm.add_zero_centered_rms_norm.bind(layer_index=8)
    assert lower is norm.add_zero_centered_rms_norm
    assert upper(*inputs)[0] is marker
    assert torch.equal(lower(*inputs)[0], torch.full((1, 3, 4), 2.0))

    def fail(*args):
        raise RuntimeError("norm launch failed")

    monkeypatch.setattr(soupunlock, "rmsnorm_full", fail)
    with pytest.raises(RuntimeError, match="norm launch failed"):
        upper(*inputs)
