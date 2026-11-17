"""The BF16-head adapter's temporary hidden rows belong to its device stream."""

from types import SimpleNamespace

import pytest
import torch

from xinfer.ops.backends.ascend import _native, core


def test_flow_workspace_reuses_one_stream_and_isolates_other_captures(monkeypatch):
    state = SimpleNamespace(stream=11, capturing=False)
    runtime = SimpleNamespace(current_stream=lambda device: SimpleNamespace(npu_stream=state.stream),
                              current_device=lambda: 0, is_current_stream_capturing=lambda: state.capturing)
    monkeypatch.setattr(torch, "npu", runtime, raising=False)
    monkeypatch.setattr(_native, "_scratch", {})
    monkeypatch.setattr(_native, "_retired", [])
    monkeypatch.setattr(_native, "core_library", lambda: object())
    writes, reads = [], []

    def launch(library, name, tensors, scalars):
        if name == "denoise_fp32head_prep":
            writes.append(tensors[1])
        elif name == "denoise_fp32head_cube":
            reads.append(tensors[2])

    monkeypatch.setattr(core, "_launch", launch)
    x = torch.zeros(1, 16, 26)
    hidden = torch.ones(1, 16, 1024, dtype=torch.bfloat16)
    weight = torch.ones(26, 1024, dtype=torch.bfloat16)

    def call():
        assert core.denoise_output_euler(x, hidden, weight, dt=0.1) is x
        assert writes[-1].data_ptr() == reads[-1].data_ptr()
        return writes[-1]

    first = call()
    assert call().data_ptr() == first.data_ptr()
    state.stream = 22
    other = call()
    assert other.data_ptr() != first.data_ptr()
    first.fill_(3)
    other.fill_(7)
    assert torch.equal(first, torch.full_like(first, 3))

    state.capturing = True
    state.stream = 11
    assert call().data_ptr() == first.data_ptr()
    state.stream = 33
    launches = len(writes)
    with pytest.raises(RuntimeError, match="before capture"):
        call()
    assert len(writes) == launches


def test_fp32_model_head_does_not_enter_the_bf16_scratch_route(monkeypatch):
    monkeypatch.setattr(_native, "core_library", lambda: object())
    monkeypatch.setattr(_native, "scratch", lambda *args: pytest.fail("unsupported head allocated workspace"))
    with pytest.raises(ValueError, match="BF16"):
        core.denoise_output_euler(torch.zeros(1, 16, 26), torch.ones(1, 16, 1024, dtype=torch.bfloat16),
                                 torch.ones(26, 1024, dtype=torch.float32), dt=0.1)
