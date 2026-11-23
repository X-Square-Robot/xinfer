"""The FP32-head adapter preserves its ABI, input ownership and stream-private workspace."""

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
    weight = torch.ones(26, 1024, dtype=torch.float32)

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


def test_bf16_head_declines_before_entering_the_fp32_launcher(monkeypatch):
    monkeypatch.setattr(_native, "core_library", lambda: object())
    monkeypatch.setattr(_native, "scratch", lambda *args: pytest.fail("unsupported head allocated workspace"))
    with pytest.raises(ValueError, match="FP32"):
        core.denoise_output_euler(torch.zeros(1, 16, 26), torch.ones(1, 16, 1024, dtype=torch.bfloat16),
                                 torch.ones(26, 1024, dtype=torch.bfloat16), dt=0.1)


def test_flow_binding_preserves_its_input_and_packs_integer_pinned_rows(monkeypatch):
    from xinfer.ops.backends.ascend import flow

    def launch(x, h, weight, *, out=None, pinned=None, **kwargs):
        assert pinned.dtype == torch.int32
        target = x if out is None else out
        target.fill_(7)
        return target

    monkeypatch.setattr(flow, "_output_euler", launch)
    x = torch.zeros(2, 16, 26)
    result = flow.flow_velocity_euler.__wrapped__(x, torch.ones(2, 16, 1024, dtype=torch.bfloat16),
                                                torch.ones(26, 1024), 0.1, pinned=torch.tensor([3, 5]))
    assert x.eq(0).all() and result.eq(7).all()
    assert result.data_ptr() != x.data_ptr()


def test_fp32_program_declines_a_prenorm_it_does_not_implement(monkeypatch):
    monkeypatch.setattr(_native, "core_library", lambda: object())
    monkeypatch.setattr(core, "_launch", lambda *args: pytest.fail("unsupported prenorm launched"))
    with pytest.raises(ValueError, match="no prenorm"):
        core.denoise_output_euler(torch.zeros(1, 16, 26), torch.ones(1, 16, 1024, dtype=torch.bfloat16),
                                 torch.ones(26, 1024), dt=0.1, prenorm="rms")


@pytest.mark.parametrize("name,value", [
    ("x", torch.zeros(1, 26, 16).transpose(1, 2)),
    ("x", torch.zeros(1, 16, 26, dtype=torch.bfloat16)),
    ("bias", torch.ones(26, dtype=torch.bfloat16)),
    ("pinned", torch.tensor([3], dtype=torch.int64)),
    ("pinned_values", torch.zeros(1, 16, 26, dtype=torch.bfloat16)),
    ("padding_velocity", torch.zeros(1, 26)),
    ("dt", torch.ones(1, dtype=torch.float64)),
])
def test_unsafe_pointer_layout_declines_before_allocating_or_launching(monkeypatch, name, value):
    monkeypatch.setattr(_native, "core_library", lambda: object())
    monkeypatch.setattr(_native, "scratch", lambda *args: pytest.fail("unsupported layout allocated workspace"))
    monkeypatch.setattr(core, "_launch", lambda *args: pytest.fail("unsupported layout launched"))
    arguments = dict(x=torch.zeros(1, 16, 26), h=torch.ones(1, 16, 1024, dtype=torch.bfloat16),
                     weight=torch.ones(26, 1024), dt=0.1, out=torch.empty(1, 16, 26))
    arguments[name] = value
    with pytest.raises(ValueError):
        core.denoise_output_euler(**arguments)


@pytest.mark.parametrize("name,value", [
    ("head_index", torch.zeros(1, 16, dtype=torch.int64)),
    ("input_mask", torch.ones(1, 16, dtype=torch.int32)),
    ("extra", torch.zeros(1, 16, 26, dtype=torch.bfloat16)),
    ("addends", (torch.zeros(1, 16, 1024, dtype=torch.bfloat16),)),
])
def test_embedding_declines_unsafe_pointer_types_before_launch(monkeypatch, name, value):
    monkeypatch.setattr(_native, "core_library", lambda: object())
    monkeypatch.setattr(core, "_launch", lambda *args: pytest.fail("unsupported embedding layout launched"))
    arguments = dict(x=torch.zeros(1, 16, 26), weight=torch.ones(1, 1024, 26))
    if name == "extra":
        arguments["weight"] = torch.ones(1, 1024, 52)
    arguments[name] = value
    with pytest.raises(ValueError):
        core.denoise_input_embed(**arguments)


def test_embedding_binding_packs_integer_head_indices(monkeypatch):
    from xinfer.ops.backends.ascend import flow

    def launch(x, weight, *, head_index, **kwargs):
        assert head_index.dtype == torch.int32
        return x

    monkeypatch.setattr(flow, "_input_embed", launch)
    x = torch.zeros(1, 16, 26)
    assert flow.flow_input_embed.__wrapped__(x, torch.ones(2, 1024, 26),
                                            head_index=torch.ones(1, 16, dtype=torch.int64)) is x
