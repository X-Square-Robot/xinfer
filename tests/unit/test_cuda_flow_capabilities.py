"""Flow route decisions include optional arguments and tensor layout."""

import pytest
import torch

from xinfer.ops.backends.cuda import flow
from xinfer.ops.registry import observe_reference_calls
from xinfer.ops.reference import flow as ref


def test_embedding_layout_falls_back_for_a_strided_mask_and_broadcast_addend(monkeypatch):
    monkeypatch.setattr(flow, "denoise_input_embed_takes", lambda *_: True)
    monkeypatch.setattr(flow, "denoise_input_embed", lambda *_, **kw: pytest.fail("invalid native layout"))
    x, weight = torch.ones(2, 4, 8), torch.ones(1, 16, 8)
    mask = torch.ones(2, 8, dtype=torch.bool)[:, ::2]
    addend = torch.ones(1, 1, 16)
    for options in ({"input_mask": mask}, {"addends": (addend,)}):
        observed = []
        with observe_reference_calls(observed.append):
            out = flow.flow_input_embed(x, weight, **options)
        assert observed == ["flow_input_embed"] and torch.equal(out, ref.flow_input_embed(x, weight, **options))


def test_aligned_eligible_flow_rows_and_weight_layout(monkeypatch):
    x = torch.ones(2, 4, 8)
    contiguous = torch.ones(2, 4, 1024)
    h = torch.ones(2 * 4 * 1024 + 1)[1:].reshape_as(contiguous)
    w = torch.ones(8, 1024)
    mask = torch.ones(2, 8, dtype=torch.bool)
    padding = torch.zeros_like(x)
    assert flow._euler_layout(x, contiguous, w, mask, padding, None, None)
    assert not flow._euler_layout(x, h, w, mask, padding, None, None)
    monkeypatch.setattr(flow, "denoise_output_euler_takes", lambda *_: True)
    with observe_reference_calls(lambda _: None):
        result = flow.flow_velocity_euler(x, h, w, 0.25, mask, padding)
    assert torch.equal(result, x + 0.25 * (h @ w.T) * mask[:, None, :])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op", ["flow_input_embed", "flow_velocity_euler"])
def test_unmeasured_routes_are_opt_in_and_match_the_reference(op, monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    if flow.arch(device) not in flow.BUILD_ARCHS:
        pytest.skip("requires the in-tree flow kernels for this device")
    generator = torch.Generator(device=device).manual_seed(213)
    def tensor(*shape):
        return torch.randn(shape, device=device, generator=generator)
    x = tensor(2, 5, 8)
    if op == "flow_input_embed":
        args = x, tensor(1, 32, 8)
        kw = {"override": tensor(2, 5, 32), "override_mask": tensor(2, 5) > 0}
    else:
        args = x, tensor(2, 5, 1024).bfloat16(), tensor(8, 1024) / 32, 0.1
        kw = {"dof_mask": tensor(2, 8) > 0, "padding_velocity": tensor(2, 5, 8)}
    expected = getattr(ref, op)(*args, **kw)
    monkeypatch.delenv("XINFER_FLOW_UNMEASURED", raising=False)
    reference_calls, routes = [], []
    with observe_reference_calls(reference_calls.append, routes.append):
        assert torch.equal(getattr(flow, op)(*args, **kw), expected)
        monkeypatch.setenv("XINFER_FLOW_UNMEASURED", "1")
        result = getattr(flow, op)(*args, **kw)
    assert reference_calls == [op] and routes == ["kernels:can_run"]
    tolerance = 1e-2 if op == "flow_input_embed" else 2e-3
    torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
