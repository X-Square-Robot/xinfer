"""Flow route decisions include optional arguments and tensor layout."""

import pytest
import torch

from xinfer.ops.backends.cuda import flow
from xinfer.ops.registry import observe_reference_calls
from xinfer.ops.reference import flow as ref


def test_embedding_layout_falls_back_for_a_strided_mask_and_broadcast_addend(monkeypatch):
    monkeypatch.setattr(flow, "_has_kernels", lambda device: True)
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
    monkeypatch.setattr(flow, "_has_kernels", lambda device: True)
    row_mask = torch.arange(4)[:, None].expand(4, 8) % 2 == 0
    references = []
    with observe_reference_calls(references.append):
        result = flow.flow_velocity_euler(x, h, w, 0.25, row_mask, padding)
    assert references == ["flow_velocity_euler"]
    assert torch.equal(result, x + 0.25 * (h @ w.T) * row_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op", ["flow_input_embed", "flow_velocity_euler"])
@pytest.mark.parametrize("tier", ["measured", "can_run", "reference"])
def test_flow_dispatch_uses_native_capabilities_without_tuning(op, tier, monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    if flow.arch(device) not in flow.BUILD_ARCHS:
        pytest.skip("requires the in-tree flow kernels for this device")
    generator = torch.Generator(device=device).manual_seed(213)

    def tensor(*shape):
        return torch.randn(shape, device=device, generator=generator)

    b, m, a = (1, 5, 26) if tier == "measured" else (2, 5, 8)
    x = tensor(b, m, a)
    if op == "flow_input_embed":
        entry = "denoise_input_embed"
        width = 17 if tier == "reference" else 32
        args = x, tensor(1, width, a)
        options = {"override": tensor(b, m, width), "override_mask": tensor(b, m) > 0}
        expected = ref.flow_input_embed(*args, **options)
    else:
        entry = "denoise_output_euler"
        width = 96 if tier == "reference" else 1024
        args = x, tensor(b, m, width).bfloat16(), tensor(a, width) / 32, 0.1
        mask = tensor(b, a) > 0
        # The native API also accepts [B, A]. Make its batch broadcast explicit for the torch reference.
        full_mask = mask[:, None, :].expand_as(x)
        padding = tensor(b, m, a)
        options = {"dof_mask": full_mask if tier == "reference" else mask, "padding_velocity": padding}
        expected = ref.flow_velocity_euler(*args, dof_mask=full_mask, padding_velocity=padding)
    for setting in (None, "0", "1"):
        if setting is None:
            monkeypatch.delenv("XINFER_OP_TUNING", raising=False)
        else:
            monkeypatch.setenv("XINFER_OP_TUNING", setting)
        references, routes = [], []
        with observe_reference_calls(references.append, routes.append):
            result = getattr(flow, op)(*args, **options)
        if tier == "reference":
            assert references == [op] and not routes
            assert torch.equal(result, expected)
        else:
            assert not references and routes == [f"{tier} ({entry})"]
            tolerance = 1e-2 if op == "flow_input_embed" else 2e-3
            torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
