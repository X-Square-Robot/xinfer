"""denoise_input_embed and denoise_output_euler against FP32 torch."""
import pytest
import torch

from xinfer.ops.backends.cuda import flow
from conftest import DEVICE, sm


def reference(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
    v = h.float() @ weight.float().t()
    if dof_mask is not None:
        v = dof_mask * v + (1 - dof_mask) * (0 if padding_velocity is None else padding_velocity)
    out = x + dt * v
    if pinned is not None:
        rows = torch.arange(x.shape[1], device=DEVICE)[None, :, None] < pinned[:, None, None]
        out = torch.where(rows, x if pinned_values is None else pinned_values, out)
    return out


@sm(110, 120)
@pytest.mark.parametrize("b,m,a", [(1, 16, 26), (2, 32, 32), (4, 1, 26)])
@pytest.mark.parametrize("h_dtype,w_dtype", [(torch.bfloat16, torch.float32), (torch.bfloat16, torch.bfloat16),
                                             (torch.float32, torch.float32)])
@pytest.mark.parametrize("options", ["plain", "dof", "dof+pinned", "pinned+values"])
def test_matches_reference(b, m, a, h_dtype, w_dtype, options, rnd):
    x, h = rnd(b, m, a, dtype=torch.float32), rnd(b, m, 1024, dtype=h_dtype)
    weight = rnd(a, 1024, dtype=w_dtype, scale=0.05)
    kw = {}
    if "dof" in options:
        kw.update(dof_mask=(rnd(b, m, a, dtype=torch.float32) > 0).float(),
                  padding_velocity=rnd(b, m, a, dtype=torch.float32))
    if "pinned" in options:
        kw.update(pinned=torch.full((b,), min(5, m), dtype=torch.int32, device=DEVICE))
    if "values" in options:
        kw.update(pinned_values=rnd(b, m, a, dtype=torch.float32))
    assert flow.denoise_output_euler_takes(x, h, weight)
    expected = reference(x, h, weight, 0.2, **kw)
    out = flow.denoise_output_euler(x, h, weight, dt=0.2, out=torch.empty_like(x), **kw)
    assert torch.allclose(out, expected, rtol=2e-3, atol=2e-3)
    assert flow.denoise_output_euler(x, h, weight, dt=0.2, **kw) is x  # in place by default
    assert torch.equal(x, out)


@sm(110, 120)
@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("options", ["plain", "addends", "mask+override"])
def test_input_embed(extra, heads, options, rnd):
    b, m, a, n = 1, 16, 26, 1024
    f32 = dict(dtype=torch.float32)
    x, weight = rnd(b, m, a, **f32), rnd(heads, n, 2 * a if extra else a, scale=0.2, **f32)
    kw = dict(extra=rnd(b, m, a, **f32)) if extra else {}
    head_index = None
    if heads == 2:
        head_index = kw["head_index"] = (torch.arange(m, device=DEVICE) % 2).to(torch.int32)[None].contiguous()
    inputs = torch.cat((x, kw["extra"]), -1) if extra else x
    w = weight[0] if head_index is None else weight[head_index.long()]
    expected = (inputs[..., None, :] * w).sum(-1)
    if "mask" in options:
        kw["input_mask"] = torch.arange(m, device=DEVICE)[None] >= 3
        expected = expected * kw["input_mask"][..., None]
    if options != "plain":
        kw["addends"] = (rnd(b, m, n, **f32), rnd(b, m, n, **f32))
        expected = expected + kw["addends"][0] + kw["addends"][1]
    if "override" in options:
        kw.update(override=rnd(b, m, n, **f32), override_mask=torch.arange(m, device=DEVICE)[None] < 2)
        expected = torch.where(kw["override_mask"][..., None], kw["override"], expected)
    assert flow.denoise_input_embed_takes(x, weight)
    out = flow.denoise_input_embed(x, weight, **kw)
    assert out.dtype == torch.bfloat16 and torch.allclose(out.float(), expected, rtol=1e-2, atol=1e-2)


def test_input_embed_takes_only_listed_calls(rnd):
    f32 = dict(dtype=torch.float32)
    assert not flow.denoise_input_embed_takes(rnd(1, 16, 26, **f32), rnd(1, 1000, 26, **f32))  # N not a multiple of 16
    assert not flow.denoise_input_embed_takes(rnd(2, 16, 26, **f32), rnd(1, 1024, 26, **f32))  # two requests


def test_takes_only_listed_calls(rnd):
    assert not flow.denoise_output_euler_takes(rnd(1, 16, 26, dtype=torch.float32), rnd(1, 16, 768),
                                             rnd(26, 768, dtype=torch.float32))
