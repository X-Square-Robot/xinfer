"""The normalizations against FP32 torch."""
import pytest
import torch
import torch.nn.functional as F

from xinfer.ops.backends.cuda import norm
from conftest import DEVICE, close

BF16 = torch.bfloat16


def reference(x, weight, eps, zero_centered=False):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (weight.float() + zero_centered)


@pytest.mark.parametrize("shape", [(3, 16), (1, 16, 1024), (2112, 1280), (1, 768, 2048)])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("weight_dtype", [BF16, torch.float32])
def test_rmsnorm(shape, zero_centered, weight_dtype, rnd):
    x, weight = rnd(*shape), rnd(shape[-1], dtype=weight_dtype)
    assert close(norm.rmsnorm(x, weight, 1e-6, zero_centered=zero_centered), reference(x, weight, 1e-6, zero_centered))


@pytest.mark.parametrize("shape", [(3, 16), (1, 16, 1024), (2112, 1280), (1, 768, 2048)])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_add_rmsnorm(shape, zero_centered, rnd):
    residual, x, weight = rnd(*shape), rnd(*shape), rnd(shape[-1])
    total, normed = norm.add_rmsnorm(residual, x, weight, 1e-6, zero_centered=zero_centered)
    assert torch.equal(total, (residual.float() + x.float()).to(BF16))  # one rounding of the FP32 sum
    assert close(normed, reference(total, weight, 1e-6, zero_centered))


@pytest.mark.parametrize("shape", [(3, 16), (868, 1024), (2, 40, 1152)])
def test_layernorm(shape, rnd):
    x, weight, bias = rnd(*shape), rnd(shape[-1]), rnd(shape[-1])
    expected = F.layer_norm(x.float(), (shape[-1],), weight.float(), bias.float(), 1e-6)
    assert close(norm.layernorm(x, weight, bias, 1e-6), expected)


def adarms_reference(x, modulation, eps):
    k = x.shape[-1]
    scale, shift, gate = modulation.float()[:, None].split(k, dim=-1)
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + scale) + shift), gate


def pinned_rows_mask(m, pinned_rows):
    return torch.arange(m, device=DEVICE)[None, :, None] < pinned_rows[:, None, None]


@pytest.mark.parametrize("b,m,k", [(1, 16, 1024), (2, 5, 256)])
@pytest.mark.parametrize("pinned", [False, True])
def test_adarmsnorm(b, m, k, pinned, rnd):
    x, modulation = rnd(b, m, k), rnd(b, 3 * k, dtype=torch.float32, scale=0.3)
    expected, gate = adarms_reference(x, modulation, 1e-6)
    kw = {}
    if pinned:
        kw = dict(pinned_modulation=rnd(b, 3 * k, dtype=torch.float32, scale=0.3),
                  pinned_rows=torch.full((b,), 3, dtype=torch.int32, device=DEVICE))
        other, other_gate = adarms_reference(x, kw["pinned_modulation"], 1e-6)
        mask = pinned_rows_mask(m, kw["pinned_rows"])
        expected, gate = torch.where(mask, other, expected), torch.where(mask, other_gate, gate)
    normed, out_gate = norm.adarmsnorm(x, modulation, 1e-6, **kw)
    assert close(normed, expected)
    assert tuple(out_gate.shape) == (b, m if pinned else 1, k) and torch.equal(out_gate, gate.to(BF16))


@pytest.mark.parametrize("pinned", [False, True])
def test_adarmsnorm_of_a_gated_residual_sum(pinned, rnd):
    b, m, k = 2, 16, 1024
    residual, branch = rnd(b, m, k), rnd(b, m, k)
    branch_gate, modulation = rnd(b, m if pinned else 1, k), rnd(b, 3 * k, dtype=torch.float32, scale=0.3)
    total = residual + (branch.float() * branch_gate.float()).to(BF16)  # the product is rounded, then the sum
    expected, gate = adarms_reference(total, modulation, 1e-6)
    kw = {}
    if pinned:
        kw = dict(pinned_modulation=rnd(b, 3 * k, dtype=torch.float32, scale=0.3),
                  pinned_rows=torch.tensor([3, 0], dtype=torch.int32, device=DEVICE))
        other, other_gate = adarms_reference(total, kw["pinned_modulation"], 1e-6)
        mask = pinned_rows_mask(m, kw["pinned_rows"])
        expected, gate = torch.where(mask, other, expected), torch.where(mask, other_gate, gate)
    normed, out_gate, out_total = norm.adarmsnorm(branch, modulation, 1e-6, residual=residual, branch_gate=branch_gate,
                                                **kw)
    assert torch.equal(out_total, total)
    assert close(normed, expected) and torch.equal(out_gate, gate.to(BF16).expand_as(out_gate))


@pytest.mark.parametrize("k", [1024, 256])  # one kernel at 1024, the two steps elsewhere
@pytest.mark.parametrize("pinned", [False, True])
def test_double_adarmsnorm_is_the_two_steps(k, pinned, rnd):
    b, m = 1, 16
    residual, branch, branch_gate = rnd(b, m, k), rnd(b, m, k), rnd(b, m if pinned else 1, k)
    first, second = (rnd(b, 3 * k, dtype=torch.float32, scale=0.3) for _ in range(2))
    kw1, kw2, kw = {}, {}, {}
    if pinned:
        rows = torch.full((b,), 5, dtype=torch.int32, device=DEVICE)
        p1, p2 = (rnd(b, 3 * k, dtype=torch.float32, scale=0.3) for _ in range(2))
        kw1, kw2 = dict(pinned_modulation=p1, pinned_rows=rows), dict(pinned_modulation=p2, pinned_rows=rows)
        kw = dict(pinned_input_modulation=p1, pinned_post_modulation=p2, pinned_rows=rows)
    normed, gate, total = norm.adarmsnorm(branch, first, 1e-6, residual=residual, branch_gate=branch_gate, **kw1)
    normed, gate, total = norm.adarmsnorm(normed, second, 1e-6, residual=total, branch_gate=gate, **kw2)
    out_gate = norm.double_adarmsnorm_(residual, branch, branch_gate, first, second, 1e-6, **kw)
    assert close(residual, total, 1e-2) and close(branch, normed, 1e-2) and torch.equal(out_gate, gate)


def test_qkv_norm_split(rnd):
    b, m, heads, kv_heads, d = 1, 464, 8, 2, 256
    packed = rnd(b, m, 2 * heads * d + 2 * kv_heads * d)  # [q_0 | gate_0 | q_1 | gate_1 | ... | K | V]
    q_weight, k_weight = rnd(d, scale=0.2), rnd(d, scale=0.2)
    q, k, v, gate = norm.qkv_norm_split(packed, q_weight, k_weight, num_heads=heads, num_kv_heads=kv_heads,
                                      head_dim=d, eps=1e-6)
    query, keys, values = packed.split((2 * heads * d, kv_heads * d, kv_heads * d), dim=-1)
    query = query.view(b, m, heads, 2, d)
    assert close(q, reference(query[..., 0, :], q_weight, 1e-6, True))
    assert close(k, reference(keys.view(b, m, kv_heads, d), k_weight, 1e-6, True))
    assert torch.equal(v, values.view(b, m, kv_heads, d))
    assert torch.equal(gate, query[..., 1, :].reshape(b, m, heads * d))


def test_strided_input(rnd):
    """A view with a stale outer stride (a slice of a longer sequence) is normalized like its dense copy."""
    x, weight = rnd(1, 64, 256), rnd(256)
    view = x[:, 8:40]
    assert torch.equal(norm.rmsnorm(view, weight, 1e-6), norm.rmsnorm(view.contiguous(), weight, 1e-6))
