"""NVFP4 reference format, the nvfp4 / nvfp4_mse linear methods, and the reference GEMM vs torch (GPU)."""

import numpy as np
import pytest
import torch

from xinfer import ops
from xinfer.ops import mxfp8, nvfp4
from xinfer.platforms import resolve_platform

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _scales(s, rows, k):
    """Scale bytes per [row, 16-column block] from the swizzled tensor."""
    off = nvfp4.scale_offsets(rows, k, s.device)[:, : k // 16]
    return s.view(torch.uint8)[off.reshape(-1)].view(rows, k // 16)


def _codes(q):
    """Unpacked E2M1 codes [M, K] of packed uint8 [M, K/2]."""
    return torch.stack((q & 15, q >> 4), -1).reshape(q.shape[0], -1)


def _e4m3_bits(value):
    return int(torch.tensor(value).to(torch.float8_e4m3fn).view(torch.uint8))


def test_e2m1_rounds_to_nearest_even_and_saturates():
    values = [0.0, 0.2, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 2.6, 3.5, 4.9, 5.0, 5.1, 7.0, 1e9, -0.0, -0.1, -2.5, -7.0]
    expect = [0, 0, 0, 1, 2, 2, 4, 4, 5, 6, 6, 6, 7, 7, 7, 8, 8, 12, 15]
    codes = nvfp4.e2m1_codes(torch.tensor(values))
    assert codes.tolist() == expect
    assert nvfp4.e2m1_values(torch.arange(16)).tolist() == list(E2M1) + [-v for v in E2M1]


def test_scale_layout_matches_the_swizzle_formula():
    off = nvfp4.scale_offsets(256, 1280)
    assert off.shape == (256, 80)
    assert off[0, 0] == 0 and off[0, 3] == 3 and off[0, 4] == 512 and off[1, 0] == 16 and off[32, 0] == 4
    assert off[128, 0] == 20 * 512 and len(set(off.reshape(-1).tolist())) == off.numel()
    assert nvfp4.scale_elements(37, 48) == 128 * 4 and nvfp4.scale_elements(256, 1280) == 256 * 80


def test_packing_scales_and_padding():
    # one row, two blocks: amax 6 gives scale 1, amax 0 gives the floor scale 2^-6
    x = torch.zeros(1, 32)
    x[0, :16] = torch.tensor([0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6, 0.25, -0.1])
    q, s = nvfp4.quantize(x.bfloat16())
    assert q.dtype == torch.uint8 and q.shape == (1, 16) and s.dtype == torch.float8_e4m3fn
    assert q[0, :8].tolist() == [0x21, 0x43, 0x65, 0x97, 0xBA, 0xDC, 0xFE, 0x80] and q[0, 8:].tolist() == [0] * 8
    assert _scales(s, 1, 32).tolist() == [[_e4m3_bits(1.0), _e4m3_bits(2.0 ** -6)]]
    assert int(s.view(torch.uint8).count_nonzero()) == 2  # padding rows and blocks are zero
    assert torch.equal(nvfp4.dequantize(q, s)[0, :14], x[0, :14])


def test_amax_scale_rounding_and_clamping():
    x = torch.zeros(4, 16)
    x[0, 0] = 6.375  # amax / 6 = 1.0625, the tie between E4M3 1.0 and 1.125: to even (1.0), element saturates
    x[1, 0] = 6.4375  # just above the tie: 1.125, element 5.72 → 6
    x[2, 0] = 1e6  # clamped to 448
    x[3, 0] = 0.05  # below 6 · 2^-6: floor scale 2^-6, element 3.2 → 3
    q, s = nvfp4.quantize(x.bfloat16())
    assert _scales(s, 4, 16)[:, 0].tolist() == [_e4m3_bits(v) for v in (1.0, 1.125, 448.0, 2.0 ** -6)]
    assert _codes(q)[:, 0].tolist() == [7, 7, 7, 5]


def test_quantize_roundtrip_error_bound():
    torch.manual_seed(0)
    x = (torch.randn(37, 384) * torch.logspace(-3, 3, 384)).bfloat16()
    q, s = nvfp4.quantize(x)
    back = nvfp4.dequantize(q, s)
    scale = _scales(s, 37, 384).view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    # half the widest E2M1 step (4 → 6) is one scale unit; saturation after the scale rounds down is less
    assert bool(((back - x.float()).abs() <= scale).all())
    assert float((back - x.float()).norm() / x.float().norm()) < 0.12


def _mse_scale_numpy(block, g):
    """Scalar FP32 transliteration of the kernel's MSE sweep for one block of 16 values."""
    f32 = np.float32
    grid = np.array(E2M1, dtype=np.float32)

    def round_e2m1(v):
        a = min(abs(v), f32(6))
        i = int(np.argmin(np.abs(grid - a)))
        if i + 1 < len(grid) and grid[i + 1] - a == a - grid[i]:  # tie: the even code
            i = i if i % 2 == 0 else i + 1
        return f32(np.copysign(grid[i], v))

    best, best_code = f32(np.finfo(np.float32).max), 1
    for code in range(1, 127):
        decode = f32(torch.tensor([code], dtype=torch.uint8).view(torch.float8_e4m3fn).item()) * f32(g)
        errors = [f32(v - f32(round_e2m1(f32(v / decode)) * decode)) for v in block]
        errors = [f32(e * e) for e in errors]
        while len(errors) > 1:
            half = len(errors) // 2
            errors = [f32(errors[i] + errors[i + half]) for i in range(half)]
        if errors[0] < best:
            best, best_code = errors[0], code
    return best_code


def test_mse_scales_match_a_scalar_transliteration():
    torch.manual_seed(1)
    w = (torch.randn(3, 64) * 0.05).bfloat16()
    w[2, 16:32] = 0  # all-zero block: every error is 0, the first candidate (code 1) wins
    g = nvfp4.global_scale(w)
    q, s = nvfp4.quantize(w, global_scale=g, mse=True)
    got = _scales(s, 3, 64)
    blocks = w.float().numpy().reshape(3, 4, 16)
    expect = [[_mse_scale_numpy(blocks[r, b], g) for b in range(4)] for r in range(3)]
    assert got.tolist() == expect and expect[2][1] == 1
    # elements are x / (scale · g), as for amax scales
    scale = got.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    assert torch.equal(_codes(q), nvfp4.e2m1_codes(w.float() / (scale * torch.tensor(g))))


def test_mse_scales_never_reconstruct_worse_than_amax_scales():
    torch.manual_seed(2)
    w = (torch.randn(16, 256) * torch.logspace(-2, 1, 256)).bfloat16()
    g = nvfp4.global_scale(w)
    mse = nvfp4.dequantize(*nvfp4.quantize(w, global_scale=g, mse=True)) * g
    amax = nvfp4.dequantize(*nvfp4.quantize(w, global_scale=g)) * g
    err = lambda back: ((back - w.float()) ** 2).view(16, 16, 16).sum(-1)  # noqa: E731
    assert bool((err(mse) <= err(amax) * (1 + 1e-6)).all()) and float(err(mse).sum()) < 0.9 * float(err(amax).sum())


def test_global_scale():
    w = torch.zeros(4, 16, dtype=torch.bfloat16)
    assert nvfp4.global_scale(w) == 1.0
    w[1, 3] = -1344.0  # amax / 2688 = 0.5
    assert nvfp4.global_scale(w) == 0.5
    w[1, 3] = 1.0
    assert nvfp4.global_scale(w) == float(np.float32(1.0 / 2688.0))


def test_rmsnorm_quantize_rounds_the_normalized_value_to_bf16():
    torch.manual_seed(3)
    x, w = (torch.randn(9, 2048) * 3).bfloat16(), 1 + 0.1 * torch.randn(2048)
    q, s = nvfp4.rmsnorm_quantize(x, w, 1e-6)
    x32 = x.float()
    total = mxfp8._sum_of_squares(x32, 256)
    normalized = (x32 * (torch.rsqrt(total / 2048 + 1e-6)[:, None] * w.bfloat16().float())).bfloat16()
    rq, rs = nvfp4.quantize(normalized)
    assert torch.equal(q, rq) and torch.equal(s.view(torch.uint8), rs.view(torch.uint8))
    # the FP32 weight is used as BF16; summing in another order changes at most rare codes
    assert torch.equal(nvfp4.rmsnorm_quantize(x, w.bfloat16(), 1e-6)[0], q)
    exact = (x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-6) * w.bfloat16().double())
    eq = nvfp4.quantize(exact.bfloat16())[0]
    assert float((_codes(eq) != _codes(q)).float().mean()) < 1e-3


def test_silu_mul_quantize_skips_the_bf16_rounding():
    torch.manual_seed(4)
    gu = (torch.randn(6, 512) * 2).bfloat16()
    q, s = nvfp4.silu_mul_quantize(gu)
    g, u = gu.float().chunk(2, 1)
    rq, rs = nvfp4.quantize_values((g / (1 + torch.exp(-g))) * u)
    assert torch.equal(q, rq) and torch.equal(s.view(torch.uint8), rs.view(torch.uint8))
    assert not torch.equal(q, nvfp4.quantize(torch.nn.functional.silu(g.bfloat16()) * u.bfloat16())[0])


def test_gemm_applies_alpha_and_bias_to_the_fp32_accumulator():
    torch.manual_seed(5)
    grid = torch.tensor(E2M1 + tuple(-v for v in E2M1))
    a = grid[torch.randint(0, 16, (5, 64))]
    w = grid[torch.randint(0, 16, (32, 64))] * 4  # every block's amax 6 (24): scales 1 and 4, exact values
    a[:, ::16], w[:, ::16] = 6.0, 24.0
    aq, as_ = nvfp4.quantize(a.bfloat16())
    wq, ws = nvfp4.quantize(w.bfloat16())
    assert torch.equal(nvfp4.dequantize(aq, as_), a) and torch.equal(nvfp4.dequantize(wq, ws), w)
    bias = torch.randn(32).bfloat16()
    alpha = float(np.float32(1 / 3))
    acc = a.double() @ w.double().T  # exact: small integers times powers of two
    expect = (acc.float() * alpha + bias.float()).bfloat16()
    assert torch.equal(nvfp4.gemm(aq, as_, wq, ws, bias, alpha=alpha), expect)
    assert torch.equal(nvfp4.gemm(aq, as_, wq, ws), acc.float().bfloat16())


# ------------------------------------------------------------ methods (CPU)
def _table():
    return ops.registry.build_table(resolve_platform("cpu"))


def _load(module, gen):
    for name, prm in list(module.named_parameters()):
        owner = module.get_submodule(name.rpartition(".")[0]) if "." in name else module
        leaf = name.rpartition(".")[2]
        val = (1 + 0.1 * torch.randn(prm.shape, generator=gen)) if leaf == "weight" and prm.ndim == 1 else \
            torch.randn(prm.shape, generator=gen) * (prm.shape[-1] ** -0.5 if prm.ndim == 2 else 0.1)
        owner._parameters[leaf] = torch.nn.Parameter(val.bfloat16(), requires_grad=False)


@pytest.mark.parametrize("method", ["nvfp4", "nvfp4_mse"])
def test_linear_forms_compose_the_reference_operators(method):
    from xinfer.layers.basic import Linear, RMSNorm
    from xinfer.quantization import QuantConfig, process_weights, use_quant

    with ops.use_ops(_table()), use_quant(QuantConfig(groups={"g": method})):
        layer, norm = Linear(128, 64, bias=True, group="g"), RMSNorm(128, 1e-6)
        down = Linear(64, 128, group="g")
        plain = Linear(128, 64, bias=True)
    gen = torch.Generator().manual_seed(6)
    for module in (layer, norm, down):
        _load(module, gen)
    plain._parameters["weight"], plain._parameters["bias"] = layer.weight, layer.bias
    w, b = layer.weight.clone(), layer.bias.clone()
    assert process_weights(layer) == {method: 1} and process_weights(down) == {method: 1}
    assert not hasattr(layer, "weight") and layer.weight_q.shape == (64, 64)
    mse = method == "nvfp4_mse"
    g = nvfp4.global_scale(w) if mse else 1.0
    assert layer.weight_global_scale == g
    wq, ws = nvfp4.quantize(w, global_scale=g, mse=mse)
    assert torch.equal(layer.weight_q, wq) and torch.equal(layer.weight_scales.view(torch.uint8), ws.view(torch.uint8))

    def gemm(quantized):
        if mse:  # global scale in the epilogue, bias added to the BF16 output
            return nvfp4.gemm(*quantized, wq, ws, alpha=g) + b
        return nvfp4.gemm(*quantized, wq, ws, b)

    x = torch.randn(3, 5, 128).bfloat16()
    flat = x.reshape(15, 128)
    assert torch.equal(layer(x), gemm(nvfp4.quantize(flat)).reshape(3, 5, 64))
    assert torch.equal(layer(x, norm=norm), gemm(nvfp4.rmsnorm_quantize(flat, norm.weight, 1e-6)).reshape(3, 5, 64))
    residual = torch.randn(3, 5, 64).bfloat16()
    assert torch.equal(layer(x, residual=residual), residual + layer(x))
    gu = torch.randn(15, 128).bfloat16()
    assert torch.equal(down(gu, input_act="silu_and_mul"),
                       nvfp4.gemm(*nvfp4.silu_mul_quantize(gu), down.weight_q, down.weight_scales,
                                  alpha=down.weight_global_scale))
    ref = plain(x).float()
    assert float((layer(x).float() - ref).norm() / ref.norm()) < 0.15


def test_qkv_and_gated_layers_run_the_plain_composition():
    from xinfer.layers.basic import GatedMLP, QKVLinear, RMSNorm
    from xinfer.layers.rotary import MRope
    from xinfer.quantization import QuantConfig, process_weights, use_quant

    with ops.use_ops(_table()), use_quant(QuantConfig(groups={"g": "nvfp4", "e": "nvfp4_mse"})):
        norm = RMSNorm(128, 1e-6)
        qkv = QKVLinear(128, 4, 2, 16, bias=True, group="e")
        mlp_gate = GatedMLP(128, 64, group="g", act_fused_into="gate_up")
        mlp_down = GatedMLP(128, 64, group="e", act_fused_into="down")
        rope_layer = MRope([2, 3, 3])
    gen = torch.Generator().manual_seed(7)
    for module in (norm, qkv, mlp_gate, mlp_down):
        _load(module, gen)
        process_weights(module)
    x = torch.randn(1, 5, 128).bfloat16()
    angles = torch.rand(3, 1, 5, 8) * 30
    emb = torch.cat((angles, angles), -1)
    rope = rope_layer.tables(emb.cos(), emb.sin(), torch.bfloat16)
    q, k, v = qkv(x, rope, norm=norm)
    eq, ek, ev = qkv.quant.apply_fused(qkv, x, norm=norm).split([64, 32, 32], -1)
    assert torch.equal(q, rope.rotate(eq.reshape(1, 5, 4, 16))) and torch.equal(k, rope.rotate(ek.reshape(1, 5, 2, 16)))
    assert torch.equal(v, ev.reshape(1, 5, 2, 16))
    gate_up = mlp_gate.gate_up.quant.apply_fused(mlp_gate.gate_up, x, norm=norm)
    hidden = torch.nn.functional.silu(gate_up[..., :64]) * gate_up[..., 64:]
    assert torch.equal(mlp_gate(x, norm=norm, residual=x), x + mlp_gate.down(hidden))
    gate_up = mlp_down.gate_up(x, norm=norm)
    assert torch.equal(mlp_down(x, norm=norm, residual=x), x + mlp_down.down(gate_up, input_act="silu_and_mul"))


def test_widths_must_be_multiples_of_16():
    from xinfer.layers.basic import Linear
    from xinfer.quantization import QuantConfig, use_quant

    for method in ("nvfp4", "nvfp4_mse"):
        with ops.use_ops(_table()), use_quant(QuantConfig(groups={"g": method})):
            Linear(4304, 1152, group="g")
            for k, n in ((1176, 64), (128, 72)):
                with pytest.raises(ValueError):
                    Linear(k, n, group="g")


# ------------------------------------------------------------ GPU
def _scaled_mm_nvfp4():
    if not (torch.cuda.is_available() and hasattr(torch, "float4_e2m1fn_x2")):
        return False
    return torch.cuda.get_device_capability() >= (10, 0)


gpu = pytest.mark.skipif(not _scaled_mm_nvfp4(), reason="CUDA sm100+ and torch float4_e2m1fn_x2 required")


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("rows,n,k", [(2112, 1280, 1280), (2112, 1280, 3456), (528, 2048, 5120), (1, 3840, 1280)])
@pytest.mark.parametrize("with_bias", [True, False])
def test_reference_gemm_matches_torch_scaled_mm(rows, n, k, with_bias):
    """``torch._scaled_mm`` reads packed E2M1 with swizzled E4M3 block-16 scales, as the Wall-X service called it."""
    torch.manual_seed(rows + n + k)
    a = torch.randn(rows, k, device="cuda").bfloat16()
    w = (torch.randn(n, k, device="cuda") * k ** -0.5).bfloat16()
    bias = torch.randn(n, device="cuda").bfloat16() if with_bias else None
    aq, as_ = nvfp4.quantize(a)
    wq, ws = nvfp4.quantize(w)
    fp4 = torch.float4_e2m1fn_x2
    out = torch._scaled_mm(aq.view(fp4), wq.view(fp4).t(), as_, ws, bias=bias, out_dtype=torch.bfloat16)
    ref = nvfp4.gemm(aq, as_, wq, ws, bias)
    err = (out.float() - ref.float()).abs()
    # the same exact products summed in another order: BF16 rounding of nearly the same FP32 value
    assert float((out == ref).float().mean()) > 0.99
    assert float(err.max()) <= 2 * float(ref.float().abs().max()) * 2 ** -8


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reference_quantizers_give_the_same_bits_on_cuda():
    torch.manual_seed(8)
    x = (torch.randn(300, 1280) * torch.logspace(-3, 3, 1280)).bfloat16()
    w = (torch.randn(64, 2048) * 0.02).bfloat16()
    g = nvfp4.global_scale(w)
    for tensor, kwargs in ((x, {}), (w, dict(global_scale=g, mse=True))):
        (q, s), (rq, rs) = nvfp4.quantize(tensor.cuda(), **kwargs), nvfp4.quantize(tensor, **kwargs)
        assert torch.equal(q.cpu(), rq) and torch.equal(s.view(torch.uint8).cpu(), rs.view(torch.uint8))
