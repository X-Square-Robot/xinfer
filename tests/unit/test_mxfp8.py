"""MXFP8 reference format and fused forms, and the MXFP8 linear method on the reference operators."""

import pytest
import torch

from xinfer import ops
from xinfer.ops.reference import mxfp8
from xinfer.ops.layout import mxfp8 as layout
from xinfer.platforms import resolve_platform


def test_scale_layout_matches_the_swizzle_formula():
    off = layout.scale_offsets(256, 1280)
    assert off.shape == (256, 40)
    assert off[0, 0] == 0 and off[0, 1] == 1 and off[0, 4] == 512
    assert off[1, 0] == 16 and off[32, 0] == 4 and off[128, 0] == 10 * 512
    assert len(set(off.reshape(-1).tolist())) == off.numel()
    assert int(off.max()) < layout.scale_elements(256, 1280)


@pytest.mark.parametrize("share", [True, False])
def test_quantize_roundtrip_error_and_scales(share):
    torch.manual_seed(0)
    x = (torch.randn(37, 384) * torch.logspace(-3, 3, 384)).bfloat16()
    q, s = mxfp8.quantize(x, share_scale_128=share)
    assert q.dtype == torch.float8_e4m3fn and s.numel() == layout.scale_elements(37, 384)
    back = mxfp8.dequantize(q, s)
    rel = ((back - x.float()).abs() / x.float().abs().clamp_min(1e-30))
    # 3 mantissa bits: 2^-4 relative error, unless the value underflows its block's scale
    big = x.float().abs() > x.float().abs().amax(-1, keepdim=True) * 2 ** -6
    assert float(rel[big].max()) <= 2 ** -4 + 1e-7
    e = s[layout.scale_offsets(37, 384)[:, :12].reshape(-1)].view(37, 12)
    if share:
        assert torch.equal(e.view(37, 3, 4), e.view(37, 3, 1, 4)[:, :, :, :1].expand(37, 3, 1, 4).reshape(37, 3, 4))


def test_zero_rows_and_exact_powers():
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    x[1, 5] = 448.0
    q, s = mxfp8.quantize(x)
    assert torch.equal(mxfp8.dequantize(q, s), x.float())


def test_gemm_reference_close_to_bf16():
    torch.manual_seed(1)
    a, w, b = torch.randn(16, 256).bfloat16(), (torch.randn(64, 256) * 0.05).bfloat16(), torch.randn(64).bfloat16()
    out = mxfp8.gemm(*mxfp8.quantize(a), *mxfp8.quantize(w), b)
    ref = (a.float() @ w.float().T + b.float())
    assert float((out.float() - ref).norm() / ref.norm()) < 0.04


def test_mxfp8_linear_with_reference_ops():
    from xinfer.models.layers.basic import Linear
    from xinfer.models.layers.quantization import QuantConfig, process_weights, use_quant

    table = ops.registry.build_table(resolve_platform("cpu"))
    torch.manual_seed(2)
    with ops.use_ops(table), use_quant(QuantConfig(groups={"g": "mxfp8"})):
        layer = Linear(128, 64, bias=True, group="g")
        plain = Linear(128, 64, bias=True, group="other")
    w, b = (torch.randn(64, 128) * 0.1).bfloat16(), torch.randn(64).bfloat16()
    for lin in (layer, plain):
        lin._parameters["weight"] = torch.nn.Parameter(w.clone(), requires_grad=False)
        lin._parameters["bias"] = torch.nn.Parameter(b.clone(), requires_grad=False)
    assert process_weights(layer) == {"mxfp8": 1} and not hasattr(layer, "weight")
    x = torch.randn(3, 5, 128).bfloat16()
    y, ref = layer(x), plain(x)
    assert y.shape == (3, 5, 64)
    assert float((y.float() - ref.float()).norm() / ref.float().norm()) < 0.04
    with ops.use_ops(table), use_quant(QuantConfig(groups={"g": "mxfp8"})), pytest.raises(ValueError):
        Linear(1176, 64, group="g")  # K must be a multiple of 32


# ------------------------------------------------------------ fused forms (CPU)
def _rotate(x, cos, sin):
    """Standard-layout rotary (rotate_half) on FP32 ``[M, heads, D]`` with half tables ``[M, D/2]``."""
    c, s = torch.cat((cos, cos), -1)[:, None], torch.cat((sin, sin), -1)[:, None]
    half = x.shape[-1] // 2
    return x * c + torch.cat((-x[..., half:], x[..., :half]), -1) * s


def test_rope_table_layout_and_roundtrip():
    cos, sin = torch.rand(5, 64), torch.rand(5, 64)
    table = layout.rope_table(cos, sin)
    assert table.shape == (32, 5, 4) and table.is_contiguous()  # [D/4, M, 4], D = 128
    assert table[3, 2].tolist() == [cos[2, 6].item(), sin[2, 6].item(), cos[2, 7].item(), sin[2, 7].item()]
    c, s = layout.rope_table_cos_sin(table)
    assert torch.equal(c, cos) and torch.equal(s, sin)


def test_row_orders():
    perm = layout.rope_pair_permutation(q_dim=8, k_dim=8, head_dim=8, n=24)
    assert perm[:8].tolist() == [0, 4, 1, 5, 2, 6, 3, 7] and perm[16:].tolist() == list(range(16, 24))
    assert layout.gate_up_interleave(3).tolist() == [0, 3, 1, 4, 2, 5]


def test_gemm_rope_reference_rotates_the_standard_projection():
    torch.manual_seed(3)
    heads, kv, hd, k = 4, 2, 16, 128
    n = (heads + 2 * kv) * hd
    a, w = torch.randn(7, k).bfloat16(), (torch.randn(n, k) * k ** -0.5).bfloat16()
    bias = torch.randn(n).bfloat16()
    perm = layout.rope_pair_permutation(heads * hd, kv * hd, hd, n)
    aq, as_ = mxfp8.quantize(a)
    wq, ws = mxfp8.quantize(w)
    pq, ps = mxfp8.quantize(w[perm].contiguous())
    assert torch.equal(pq.view(torch.uint8), wq.view(torch.uint8)[perm])  # quantization is per row
    angles = torch.rand(7, hd // 2) * 50
    out = mxfp8.gemm_rope(aq, as_, pq, ps, bias[perm], layout.rope_table(angles.cos(), angles.sin()),
                          heads * hd, kv * hd, hd)
    acc = mxfp8.dequantize(aq, as_) @ mxfp8.dequantize(wq, ws).T + bias.float()
    rot = (heads + kv) * hd
    qk = _rotate(acc[:, :rot].view(7, heads + kv, hd), angles.cos(), angles.sin()).reshape(7, rot)
    expect = torch.cat((qk, acc[:, rot:]), 1)
    torch.testing.assert_close(out.float(), expect, rtol=2 ** -8, atol=1e-5)


def test_gemm_silu_mul_and_residual_references():
    torch.manual_seed(4)
    a, w = torch.randn(5, 256).bfloat16(), (torch.randn(96, 256) * 0.06).bfloat16()
    aq, as_ = mxfp8.quantize(a)
    wq, ws = mxfp8.quantize(w)
    acc = mxfp8.dequantize(aq, as_) @ mxfp8.dequantize(wq, ws).T
    iq, is_ = mxfp8.quantize(w[layout.gate_up_interleave(48)].contiguous())
    gated = mxfp8.gemm_silu_mul(aq, as_, iq, is_)
    torch.testing.assert_close(gated.float(), torch.nn.functional.silu(acc[:, :48]) * acc[:, 48:],
                               rtol=2 ** -8, atol=1e-5)
    residual = torch.randn(5, 96).bfloat16()
    out = mxfp8.gemm_residual(aq, as_, wq, ws, residual)
    torch.testing.assert_close(out.float(), acc + residual.float(), rtol=2 ** -8, atol=1e-5)


def test_fused_quantizers_skip_the_bf16_rounding():
    torch.manual_seed(5)
    x, w = (torch.randn(9, 512) * 3).bfloat16(), (1 + 0.1 * torch.randn(512)).bfloat16()
    q, s = mxfp8.rmsnorm_quantize(x, w, 1e-6)
    x32 = x.double()
    exact = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6) * w.double()).float()
    rq, rs = mxfp8.quantize_values(exact)
    # same values up to FP32 sum order: at most a rare E4M3 rounding flip
    assert torch.equal(s, rs)
    assert float((q.view(torch.uint8) != rq.view(torch.uint8)).float().mean()) < 1e-3
    gu = (torch.randn(6, 256) * 2).bfloat16()
    q, s = mxfp8.silu_mul_quantize(gu)
    g, u = gu.double().chunk(2, 1)
    rq, rs = mxfp8.quantize_values((g / (1 + torch.exp(-g)) * u).float())
    assert torch.equal(s, rs)
    assert float((q.view(torch.uint8) != rq.view(torch.uint8)).float().mean()) < 1e-3
    assert not torch.equal(q.view(torch.uint8), mxfp8.quantize(torch.nn.functional.silu(g.bfloat16()) *
                                                                u.bfloat16())[0].view(torch.uint8))


def _layers(method, *, seed=6, hidden=128, heads=4, kv=2, hd=16, inter=64):
    from xinfer.models.layers.basic import GatedMLP, Linear, QKVLinear, RMSNorm
    from xinfer.models.layers.quantization import QuantConfig, process_weights, use_quant

    torch.manual_seed(seed)
    table = ops.registry.build_table(resolve_platform("cpu"))
    with ops.use_ops(table), use_quant(QuantConfig(groups={"g": method})):
        mods = dict(norm=RMSNorm(hidden, 1e-6), qkv=QKVLinear(hidden, heads, kv, hd, bias=True, group="g"),
                    o=Linear(heads * hd, hidden, group="g"),
                    mlp_gate=GatedMLP(hidden, inter, group="g", act_fused_into="gate_up"),
                    mlp_down=GatedMLP(hidden, inter, group="g", act_fused_into="down"))
    gen = torch.Generator().manual_seed(seed)
    for m in mods.values():
        for name, prm in list(m.named_parameters()):
            owner = m.get_submodule(name.rpartition(".")[0]) if "." in name else m
            leaf = name.rpartition(".")[2]
            val = (1 + 0.1 * torch.randn(prm.shape, generator=gen)) if leaf == "weight" and prm.ndim == 1 else \
                torch.randn(prm.shape, generator=gen) * (prm.shape[-1] ** -0.5 if prm.ndim == 2 else 0.1)
            owner._parameters[leaf] = torch.nn.Parameter(val.bfloat16(), requires_grad=False)
        process_weights(m)
    return mods


def _rope(rows, hd=16, seed=7):
    from xinfer.models.layers.rotary import MRope

    torch.manual_seed(seed)
    angles = torch.rand(3, 1, rows, hd // 2) * 30
    emb = torch.cat((angles, angles), -1)
    with ops.use_ops(ops.registry.build_table(resolve_platform("cpu"))):
        return MRope([2, 3, 3]).tables(emb.cos(), emb.sin(), torch.bfloat16)


def test_bf16_fusable_forms_compose_the_eager_layer():
    m = _layers("bf16")
    x = torch.randn(1, 5, 128).bfloat16()
    rope = _rope(5)
    q, k, v = m["qkv"](x, rope, norm=m["norm"])
    full = torch.nn.functional.linear(m["norm"](x), m["qkv"].weight, m["qkv"].bias)
    eq, ek, ev = full.split([64, 32, 32], -1)
    assert torch.equal(q, rope.rotate(eq.reshape(1, 5, 4, 16)))
    assert torch.equal(k, rope.rotate(ek.reshape(1, 5, 2, 16))) and torch.equal(v, ev.reshape(1, 5, 2, 16))
    o_in = torch.randn(1, 5, 64).bfloat16()
    assert torch.equal(m["o"](o_in, residual=x), x + m["o"](o_in))
    for key in ("mlp_gate", "mlp_down"):
        mlp = m[key]
        gu = torch.nn.functional.linear(m["norm"](x), mlp.gate_up.weight)
        expect = x + torch.nn.functional.linear(torch.nn.functional.silu(gu[..., :64]) * gu[..., 64:], mlp.down.weight)
        assert torch.equal(mlp(x, norm=m["norm"], residual=x), expect), key


def test_mxfp8_fused_forms_track_bf16_and_refuse_the_plain_form():
    ref, fused = _layers("bf16"), _layers("mxfp8")
    x = torch.randn(1, 5, 128).bfloat16()
    rope = _rope(5)

    def close(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm())

    for got, want in zip(fused["qkv"](x, rope, norm=fused["norm"]), ref["qkv"](x, rope, norm=ref["norm"])):
        assert close(got, want) < 0.06
    o_in = torch.randn(1, 5, 64).bfloat16()
    assert close(fused["o"](o_in, residual=x), ref["o"](o_in, residual=x)) < 0.06
    for key in ("mlp_gate", "mlp_down"):
        assert close(fused[key](x, norm=fused["norm"], residual=x), ref[key](x, norm=ref["norm"], residual=x)) < 0.06
    assert fused["mlp_gate"].gate_up.row_order is not None and fused["mlp_down"].gate_up.row_order is None
    with pytest.raises(RuntimeError):  # rows reordered for the rope epilogue
        fused["qkv"].quant.apply(fused["qkv"], x)


def _pair(*shape, scale=1.0, seed=0):
    """A BF16 tensor and its MXFP8 (1x128) pair."""
    x = (torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale).bfloat16()
    return x, mxfp8.quantize(x)


def test_gemm_family_gemm_epilogue_forms():
    _, x = _pair(24, 256, seed=3)
    _, w = _pair(128, 256, scale=0.05, seed=4)
    bias, residual = torch.randn(128).bfloat16(), torch.randn(24, 128).bfloat16()
    plain = mxfp8.gemm(*x, *w, bias)
    assert torch.equal(mxfp8.gemm_epilogue(x, w, bias), plain)
    # the residual is added to the rounded product (two roundings)
    two_roundings = (plain.float() + residual.float()).bfloat16()
    assert torch.equal(mxfp8.gemm_epilogue(x, w, bias, residual=residual), two_roundings)
    keep = torch.tensor([1.0] * 20 + [0.0] * 4)
    masked = mxfp8.gemm_epilogue(x, w, row_mask=keep)
    assert torch.equal(masked[:20], mxfp8.gemm(*x, *w)[:20]) and not masked[20:].any()
    acc = mxfp8.dequantize(*x) @ mxfp8.dequantize(*w).T + bias.float()
    q, s = mxfp8.gemm_epilogue(x, w, bias, activation="gelu_tanh", out_quant=True)
    expected = mxfp8.quantize_values(torch.nn.functional.gelu(acc, approximate="tanh"))
    assert torch.equal(q.view(torch.uint8), expected[0].view(torch.uint8)) and torch.equal(s, expected[1])


def test_gemm_family_glu_and_qkv_rope_forms():
    from xinfer.ops import reference

    _, x = _pair(16, 128, seed=5)
    _, w = _pair(256, 128, scale=0.05, seed=6)
    acc = mxfp8.dequantize(*x) @ mxfp8.dequantize(*w).T
    q, s = mxfp8.gemm_glu_quantize(x, w)
    expected = mxfp8.quantize_values(torch.nn.functional.silu(acc[:, :128]) * acc[:, 128:])
    assert torch.equal(q.view(torch.uint8), expected[0].view(torch.uint8)) and torch.equal(s, expected[1])
    _, wqkv = _pair(3 * 2 * 32, 128, scale=0.05, seed=7)
    bias = torch.randn(192).bfloat16()
    angles = torch.rand(16, 16) * 6
    out = mxfp8.qkv_rope_epilogue(x, wqkv, bias, angles.cos(), angles.sin(), 2, 32)
    want = reference.qkv_rope_epilogue(mxfp8.dequantize(*x), mxfp8.dequantize(*wqkv), bias, angles.cos(),
                                       angles.sin(), 2, 32)
    assert torch.equal(out, want.bfloat16())


def test_gemm_family_norms_quantize_the_bf16_norm():
    from xinfer.ops import reference

    x = (torch.randn(2, 5, 256, generator=torch.Generator().manual_seed(8)) * 2).bfloat16()
    residual = torch.randn(2, 5, 256).bfloat16()
    weight, bias = torch.randn(256) * 0.1, torch.randn(256).bfloat16()
    for got, want in ((mxfp8.layer_norm_quantize(x, weight.bfloat16(), bias, 1e-6),
                       reference.layer_norm_cast(x, weight.bfloat16(), bias, 1e-6)),
                      (mxfp8.zero_centered_rms_norm_quantize(x, weight, 1e-6),
                       reference.zero_centered_rms_norm(x, weight, 1e-6))):
        expected = mxfp8.quantize(want.reshape(10, 256))
        assert torch.equal(got[0].view(torch.uint8), expected[0].view(torch.uint8))
        assert torch.equal(got[1], expected[1])
    summed, (q, s) = mxfp8.add_zero_centered_rms_norm_quantize(x, residual, weight, 1e-6)
    s32 = x.float() + residual.float()
    assert torch.equal(summed, s32.bfloat16())
    expected = mxfp8.quantize(reference.zero_centered_rms_norm(s32, weight, 1e-6).bfloat16().reshape(10, 256))
    assert torch.equal(q.view(torch.uint8), expected[0].view(torch.uint8)) and torch.equal(s, expected[1])


def test_mxfp8_packings_go_with_their_weight(monkeypatch):
    """An MXFP8 weight is packed once per operator; its entries go when its values or scales are freed, so a weight
    allocated later at the same address gets its own packing."""
    import gc
    from types import SimpleNamespace

    from xinfer.ops.backends.cuda import gemm

    packed = []
    monkeypatch.setattr(gemm, "xc", SimpleNamespace(prepack_weight=lambda q, op, scale: packed.append(op) or op))
    monkeypatch.setattr(gemm, "_packed", {})
    q, scales = mxfp8.quantize(torch.ones(32, 128, dtype=torch.bfloat16))
    assert gemm._mxfp8_weight("gemm", (q, scales)) == "gemm" == gemm._mxfp8_weight("gemm", (q, scales))
    assert gemm._mxfp8_weight("gemm_glu", (q, scales)) == "gemm_glu" and packed == ["gemm", "gemm_glu"]
    del q
    gc.collect()
    assert gemm._packed == {}


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_xcompute_gemm_family_forms_match_the_references_at_wallx2b_shapes():
    """The CUDA operators (xcompute's MXFP8 GEMM family) against the references at Wall-X 2B's shapes: ViT rows
    1568 (hidden 1024, 16 heads of 64, MLP 4096) and a prompt of 640 (hidden 2048, MLP 6144, DeltaNet 8224)."""
    table = ops.registry.build_table(resolve_platform("cuda"))
    if table.selection("mxfp8_gemm_epilogue").library != "xcompute":
        pytest.skip("needs an xcompute build with the MXFP8 GEMM family")
    g = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).bfloat16()

    def relative(got, want):
        got, want = (mxfp8.dequantize(*v) if isinstance(v, tuple) else v.float() for v in (got, want))
        return float((got - want).norm() / want.norm())

    def check(name, *args, bound, **kwargs):
        got = getattr(table, name)(*args, **kwargs)
        want = ops.registry.spec(name).reference(*args, **kwargs)
        if name.startswith("mxfp8_add_"):
            assert torch.equal(got[0], want[0]), name
            got, want = got[1], want[1]
        assert relative(got, want) < bound, name

    vit, prompt = randn(1568, 1024, scale=3), randn(1, 640, 2048, scale=2)
    check("mxfp8_quantize", vit, bound=1e-6)
    check("mxfp8_layer_norm_quantize", vit, randn(1024, scale=0.2), randn(1024, scale=0.1), 1e-6, bound=0.01)
    norm = torch.randn(2048, device="cuda", generator=g) * 0.1
    check("mxfp8_zero_centered_rms_norm_quantize", prompt, norm, 1e-6, bound=0.01)
    check("mxfp8_add_zero_centered_rms_norm_quantize", randn(1, 640, 2048), prompt, norm, 1e-6, bound=0.01)
    x1024, x2048, x6144 = (mxfp8.quantize(randn(m, k)) for m, k in ((1568, 1024), (640, 2048), (640, 6144)))
    def weight(n, k):
        return mxfp8.quantize(randn(n, k, scale=0.03))

    angles = torch.rand(1568, 32, device="cuda", generator=g) * 6
    check("mxfp8_qkv_rope_epilogue", x1024, weight(3072, 1024), randn(3072), angles.cos(), angles.sin(), 16, 64,
          bound=0.01)
    check("mxfp8_gemm_epilogue", x1024, weight(4096, 1024), randn(4096), activation="gelu_tanh", out_quant=True,
          bound=0.05)
    check("mxfp8_gemm_epilogue", x1024, weight(1024, 1024), randn(1024), residual=randn(1568, 1024), bound=0.01)
    keep = (torch.arange(640, device="cuda") < 600).float()
    check("mxfp8_gemm_epilogue", x2048, weight(8224, 2048), row_mask=keep, bound=0.01)
    check("mxfp8_gemm_epilogue", x6144, weight(2048, 6144), bound=0.01)
    check("mxfp8_gemm_glu_quantize", x2048, weight(12288, 2048), bound=0.05)
