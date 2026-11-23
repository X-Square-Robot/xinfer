"""NVFP4: quantization, the GEMM entries on NVFP4 operands and the norms that write NVFP4."""
import pytest
import torch
import torch.nn.functional as F

from xinfer.ops.backends.cuda import gemm, norm
from xinfer.ops.layout import nvfp4 as layout
from xinfer.ops.reference import nvfp4
from conftest import DEVICE, NVFP4_TOLERANCE, close, dequantize, sm

needs_fp4_gemm = sm(120)  # cuBLASLt has block-scaled FP4 GEMMs from SM120 on


def scales_of(pair):
    """The scale bytes of an NVFP4 pair at their positions, and whether the padding bytes of the storage are zero."""
    data, scales = pair
    rows, k = data.shape[0], 2 * data.shape[1]
    offsets = layout.scale_offsets(rows, k, DEVICE)
    stored = scales.view(torch.uint8)
    used = torch.zeros(stored.numel(), dtype=torch.bool, device=DEVICE)
    used[offsets[:, : k // 16].reshape(-1)] = True
    return stored[offsets][:, : k // 16], bool((stored[~used] == 0).all())


@pytest.mark.parametrize("m,k", [(1, 16), (37, 96), (868, 1024)])
@pytest.mark.parametrize("mse", [False, True])
def test_quantize(m, k, mse, rnd):
    """The kernel divides as the Wall-X kernels do (``a * rcp.approx(b)``): against the exact reference, only values
    at a rounding tie may differ."""
    x = rnd(m, k) * torch.logspace(-3, 3, k, device=DEVICE).to(torch.bfloat16)  # blocks of very different size
    g = x.float().abs().amax().item() / (448 * 6) if mse else 1.0
    data, scales = gemm.quantize_nvfp4(x, global_scale=g, mse=mse)
    want = nvfp4.quantize(x, global_scale=g, mse=mse)
    assert data.shape == (m, k // 2) and scales.numel() == layout.scale_elements(m, k)
    assert (data != want[0]).float().mean().item() < 2e-3
    got_scales, padding_zero = scales_of((data, scales))
    assert (got_scales != scales_of(want)[0]).float().mean().item() < 2e-3 and padding_zero
    assert close(dequantize((data, scales, g)), x, NVFP4_TOLERANCE)
    with pytest.raises(ValueError):
        gemm.quantize_nvfp4(rnd(4, 40))


@needs_fp4_gemm
@pytest.mark.parametrize("m,n,k", [(1, 32, 32), (7, 96, 64), (300, 1024, 1024)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("options", ["plain", "residual", "gelu_tanh+quantize", "glu+quantize"])
def test_generic_gemm(m, n, k, bias, options, rnd):
    """The generic path: ``bf16(g * x.W^T + bias)``, then the epilogue on that BF16 tensor."""
    x = gemm.quantize_nvfp4(rnd(m, k))
    w = rnd(2 * n if options == "glu+quantize" else n, k, scale=0.05)
    g = w.float().abs().amax().item() / (448 * 6)
    weight = (*gemm.quantize_nvfp4(w, global_scale=g, mse=True), g)
    b = rnd(weight[0].shape[0]) if bias else None
    y = F.linear(dequantize(x), dequantize(weight), None if b is None else b.float())
    assert gemm.kernel_name("gemm", x, weight, b) == "generic"
    if options == "plain":
        assert close(gemm.gemm(x, weight, b), y)
    elif options == "residual":
        residual = rnd(m, n)
        out = gemm.gemm(x, weight, b, residual=residual)
        assert torch.equal(out, (gemm.gemm(x, weight, b).float() + residual.float()).to(torch.bfloat16))
    elif options == "gelu_tanh+quantize":
        out = gemm.gemm(x, weight, b, activation="gelu_tanh", quantize=True)
        bf16 = gemm.gemm(x, weight, b, activation="gelu_tanh")
        assert all(torch.equal(a, c) for a, c in zip(scales_of(out)[:1] + (out[0],),
                                                       scales_of(gemm.quantize_nvfp4(bf16))[:1]
                                                       + (gemm.quantize_nvfp4(bf16)[0],)))
        assert close(dequantize(out), F.gelu(y, approximate="tanh"), NVFP4_TOLERANCE)
    elif options == "glu+quantize":
        out = gemm.gemm_glu(x, weight, b, activation="silu", quantize=True)
        assert close(dequantize(out), F.silu(y[:, :n]) * y[:, n:], NVFP4_TOLERANCE)


@needs_fp4_gemm
@pytest.mark.parametrize("m", [1, 300])
def test_generic_qkv_proj_rope(m, rnd):
    """Rotate-half RoPE on the BF16 Q and K heads of ``bf16(g * x.W^T + bias)``, rounded again."""
    heads, d, k = 4, 64, 1024
    x = gemm.quantize_nvfp4(rnd(m, k))
    w = rnd(3 * heads * d, k, scale=0.05)
    g = w.float().abs().amax().item() / (448 * 6)
    weight, b = (*gemm.quantize_nvfp4(w, global_scale=g, mse=True), g), rnd(3 * heads * d)
    angles = torch.randn(m, d // 2, device=DEVICE)
    out = gemm.qkv_proj_rope(x, weight, b, cos=angles.cos(), sin=angles.sin(), num_heads=heads, head_dim=d)
    y = gemm.gemm(x, weight, b).float().view(m, 3, heads, d)
    c, s = angles.cos()[:, None, None], angles.sin()[:, None, None]
    lo, hi = y[:, :2, :, : d // 2], y[:, :2, :, d // 2:]
    want = torch.cat((torch.cat((lo * c - hi * s, hi * c + lo * s), -1), y[:, 2:]), 1).reshape(m, -1)
    assert close(out, want.to(torch.bfloat16))
    assert torch.equal(out[:, 2 * heads * d:], y[:, 2:].reshape(m, -1).to(torch.bfloat16))  # V passes through


@pytest.mark.parametrize("rows,k", [(17, 1024), (868, 1024), (5, 1040)])
def test_layernorm_nvfp4_is_the_quantized_bf16_layernorm(rows, k, rnd):
    """One kernel where it takes the rows (K % 128 == 0, K <= 2048), else the norm and the quantization: the same
    bits either way, padding scales zero."""
    x, w, b = rnd(rows, k), rnd(k, scale=0.1) + 1, rnd(k, scale=0.1)
    out = norm.layernorm(x, w, b, 1e-6, quantize="nvfp4")
    want = gemm.quantize_nvfp4(norm.layernorm(x, w, b, 1e-6))
    assert torch.equal(out[0], want[0])
    got, padding_zero = scales_of(out)
    assert torch.equal(got, scales_of(want)[0]) and padding_zero


@pytest.mark.parametrize("rows,k", [(1, 2048), (464, 2048), (3, 1040)])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_add_rmsnorm_nvfp4_is_the_quantized_bf16_norm(rows, k, weight_dtype, rnd):
    """The wide add + RMSNorm kernel quantizing what it writes (K % 64 == 0), else the two steps: the sum and the
    NVFP4 of the zero-centered BF16 norm, bit for bit."""
    x, residual, w = rnd(rows, k), rnd(rows, k, scale=4.0), rnd(k, dtype=weight_dtype, scale=0.1)
    total, out = norm.add_rmsnorm(residual, x, w, 1e-6, zero_centered=True, quantize="nvfp4")
    want_total, normed = norm.add_rmsnorm(residual, x, w, 1e-6, zero_centered=True)
    want = gemm.quantize_nvfp4(normed)
    assert torch.equal(total, want_total) and torch.equal(out[0], want[0])
    got, padding_zero = scales_of(out)
    assert torch.equal(got, scales_of(want)[0]) and padding_zero


@needs_fp4_gemm
@pytest.mark.parametrize("n,k,epilogue", [
    (512, 2048, "plain"), (1024, 1024, "residual"), (4096, 1024, "gelu_tanh"),
])
def test_sm120_tile_matches_generic_gemm(n, k, epilogue, rnd, monkeypatch):
    m = 17
    x = gemm.quantize_nvfp4(rnd(m, k))
    w = rnd(n, k, scale=0.05)
    g = w.float().abs().amax().item() / (448 * 6)
    weight = (*gemm.quantize_nvfp4(w, global_scale=g, mse=True), g)
    bias = None if epilogue == "plain" else rnd(n)
    options = {}
    if epilogue == "residual":
        options["residual"] = rnd(m, n)
    elif epilogue == "gelu_tanh":
        options.update(activation="gelu_tanh", quantize=True)
    assert gemm.kernel_name("gemm", x, weight, bias, **options) == "cutlass_nvfp4_sm120"
    actual = gemm.gemm(x, weight, bias, **options)
    monkeypatch.setattr(gemm, "_kernel_for", lambda *args: None)
    expected = gemm.gemm(x, weight, bias, **options)
    if epilogue == "gelu_tanh":
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(scales_of(actual)[0], scales_of(expected)[0])
    else:
        assert torch.equal(actual, expected)
