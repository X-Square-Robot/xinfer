"""MXFP8: quantization, the GEMM entries on MXFP8 operands and the norms that write MXFP8."""
import pytest
import torch
import torch.nn.functional as F

from xinfer.ops.backends.cuda import gemm, norm
from xinfer.ops.layout import mxfp8 as layout
from conftest import DEVICE, MXFP8_TOLERANCE, close, dequantize, sm
from test_kernel_qkv_proj_rope import angles
from test_kernel_qkv_proj_rope import reference as qkv_reference

needs_fp8_gemm = sm(120)  # cuBLASLt has block-scaled FP8 GEMMs from SM120 on


@pytest.mark.parametrize("m,k", [(1, 32), (37, 96), (868, 1024)])
@pytest.mark.parametrize("scale_block", [32, 128])
def test_quantize(m, k, scale_block, rnd):
    x = rnd(m, k) * torch.logspace(-3, 3, k, device=DEVICE).to(torch.bfloat16)  # columns of very different size
    values, scales = gemm.quantize_mxfp8(x, scale_block=scale_block)
    assert values.dtype == torch.float8_e4m3fn and scales.numel() == layout.scale_elements(m, k)
    assert close(dequantize((values, scales)), x, MXFP8_TOLERANCE)
    exponent = scales[layout.scale_offsets(m, k, DEVICE)][:, : k // 32]
    if scale_block == 128 and k % 128 == 0:
        assert bool((exponent.view(m, -1, 4) == exponent.view(m, -1, 4)[..., :1]).all())  # shared by 128 columns
    with pytest.raises(ValueError):
        gemm.quantize_mxfp8(rnd(4, 40))


@needs_fp8_gemm
@pytest.mark.parametrize("m,n,k", [(1, 32, 32), (7, 96, 64), (300, 1024, 1024)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("options", ["plain", "silu", "residual", "gelu_tanh+quantize"])
def test_generic_gemm(m, n, k, bias, options, rnd):
    x, weight = gemm.quantize_mxfp8(rnd(m, k)), gemm.quantize_mxfp8(rnd(n, k, scale=0.05))
    b = rnd(n) if bias else None
    y = F.linear(dequantize(x), dequantize(weight), None if b is None else b.float())
    if options == "residual":
        residual = rnd(m, n)
        out = gemm.gemm(x, weight, b, residual=residual)
        assert torch.equal(out, (gemm.gemm(x, weight, b).float() + residual.float()).to(torch.bfloat16))
        assert close(out, y + residual.float())
        with pytest.raises(ValueError):
            gemm.gemm(x, weight, b, residual=residual, out=residual)
    elif options == "gelu_tanh+quantize":
        out = gemm.gemm(x, weight, b, activation="gelu_tanh", quantize=True)
        assert close(dequantize(out), F.gelu(y, approximate="tanh"), MXFP8_TOLERANCE)
    else:
        activation = None if options == "plain" else options
        assert close(gemm.gemm(x, weight, b, activation=activation), y if activation is None else F.silu(y))
    with pytest.raises(ValueError):
        gemm.gemm(x, rnd(n, k))  # one MXFP8 operand, one BF16


@needs_fp8_gemm
@pytest.mark.parametrize("m,n,k", [(5, 32, 64), (464, 6144, 2048)])
@pytest.mark.parametrize("quantize", [False, True])
def test_generic_gemm_glu(m, n, k, quantize, rnd):
    x, weight = gemm.quantize_mxfp8(rnd(m, k)), gemm.quantize_mxfp8(rnd(2 * n, k, scale=0.05))
    y = F.linear(dequantize(x), dequantize(weight))
    expected = F.silu(y[:, :n]) * y[:, n:]
    out = gemm.gemm_glu(x, weight, quantize=quantize)
    assert close(dequantize(out) if quantize else out, expected, MXFP8_TOLERANCE if quantize else 5e-3)


@needs_fp8_gemm
@pytest.mark.parametrize("m", [1, 868, 2240, 2241])
@pytest.mark.parametrize("table", [False, True])
def test_qkv_proj_rope(m, table, rnd):
    """Heads of 64 with a packed table run the tile kernel inside its range; every call gives the generic bits."""
    k, heads, d = 1024, 16, 64
    n = 3 * heads * d
    x, weight, bias = gemm.quantize_mxfp8(rnd(m, k)), gemm.quantize_mxfp8(rnd(n, k, scale=0.05)), rnd(n)
    cos, sin = angles(m, d)
    tables = dict(rope_table=layout.rope_table(cos, sin)) if table else dict(cos=cos, sin=sin)
    name = gemm.kernel_name("qkv_proj_rope", x, weight, bias, num_heads=heads, head_dim=d, **tables)
    assert name == ("tile_qkv_rope_mxfp8_sm120" if table and 1 <= m <= 2240 else "generic")
    out = gemm.qkv_proj_rope(x, weight, bias, num_heads=heads, head_dim=d, **tables)
    expected = qkv_reference(dequantize(x), dequantize(weight), bias, cos, sin, heads, heads, d)
    assert close(out, expected)
    generic = gemm.qkv_proj_rope(x, weight, bias, num_heads=heads, head_dim=d, cos=cos, sin=sin)
    assert torch.equal(out, generic)


@needs_fp8_gemm
def test_tile_residual_gives_the_generic_bits(rnd):
    m, n, k = 868, 1024, 4096
    x, weight = gemm.quantize_mxfp8(rnd(m, k)), gemm.quantize_mxfp8(rnd(n, k, scale=0.05))
    bias, residual = rnd(n), rnd(m, n)
    assert gemm.kernel_name("gemm", x, weight, bias, residual=residual) == "tile_residual_mxfp8_sm120"
    out = gemm.gemm(x, weight, bias, residual=residual)
    assert torch.equal(out, (gemm.gemm(x, weight, bias).float() + residual.float()).to(torch.bfloat16))


def norm_reference(x, weight, eps, zero_centered):
    x = x.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (weight.float() + zero_centered)
    return normed.to(torch.bfloat16)


@pytest.mark.parametrize("shape", [(464, 2048), (1, 16, 1024), (3, 96)])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_rmsnorm_quantize(shape, zero_centered, rnd):
    x, residual, weight = rnd(*shape), rnd(*shape), rnd(shape[-1], dtype=torch.float32)
    values, scales = norm.rmsnorm(x, weight, 1e-6, zero_centered=zero_centered, quantize=True)
    assert tuple(values.shape) == shape
    flat = (values.reshape(-1, shape[-1]), scales)
    assert close(dequantize(flat), norm_reference(x, weight, 1e-6, zero_centered).reshape(-1, shape[-1]),
                 MXFP8_TOLERANCE)
    total, (values, scales) = norm.add_rmsnorm(residual, x, weight, 1e-6, zero_centered=zero_centered, quantize=True)
    assert torch.equal(total, (residual.float() + x.float()).to(torch.bfloat16))
    assert close(dequantize((values.reshape(-1, shape[-1]), scales)),
                 norm_reference(residual.float() + x.float(), weight, 1e-6, zero_centered).reshape(-1, shape[-1]),
                 MXFP8_TOLERANCE)


@pytest.mark.parametrize("shape", [(868, 1024), (5, 96)])  # one kernel at 1024, the norm then the quantization at 96
def test_layernorm_quantize_is_the_norm_then_quantize(shape, rnd):
    x, weight, bias = rnd(*shape), rnd(shape[-1]), rnd(shape[-1])
    values, scales = norm.layernorm(x, weight, bias, 1e-6, quantize=True)
    expected = gemm.quantize_mxfp8(norm.layernorm(x, weight, bias, 1e-6))
    assert close(dequantize((values, scales)), dequantize(expected), 1e-2)
    assert close(dequantize((values, scales)), F.layer_norm(x.float(), (shape[-1],), weight.float(), bias.float(),
                                                             1e-6), MXFP8_TOLERANCE)
