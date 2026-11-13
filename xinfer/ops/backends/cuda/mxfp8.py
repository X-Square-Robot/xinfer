"""MXFP8 GEMMs, quantization and norm epilogues for the CUDA xcompute backend.

Quantized operands pair E4M3 values and E8M0 scales. A build with the MXFP8 GEMM family can
fuse the norm or GEMM output with the next projection's activation quantization.
"""

from __future__ import annotations

import torch

from xinfer.ops.backends._xcompute import XCOMPUTE, library, xc, xcompute_with
from xinfer.ops.layout.mxfp8 import scale_elements
from xinfer.ops.registry import register

# The SM120 projections use tiled kernels where available, otherwise xcompute's block-scaled cuBLASLt GEMM.
# Older builds may export activation quantization without the GEMM or quantized normalization paths.
FUNCTIONS = ("gemm_generic_mxfp8", "quantize_mxfp8_into", "layernorm", "layernorm_quant_mxfp8",
             "layernorm_quant_mxfp8_selected_backend", "rmsnorm_quant_mxfp8_into")
MXFP8 = (dict(xcompute_with("layernorm", functions=FUNCTIONS), platforms=("sm120",))
         if "sm120" in XCOMPUTE["platforms"] else dict(XCOMPUTE, requires=None))
QUANT = dict(out_quant="mxfp8", scale_block=128)


def _weight(op, weight, **heads):
    """Keep the packing with its quantized weight while that weight is in use."""
    q, scales = weight
    cache = getattr(q, "_xinfer_mxfp8_packings", None)
    if cache is None:
        cache = q._xinfer_mxfp8_packings = {}
    key = (op, tuple(heads.items()))
    cached = cache.get(key)
    if cached is None or cached[0] is not scales:
        cached = (scales, xc.prepack_weight(q, op=op, scale=scales, **heads))
        cache[key] = cached
    return cached[1]


@register("mxfp8_quantize", backend="cuda", **MXFP8)
def mxfp8_quantize(x, *, share_scale_128=True):
    x = x.contiguous()
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(scale_elements(*x.shape), dtype=torch.uint8, device=x.device)
    library.call("quantize_mxfp8_into", x, q, scales, bool(share_scale_128))
    return q, scales


@register("mxfp8_layer_norm_quantize", backend="cuda", **MXFP8)
def mxfp8_layer_norm_quantize(x, weight, bias, eps):
    return xc.layernorm(x.reshape(-1, x.shape[-1]), weight, bias, float(eps), quant="mxfp8", scale_block=128)


@register("mxfp8_zero_centered_rms_norm_quantize", backend="cuda", **MXFP8)
def mxfp8_zero_centered_rms_norm_quantize(x, weight, eps):
    return xc.rmsnorm(x.reshape(-1, x.shape[-1]), weight, float(eps), zero_centered=True, quant="mxfp8",
                      scale_block=128, round_to_bf16=True)


@register("mxfp8_add_zero_centered_rms_norm_quantize", backend="cuda", **MXFP8)
def mxfp8_add_zero_centered_rms_norm_quantize(x, residual, weight, eps):
    k = x.shape[-1]
    summed, q, scales = xc.rmsnorm(x.reshape(-1, k), weight, float(eps), residual=residual.reshape(-1, k),
                                   zero_centered=True, quant="mxfp8", scale_block=128, round_to_bf16=True)
    return summed.view(x.shape), (q, scales)


@register("mxfp8_gemm_epilogue", backend="cuda", **MXFP8)
def mxfp8_gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, row_mask=None, out_quant=False):
    q, scales = x
    if row_mask is not None:
        q = (q.view(torch.uint8) * row_mask.to(torch.uint8)[:, None]).view(torch.float8_e4m3fn)
    options = dict(bias=bias, activation=activation, residual=residual, **(QUANT if out_quant else {}))
    return xc.gemm((q, scales), _weight("gemm", weight), **options, backend="auto")


@register("mxfp8_gemm_glu_quantize", backend="cuda", **MXFP8)
def mxfp8_gemm_glu_quantize(x, weight):
    return xc.gemm_glu(x, _weight("gemm_glu", weight), activation="silu", **QUANT, backend="auto")


@register("mxfp8_qkv_rope_epilogue", backend="cuda", **MXFP8)
def mxfp8_qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim):
    heads = dict(num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim)
    return xc.qkv_proj_rope(x, _weight("qkv_proj_rope", weight, **heads), bias, rope_table=xc.pack_rope_table(cos, sin),
                            **heads, backend="auto")
