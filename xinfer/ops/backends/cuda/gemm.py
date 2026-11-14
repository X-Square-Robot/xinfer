"""GEMM operators with fused epilogues on CUDA, through xcompute (``_xcompute``).

Stream-K GEMMs may differ at rounding level across runs (``deterministic=False``). The MXFP8 operators
(``mxfp8_quantize``: ``quantize_mxfp8_into``; ``mxfp8_gemm_epilogue``, ``mxfp8_gemm_glu_quantize``,
``mxfp8_qkv_rope_epilogue``: ``gemm``, ``gemm_glu``, ``qkv_proj_rope`` with MXFP8 inputs) register with a build that has
the MXFP8 GEMM family (``mxfp8_registration``) and write one scale per 128 columns.
"""

from __future__ import annotations

import weakref

import torch

from xinfer.ops.backends._xcompute import XCOMPUTE, library, mxfp8_registration, xc, xcompute_with
from xinfer.ops.layout.mxfp8 import scale_elements
from xinfer.ops.registry import register

MXFP8 = mxfp8_registration()
MXFP8_OUT = dict(out_quant="mxfp8", scale_block=128)

_supports_kn = True
_packed = {}  # (op, weight key) -> packed weight, or None when packing does not help


def _key(weight):
    return weight.data_ptr(), tuple(weight.shape), tuple(weight.stride())


def _cached(cache, key, weights, make):
    """``cache[key]``, made on the first call. The entry is dropped when the tensor owning any of ``weights``' storage
    is freed, so a later weight allocated at the same address never gets another weight's packing."""
    if key not in cache:
        cache[key] = make()
        for weight in weights:
            weakref.finalize(weight if weight._base is None else weight._base, cache.pop, key, None)
    return cache[key]


def _listed_weight(op, x, weight, **options):
    """``weight``, or its xcompute packing when the call would run the generic path on the raw ``[N, K]`` weight
    and runs a listed kernel on the packed one.

    A weight is packed once, on its first call that would run the generic path (eager: the engine runs every new
    graph key before capturing it), and the packing is kept only if that call runs a listed kernel with it.
    """
    if xc.selected_backend(op, x, weight, **options) != "generic":
        return weight

    def pack():
        packed = xc.prepack_weight(weight, op=op)
        return packed if xc.selected_backend(op, x, packed, **options) != "generic" else None

    packed = _cached(_packed, (op, *_key(weight)), (weight,), pack)
    return weight if packed is None or xc.selected_backend(op, x, packed, **options) == "generic" else packed


def _kn(weight):
    """Share the transposed weight when xcompute supports ``kn``. Older builds need a dense ``[N, K]`` weight."""
    global _supports_kn
    if not _supports_kn:
        return weight.contiguous()
    try:
        packed = xc.prepack_weight(weight, op="gemm", layout="kn")
    except ValueError as exc:
        if "layout= only takes 'nk_k64blk'" not in str(exc):
            raise
        _supports_kn = False
        return weight.contiguous()
    if packed.tensors["w"].data_ptr() != weight.data_ptr():
        raise ValueError("xcompute packed the GEMM weight into a copy")
    return packed


@register("gemm_epilogue", backend="cuda", deterministic=False, **XCOMPUTE)
def gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, residual_gate=None, row_class=None,
                  row_scale=None):
    x = x.contiguous()
    options = dict(bias=bias, activation=activation, residual=residual, residual_gate=residual_gate,
                   row_class=row_class, row_scale=row_scale)
    if not weight.is_contiguous() and weight.t().is_contiguous():
        weight = _kn(weight)
    else:
        weight = _listed_weight("gemm", x, weight, **options)
    return xc.gemm(x, weight, **options, backend="auto")


@register("gemm_glu_epilogue", backend="cuda", **XCOMPUTE)
def gemm_glu_epilogue(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
    k = x.shape[-1]
    prenorm = None if modulation is None else xc.AdaRms(scale=modulation[:, :k], shift=modulation[:, k:2 * k],
                                                        row_class=row_class, eps=float(eps))
    x, options = x.contiguous(), dict(bias=bias, activation="silu", prenorm=prenorm)
    return xc.gemm_glu(x, _listed_weight("gemm_glu", x, weight, **options), **options, backend="auto")


@register("qkv_rope_epilogue", backend="cuda", deterministic=False, **XCOMPUTE)
def qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim):
    return xc.qkv_proj_rope(x.contiguous(), weight, bias, rope_table=xc.pack_rope_table(cos, sin),
                            num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim, backend="auto")


@register("qkv_norm_rope_kvcache", backend="cuda", deterministic=False, **xcompute_with("QkNorm"))
def qkv_norm_rope_kvcache(x, weight, q_weight, k_weight, cos, sin, k_cache, v_cache, start, num_heads,
                          num_kv_heads, head_dim, eps):
    return xc.qkv_proj_rope(
        x.contiguous(), weight, cos=cos, sin=sin, rotary_dim=2 * cos.shape[-1], num_heads=num_heads,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        extra_cols=weight.shape[0] - (num_heads + 2 * num_kv_heads) * head_dim,
        qk_norm=xc.QkNorm(q_weight, k_weight, float(eps), zero_centered=True),
        kv_cache=(k_cache, v_cache, int(start)), backend="auto")


@register("qkv_rope_kvcache", backend="cuda", deterministic=False, **XCOMPUTE)
def qkv_rope_kvcache(x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads, head_dim):
    return xc.qkv_proj_rope(x.contiguous(), weight, bias, cos=cos, sin=sin,
                            rotary_dim=2 * cos.shape[-1], num_heads=num_heads,
                            num_kv_heads=num_kv_heads, head_dim=head_dim,
                            kv_cache=(k_cache, v_cache, int(start)), backend="auto")


def _mxfp8_weight(op, weight, **heads):
    """The xcompute packing of an MXFP8 weight ``(q [N, K], scales)`` for ``op``, made on its first call."""
    q, scales = weight
    return _cached(_packed, (op, *_key(q)), (q, scales), lambda: xc.prepack_weight(q, op, scale=scales, **heads))


@register("mxfp8_quantize", backend="cuda", **MXFP8)
def mxfp8_quantize(x, *, share_scale_128=True):
    x = x.contiguous()
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(scale_elements(*x.shape), dtype=torch.uint8, device=x.device)
    library.call("quantize_mxfp8_into", x, q, scales, bool(share_scale_128))
    return q, scales


@register("mxfp8_gemm_epilogue", backend="cuda", **MXFP8)
def mxfp8_gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, row_mask=None, out_quant=False):
    q, scales = x
    if row_mask is not None:
        q = (q.view(torch.uint8) * row_mask.to(torch.uint8)[:, None]).view(torch.float8_e4m3fn)
    options = dict(bias=bias, activation=activation, residual=residual, **(MXFP8_OUT if out_quant else {}))
    return xc.gemm((q, scales), _mxfp8_weight("gemm", weight), **options, backend="auto")


@register("mxfp8_gemm_glu_quantize", backend="cuda", **MXFP8)
def mxfp8_gemm_glu_quantize(x, weight):
    return xc.gemm_glu(x, _mxfp8_weight("gemm_glu", weight), activation="silu", **MXFP8_OUT, backend="auto")


@register("mxfp8_qkv_rope_epilogue", backend="cuda", **MXFP8)
def mxfp8_qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim):
    heads = dict(num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim)
    return xc.qkv_proj_rope(x, _mxfp8_weight("qkv_proj_rope", weight, **heads), bias,
                            rope_table=xc.pack_rope_table(cos, sin), **heads, backend="auto")
