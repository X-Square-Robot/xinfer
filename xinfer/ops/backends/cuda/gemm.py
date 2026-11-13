"""GEMM operators with fused epilogues on CUDA, through xcompute (``_xcompute``).

Stream-K GEMMs may differ at rounding level across runs (``deterministic=False``).
"""

from __future__ import annotations

from weakref import WeakKeyDictionary

from xinfer.ops.backends._xcompute import XCOMPUTE, xc, xcompute_with
from xinfer.ops.registry import register

_supports_kn = True
_packed = WeakKeyDictionary()  # source storage -> its views and their packed weights


def _listed_weight(op, x, weight, **options):
    """``weight``, or its xcompute packing when the call would run the generic path on the raw ``[N, K]`` weight
    and runs a listed kernel on the packed one.

    A weight is packed once, on its first call that would run the generic path (eager: the engine runs every new
    graph key before capturing it), and the packing is kept only if that call runs a listed kernel with it.
    """
    if xc.selected_backend(op, x, weight, **options) != "generic":
        return weight
    storage = weight.untyped_storage()
    views = _packed.setdefault(storage, {})
    key = (op, weight.storage_offset(), weight.shape, weight.stride(), weight.dtype)
    if key not in views:
        packed = xc.prepack_weight(weight, op=op)
        if xc.selected_backend(op, x, packed, **options) == "generic":
            packed = None
        elif packed.tensors["w"].untyped_storage() is storage:
            return packed  # caching a zero-copy packing would keep the weak key alive
        views[key] = packed
    packed = views[key]
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
