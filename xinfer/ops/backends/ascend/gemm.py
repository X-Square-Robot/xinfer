"""Fused Ascend projections using the paired xcompute package.

Explicit xcompute exports with the OpSpec names take precedence. Otherwise CANN bridges handle supported
projection and AdaRMS calls in eager execution and graph capture. Builds
without these bridges use the references. Other layouts/options also run the reference, with support cached
per call layout; legacy eager-only raw entries are not registered.
Stream-K reductions may differ at rounding level across runs.

``gemm_add_rms_norm`` and ``mlp_add_rms_norm`` call this backend's ``gemm_epilogue``, ``gemm_glu_epilogue`` and
``add_rms_norm`` in turn (the default selections of those operators), so their results are those calls'.
"""

import torch

from xinfer.ops.backends.ascend._support import VERSION, bridge, canonical, partial, xc
from xinfer.ops.backends.ascend.norm import add_rms_norm, norm_library
from xinfer.ops.reference import gemm as reference
from xinfer.ops.registry import register


def _projection():
    module = bridge("cann_projection")
    factory = getattr(module, "CannProjection", None)
    try:
        return factory() if callable(factory) else None
    except getattr(xc, "MissingBackendError", ()):
        return None  # the constructor binds exports missing from builds without the CANN bridge


def _action(projection):
    if not callable(getattr(projection, "accumulator", None)):
        return None
    factory = getattr(bridge("cann_action"), "CannAction", None)
    return factory(projection.accumulator, precision="bf16-split2") if callable(factory) else None


canonical_gemm = canonical("gemm_epilogue", reference.gemm_epilogue)
canonical_glu = canonical("gemm_glu_epilogue", reference.gemm_glu_epilogue)
canonical_qkv = canonical("qkv_rope_epilogue", reference.qkv_rope_epilogue)
canonical_cache = canonical("qkv_rope_kvcache", reference.qkv_rope_kvcache)
cann_projection = (_projection() if any(function is None for function in
                                       (canonical_gemm, canonical_glu, canonical_qkv, canonical_cache)) else None)
cann_action = _action(cann_projection) if canonical_glu is None or canonical_cache is None else None
native_gemm = canonical_gemm or getattr(cann_projection, "gemm_epilogue", None)
cann_glu = getattr(cann_projection, "gemm_glu_epilogue", None)
native_qkv = canonical_qkv or getattr(cann_projection, "qkv_rope_epilogue", None)
native_cache = canonical_cache or getattr(cann_action, "qkv_rope_kvcache", None)


@register("gemm_epilogue", backend="ascend", requires=xc if callable(native_gemm) else None,
          version=VERSION)
@partial(reference.gemm_epilogue)
def gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, residual_gate=None,
                  row_class=None, row_scale=None):
    return native_gemm(x, weight, bias, activation=activation, residual=residual, residual_gate=residual_gate,
                       row_class=row_class, row_scale=row_scale)


def _glu_supported(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
    return canonical_glu is not None or modulation is None or callable(getattr(cann_action, "gemm_glu_epilogue", None))


@register("gemm_glu_epilogue", backend="ascend",
          requires=xc if canonical_glu is not None or callable(cann_glu) else None,
          version=VERSION)
@partial(reference.gemm_glu_epilogue, check=_glu_supported)
def gemm_glu_epilogue(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
    if canonical_glu is not None:
        return canonical_glu(x, weight, bias=bias, modulation=modulation, row_class=row_class, eps=eps)
    if modulation is None:
        return cann_glu(x, weight, bias=bias, row_class=row_class, eps=eps)
    return cann_action.gemm_glu_epilogue(x, weight, bias=bias, modulation=modulation,
                                        row_class=row_class, eps=eps)


@register("gemm_add_rms_norm", backend="ascend",
          requires=xc if callable(native_gemm) and norm_library is not None else None, version=VERSION)
def gemm_add_rms_norm(x, projection, residual, weight, eps, bias=None):
    return add_rms_norm(gemm_epilogue(x, projection, bias).view(residual.shape), residual, weight, eps)


@register("mlp_add_rms_norm", backend="ascend",
          requires=xc if (callable(native_gemm) and norm_library is not None
                          and (canonical_glu is not None or callable(cann_glu))) else None, version=VERSION)
def mlp_add_rms_norm(x, gate_up, down, residual, weight, eps):
    return gemm_add_rms_norm(gemm_glu_epilogue(x, gate_up), down, residual, weight, eps)


@register("qkv_rope_epilogue", backend="ascend", requires=xc if callable(native_qkv) else None,
          version=VERSION)
@partial(reference.qkv_rope_epilogue)
def qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim, *, qk_order="standard"):
    return native_qkv(x, weight, bias, cos, sin, num_heads, head_dim)  # the standard order serves "any" too


_fused_projection = getattr(xc, "qkv_proj_rope", None) if xc is not None else None
_qk_norm = getattr(xc, "QkNorm", None) if xc is not None else None


# ``qkv_rope_kvcache`` takes the unified projection entry first: the same ``qkv_proj_rope`` call with its
# ``kv_cache`` epilogue that the end-to-end precision gate validated. The CANN action bridge also serves the
# operator, but its composition fails that gate for Wall-OSS (every other operator family passes), so it stays the
# fallback for builds without the entry.
@register("qkv_rope_kvcache", backend="ascend",
          requires=xc if callable(_fused_projection) or callable(native_cache) else None,
          version=VERSION, deterministic=False)
@partial(reference.qkv_rope_kvcache)
def qkv_rope_kvcache(x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads, head_dim):
    if callable(_fused_projection):
        return _fused_projection(x.contiguous(), weight, bias, cos=cos, sin=sin, rotary_dim=2 * cos.shape[-1],
                                 num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                                 kv_cache=(k_cache, v_cache, int(start)), backend="auto")
    return native_cache(x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads, head_dim)


# The prompt stream's plain projection (``F.linear``'s contract) and its gated Q/K split. The GEMM family's
# unified entry serves both on Ascend: ``xcompute.gemm`` runs the plain arithmetic (bit for bit the torch
# reference at the served shapes), and ``qkv_norm_rope_kvcache`` below is the same family's fused projection.
_plain_gemm = getattr(xc, "gemm", None) if xc is not None else None
_split_forward = getattr(xc.xcompute_lib, "qwen35_vla_qkv_norm_replace_forward", None) if xc is not None else None


def _linear_supported(x, weight, bias=None):
    """Rows with a 2-D weight of the same dtype and width, as the unified entry takes them."""
    return (x.device.type == "npu" and x.ndim >= 2 and weight.device == x.device and weight.ndim == 2
            and weight.shape[-1] == x.shape[-1]
            and x.dtype == weight.dtype and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and (bias is None or (bias.device == x.device and bias.dtype == x.dtype
                                  and bias.shape == (weight.shape[0],))))


@register("linear", backend="ascend", requires=xc if callable(_plain_gemm) else None,
          version=VERSION)
@partial(reference.linear, check=_linear_supported)
def linear(x, weight, bias=None):
    # The entry takes rank-matching operands, so flatten the leading dimensions of a higher-rank input and
    # restore them (the same shape handling the official-operator alternative performs).
    lead = x.shape[:-1]
    out = _plain_gemm(x.reshape(-1, x.shape[-1]), weight, bias)
    return out.reshape(*lead, weight.shape[0]) if lead else out


_no_action_rows = {}  # (batch, device) -> zero action-row starts/ends [B] and offsets [B + 1] (int64)


@register("qkv_norm_split", backend="ascend", requires=xc if callable(_split_forward) else None,
          version=VERSION)
@partial(reference.qkv_norm_split)
def qkv_norm_split(packed, q_weight, k_weight, num_heads, num_kv_heads, head_dim, eps):
    b, n = packed.shape[:2]
    key = (b, packed.device)
    if key not in _no_action_rows:  # every row is a text row
        _no_action_rows[key] = (torch.zeros(b, dtype=torch.int64, device=packed.device),
                                torch.zeros(b + 1, dtype=torch.int64, device=packed.device))
    rows, offsets = _no_action_rows[key]
    q = torch.empty((b, n, num_heads, head_dim), dtype=packed.dtype, device=packed.device)
    k = torch.empty((b, n, num_kv_heads, head_dim), dtype=packed.dtype, device=packed.device)
    v = torch.empty_like(k)
    gate = torch.empty((b, n, num_heads * head_dim), dtype=packed.dtype, device=packed.device)
    # The kernel takes a text and an action projection with their norm weights; with no action rows it reads the
    # text ones only. With every row a text row the zero action projection is never read.
    _split_forward(q, k, v, gate, packed, packed, rows, rows, offsets, q_weight, k_weight, q_weight, k_weight,
                   int(num_heads), int(num_kv_heads), int(head_dim), float(eps))
    return q, k, v, gate


# ``qkv_norm_rope_kvcache`` is the unified projection entry with the norm, RoPE and cache writes as epilogues.
# The recorded reason for keeping it on the reference (a workspace the caller had to size, so it could not be
# captured) no longer holds: this build sizes its workspace per stream, the lineage binding of this same entry
# runs captured, and the fused path is far ahead of the composition (about 60 us against more than 300 per call
# at the action shape). The raw ``[N, K]`` weight takes the unified entry's generic route; prepacking at load
# time is the model-side follow-up that would take the fast kernels.


@register("qkv_norm_rope_kvcache", backend="ascend",
          requires=xc if callable(_fused_projection) and callable(_qk_norm) else None,
          version=VERSION, deterministic=False)
@partial(reference.qkv_norm_rope_kvcache)
def qkv_norm_rope_kvcache(x, weight, q_weight, k_weight, cos, sin, k_cache, v_cache, start, num_heads,
                          num_kv_heads, head_dim, eps):
    return _fused_projection(
        x.contiguous(), weight, cos=cos, sin=sin, rotary_dim=2 * cos.shape[-1], num_heads=num_heads,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        extra_cols=weight.shape[0] - (num_heads + 2 * num_kv_heads) * head_dim,
        qk_norm=_qk_norm(q_weight, k_weight, float(eps), zero_centered=True),
        kv_cache=(k_cache, v_cache, int(start)), backend="auto")
