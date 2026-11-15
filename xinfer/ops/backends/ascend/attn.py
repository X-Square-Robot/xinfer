"""Ascend attention through OpSpec-named exports, optional CANN bridges and raw xcompute entries.

Unsupported layouts fall back per cached call key to the graph-safe Ascend Torch composition;
the platform-independent reference remains SDPA. Legacy eager-only variable-length entries are not registered.
"""

import math

import torch

from xinfer.ops.backends.ascend import _torch_attn
from xinfer.ops.backends.ascend._support import (
    VERSION,
    bridge,
    call_key,
    canonical,
    entry,
    has_functions,
    partial,
    xc,
)
from xinfer.ops.registry import register
from xinfer.ops.reference import attn as reference

_CACHE_HEADS = ((8, 1, 256), (8, 2, 256), (16, 2, 128))
_CACHE_SYMBOLS = ("attention_kvcache_splitkv", "attention_kvcache_ascend")


def _cache_supported(q, k, v, prefix, new_pos, mask, scale):
    if (q.device.type != "npu" or q.ndim != 4 or k.ndim != 4 or v.shape != k.shape
            or not all(t.dtype == torch.bfloat16 and t.device == q.device and t.stride(-1) == 1
                       for t in (q, k, v))):
        return False
    b, m, h, d = q.shape
    length, kv = k.shape[1:3]
    if (h, kv, d) not in _CACHE_HEADS or k.shape[0] != b or k.shape[-1] != d or not b or not m:
        return False
    if not (0 <= new_pos <= length - m and (scale is None or (math.isfinite(scale) and scale > 0))):
        return False
    if prefix.dtype != torch.int32 or prefix.shape != (b,) or prefix.device != q.device or not prefix.is_contiguous():
        return False
    return mask is None or (mask.shape == (b, 1, length, 2) and mask.dtype == torch.int32
                            and mask.device == q.device and mask.is_contiguous() and mask.data_ptr() % 8 == 0)


def _attention():
    factory = getattr(bridge("cann_attention"), "CannAttention", None)
    return factory() if callable(factory) else None


def _route(attention, name, launch, query, symbols):
    bound = (getattr(attention, launch, None), getattr(attention, query, None))
    capable = all(callable(fn) for fn in bound) if any(callable(fn) for fn in bound) else has_functions(*symbols)
    function = getattr(attention, name, None)
    return function if capable and callable(function) else None


canonical_varlen = canonical("varlen_attention", reference.varlen_attention)
canonical_causal = canonical("causal_attention", reference.causal_attention)
canonical_cache = canonical("prefix_kvcache_attention", reference.prefix_kvcache_attention)
attention = (_attention() if any(function is None for function in
                                (canonical_varlen, canonical_causal, canonical_cache)) else None)
native_varlen = canonical_varlen or _route(attention, "varlen_attention", "_tnd_launch", "_tnd_query",
                                         ("cann_fia_vision_probe", "cann_fia_vision_probe_workspace_size"))
native_causal = canonical_causal or _route(attention, "causal_attention", "_bnsd_launch", "_bnsd_query",
                                         ("cann_fia_v5_probe", "cann_fia_v5_probe_workspace_size"))
native_cache = canonical_cache or _route(attention, "prefix_kvcache_attention", "_bnsd_launch", "_bnsd_query",
                                        ("cann_fia_v5_probe", "cann_fia_v5_probe_workspace_size"))
raw_cache = (entry("attention_kvcache") if native_cache is None
             and any(has_functions(name) for name in _CACHE_SYMBOLS) else None)


def _varlen_key(q, k, v, cu_seqlens, *, scale=None):
    # Segment values define this call's layout. A tensor is read back, as in the reference and CUDA backend;
    # callers use a host sequence for graph capture.
    bounds = tuple(cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else cu_seqlens)
    return call_key((q, k, v, bounds), {"scale": scale})


@register("varlen_attention", backend="ascend", requires=xc if callable(native_varlen) else None,
          version=VERSION)
@partial(_torch_attn.varlen_attention, key=_varlen_key)
def varlen_attention(q, k, v, cu_seqlens, *, scale=None):
    return native_varlen(q, k, v, cu_seqlens, scale=scale)


@register("causal_attention", backend="ascend", requires=xc if callable(native_causal) else None,
          version=VERSION)
@partial(_torch_attn.causal_attention)
def causal_attention(q, k, v, *, scale=None, return_bnsd=False):
    return native_causal(q, k, v, scale=scale, return_bnsd=return_bnsd)


def _prefix_supported(q, k_cache, v_cache, prefix_len, new_pos, *, flash_mask=None, scale=None, all_keep=False,
                      return_bnsd=False):
    return callable(native_cache) or _cache_supported(q, k_cache, v_cache, prefix_len, new_pos, flash_mask, scale)


@register("prefix_kvcache_attention", backend="ascend", requires=xc if callable(native_cache) else raw_cache,
          version=VERSION)
@partial(_torch_attn.prefix_kvcache_attention, check=_prefix_supported)
def prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, *, flash_mask=None, scale=None,
                             all_keep=False, return_bnsd=False):
    # The OpSpec contract is [B, M, HD]: a provider that returns [B, M, H, D] must be flattened here, and for a
    # provider that already returns [B, M, HD] this reshape is the identity. Never trust the provider's shape.
    if callable(native_cache):
        out = native_cache(q, k_cache, v_cache, prefix_len, new_pos, flash_mask=flash_mask, scale=scale,
                           all_keep=all_keep, return_bnsd=return_bnsd)
    else:
        if all_keep and flash_mask is not None:
            raise ValueError("all_keep cannot be combined with a flash mask")
        out = xc.attention_kvcache(q, k_cache, v_cache, prefix_len=prefix_len, new_pos=new_pos,
                                   flash_mask=None if all_keep else flash_mask, softmax_scale=scale,
                                   backend="ascend_tilelang", pdl=False)
        if return_bnsd:
            b, m, h, d = q.shape
            out = out.view(b, m, h, d).transpose(1, 2)
    if return_bnsd:
        return out
    return out.reshape(q.shape[0], q.shape[1], -1)


# The cache-attention metadata scope rides the prefix attention: models open it around their action loop so the
# dense drop mask is built once per forward from the live flash mask instead of once per call. The torch
# composition keeps a null context (the model's getattr default).
_action_scope = getattr(attention, "action_scope", None) if attention is not None else None
if callable(_action_scope):
    prefix_kvcache_attention.action_scope = _action_scope


# ---------------------------------------------------------------- Wall-X 2B attention family
# One packed-segment entry serves both 2B attention operators: ``causal`` picks the mask and ``out_gate`` folds the
# output gate into the same kernel, which is exactly how the reference composes them.
_attention_varlen = getattr(xc, "attention_varlen", None) if xc is not None else None
_varlen_backend = getattr(xc, "attention_varlen_backend", None) if xc is not None else None
_gated_delta = getattr(xc, "CannGatedDeltaNet", None) if xc is not None else None
_gated_delta_net = _gated_delta() if callable(_gated_delta) else None


def _packed_supported(q, k, v) -> bool:
    """``attention_varlen`` reads dense-last-dimension [T, H, D] BF16 rows."""
    return (q.device.type == "npu" and q.ndim == 3 and k.ndim == 3 and v.ndim == 3
            and all(t.dtype == torch.bfloat16 and t.device == q.device and t.stride(-1) == 1 for t in (q, k, v))
            and k.shape == v.shape and q.shape[-1] == k.shape[-1]
            and q.shape[1] % k.shape[1] == 0)


def _varlen_accepts(q, k, v, **options) -> bool:
    """Ask the entry whether it serves this layout before calling it.

    `attention_varlen` raises for a layout outside its white list. Probing first keeps that rejection on the
    partial-support path (a cached "unsupported" and the reference) instead of letting a runtime error escape,
    which is the same shape of check the CUDA backend uses for its kernel lists.
    """
    if not callable(_varlen_backend):
        return False
    try:
        _varlen_backend(q, k, v, **options)
    except (ValueError, RuntimeError):
        return False
    return True


def _segment_bounds(cu_seqlens):
    """The CPU int32 tensor the host-planning entries want; a host sequence is uploaded once per distinct set."""
    if torch.is_tensor(cu_seqlens):
        return cu_seqlens if (cu_seqlens.dtype == torch.int32 and cu_seqlens.device.type == "cpu") \
            else cu_seqlens.to("cpu", torch.int32)
    bounds = tuple(cu_seqlens)
    cached = _SEGMENT_BOUNDS.get(bounds)
    if cached is None:
        cached = torch.tensor(bounds, dtype=torch.int32)  # the ascend_fia entries plan on the host
        _SEGMENT_BOUNDS[bounds] = cached
    return cached


def _segment_longest(cu_seqlens):
    """The longest segment of a bounds sequence, the ``max_seqlen`` the packed entries ask for."""
    bounds = tuple(cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else cu_seqlens)
    return max(b - a for a, b in zip(bounds[:-1], bounds[1:]))


_SEGMENT_BOUNDS: dict[tuple[int, ...], torch.Tensor] = {}


def _segment_supported(q, k, v, cu_seqlens, *, scale=None):
    """The check receives the operator's own parameter names, so a keyword call resolves here too."""
    return (_packed_supported(q, k, v)
            and _varlen_accepts(q, k, v, cu_seqlens=_segment_bounds(cu_seqlens),
                                max_seqlen=_segment_longest(cu_seqlens), causal=False))


@register("segment_attention", backend="ascend", requires=xc if callable(_attention_varlen) else None,
          version=VERSION)
@partial(_torch_attn.segment_attention, check=_segment_supported)
def segment_attention(q, k, v, cu_seqlens, *, scale=None):
    t, heads, dim = q.shape
    out = _attention_varlen(q, k, v, cu_seqlens=_segment_bounds(cu_seqlens), max_seqlen=_segment_longest(cu_seqlens),
                            causal=False, softmax_scale=scale)
    return out.reshape(t, heads, dim)


_bridge_gated_causal = getattr(attention, "gated_causal_attention", None) if attention is not None else None


@register("gated_causal_attention", backend="ascend", requires=xc if callable(_attention_varlen) else None,
          version=VERSION)
@partial(_torch_attn.gated_causal_attention,
         check=lambda q, k, v, gate, *, scale=None: _packed_supported(q, k, v)
         and gate.dtype == torch.bfloat16 and gate.device == q.device
         and gate.shape == (q.shape[0], q.shape[1] * q.shape[2])
         and (_bridge_gated_causal is not None
              or _varlen_accepts(q, k, v, seg_len=q.shape[0], causal=True, out_gate=gate)))
def gated_causal_attention(q, k, v, gate, *, scale=None):
    # The entry expects one head count for queries and keys, as the reference's helper does: repeat the grouped
    # keys and values up to the query heads. One sequence per call, so the segment length is the row count; the
    # entry returns [T, H*D] with the gate applied, which is this operator's contract.
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    if _bridge_gated_causal is not None:
        # The packed entry's white list has no head set for this prompt shape (8 x 256 on Ascend950); the CANN
        # attention bridge's own gated causal composition serves every layout its probe takes, with the same
        # two steps the reference performs.
        return _bridge_gated_causal(q, k, v, gate, scale=scale)
    return _attention_varlen(q, k, v, seg_len=q.shape[0], causal=True, softmax_scale=scale, out_gate=gate)


@register("gated_kvcache_attention", backend="ascend",
          requires=xc if callable(native_cache) or raw_cache is not None else None,
          version=VERSION)
@partial(_torch_attn.gated_kvcache_attention,
         check=lambda q, k_cache, v_cache, prefix_len, new_pos, gate, *, flash_mask=None, scale=None:
         _cache_supported(q, k_cache, v_cache, prefix_len, new_pos, flash_mask, scale)
         and gate.dtype == torch.bfloat16 and gate.device == q.device
         and gate.shape == (q.shape[0], q.shape[1], q.shape[2] * q.shape[3]))
def gated_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, gate, *, flash_mask=None, scale=None):
    # FIA-FUSE (env XINFER_FIA_FUSE, default off): the 2B decode family takes the
    # specialized scan+reduce pair that fuses the cache attention call and the
    # gated pack tail; every other shape falls through unchanged.
    try:
        from xinfer.ops.backends.ascend import fia_fuse_route as _fia_fuse

        _fused = _fia_fuse.try_route(
            q, k_cache, v_cache, prefix_len, new_pos, gate, flash_mask, scale)
    except ImportError:
        _fused = None
    if _fused is not None:
        return _fused
    # The same two steps as the reference — the cache attention, then the BF16 gated product — with the attention
    # served by the native entry above instead of SDPA.
    out = prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, flash_mask=flash_mask, scale=scale)
    return _sigmoid_gate(out, gate)


# The same metadata scope the prefix entry carries: the Wall-X 2B action loop drives this operator
# once per layer per flow step and opens the scope around the whole forward, so all of the loop's
# calls share the one dense drop mask instead of each rebuilding it.
if callable(_action_scope):
    gated_kvcache_attention.action_scope = _action_scope


def _sigmoid_gate(out, gate):
    """``bf16(bf16(out) * bf16(sigmoid(gate)))``: the two roundings ``gate_output`` performs."""
    native = getattr(xc, "sigmoid_gate_mul_inplace", None) if xc is not None else None
    if callable(native):
        native(out, gate)
        return out
    return (out.to(gate.dtype) * torch.sigmoid(gate)).to(out.dtype)


@register("gated_delta_net", backend="ascend", requires=xc if _gated_delta_net is not None else None,
          version=VERSION)
@partial(_torch_attn.gated_delta_net,
         check=lambda qkv, zba, conv_weight, A_log, dt_bias, norm_weight, num_k_heads, num_v_heads,
         head_k_dim, head_v_dim, eps:
         qkv.device.type == "npu" and qkv.ndim == 3 and qkv.shape[0] == 1
         and qkv.dtype == torch.bfloat16 and zba.dtype == torch.bfloat16
         and num_k_heads == 16 and num_v_heads in (16, 32) and (head_k_dim, head_v_dim) == (128, 128))
def gated_delta_net(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, num_k_heads, num_v_heads,
                    head_k_dim, head_v_dim, eps):
    # The entry's parameters are this operator's parameters in order; it returns [1, T, Hv*Dv] BF16, the
    # reference's [B, L, Hv*Dv] with B == 1.
    return _gated_delta_net(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, num_k_heads, num_v_heads,
                            head_k_dim, head_v_dim, float(eps))
