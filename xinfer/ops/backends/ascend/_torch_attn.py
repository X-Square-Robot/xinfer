"""Graph-safe Torch attention on Ascend when no native implementation is available.

The FP32 matmul/softmax primitive avoids SDPA's unjoined side stream during NPU graph capture. Layouts, masks
and cache updates use the reference composition helpers; the CPU/CUDA reference keeps its SDPA primitive.
"""

import torch

from xinfer.ops.reference import attn as reference
from xinfer.ops.registry import register


def _graph_safe_attention(q, k, v, *, attn_mask=None, is_causal=False, scale=None):
    """FP32 attention using Torch primitives that run inside a graph capture."""
    mask, causal = attn_mask, is_causal
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.float()
    if causal:
        keep = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~keep, float("-inf"))
    if mask is None and not causal:
        return (scores.softmax(-1) @ v.float()).to(q.dtype)
    # SDPA returns zero for rows that cannot attend to any key.
    empty = (scores == float("-inf")).all(dim=-1, keepdim=True)
    probabilities = scores.masked_fill(empty, 0).softmax(-1)
    return (probabilities @ v.float()).masked_fill(empty, 0).to(q.dtype)


@register("varlen_attention", backend="ascend", requires=torch, priority=0)
def varlen_attention(q, k, v, cu_seqlens, *, scale=None):
    return reference._varlen_attention(_graph_safe_attention, q, k, v, cu_seqlens, scale=scale)


@register("causal_attention", backend="ascend", requires=torch, priority=0)
def causal_attention(q, k, v, *, scale=None):
    return reference._causal_attention(_graph_safe_attention, q, k, v, scale=scale)


@register("action_attention", backend="ascend", requires=torch, priority=0)
def action_attention(q, k, v, k_prompt, v_prompt, num_pinned, prompt_lengths=None, *, scale=None):
    return reference._action_attention(
        _graph_safe_attention, q, k, v, k_prompt, v_prompt, num_pinned, prompt_lengths, scale=scale)


@register("masked_joint_attention", backend="ascend", requires=torch, priority=0)
def masked_joint_attention(q, k, v, k_prompt, v_prompt, mask, *, scale=None):
    return reference._masked_joint_attention(_graph_safe_attention, q, k, v, k_prompt, v_prompt, mask, scale=scale)


@register("prefix_attention", backend="ascend", requires=torch, priority=0)
def prefix_attention(q, k, v, valid, *, scale=None):
    return reference._prefix_attention(_graph_safe_attention, q, k, v, valid, scale=scale)


@register("kvcache_attention", backend="ascend", requires=torch, priority=0)
def kvcache_attention(q, k, v, k_cache, v_cache, cache_start, cache_valid, *, scale=None):
    return reference._kvcache_attention(
        _graph_safe_attention, q, k, v, k_cache, v_cache, cache_start, cache_valid, scale=scale)


@register("prefix_kvcache_attention", backend="ascend", requires=torch, priority=0)
def prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, *, flash_mask=None, scale=None):
    return reference._prefix_kvcache_attention(
        _graph_safe_attention, q, k_cache, v_cache, prefix_len, new_pos, flash_mask=flash_mask, scale=scale)


# ---------------------------------------------------------------- Wall-X 2B attention family
# These three compose the same reference helpers, with the graph-safe primitive injected, so the 2B stages keep
# working under NPU graph capture when no native entry serves the layout.


@register("segment_attention", backend="ascend", requires=torch, priority=0)
def segment_attention(q, k, v, cu_seqlens, *, scale=None):
    return reference._varlen_attention(_graph_safe_attention, q, k, v, cu_seqlens, scale=scale)


@register("gated_causal_attention", backend="ascend", requires=torch, priority=0)
def gated_causal_attention(q, k, v, gate, *, scale=None):
    t = q.shape[0]
    # Packed [T, H, D] rows become the [B, H, L, D] the primitive takes, with the causal flag rather than an
    # explicit mask: the same composition the reference uses, only the SDPA primitive differs.
    out = reference._attention(_graph_safe_attention, q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                               v.transpose(0, 1)[None], scale=scale, causal=True)
    return reference.gate_output(out[0].transpose(0, 1).reshape(t, -1), gate)


@register("gated_kvcache_attention", backend="ascend", requires=torch, priority=0)
def gated_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, gate, *, flash_mask=None, scale=None):
    out = reference._prefix_kvcache_attention(
        _graph_safe_attention, q, k_cache, v_cache, prefix_len, new_pos, flash_mask=flash_mask, scale=scale)
    return reference.gate_output(out, gate)


@register("gated_delta_net", backend="ascend", requires=torch, priority=0)
def gated_delta_net(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, num_k_heads, num_v_heads,
                    head_k_dim, head_v_dim, eps):
    # The chunked recurrence is plain torch (no SDPA side stream), so the reference is already capture-safe.
    return reference.gated_delta_net(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, num_k_heads,
                                     num_v_heads, head_k_dim, head_v_dim, eps)
