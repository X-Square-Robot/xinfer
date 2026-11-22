"""Qwen3.5 hybrid decoder layers and the adaptive-norm action expert layer used by WALL-OSS-1.0-Nano.

The prompt stack alternates Gated DeltaNet layers (linear attention with a short causal convolution) and
gated full-attention layers (``out · sigmoid(gate)``, per-head q/k RMSNorm, partial rotary). The action expert
runs the same layer pattern at a smaller width: its linear-attention layers have no mixer (the normed input
passes through), its full-attention layers attend to the prompt K/V and the action tokens, and its norms are
adaptive (scale, shift and gate from the flow-time condition).

Numerics follow the Wall-X model under BF16 autocast. Qwen3.5 RMSNorms are zero-centered (``x̂ · (1 + w)``);
the layers keep ``1 + w`` as their weight, which the weight mapping computes once in FP32, the same value the
eager norm forms on every call.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from xinfer.models.layers.basic import GatedMLP, Linear, RMSNorm, _meta_param
from xinfer.ops import active_ops


def prompt_keep_mask(valid_lengths: torch.Tensor, width: int, dtype: torch.dtype) -> torch.Tensor:
    """Return a ``[B, width, 1]`` mask: 1 for valid leading prompt tokens, 0 for padding."""
    positions = torch.arange(width, device=valid_lengths.device)
    return (positions[None] < valid_lengths.reshape(-1, 1))[..., None].to(dtype)


def partial_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotate) -> torch.Tensor:
    """Rotate the first ``cos.shape[-1]`` channels of every head and keep the rest.

    Args:
        x: ``[B, L, heads, D]``.
        cos: ``[B, L, 1, R]`` FP32 table of the rotary channels (``R <= D``).
        sin: ``[B, L, 1, R]`` FP32.
        rotate: the ``rotary_embedding`` operator.

    Returns:
        ``[B, L, heads, D]`` in ``x.dtype``.
    """
    width = cos.shape[-1]
    return torch.cat([rotate(x[..., :width], cos, sin), x[..., width:]], dim=-1)


class GatedDeltaNet(nn.Module):
    """Qwen3.5 linear-attention mixer over a whole prompt from a zero state.

    ``in_qkv`` → depthwise causal convolution + SiLU → q, k, v heads (q and k repeated to the value heads);
    ``beta = sigmoid(in_b(x))``, ``g = -exp(A_log) · softplus(in_a(x) + dt_bias)`` in FP32; the chunked gated
    delta rule; a per-head gated RMSNorm with ``silu(in_z(x))``; ``out_proj``.

    Args:
        hidden: Layer width.
        num_k_heads: Query/key heads.
        num_v_heads: Value heads (a multiple of ``num_k_heads``).
        head_k_dim: Query/key head dimension.
        head_v_dim: Value head dimension.
        conv_kernel: Short convolution length.
        eps: Epsilon of the output norm.
        group: Quantization group of the projections.
    """

    def __init__(self, hidden: int, num_k_heads: int, num_v_heads: int, head_k_dim: int, head_v_dim: int,
                 conv_kernel: int, eps: float, group: str | None = None):
        super().__init__()
        self.num_k_heads, self.num_v_heads = num_k_heads, num_v_heads
        self.head_k_dim, self.head_v_dim, self.eps = head_k_dim, head_v_dim, eps
        self.key_dim, self.value_dim = num_k_heads * head_k_dim, num_v_heads * head_v_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        self.in_qkv = Linear(hidden, conv_dim, group=group)
        self.in_z = Linear(hidden, self.value_dim, group=group)
        self.in_b = Linear(hidden, num_v_heads, group=group)
        self.in_a = Linear(hidden, num_v_heads, group=group)
        self.conv_weight = _meta_param(conv_dim, 1, conv_kernel)
        self.A_log = _meta_param(num_v_heads)
        self.dt_bias = _meta_param(num_v_heads)
        self.norm_weight = _meta_param(head_v_dim)
        self.out_proj = Linear(self.value_dim, hidden, group=group)
        ops = active_ops()
        self._conv, self._delta_rule, self._gated_norm = ops.causal_conv1d, ops.chunk_gated_delta_rule, \
            ops.gated_rms_norm

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """``x [B, L, hidden]`` (normed) → ``[B, L, hidden]``.

        Args:
            x: Normed hidden states.
            keep: ``[B, L, 1]`` rows that take part in the scan (padding rows are zeroed first, as the source
                model does; rows after the last kept row do not affect the kept rows).

        Returns:
            The mixer output, ``x.dtype``.
        """
        b, n, _ = x.shape
        x = x * keep
        mixed = self._conv(self.in_qkv(x).transpose(1, 2), self.conv_weight).transpose(1, 2)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, n, -1, self.head_k_dim)
        k = k.reshape(b, n, -1, self.head_k_dim)
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            # As the source: with equal head counts q and k stay views of the convolution output, and that
            # layout decides the bits of the reference L2 norm's reduction (torch 2.10).
            q, k = q.repeat_interleave(repeat, dim=2), k.repeat_interleave(repeat, dim=2)
        v = v.reshape(b, n, -1, self.head_v_dim)
        beta = self.in_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_a(x).float() + self.dt_bias)
        o = self._delta_rule(q, k, v, g, beta)
        z = self.in_z(x).reshape(-1, self.head_v_dim)
        o = self._gated_norm(o.reshape(-1, self.head_v_dim), z, self.norm_weight, self.eps)
        return self.out_proj(o.reshape(b, n, -1))


class GatedAttention(nn.Module):
    """Qwen3.5 full-attention projections: ``q_proj`` gives ``[query | gate]`` per head; q/k RMSNorm; rotary.

    ``q_proj``, ``k_proj`` and ``v_proj`` stay separate GEMMs, as in the source model. The attention itself is
    left to the caller (causal over the prompt, or joint with the prompt K/V for the action expert).

    Args:
        hidden: Layer width.
        num_heads: Query heads ``H``.
        num_kv_heads: K/V heads ``KV``.
        head_dim: Head dimension ``D``.
        eps: q/k norm epsilon.
        group: Quantization group of the projections.
    """

    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float,
                 group: str | None = None):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.q_proj = Linear(hidden, 2 * num_heads * head_dim, group=group)
        self.k_proj = Linear(hidden, num_kv_heads * head_dim, group=group)
        self.v_proj = Linear(hidden, num_kv_heads * head_dim, group=group)
        self.o_proj = Linear(num_heads * head_dim, hidden, group=group)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self._rotary = active_ops().rotary_embedding

    def qkv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Return ``q [B, L, H, D]``, ``k``/``v [B, L, KV, D]`` (q and k normed and rotated) and ``gate [B, L, H·D]``.

        Args:
            x: ``[B, L, hidden]`` normed hidden states.
            cos: ``[B, L, 1, R]`` FP32 rotary table of the first ``R`` channels.
            sin: ``[B, L, 1, R]`` FP32.
        """
        b, n, _ = x.shape
        d = self.head_dim
        qg = self.q_proj(x).reshape(b, n, self.num_heads, 2 * d)
        q = self.q_norm(qg[..., :d].contiguous())
        gate = qg[..., d:].reshape(b, n, self.num_heads * d)
        k = self.k_norm(self.k_proj(x).reshape(b, n, self.num_kv_heads, d).contiguous())
        v = self.v_proj(x).reshape(b, n, self.num_kv_heads, d).contiguous()
        return partial_rotary(q, cos, sin, self._rotary), partial_rotary(k, cos, sin, self._rotary), v, gate

    def output(self, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``o_proj(o · sigmoid(gate))`` for the attention output ``o [B, L, H, D]``."""
        b, n = o.shape[:2]
        return self.o_proj(o.reshape(b, n, -1) * torch.sigmoid(gate))


class PrefillLayer(nn.Module):
    """One prompt-stack layer: ``h + mixer(norm(h))``, then ``h + mlp(norm(h))``.

    Args:
        linear: Gated DeltaNet (True) or gated full attention (False).
        hidden: Layer width.
        intermediate: MLP width.
        attention: ``(num_heads, num_kv_heads, head_dim)`` of the full-attention layers.
        linear_heads: ``(num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel)`` of the DeltaNet.
        eps: RMSNorm epsilon.
        group: Quantization group.
    """

    def __init__(self, linear: bool, hidden: int, intermediate: int, attention, linear_heads, eps: float,
                 group: str | None = None):
        super().__init__()
        self.linear = linear
        self.input_norm = RMSNorm(hidden, eps)
        self.post_norm = RMSNorm(hidden, eps)
        if linear:
            self.mixer = GatedDeltaNet(hidden, *linear_heads, eps=eps, group=group)
        else:
            self.mixer = GatedAttention(hidden, *attention, eps=eps, group=group)
        self.mlp = GatedMLP(hidden, intermediate, group=group, act_fused_into="down")
        self._causal_attention = active_ops().causal_attention

    def forward(self, h, cos, sin, keep, kv_only: bool = False):
        """Return ``(h_out, k, v)``; ``k``/``v [B, L, KV, D]`` for full-attention layers, else ``None``.

        Args:
            h: ``[B, L, hidden]`` hidden states.
            cos: ``[B, L, 1, R]`` FP32 rotary table.
            sin: ``[B, L, 1, R]`` FP32.
            keep: ``[B, L, 1]`` non-padding rows (DeltaNet input mask).
            kv_only: For a full-attention layer, return after computing its K/V (``h_out`` is ``None``).
        """
        x = self.input_norm(h)
        if self.linear:
            h = h + self.mixer(x, keep)
            k = v = None
        else:
            q, k, v, gate = self.mixer.qkv(x, cos, sin)
            if kv_only:
                return None, k, v
            h = h + self.mixer.output(self._causal_attention(q, k, v), gate)
        h = h + self.mlp(self.post_norm(h))
        return h, k, v


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm modulated by a condition: ``(scale, shift, gate) = dense(cond)``; returns the normed input and gate.

    Args:
        dim: Normalized width.
        cond_dim: Condition width.
        eps: Variance epsilon.
        group: Quantization group of ``dense``.
    """

    def __init__(self, dim: int, cond_dim: int, eps: float, group: str | None = None):
        super().__init__()
        self.eps = eps
        self.dense = Linear(cond_dim, 3 * dim, bias=True, group=group)
        self._ada_rms_norm = active_ops().ada_rms_norm

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """``x [B, L, dim]``, ``cond [B, L, cond_dim]`` (``x.dtype``) → ``(normed, gate)``, both ``x.dtype``."""
        scale, shift, gate = self.dense(cond).chunk(3, dim=-1)
        return self._ada_rms_norm(x, scale, shift, self.eps), gate


class ActionLayer(nn.Module):
    """One action-expert layer with adaptive norms and gated residuals.

    Linear-attention layers have no mixer: ``a + norm(a) · gate``. Full-attention layers attend to the prompt
    K/V of the same layer and to the action tokens (``masked_joint_attention``). Then ``a + mlp(norm(a)) · gate``.

    Args:
        linear: Linear-attention position (identity mixer) or full attention.
        hidden: Action expert width.
        intermediate: Action MLP width.
        attention: ``(num_heads, num_kv_heads, head_dim)``.
        cond_dim: AdaRMS condition width.
        eps: Norm epsilon.
        group: Quantization group.
    """

    def __init__(self, linear: bool, hidden: int, intermediate: int, attention, cond_dim: int, eps: float,
                 group: str | None = None):
        super().__init__()
        self.linear = linear
        self.input_norm = AdaptiveRMSNorm(hidden, cond_dim, eps, group=group)
        self.post_norm = AdaptiveRMSNorm(hidden, cond_dim, eps, group=group)
        if not linear:
            self.attn = GatedAttention(hidden, *attention, eps=eps, group=group)
        self.mlp = GatedMLP(hidden, intermediate, group=group, act_fused_into="down")
        ops = active_ops()
        self._joint_attention, self._gated_residual = ops.masked_joint_attention, ops.gated_residual

    def forward(self, a, cond, cos, sin, prompt_kv, mask):
        """Run the layer over the action tokens.

        Args:
            a: ``[B, L, hidden]`` action hidden states.
            cond: ``[B, L, cond_dim]`` condition in ``a.dtype``.
            cos: ``[B, L, 1, R]`` FP32 rotary table of the action positions.
            sin: ``[B, L, 1, R]`` FP32.
            prompt_kv: ``(k, v)`` of this layer's prompt, each ``[B, KV, P, D]``; unused by linear layers.
            mask: ``[B, 1, L, P + L]`` visibility of prompt and action keys.

        Returns:
            ``[B, L, hidden]`` in ``a.dtype``.
        """
        x, gate = self.input_norm(a, cond)
        if self.linear:
            a = self._gated_residual(a, x, gate)
        else:
            q, k, v, attn_gate = self.attn.qkv(x, cos, sin)
            o = self._joint_attention(q, k, v, prompt_kv[0], prompt_kv[1], mask)
            a = self._gated_residual(a, self.attn.output(o, attn_gate), gate)
        x, gate = self.post_norm(a, cond)
        return self._gated_residual(a, self.mlp(x), gate)
