"""Pre-norm grouped-query decoder layer with optional external prompt K/V.

The same layer serves a plain decoder stack and the action expert of a
mixture-of-transformers model, whose queries also attend to the K/V another
stack produced for the prompt (joint attention).

The layer requests the fusable forms of its projections (norm into the QKV
and gate/up input, rotary into the QKV output, residual adds into the o_proj
and down outputs, the gated activation into the gate/up output). Unquantized,
they compose to the eager layer; a quantized method decides what it fuses.
"""

from __future__ import annotations

import torch
from torch import nn

from xinfer.models.layers.basic import GatedMLP, Linear, QKVLinear, RMSNorm
from xinfer.ops import active_ops


class DecoderLayer(nn.Module):
    """Pre-norm grouped-query decoder layer: ``h + attn(norm(h))``, then ``h + mlp(norm(h))``.

    Args:
        hidden: Layer width.
        num_heads: Query heads ``H``.
        num_kv_heads: K/V heads ``KV``.
        head_dim: Head dimension ``D``.
        intermediate: MLP width (one half of the fused gate/up weight).
        eps: RMSNorm epsilon.
        qkv_bias: Whether the QKV projection has a bias.
        group: Quantization group of the projections.

    Attributes:
        input_norm: RMSNorm before attention, fused into the QKV projection's input.
        qkv: ``QKVLinear`` producing rotated ``q``/``k`` and ``v``.
        o_proj: ``Linear`` ``H·D → hidden`` with the residual add requested in its epilogue.
        post_norm: RMSNorm before the MLP, fused into the gate/up input.
        mlp: ``GatedMLP`` with the activation in the gate/up epilogue and the residual in the down epilogue.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate: int,
        eps: float,
        qkv_bias: bool = True,
        group: str | None = None,
    ):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.input_norm = RMSNorm(hidden, eps)
        self.qkv = QKVLinear(hidden, num_heads, num_kv_heads, head_dim, bias=qkv_bias, group=group)
        self.o_proj = Linear(num_heads * head_dim, hidden, bias=False, group=group)
        self.post_norm = RMSNorm(hidden, eps)
        self.mlp = GatedMLP(hidden, intermediate, group=group, act_fused_into="gate_up")
        ops = active_ops()
        self._causal_attention, self._action_attention = ops.causal_attention, ops.action_attention

    def qkv_states(self, h: torch.Tensor, rope):
        """Return rotated ``q [B, L, H, D]``, ``k`` and ``v [B, L, KV, D]``; ``rope`` is a ``RopeTables``.

        Only ``q`` and ``k`` are rotated. Used alone for a last layer whose attention output is not needed.

        Args:
            h: Hidden states ``[B, L, hidden]``.
            rope: ``RopeTables`` for ``[B, L]``.

        Returns:
            ``(q, k, v)``.
        """
        return self.qkv(h, rope, norm=self.input_norm)

    def forward(self, h, rope, *, prompt_kv=None, num_pinned=None, prompt_lengths=None):
        """Return ``(h_out, k, v)``; ``k``/``v [B, L, KV, D]`` are this layer's own rotated K/V.

        Without ``prompt_kv`` the layer runs causal self-attention. With
        ``prompt_kv = (k, v)`` (``[B, KV, P, D]``, computed elsewhere) the
        queries attend to the prompt and to their own tokens
        (``action_attention``; ``num_pinned [B]`` makes rows ``< num_pinned`` a closed
        group, the RTC semantics; ``prompt_lengths [B]`` marks left-padded
        prompts in a batch of different prompt lengths).

        Args:
            h: Hidden states ``[B, L, hidden]``.
            rope: ``RopeTables`` for ``[B, L]``.
            prompt_kv: ``None`` or ``(k, v)`` of the prompt, each ``[B, KV, P, D]``.
            num_pinned: ``[B]`` int64 pinned-row counts (``None``: zeros); only with ``prompt_kv``.
            prompt_lengths: ``[B]`` own prompt length of each row, or ``None`` (all ``P``); only with
                ``prompt_kv``.

        Returns:
            ``(h_out [B, L, hidden], k, v)``; ``h_out`` has the dtype of ``h``.
        """
        q, k, v = self.qkv_states(h, rope)
        if prompt_kv is None:
            o = self._causal_attention(q, k, v)
        else:
            if num_pinned is None:
                num_pinned = torch.zeros(h.shape[0], dtype=torch.int64, device=h.device)
            o = self._action_attention(q, k, v, prompt_kv[0], prompt_kv[1], num_pinned, prompt_lengths)
        b, n = o.shape[:2]
        h = self.o_proj(o.reshape(b, n, -1), residual=h)
        h = self.mlp(h, norm=self.post_norm, residual=h)
        return h, k, v
