"""The ViT and prompt stream of the Wall-X Qwen3.5 models on xcompute (``numerics="optimized"``).

The Wall-X 2B stages use one operator per fusion from ``xinfer/ops/reference/fused.py``.
xcompute selects the kernel for each shape. GEMM weights are ``[N, K]``. Differences to the
reference composition (``numerics="reference"``, the wall-x model under autocast):

- ViT: every weight in BF16, including the position table, which one kernel interpolates; the patch embedding is a
  GEMM; the LayerNorms run ``layer_norm_cast``; QKV, bias and RoPE are one GEMM with the rotation on the FP32
  accumulator; the bias and residual add of the projection and fc2, the GELUs of fc1 and the merger are GEMM
  epilogues; attention runs ``segment_attention``;
- prompt stream: the DeltaNet input projections ``[qkv | z | b | a]`` are one GEMM that zeroes the padding rows (the
  padding follows every real token, so real tokens do not see it); the DeltaNet mixer from the convolution to the
  gated norm is one call (``gated_delta_net``); ``[Q·gate | K | V]`` is one GEMM, the q/k norms and split one
  kernel and RoPE another; attention and its output gate are one call; the MLP's SiLU-mul is the gate/up GEMM's
  epilogue; the MLP's down projection stays torch's GEMM (xcompute lists no kernel for it, and its cuBLASLt
  fallback with a fixed workspace is slower there).
"""

from __future__ import annotations

import torch
from torch import nn

from xinfer.models.layers.basic import ParamGroup, _meta_param
from xinfer.loader import Source
from xinfer.ops import active_ops

DTYPE = torch.bfloat16
VISION_STAGE_OPS = ("gemm_epilogue", "qkv_rope_epilogue", "layer_norm_cast", "pos_embed_interp", "rot_pos",
                    "segment_attention")
PREFILL_STAGE_OPS = ("gemm_epilogue", "gemm_glu_epilogue", "linear", "zero_centered_rms_norm",
                     "add_zero_centered_rms_norm", "qkv_norm_split", "rope_full", "gated_causal_attention",
                     "gated_delta_net")


class OptimizedVision(nn.Module):
    """The ViT and patch merger (BF16 weights, ``[out, in]`` as in the checkpoint).

    Args:
        config: ``Qwen35VisionConfig`` of the checkpoint.
    """

    def __init__(self, config):
        super().__init__()
        self.config = c = config
        h, i = c.hidden_size, c.intermediate_size
        self.patch = ParamGroup(weight=(h, c.in_features), bias=(h,))
        self.pos_embed = _meta_param(c.num_position_embeddings, h)
        self.blocks = nn.ModuleList(ParamGroup(norm1_weight=(h,), norm1_bias=(h,), qkv_weight=(3 * h, h),
                                               qkv_bias=(3 * h,), proj_weight=(h, h), proj_bias=(h,),
                                               norm2_weight=(h,), norm2_bias=(h,), fc1_weight=(i, h),
                                               fc1_bias=(i,), fc2_weight=(h, i), fc2_bias=(h,))
                                    for _ in range(c.depth))
        width = h * c.spatial_merge_size ** 2
        self.merger = ParamGroup(norm_weight=(h,), norm_bias=(h,), fc1_weight=(width, width), fc1_bias=(width,),
                                 fc2_weight=(c.out_hidden_size, width), fc2_bias=(c.out_hidden_size,))
        half = c.head_dim // 2
        self.register_buffer("inv_freq", 1.0 / (c.rope_theta ** (torch.arange(0, half, 2, dtype=torch.float) / half)),
                             persistent=False)
        self.ops = active_ops()

    def forward(self, pixel_values, grids, cu_seqlens):
        """Image token embeddings of packed patches.

        Args:
            pixel_values: ``[N, in_features]`` FP32 patch rows in merge-block order.
            grids: Host list of ``(t, h, w)`` per image.
            cu_seqlens: Host list of frame boundaries (attention stays within a frame).

        Returns:
            ``[N / merge², out_hidden_size]`` BF16.
        """
        c, ops = self.config, self.ops
        n, h, heads, d = pixel_values.shape[0], c.hidden_size, c.num_heads, c.head_dim
        x = ops.gemm_epilogue(pixel_values.to(DTYPE), self.patch.weight, self.patch.bias)
        x = x + ops.pos_embed_interp(self.pos_embed, grids, c.spatial_merge_size)
        angles = ops.rot_pos(self.inv_freq, grids, c.spatial_merge_size)
        cos, sin = angles.cos(), angles.sin()
        for block in self.blocks:
            normed = ops.layer_norm_cast(x, block.norm1_weight, block.norm1_bias, c.eps)
            qkv = ops.qkv_rope_epilogue(normed, block.qkv_weight, block.qkv_bias, cos, sin, heads, d)
            q, k, v = qkv.view(n, 3, heads, d).unbind(1)
            o = ops.segment_attention(q, k, v, cu_seqlens, scale=d ** -0.5).reshape(n, h)
            x = ops.gemm_epilogue(o, block.proj_weight, block.proj_bias, residual=x)
            normed = ops.layer_norm_cast(x, block.norm2_weight, block.norm2_bias, c.eps)
            hidden = ops.gemm_epilogue(normed, block.fc1_weight, block.fc1_bias, activation="gelu_tanh")
            x = ops.gemm_epilogue(hidden, block.fc2_weight, block.fc2_bias, residual=x)
        merged = ops.layer_norm_cast(x, self.merger.norm_weight, self.merger.norm_bias, c.eps)
        merged = ops.gemm_epilogue(merged.reshape(-1, self.merger.fc1_weight.shape[1]), self.merger.fc1_weight,
                                   self.merger.fc1_bias, activation="gelu")
        return ops.gemm_epilogue(merged, self.merger.fc2_weight, self.merger.fc2_bias)


class OptimizedPrefill(nn.Module):
    """The prompt stream up to the last full-attention layer's K/V (``[N, K]`` GEMM weights).

    Args:
        config: The checkpoint's text configuration: ``hidden_size``, ``intermediate_size``, the attention and
            DeltaNet geometry (``num_attention_heads``, ``num_key_value_heads``, ``head_dim``, ``rotary_dim``,
            ``linear_heads``), ``layer_types``, ``full_attention_layers`` and ``rms_norm_eps``.
    """

    def __init__(self, config):
        super().__init__()
        self.config = c = config
        hk, hv, dk, dv, kernel = c.linear_heads
        d, heads, kv = c.hidden_size, c.num_attention_heads, c.num_key_value_heads
        conv = 2 * hk * dk + hv * dv
        self.layers = nn.ModuleList()
        for kind in c.layer_types[:c.full_attention_layers[-1] + 1]:
            shapes = dict(input_norm=(d,), post_norm=(d,), gate_up=(2 * c.intermediate_size, d),
                          down=(d, c.intermediate_size))
            if kind == "full_attention":
                shapes.update(qkv=((2 * heads + 2 * kv) * c.head_dim, d), q_norm=(c.head_dim,),
                              k_norm=(c.head_dim,), o=(d, heads * c.head_dim))
            else:
                shapes.update(qkv_zba=(conv + hv * dv + 2 * hv, d), conv_weight=(conv, kernel), A_log=(hv,),
                              dt_bias=(hv,), norm_weight=(dv,), out=(d, hv * dv))
            self.layers.append(ParamGroup(**shapes))
        self.ops = active_ops()

    def forward(self, h, cos, sin, prompt_kv, keep):
        """Run the prompt stream and write the post-RoPE K and the V of every full-attention layer.

        Args:
            h: ``[1, P, hidden]`` BF16 prompt embeddings (padding included).
            cos: FP32 ``[1, P, rotary_dim / 2]`` compact cosines.
            sin: FP32 ``[1, P, rotary_dim / 2]`` compact sines.
            prompt_kv: ``[full-attention layers, 2, 1, kv_heads, P, head_dim]`` output.
            keep: FP32 ``[P]``, 1 for the real tokens and 0 for the padding.
        """
        c, ops = self.config, self.ops
        hk, hv, dk, dv, _ = c.linear_heads
        heads, kv_heads, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        eps, full = c.rms_norm_eps, c.full_attention_layers
        p, conv = h.shape[1], 2 * hk * dk + hv * dv
        normed = ops.zero_centered_rms_norm(h, self.layers[0].input_norm, eps)
        for i, layer in enumerate(self.layers):
            flat = normed.view(p, -1)
            if c.layer_types[i] == "full_attention":
                packed = ops.gemm_epilogue(flat, layer.qkv).view(1, p, -1)
                q, k, v, gate = ops.qkv_norm_split(packed, layer.q_norm, layer.k_norm, heads, kv_heads, d, eps)
                q, k = ops.rope_full(q, k, cos, sin, c.rotary_dim)
                slot = full.index(i)
                prompt_kv[slot, 0].copy_(k.transpose(1, 2))
                prompt_kv[slot, 1].copy_(v.transpose(1, 2))
                if i == full[-1]:
                    return
                o = ops.gated_causal_attention(q[0], k[0], v[0], gate.reshape(p, -1), scale=d ** -0.5)
                mixed = ops.gemm_epilogue(o, layer.o)
            else:
                qkv_zba = ops.gemm_epilogue(flat, layer.qkv_zba, row_scale=keep)
                o = ops.gated_delta_net(qkv_zba[:, :conv].view(1, p, -1), qkv_zba[:, conv:].view(1, p, -1),
                                        layer.conv_weight, layer.A_log, layer.dt_bias, layer.norm_weight, hk, hv,
                                        dk, dv, eps)
                mixed = ops.gemm_epilogue(o.view(p, -1), layer.out)
            h, normed = ops.add_zero_centered_rms_norm(mixed.view(1, p, -1), h, layer.post_norm, eps)
            hidden = ops.gemm_glu_epilogue(normed.view(p, -1), layer.gate_up)
            # torch's GEMM: no xcompute kernel is listed for this shape, and its cuBLASLt fallback is slower here
            h, normed = ops.add_zero_centered_rms_norm(ops.linear(hidden, layer.down).view(1, p, -1), h,
                                                       self.layers[i + 1].input_norm, eps)


def column_ranges(mask: torch.Tensor) -> torch.Tensor:
    """FlashMask ``(LTS, UTE)`` per key column of a boolean ``[1, 1, M, L]`` mask whose visible rows per column are
    contiguous: column ``c`` is visible to rows ``UTE <= t < LTS`` (``0, 0`` when no row sees it)."""
    vis = mask.reshape(mask.shape[-2], mask.shape[-1])
    m = vis.shape[0]
    seen = vis.any(0)
    first = vis.to(torch.int32).argmax(0)
    last = m - 1 - vis.flip(0).to(torch.int32).argmax(0)
    lts = torch.where(seen, last + 1, torch.zeros_like(last))
    ute = torch.where(seen, first, torch.zeros_like(first))
    return torch.stack((lts, ute), dim=-1).to(torch.int32)[None, None].contiguous()


def q_k_v_gate(q_gate: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    """Rows ``[q_h | gate_h]`` per head, ``k``, ``v`` → ``[q | k | v | gate]``."""
    pairs = q_gate.view(heads, 2, -1, q_gate.shape[-1])
    return torch.cat((pairs[:, 0].flatten(0, 1), k, v, pairs[:, 1].flatten(0, 1))).contiguous()


def vision_prompt_weight_mapping(vision_depth: int, layer_types, prompt_layers: int) -> dict[str, Source]:
    """Checkpoint key → tensor of ``OptimizedVision`` (``vision.*``), ``OptimizedPrefill`` (``prefill_stack.*``)
    and the token embedding, in their dtypes and orientations.

    The checkpoint is FP32. The ViT keeps the checkpoint's ``[out, in]`` weights, all BF16. The prompt stream takes
    ``[N, K]`` BF16 GEMM weights (``[qkv | z | b | a]`` and ``[q·gate | k | v]`` concatenated); its norms are FP32
    and raw, the q/k norms BF16 and raw, ``A_log`` and ``dt_bias`` their BF16 rounding kept in FP32.

    Args:
        vision_depth: ViT blocks.
        layer_types: ``"linear_attention"`` / ``"full_attention"`` per text layer.
        prompt_layers: Prompt-stream layers run (up to the last full-attention layer).

    Returns:
        ``{model parameter name: Source}``.
    """
    bf16, fp32 = torch.bfloat16, torch.float32
    lm, vis = "model.language_model", "model.visual"
    one = lambda key: Source.one(key, bf16)  # noqa: E731
    cat = lambda *keys: Source(tuple(keys), bf16, combine=lambda w: torch.cat(w, 0).contiguous())  # noqa: E731
    mapping = {
        "embed_tokens": one(f"{lm}.embed_tokens.weight"),
        "vision.patch.weight": Source((f"{vis}.patch_embed.proj.weight",), bf16,
                                      combine=lambda w: w[0].reshape(w[0].shape[0], -1)),
        "vision.patch.bias": one(f"{vis}.patch_embed.proj.bias"),
        "vision.pos_embed": one(f"{vis}.pos_embed.weight"),
    }
    for name, key in (("norm_weight", "norm.weight"), ("norm_bias", "norm.bias"), ("fc1_weight", "linear_fc1.weight"),
                      ("fc1_bias", "linear_fc1.bias"), ("fc2_weight", "linear_fc2.weight"),
                      ("fc2_bias", "linear_fc2.bias")):
        mapping[f"vision.merger.{name}"] = one(f"{vis}.merger.{key}")
    for i in range(vision_depth):
        source, target = f"{vis}.blocks.{i}", f"vision.blocks.{i}"
        for name, key in (("norm1", "norm1"), ("qkv", "attn.qkv"), ("proj", "attn.proj"), ("norm2", "norm2"),
                          ("fc1", "mlp.linear_fc1"), ("fc2", "mlp.linear_fc2")):
            for part in ("weight", "bias"):
                mapping[f"{target}.{name}_{part}"] = one(f"{source}.{key}.{part}")
    for i, kind in enumerate(layer_types[:prompt_layers]):
        source, prompt = f"{lm}.layers.{i}", f"prefill_stack.layers.{i}"
        mapping[f"{prompt}.input_norm"] = Source.one(f"{source}.input_layernorm.weight", fp32)
        mapping[f"{prompt}.post_norm"] = Source.one(f"{source}.post_attention_layernorm.weight", fp32)
        mapping[f"{prompt}.gate_up"] = one(f"{source}.mlp.gate_up_proj.weight")
        mapping[f"{prompt}.down"] = one(f"{source}.mlp.down_proj.weight")
        if kind == "full_attention":
            attn = f"{source}.self_attn"
            mapping[f"{prompt}.qkv"] = cat(f"{attn}.q_proj.weight", f"{attn}.k_proj.weight", f"{attn}.v_proj.weight")
            mapping[f"{prompt}.q_norm"] = one(f"{attn}.q_norm.weight")
            mapping[f"{prompt}.k_norm"] = one(f"{attn}.k_norm.weight")
            mapping[f"{prompt}.o"] = one(f"{attn}.o_proj.weight")
        else:
            linear = f"{source}.linear_attn"
            mapping[f"{prompt}.qkv_zba"] = cat(f"{linear}.in_proj_qkv.weight", f"{linear}.in_proj_z.weight",
                                               f"{linear}.in_proj_b.weight", f"{linear}.in_proj_a.weight")
            mapping[f"{prompt}.conv_weight"] = Source((f"{linear}.conv1d.weight",), bf16,
                                                      combine=lambda w: w[0].squeeze(1).contiguous())
            for name in ("A_log", "dt_bias"):
                mapping[f"{prompt}.{name}"] = Source((f"{linear}.{name}",), fp32,
                                                     combine=lambda w: w[0].to(bf16).to(fp32))
            mapping[f"{prompt}.norm_weight"] = one(f"{linear}.norm.weight")
            mapping[f"{prompt}.out"] = one(f"{linear}.out_proj.weight")
    return mapping
