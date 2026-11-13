"""Wall-OSS on the CUDA backend's kernels (``numerics="optimized"``, the default), BF16.

Each stage runs the steps of the reference composition (``numerics="reference"``, the Wall-X eager model) on
operators of ``xinfer/ops/reference/`` (each a kernel call where the CUDA backend lists a kernel, else its torch
reference). GEMM weights are ``[N, K]``; the action stage's output and down projections are stored ``[K, N]``,
which selects the small-M kernel. On the Ascend provider the ``[N, K]`` projections are read as given (a
transposed view would cost the v22 CANN route a full materialization per call) and the action loop feeds its
projections the attention output's native BNSD form. Differences to the reference:

- every residual add and the RMSNorm after it are one call (``add_rms_norm``), the sum rounded to BF16 as before;
  in the ViT the projection before it joins them (``gemm_add_rms_norm``), which on integrated GPUs adds the residual
  to the FP32 product (one rounding);
- the QKV projections add their bias and rotate q and k on the FP32 accumulator (one rounding); in the prefill and
  action stages the same call writes K and V to the head-major cache;
- the gated MLPs apply SiLU-mul to the FP32 accumulator (one rounding); the merger's GELU is the fc1 epilogue;
- the action stage attends to the prompt K/V and its own keys from one head-major cache, with prompt padding and
  the RTC visibility as per-column row ranges, and reads out, blends the uncommanded dimensions and steps the
  actions in one call; one request per dispatch.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from xinfer.models.layers.basic import ParamGroup
from xinfer.models.layers.rotary import inverse_frequencies, mrope_cos_sin, vision_rotary_angles
from xinfer.loader import Source
from xinfer.ops.backends.copykill_r2 import enabled as _copykill_enabled
from xinfer.ops import active_ops

DTYPE = torch.bfloat16
FLOW_TIME_SCALE = 0.999

OPTIMIZED_STAGE_OPS = {
    "vision": ("gemm_epilogue", "gemm_glu_epilogue", "gemm_add_rms_norm", "qkv_rope_epilogue", "rms_norm",
               "varlen_attention"),
    "prefill": ("gemm_epilogue", "qkv_rope_kvcache", "mlp_add_rms_norm", "rms_norm", "add_rms_norm",
                "causal_attention"),
    "action": ("gemm_epilogue", "gemm_glu_epilogue", "qkv_rope_kvcache", "prefix_kvcache_attention",
               "rms_norm", "add_rms_norm", "flow_velocity_euler", "linear", "prefetch_weights", "join_prefetch"),
}


def compact_rope(inv_freq: torch.Tensor, positions: torch.Tensor, section) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 ``(cos, sin)`` ``[L, D/2]`` of M-RoPE positions ``[3, 1, L]``, rounded through BF16 and band-selected as
    the reference's ``RopeTables.rotate`` uses them."""
    cos, sin = mrope_cos_sin(inv_freq, positions, torch.float32)
    tables = []
    for table in (cos, sin):
        half = table.to(DTYPE).float()[:, 0, :, :table.shape[-1] // 2]  # [3, L, D/2]
        parts = half.split(list(section), dim=-1)
        tables.append(torch.cat([part[i % 3] for i, part in enumerate(parts)], dim=-1).contiguous())
    return tables[0], tables[1]


class OptimizedVision(nn.Module):
    """The Qwen2.5-VL ViT and patch merger (BF16 weights, ``[out, in]`` as in the checkpoint).

    Args:
        config: ``VisionTowerConfig`` of the checkpoint.
    """

    def __init__(self, config):
        super().__init__()
        self.config = c = config
        h, i = c.hidden_size, c.mlp_size
        self.patch = ParamGroup(weight=(h, c.in_features))
        self.blocks = nn.ModuleList(ParamGroup(norm1=(h,), qkv_weight=(3 * h, h), qkv_bias=(3 * h,),
                                               proj_weight=(h, h), proj_bias=(h,), norm2=(h,),
                                               gate_up_weight=(2 * i, h), gate_up_bias=(2 * i,),
                                               down_weight=(h, i), down_bias=(h,))
                                    for _ in range(c.depth))
        width = h * c.spatial_merge_size ** 2
        self.merger = ParamGroup(norm=(h,), fc1_weight=(width, width), fc1_bias=(width,),
                                 fc2_weight=(c.out_hidden_size, width), fc2_bias=(c.out_hidden_size,))
        half = c.head_dim // 2
        self.register_buffer("inv_freq", 1.0 / (c.rope_theta ** (torch.arange(0, half, 2, dtype=torch.float) / half)),
                             persistent=False)
        self.ops = active_ops()

    def forward(self, pixel_values, layout):
        """Image token embeddings (see ``VisionTower.forward``; the same layout and output order).

        Args:
            pixel_values: ``[N, in_features]`` FP32 patch rows.
            layout: ``vision_layout`` of the images.

        Returns:
            ``[N / merge², out_hidden_size]`` BF16.
        """
        c, ops = self.config, self.ops
        unit, eps, heads, d = c.spatial_merge_size ** 2, c.eps, c.num_heads, c.head_dim
        x = ops.gemm_epilogue(pixel_values.to(DTYPE), self.patch.weight)
        n = x.shape[0]
        window_index = layout["window_index"]
        angles = vision_rotary_angles(self.inv_freq, layout["positions"])
        x = x.reshape(n // unit, unit, -1)[window_index].reshape(n, -1)
        angles = angles.reshape(n // unit, unit, -1)[window_index].reshape(n, -1)
        cos, sin = angles.cos(), angles.sin()
        normed = ops.rms_norm(x, self.blocks[0].norm1, eps)
        for i, block in enumerate(self.blocks):
            cu_seqlens = layout["cu_full"] if i in c.fullatt_block_indexes else layout["cu_window"]
            qkv = ops.qkv_rope_epilogue(normed, block.qkv_weight, block.qkv_bias, cos, sin, heads, d,
                                        qk_order="any")  # Q and K only feed the attention
            if _copykill_enabled("k3"):
                qkv_split = qkv.view(n, 3, heads, d).permute(1, 0, 2, 3).contiguous()
                q, k, v = qkv_split.unbind(0)
            else:
                q, k, v = qkv.view(n, 3, heads, d).unbind(1)
            o = ops.varlen_attention(q, k, v, cu_seqlens).reshape(n, -1)
            x, normed = ops.gemm_add_rms_norm(o, block.proj_weight, x, block.norm2, eps, bias=block.proj_bias)
            hidden = ops.gemm_glu_epilogue(normed, block.gate_up_weight, bias=block.gate_up_bias)
            following = self.blocks[i + 1].norm1 if i + 1 < len(self.blocks) else self.merger.norm
            x, normed = ops.gemm_add_rms_norm(hidden, block.down_weight, x, following, eps, bias=block.down_bias)
        merged = ops.gemm_epilogue(normed.reshape(-1, self.merger.fc1_weight.shape[1]), self.merger.fc1_weight,
                                   self.merger.fc1_bias, activation="gelu")
        return ops.gemm_epilogue(merged, self.merger.fc2_weight, self.merger.fc2_bias)[layout["reverse_index"]]


def _layer_params(hidden, intermediate, qkv_rows, q_width):
    """Parameters of one decoder layer; the output and down projections stored ``[N, K]``."""
    shapes = dict(input_norm=(hidden,), qkv=(qkv_rows, hidden), qkv_bias=(qkv_rows,), post_norm=(hidden,),
                  gate_up=(2 * intermediate, hidden), o=(hidden, q_width), down=(hidden, intermediate))
    return ParamGroup(**shapes)


class OptimizedPrefill(nn.Module):
    """Expert 0 over the prompt: every layer's post-RoPE K and V.

    Args:
        config: ``WallOSSConfig`` of the checkpoint.
    """

    def __init__(self, config):
        super().__init__()
        self.config = c = config
        expert = c.experts[0]
        q_width = c.num_attention_heads * c.head_dim
        rows = q_width + 2 * c.num_key_value_heads * c.head_dim
        self.layers = nn.ModuleList(_layer_params(expert.hidden_size, expert.intermediate_size, rows, q_width)
                                    for _ in range(c.num_hidden_layers))
        self.ops = active_ops()

    def forward(self, h, cos, sin, prompt_kv):
        """Run expert 0 and write ``prompt_kv`` ``[layers, 2, 1, kv_heads, P, head_dim]``.

        Args:
            h: ``[1, P, hidden]`` BF16 prompt embeddings.
            cos: FP32 ``[P, head_dim / 2]`` compact cosines (``compact_rope``).
            sin: FP32 ``[P, head_dim / 2]``.
            prompt_kv: The output.
        """
        c, ops = self.config, self.ops
        heads, kv_heads, d, eps = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rms_norm_eps
        p = h.shape[1]
        normed = ops.rms_norm(h, self.layers[0].input_norm, eps)
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            keys, values = prompt_kv[i, 0, 0], prompt_kv[i, 1, 0]  # [KV, P, D]
            q = ops.qkv_rope_kvcache(normed.view(p, -1), layer.qkv, layer.qkv_bias, cos, sin, keys, values, 0, heads,
                                     kv_heads, d)
            if i == last:
                return
            # The pinned prefill finish writes Q head-major BNSD [H, P, D]; the strided [1, P, H, D] view
            # turns contiguous inside the attention transpose, so the per-call Q copy disappears. Fallback
            # widths keep the packed view.
            bnsd = q.ndim == 3
            q_arg = q.transpose(0, 1)[None] if bnsd else q.view(1, p, heads, d)
            # With ``return_bnsd`` the pinned causal op hands back its native BNSD [1, H, P, D] buffer and the
            # o-projection reads it directly (the M768 Cube route folds the packed reshape copy and the finish
            # round); fallback widths keep the packed view and the old chain.
            o = ops.causal_attention(q_arg, keys.transpose(0, 1)[None], values.transpose(0, 1)[None],
                                     return_bnsd=bnsd)
            rows = o[0] if bnsd else o.reshape(p, -1)
            h, normed = ops.add_rms_norm(ops.gemm_epilogue(rows, layer.o).view(1, p, -1), h,
                                         layer.post_norm, eps)
            h, normed = ops.mlp_add_rms_norm(normed.view(p, -1), layer.gate_up, layer.down, h,
                                             self.layers[i + 1].input_norm, eps)


class OptimizedAction(nn.Module):
    """Expert 1 over the ``H`` action tokens and the model's FP32 ``ActionHead``; one request per dispatch.

    Args:
        config: ``WallOSSConfig`` of the checkpoint.
        action_head: The model's ``ActionHead``.
    """

    def __init__(self, config, action_head):
        super().__init__()
        self.config = c = config
        expert = c.experts[1]
        q_width = c.num_attention_heads * c.head_dim
        rows = q_width + 2 * c.num_key_value_heads * c.head_dim
        self.layers = nn.ModuleList(_layer_params(expert.hidden_size, expert.intermediate_size, rows, q_width)
                                    for _ in range(c.num_hidden_layers))
        self.final_norm = nn.Parameter(torch.empty(expert.hidden_size, device="meta"), requires_grad=False)
        self.action_head = action_head
        self.register_buffer("inv_freq", inverse_frequencies(c.head_dim, c.rope_theta), persistent=False)
        self.ops = active_ops()
        self._steps: dict[int, list[float]] = {}

    def step_sizes(self, steps: int, device) -> list[float]:
        """``t[k + 1] − t[k]`` of the FP32 flow times ``linspace(0, 1, steps + 1) · 0.999`` on ``device``."""
        if steps not in self._steps:
            times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32) * FLOW_TIME_SCALE
            self._steps[steps] = (times[1:] - times[:-1]).tolist()
        return self._steps[steps]

    def forward(self, x_in, prompt_kv, out):
        """Euler flow over the action tokens (inputs and semantics as ``WallOSS.denoise``, one request).

        Args:
            x_in: The stage inputs.
            prompt_kv: ``[layers, 2, 1, kv_heads, P, head_dim]`` BF16.
            out: ``[1, H, D]`` FP32 normalized actions.
        """
        c, ops = self.config, self.ops
        heads, kv_heads, d, eps = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rms_norm_eps
        horizon, device = c.action_horizon, prompt_kv.device
        p = prompt_kv.shape[-2]
        steps = int(x_in["num_steps"])
        dts = self.step_sizes(steps, device)
        times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32) * FLOW_TIME_SCALE
        rows = torch.arange(horizon, device=device)
        rtc = "rtc_actions" in x_in
        delay = x_in["inference_delay"].reshape(1) if rtc else torch.zeros(1, dtype=torch.int64, device=device)
        pinned = (rows < delay)[None]  # [1, H]
        positions = x_in["action_position_base"].reshape(1, 1) + rows[None]
        if rtc:
            shifted = (rows >= delay) & (delay > 0) & (delay < horizon)
            positions = positions + shifted.to(positions.dtype)[None] * c.rtc_rope_gap
        cos, sin = compact_rope(self.inv_freq, positions[None].expand(3, 1, horizon), c.mrope_section)
        # The host all-keep guarantee (processor-computed: delay == 0 and prompt_valid == p). A host input the
        # stage reads keys the NPU graph, so the per-forward flash-mask build (13.4 us) and the FIA mask read
        # disappear on the flow path; RTC and padded prompts keep the mask.
        all_keep = "mask_all_keep" in x_in and bool(x_in["mask_all_keep"])
        if all_keep:
            flash_mask = None
        else:
            valid = x_in["prompt_valid"].reshape(1) if "prompt_valid" in x_in else torch.full((1,), p, device=device)
            columns = torch.arange(p + horizon, device=device)
            lts = torch.where((columns < valid) | (columns >= p), horizon, 0)
            ute = torch.where(columns >= p + delay, delay, 0)
            flash_mask = torch.stack((lts, ute), dim=-1).to(torch.int32)[None, None].contiguous()
        cache = torch.empty((*prompt_kv.shape[:-2], p + horizon, d), dtype=prompt_kv.dtype, device=device)
        cache[..., :p, :].copy_(prompt_kv)
        start = torch.full((1,), p, dtype=torch.int32, device=device)
        dof = x_in["dof_mask"].to(torch.float32)
        x = x_in["noise"].to(torch.float32)
        prefix = None
        if rtc:
            prefix = x_in["rtc_actions"].to(torch.float32)[:, :horizon]
            if prefix.shape[1] < horizon:  # repeat the last pinned row
                prefix = torch.cat([prefix, prefix[:, -1:].expand(-1, horizon - prefix.shape[1], -1)], dim=1)
            x = torch.where(pinned[..., None], prefix, x)
        dof_mask = padding = None
        if not x_in["dof_mask_all"]:
            dof_mask, padding = dof, x_in["padding_action"].to(torch.float32) - x
        pinned_rows = delay.to(torch.int32) if rtc else None
        last = len(self.layers) - 1
        # The prefix-attention metadata scope: the native cache-attention bridge exposes it, and one open
        # scope per forward lets all of the loop's prefix calls share the dense drop mask (built once from
        # the live flash_mask/prefix_len instead of once per call). The torch reference keeps a null context.
        scope = getattr(ops.prefix_kvcache_attention, "action_scope", nullcontext)
        # The sinusoidal time embedding is a pure function of the step schedule (linspace(0, 1, steps + 1)
        # * FLOW_TIME_SCALE; pinned RTC rows read 1.0), so its rows are precomputed once per (steps, device)
        # and selected per step: sin/cos are elementwise in tau, every row value comes from the same scalar
        # through the same op chain, and the per-step selection only picks rows — bitwise-equal to
        # time_embedding(tau). The per-step arange/exp/sin/cos/cat chain (~8 tiny kernels inside the action
        # graph) collapses to a row broadcast.
        te_key = (steps, device.index if device.index is not None else 0)
        te_cache = getattr(self, "_te_rows", None)
        if te_cache is None or te_key not in te_cache:
            ah = self.action_head
            with torch.no_grad():
                sched = ah.time_embedding(times[:-1]).reshape(steps, 1, -1)
                one = ah.time_embedding(torch.ones_like(times[:1])).reshape(1, 1, -1)
            te_cache = {} if te_cache is None else te_cache
            te_cache[te_key] = (sched, one)
            self._te_rows = te_cache
        te_sched, te_one = self._te_rows[te_key]
        ah = self.action_head
        with scope():
            for step in range(steps):
                tau_emb = (torch.where(pinned[..., None], te_one, te_sched[step]) if rtc
                           else te_sched[step].expand(1, horizon, -1))
                af = ah.w1(torch.cat([x, dof], dim=-1))
                jf = ah.w2(torch.cat([af, tau_emb], dim=-1))
                h = ah.w3(ah._silu(jf)).to(DTYPE)
                normed = ops.rms_norm(h, self.layers[0].input_norm, eps)
                for i, layer in enumerate(self.layers):
                    # this layer's later weights and the next layer's QKV, read while the QKV and attention run
                    following_qkv = self.layers[(i + 1) % len(self.layers)].qkv
                    ops.prefetch_weights((layer.o, layer.gate_up, layer.down, following_qkv))
                    keys, values = cache[i, 0], cache[i, 1]  # [1, KV, P + H, D]
                    q = ops.qkv_rope_kvcache(normed[0], layer.qkv, layer.qkv_bias, cos, sin, keys[0], values[0], p, heads,
                                             kv_heads, d)
                    # The NPU M16 epilogue returns Q already in BNSD [H, M, D] order (ACLNN FIA rejects the
                    # strided-query descriptor, error 561103, so the kernel writes the layout the attention
                    # consumes). The strided [1, M, H, D] view turns contiguous inside the attention
                    # transpose; the reference and CUDA backends return packed [M, H*D].
                    q_arg = q.transpose(0, 1)[None] if q.ndim == 3 else q.view(1, horizon, heads, d)
                    o = ops.prefix_kvcache_attention(q_arg, keys.transpose(1, 2),
                                                     values.transpose(1, 2), start, p, flash_mask=flash_mask,
                                                     all_keep=all_keep, return_bnsd=True)
                    # The BNSD [H, M, D] output feeds the o-projection directly: the pinned CANN route reads
                    # it with the transpose folded into the Cube A-load; every other backend permutes.
                    # The fused gemm+add_rms entries (RMSFUSE RF3 gemm_add_rms_r4, then COPYKILL K2's
                    # gemm_add_rms) take the [H, M, D] operand directly and return the (sum, normalized)
                    # pair; both need the delivered xcompute package and stay absent otherwise, and the
                    # composition below is then exactly the previous chain.
                    _fused_entry = getattr(ops.add_rms_norm, "gemm_add_rms_r4", None)
                    _pair = _fused_entry(o[0], layer.o, h, layer.post_norm, eps) if _fused_entry is not None else None
                    if _pair is not None:
                        h, normed = _pair
                    else:
                        o_proj_fused = (getattr(ops.add_rms_norm, "gemm_add_rms", None)
                                        if _copykill_enabled("k2") else None)
                        if o_proj_fused is not None:
                            try:
                                h, normed = o_proj_fused(o[0], layer.o, h, layer.post_norm, eps)
                            except ValueError:
                                o_proj_fused = None
                        if o_proj_fused is None:
                            h, normed = ops.add_rms_norm(ops.gemm_epilogue(o[0], layer.o)[None], h, layer.post_norm, eps)
                    hidden = ops.gemm_glu_epilogue(normed[0], layer.gate_up)
                    following = self.final_norm if i == last else self.layers[i + 1].input_norm
                    _pair = _fused_entry(hidden, layer.down, h, following, eps) if _fused_entry is not None else None
                    if _pair is not None:
                        h, normed = _pair
                    else:
                        h, normed = ops.add_rms_norm(ops.gemm_epilogue(hidden, layer.down)[None], h, following, eps)
                x = ops.flow_velocity_euler(x, normed, self.action_head.proj_back.weight, dts[step], dof_mask, padding,
                                            pinned_rows, prefix)
            ops.join_prefetch()
        out.copy_(x)


def optimized_weight_mapping(model) -> dict[str, Source]:
    """Checkpoint key → tensor of the optimized composition, in its dtypes and orientations.

    The ViT, the token embedding and the action head load as for the reference composition (BF16; the action head
    FP32), the language-model RMSNorm weights FP32. Both stages' output and down projections are stored ``[N, K]``:
    the v22 CannProjection gemm route reads the weight as given, and a transposed view would cost it a full
    materialization per call (the ``[K, N]`` layout predates that route).

    Args:
        model: The ``WallOSS`` shell built with ``numerics="optimized"``.

    Returns:
        ``{model parameter name: Source}``.
    """
    bf16, fp32 = torch.bfloat16, torch.float32
    one = lambda key: Source.one(key, bf16)  # noqa: E731
    vision = model.vision.config
    mapping = {
        "vision.patch.weight": Source(("visual.patch_embed.proj.weight",), bf16,
                                      combine=lambda w: w[0].reshape(vision.hidden_size, -1)),
        "vision.merger.norm": one("visual.merger.ln_q.weight"),
        "vision.merger.fc1_weight": one("visual.merger.mlp.0.weight"),
        "vision.merger.fc1_bias": one("visual.merger.mlp.0.bias"),
        "vision.merger.fc2_weight": one("visual.merger.mlp.2.weight"),
        "vision.merger.fc2_bias": one("visual.merger.mlp.2.bias"),
        "embed_tokens": one("model.embed_tokens.weight"),
        "action.final_norm": Source.one("model.norms.1.weight", fp32),
    }
    for name in ("w1", "w2", "w3"):
        mapping[f"action_head.{name}.weight"] = Source.one(f"action_preprocessor.{name}.weight", fp32)
    mapping["action_head.proj_back.weight"] = Source.one("action_preprocessor.action_proj_back.weight", fp32)
    for i in range(vision.depth):
        source, target = f"visual.blocks.{i}", f"vision.blocks.{i}"
        mapping[f"{target}.norm1"] = one(f"{source}.norm1.weight")
        mapping[f"{target}.norm2"] = one(f"{source}.norm2.weight")
        for name, key in (("qkv", "attn.qkv"), ("proj", "attn.proj"), ("gate_up", "mlp.gate_up_proj"),
                          ("down", "mlp.down_proj")):
            mapping[f"{target}.{name}_weight"] = one(f"{source}.{key}.weight")
            mapping[f"{target}.{name}_bias"] = one(f"{source}.{key}.bias")
    for i in range(model.config.num_hidden_layers):
        source = f"model.layers.{i}"
        for expert, stack in ((0, "prefill_stack"), (1, "action")):
            target = f"{stack}.layers.{i}"
            mapping[f"{target}.input_norm"] = Source.one(f"{source}.input_layernorms.{expert}.weight", fp32)
            mapping[f"{target}.post_norm"] = Source.one(f"{source}.post_attention_layernorms.{expert}.weight", fp32)
            mapping[f"{target}.qkv"] = one(f"{source}.self_attn.qkv_proj_experts.{expert}.weight")
            mapping[f"{target}.qkv_bias"] = one(f"{source}.self_attn.qkv_proj_experts.{expert}.bias")
            mapping[f"{target}.gate_up"] = one(f"{source}.moe.experts.{expert}.gate_up_proj.weight")
            o = f"{source}.self_attn.o_proj_experts.{expert}.weight"
            down = f"{source}.moe.experts.{expert}.down_proj.weight"
            mapping[f"{target}.o"], mapping[f"{target}.down"] = one(o), one(down)
    return mapping
