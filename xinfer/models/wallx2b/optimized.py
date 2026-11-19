"""Wall-X 2B on the CUDA backend's kernels (``numerics="optimized"``, the default).

The ViT and the prompt stream are the shared ``xinfer/models/wallx/optimized.py``. The action stage runs the steps
of the reference composition (``numerics="reference"``, the wall-x model under autocast) on operators of
``xinfer/ops/reference/`` (each a kernel call) chosen for its 16 rows and width 1024, where the GEMMs with
fused norms and residuals have no fast kernel: the adaptive norms are separate calls that also add the gated residual
(the two norms of a linear layer, around its identity mixer, in one call), and the GEMMs are plain (the output
projections and the MLP down projection read ``[K, N]`` weights, which select the small-M kernel).
Differences to the reference:

- conditioning: the flow condition and every adaptive norm's modulation are computed in FP32 once at load, for the
  fixed flow times (``t_k`` of the generated rows, 1 of the pinned RTC rows), and the norms take them in FP32. The
  reference projects the modulations in BF16;
- each step embeds ``[z | dof mask]`` in FP32 in one call; the full layers' ``[q | k | v | gate]`` GEMM, q/k norms,
  RoPE and K/V append are one call, and attention reads the prompt K/V and the action keys from the head-major
  cache, with the action mask as per-column row ranges;
- the readout uses the FP32 weight on the normed rows (FP32 accumulation) and the velocity stays FP32; the
  readout, the velocity of the uncommanded dimensions, the Euler step and the pinned rows are one call. The
  reference reads out with a BF16 GEMM and rounds the velocity to BF16.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from xinfer.loader import Source
from xinfer.models.wallx.optimized import (PREFILL_STAGE_OPS, VISION_STAGE_OPS, ParamGroup, column_ranges,
                                           q_k_v_gate, vision_prompt_weight_mapping)
from xinfer.ops import active_ops

DTYPE = torch.bfloat16

OPTIMIZED_STAGE_OPS = {
    "vision": VISION_STAGE_OPS,
    "prefill": PREFILL_STAGE_OPS,
    "action": ("gemm_epilogue", "gemm_glu_epilogue", "qkv_norm_rope_kvcache", "gated_kvcache_attention",
               "modulated_rms_norm", "gated_residual_modulated_rms_norm", "identity_layer_norms_", "flow_input_embed",
               "flow_velocity_euler"),
}


class OptimizedAction(nn.Module):
    """The action stream over the ``H`` action rows and the model's FP32 flow head.

    GEMM weights are ``[N, K]``; the output and down projections are stored ``[K, N]`` (``o_t``, ``down_t``).
    ``prepare``
    builds the modulation tables after the weights are loaded and releases the FP32 projections.

    Args:
        config: ``Wallx2bConfig`` of the checkpoint.
        flow_head: The model's ``FlowHead`` (input projection, flow-time condition, readout).
    """

    def __init__(self, config, flow_head):
        super().__init__()
        self.config = config
        m = config.model
        a, i, cond = m.action_hidden_size, m.action_intermediate_size, m.adarms_cond_dim
        heads, kv, d = m.num_attention_heads, m.num_key_value_heads, m.head_dim
        self.layers = nn.ModuleList()
        for kind in m.layer_types:
            shapes = dict(gate_up=(2 * i, a), down_t=(i, a))
            if kind == "full_attention":  # qkv rows [q | k | v | gate]
                shapes.update(qkv=((2 * heads + 2 * kv) * d, a), q_norm=(d,), k_norm=(d,), o_t=(heads * d, a))
            self.layers.append(ParamGroup(**shapes))
        # The adaptive norms' projections in call order (layer i input, then post; the final norm), FP32.
        self.projections = nn.ModuleList(ParamGroup(weight=(3 * a, cond), bias=(3 * a,))
                                         for _ in range(2 * len(m.layer_types) + 1))
        self.flow_head = flow_head
        self.ops = active_ops()
        self.host_steps: list[float] = []

    def times(self, device) -> torch.Tensor:
        """FP32 flow times ``linspace(0, 1, steps + 1)``, shifted, times ``s`` (as ``Wallx2b.denoise``)."""
        m = self.config.model
        times = torch.linspace(0.0, 1.0, self.config.num_steps + 1, device=device, dtype=torch.float32)
        if m.time_shift != 1.0:
            times = (m.time_shift * times) / (1 + (m.time_shift - 1) * times)
        return times * m.flow_time_scale

    @torch.no_grad()
    def prepare(self) -> None:
        """After loading, on the weights' device: the modulations of every step and norm, the step sizes on the
        host and the stacked input projection; then the FP32 projections are released.

        ``modulations`` ``[steps + 1, norms, 1, 3·hidden]`` FP32: row ``k < steps`` holds every norm's modulation for
        flow time ``t_k`` (the generated rows), row ``steps`` for flow time 1 (the pinned RTC rows).
        """
        head = self.flow_head
        times = self.times(head.w1.weight.device)
        conditions = head.condition(torch.cat((times[:-1], times.new_ones(1))))  # [steps + 1, cond] FP32
        mods = torch.stack([F.linear(conditions, p.weight, p.bias) for p in self.projections], dim=1)
        self.modulations = mods[:, :, None].contiguous()
        self.host_steps = (times[1:] - times[:-1]).tolist()
        self.embed_weight = head.w1.weight[None].float().contiguous()
        del self.projections

    def forward(self, x, prompt_kv, cos, sin, out):
        """Euler flow over the action rows; writes ``out`` ``[1, H, D]`` FP32 (inputs as in ``Wallx2b.denoise``).

        Args:
            x: The stage inputs.
            prompt_kv: ``[full-attention layers, 2, 1, kv_heads, P, head_dim]`` BF16.
            cos: FP32 ``[H, rotary_dim / 2]`` compact cosines of the action positions.
            sin: FP32 ``[H, rotary_dim / 2]`` compact sines.
            out: Output ``[1, H, D]`` FP32.
        """
        # The cache-attention metadata scope of the 2B action bridge: one open scope per forward
        # lets every flow step's ``gated_kvcache_attention`` call share the dense drop mask the
        # bridge builds from the live flash mask, instead of each call rebuilding it. The torch
        # composition exposes no scope and keeps the getattr default (a null context).
        scope = getattr(self.ops.gated_kvcache_attention, "action_scope", nullcontext)
        with scope():
            return self._forward_inner(x, prompt_kv, cos, sin, out)

    def _forward_inner(self, x, prompt_kv, cos, sin, out):
        c, ops, m = self.config, self.ops, self.config.model
        heads, kv_heads, d, eps = m.num_attention_heads, m.num_key_value_heads, m.head_dim, m.rms_norm_eps
        noise, dof = x["noise"].to(torch.float32), x["dof_mask"].to(torch.float32)
        rows, p = noise.shape[1], prompt_kv.shape[-2]
        dof_mask = padding = pinned = prefix = None
        if not x["dof_mask_all"]:
            dof_mask, padding = dof, x["padding_action"].to(torch.float32) - noise
        z = noise
        if "rtc_actions" in x:
            pinned = x["inference_delay"].reshape(1).to(torch.int32)
            prefix = x["rtc_actions"].to(torch.float32)
            z = torch.where((torch.arange(rows, device=noise.device) < pinned)[None, :, None], prefix, noise)
        cache = torch.empty((*prompt_kv.shape[:-2], p + rows, prompt_kv.shape[-1]), dtype=prompt_kv.dtype,
                            device=prompt_kv.device)
        cache[..., :p, :].copy_(prompt_kv)
        flash_mask = column_ranges(x["action_mask"])
        start = torch.full((1,), p, dtype=torch.int32, device=noise.device)
        full = m.full_attention_layers
        pinned_mods = self.modulations[c.num_steps]

        def mod(step_mods, j, name="modulation"):
            """The modulation arguments of norm ``j``: the step's, and the pinned rows' with RTC."""
            if pinned is None:
                return {name: step_mods[j]}
            return {name: step_mods[j], f"pinned_{name}": pinned_mods[j], "pinned_rows": pinned}

        for step in range(c.num_steps):
            mods = self.modulations[step]
            h = ops.flow_input_embed(z, self.embed_weight, extra=dof)
            branch = gate = None
            for i, layer in enumerate(self.layers):
                if i in full:
                    if branch is None:
                        normed, gate_in = ops.modulated_rms_norm(h, eps=eps, **mod(mods, 2 * i))
                    else:
                        normed, gate_in, h = ops.gated_residual_modulated_rms_norm(h, branch, gate, eps=eps,
                                                                                   **mod(mods, 2 * i))
                    keys, vals = cache[full.index(i), 0], cache[full.index(i), 1]  # [1, KV, P + H, D]
                    packed = ops.qkv_norm_rope_kvcache(normed[0], layer.qkv, layer.q_norm, layer.k_norm, cos, sin,
                                                       keys[0], vals[0], p, heads, kv_heads, d, eps)
                    if isinstance(packed, tuple):
                        # Normalize the native head-major query to the OpSpec layout by view.
                        q_attn, gate_rows = packed
                        o = ops.gated_kvcache_attention(q_attn.transpose(1, 2), keys.transpose(1, 2),
                                                        vals.transpose(1, 2), start, p, gate_rows,
                                                        flash_mask=flash_mask, scale=d ** -0.5)
                    else:
                        o = ops.gated_kvcache_attention(packed[:, :heads * d].view(1, rows, heads, d),
                                                        keys.transpose(1, 2), vals.transpose(1, 2), start, p,
                                                        packed[None, :, heads * d:], flash_mask=flash_mask,
                                                        scale=d ** -0.5)
                    mixed = ops.gemm_epilogue(o[0], layer.o_t.t())[None]
                    normed, gate, h = ops.gated_residual_modulated_rms_norm(h, mixed, gate_in, eps=eps,
                                                                            **mod(mods, 2 * i + 1))
                elif branch is None:
                    normed, gate_in = ops.modulated_rms_norm(h, eps=eps, **mod(mods, 2 * i))
                    normed, gate, h = ops.gated_residual_modulated_rms_norm(h, normed, gate_in, eps=eps,
                                                                            **mod(mods, 2 * i + 1))
                else:
                    gate = ops.identity_layer_norms_(h, branch, gate, eps=eps,
                                                     **{**mod(mods, 2 * i, "input_modulation"),
                                                        **mod(mods, 2 * i + 1, "post_modulation")})
                    normed = branch
                hidden = ops.gemm_glu_epilogue(normed[0], layer.gate_up)
                branch = ops.gemm_epilogue(hidden, layer.down_t.t())[None]
            normed, _, h = ops.gated_residual_modulated_rms_norm(h, branch, gate, eps=eps, **mod(mods, -1))
            z = ops.flow_velocity_euler(z, normed, self.flow_head.action_proj_back.weight, self.host_steps[step],
                                        dof_mask, padding, pinned, prefix)
        out.copy_(z)


def optimized_weight_mapping(model) -> dict[str, Source]:
    """Checkpoint key → tensor of the optimized composition, in its dtypes and orientations.

    The ViT, prompt stream and token embedding load as ``vision_prompt_weight_mapping`` gives them; the flow head as
    for the reference composition, except the readout in FP32. The action stream takes ``[N, K]`` BF16 GEMM weights
    (``[q | k | v | gate]`` concatenated), the output and down projections ``[K, N]``, the q/k norms BF16 and raw,
    and the adaptive norms' projections FP32.

    Args:
        model: The ``Wallx2b`` shell built with ``numerics="optimized"``.

    Returns:
        ``{model parameter name: Source}``.
    """
    from xinfer.models.wallx2b.model import flow_head_mapping

    bf16, fp32 = torch.bfloat16, torch.float32
    m = model.config.model
    lm = "model.language_model"
    one = lambda key: Source.one(key, bf16)  # noqa: E731
    t = lambda key: Source((key,), bf16, combine=lambda w: w[0].t().contiguous())  # noqa: E731
    mapping = flow_head_mapping()
    mapping["flow_head.action_proj_back.weight"] = Source.one("model.action_processor.action_proj_back.weight", fp32)
    mapping.update(vision_prompt_weight_mapping(m.vision.depth, m.layer_types, len(model.prefill_stack.layers)))
    norms = []
    for i, kind in enumerate(m.layer_types):
        source, action = f"{lm}.layers.{i}", f"action.layers.{i}"
        attn = f"{source}.self_attn"
        norms += [f"{source}.input_layernorm_action", f"{source}.post_attention_layernorm_action"]
        mapping[f"{action}.gate_up"] = one(f"{source}.mlp_action_expert.gate_up_proj.weight")
        mapping[f"{action}.down_t"] = t(f"{source}.mlp_action_expert.down_proj.weight")
        if kind == "full_attention":  # [q·gate per head | k | v] → [q | k | v | gate]
            mapping[f"{action}.qkv"] = Source(
                (f"{attn}.q_proj_action.weight", f"{attn}.k_proj_action.weight", f"{attn}.v_proj_action.weight"), bf16,
                combine=lambda w, h=m.num_attention_heads: q_k_v_gate(*w, h))
            mapping[f"{action}.o_t"] = t(f"{attn}.o_proj_action.weight")
            mapping[f"{action}.q_norm"] = one(f"{attn}.q_norm_action.weight")
            mapping[f"{action}.k_norm"] = one(f"{attn}.k_norm_action.weight")
    for j, norm in enumerate(norms + [f"{lm}.norm_action"]):
        for part in ("weight", "bias"):
            mapping[f"action.projections.{j}.{part}"] = Source.one(f"{norm}.dense.{part}", fp32)
    return mapping
