"""Wall-X 2B: Qwen3.5 hybrid backbone with a mixture-of-transformers flow action expert.

The checkpoint is a two-stream mixture of transformers on the Qwen3.5-2B backbone (18 Gated DeltaNet layers and 6
gated full-attention layers). The prompt stream (width 2048) processes the prompt; the action stream (width 1024,
adaptive norms conditioned on the flow time) processes the ``H`` action tokens. Action tokens bypass the
linear-attention layers and attend to the prompt only through the full-attention layers, and prompt tokens never
attend to action tokens, so the model splits into three stages:

- ``vision``: views → patch rows (``ImagePack``) → image token embeddings (Qwen3.5 ViT);
- ``prefill``: the prompt stream over the prompt alone → the prompt K/V of every full-attention layer. The source
  sampler evaluates the prompt together with the ``H`` action tokens (their rows are zeroed in the DeltaNet scans
  and invisible to prompt queries), the same arithmetic in another order, which gives other bits;
- ``action``: ten Euler steps of the action stream over the action tokens with the prompt K/V.

RTC (``rtc_delay`` > 0) pins the first ``delay`` rows to the request's normalized prefix: they enter every step as
clean data with the flow-time-1 condition, do not see the other action rows, keep zero velocity and are restored
after the last step. Only the other rows' positions skip ``rtc_rope_gap``; without RTC every action row does.

The ``numerics`` model option selects the composition. ``"optimized"`` (the default) runs the steps on xcompute's
operators, with the flow conditioning in FP32 (``optimized.py``). ``"reference"`` follows the
Wall-X model under BF16 autocast: BF16 weights and activations, FP32 prompt-stream RMSNorm weights, the action
projection and time condition in FP32, the readout as a BF16 GEMM.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from xinfer.models.layers.basic import Linear
from xinfer.loader import Source
from xinfer.models.interface import CameraSpec, Capabilities, IOSchema, OutputSpec
from xinfer.models.wallx.qwen35 import DTYPE, WallXQwen35, reference_weight_mapping, text_rope_tables
from xinfer.models.wallx2b.config import ModelGeometry, Wallx2bConfig
from xinfer.models.wallx2b.optimized import OPTIMIZED_STAGE_OPS, OptimizedAction, optimized_weight_mapping
from xinfer.ops import active_ops

TIME_EMBED_MAX_PERIOD = 10000.0


class FlowHead(nn.Module):
    """The action projection, flow-time condition and readout (wall-x ``ActionProcessor``, AdaRMS variant).

    ``w1`` maps ``[noisy action | dof mask]`` to the action stream width; the condition is
    ``silu(time_mlp_out(silu(time_mlp_in(time_embedding(t)))))``; both run in FP32 (the source runs them under FP32
    autocast). ``action_proj_back`` reads the normed action stream out as a BF16 GEMM (the source calls it under the
    sampler's BF16 autocast).

    Args:
        action_dim: Action values per row ``D``.
        hidden: Action stream width.
    """

    def __init__(self, action_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.w1 = Linear(2 * action_dim, hidden, group="action_head")
        self.time_mlp_in = Linear(hidden, hidden, bias=True, group="action_head")
        self.time_mlp_out = Linear(hidden, hidden, bias=True, group="action_head")
        self.action_proj_back = Linear(hidden, action_dim, group="action_head")
        self._silu = active_ops().silu

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """``[B]`` FP32 flow times → ``[B, hidden]`` FP32: ``sin`` half, then ``cos`` half."""
        half = self.hidden // 2
        scale = math.log(TIME_EMBED_MAX_PERIOD) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        angles = t[:, None] * freqs[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def condition(self, t: torch.Tensor) -> torch.Tensor:
        """``[B]`` FP32 flow times → ``[B, hidden]`` FP32 AdaRMS condition."""
        return self._silu(self.time_mlp_out(self._silu(self.time_mlp_in(self.time_embedding(t)))))

    def embed(self, z: torch.Tensor, dof_mask: torch.Tensor) -> torch.Tensor:
        """``z``, ``dof_mask`` ``[B, H, D]`` FP32 → action stream input ``[B, H, hidden]`` BF16."""
        return self.w1(torch.cat([z, dof_mask], dim=-1)).to(DTYPE)


class Wallx2b(WallXQwen35):
    """Wall-X 2B served as three stages: ``vision`` → ``prefill`` → ``action`` (see the module docstring).

    The backbone, loading and the ``vision`` and ``prefill`` stages are ``WallXQwen35``'s; this class adds the flow
    action expert: ``FlowHead``, the action stage with RTC and the processing.
    """

    config_class = Wallx2bConfig

    @property
    def geometry(self) -> ModelGeometry:
        return self.config.model

    def image_normalization(self):
        return self.config.image_mean, self.config.image_std

    def build_flow_head(self) -> FlowHead:
        return FlowHead(self.config.action_dim, self.config.model.action_hidden_size)

    def build_optimized_action(self) -> OptimizedAction:
        return OptimizedAction(self.config, self.flow_head)

    def checkpoint_mapping(self) -> dict[str, Source]:
        return (optimized_weight_mapping if self.numerics == "optimized" else weight_mapping)(self)

    # ------------------------------------------------------- model interface
    def io_schema(self) -> IOSchema:
        """Wire-level request layout of the desktop robot protocol (the Wall-X 2606 service's)."""
        return IOSchema(
            cameras=tuple(CameraSpec(camera.wire_keys[0]) for camera in self.config.cameras),
            state_keys=("follow_left_position", "follow_left_rotation", "follow_left_gripper",
                        "follow_right_position", "follow_right_rotation", "follow_right_gripper"),
            action_keys=("follow1_pos", "follow2_pos"),
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
        )

    @classmethod
    def build_processors(cls, checkpoint, processing=None, **_ignored):
        """Build the CPU processors and the wire metadata frame without loading weights.

        Args:
            checkpoint: Checkpoint directory (configs, tokenizer, normalizers).
            processing: ``ProcessingConfig``, dict or ``None``.
            **_ignored: Other model options, unused here.

        Returns:
            ``(Wallx2bInputProcessor, Wallx2bOutputProcessor, wire_metadata(config))``.
        """
        from xinfer.models.wallx2b.processing import (Wallx2bInputProcessor, Wallx2bOutputProcessor,
                                                      processing_config, wire_metadata)

        config = Wallx2bConfig.load(checkpoint)
        return (Wallx2bInputProcessor(checkpoint, config, processing_config(processing)),
                Wallx2bOutputProcessor(config), wire_metadata(config))

    def shape_buckets(self):
        from xinfer.models.wallx2b.processing import processing_config

        return {"prompt": processing_config(self.processing).bucket_spec(
            self.config.max_seq_length - self.config.action_horizon)}

    def capabilities(self) -> Capabilities:
        """One request per dispatch in every stage; ``config.num_steps`` Euler steps."""
        stage_ops = OPTIMIZED_STAGE_OPS if self.numerics == "optimized" else self.stage_ops
        return Capabilities(batch_sizes={"vision": (1,), "prefill": (1,), "action": (1,)},
                            num_steps=self.config.num_steps, stage_ops=stage_ops)

    def action_outputs(self, batch):
        """``predict_action_norm`` ``[B, H, D]`` FP32, the actions in normalized units."""
        c = self.config
        return {"predict_action_norm": OutputSpec((batch.batch_size, c.action_horizon, c.action_dim), torch.float32)}

    # ----------------------------------------------------------------- stages
    def denoise(self, batch):
        """Euler flow over the action rows with the prompt K/V reused (wall-x ``generate_flow_action``).

        Each step embeds ``[z | dof mask]``, runs the action stream with the prompt K/V of each full-attention
        layer, reads out the velocity and updates ``z`` by ``(t[k+1] - t[k]) · v``; flow times are
        ``linspace(0, 1, steps + 1) · s``. Uncommanded dimensions move toward the normalized zero action:
        ``v = (padding_action - noise) · (1 - dof) + v · dof``. An RTC request pins its first ``inference_delay``
        rows (see the module docstring).

        Args:
            batch: ``StageBatch`` of one request. Reads ``prompt_kv`` and:

                - ``noise`` ``[1, H, D]`` FP32, ``dof_mask`` ``[1, H, D]``, ``dof_mask_all`` (host bool: every
                  dimension commanded) and ``padding_action`` ``[1, H, D]`` FP32;
                - ``action_positions`` ``[3, 1, H]`` int64 (gap included) and ``action_mask`` ``[1, 1, H, P + H]``
                  bool (visibility of prompt and action keys per action row);
                - RTC only: ``rtc_actions`` ``[1, H, D]`` FP32 (normalized; rows from the delay on unused) and
                  ``inference_delay`` ``[1]`` int64.

        Writes:
            ``batch.out["predict_action_norm"]`` ``[1, H, D]`` FP32: the final actions in normalized units.
        """
        x, c, m = batch.inputs, self.config, self.config.model
        kv = x["prompt_kv"]
        if self.numerics == "optimized":
            cos, sin = text_rope_tables(self.inv_freq, x["action_positions"], m.mrope_section)
            half = m.rotary_dim // 2
            self.action(x, kv, cos[0, :, 0, :half].contiguous(), sin[0, :, 0, :half].contiguous(),
                        batch.out["predict_action_norm"])
            return
        noise, dof = x["noise"].to(torch.float32), x["dof_mask"].to(torch.float32)
        times = torch.linspace(0.0, 1.0, c.num_steps + 1, device=noise.device, dtype=torch.float32)
        if m.time_shift != 1.0:
            times = (m.time_shift * times) / (1 + (m.time_shift - 1) * times)
        times = times * m.flow_time_scale
        cos, sin = text_rope_tables(self.inv_freq, x["action_positions"], m.mrope_section)
        mask = x["action_mask"]
        v_padding = None if x["dof_mask_all"] else x["padding_action"].to(torch.float32) - noise
        rtc = "rtc_actions" in x
        if rtc:
            rows = torch.arange(c.action_horizon, device=noise.device)
            pinned = (rows < x["inference_delay"].reshape(-1, 1))[..., None]  # [1, H, 1]
            prefix = x["rtc_actions"].to(torch.float32)
            pinned_cond = self.flow_head.condition(torch.ones(1, device=noise.device))[:, None]
        z = noise.clone()
        for k in range(c.num_steps):
            t = times[k]
            cond = self.flow_head.condition(t.reshape(1))[:, None]
            if rtc:
                # The source projects one condition per row for RTC and one per request otherwise.
                z = torch.where(pinned, prefix, z)
                cond = torch.where(pinned, pinned_cond, cond)
            cond = cond.to(DTYPE)
            a = self.flow_head.embed(z, dof)
            for i, layer in enumerate(self.action_layers):
                layer_kv = None if layer.linear else (kv[m.full_attention_layers.index(i), 0],
                                                      kv[m.full_attention_layers.index(i), 1])
                a = layer(a, cond, cos, sin, layer_kv, mask)
            a, _ = self.action_norm(a, cond)
            velocity = self.flow_head.action_proj_back(a)  # BF16, as under the source's autocast
            if v_padding is not None:
                velocity = v_padding * (1 - dof) + velocity * dof
            if rtc:
                velocity = torch.where(pinned, 0.0, velocity)
            z = z + (times[k + 1] - t) * velocity
        if rtc:
            z = torch.where(pinned, prefix, z)
        batch.out["predict_action_norm"].copy_(z)


def flow_head_mapping() -> dict[str, Source]:
    """Checkpoint key → ``FlowHead`` tensor: the input projection and time MLP FP32, the readout BF16."""
    ap = "model.action_processor"
    mapping = {"flow_head.w1.weight": Source.one(f"{ap}.w1.weight", torch.float32),
               "flow_head.action_proj_back.weight": Source.one(f"{ap}.action_proj_back.weight", torch.bfloat16)}
    for name in ("time_mlp_in", "time_mlp_out"):
        for part in ("weight", "bias"):
            mapping[f"flow_head.{name}.{part}"] = Source.one(f"{ap}.{name}.{part}", torch.float32)
    return mapping


def weight_mapping(model: Wallx2b) -> dict[str, Source]:
    """Checkpoint key → model tensor of the reference composition: the backbone (``reference_weight_mapping``) and
    the flow head (the input projection and time MLP FP32, the readout BF16)."""
    return {**flow_head_mapping(), **reference_weight_mapping(model)}
