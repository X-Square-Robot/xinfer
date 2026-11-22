"""WALL-OSS-0.5 checkpoint configuration.

A WALL-OSS-0.5 checkpoint directory holds ``config.json`` (model structure, a
Qwen2.5-VL config extended with the action expert) and ``config.yml`` (the
training configuration: action/state layout, normalization, RTC training).
This module reads the values inference needs and rejects layouts the
implementation does not support, instead of guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from xinfer.models.checkpoint_identity import validate_checkpoint_model
from xinfer.models.registry import WALL_OSS_0_5_ID

# Tokens added after the 151665 base Qwen2.5 tokens, in this order, by the
# Wall-X trainer (wall_x/trainer/trainer_utils.py). Only the first two can
# occur in a prompt; the rest come from the AR action tokenizer.
BASE_VOCAB_SIZE = 151665
PROPRI_TOKEN = "<|propri|>"
ACTION_TOKEN = "<|action|>"
PROPRI_TOKEN_ID = 151665
ACTION_TOKEN_ID = 151666
PAD_TOKEN_ID = 151643  # <|endoftext|>, the Qwen2.5 tokenizer's padding token


def _expect(name, value, expected):
    if value != expected:
        raise ValueError(f"checkpoint {name}={value!r}; supported: {expected!r}")


@dataclass(frozen=True)
class VisionConfig:
    """Qwen2.5-VL ViT structure from ``config.json``'s ``vision_config``.

    Attributes:
        depth: Number of ViT blocks (WALL-OSS-0.5: 32).
        hidden_size: ViT width (WALL-OSS-0.5: 1280).
        intermediate_size: MLP width as configured (``model.vision_config`` pads it to a multiple of 128).
        num_heads: Attention heads.
        in_channels: Image channels (``in_chans``, default 3).
        patch_size: Spatial patch size in pixels (WALL-OSS-0.5: 14).
        temporal_patch_size: Frames per patch (WALL-OSS-0.5: 2; the processor repeats the single image).
        spatial_merge_size: Patches merged per side into one image token (WALL-OSS-0.5: 2, i.e. 2x2 patches).
        window_size: Window side in pixels for window attention (WALL-OSS-0.5: 112).
        out_hidden_size: Width of the merged image tokens (the language model width).
        fullatt_block_indexes: Blocks that use full attention instead of window attention.
    """

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    window_size: int
    out_hidden_size: int
    fullatt_block_indexes: tuple[int, ...]

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class ExpertConfig:
    """Width of one mixture-of-transformers expert (``config.json`` ``experts[i]``).

    Attributes:
        hidden_size: Expert width (WALL-OSS-0.5: 2048 for expert 0, 1024 for expert 1).
        intermediate_size: MLP width of the expert.
    """

    hidden_size: int
    intermediate_size: int


@dataclass(frozen=True)
class WallOSS05Config:
    """The values of ``config.json`` and ``config.yml`` that inference needs; build it with ``load``.

    Attributes:
        hidden_size: Language model (expert 0) width.
        num_hidden_layers: Decoder layers, shared by both experts.
        num_attention_heads: Query heads of the shared attention.
        num_key_value_heads: K/V heads (grouped-query attention).
        rms_norm_eps: Epsilon of the language-model RMSNorms.
        rope_theta: RoPE base.
        mrope_section: M-RoPE temporal/height/width band sizes over the half head dimension.
        experts: ``(expert 0, expert 1)`` widths: the prompt expert and the action expert.
        vocab_size: ``config.json`` vocabulary size (the embedding table may have more rows).
        image_token_id: Id of ``<|image_pad|>``.
        vision_start_token_id: Id of ``<|vision_start|>``.
        vision_end_token_id: Id of ``<|vision_end|>``.
        vision: The ViT configuration.
        action_horizon: Action tokens per request ``H`` (``data.action_horizon_flow``).
        dof_config: ``(name, dims)`` of the action layout, in order; sums to ``action_dim``.
        agent_pos_config: ``(name, dims)`` of the proprioceptive state layout; sums to ``state_dim``.
        resolution: Per-view long-side target in pixels (``data.resolution``; missing views keep their size).
        use_state_string_representation: Whether the state enters the prompt as a binned string (the input
            processor requires it).
        use_per_token_tau: ``config.yml`` flag; not checked (the action stage always uses per-token time).
        rtc_rope_gap: Position shift of the action tokens after the pinned RTC prefix.
        rtc_max_delay: ``rtc_train.max_delay``; upper bound of the RTC delay the model accepts.
        per_step_normalization: ``config.yml`` flag; not checked (statistics are always ``[H, D]``).
        disable_normalizer_clamp: ``config.yml`` flag; not checked (normalization never clamps).
        causal_action_attention_mask: ``config.json`` flag; not read by inference.
        use_x_pred: ``config.json`` flag; not read by inference (the action stage predicts velocity).
        noise_scheduler: ``config.json`` noise scheduler settings; not read by inference.
        raw_json: The parsed ``config.json``.
        raw_yaml: The parsed ``config.yml``.
    """

    # Language backbone (expert 0) and shared attention geometry.
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    mrope_section: tuple[int, int, int]
    experts: tuple[ExpertConfig, ExpertConfig]
    vocab_size: int
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    vision: VisionConfig
    # Action space and training semantics (config.yml).
    action_horizon: int
    dof_config: tuple[tuple[str, int], ...]
    agent_pos_config: tuple[tuple[str, int], ...]
    resolution: dict
    use_state_string_representation: bool
    use_per_token_tau: bool
    rtc_rope_gap: int
    rtc_max_delay: int
    per_step_normalization: bool
    disable_normalizer_clamp: bool
    causal_action_attention_mask: bool
    use_x_pred: bool
    noise_scheduler: dict
    raw_json: dict
    raw_yaml: dict

    @property
    def head_dim(self) -> int:
        """Attention head dimension ``hidden_size / num_attention_heads`` (WALL-OSS-0.5: 128)."""
        return self.hidden_size // self.num_attention_heads

    @property
    def action_dim(self) -> int:
        """Action dimension ``D``: the sum of ``dof_config`` (WALL-OSS-0.5: 26)."""
        return sum(d for _, d in self.dof_config)

    @property
    def state_dim(self) -> int:
        """Proprioceptive state dimension: the sum of ``agent_pos_config``."""
        return sum(d for _, d in self.agent_pos_config)

    @classmethod
    def load(cls, checkpoint: str | Path) -> WallOSS05Config:
        """Read ``config.json`` and ``config.yml`` from a checkpoint directory.

        Args:
            checkpoint: Checkpoint directory.

        Returns:
            The parsed configuration.

        Raises:
            ValueError: The checkpoint is not a supported layout (a two-expert ``qwen2_5_vl`` with
                attention/MLP/norm MoE, SiLU, M-RoPE, no sliding window, a SiLU ViT, ``experts[0]`` as wide as
                ``hidden_size`` and ``config.yml`` ``model_type: qwen2_5``), or ``config.yml`` lacks
                ``data.action_horizon_flow``.
        """
        root = Path(checkpoint)
        validate_checkpoint_model(root, WALL_OSS_0_5_ID)
        cj = json.loads((root / "config.json").read_text())
        cy = yaml.safe_load((root / "config.yml").read_text())

        _expect("model_type", cj.get("model_type"), "qwen2_5_vl")
        _expect("num_experts", cj.get("num_experts"), 2)
        for flag in ("attention_moe", "mlp_moe", "norm_moe"):
            _expect(flag, cj.get(flag), True)
        _expect("hidden_act", cj.get("hidden_act"), "silu")
        _expect("rope_scaling.type", (cj.get("rope_scaling") or {}).get("type"), "mrope")
        _expect("use_sliding_window", cj.get("use_sliding_window", False), False)
        _expect("model_type (config.yml)", cy.get("model_type"), "qwen2_5")

        v = cj["vision_config"]
        vision = VisionConfig(
            depth=v["depth"],
            hidden_size=v["hidden_size"],
            intermediate_size=v["intermediate_size"],
            num_heads=v["num_heads"],
            in_channels=v.get("in_chans", 3),
            patch_size=v["patch_size"],
            temporal_patch_size=v["temporal_patch_size"],
            spatial_merge_size=v["spatial_merge_size"],
            window_size=v["window_size"],
            out_hidden_size=v["out_hidden_size"],
            fullatt_block_indexes=tuple(v["fullatt_block_indexes"]),
        )
        _expect("vision hidden_act", v.get("hidden_act"), "silu")
        experts = tuple(ExpertConfig(e["hidden_size"], e["intermediate_size"]) for e in cj["experts"])
        _expect("experts[0].hidden_size", experts[0].hidden_size, cj["hidden_size"])
        data = cy.get("data") or {}
        horizon = data.get("action_horizon_flow")
        if horizon is None:
            raise ValueError("config.yml lacks data.action_horizon_flow")
        return cls(
            hidden_size=cj["hidden_size"],
            num_hidden_layers=cj["num_hidden_layers"],
            num_attention_heads=cj["num_attention_heads"],
            num_key_value_heads=cj["num_key_value_heads"],
            rms_norm_eps=cj["rms_norm_eps"],
            rope_theta=cj["rope_theta"],
            mrope_section=tuple(cj["rope_scaling"]["mrope_section"]),
            experts=experts,
            vocab_size=cj["vocab_size"],
            image_token_id=cj["image_token_id"],
            vision_start_token_id=cj["vision_start_token_id"],
            vision_end_token_id=cj["vision_end_token_id"],
            vision=vision,
            action_horizon=int(horizon),
            dof_config=tuple((k, int(d)) for k, d in cy["dof_config"].items()),
            agent_pos_config=tuple((k, int(d)) for k, d in cy["agent_pos_config"].items()),
            resolution=dict(data.get("resolution") or {}),
            use_state_string_representation=bool(cy.get("use_state_string_representation", False)),
            use_per_token_tau=bool(cy.get("use_per_token_tau", False)),
            rtc_rope_gap=int(cy.get("rtc_rope_gap", 0)),
            rtc_max_delay=int((cy.get("rtc_train") or {}).get("max_delay", 0)),
            per_step_normalization=bool(cy.get("per_step_normalization", False)),
            disable_normalizer_clamp=bool(cy.get("disable_normalizer_clamp", False)),
            causal_action_attention_mask=bool(cj.get("causal_action_attention_mask", False)),
            use_x_pred=bool(cj.get("use_x_pred", False)),
            noise_scheduler=dict(cj.get("noise_scheduler") or {}),
            raw_json=cj,
            raw_yaml=cy,
        )
