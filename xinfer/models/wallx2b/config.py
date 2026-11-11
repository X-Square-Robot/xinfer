"""Wall-X 2606 checkpoint configuration.

A Wall-X 2606 checkpoint directory holds ``config.json`` (model structure, a Qwen3.5 config extended with
``vla_config``), ``config.yml`` (the training configuration, including the Dataset V2 snapshot
``data.dataset_config_effective`` whose processors define the prompt, image and state preprocessing),
``preprocessor_config.json`` (the Qwen image processor), the tokenizer files and ``normalizer_{action,propri}.pth``.
This module reads the values inference needs and rejects layouts the implementation does not support, instead of
guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from xinfer.models.layers.vision.qwen3_5_vit import Qwen35VisionConfig

# Tokens the Wall-X trainer adds after the tokenizer's own added tokens (tokenizer.json plus the
# ``added_tokens_decoder`` of tokenizer_config.json), in this order; the prompt contains only the action token.
PROPRI_TOKEN = "<|propri|>"
ACTION_TOKEN = "<|action|>"
PROPRI_TOKEN_ID = 248077
ACTION_TOKEN_ID = 248078

# Normalizer key of the desktop deployment; it also names the dataset in the system prompt.
NORM_KEY = "x2_normal"
# x2robot_utils FREQ_MAPPING: control frequency the system prompt states for a dataset.
FREQUENCY_HZ = {"x2_normal": 32, "ex_normal": 32}


def _expect(name, value, expected):
    if value != expected:
        raise ValueError(f"checkpoint {name}={value!r}; supported: {expected!r}")


@dataclass(frozen=True)
class Camera:
    """One camera: how the model and prompt name it and where its image arrives on the wire.

    Attributes:
        view: Model view name, the key of ``Wallx2bConfig.resolution`` (e.g. ``"face_view"``).
        label: Name of the view in the prompt (x2robot_utils ``CAM_NAME_MAPPING``).
        wire_keys: Keys under ``request["views"]`` that carry the image; the first present one is used.
    """

    view: str
    label: str
    wire_keys: tuple[str, ...]


# The desktop cameras in prompt order, with the wire aliases the reference service accepts.
CAMERAS = (
    Camera("face_view", "front view", ("camera_front", "face_view", "front_view")),
    Camera("left_wrist_view", "left wrist view", ("camera_left", "left_wrist_view", "left_view")),
    Camera("right_wrist_view", "right wrist view", ("camera_right", "right_wrist_view", "right_view")),
)


@dataclass(frozen=True)
class Normalizer:
    """Min/range statistics of one normalizer file; values map to ``(x - min) / delta · 2 - 1``.

    Attributes:
        min: Minimum, float32 (``[H, D]`` per action step, ``[state_dim]`` for the state).
        delta: Range, float32, same shape as ``min``.
    """

    min: np.ndarray
    delta: np.ndarray

    @classmethod
    def load(cls, path: Path, key: str) -> Normalizer:
        """Read ``min.<key>`` and ``delta.<key>`` from a ``torch.save`` file as float32 arrays."""
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(state[f"min.{key}"].numpy().astype(np.float32), state[f"delta.{key}"].numpy().astype(np.float32))


@dataclass(frozen=True)
class ModelGeometry:
    """Structure of the backbone, the ViT and the action expert, from ``config.json``; build it with ``load``.

    Attributes:
        hidden_size: Prompt-stream width (2048).
        layer_types: ``"linear_attention"`` / ``"full_attention"`` per layer (both streams).
        intermediate_size: Prompt-stream MLP width.
        num_attention_heads, num_key_value_heads, head_dim: Full-attention geometry.
        linear_heads: ``(num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel)`` of the DeltaNet.
        rms_norm_eps: RMSNorm epsilon.
        rope_theta: RoPE base.
        rotary_dim: Rotated channels per head (``head_dim · partial_rotary_factor``).
        mrope_section: Interleaved M-RoPE band sizes (temporal, height, width).
        vision: The ViT configuration.
        action_hidden_size, action_intermediate_size, adarms_cond_dim: Action stream geometry.
        flow_time_scale: Last flow time (noise scheduler ``s``).
        time_shift: Flow-time shift of the noise scheduler (1 = none).
    """

    hidden_size: int
    layer_types: tuple[str, ...]
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_heads: tuple[int, int, int, int, int]
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    mrope_section: tuple[int, int, int]
    vision: Qwen35VisionConfig
    action_hidden_size: int
    action_intermediate_size: int
    adarms_cond_dim: int
    flow_time_scale: float
    time_shift: float

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        """Indices of the full-attention layers (the layers with prompt K/V)."""
        return tuple(i for i, t in enumerate(self.layer_types) if t == "full_attention")

    @classmethod
    def load(cls, path) -> "ModelGeometry":
        """Read a checkpoint directory's ``config.json``.

        Raises:
            ValueError: The checkpoint uses a structure this implementation does not support.
        """
        path = Path(path)
        raw = json.loads((path / "config.json").read_text())
        text, vision, vla = raw["text_config"], raw["vision_config"], raw["vla_config"]
        rope = text["rope_parameters"]
        _expect("text_config.attn_output_gate", text.get("attn_output_gate"), True)
        _expect("text_config.hidden_act", text.get("hidden_act"), "silu")
        _expect("text_config.attention_bias", text.get("attention_bias"), False)
        _expect("rope_parameters.rope_type", rope.get("rope_type"), "default")
        _expect("rope_parameters.mrope_interleaved", rope.get("mrope_interleaved"), True)
        _expect("vision_config.hidden_act", vision.get("hidden_act"), "gelu_pytorch_tanh")
        _expect("vision_config.deepstack_visual_indexes", vision.get("deepstack_visual_indexes", []), [])
        for key, expected in (("use_mot", True), ("norm_moe", True), ("mlp_moe", True), ("use_adarms", True),
                              ("action_linear_attention", False), ("causal_action_attention_mask", False),
                              ("use_state_string_representation", True)):
            _expect(f"vla_config.{key}", vla.get(key), expected)
        _expect("vla_config.use_x_pred", vla.get("use_x_pred", False), False)
        _expect("vla_config.dim_inputs", list(vla["dim_inputs"]), [text["hidden_size"], vla["action_hidden_size"]])
        scheduler = vla["noise_scheduler"]
        return cls(
            hidden_size=text["hidden_size"],
            layer_types=tuple(text["layer_types"]),
            intermediate_size=text["intermediate_size"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            linear_heads=(text["linear_num_key_heads"], text["linear_num_value_heads"], text["linear_key_head_dim"],
                          text["linear_value_head_dim"], text["linear_conv_kernel_dim"]),
            rms_norm_eps=text["rms_norm_eps"],
            rope_theta=float(rope["rope_theta"]),
            rotary_dim=int(text["head_dim"] * rope.get("partial_rotary_factor", 1.0)),
            mrope_section=tuple(rope["mrope_section"]),
            vision=Qwen35VisionConfig(
                depth=vision["depth"], hidden_size=vision["hidden_size"],
                intermediate_size=vision["intermediate_size"], num_heads=vision["num_heads"],
                in_channels=vision["in_channels"], patch_size=vision["patch_size"],
                temporal_patch_size=vision["temporal_patch_size"], spatial_merge_size=vision["spatial_merge_size"],
                out_hidden_size=vision["out_hidden_size"],
                num_position_embeddings=vision["num_position_embeddings"]),
            action_hidden_size=vla["action_hidden_size"],
            action_intermediate_size=vla["action_mlp_intermediate_size"],
            adarms_cond_dim=vla["adarms_cond_dim"],
            flow_time_scale=float(scheduler.get("s", 0.999)),
            time_shift=float(scheduler.get("time_shift", 1.0)),
        )


@dataclass(frozen=True)
class Wallx2bConfig:
    """The checkpoint values inference needs; build it with ``load``.

    Attributes:
        model: Structure of the backbone, ViT and action expert.
        image_token_id: Id of ``<|image_pad|>``.
        vision_start_token_id: Id of ``<|vision_start|>``.
        vision_end_token_id: Id of ``<|vision_end|>``.
        action_horizon: Action tokens per request ``H``.
        dof_config: ``(name, dims)`` of the action layout, in order; sums to ``action_dim``.
        agent_pos_config: ``(name, dims)`` of the proprioceptive state layout; sums to ``state_dim``.
        num_steps: Euler steps per request (``vla_config.noise_scheduler.num_inference_timesteps``).
        rtc_rope_gap: Position shift of the action tokens after the pinned RTC prefix.
        rtc_max_delay: Largest RTC delay seen in training (``rtc_train.max_delay``).
        action_norm: Per-step action statistics ``[H, D]``.
        state_norm: State statistics ``[state_dim]``.
        clamp_normalized: Whether normalized state and padding actions are clamped to ``[-1, 1]``
            (``normalizer_config.clip``); the RTC prefix is never clamped.
        cameras: Camera bindings; their order is the image order in the prompt.
        resolution: Per-view long-side target in pixels of the first resize.
        resize_factor: ``smart_resize`` grid of the first resize (Dataset V2 ``image_factor``).
        resize_min_pixels: Lower bound on the pixel count of the first resize.
        resize_max_pixels: Upper bound on the pixel count of the first resize.
        min_pixels: Lower bound on the pixel count of the image processor's resize (``size.shortest_edge``).
        max_pixels: Upper bound on the pixel count of the image processor's resize (``size.longest_edge``).
        patch_size: ViT patch side in pixels.
        temporal_patch_size: Frames per patch (the single image is repeated).
        merge_size: Patches merged per side into one image token.
        image_mean: Per-channel RGB mean on the [0, 1] scale.
        image_std: Per-channel RGB standard deviation on the [0, 1] scale.
        norm_key: Normalizer key; also the dataset name of the system prompt.
        embodiment: Embodiment line of the system prompt (``embodiment_type_to_name`` of the checkpoint's
            ``embodiment_type``).
        frequency_hz: Control frequency stated in the system prompt.
        relative_action: Whether the system prompt states a relative action space.
        state_bins: Bins of the state string in the prompt.
        max_instruction_words: Longest admitted instruction, in words.
        max_seq_length: Longest admitted token sequence, actions included.
        special_tokens: The checkpoint's ``model.new_special_tokens``, added to the tokenizer after the action
            token.
    """

    model: ModelGeometry
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    action_horizon: int
    dof_config: tuple[tuple[str, int], ...]
    agent_pos_config: tuple[tuple[str, int], ...]
    num_steps: int
    rtc_rope_gap: int
    rtc_max_delay: int
    action_norm: Normalizer
    state_norm: Normalizer
    clamp_normalized: bool
    cameras: tuple[Camera, ...]
    resolution: dict
    resize_factor: int
    resize_min_pixels: int
    resize_max_pixels: int
    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    norm_key: str
    embodiment: str
    frequency_hz: int
    relative_action: bool
    state_bins: int
    max_instruction_words: int
    max_seq_length: int
    special_tokens: tuple[str, ...]

    @property
    def action_dim(self) -> int:
        """Action dimension ``D``: the sum of ``dof_config`` (Wall-X 2606: 26)."""
        return sum(d for _, d in self.dof_config)

    @property
    def state_dim(self) -> int:
        """Proprioceptive state dimension: the sum of ``agent_pos_config`` (Wall-X 2606: 26)."""
        return sum(d for _, d in self.agent_pos_config)

    @classmethod
    def load(cls, checkpoint: str | Path, norm_key: str = NORM_KEY) -> Wallx2bConfig:
        """Read the configuration files and normalizer statistics of a checkpoint directory.

        Args:
            checkpoint: Checkpoint directory.
            norm_key: Normalizer key.

        Returns:
            The parsed configuration.

        Raises:
            ValueError: The checkpoint is not a supported layout: ``config.yml`` ``model_type`` other than
                ``qwen3_5``, no embedded Dataset V2 snapshot, a text processor other than
                ``wallx_task_instruction`` with an embodied prompt, a vision processor other than ``qwen3_5``,
                a state that is not written into the prompt, an image processor other than
                ``Qwen2VLImageProcessorFast`` with 2-frame patches, an unknown embodiment, or normalizer
                statistics that do not match the action/state layout.
        """
        root = Path(checkpoint)
        cj = json.loads((root / "config.json").read_text())
        cy = yaml.safe_load((root / "config.yml").read_text())
        image = json.loads((root / "preprocessor_config.json").read_text())
        _expect("model_type (config.yml)", cy.get("model_type"), "qwen3_5")
        _expect("image_processor_type", image.get("image_processor_type"), "Qwen2VLImageProcessorFast")
        _expect("temporal_patch_size", image.get("temporal_patch_size"), 2)

        task = cy["task"]
        data = cy.get("data") or {}
        dataset = (data.get("dataset_config_effective") or {}).get("dataset")
        if not dataset:
            raise ValueError("config.yml lacks the Dataset V2 snapshot data.dataset_config_effective")
        processors = dataset["processors"]
        _expect("vision processor", processors["vision"]["type"], "qwen3_5")
        _expect("text processor", processors["text"]["type"], "wallx_task_instruction")
        vision = processors["vision"]["params"]
        text = processors["text"]["params"]
        epilogue = processors["epilogue"]["params"]
        rtc = epilogue.get("rtc_train") or {}
        if text.get("use_embodied_prompt_ratio", 0) <= 0:
            raise ValueError("only checkpoints trained with the embodied system prompt are supported")
        _expect("use_state_string_representation", epilogue.get("use_state_string_representation"), True)
        _expect("action_chunk_size", text.get("action_chunk_size"), task["action_horizon_flow"])
        embodiment_type = (dataset.get("globals") or {}).get("embodiment_type")
        names = text.get("embodiment_type_to_name") or {}
        if embodiment_type not in names:
            raise ValueError(f"embodiment_type {embodiment_type!r} has no prompt name; names: {sorted(names)}")
        normalizer = data.get("normalizer_config") or {}

        horizon = int(task["action_horizon_flow"])
        dof_config = tuple((k, int(d)) for k, d in task["dof_config"].items())
        agent_pos_config = tuple((k, int(d)) for k, d in task["agent_pos_config"].items())
        action_norm = Normalizer.load(root / "normalizer_action.pth", norm_key)
        state_norm = Normalizer.load(root / "normalizer_propri.pth", norm_key)
        if (action_norm.min.shape != (horizon, sum(d for _, d in dof_config))
                or state_norm.min.shape != (sum(d for _, d in agent_pos_config),)):
            raise ValueError("normalizer statistics do not match the checkpoint's action/state layout")
        return cls(
            model=ModelGeometry.load(root),
            image_token_id=cj["image_token_id"],
            vision_start_token_id=cj["vision_start_token_id"],
            vision_end_token_id=cj["vision_end_token_id"],
            action_horizon=horizon,
            dof_config=dof_config,
            agent_pos_config=agent_pos_config,
            num_steps=int(cj["vla_config"]["noise_scheduler"]["num_inference_timesteps"]),
            rtc_rope_gap=int(rtc.get("rope_gap", 0)),
            rtc_max_delay=int(rtc["max_delay"]),
            action_norm=action_norm,
            state_norm=state_norm,
            clamp_normalized=bool(normalizer.get("clip", not normalizer.get("disable_normalizer_clamp", False))),
            cameras=CAMERAS,
            resolution=dict(vision["resolution"]),
            # Dataset V2 defaults of the qwen3_5 vision processor.
            resize_factor=int(vision.get("image_factor", 32)),
            resize_min_pixels=int(vision.get("min_pixels", 4096)),
            resize_max_pixels=int(vision.get("max_pixels", 16777216)),
            min_pixels=int(image["size"]["shortest_edge"]),
            max_pixels=int(image["size"]["longest_edge"]),
            patch_size=int(image["patch_size"]),
            temporal_patch_size=int(image["temporal_patch_size"]),
            merge_size=int(image["merge_size"]),
            image_mean=tuple(image["image_mean"]),
            image_std=tuple(image["image_std"]),
            norm_key=norm_key,
            embodiment=names[embodiment_type],
            frequency_hz=FREQUENCY_HZ[norm_key],
            relative_action=bool(text.get("use_relative_action", False)),
            state_bins=int(epilogue["state_bins"]),
            max_instruction_words=int(text.get("max_instruction_words", 256)),
            max_seq_length=int(epilogue["max_seq_length"]),
            special_tokens=tuple((cy.get("model") or {}).get("new_special_tokens") or ()),
        )
