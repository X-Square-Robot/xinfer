"""Wall-OSS input and output processing (CPU only).

The InputProcessor turns a robot request into model inputs; the
OutputProcessor turns the model's actions (checkpoint units) into the robot
response. Behavior follows the embodied-serving Wall-X 4B service (es 9633ffd
+ Wall-X 02c5351d), which is the reference these functions are tested
against bit for bit; HarriX's bitwise port of the same service is the second
source. Deployment choices of that service (fixed instruction, camera wire
keys, embodiment text, admitted RTC delays) are explicit fields of
``ProcessingConfig`` instead of hidden constants.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xinfer.layers.vision.qwen2_5_vit import vision_layout
from xinfer.models.interface import BucketSpec, PreparedRequest
from xinfer.models.processing import (
    SegmentTokenizer,
    text_length_range,
    check_image_preprocess_device,
    device_image,
    is_jpeg,
    map_views,
    source_size,
)
from xinfer.models.wall_oss import geometry
from xinfer.models.wall_oss.image_processing import (
    decode_image, decode_view, mrope_position_ids, smart_resize, wire_view,
)
from xinfer.models.wall_oss.config import (
    ACTION_TOKEN,
    ACTION_TOKEN_ID,
    PAD_TOKEN_ID,
    PROPRI_TOKEN,
    PROPRI_TOKEN_ID,
    WallOSSConfig,
)

IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
STATE_BINS = 512


@dataclass(frozen=True)
class CameraBinding:
    """One camera: where its image arrives on the wire and how the model and prompt name it.

    Attributes:
        view: Model view name, also the key of ``data.resolution`` (e.g. ``"face_view"``).
        wire_key: Key of the image under ``request["views"]`` (e.g. ``"camera_front"``).
        label: Name of the view in the prompt (e.g. ``"front view"``).
    """

    view: str  # model view name, also the key of data.resolution
    wire_key: str  # key under request["views"]
    label: str  # name used in the prompt


@dataclass(frozen=True)
class ProcessingConfig:
    """Deployment options. Defaults reproduce the reference desktop deployment.

    Attributes:
        cameras: Camera bindings; their order is the image order in the prompt and of the ``image_<i>`` inputs.
        norm_key: Key suffix of the action and state statistics in ``normalizer_*.pth``.
        embodiment: Embodiment line of the system prompt.
        frequency_hz: Control frequency stated in the system prompt.
        action_space: Action space stated in the system prompt.
        instruction_override: Fixed task string replacing the client's ``instruction``; ``None`` keeps it.
        num_steps: Euler steps per request.
        allow_client_num_steps: Whether RTC requests may set the step count (``num_flow_steps`` /
            ``num_steps``).
        rtc_wire_delay_range: Inclusive range admitted for the raw wire ``inference_delay`` of RTC requests.
        noise_seed: Seed of the CPU generator of the initial flow noise; ``None`` seeds from the OS.
        prompt_bucket: Prompts are padded at their end, before the action tokens, to a multiple of this many
            tokens (``pad_prompt``); 0 keeps their exact length. The padding is masked out, so it changes the
            actions at rounding level only.
        resize_factor: ``smart_resize`` grid factor in pixels (patch size · merge size).
        min_pixels: ``smart_resize`` lower bound on the pixel count.
        max_pixels: ``smart_resize`` upper bound on the pixel count.
        image_mean: Per-channel RGB mean on the [0, 1] scale.
        image_std: Per-channel RGB standard deviation on the [0, 1] scale.
        patch_size: ViT patch side in pixels.
        temporal_patch_size: Frames per patch (the single image is repeated).
        merge_size: Patches merged per side into one image token.
        image_preprocess_device: Where the views are decoded and resized: ``"cpu"`` in the processor (OpenCV,
            the reference path), or ``"gpu"`` by the engine on the CUDA device (``DeviceImage`` inputs: nvJPEG and
            an FP32 bicubic resize), which saves the processor's decode and resize time but does not give the CPU
            path's pixels bit for bit.
    """

    cameras: tuple[CameraBinding, ...] = (
        CameraBinding("face_view", "camera_front", "front view"),
        CameraBinding("left_wrist_view", "camera_left", "left wrist view"),
        CameraBinding("right_wrist_view", "camera_right", "right wrist view"),
    )
    norm_key: str = "ex_normal"
    embodiment: str = "X2 ex001_desktop"
    frequency_hz: int = 32
    action_space: str = "Rel EEF"
    # The reference service replaces every client instruction with a fixed
    # task string; ``None`` uses the client's instruction.
    instruction_override: str | None = None
    num_steps: int = 5
    allow_client_num_steps: bool = True  # RTC requests may set num_flow_steps/num_steps
    rtc_wire_delay_range: tuple[int, int] = (3, 8)  # admission on the raw wire value
    # Initial flow noise is drawn here on the CPU, so the action stage reads no
    # random state; ``None`` seeds from the OS.
    noise_seed: int | None = None
    # The state is written into the prompt as text, so the prompt length changes with the arm poses; CUDA graphs
    # are keyed by the padded length.
    prompt_bucket: int = 16
    # Image geometry (checkpoint preprocessor_config.json).
    resize_factor: int = 28
    min_pixels: int = 3136
    max_pixels: int = 12845056
    image_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    patch_size: int = 14
    temporal_patch_size: int = 2
    merge_size: int = 2
    image_preprocess_device: str = "cpu"

    def __post_init__(self):
        check_image_preprocess_device(self.image_preprocess_device)

    def bucket_spec(self) -> BucketSpec:
        """The prompt policy: pad to multiples of ``prompt_bucket`` tokens (0: exact lengths)."""
        return BucketSpec(self.prompt_bucket)


# Resize recipe of the device path (DeviceImage.resize): one bicubic resize (a = -0.75) without
# antialiasing, as OpenCV INTER_CUBIC.
DEVICE_RESIZE = (("bicubic", False),)


def processing_config(value) -> ProcessingConfig:
    """Accept a ProcessingConfig, None, or a JSON-style dict of its fields.

    Args:
        value: ``ProcessingConfig`` (returned as is), ``None`` (defaults) or a dict of fields; dict
            ``cameras`` may be dicts or ``(view, wire_key, label)`` sequences, and list-valued tuple fields
            are converted to tuples.

    Returns:
        The ``ProcessingConfig``.
    """
    if value is None:
        return ProcessingConfig()
    if isinstance(value, ProcessingConfig):
        return value
    fields = dict(value)
    if "cameras" in fields:
        fields["cameras"] = tuple(CameraBinding(**c) if isinstance(c, dict) else CameraBinding(*c)
                                  for c in fields["cameras"])
    for name in ("rtc_wire_delay_range", "image_mean", "image_std"):
        if name in fields:
            fields[name] = tuple(fields[name])
    return ProcessingConfig(**fields)


def wire_metadata(model_config: WallOSSConfig) -> dict:
    """The frame the desktop x2robot client reads on connect.

    Args:
        model_config: The checkpoint's configuration.

    Returns:
        ``{"batch_enabled": True, "model": "wall-x", "action_horizon": H, "action_dim": D,
        "rtc_ee_chunk_dim": 14}``.
    """
    return {"batch_enabled": True, "model": "wall-x", "action_horizon": model_config.action_horizon,
            "action_dim": model_config.action_dim, "rtc_ee_chunk_dim": 14}


# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------


@dataclass
class Normalizer:
    """Min/range statistics of one normalizer file; values map to ``(x - min) / delta · 2 - 1``.

    Attributes:
        min: Minimum, float32 (``[H, D]`` for actions, ``[state_dim]`` for the state).
        delta: Range, float32, same shape as ``min``.
    """

    min: np.ndarray  # float32
    delta: np.ndarray  # float32

    @classmethod
    def load(cls, path: Path, key: str) -> Normalizer:
        """Read ``min.<key>`` and ``delta.<key>`` from a ``torch.save`` file as float32 arrays.

        Args:
            path: ``normalizer_action.pth`` or ``normalizer_propri.pth``.
            key: Statistics key suffix (``ProcessingConfig.norm_key``).

        Returns:
            The ``Normalizer``.
        """
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            state[f"min.{key}"].numpy().astype(np.float32),
            state[f"delta.{key}"].numpy().astype(np.float32),
        )


def load_tokenizer(checkpoint: Path):
    """Load ``tokenizer.json`` and add the Wall-X tokens ``<|propri|>`` (151665) and ``<|action|>`` (151666).

    Args:
        checkpoint: Checkpoint directory.

    Returns:
        A ``tokenizers.Tokenizer``.

    Raises:
        ValueError: The tokenizer does not end at the base Qwen2.5 vocabulary, or the added tokens did not
            receive their training ids.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    if tokenizer.get_vocab_size(with_added_tokens=True) != PROPRI_TOKEN_ID:
        raise ValueError("tokenizer.json does not end at the base Qwen2.5 vocabulary")
    tokenizer.add_tokens([PROPRI_TOKEN, ACTION_TOKEN])
    if tokenizer.token_to_id(PROPRI_TOKEN) != PROPRI_TOKEN_ID or tokenizer.token_to_id(ACTION_TOKEN) != ACTION_TOKEN_ID:
        raise ValueError("added Wall-X tokens did not receive their training ids")
    return tokenizer


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------


def resize_view(image: np.ndarray, target: int, config: ProcessingConfig) -> np.ndarray:
    """Scale the long side to ``target`` then snap to the patch grid; one cubic resize.

    The intermediate size (long side ``target``, short side truncated to int) only feeds ``smart_resize``;
    the image itself is resized once, with OpenCV ``INTER_CUBIC``.

    Args:
        image: ``[H, W, 3]`` uint8 RGB.
        target: Long-side target in pixels (``data.resolution`` of the view); ``-1`` keeps the input size.
        config: Processing options (``resize_factor``, ``min_pixels``, ``max_pixels``).

    Returns:
        ``[h, w, 3]`` uint8 RGB with ``h`` and ``w`` multiples of ``config.resize_factor``.
    """
    import cv2

    hs, ws = resize_size(*image.shape[:2], target, config)
    return cv2.resize(image, (ws, hs), interpolation=cv2.INTER_CUBIC)


def resize_size(h0: int, w0: int, target: int, config: ProcessingConfig) -> tuple[int, int]:
    """``(h, w)`` that ``resize_view`` gives an ``h0 × w0`` image."""
    if target == -1:
        h1, w1 = h0, w0
    elif w0 > h0:
        w1, h1 = target, int(target * h0 / w0)
    else:
        h1, w1 = target, int(target * w0 / h0)
    return smart_resize(h1, w1, config.resize_factor, config.min_pixels, config.max_pixels)


# --------------------------------------------------------------------------
# State and actions
# --------------------------------------------------------------------------

# Wire source of each agent_pos_config entry (desktop robot form).
_ARM_FIELDS = {"left": "follow1_pos", "right": "follow2_pos"}
_OPTIONAL_FIELDS = {"velocity_decomposed": "velocity_decomposed", "height": "lift", "head_actions": "head_pos"}
# dof dims the desktop robot never commands (velocity, height, head).
_DESKTOP_MASKED_DOF = ("velocity_decomposed", "height", "head_actions")


def _arm_block(state: Mapping[str, Any], field: str, rtc: bool) -> dict[str, np.ndarray]:
    """Split the first 7 values of one arm's wire state into ``pos``, ``rpy`` and ``grip``.

    Returns ``{"pos": [1, 3], "rpy": [1, 3], "grip": [1, 1]}``: the end-effector xyz position (metres),
    its roll/pitch/yaw (ZYX Euler angles, radians) and the gripper value. RTC requests cast the values to
    float32; otherwise the wire dtype is kept.
    """
    raw = np.asarray(state[field], dtype=np.float32) if rtc else np.asarray(state[field])
    raw = raw[:7]
    return dict(pos=raw[0:3].reshape(1, 3), rpy=raw[3:6].reshape(1, 3), grip=raw[6:7].reshape(1, 1))


def build_agent_pos(state: Mapping[str, Any], config: WallOSSConfig, rtc: bool):
    """Return ``(agent_pos [1,26] float64, mask [1,26] float64, arm blocks)``.

    The state is assembled in ``config.agent_pos_config`` order (26 dims for Wall-OSS). Arm entries come from
    ``follow1_pos`` (left) and ``follow2_pos`` (right): xyz position, the rotation as 6D (first two rows of
    the ZYX Euler rotation matrix) and the gripper. ``velocity_decomposed``, ``height`` and ``head_actions``
    come from the optional wire fields ``velocity_decomposed``, ``lift`` and ``head_pos``; a missing field
    gives zeros and a zero mask.

    Args:
        state: Wire ``state`` mapping.
        config: Model configuration (``agent_pos_config``).
        rtc: Whether the request is RTC (arm values are cast to float32 first).

    Returns:
        ``(agent_pos, mask, arms)``: ``agent_pos`` ``[1, state_dim]`` float64; ``mask`` ``[1, state_dim]``
        float64 (1 = present); ``arms`` ``{"left", "right"}`` → ``{"pos", "rpy", "grip"}`` arrays of the
        wire values (see ``_arm_block``), the reference pose for the relative actions.

    Raises:
        ValueError: An unsupported ``agent_pos_config`` key, or a value of the wrong size.
    """
    arms = {side: _arm_block(state, field, rtc) for side, field in _ARM_FIELDS.items()}
    parts, mask = [], []
    for key, dim in config.agent_pos_config:
        side = "left" if "_left_" in key else "right" if "_right_" in key else None
        present = True
        if side is not None:
            arm = arms[side]
            if key.endswith("ee_cartesian_pos"):
                value = arm["pos"]
            elif key.endswith("ee_rotation_6D"):
                value = geometry.euler_to_matrix_zyx_6d_nb(arm["rpy"])
            elif key.endswith("gripper"):
                value = arm["grip"]
            else:
                raise ValueError(f"unsupported arm state key {key!r}")
        elif key in _OPTIONAL_FIELDS:
            wire = state.get(_OPTIONAL_FIELDS[key])
            if wire is None:
                value, present = np.zeros((1, dim)), False
            else:
                value = np.asarray(wire).reshape(1, -1)
        else:
            raise ValueError(f"unsupported state key {key!r}")
        if value.shape != (1, dim):
            raise ValueError(f"state {key!r} has shape {value.shape}, expected (1, {dim})")
        parts.append(value)
        mask.append(np.full((1, dim), 1.0 if present else 0.0))
    agent_pos = np.concatenate(parts, axis=-1)
    return agent_pos, np.concatenate(mask, axis=-1), arms


def dof_slices(config: WallOSSConfig) -> dict[str, slice]:
    """Map each ``dof_config`` name to its slice of the action vector ``[D]``."""
    out, start = {}, 0
    for key, dim in config.dof_config:
        out[key] = slice(start, start + dim)
        start += dim
    return out


def _segment(config: WallOSSConfig, side: str, suffix: str) -> slice:
    return dof_slices(config)[f"follow_{side}_{suffix}"]


def rtc_actions_raw(chunk: np.ndarray, arms: dict, config: WallOSSConfig) -> np.ndarray:
    """Absolute EE chunk ``[N, 14]`` → relative model-layout rows ``[N-1, D]`` (float32).

    Row 0 of the chunk is dropped. For each arm, relative to the request's state: position minus the state
    position, rotation ``R_abs · R_stateᵀ`` as 6D (first two rows), and the gripper value unchanged; the
    other action dims stay 0. Computed in float64 from the float32 state.

    Args:
        chunk: ``[N, 14]`` absolute poses, left arm in columns 0–6 and right arm in 7–13, each
            ``[x, y, z, roll, pitch, yaw, gripper]`` (ZYX Euler angles).
        arms: Arm blocks from ``build_agent_pos`` (the current state).
        config: Model configuration (``dof_config``).

    Returns:
        ``[N - 1, D]`` float32 in checkpoint units (not normalized).
    """
    reference = np.concatenate([np.concatenate([arms[s]["pos"], arms[s]["rpy"], arms[s]["grip"]], -1)[0]
                          for s in ("left", "right")]).astype(np.float32)
    act = chunk[1:].astype(np.float64)
    refd = reference.astype(np.float64)
    out = np.zeros((act.shape[0], config.action_dim), dtype=np.float32)
    for side, off in (("left", 0), ("right", 7)):
        pos_rel = act[:, off:off + 3] - refd[off:off + 3]
        rot_abs = geometry.euler_to_matrix_zyx_6d_nb(np.ascontiguousarray(act[:, off + 3:off + 6]))
        rot_ref = geometry.euler_to_matrix_zyx_6d_nb(np.ascontiguousarray(refd[off + 3:off + 6][None]))[0]
        rot_rel = geometry.compute_delta_from_state_and_abs_rot(rot_abs, rot_ref)
        out[:, _segment(config, side, "ee_cartesian_pos_relative")] = pos_rel
        out[:, _segment(config, side, "ee_rotation_6D_relative")] = rot_rel
        out[:, _segment(config, side, "gripper")] = act[:, off + 6:off + 7]
    return out


# --------------------------------------------------------------------------
# Processors
# --------------------------------------------------------------------------


def pad_prompt(tensors: Mapping[str, Any], width: int) -> dict[str, Any]:
    """A request's tensors with its prompt padded at its end to ``width`` tokens.

    The ``P`` valid prompt tokens (``prompt_valid``, or ``prompt_len`` of an unpadded request) keep their ids
    and M-RoPE positions; pad tokens (``PAD_TOKEN_ID``, positions 0) fill ``[P, width)``, and the ``H`` action
    tokens follow. Under causal attention no valid token attends to a later pad token, so the prefill K/V of the
    valid tokens do not change, and the action stage reads only the first ``prompt_valid`` prompt columns. The
    action positions (``action_position_base``) stay those of the unpadded prompt. A ``width`` below ``P``
    cuts the prompt to ``width`` tokens, which only serves to build warmup inputs of a smaller bucket
    (``Engine.warmup``).

    Args:
        tensors: a prepared request's tensors (``input_ids``, ``position_ids``, ``prompt_len`` and, once
            padded, ``prompt_valid``).
        width: padded prompt length.

    Returns:
        A new dict with ``input_ids`` ``[1, width + H]``, ``position_ids`` ``[3, 1, width + H]``,
        ``prompt_len = width`` and ``prompt_valid`` ``[1]`` int32 (``min(P, width)``); other entries are shared.

    Raises:
        ValueError: ``width`` would cut image tokens.
    """
    start = int(tensors["prompt_len"])  # first action token
    valid = int(tensors["prompt_valid"][0]) if "prompt_valid" in tensors else start
    keep = min(valid, width)
    if tensors["image_token_index"].numel() and int(tensors["image_token_index"].max()) >= keep:
        raise ValueError(f"a prompt of {width} tokens would cut image tokens")
    ids, positions = tensors["input_ids"], tensors["position_ids"]
    pad = width - keep
    out = dict(tensors)
    out["input_ids"] = torch.cat([ids[:, :keep], torch.full((1, pad), PAD_TOKEN_ID, dtype=ids.dtype),
                                  ids[:, start:]], dim=1)
    out["position_ids"] = torch.cat([positions[:, :, :keep], positions.new_zeros((3, 1, pad)),
                                     positions[:, :, start:]], dim=2)
    out["prompt_len"] = width
    out["prompt_valid"] = torch.tensor([keep], dtype=torch.int32)
    return out


class WallOSSInputProcessor:
    """Turn a desktop robot observation into a ``PreparedRequest`` (CPU only).

    Args:
        checkpoint: Checkpoint directory (``tokenizer.json``, ``normalizer_action.pth``,
            ``normalizer_propri.pth``).
        model_config: The checkpoint's configuration.
        config: ``ProcessingConfig``, dict or ``None`` (defaults).

    Attributes:
        config: The resolved ``ProcessingConfig``.
        model_config: The checkpoint's configuration.
        tokenizer: Checkpoint tokenizer with the Wall-X tokens added.
        action_norm: Action statistics ``[H, D]``.
        state_norm: State statistics ``[state_dim]``.
        vision_config: ``VisionTowerConfig`` used for ``vision_layout``.
        padding_action: Normalized zero action ``[1, H, D]`` float32.
        dof_mask: ``[1, H, D]`` float64; 0 on the dims the desktop robot never commands (velocity, height,
            head), 1 elsewhere.

    Raises:
        ValueError: The normalizer shapes do not match the checkpoint's action/state layout, or the checkpoint
            does not use the state string representation.
    """

    def __init__(self, checkpoint: str | Path, model_config: WallOSSConfig, config: ProcessingConfig | None = None):
        checkpoint = Path(checkpoint)
        self.config = processing_config(config)
        self.bucket = self.config.bucket_spec()
        self.model_config = model_config
        self.tokenizer = load_tokenizer(checkpoint)
        self._tokenize = SegmentTokenizer(self.tokenizer)
        self.action_norm = Normalizer.load(checkpoint / "normalizer_action.pth", self.config.norm_key)
        self.state_norm = Normalizer.load(checkpoint / "normalizer_propri.pth", self.config.norm_key)
        h, d = model_config.action_horizon, model_config.action_dim
        if self.action_norm.min.shape != (h, d) or self.state_norm.min.shape != (model_config.state_dim,):
            raise ValueError("normalizer statistics do not match the checkpoint's action/state layout")
        if not model_config.use_state_string_representation:
            raise ValueError("only checkpoints with use_state_string_representation are supported")
        self._noise = torch.Generator()
        if self.config.noise_seed is None:
            self._noise.seed()
        else:
            self._noise.manual_seed(self.config.noise_seed)
        from xinfer.models.wall_oss.model import vision_config

        self.vision_config = vision_config(model_config)
        self.padding_action = self._normalize_actions(np.zeros((1, h, d), np.float32))
        slices = dof_slices(model_config)
        self.dof_mask = np.ones((1, h, d), dtype=np.float64)
        for key in _DESKTOP_MASKED_DOF:
            if key in slices:
                self.dof_mask[:, :, slices[key]] = 0.0

    # Normalization: four separate IEEE operations, no clamp.
    def _normalize_actions(self, x: np.ndarray) -> np.ndarray:
        """``(x - min) / delta · 2 - 1`` with the float32 action statistics; ``x`` is ``[..., H, D]``."""
        return ((x - self.action_norm.min) / self.action_norm.delta) * 2 - 1

    def _normalize_state(self, x: np.ndarray) -> np.ndarray:
        """``(x - min) / delta · 2 - 1`` in float64; ``x`` is ``[..., state_dim]``."""
        return ((x - self.state_norm.min.astype(np.float64)) / self.state_norm.delta.astype(np.float64)) * 2 - 1

    def _rtc(self, observation: Mapping[str, Any]):
        """Admit the request's RTC fields.

        Reads ``use_rtc``, ``inference_delay`` and ``prev_action_chunk`` (``[N, 14]`` or ``[1, N, 14]``).

        Returns:
            ``(chunk [N, 14], delay)`` with ``delay = min(wire delay, N - 1, H, rtc_max_delay)``, or ``None``
            for a flow request: RTC not requested, no chunk, wire delay 0, a chunk not of width 14, or a
            resulting delay of 0.

        Raises:
            ValueError: An RTC request whose wire delay is outside ``config.rtc_wire_delay_range``.
        """
        config, model_config = self.config, self.model_config
        use_rtc = bool(observation.get("use_rtc"))
        wire_delay = int(observation.get("inference_delay") or 0)
        min_delay, max_delay = config.rtc_wire_delay_range
        if use_rtc and not min_delay <= wire_delay <= max_delay:
            raise ValueError(f"inference_delay {wire_delay} outside admitted range [{min_delay}, {max_delay}]")
        chunk = observation.get("prev_action_chunk")
        if not (use_rtc and chunk is not None and wire_delay > 0):
            return None
        chunk = np.asarray(chunk)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[1] != 14:
            return None
        delay = min(wire_delay, chunk.shape[0] - 1, model_config.action_horizon, model_config.rtc_max_delay)
        if delay <= 0:
            return None
        return chunk, delay

    def __call__(self, observation: Mapping[str, Any]) -> PreparedRequest:
        """Prepare one request: images, state string, prompt, positions, vision layout and RTC actions.

        Args:
            observation: Desktop robot observation. Reads:

                - ``views``: ``{wire_key: image}`` for every ``config.cameras`` entry (HWC uint8 RGB array or
                  raw/base64 JPEG);
                - ``state``: ``follow1_pos`` / ``follow2_pos`` (left/right arm, first 7 values: xyz in
                  metres, roll/pitch/yaw in radians, gripper) and the optional ``velocity_decomposed``,
                  ``lift`` and ``head_pos``;
                - ``instruction``: task text (not read when ``config.instruction_override`` is set);
                - RTC: ``use_rtc``, ``inference_delay`` (wire delay), ``prev_action_chunk`` ``[N, 14]``
                  absolute poses and, if ``config.allow_client_num_steps``, ``num_flow_steps`` /
                  ``num_steps``.

        Returns:
            ``PreparedRequest`` with CPU ``tensors`` (``P`` prompt tokens, ``H`` horizon, ``D`` action dim):

            - ``input_ids`` ``[1, P + H]`` int64: the prompt followed by ``H`` ``<|action|>`` tokens;
            - ``position_ids`` ``[3, 1, P + H]`` int64: M-RoPE positions;
            - ``image_<i>`` ``[h_i, w_i, 3]`` uint8 RGB: the resized views in camera order (the vision stage
              normalizes and packs them, ``xinfer.layers.image_pack``); with ``image_preprocess_device="gpu"``
              instead ``DeviceImage`` values of the views as they arrived (``device_image``), which the engine
              turns into the same views on the device;
            - ``image_grid_thw`` ``[num_views, 3]`` int64, ``proprioception`` ``[1, 1, state_dim]`` float64
              (normalized state) and ``agent_pos_mask`` ``[1, 1, state_dim]`` float64: compared with the
              reference service in tests; the stages do not read them;
            - ``dof_mask`` ``[1, H, D]`` float64, ``dof_mask_all`` (bool), ``padding_action`` ``[1, H, D]``
              float32 and ``noise`` ``[1, H, D]`` float32 (drawn from the processor's generator);
            - ``prompt_len`` (int ``P``), ``image_token_index`` ``[N_tokens]`` int64 and
              ``action_position_base`` ``[1]`` int64 (M-RoPE position of the first action token);
            - with ``config.prompt_bucket``: the prompt padded by ``pad_prompt`` to ``P`` rounded up to the
              bucket, so ``P`` above is the padded length, and ``prompt_valid`` ``[1]`` int32, the number of
              valid prompt tokens;
            - ``vision_positions``, ``vision_window_index``, ``vision_reverse_index``, ``vision_cu_window``,
              ``vision_cu_full``: the ``vision_layout`` of the image grids;
            - ``inference_delay`` ``[1]`` int64 (0 for flow requests) and ``num_steps`` (int);
            - RTC only: ``rtc_actions`` ``[1, T, D]`` float32, the normalized relative previous actions,
              ``T = min(N - 1, H)``.

            ``meta`` holds ``arms`` (the state pose, see ``build_agent_pos``), ``image_size`` (``(w, h)`` of
            each resized view), ``rtc``, ``inference_delay`` (int) and ``prompt_len_range``: the shortest and
            longest unpadded prompt any state gives with this request's instruction and images (each state value
            is written with 1 to 3 digits, one token each), from which ``Engine.warmup`` derives the
            prompt buckets to warm up.

        Raises:
            ValueError: Inadmissible RTC delay, undecodable or malformed images, malformed state, or a
                prompt whose action tokens are not the last ``H`` tokens.
        """
        config, model_config = self.config, self.model_config
        rtc = self._rtc(observation)
        state = observation["state"]

        # Images, in prompt order.
        views = observation["views"]
        targets = [model_config.resolution.get(camera.view, -1) for camera in config.cameras]
        if config.image_preprocess_device == "gpu":
            # JPEG bytes go to the device decoder; other encodings are decoded here and resized on the device.
            originals = [wire_view(views[camera.wire_key]) for camera in config.cameras]
            originals = [image if isinstance(image, np.ndarray) or is_jpeg(image) else decode_image(image)
                         for image in originals]
            shapes = [resize_size(*source_size(image), target, config) for image, target in zip(originals, targets)]
            images = [device_image(image, (shape,), DEVICE_RESIZE) for image, shape in zip(originals, shapes)]
        else:
            resized = map_views(lambda item: resize_view(decode_view(views[item[0].wire_key]), item[1], config),
                                list(zip(config.cameras, targets)))
            images = [torch.from_numpy(image) for image in resized]
            shapes = [image.shape[:2] for image in resized]
        grids = [(1, h // config.patch_size, w // config.patch_size) for h, w in shapes]
        sizes = [(w, h) for h, w in shapes]

        # State.
        agent_pos, agent_mask, arms = build_agent_pos(state, model_config, rtc is not None)
        proprio = self._normalize_state(agent_pos)[None]  # [1, 1, D] float64
        bins = np.linspace(-1, 1, STATE_BINS + 1)[:-1]
        digits = np.digitize(proprio, bins) - 1
        state_text = " ".join(str(digits[0, 0, i]) for i in range(proprio.shape[-1]) if agent_mask[0, i] != 0)

        # Prompt.
        merge = config.merge_size ** 2
        labels = ", ".join(c.label for c in config.cameras)
        system = (
            "You are an embodied vision-language-action (VLA) model controlling the robot with "
            f"language instructions.\n Embodiment: {config.embodiment}\n Camera Setup: {labels},\n"
            f" Frequency: {config.frequency_hz}HZ\n Action Space: {config.action_space}\n"
        )
        observation_text = " ".join(
            f"{c.label}: {VISION_START}{IMAGE_PAD * (g[0] * g[1] * g[2] // merge)}{VISION_END}"
            for c, g in zip(config.cameras, grids)
        )
        instruction = config.instruction_override
        if instruction is None:
            instruction = observation["instruction"]
        text = (
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nObservation: {observation_text}\n"
            f"Instruction: {instruction}\nPredict the next action in robot action.\n"
            f"Proprioception: {state_text}\n<|im_end|>\n<|im_start|>assistant\n"
            + ACTION_TOKEN * model_config.action_horizon
        )
        token_ids = np.asarray(self._tokenize(text), dtype=np.int64)[None]
        action_positions = np.flatnonzero(token_ids[0] == ACTION_TOKEN_ID)
        if (action_positions.size != model_config.action_horizon
                or action_positions[0] != token_ids.shape[1] - model_config.action_horizon):
            raise ValueError("the action tokens must be the last action_horizon tokens")
        position_ids = mrope_position_ids(token_ids, grids, model_config.image_token_id, config.merge_size)
        prompt_len = token_ids.shape[1] - model_config.action_horizon
        layout = vision_layout(grids, self.vision_config)

        num_steps = config.num_steps
        tensors = dict(
            input_ids=torch.from_numpy(token_ids),
            position_ids=torch.from_numpy(position_ids),
            **{f"image_{i}": image for i, image in enumerate(images)},
            image_grid_thw=torch.tensor(grids, dtype=torch.int64),
            proprioception=torch.from_numpy(proprio),
            agent_pos_mask=torch.from_numpy(agent_mask[None]),
            dof_mask=torch.from_numpy(self.dof_mask.copy()),
            padding_action=torch.from_numpy(self.padding_action.copy()),
            noise=torch.randn((1, model_config.action_horizon, model_config.action_dim), generator=self._noise),
            # Model-side layout derived from the prompt and image grids. Host
            # values (ints, lists, bools) select the CUDA graph; per-request
            # values the device needs are tensors.
            prompt_len=prompt_len,
            image_token_index=torch.from_numpy(np.flatnonzero(token_ids[0] == model_config.image_token_id)),
            action_position_base=torch.tensor([position_ids[0, 0, prompt_len]], dtype=torch.int64),
            dof_mask_all=bool(self.dof_mask.all()),
            vision_positions=layout["positions"],
            vision_window_index=layout["window_index"],
            vision_reverse_index=layout["reverse_index"],
            vision_cu_window=layout["cu_window"],
            vision_cu_full=layout["cu_full"],
        )
        # Each state value takes 1 to 3 digit tokens (bins 0..511), the rest of the prompt does not depend on it.
        state_digits = sum(c.isdigit() for c in state_text)
        values = int((agent_mask != 0).sum())
        max_digits = len(str(STATE_BINS - 1))
        meta = dict(arms=arms, image_size=sizes, rtc=rtc is not None, inference_delay=0,
                    prompt_len_range=text_length_range(prompt_len, state_digits, values, max_tokens=max_digits))
        if rtc is not None:
            chunk, delay = rtc
            raw = rtc_actions_raw(chunk, arms, model_config)
            t = raw.shape[0]
            p = raw[None]
            if t < model_config.action_horizon:
                fill = np.zeros((1, model_config.action_horizon - t, model_config.action_dim), np.float32)
                p = np.concatenate([p, fill], 1)
            p = self._normalize_actions(p[:, : model_config.action_horizon])[:, :t]
            tensors["rtc_actions"] = torch.from_numpy(np.ascontiguousarray(p))
            meta["inference_delay"] = delay
            requested = observation.get("num_flow_steps", observation.get("num_steps"))
            if config.allow_client_num_steps and requested is not None and int(requested) > 0:
                num_steps = int(requested)
        tensors["inference_delay"] = torch.tensor([meta["inference_delay"]], dtype=torch.int64)
        tensors["num_steps"] = num_steps
        if config.prompt_bucket:
            tensors = pad_prompt(tensors, self.bucket.width(prompt_len))
        return PreparedRequest(tensors=tensors, meta=meta)


class WallOSSOutputProcessor:
    """``predict_action`` [1, H, D] (checkpoint units) → desktop robot response.

    Args:
        model_config: The checkpoint's configuration (``dof_config`` slices).
    """

    def __init__(self, model_config: WallOSSConfig):
        self.model_config = model_config

    def __call__(self, outputs: Mapping[str, Any], request: PreparedRequest) -> dict[str, Any]:
        """Compose the relative actions with the request's state into absolute arm poses.

        For each arm: position = state position + relative position; rotation = 6D delta composed with the
        state rotation (``R_delta · R_state``), converted to canonical ZYX Euler angles; gripper taken as is.

        Args:
            outputs: Final stage outputs; reads ``predict_action`` ``[1, H, D]`` FP32 (row 0 only).
            request: The ``PreparedRequest`` of this request; reads ``meta["arms"]``.

        Returns:
            Response dict:

            - ``follow1_pos`` / ``follow2_pos``: left / right arm, lists of ``H + 1`` rows
              ``[x, y, z, roll, pitch, yaw, gripper]``; row 0 is the request's state, rows 1..H the
              predicted absolute poses;
            - ``predict_relative_chunk``: ``[H + 1, 20]`` float32, per arm (left then right) 10 columns
              ``[dx, dy, dz, 6D rotation delta, gripper]``; row 0 is the identity (zero offset, identity
              rotation, the state's gripper);
            - ``action_pred``: ``[H + 1, 14]`` float32, ``follow1_pos`` and ``follow2_pos`` side by side.
        """
        model_config = self.model_config
        action = np.asarray(outputs["predict_action"])[0]  # float32 [H, D]
        arms = request.meta["arms"]
        rows, relative = {}, {}
        for side in ("left", "right"):
            s = arms[side]
            pos_c = _segment(model_config, side, "ee_cartesian_pos_relative")
            rot_c = _segment(model_config, side, "ee_rotation_6D_relative")
            grip_c = _segment(model_config, side, "gripper")
            pos_abs = s["pos"] + action[:, pos_c]
            state6 = geometry.euler_to_matrix_zyx_6d_nb(s["rpy"])[0]
            abs6 = geometry.compose_state_and_delta_to_abs_6d(action[:, rot_c], state6)
            rpy = geometry.so3_to_euler_zyx_batch_nb(abs6)
            arm = np.concatenate([
                np.concatenate([s["pos"], pos_abs]),
                np.concatenate([s["rpy"], rpy]),
                np.concatenate([s["grip"], action[:, grip_c]]),
            ], axis=1)
            rows[side] = arm
            head = np.zeros((1, 10), dtype=np.float32)
            head[0, 3] = 1.0
            head[0, 7] = 1.0
            head[0, 9] = np.float32(s["grip"][0, 0])
            relative[side] = np.concatenate(
                [head, np.concatenate([action[:, pos_c], action[:, rot_c], action[:, grip_c]], -1)]
            )
        follow1, follow2 = rows["left"].tolist(), rows["right"].tolist()
        return {
            "follow1_pos": follow1,
            "follow2_pos": follow2,
            "predict_relative_chunk": np.concatenate([relative["left"], relative["right"]], axis=-1),
            "action_pred": np.concatenate(
                [np.asarray(follow1, dtype=np.float32), np.asarray(follow2, dtype=np.float32)], axis=-1
            ),
        }
