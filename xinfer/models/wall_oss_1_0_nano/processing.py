"""WALL-OSS-1.0-Nano input and output processing (CPU only).

The InputProcessor turns a desktop robot request into model inputs; the OutputProcessor turns the model's
normalized actions into the robot response. Behavior follows the production Wall-X 2606 service (with wall-x d882c37
and x2robot_dataset e8ebcf9) as deployed with its rtx5090 profile: uint8 patch rows normalized
by a lookup table on the device, responses of ``H`` future poses without the state row (``omit_state_head``), and
prompts without padding (``prompt_bucket=0``). The default pads prompts to 16-token buckets with masked padding.
The exact-length configuration is tested against that service bit for bit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xinfer.models.layers.vision.qwen3_5_vit import vision_layout
from xinfer.models.interface import BucketSpec, PreparedRequest
from xinfer.models.registry import WALL_OSS_1_0_NANO_ID
from xinfer.models.processing import (
    SegmentTokenizer,
    check_image_preprocess_device,
    device_image,
    is_jpeg,
    map_views,
    source_size,
    text_length_range,
)
from xinfer.models.wallx import geometry
from xinfer.models.wallx.processing import (
    decode_image,
    decode_view,
    mrope_position_ids,
    pad_prompt,
    smart_resize,
    wire_view,
)
from xinfer.models.wall_oss_1_0_nano.config import (
    ACTION_TOKEN, ACTION_TOKEN_ID, PROPRI_TOKEN, PROPRI_TOKEN_ID, WallOSS10NanoConfig,
)

IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
# Limits on the pinned RTC poses (columns of both arms): positions in metres, angles in radians (slightly above π),
# grippers. The reference service rejects larger values, which have overflowed its forward pass to NaN.
RTC_POSE_LIMITS = (((0, 1, 2, 7, 8, 9), 5.0), ((3, 4, 5, 10, 11, 12), 3.2), ((6, 13), 5.0))
# Optional state fields: agent_pos_config name -> wire key under request["state"].
_OPTIONAL_STATE = {"velocity_decomposed": "velocity_decomposed", "height": "lift", "head_actions": "head_pos"}
# Action dims the desktop robot never commands.
_DESKTOP_MASKED_DOF = ("velocity_decomposed", "height", "head_actions")
_ARMS = (("left", "follow1_pos"), ("right", "follow2_pos"))


@dataclass(frozen=True)
class ProcessingConfig:
    """Deployment options.

    Attributes:
        noise_seed: Seed of the CPU generator of the initial flow noise when a request has no ``init_noise``;
            ``None`` seeds from the OS.
        prompt_bucket: Round prompt widths up to this multiple; zero keeps the actual length.
        image_preprocess_device: Where the views are decoded and resized: ``"cpu"`` in the processor (OpenCV,
            Pillow and torch, the reference path), or ``"gpu"`` by the engine on the CUDA device (``DeviceImage``
            inputs: nvJPEG and FP32 antialiased bicubic resizes), which saves the processor's decode and resize time
            but does not give the CPU path's pixels bit for bit.
    """

    noise_seed: int | None = None
    prompt_bucket: int = 16
    image_preprocess_device: str = "cpu"

    def __post_init__(self):
        check_image_preprocess_device(self.image_preprocess_device)

    def bucket_spec(self, max_length: int) -> BucketSpec:
        return BucketSpec(self.prompt_bucket, max_length=max_length)


def processing_config(value) -> ProcessingConfig:
    """Accept a ProcessingConfig, None (defaults) or a dict of its fields."""
    if value is None:
        return ProcessingConfig()
    if isinstance(value, ProcessingConfig):
        return value
    return ProcessingConfig(**value)


def wire_metadata(model_config: WallOSS10NanoConfig) -> dict:
    """The frame sent to each client on connect.

    Args:
        model_config: The checkpoint's configuration.

    Returns:
        Model name and family, action horizon and dimension, normalizer key, RTC position gap, flow steps, the RTC
        feature and its wire format (``H`` future rows, no state row).
    """
    return {"model": WALL_OSS_1_0_NANO_ID, "model_family": "qwen3_5", "action_horizon": model_config.action_horizon,
            "action_dim": model_config.action_dim, "norm_key": model_config.norm_key,
            "rtc_rope_gap": model_config.rtc_rope_gap, "flow_steps": model_config.num_steps,
            "features": {"rtc": True}, "rtc_wire_format": "desktop_future_only_h"}


def load_tokenizer(checkpoint: Path, config: WallOSS10NanoConfig):
    """Load the checkpoint tokenizer as the reference's transformers processor builds it.

    ``tokenizer.json`` is extended by the further added tokens of ``tokenizer_config.json`` (transformers adds
    them on load), then by the Wall-X tokens ``<|propri|>`` and ``<|action|>`` and the checkpoint's special tokens.

    Raises:
        ValueError: The Wall-X tokens did not receive their training ids.
    """
    from tokenizers import AddedToken, Tokenizer

    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    declared = json.loads((checkpoint / "tokenizer_config.json").read_text()).get("added_tokens_decoder", {})
    for index in sorted(map(int, declared)):
        if index >= tokenizer.get_vocab_size(with_added_tokens=True):
            spec = declared[str(index)]
            tokenizer.add_tokens([AddedToken(spec["content"], single_word=spec["single_word"], lstrip=spec["lstrip"],
                                             rstrip=spec["rstrip"], normalized=spec["normalized"],
                                             special=spec["special"])])
    tokenizer.add_tokens([PROPRI_TOKEN, ACTION_TOKEN, *config.special_tokens])
    if tokenizer.token_to_id(PROPRI_TOKEN) != PROPRI_TOKEN_ID or tokenizer.token_to_id(ACTION_TOKEN) != ACTION_TOKEN_ID:
        raise ValueError("added Wall-X tokens did not receive their training ids")
    return tokenizer


def resize_view(image: np.ndarray, target: int, config: WallOSS10NanoConfig) -> np.ndarray:
    """Resize a view as the reference does: Dataset V2's resize, then the image processor's.

    First the long side is scaled to ``target`` (the short side truncated to int), snapped to the
    ``resize_factor`` grid by ``smart_resize`` and the image resized with Pillow ``BICUBIC``. The Qwen2-VL fast
    image processor then applies ``smart_resize`` with its own pixel bounds and resizes again with torch's
    antialiased bicubic interpolation on uint8, which upscales the small desktop views (``min_pixels``).

    Args:
        image: ``[H, W, 3]`` uint8 RGB.
        target: Long-side target in pixels; ``-1`` keeps the input size.
        config: The checkpoint's configuration.

    Returns:
        ``[h, w, 3]`` uint8 RGB, ``h`` and ``w`` multiples of ``patch_size · merge_size``.
    """
    from PIL import Image

    (hs, ws), (h, w) = resize_sizes(*image.shape[:2], target, config)
    resized = np.array(Image.fromarray(image).resize((ws, hs), Image.Resampling.BICUBIC))
    if (h, w) == (hs, ws):
        return resized
    # [1, 3, h, w] in channels-last layout, as torchvision receives the stacked Pillow images.
    pixels = torch.from_numpy(resized).permute(2, 0, 1)[None]
    pixels = torch.nn.functional.interpolate(pixels, size=[h, w], mode="bicubic", align_corners=False,
                                             antialias=True)
    return pixels[0].permute(1, 2, 0).contiguous().numpy()


def resize_sizes(h0: int, w0: int, target: int, config: WallOSS10NanoConfig) -> tuple[tuple[int, int], tuple[int, int]]:
    """``(h, w)`` after each of ``resize_view``'s two resizes of an ``h0 × w0`` image (equal when the second is
    skipped)."""
    if target == -1:
        h1, w1 = h0, w0
    elif w0 > h0:
        w1, h1 = target, int(target * h0 / w0)
    else:
        h1, w1 = target, int(target * w0 / h0)
    first = smart_resize(h1, w1, config.resize_factor, config.resize_min_pixels, config.resize_max_pixels)
    # On a grid-aligned size within max_pixels this equals the transformers smart_resize the processor uses.
    return first, smart_resize(*first, config.patch_size * config.merge_size, config.min_pixels, config.max_pixels)


# Resize recipe of the device path (DeviceImage.resize), as resize_view: an antialiased bicubic resize (Pillow
# BICUBIC), then, when the size changes, another (torch, antialiased bicubic), each stored as uint8.
DEVICE_RESIZE = (("bicubic", True), ("bicubic", True))


# --------------------------------------------------------------------------
# State and actions
# --------------------------------------------------------------------------


def _segments(layout: tuple[tuple[str, int], ...]) -> dict[str, slice]:
    """Map each layout name, without its ``follow_`` / ``master_`` role prefix, to its slice."""
    out, start = {}, 0
    for key, dim in layout:
        out[key.removeprefix("follow_").removeprefix("master_")] = slice(start, start + dim)
        start += dim
    return out


def arm_state(state: Mapping[str, Any], side: str, legacy: str) -> np.ndarray:
    """One arm's ``[x, y, z, roll, pitch, yaw, gripper]`` state, float64 ``[7]``.

    Reads ``follow_<side>_position`` / ``_rotation`` / ``_gripper`` or, for each one missing, the matching part
    of the legacy 7-value field ``legacy`` (``follow1_pos`` / ``follow2_pos``).

    Raises:
        ValueError: A legacy field without 7 finite values, or a part of the wrong size or not finite.
    """
    if legacy in state:
        values = np.asarray(state[legacy], dtype=np.float64).reshape(-1)
        if values.size != 7 or not np.isfinite(values).all():
            raise ValueError(f"{legacy} must contain 7 finite values")
    parts = []
    for part, width, columns in (("position", 3, slice(0, 3)), ("rotation", 3, slice(3, 6)),
                                 ("gripper", 1, slice(6, 7))):
        key = f"follow_{side}_{part}"
        value = state[key] if key in state else values[columns] if legacy in state else []
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if value.size != width or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain {width} finite values")
        parts.append(value)
    return np.concatenate(parts)


def build_state(state: Mapping[str, Any], config: WallOSS10NanoConfig):
    """Return ``(agent_pos [1, state_dim] float64, mask [1, state_dim] float64, arms)``.

    The state follows ``config.agent_pos_config``: per arm the position, the rotation as 6D (first two rows of the
    ZYX Euler rotation matrix) and the gripper, from ``arm_state``; ``velocity_decomposed``, ``height`` and
    ``head_actions`` from the optional wire fields ``velocity_decomposed``, ``lift`` and ``head_pos``, where a
    missing field, or one of the wrong size, gives zeros and a zero mask.

    Returns:
        ``(agent_pos, mask, arms)``; ``arms`` maps ``"left"`` / ``"right"`` to the ``arm_state`` arrays, the
        reference pose of the relative actions.

    Raises:
        ValueError: Malformed arm state or an unsupported ``agent_pos_config`` key.
    """
    arms = {side: arm_state(state, side, legacy) for side, legacy in _ARMS}
    parts, mask = [], []
    for key, dim in config.agent_pos_config:
        name = key.removeprefix("follow_").removeprefix("master_")
        side, _, field = name.partition("_")
        value = None
        if side in arms and field == "ee_cartesian_pos":
            value = arms[side][None, 0:3]
        elif side in arms and field == "ee_rotation_6D":
            value = geometry.euler_to_matrix_zyx_6d_nb(arms[side][None, 3:6])
        elif side in arms and field == "gripper":
            value = arms[side][None, 6:7]
        elif name in _OPTIONAL_STATE:
            wire = state.get(_OPTIONAL_STATE[name])
            if wire is not None and np.asarray(wire).size == dim:
                value = np.asarray(wire).reshape(1, dim)
        else:
            raise ValueError(f"unsupported state key {key!r}")
        parts.append(np.zeros((1, dim)) if value is None else value)
        mask.append(np.full((1, dim), 0.0 if value is None else 1.0))
    return np.concatenate(parts, axis=-1).astype(np.float64), np.concatenate(mask, axis=-1), arms


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"RTC {name} must be an integer")
    return int(value)


def parse_rtc(observation: Mapping[str, Any], config: WallOSS10NanoConfig):
    """Admit a request's RTC fields, as the reference's desktop protocol (fixed shift, no state row).

    The client sends ``prev_action_chunk`` ``[N, 14]`` (or ``[1, N, 14]``): its previous response shifted by
    ``s_model`` executed rows and padded at the end. ``N = H`` is a response of future poses only; ``N = H + 1``
    starts with an anchor row, which is not used. ``s_model`` bounds the rows still valid, and the delay is clamped
    to them; a chunk with no valid row left is served as a flow request.

    Returns:
        ``(poses [d, 14] float32, d)``: the pinned absolute poses, left arm in columns 0–6 and right arm in 7–13,
        each ``[x, y, z, roll, pitch, yaw, gripper]``; ``None`` for a flow request.

    Raises:
        ValueError: ``rtc_mode`` other than ``training_time``; a delay that is not an integer in
            ``[1, min(H, rtc_max_delay)]``; a chunk not ``[H or H + 1, 14]``; ``s_model`` not an integer in range;
            non-finite pinned poses or poses beyond ``RTC_POSE_LIMITS``.
    """
    if not observation.get("use_rtc"):
        return None
    if observation.get("rtc_mode") not in (None, "training_time"):
        raise ValueError("only rtc_mode 'training_time' is supported")
    horizon = config.action_horizon
    delay = _integer(observation.get("inference_delay"), "inference_delay")
    if not 0 < delay <= min(horizon, config.rtc_max_delay):
        raise ValueError(f"inference_delay {delay} outside [1, {min(horizon, config.rtc_max_delay)}]")
    try:
        chunk = np.asarray(observation.get("prev_action_chunk"), dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("prev_action_chunk must be numeric") from error
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim != 2 or chunk.shape[1] != 14 or chunk.shape[0] not in (horizon, horizon + 1):
        raise ValueError(f"prev_action_chunk must be [{horizon} or {horizon + 1}, 14], got {chunk.shape}")
    anchor = chunk.shape[0] - horizon
    shift = _integer(observation.get("s_model"), "s_model")
    if not 0 <= shift <= horizon:
        raise ValueError(f"s_model {shift} outside [0, {horizon}]")
    available = horizon - shift
    if available <= 0:
        return None
    delay = min(delay, available)
    poses = chunk[anchor:anchor + delay]
    if not np.isfinite(poses).all():
        raise ValueError("prev_action_chunk has non-finite pinned poses")
    for columns, limit in RTC_POSE_LIMITS:
        if np.abs(poses[:, columns]).max() > limit:
            raise ValueError(f"prev_action_chunk pinned poses exceed {limit} in columns {columns}")
    return poses, delay


def rtc_prefix(poses: np.ndarray, reference: np.ndarray, config: WallOSS10NanoConfig) -> np.ndarray:
    """Absolute pinned poses ``[d, 14]`` → relative actions in the checkpoint layout ``[d, D]`` (float32).

    For each arm, relative to the reference state: position minus the state position, rotation
    ``R_abs · R_stateᵀ`` as 6D (first two rows) and the gripper unchanged; the other action dims stay 0. Computed in
    float64 from the float32 poses and state.

    Args:
        poses: Pinned poses from ``parse_rtc``.
        reference: ``[14]`` float32 state, left then right ``arm_state``.
        config: The checkpoint's configuration.
    """
    act = poses.astype(np.float64)
    ref = reference.astype(np.float64)
    segments = _segments(config.dof_config)
    out = np.zeros((act.shape[0], config.action_dim), dtype=np.float32)
    for side, off in (("left", 0), ("right", 7)):
        rot_ref = geometry.euler_to_matrix_zyx_6d_nb(ref[off + 3:off + 6][None])[0]
        rot_abs = geometry.euler_to_matrix_zyx_6d_nb(act[:, off + 3:off + 6])
        out[:, segments[f"{side}_ee_cartesian_pos_relative"]] = act[:, off:off + 3] - ref[off:off + 3][None]
        out[:, segments[f"{side}_ee_rotation_6D_relative"]] = geometry.compute_delta_from_state_and_abs_rot(
            rot_abs, rot_ref)
        out[:, segments[f"{side}_gripper"]] = act[:, off + 6:off + 7]
    return out


# --------------------------------------------------------------------------
# Processors
# --------------------------------------------------------------------------


class WallOSS10NanoInputProcessor:
    """Turn a desktop robot request into a ``PreparedRequest`` (CPU only, safe to call concurrently).

    Args:
        checkpoint: Checkpoint directory (``tokenizer.json``, ``tokenizer_config.json``).
        model_config: The checkpoint's configuration.
        config: ``ProcessingConfig``, dict or ``None`` (defaults).

    Attributes:
        config: The resolved ``ProcessingConfig``.
        model_config: The checkpoint's configuration.
        tokenizer: Checkpoint tokenizer with the Wall-X tokens added.
        padding_action: Normalized zero action ``[1, H, D]`` float32, clamped like the action normalizer.
        dof_mask: ``[1, H, D]`` float64; 0 on the dims the desktop robot never commands (velocity, height,
            head), 1 elsewhere.

    Raises:
        ValueError: The action layout lacks a relative position, 6D rotation or gripper segment for an arm.
    """

    def __init__(
        self, checkpoint: str | Path, model_config: WallOSS10NanoConfig, config: ProcessingConfig | None = None,
    ):
        checkpoint = Path(checkpoint)
        self.config = processing_config(config)
        self.model_config = model_config
        self.bucket = self.config.bucket_spec(model_config.max_seq_length - model_config.action_horizon)
        self.tokenizer = load_tokenizer(checkpoint, model_config)
        self._tokenize = SegmentTokenizer(self.tokenizer)
        segments = _segments(model_config.dof_config)
        for side in ("left", "right"):
            for field in ("ee_cartesian_pos_relative", "ee_rotation_6D_relative", "gripper"):
                if f"{side}_{field}" not in segments:
                    raise ValueError(f"action layout lacks {side}_{field}")
        self._noise = torch.Generator()
        if self.config.noise_seed is None:
            self._noise.seed()
        else:
            self._noise.manual_seed(self.config.noise_seed)
        norm = model_config.action_norm
        padding = ((np.zeros_like(norm.min) - norm.min) / norm.delta) * 2 - 1
        if model_config.clamp_normalized:
            padding = np.clip(padding, -1, 1)
        self.padding_action = padding[None]
        h, d = model_config.action_horizon, model_config.action_dim
        self.dof_mask = np.ones((1, h, d), dtype=np.float64)
        start = 0
        for key, dim in model_config.dof_config:
            if key in _DESKTOP_MASKED_DOF:
                self.dof_mask[:, :, start:start + dim] = 0.0
            start += dim

    def _normalize_state(self, x: np.ndarray) -> np.ndarray:
        """``(x - min) / delta · 2 - 1`` in float64, clamped to ``[-1, 1]`` when the checkpoint clamps."""
        norm = self.model_config.state_norm
        x = ((x - norm.min.astype(np.float64)) / norm.delta.astype(np.float64)) * 2 - 1
        return np.clip(x, -1, 1) if self.model_config.clamp_normalized else x

    def _normalize_prefix(self, x: np.ndarray) -> np.ndarray:
        """``(x - min) / delta · 2 - 1`` with the float32 statistics of steps ``0 .. d-1``, never clamped, so the
        pinned rows map back exactly."""
        norm = self.model_config.action_norm
        d = x.shape[0]
        return ((x - norm.min[:d]) / norm.delta[:d]) * 2 - 1

    def _prompt(self, instruction: str, state_text: str, grids) -> str:
        c = self.model_config
        merge = c.merge_size ** 2
        system = (
            "<|im_start|>system\nYou are an embodied vision-language-action (VLA) model controlling the robot with "
            f"language instructions.\n Embodiment: {c.embodiment}\n Camera Setup:"
            + "".join(f" {camera.label}," for camera in c.cameras)
            + f"\n Frequency: {c.frequency_hz}HZ\n Action Space: {'Rel' if c.relative_action else 'Abs'} EEF\n"
            "<|im_end|>\n"
        )
        observation = "".join(f" {camera.label}: {VISION_START}{IMAGE_PAD * (t * gh * gw // merge)}{VISION_END}"
                              for camera, (t, gh, gw) in zip(c.cameras, grids))
        task = f"Task: {instruction}" if instruction else ""
        return (
            f"{system}<|im_start|>user\nObservation:{observation}\n{task}\nPredict the next action in robot action.\n"
            f"Proprioception: {state_text}\n<|im_end|>\n<|im_start|>assistant\n" + ACTION_TOKEN * c.action_horizon
        )

    def __call__(self, observation: Mapping[str, Any]) -> PreparedRequest:
        """Prepare one request: images, state string, prompt, positions and RTC prefix.

        Args:
            observation: Desktop robot request. Reads:

                - ``views``: the image of every camera under one of its ``wire_keys`` (HWC uint8 RGB array or
                  raw/base64 JPEG);
                - ``state``: per arm ``follow_<side>_position`` / ``_rotation`` / ``_gripper`` (xyz in metres,
                  roll/pitch/yaw ZYX Euler angles in radians, gripper) or the legacy ``follow1_pos`` /
                  ``follow2_pos``, and the optional ``velocity_decomposed``, ``lift`` and ``head_pos``;
                - ``instruction``: task text (whitespace collapsed; it becomes the prompt's ``Task:`` line);
                - ``num_flow_steps`` / ``num_steps``: optional, must equal ``model_config.num_steps``;
                - ``init_noise``: optional ``[H, D]`` initial flow state;
                - RTC: ``use_rtc``, ``rtc_mode``, ``inference_delay``, ``s_model`` and ``prev_action_chunk``
                  (see ``parse_rtc``).

        Returns:
            ``PreparedRequest`` with CPU ``tensors`` (``P`` prompt tokens, ``H`` horizon, ``D`` action dim):

            - ``input_ids`` ``[1, P]`` int64 and ``position_ids`` ``[3, 1, P]`` int64: the prompt and its M-RoPE
              temporal/height/width positions (computed over the prompt followed by ``H`` ``<|action|>`` tokens,
              as the reference does);
            - ``image_<i>`` ``[h_i, w_i, 3]`` uint8 RGB: the resized views in camera order. The vision stage
              normalizes and packs them with ``xinfer.models.layers.image_pack.ImagePack`` and the WALL-OSS-0.5
              ``pixel_lookup_table`` of ``image_mean`` / ``image_std`` (the same Qwen2-VL fast processor
              arithmetic), then casts to BF16 as the reference does;
            - ``vision_positions``, ``vision_pos_index``, ``vision_pos_weight`` and the host list
              ``vision_cu_seqlens``: the ``vision_layout`` of the image grids;
            - ``image_token_index`` ``[N_image_tokens]`` int64: positions of the ``<|image_pad|>`` tokens;
            - ``prompt_len`` (int ``P``): bucket width; ``prompt_valid`` ``[1]`` int64: valid token count;
            - ``dof_mask`` ``[1, H, D]`` float32 and ``dof_mask_all`` (bool): commanded action dims;
            - ``padding_action`` ``[1, H, D]`` float32: the normalized zero action, the target of the
              uncommanded dims;
            - ``noise`` ``[1, H, D]`` float32: ``init_noise``, else drawn from the processor's generator;
            - ``action_positions`` ``[3, 1, H]`` int64: the action tokens' positions; the rows after the pinned
              RTC rows (every row without RTC) skip ``rtc_rope_gap`` positions;
            - ``action_mask`` ``[1, 1, H, P + H]`` bool: the prompt and action keys each action row sees (pinned
              rows do not see the other action rows);
            - RTC only: ``rtc_actions`` ``[1, H, D]`` float32, the normalized relative pinned actions in the first
              ``d`` rows (zeros after), and ``inference_delay`` ``[1]`` int64 (``d``).

            ``meta`` holds ``arms`` (the state pose, see ``build_state``), ``inference_delay`` (int), and
            ``prompt_len_range``: prompt lengths reachable by varying the state, used for bucket warmup.

        Raises:
            ValueError: Inadmissible step count, RTC fields or ``init_noise``; missing, undecodable or malformed
                images; malformed state; an instruction over ``max_instruction_words`` words; or a sequence over
                ``max_seq_length`` tokens.
            TypeError: An instruction that is not a string.
        """
        c = self.model_config
        h, d = c.action_horizon, c.action_dim
        steps = observation.get("num_flow_steps", observation.get("num_steps", c.num_steps))
        if type(steps) is not int or steps != c.num_steps:
            raise ValueError(f"this model requires {c.num_steps} flow steps")
        rtc = parse_rtc(observation, c)
        state = observation.get("state") or {}
        instruction = observation.get("instruction", "")
        if isinstance(instruction, np.ndarray):
            instruction = instruction.item()
        if not isinstance(instruction, str):
            raise TypeError(f"instruction must be a string, got {type(instruction).__name__}")
        words = instruction.split()
        if len(words) > c.max_instruction_words:
            raise ValueError(f"instruction has {len(words)} words, more than {c.max_instruction_words}")

        # Images, in prompt order.
        views = observation["views"]

        def wire(camera):
            key = next((k for k in camera.wire_keys if views.get(k) is not None), None)
            if key is None:
                raise ValueError(f"request has no {camera.view} image (keys {camera.wire_keys})")
            return views[key]

        if self.config.image_preprocess_device == "gpu":
            # JPEG bytes go to the device decoder; other encodings are decoded here and resized on the device.
            images = [wire_view(wire(camera)) for camera in c.cameras]
            images = [image if isinstance(image, np.ndarray) or is_jpeg(image) else decode_image(image)
                      for image in images]
            sizes = [resize_sizes(*source_size(image), c.resolution.get(camera.view, -1), c)
                     for image, camera in zip(images, c.cameras)]
            images = [device_image(image, size, DEVICE_RESIZE) for image, size in zip(images, sizes)]
            shapes = [size[-1] for size in sizes]
        else:
            images = [torch.from_numpy(image) for image in map_views(
                lambda camera: resize_view(decode_view(wire(camera)), c.resolution.get(camera.view, -1), c), c.cameras)]
            shapes = [image.shape[:2] for image in images]
        grids = [(1, h // c.patch_size, w // c.patch_size) for h, w in shapes]

        # State: normalized, then binned into the prompt.
        agent_pos, agent_mask, arms = build_state(state, c)
        proprio = self._normalize_state(agent_pos)[None]  # [1, 1, state_dim]
        digits = np.digitize(proprio, np.linspace(-1, 1, c.state_bins + 1)[:-1]) - 1
        values = digits[0, 0][agent_mask[0] != 0]
        state_text = " ".join(map(str, values))

        token_ids = np.asarray(self._tokenize(self._prompt(" ".join(words), state_text, grids)), dtype=np.int64)[None]
        if token_ids.shape[1] > c.max_seq_length:
            raise ValueError(f"sequence of {token_ids.shape[1]} tokens exceeds max_seq_length {c.max_seq_length}")
        prompt_len = token_ids.shape[1] - h
        noise = observation.get("init_noise")
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            if noise.shape != (h, d) or not np.isfinite(noise).all():
                raise ValueError(f"init_noise must be finite [{h}, {d}]")
            noise = torch.from_numpy(noise.copy())[None]
        else:
            noise = torch.randn((1, h, d), generator=self._noise)
        p, h = prompt_len, c.action_horizon
        positions = mrope_position_ids(token_ids, grids, c.image_token_id, c.merge_size)  # [3, 1, P + H]
        delay = 0
        rtc_actions = None
        if rtc is not None:
            poses, delay = rtc
            # The reference state of the prefix is the float32 wire pose, unlike the float64 prompt state.
            reference = np.concatenate([arms["left"], arms["right"]]).astype(np.float32)
            rtc_actions = self._normalize_prefix(rtc_prefix(poses, reference, c))
        # Action rows after the pinned RTC rows (all rows without RTC) skip rtc_rope_gap positions; pinned rows do
        # not see the other action rows.
        action_positions = positions[..., p:].copy()
        action_positions[..., delay:] += c.rtc_rope_gap
        action_mask = np.ones((1, 1, h, p + h), dtype=bool)
        action_mask[0, 0, :delay, p + delay:] = False
        layout = vision_layout(grids, c.model.vision)
        tensors = dict(
            input_ids=torch.from_numpy(token_ids[:, :p]),
            position_ids=torch.from_numpy(positions[..., :p].copy()),
            **{f"image_{i}": image for i, image in enumerate(images)},
            vision_positions=layout["positions"],
            vision_pos_index=layout["pos_index"],
            vision_pos_weight=layout["pos_weight"],
            vision_cu_seqlens=layout["cu_seqlens"],
            image_token_index=torch.from_numpy(np.flatnonzero(token_ids[0, :p] == c.image_token_id)),
            prompt_len=p,
            prompt_valid=torch.tensor([p]),
            dof_mask=torch.from_numpy(self.dof_mask.astype(np.float32)),
            dof_mask_all=bool(self.dof_mask.all()),
            padding_action=torch.from_numpy(self.padding_action.copy()),
            noise=noise,
            action_positions=torch.from_numpy(action_positions),
            action_mask=torch.from_numpy(action_mask),
        )
        if rtc_actions is not None:
            padded = np.zeros((1, h, c.action_dim), dtype=np.float32)
            padded[0, :delay] = rtc_actions
            tensors["rtc_actions"] = torch.from_numpy(padded)
            tensors["inference_delay"] = torch.tensor([delay], dtype=torch.int64)
        # The -1 sentinel shares its sign with the separator token; only digit tokens vary.
        length_range = text_length_range(p, sum(len(str(abs(v))) for v in values), len(values),
                                          max_tokens=len(str(c.state_bins - 1)), max_length=self.bucket.max_length)
        return PreparedRequest(pad_prompt(tensors, self.bucket.width(p)),
                               dict(arms=arms, inference_delay=delay, prompt_len_range=length_range))


class WallOSS10NanoOutputProcessor:
    """``predict_action_norm`` ``[1, H, D]`` (normalized) → desktop robot response.

    Args:
        model_config: The checkpoint's configuration (action statistics, ``dof_config`` slices).
    """

    def __init__(self, model_config: WallOSS10NanoConfig):
        self.model_config = model_config
        self._segments = _segments(model_config.dof_config)

    def __call__(self, outputs: Mapping[str, Any], request: PreparedRequest) -> dict[str, Any]:
        """Denormalize the actions and compose them with the request's state into absolute arm poses.

        Denormalization ``(x + 1) / 2 · delta + min`` runs here in float32 with the per-step statistics. For each
        arm: position = state position + relative position; rotation = 6D delta composed with the state rotation
        (``R_delta · R_state``), converted to canonical ZYX Euler angles; gripper taken as is; computed in float64
        from the float64 state, then cast to float32.

        Args:
            outputs: Final stage outputs; reads ``predict_action_norm`` ``[1, H, D]`` float32.
            request: The ``PreparedRequest`` of this request; reads ``meta["arms"]``.

        Returns:
            Response dict:

            - ``follow1_pos`` / ``follow2_pos``: left / right arm, lists of ``H`` rows
              ``[x, y, z, roll, pitch, yaw, gripper]`` (float32 values);
            - ``predict_action_norm``: ``[1, H, D]`` float32, the model output;
            - ``action_pred``: ``[H, 14]`` float32, ``follow1_pos`` and ``follow2_pos`` side by side.

        Raises:
            FloatingPointError: A non-finite pose.
        """
        c = self.model_config
        normalized = np.asarray(outputs["predict_action_norm"], dtype=np.float32).reshape(1, c.action_horizon,
                                                                                            c.action_dim)
        action = ((normalized[0] + 1) / 2) * c.action_norm.delta + c.action_norm.min
        response = {}
        for side, key in _ARMS:
            state = request.meta["arms"][side]
            position = state[None, 0:3] + action[:, self._segments[f"{side}_ee_cartesian_pos_relative"]]
            state6 = geometry.euler_to_matrix_zyx_6d_nb(state[None, 3:6])[0]
            rotation = geometry.compose_state_and_delta_to_abs_6d(
                action[:, self._segments[f"{side}_ee_rotation_6D_relative"]], state6)
            arm = np.concatenate([position.astype(np.float32),
                                  geometry.so3_to_euler_zyx_batch_nb(rotation).astype(np.float32),
                                  action[:, self._segments[f"{side}_gripper"]]], axis=-1)
            if not np.isfinite(arm).all():
                raise FloatingPointError(f"non-finite {key}")
            response[key] = arm.tolist()
        response["predict_action_norm"] = normalized
        response["action_pred"] = np.concatenate(
            [np.asarray(response["follow1_pos"], dtype=np.float32), np.asarray(response["follow2_pos"], np.float32)],
            axis=-1)
        return response
