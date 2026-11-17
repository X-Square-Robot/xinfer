"""Wall-X 2606 input and output processing (CPU only).

The InputProcessor turns a desktop robot request into model inputs; the OutputProcessor turns the model's
normalized actions into the robot response. Behavior follows the production Wall-X 2606 service (with wall-x d882c37
and x2robot_dataset e8ebcf9) as deployed with its rtx5090 profile: uint8 patch rows normalized
by a lookup table on the device, responses of ``H`` future poses without the state row (``omit_state_head``), and
prompts without padding (``prompt_bucket=0``). The default pads prompts to 16-token buckets with masked padding.
The exact-length configuration is tested against that service bit for bit.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xinfer.models.layers.vision.qwen3_5_vit import vision_layout
from xinfer.models.interface import BucketSpec, PreparedRequest
from xinfer.models.processing import (
    _view_pool,
    _wirefast,
    SegmentTokenizer,
    check_image_preprocess_device,
    device_image,
    device_images_ready,
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
from xinfer.models.wallx2b.config import ACTION_TOKEN, ACTION_TOKEN_ID, PROPRI_TOKEN, PROPRI_TOKEN_ID, Wallx2bConfig

try:  # PREPARE-C PC1/PC2: built GIL-releasing prepare pack (pure CPython+NumPy C API).
    from xinfer import _preparefast  # built for this platform; every user falls back below
except Exception:  # noqa: BLE001 - the pack is an optimization; prepare must not fail without it
    _preparefast = None

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


def _wire_view_fast(value):
    """``wire_view`` with the GIL released for the wire decodes (the 2B-HOST port of the r11 knife A).

    The base64 decode of a view is a single C call that holds the GIL for its whole
    execution, so the view pool's parallelism is nominal and the device tickets are not
    put until late in prepare (probe-timeline-r10, wall_oss lineage). The built
    ``_wirefast`` helper decodes canonical base64 and copies bytes-like buffers under
    ``Py_BEGIN_ALLOW_THREADS``; anything it declines falls back to ``wire_view``, so the
    decoded values are equal by construction. Arrays take ``wire_view`` unchanged.
    """
    if isinstance(value, str):
        if _wirefast is not None:
            decoded = _wirefast.b64decode(value)
            if decoded is not None:
                return decoded
    elif isinstance(value, (bytearray, memoryview)):
        if _wirefast is not None:
            copied = _wirefast.to_bytearray(value)
            if copied is not None:
                return copied
    return wire_view(value)


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


def wire_metadata(model_config: Wallx2bConfig) -> dict:
    """The frame sent to each client on connect (the Wall-X 2606 service's fields).

    Args:
        model_config: The checkpoint's configuration.

    Returns:
        Model name and family, action horizon and dimension, normalizer key, RTC position gap, flow steps, the RTC
        feature and its wire format (``H`` future rows, no state row).
    """
    return {"model": "wallx2606", "model_family": "qwen3_5", "action_horizon": model_config.action_horizon,
            "action_dim": model_config.action_dim, "norm_key": model_config.norm_key,
            "rtc_rope_gap": model_config.rtc_rope_gap, "flow_steps": model_config.num_steps,
            "features": {"rtc": True}, "rtc_wire_format": "desktop_future_only_h"}


def load_tokenizer(checkpoint: Path, config: Wallx2bConfig):
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


def resize_view(image: np.ndarray, target: int, config: Wallx2bConfig) -> np.ndarray:
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


def resize_sizes(h0: int, w0: int, target: int, config: Wallx2bConfig) -> tuple[tuple[int, int], tuple[int, int]]:
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


def build_state(state: Mapping[str, Any], config: Wallx2bConfig):
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


def parse_rtc(observation: Mapping[str, Any], config: Wallx2bConfig):
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


def rtc_prefix(poses: np.ndarray, reference: np.ndarray, config: Wallx2bConfig) -> np.ndarray:
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


class Wallx2bInputProcessor:
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

    def __init__(self, checkpoint: str | Path, model_config: Wallx2bConfig, config: ProcessingConfig | None = None):
        self._image_prefetch = None
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
        # HOST-GAP K5 hotfix: pinning the prepared tensors needs an accelerator in THIS
        # process (frontend processes under ``xinfer serve`` have none; capacity cap6
        # crash 2026-11-15). Probe once and fall back to plain tensors when absent.
        try:
            self._pin_prepared = torch.zeros(1).pin_memory() is not None
        except Exception:  # noqa: BLE001 - any deviceless or pinned-alloc failure disables it
            self._pin_prepared = False
        # HOST-FINAL HF3: with batched admission uploads on (AG2, XINFER_UPLOAD_BATCH,
        # default on) the engine packs every CPU tensor of the request into its own
        # pinned slab and issues one H2D copy, so per-tensor pinning in prepare
        # (K5/PC3, ~0.17 ms of CPU per request inside the DVPP wave window) buys
        # nothing on the batched path, and its memcpys compete with the wave's host
        # supply chain (the PC5 lesson). Keep the pin only when batching is off (the
        # fallback _to_device pins on the engine thread then, as before K5). Values,
        # shapes and every uploaded byte are unchanged; pinned-ness is not observable
        # downstream because AG2 packs from any host memory.
        if self._pin_prepared and os.environ.get("XINFER_UPLOAD_BATCH", "1") not in ("0", "false", "False"):
            self._pin_prepared = False
        # PREPARE-C PC3: the built _pincpp pins the prepared tensors with the GIL
        # released (the python pin loop held it ~0.5 ms per request, the largest
        # main-thread GIL item left on the K1-K6 arm; probe start.gilphase
        # 2026-11-15). Probed once on the same K5 guard: an unpinned tensor must
        # come back pinned and a non-tensor must pass through, else python pins.
        self._pin_cpp = None
        if self._pin_prepared:
            try:
                from xinfer import _pincpp

                out = _pincpp.pin_batch({"probe": torch.zeros(1), "host": "x"})
                if out["probe"].is_pinned() and out["host"] == "x":
                    self._pin_cpp = _pincpp.pin_batch
            except Exception:  # noqa: BLE001 - any ABI or probe failure falls back to python pin
                self._pin_cpp = None
        # PREPARE-C PC2: the state digitize bins and the float64 normalization
        # statistics are frozen per config; precompute them once instead of per
        # request (same values numpy recomputed every call).
        self._state_bins = np.linspace(-1, 1, model_config.state_bins + 1)[:-1]
        self._state_min64 = model_config.state_norm.min.astype(np.float64)
        self._state_delta64 = model_config.state_norm.delta.astype(np.float64)

    def bind_image_prefetch(self, prefetch) -> None:
        self._image_prefetch = prefetch

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
              normalizes and packs them with ``xinfer.models.layers.image_pack.ImagePack`` and the Wall-OSS
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
            # Per-view handoff (the 2B-HOST port of the walloss r10/r11 prepare knives): every wire
            # decode runs in the view pool through _wire_view_fast, which releases the GIL for the
            # base64 decode, and each result is waited for in camera order (each wait also
            # releases the GIL); device_image then claims a per-view prefetch slot and hands the
            # view to the engine's device-image thread immediately, so the DVPP chain overlaps
            # the remaining host prepare (state, prompt, tokenizer, positions) instead of
            # running after it. Same decodes, same sizes, same values, camera order.
            pool = _view_pool()
            # HOST-GAP K6: the whole per-view chain (wire decode, size, device_image, prefetch
            # ticket) runs inside the pool worker. The ticket used to be claimed after the main
            # thread waited for the view (ticket0 ~0.6 ms into prepare, timeline probe
            # 2026-11-15), serializing the DVPP wave behind the main thread GIL bursts; from
            # the worker the ticket goes out as soon as that view bytes exist, the wave starts
            # ~0.4 ms earlier and its tail no longer pokes into the engine window. claim and
            # submit are lock-guarded, every helper in the chain is a pure function, and the
            # results are still consumed in camera order with the same values.
            def handoff(camera):
                value = wire(camera)
                # PREPARE-C PC4: the wire decode, the JPEG header scan and both
                # smart_resize calls of this worker chain in one GIL-releasing C
                # call (b64 already ran GIL-released via _wirefast; this also
                # removes the python wrapper, is_jpeg, jpeg_size's python marker
                # walk and resize_sizes' arithmetic from the pool worker, whose
                # GIL slices serialize the wave's ticket handoffs). device_image
                # (frombuffer + prefetch ticket) stays python. None = the python
                # chain below unchanged, byte for byte.
                if _preparefast is not None:
                    fast = _preparefast.handoff(value, c.resolution.get(camera.view, -1),
                                                c.resize_factor, c.resize_min_pixels,
                                                c.resize_max_pixels, c.patch_size * c.merge_size,
                                                c.min_pixels, c.max_pixels)
                    if fast is not None:
                        data, _src, size = fast
                        return device_image(data, size, DEVICE_RESIZE, prefetch=self._image_prefetch), size[-1]
                image = _wire_view_fast(value)
                if not (isinstance(image, np.ndarray) or is_jpeg(image)):
                    image = decode_image(image)
                size = resize_sizes(*source_size(image), c.resolution.get(camera.view, -1), c)
                return device_image(image, size, DEVICE_RESIZE, prefetch=self._image_prefetch), size[-1]

            futures = [pool.submit(handoff, camera) for camera in c.cameras]
            images, shapes = [], []
            for future in futures:
                image, shape = future.result()
                images.append(image)
                shapes.append(shape)
            # Close this request's hand-over: the device-image thread starts the views' decode
            # wave now, before the request's admission item reaches it.
            device_images_ready(self._image_prefetch)
        else:
            images = [torch.from_numpy(image) for image in map_views(
                lambda camera: resize_view(decode_view(wire(camera)), c.resolution.get(camera.view, -1), c), c.cameras)]
            shapes = [image.shape[:2] for image in images]
        grids = [(1, h // c.patch_size, w // c.patch_size) for h, w in shapes]

        # State: normalized, then binned into the prompt.
        agent_pos, agent_mask, arms = build_state(state, c)
        # PREPARE-C PC2: normalize + digitize + state_text in one GIL-releasing C
        # call (same float64 arithmetic in numpy's op order, digitize's
        # searchsorted-'right' semantics, str()'s digits). None = run the python
        # path below unchanged.
        state_fast = None
        if _preparefast is not None:
            state_fast = _preparefast.state_text(agent_pos, agent_mask, self._state_min64,
                                                 self._state_delta64, self._state_bins,
                                                 c.clamp_normalized)
        if state_fast is not None:
            state_text, values, n_values, sum_digits = state_fast
        else:
            proprio = self._normalize_state(agent_pos)[None]  # [1, 1, state_dim]
            digits = np.digitize(proprio, self._state_bins) - 1
            values = digits[0, 0][agent_mask[0] != 0]
            state_text = " ".join(map(str, values))
            n_values = len(values)
            sum_digits = sum(len(str(abs(v))) for v in values)

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
        delay = 0
        rtc_actions = None
        if rtc is not None:
            poses, delay = rtc
            # The reference state of the prefix is the float32 wire pose, unlike the float64 prompt state.
            reference = np.concatenate([arms["left"], arms["right"]]).astype(np.float32)
            rtc_actions = self._normalize_prefix(rtc_prefix(poses, reference, c))
        layout = vision_layout(grids, c.model.vision)
        # PREPARE-C PC1: mrope + action positions/mask + pad_prompt + rtc zero-pad
        # + image_token_index in one GIL-releasing C call that returns the FINAL
        # padded arrays (K4's integer walk, K3's padding values, the RTC pattern),
        # fresh per request exactly as the python path built them. None = the
        # python path below runs unchanged (and pads after building the dict).
        tail = None
        if _preparefast is not None:
            tail = _preparefast.tail(token_ids, c.image_token_id, c.merge_size, grids, delay,
                                     c.rtc_rope_gap, self.bucket.width(p), rtc_actions, p, h)
        if tail is not None:
            tensors = dict(
                input_ids=torch.from_numpy(tail["input_ids"]),
                position_ids=torch.from_numpy(tail["position_ids"]),
                **{f"image_{i}": image for i, image in enumerate(images)},
                vision_positions=layout["positions"],
                vision_pos_index=layout["pos_index"],
                vision_pos_weight=layout["pos_weight"],
                vision_cu_seqlens=layout["cu_seqlens"],
                image_token_index=torch.from_numpy(tail["image_token_index"]),
                prompt_len=self.bucket.width(p),
                prompt_valid=torch.from_numpy(tail["prompt_valid"]),
                dof_mask=torch.from_numpy(self.dof_mask.astype(np.float32)),
                dof_mask_all=bool(self.dof_mask.all()),
                padding_action=torch.from_numpy(self.padding_action.copy()),
                noise=noise,
                action_positions=torch.from_numpy(tail["action_positions"]),
                action_mask=torch.from_numpy(tail["action_mask"]),
            )
            if "rtc_actions" in tail:
                tensors["rtc_actions"] = torch.from_numpy(tail["rtc_actions"])
                tensors["inference_delay"] = torch.tensor([delay], dtype=torch.int64)
            padded = tensors  # already padded to the bucket width
        else:
            positions = mrope_position_ids(token_ids, grids, c.image_token_id, c.merge_size)  # [3, 1, P + H]
            # Action rows after the pinned RTC rows (all rows without RTC) skip rtc_rope_gap positions; pinned rows do
            # not see the other action rows.
            action_positions = positions[..., p:].copy()
            action_positions[..., delay:] += c.rtc_rope_gap
            action_mask = np.ones((1, 1, h, p + h), dtype=bool)
            action_mask[0, 0, :delay, p + delay:] = False
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
            padded = pad_prompt(tensors, self.bucket.width(p))
        # The -1 sentinel shares its sign with the separator token; only digit tokens vary.
        length_range = text_length_range(p, sum_digits, n_values,
                                          max_tokens=len(str(c.state_bins - 1)), max_length=self.bucket.max_length)
        # HOST-GAP K5: pin the prepared CPU tensors here, inside the DVPP wave window the
        # prefetch tickets opened (prepare overlaps the device decode), instead of at engine
        # admission where _to_device pinned them one by one on the engine thread (~0.2 ms of
        # GIL bursts between the wave end and the vision dispatch; timeline probe 2026-11-15).
        # pin_memory returns fresh pinned tensors, so the shared cached vision_layout tensors
        # are copied, never mutated; admission then uploads with non_blocking copies only.
        # PREPARE-C PC3: _pincpp.pin_batch runs the same ATen pin_memory() per
        # tensor with the GIL released (the python loop held it ~0.5 ms/request,
        # probe start.gilphase 2026-11-15); probed at construction, python loop
        # as the fallback.
        if self._pin_prepared:
            if self._pin_cpp is not None:
                prepared_tensors = self._pin_cpp(padded)
            else:
                prepared_tensors = {name: (value.pin_memory() if torch.is_tensor(value)
                                           and value.device.type == "cpu" and not value.is_pinned() else value)
                                    for name, value in padded.items()}
        else:
            prepared_tensors = padded
        return PreparedRequest(prepared_tensors,
                               dict(arms=arms, inference_delay=delay, prompt_len_range=length_range))


class Wallx2bOutputProcessor:
    """``predict_action_norm`` ``[1, H, D]`` (normalized) → desktop robot response.

    Args:
        model_config: The checkpoint's configuration (action statistics, ``dof_config`` slices).
    """

    def __init__(self, model_config: Wallx2bConfig):
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
        arms_f32 = []
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
            arms_f32.append(arm)
        response["predict_action_norm"] = normalized
        # HOST-FINAL HF1a: concatenate the arm arrays themselves instead of the
        # list round trip. tolist boxes float32 values as exact python floats,
        # and np.asarray(..., float32) casts them straight back -- a lossless
        # round trip, so values, dtype and layout are bitwise identical to the
        # lists it used to rebuild (~12 us per request).
        response["action_pred"] = np.concatenate(arms_f32, axis=-1)
        return response
