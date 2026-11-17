"""Image decoding, normalization, prompt padding and image-only M-RoPE helpers shared by Wall-X models."""

from __future__ import annotations

import base64
import math
from typing import Any

import numpy as np


def pad_prompt(tensors, width: int) -> dict:
    """Resize a prefix-only prompt and its action mask, keeping action positions and own-token masks intact.

    The Qwen3.5 backbone masks padded prompt rows before prefill and in action attention. Shrinking is used only
    for warmup; the real-token count is capped with it. Inputs are never modified in place.
    """
    import torch

    old_width = tensors["prompt_len"]
    if width == old_width:
        return dict(tensors)
    tokens, positions, mask = (tensors[k] for k in ("input_ids", "position_ids", "action_mask"))
    # HOST-GAP K3: the same padded values through numpy instead of six torch CPU dispatches
    # (new_zeros, new_ones and three cat plus clamp, ~0.45 ms per request). The concatenated
    # arrays are fresh per request, so nothing is shared; values, dtypes and shapes are
    # unchanged and the tensors are read-only downstream exactly as the torch results were.
    ids = tokens.numpy()
    pos = positions.numpy()
    msk = mask.numpy()
    valid = tensors["prompt_valid"].numpy()
    keep = min(width, old_width)
    padding = max(0, width - old_width)
    return dict(
        tensors,
        input_ids=torch.from_numpy(np.concatenate(
            (ids[:, :width], np.zeros((ids.shape[0], padding), dtype=ids.dtype)), axis=-1)),
        position_ids=torch.from_numpy(np.concatenate(
            (pos[..., :width], np.ones((*pos.shape[:-1], padding), dtype=pos.dtype)), axis=-1)),
        prompt_len=width,
        prompt_valid=torch.from_numpy(np.minimum(valid, width)),
        action_mask=torch.from_numpy(np.concatenate(
            (msk[..., :keep], np.zeros((*msk.shape[:-1], padding), dtype=msk.dtype),
             msk[..., old_width:]), axis=-1)),
    )


def pixel_lookup_table(mean, std) -> np.ndarray:
    """Per-channel uint8 → normalized float32 table, [3, 256].

    The Qwen2-VL fast processor folds the 1/255 rescale into mean and std
    (mean·255, std·255 in float32) and normalizes ``(v - m) / s`` in float32.

    Args:
        mean: Per-channel RGB mean on the [0, 1] scale.
        std: Per-channel RGB standard deviation on the [0, 1] scale.

    Returns:
        ``[3, 256]`` float32; row ``c`` maps a uint8 value of channel ``c`` to its normalized value.
    """
    ramp = np.arange(256, dtype=np.float32)
    rows = []
    for m, s in zip(mean, std):
        m255 = np.float32(np.float32(m) * np.float32(255.0))
        s255 = np.float32(np.float32(s) * np.float32(255.0))
        rows.append(((ramp - m255) / s255).astype(np.float32))
    return np.stack(rows)


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """qwen-vl-utils 0.0.14 ``smart_resize``.

    Rounds each side to a multiple of ``factor``, then rescales both sides (keeping the aspect ratio
    approximately) when the pixel count is outside ``[min_pixels, max_pixels]``.

    Args:
        height: Input height in pixels.
        width: Input width in pixels.
        factor: Grid factor; both output sides are multiples of it.
        min_pixels: Lower bound on ``h · w``.
        max_pixels: Upper bound on ``h · w``.

    Returns:
        ``(height, width)`` of the resized image.

    Raises:
        ValueError: The aspect ratio exceeds 200.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio of {height}x{width} exceeds 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def decode_view(value: Any) -> np.ndarray:
    """Return an HWC uint8 RGB image from an ndarray or a base64/raw JPEG.

    Args:
        value: Encoded image bytes (raw or base64 ``str``; decoded with OpenCV to RGB), or an array-like
            ``[H, W, 3]`` (or ``[1, H, W, 3]``) uint8 RGB image.

    Returns:
        ``[H, W, 3]`` uint8 RGB.

    Raises:
        ValueError: The bytes cannot be decoded or the array is not HWC uint8 with 3 channels.
    """
    return decode_image(wire_view(value))


def decode_image(image: bytes | np.ndarray) -> np.ndarray:
    """``decode_view`` of a ``wire_view`` result: encoded bytes decoded with OpenCV to RGB, an array as it is."""
    import cv2

    if isinstance(image, np.ndarray):
        return image
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise ValueError("image bytes could not be decoded")
    return image


def wire_view(value: Any) -> bytes | np.ndarray:
    """A wire view as it arrived: the encoded bytes (raw, or from a base64 ``str``), or an HWC uint8 RGB array.

    Raises:
        ValueError: The array is not HWC (or ``[1, H, W, 3]``) uint8 with 3 channels.
    """
    if isinstance(value, str):
        return base64.b64decode(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"image must be HWC uint8 RGB, got {image.dtype} {image.shape}")
    return image


def mrope_position_ids(input_ids: np.ndarray, grids: list[tuple[int, int, int]], image_token_id: int,
                       merge_size: int) -> np.ndarray:
    """3-axis M-RoPE positions for an image-only prompt, ``int64 [3, 1, L]``.

    Text tokens advance all axes together. An image block at position ``position``
    gets ``(position, position + row, position + col)`` per merged token, then ``position`` advances
    by ``max(rows, cols)`` (Qwen2.5-VL ``get_rope_index`` with t = 1).

    Args:
        input_ids: Token ids ``[1, L]`` (or ``[L]``) int64, including the action tokens.
        grids: Patch grid ``(t, gh, gw)`` of each image, in prompt order.
        image_token_id: Token id marking an image patch in the prompt.
        merge_size: Patches merged per side into one image token.

    Returns:
        ``[3, 1, L]`` int64 temporal/height/width positions.

    Raises:
        ValueError: An image token run does not match its grid, or the number of images differs from
            ``len(grids)``.
    """
    token_ids = input_ids.reshape(-1)
    n = token_ids.shape[0]
    out = np.empty((3, n), dtype=np.int64)
    # HOST-GAP K4: the per-token python walk is a per-grid pass plus vectorized positions
    # (~0.15 ms per request saved). Identical integer arithmetic: a text token advances every
    # axis by one, an image block carries (position, position + row, position + col) and
    # advances by max(rows, cols); each grid consumes one uninterrupted run of exactly its
    # token count, so adjacent blocks (no text between) split exactly as the walk did.
    is_image = token_ids == image_token_id
    advance = np.where(is_image, 0, 1).astype(np.int64)
    starts, r_all, c_all = [], [], []
    i = 0
    for t, gh, gw in grids:
        rows, cols = gh // merge_size, gw // merge_size
        count = rows * cols
        start = i + int(np.argmax(is_image[i:])) if is_image[i:].any() else n
        end = start + count
        if end > n or not is_image[start:end].all():
            raise ValueError("image token run does not match its grid")
        advance[end - 1] += max(rows, cols)
        starts.append(start)
        r_all.append(np.repeat(np.arange(rows), cols))
        c_all.append(np.tile(np.arange(cols), rows))
        i = end
    if is_image[i:].any():
        raise ValueError(f"prompt has {len(grids) + 1} images, request has {len(grids)}")
    position = np.concatenate(((0,), np.cumsum(advance)[:-1])) if n else np.zeros(0, dtype=np.int64)
    out[0] = position
    out[1] = position
    out[2] = position
    for start, r, c in zip(starts, r_all, c_all):
        block = position[start]
        out[0, start : start + r.size] = block
        out[1, start : start + r.size] = block + r
        out[2, start : start + r.size] = block + c
    return out[:, None, :]
