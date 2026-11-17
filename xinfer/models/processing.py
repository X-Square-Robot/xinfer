"""CPU helpers shared by the models' input processors.

Camera views cross from a processor to the model as resized ``[h, w, 3]`` uint8 tensors named ``image_<i>`` in
camera order; the model's first stage normalizes them on its device (``xinfer.models.layers.image_pack``). With
``image_preprocess_device="gpu"`` the processor sends each view undecoded and unresized instead, as a
``DeviceImage`` (``device_image``), and the engine decodes and resizes it on the device before the first stage.
"""

from __future__ import annotations

import functools
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from xinfer.models.interface import DeviceImage

try:
    from xinfer import _wirefast
except ImportError:  # optional host extension
    _wirefast = None

# Where camera views are decoded and resized: in the processor ("cpu") or by the engine on the device ("gpu").
IMAGE_PREPROCESS_DEVICES = ("cpu", "gpu")


def text_length_range(length: int, value_tokens: int, values: int, *, max_tokens: int = 3,
                      max_length: int | None = None) -> tuple[int, int]:
    """Reachable prompt lengths when only the token cost of its state values changes.

    ``value_tokens`` counts the variable tokens in the current prompt (digits for
    the bundled models), with one to ``max_tokens`` per value. Separators stay in the fixed part. A truncated prompt
    may give an overly broad lower bound, which safely warms extra shapes.
    """
    fixed = length - value_tokens
    shortest, longest = max(1, fixed + values), fixed + values * max_tokens
    return shortest, min(longest, max_length) if max_length is not None else longest


def check_image_preprocess_device(value: str) -> None:
    """Raise ``ValueError`` unless ``value`` is one of ``IMAGE_PREPROCESS_DEVICES``."""
    if value not in IMAGE_PREPROCESS_DEVICES:
        raise ValueError(f"image_preprocess_device must be one of {IMAGE_PREPROCESS_DEVICES}, got {value!r}")


@functools.cache
def _view_pool() -> ThreadPoolExecutor:
    # Python's default limit. A thread starts only when every started one is busy, so the pool grows to the views
    # in flight (a frontend's processing threads times the cameras), not to the limit.
    return ThreadPoolExecutor(thread_name_prefix="xinfer-views")


def map_views(fn, items) -> list:
    """``[fn(item) for item in items]``, computed concurrently, in order; an exception reaches the caller.

    Per-view work (decoding, resizing) runs in OpenCV and Pillow, which release the GIL, so a request's views
    overlap. One thread pool per process serves every processor and every request processed at the same time.
    """
    return list(_view_pool().map(fn, items))


def is_jpeg(data) -> bool:
    """Whether encoded image bytes start like a JPEG (an SOI marker)."""
    return data[:2] == b"\xff\xd8"


def jpeg_size(data) -> tuple[int, int]:
    """``(height, width)`` of a JPEG from its frame header, without decoding it.

    Raises:
        ValueError: the data is not a JPEG, or it ends before a frame header.
    """
    if not is_jpeg(data):
        raise ValueError("image bytes are not a JPEG")
    i, n = 2, len(data)
    while i + 9 <= n:
        if data[i] != 0xFF:
            raise ValueError("malformed JPEG header")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
        elif marker == 0x01 or 0xD0 <= marker <= 0xD8:  # markers without a length
            i += 2
        elif 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # start of frame
            return int.from_bytes(data[i + 5:i + 7], "big"), int.from_bytes(data[i + 7:i + 9], "big")
        else:
            i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    raise ValueError("JPEG has no frame header")


def source_size(image) -> tuple[int, int]:
    """``(height, width)`` of a view as it arrived: JPEG bytes (from the header) or an ``[H, W, 3]`` frame."""
    return jpeg_size(image) if isinstance(image, (bytes, bytearray)) else tuple(image.shape[:2])


def device_image(image, sizes, resize, *, prefetch=None) -> DeviceImage:
    """A view as it arrived, for the engine to decode and resize on the device (``image_preprocess_device="gpu"``).

    Args:
        image: JPEG bytes, or a uint8 ``[H, W, 3]`` RGB frame.
        sizes: ``(h, w)`` after each step of ``resize``.
        resize: the model's resize recipe, ``(mode, antialias)`` per step.
        prefetch: optional image queue bound to this processor's engine.
    """
    if isinstance(image, (bytes, bytearray)):
        # HOST-GAP K1: the JPEG bytes are only ever read from here (the engine uploads them
        # to the device decoder), so the immutable ``bytes`` of the wire decode goes straight
        # into a read-only tensor. The previous writeable copy (``to_bytearray``, one GIL-held
        # pass over every JPEG on the prepare thread, ~0.2 ms per view) is gone; torch emits
        # its non-writable-buffer UserWarning once per process and nothing writes this tensor.
        data = torch.frombuffer(image, dtype=torch.uint8)
    elif (_wirefast is not None and isinstance(image, np.ndarray) and image.dtype == np.uint8
          and image.flags.c_contiguous and not image.flags.writeable):
        # a read-only wire frame: np.require's multi-MB copy holds the GIL (probe-timeline-r10);
        # one GIL-released copy to a writeable view instead, then require passes it through.
        buffer = _wirefast.to_bytearray(image)
        if buffer is not None:
            image = np.frombuffer(buffer, dtype=np.uint8).reshape(image.shape)
        data = torch.from_numpy(np.require(image, requirements=("C", "W")))
    else:
        data = torch.from_numpy(np.require(image, requirements=("C", "W")))  # copies a read-only wire frame
    value = DeviceImage(data, tuple(tuple(size) for size in sizes), tuple(resize))
    return value if prefetch is None else prefetch.hand_off(value)


def view_inputs(inputs) -> list:
    """The ``image_<i>`` inputs of a request, in camera order."""
    views = []
    while f"image_{len(views)}" in inputs:
        views.append(inputs[f"image_{len(views)}"])
    return views


def device_images_ready(prefetch=None) -> None:
    """Close the current request's hand-over of views to the engine's device-image thread.

    Call it once, after a request's ``DeviceImage`` values were built with ``device_image``: the
    engine decodes that request's views as one wave, and this lets it start the wave before the
    request is admitted. A no-op without this processor's bound engine handle.
    """
    if prefetch is not None:
        prefetch.flush()


_QWEN_NUMBER_SPLITS = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)


def _number_cache_supported(tokenizer) -> bool:
    """Admit only the Qwen components whose ASCII numbers have independent pre-token boundaries."""
    if type(getattr(tokenizer, "model", None)).__name__ != "BPE" or tokenizer.pre_tokenizer is None:
        return False
    def state(component):
        return None if component is None else json.loads(component.__getstate__())

    if state(tokenizer.normalizer) not in (None, {"type": "NFC"}):
        return False
    pre = state(tokenizer.pre_tokenizer)
    post = state(tokenizer.post_processor)
    byte_levels = [{"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": trim, "use_regex": False}
                   for trim in (False, True)]
    return (post is None or post in byte_levels) and any(
        pre == {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False}, byte]}
        for pattern in _QWEN_NUMBER_SPLITS for byte in byte_levels)


class SegmentTokenizer:
    """``tokenizer.encode(text, add_special_tokens=False).ids`` of a ``tokenizers.Tokenizer``, encoding the text
    between added tokens once.

    The tokenizer splits its input at added tokens before normalizing it, so the pieces between them encode
    independently. The fixed prompt pieces (system prompt, camera labels) come from a bounded cache and each
    image pad maps straight to its id; only the variable pieces (instruction, state text) are encoded per request.
    """

    def __init__(self, tokenizer, cache_size: int = 256):
        self.added = {token.content: index for index, token in tokenizer.get_added_tokens_decoder().items()}
        self._split = (re.compile("(" + "|".join(map(re.escape, sorted(self.added, key=len, reverse=True))) + ")")
                       if self.added else None)
        self._encode = functools.lru_cache(cache_size)(
            lambda piece: tuple(tokenizer.encode(piece, add_special_tokens=False).ids))
        # Repeated prompts reuse their complete token sequence.
        self._whole = functools.lru_cache(cache_size)(self._encode_whole)
        self._hl1 = (os.environ.get("XINFER_HL1_NUMCACHE", "on").strip().lower() != "off"
                     and _number_cache_supported(tokenizer))
        self._num_run = re.compile(r"[0-9]+(?: [0-9]+)*")
        self._num_token = functools.lru_cache(max(8 * cache_size, 4096))(
            lambda token: tuple(tokenizer.encode(token, add_special_tokens=False).ids))

    def _num_ids(self, run: str) -> tuple[int, ...]:
        """The ids of a single-space digit run, assembled from per-number tuples."""
        out: list[int] = []
        first = True
        for number in run.split(" "):
            out.extend(self._num_token(number if first else " " + number))
            first = False
        return tuple(out)

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        """Cache ASCII integer runs only at the admitted tokenizer's independent boundaries.

        NFC cannot combine across the ASCII cuts. Qwen isolates each digit and a single preceding
        space; multiple whitespace characters can split differently before a number and at piece end.
        """
        if self._hl1:
            for m in self._num_run.finditer(piece):
                run = m.group(0)
                if len(run.split(" ")) < 4:
                    continue
                start, end = m.span()
                if start and (piece[start - 1] != " " or (start > 1 and piece[start - 2].isspace())):
                    continue
                if end != len(piece) and piece[end] != "\n":
                    continue
                head, tail = piece[:start], piece[end:]
                out: list[int] = []
                if head:
                    out.extend(self._encode(head))
                out.extend(self._num_ids(run))
                if tail:
                    out.extend(self._encode(tail))
                return tuple(out)
        return self._encode(piece)

    def _encode_whole(self, text: str) -> tuple[int, ...]:
        ids = []
        pieces = (text,) if self._split is None else self._split.split(text)
        for piece in pieces:
            if piece in self.added:
                ids.append(self.added[piece])
            elif piece:
                ids.extend(self._encode_piece(piece))
        return tuple(ids)

    def __call__(self, text: str) -> list[int]:
        return list(self._whole(text))
