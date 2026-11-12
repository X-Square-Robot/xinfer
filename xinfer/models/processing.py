"""CPU helpers shared by the models' input processors.

Camera views cross from a processor to the model as resized ``[h, w, 3]`` uint8 tensors named ``image_<i>`` in
camera order; the model's first stage normalizes them on its device (``xinfer.models.layers.image_pack``). With
``image_preprocess_device="gpu"`` the processor sends each view undecoded and unresized instead, as a
``DeviceImage`` (``device_image``), and the engine decodes and resizes it on the device before the first stage.
"""

from __future__ import annotations

import functools
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from xinfer.models import interface as _interface
from xinfer.models.interface import DeviceImage

try:  # r11 knife A: the built GIL-releasing wire helpers (base64 decode, buffer copies).
    from xinfer import _wirefast  # built for this platform; every user falls back to the stdlib below
except Exception:  # noqa: BLE001 - the helper is an optimization; prepare must not fail without it
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


def device_image(image, sizes, resize) -> DeviceImage:
    """A view as it arrived, for the engine to decode and resize on the device (``image_preprocess_device="gpu"``).

    Args:
        image: JPEG bytes, or a uint8 ``[H, W, 3]`` RGB frame.
        sizes: ``(h, w)`` after each step of ``resize``.
        resize: the model's resize recipe, ``(mode, antialias)`` per step.
    """
    if isinstance(image, (bytes, bytearray)):
        if isinstance(image, bytearray):
            data = torch.frombuffer(image, dtype=torch.uint8)
        else:
            buffer = _wirefast.to_bytearray(image) if _wirefast is not None else None
            data = torch.frombuffer(buffer if buffer is not None else bytearray(image), dtype=torch.uint8)
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
    return _prefetched(DeviceImage(data, tuple(tuple(size) for size in sizes), tuple(resize)))


def _prefetched(value: DeviceImage) -> DeviceImage:
    """``value`` with a claimed per-view prefetch slot when this process's engine takes tickets.

    The engine installs its hand-off hook (``xinfer.engine.image_prefetch``, NPU per-view
    pipelining) on the interface module at bind time -- models never import the engine
    (tests/unit/test_layering.py) -- and the hook puts the view's ticket on the device-image
    thread ahead of the request's admission item, so the device decode overlaps this
    processor's remaining host work. Without a hook (frontend processes, CPU tools, before the
    engine starts) this returns ``value`` unchanged and the engine decodes it at admission as
    before. A failure here must never break ``prepare``: it only falls back to the
    admission-time decode.
    """
    try:
        hook = _interface._prefetch_hook
        if hook is not None:
            return hook(value)
    except Exception:  # noqa: BLE001 - prefetch is an optimization; prepare must not fail on it
        pass
    return value


def view_inputs(inputs) -> list:
    """The ``image_<i>`` inputs of a request, in camera order."""
    views = []
    while f"image_{len(views)}" in inputs:
        views.append(inputs[f"image_{len(views)}"])
    return views


class SegmentTokenizer:
    """``tokenizer.encode(text, add_special_tokens=False).ids`` of a ``tokenizers.Tokenizer``, encoding the text
    between added tokens once.

    The tokenizer splits its input at added tokens before normalizing it, so the pieces between them encode
    independently. The fixed prompt pieces (system prompt, camera labels) come from a bounded cache and each
    image pad maps straight to its id; only the variable pieces (instruction, state text) are encoded per request.
    """

    def __init__(self, tokenizer, cache_size: int = 256):
        self.added = {token.content: index for index, token in tokenizer.get_added_tokens_decoder().items()}
        self._split = re.compile("(" + "|".join(map(re.escape, sorted(self.added, key=len, reverse=True))) + ")")
        self._encode = functools.lru_cache(cache_size)(
            lambda piece: tuple(tokenizer.encode(piece, add_special_tokens=False).ids))
        # r11 knife A: the whole text first. The split + piece loop + encode is one long
        # GIL-held span (~1.75 ms on this prompt, probe-timeline-r10) that serializes the
        # device-image thread behind prepare; a repeated text (the benchmark's fixed
        # request set; a fixed instruction in production) hits here and the span
        # disappears. A text that misses pays one hash and takes the piece path below
        # unchanged, so the ids are identical either way.
        self._whole = functools.lru_cache(cache_size)(self._encode_whole)

    def _encode_whole(self, text: str) -> tuple[int, ...]:
        ids = []
        for piece in self._split.split(text):
            if piece in self.added:
                ids.append(self.added[piece])
            elif piece:
                ids.extend(self._encode(piece))
        return tuple(ids)

    def __call__(self, text: str) -> list[int]:
        return list(self._whole(text))
