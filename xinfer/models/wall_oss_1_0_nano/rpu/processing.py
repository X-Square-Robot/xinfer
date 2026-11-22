"""Exact wall_oss_1_0_nano CPU processing; no WALL-OSS-0.5 or RhinoForge runtime dependency.

The banded image pipeline is adapted from RhinoForge's Wall Qwen3.5 adapter;
public source links are listed in docs/models/wall_oss_1_0_nano_rpu.md.

Two resizes are intentional: Dataset-V2 PIL BICUBIC, then the checkpoint's
Torchvision uint8 bicubic/antialias resize. FP32 noise is retained before the
once-only FP16 packing, with both byte hashes available for paired comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
import hashlib
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from xinfer.models.interface import PreparedRequest
from .geometry import relative26_to_absolute14

DEFAULT_CAMERAS = ("face_view", "left_wrist_view", "right_wrist_view")
CAMERA_LABELS = ("front view", "left wrist view", "right wrist view")
IMAGE_GRIDS = ((1, 14, 22), (1, 14, 20), (1, 14, 20))
PROFILE_MASK = (1.0,) * 20 + (0.0,) * 6
FLOW_TOKENS = ("<|propri|>", "<|action|>", "<point>", "</point>", "<box>", "</box>")
BASE_VOCAB_SIZE, CHECKPOINT_VOCAB_SIZE = 248077, 256277
_AUDIO_TAIL = (
    "<|audio_start|>",
    "<|audio_end|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<|audio_pad|>",
)
_UINT8_TO_FP16 = ((np.arange(256, dtype=np.float32) - np.float32(127.5)) / np.float32(127.5)).astype(np.float16)
# Every packed pixel on every path is an entry of this constant table
# (fancy-index gather or np.take with clip), so validating it once at
# import makes the per-request exponent scan vacuous.
if not bool(np.isfinite(_UINT8_TO_FP16).all()):
    raise RuntimeError("non-finite entries in the uint8->fp16 pixel table")
_UINT8_TO_FP16.setflags(write=False)


def array_manifest(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    return dict(dtype=value.dtype.str, shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())


def extend_flow_tokenizer(tokenizer):
    """Install and validate the exact Qwen3.5 flow-token ABI once at startup."""
    current = len(tokenizer)
    if current not in (BASE_VOCAB_SIZE, CHECKPOINT_VOCAB_SIZE):
        raise ValueError("WALL-OSS-1.0-Nano tokenizer must be pristine or exactly flow-extended")
    for offset, token in enumerate(_AUDIO_TAIL):
        if tokenizer.convert_ids_to_tokens(BASE_VOCAB_SIZE - len(_AUDIO_TAIL) + offset) != token:
            raise ValueError("WALL-OSS-1.0-Nano tokenizer does not have the Qwen3.5 audio-tail ABI")
    reserve_count = CHECKPOINT_VOCAB_SIZE - BASE_VOCAB_SIZE - len(FLOW_TOKENS)
    if current == BASE_VOCAB_SIZE:
        if tokenizer.add_special_tokens(
            {"additional_special_tokens": list(FLOW_TOKENS)}, replace_extra_special_tokens=False
        ) != len(FLOW_TOKENS):
            raise ValueError("WALL-OSS-1.0-Nano flow-token extension failed")
        if (
            tokenizer.add_tokens([f"<|flow_only_reserved_{i}|>" for i in range(reserve_count)], special_tokens=False)
            != reserve_count
        ):
            raise ValueError("WALL-OSS-1.0-Nano reserved-token extension failed")
    if len(tokenizer) != CHECKPOINT_VOCAB_SIZE:
        raise ValueError("WALL-OSS-1.0-Nano vocabulary must contain 256277 tokens")
    special = set(tokenizer.all_special_tokens)
    for offset, token in enumerate(FLOW_TOKENS):
        if tokenizer.convert_tokens_to_ids(token) != BASE_VOCAB_SIZE + offset or token not in special:
            raise ValueError(f"invalid wall_oss_1_0_nano flow-token ABI for {token}")
    for i in range(reserve_count):
        token = f"<|flow_only_reserved_{i}|>"
        if tokenizer.convert_tokens_to_ids(token) != BASE_VOCAB_SIZE + len(FLOW_TOKENS) + i or token in special:
            raise ValueError(f"invalid wall_oss_1_0_nano reserved-token ABI for {token}")
    return tokenizer


def dataset_v2_size(size: tuple[int, int]) -> tuple[int, int]:
    """Return (height,width), retaining Dataset-V2's rounding in both bounds."""
    width, height = size
    if width <= 0 or height <= 0 or max(width, height) / min(width, height) > 200:
        raise ValueError("image must have positive dimensions and aspect ratio <=200")
    if width > height:
        width, height = 224, max(1, int(224 * height / width))
    else:
        width, height = max(1, int(224 * width / height)), 224
    h, w = max(32, round(height / 32) * 32), max(32, round(width / 32) * 32)
    for bound, applies in ((4096 * 4096, h * w > 4096 * 4096), (4096, h * w < 4096)):
        if applies:
            scale = math.sqrt(bound / (height * width))
            h, w = max(32, round(height * scale / 32) * 32), max(32, round(width * scale / 32) * 32)
    return h, w


def processor_size(height: int, width: int, *, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    h, w = round(height / 32) * 32, round(width / 32) * 32
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = max(32, math.floor(height / beta / 32) * 32), max(32, math.floor(width / beta / 32) * 32)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / 32) * 32, math.ceil(width * beta / 32) * 32
    return h, w


def pack_image_array(image: np.ndarray) -> np.ndarray:
    """FP16 normalize RGB uint8 and pack [merge_h,merge_w,2,2,C,T,P,P]."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("packed image must be uint8 RGB HWC")
    height, width = image.shape[:2]
    if height % 32 or width % 32:
        raise ValueError("packed image dimensions must be multiples of 32")
    h, w = height // 16, width // 16
    spatial = _UINT8_TO_FP16[image].reshape(h // 2, 2, 16, w // 2, 2, 16, 3).transpose(0, 3, 1, 4, 6, 2, 5)
    pixels = np.empty((h * w, 1536), dtype=np.float16)
    pixels.reshape(h // 2, w // 2, 2, 2, 3, 2, 16, 16)[...] = spatial[..., None, :, :]
    return pixels


def normalize_state(value: Any, minimum: Any, delta: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    if state.shape not in ((26,), (1, 26), (1, 1, 26)) or not np.isfinite(state).all():
        raise ValueError("state must be finite [26], [1,26], or [1,1,26]")
    low, span = np.asarray(minimum, dtype=np.float32), np.asarray(delta, dtype=np.float32)
    if (
        low.shape != (26,)
        or span.shape != (26,)
        or not np.isfinite(low).all()
        or not np.isfinite(span).all()
        or (span <= 0).any()
    ):
        raise ValueError("state normalizer requires finite [26] arrays and positive delta")
    normalized = state.reshape(1, 1, 26) - low.reshape(1, 1, 26)
    normalized /= span.reshape(1, 1, 26)
    normalized *= np.float32(2)
    normalized -= np.float32(1)
    np.clip(normalized, -1, 1, out=normalized)
    return normalized


def digitize_state(normalized: np.ndarray) -> str:
    boundaries = torch.linspace(-1.0, 1.0, 513, dtype=torch.float32).numpy()[:-1]
    digits = np.searchsorted(boundaries, normalized.reshape(26), side="right") - 1
    return " ".join(map(str, digits[:20].tolist()))


def build_prompt(instruction: str | Mapping[str, Any], *, state_string: str) -> tuple[str, str]:
    def sanitize(value):
        words = str(value or "").strip().split()
        if len(words) > 256:
            raise ValueError("runtime instruction exceeds the 256-word profile")
        return " ".join(words)

    if isinstance(instruction, Mapping):
        task, detail = sanitize(instruction.get("task", "")), sanitize(instruction.get("detail", ""))
        instruction_text = "\n".join(
            ([f"Task: {task}"] if task else []) + ([f"Instruction: {detail}"] if detail else [])
        )
    else:
        instruction_text = f"Task: {sanitize(instruction)}"
    cameras = "".join(f" {label}," for label in CAMERA_LABELS)
    prefix = (
        "<|im_start|>system\nYou are an embodied vision-language-action (VLA) model controlling "
        "the robot with language instructions.\n Embodiment: X2 EX001 Desktop"
        f"\n Camera Setup:{cameras}\n Frequency: 32HZ\n Action Space: Rel EEF\n<|im_end|>\n"
        "<|im_start|>user\nObservation:"
    )
    for label in CAMERA_LABELS:
        prefix += f" {label}: <|vision_start|><|image_pad|><|vision_end|>"
    prefix += (
        f"\n{instruction_text}\nPredict the next action in robot action.\n"
        f"Proprioception: {state_string}\n<|im_end|>\n<|im_start|>assistant\n"
    )
    return prefix, "<|action|>" * 16


def mrope_position_ids(
    input_ids: np.ndarray, image_token_id: int, grids: Sequence[Sequence[int]] = IMAGE_GRIDS
) -> np.ndarray:
    """Batch-one image-only Qwen3.5 positions, independent of physical buckets."""
    types = (np.asarray(input_ids).reshape(-1) == image_token_id).astype(np.int32)
    if not len(types):
        raise ValueError("empty multimodal token sequence")
    boundaries = [0, *(np.flatnonzero(types[1:] != types[:-1]) + 1).tolist(), len(types)]
    positions = np.empty((3, 1, len(types)), dtype=np.int64)
    current, image_index = 0, 0
    for start, end in zip(boundaries, boundaries[1:]):
        length = end - start
        if types[start] == 0:
            positions[:, 0, start:end] = np.arange(current, current + length)
            current += length
        elif image_index < len(grids):
            t, h, w = grids[image_index]
            if t != 1 or h % 2 or w % 2 or length != h * w // 4:
                raise ValueError("image token/grid length mismatch")
            h, w = h // 2, w // 2
            positions[0, 0, start:end] = current
            positions[1, 0, start:end] = np.repeat(np.arange(h), w) + current
            positions[2, 0, start:end] = np.tile(np.arange(w), h) + current
            current += max(h, w)
            image_index += 1
        else:
            raise ValueError("image token/grid count mismatch")
    if image_index != len(grids):
        raise ValueError("image token/grid count mismatch")
    return positions


def prepare_noise(initial_noise: Any | None = None, noise_seed: int | None = None):
    if initial_noise is not None and noise_seed is not None:
        raise ValueError("provide either initial_noise or noise_seed, not both")
    if noise_seed is not None and (isinstance(noise_seed, bool) or not isinstance(noise_seed, int)):
        raise ValueError("noise_seed must be an integer or None")
    if initial_noise is None:
        generator = torch.Generator(device="cpu").manual_seed(0 if noise_seed is None else noise_seed)
        noise = torch.randn((1, 16, 26), generator=generator, dtype=torch.float32)
        original = noise.numpy()
    else:
        if isinstance(initial_noise, torch.Tensor) and initial_noise.device.type != "cpu":
            raise ValueError("initial_noise must be on CPU")
        original = np.asarray(initial_noise)
        noise = torch.as_tensor(initial_noise, dtype=torch.float32)
        if noise.shape == (16, 26):
            noise = noise.unsqueeze(0)
        if noise.shape != (1, 16, 26):
            raise ValueError("initial_noise must have shape [16,26] or [1,16,26]")
        noise = noise.contiguous().clone()
    if not bool(torch.isfinite(noise).all()) or not bool(torch.isfinite(noise.half()).all()):
        raise ValueError("initial_noise must remain finite after FP16 packing")
    packed = noise.half()
    return (
        noise,
        packed,
        dict(
            source="explicit" if initial_noise is not None else "cpu_seed",
            seed=None if initial_noise is not None else (0 if noise_seed is None else noise_seed),
            supplied=array_manifest(original),
            fp32=array_manifest(noise),
            fp16=array_manifest(packed),
        ),
    )


def _load_normalizers(checkpoint: Path) -> dict[str, np.ndarray]:
    result = {}
    for kind, name, expected in (("propri", "state", (26,)), ("action", "action", (16, 26))):
        payload = torch.load(checkpoint / f"normalizer_{kind}.pth", map_location="cpu", weights_only=True)
        for stat in ("min", "delta"):
            value = payload[f"{stat}.x2_normal"].detach().float().contiguous().numpy()
            if value.shape != expected or not np.isfinite(value).all() or (stat == "delta" and (value <= 0).any()):
                raise ValueError(f"{kind} normalizer requires finite {expected} and positive delta")
            result[f"{name}_{stat}"] = value.copy()
    return result


def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, Image.Image):
        return value if value.mode == "RGB" else value.convert("RGB")
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("camera images must be PIL/path or uint8 RGB HWC arrays")
    return Image.fromarray(image)


def _image_size(value: Any) -> tuple[int, int]:
    """Header-only size probe: never decodes pixels.

    Mirrors _decode_image's accepted inputs so the production path can size
    the prompt and submit pool decode jobs without paying serial decode on
    the request thread.
    """
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.size
    if isinstance(value, Image.Image):
        return value.size
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("camera images must be PIL/path or uint8 RGB HWC arrays")
    return (int(image.shape[1]), int(image.shape[0]))


class WallOSS10NanoRPUInputProcessor:
    """Prepare the exact B1/H16/non-RTC/NFE8 profile on CPU.

    ``processor`` and ``normalizers`` are explicit injection points for CPU
    parity tests. Production uses the admitted checkpoint's HF processor and
    normalizers. This processor owns a bounded six-worker image pool.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        model_config=None,
        *,
        processor=None,
        normalizers: Mapping[str, Any] | None = None,
        image_workers: int = 6,
        audit_inputs: bool = False,
    ):
        checkpoint = Path(checkpoint)
        if image_workers < 0 or image_workers > 6:
            raise ValueError("image_workers must be in [0,6]")
        self.model_config = model_config
        self.audit_inputs = bool(audit_inputs)
        if processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        self.processor = processor
        self.tokenizer = extend_flow_tokenizer(processor.tokenizer)
        self.normalizers = {
            k: np.asarray(v, dtype=np.float32).copy()
            for k, v in (normalizers if normalizers is not None else _load_normalizers(checkpoint)).items()
        }
        for name, shape in (("state", (26,)), ("action", (16, 26))):
            low, delta = self.normalizers[f"{name}_min"], self.normalizers[f"{name}_delta"]
            if (
                low.shape != shape
                or delta.shape != shape
                or not np.isfinite(low).all()
                or not np.isfinite(delta).all()
                or (delta <= 0).any()
            ):
                raise ValueError(f"{name} normalizer requires finite {shape} and positive delta")
        image = processor.image_processor
        attrs = (
            getattr(image, "patch_size", None),
            getattr(image, "merge_size", None),
            getattr(image, "temporal_patch_size", None),
        )
        if attrs != (16, 2, 2) or getattr(image, "do_resize", None) is not True:
            raise ValueError("WALL-OSS-1.0-Nano requires checkpoint resize and patch/merge/temporal sizes 16/2/2")
        if (
            not image.do_rescale
            or not image.do_normalize
            or image.rescale_factor != 1 / 255
            or tuple(image.image_mean) != (0.5,) * 3
            or tuple(image.image_std) != (0.5,) * 3
            or int(image.resample) != int(Image.Resampling.BICUBIC)
        ):
            raise ValueError("WALL-OSS-1.0-Nano requires the checkpoint bicubic/normalization image profile")
        size = image.size
        get = size.get if isinstance(size, Mapping) else lambda k: getattr(size, k, None)
        self._bounds = (int(get("shortest_edge") or 0), int(get("longest_edge") or 0))
        if not 0 < self._bounds[0] <= self._bounds[1]:
            raise ValueError("invalid checkpoint image processor resize bounds")
        self._pool = (
            ThreadPoolExecutor(max_workers=image_workers, thread_name_prefix="wall_oss_1_0_nano-image")
            if image_workers else None
        )
        self._closed = False
        # Per-thread pack staging buffers for the image pool (see
        # _pack_image_rows); sized on first use, reused across requests.
        self._pack_scratch = threading.local()
        # Request-invariant input tensors, built once outside request
        # timing. Values are constants of the admitted profile (validated
        # per request where they depend on the observation); sharing the
        # tensor objects is safe because every consumer only reads them.
        mask26 = torch.tensor(PROFILE_MASK, dtype=torch.float32).reshape(1, 1, 26)
        times = torch.linspace(0, 1, 9, dtype=torch.float32) * 0.999
        dt = (times[1:] - times[:-1]).half()
        if not bool(dt.eq(dt[0]).all()):
            raise RuntimeError("nonuniform FP16 Euler schedule")
        self._constants = {
            "agent_pos_mask": mask26,
            "dof_mask": mask26.clone(),
            "image_grid_thw": torch.tensor(IMAGE_GRIDS, dtype=torch.int64),
            "time_values": times,
            "dt": dt,
        }
        self._action_id = self.tokenizer.convert_tokens_to_ids("<|action|>")
        self._image_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # Start each worker in preparation, outside request timing.
        if self._pool:
            from threading import Barrier

            barrier = Barrier(image_workers)
            warmup = [self._pool.submit(barrier.wait) for _ in range(image_workers)]
            for future in warmup:
                future.result()

    def _camera(self, image: Image.Image, stage1_size: tuple[int, int], stage2_size: tuple[int, int]):
        import torchvision.transforms.v2.functional as tvf
        from torchvision.transforms import InterpolationMode

        h, w = stage1_size
        stage1 = image.resize((w, h), Image.Resampling.BICUBIC)
        stage2 = tvf.resize(
            tvf.pil_to_tensor(stage1), list(stage2_size), interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        stage2 = stage2.permute(1, 2, 0).contiguous().numpy()
        audit = (
            dict(stage1=array_manifest(np.asarray(stage1)), stage2=array_manifest(stage2))
            if self.audit_inputs
            else None
        )
        return pack_image_array(stage2), audit

    # --- Banded pool pipeline (ported from the original rpu_backend adapter) ---
    # The production pool path reproduces the serial two-stage resize
    # bit-for-bit while removing the per-camera serial bottleneck: quarter-row
    # PIL band resizes keep every band row's sample center and filter window
    # identical to the full Dataset-V2 resize, the bands reassemble into the
    # same stage-1 image, and the processor resize plus patch packing run as
    # chained pool jobs so no pool thread ever blocks on another pool job.

    def _stage1_bands(self, image: Image.Image, height: int):
        """Output-row bands that reproduce one full Dataset-V2 PIL resize.

        PIL samples the whole source image around each box. With quarter
        boundaries on integral source rows and a scale with at most four
        binary fraction digits, each band row has the same sample center and
        filter window as the full resize. Other geometries use one full
        resize.
        """
        if (image.mode == "RGB" and height % 4 == 0 and image.height % 4 == 0
                and (16 * image.height) % height == 0):
            return [(index * height // 4, (index + 1) * height // 4) for index in range(4)]
        return [(0, height)]

    def _resize_band(self, image: Image.Image, width: int, height: int, start: int, end: int):
        if (start, end) == (0, height):
            return image.resize((width, height), Image.Resampling.BICUBIC)
        return image.resize(
            (width, end - start), Image.Resampling.BICUBIC,
            (0, start * image.height // height, image.width,
             end * image.height // height),
        )

    def _processor_resize_bands(self, parts, stage1_size, stage2_size):
        """Compose the Dataset-V2 bands and run the processor resize."""
        if len(parts) == 1:
            image = parts[0]
        else:
            image = Image.new("RGB", (stage1_size[1], stage1_size[0]))
            top = 0
            for part in parts:
                image.paste(part, (0, top))
                top += part.height
        import torchvision.transforms.v2.functional as tvf
        from torchvision.transforms import InterpolationMode

        resized = tvf.resize(
            tvf.pil_to_tensor(image), list(stage2_size),
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        )
        return resized.permute(1, 2, 0).contiguous().numpy()

    def _pack_split(self, stage2_size):
        """Two-way split of one camera's packed patch rows, or None.

        Packed rows are laid out per merge block (2x2 patches) with row index
        ``4 * (merge_row * merge_columns + merge_col) + 2 * a + b`` for patch
        row ``2*merge_row+a`` and patch column ``2*merge_col+b``, so any whole
        16-pixel output-row band owns a fixed set of packed rows and the two
        halves can be packed by independent pool jobs. Returns the pixel-row
        split point, or None when the height does not admit one.
        """
        height = stage2_size[0]
        if height < 64 or height % 2:
            return None
        half = height // 2
        return half if half % 16 == 0 else None

    def _pack_image_rows(self, array: np.ndarray, output: np.ndarray, patch_row0: int):
        """Pack consecutive 16-pixel patch rows into a camera patch buffer.

        Same per-patch content as :func:`pack_image_array`, restricted to
        whole patch rows starting at camera patch row ``patch_row0``; the
        rows scatter to their merge-block positions. Pure data movement.
        """
        height, width = array.shape[:2]
        patch_rows = height // 16
        patch_columns = width // 16
        channels = array.shape[2]
        # np.take is the same gather as _UINT8_TO_FP16[array] but about 2.4x
        # faster: fancy indexing on a 256-entry LUT buffers byte by byte.
        # Taking into a reusable per-thread scratch avoids a fresh
        # multi-hundred-KB mmap (plus first-touch page faults) per pack job.
        scratch = getattr(self._pack_scratch, "buf", None)
        need = height * width * channels
        if scratch is None or scratch.size < need:
            scratch = np.empty(need, dtype=np.float16)
            self._pack_scratch.buf = scratch
        values = np.take(_UINT8_TO_FP16, array, mode="clip",
                         out=scratch[:need].reshape(array.shape))
        patches = values.reshape(patch_rows, 16, patch_columns, 16, channels)
        # (patch_row, patch_col, channel, row, col); the packed row duplicates
        # each patch across the merge sub-axis exactly as the full pack does,
        # so the duplicate is broadcast into the destination instead of
        # materialized.
        patches = patches.transpose(0, 2, 4, 1, 3)
        patch = patch_row0 + np.arange(patch_rows)[:, None]
        column = np.arange(patch_columns)[None, :]
        targets = (
            4 * ((patch // 2) * (width // 32) + column // 2)
            + (patch % 2) * 2 + column % 2
        )
        # 2-D targets with the axis-inserted RHS let numpy stream the fancy
        # scatter straight from the transposed view — no merged
        # (patch_rows*patch_columns) materialization per job.
        output.reshape(-1, channels, 2, 16, 16)[targets] = patches[:, :, :, None]

    def _submit_camera(self, executor, view, camera_index, stage1_size, stage2_size, output, pixel_sink):
        """Chain one camera on the pool: decode -> banded stage-1 resize ->
        processor resize -> patch-row packs. Each link is scheduled by the
        completion callbacks of the jobs it depends on, so no pool thread
        blocks on another pool job. The decode runs on the pool so the
        request thread only pays a header probe (see _image_size). When a
        pixel_sink is provided, each camera's packed rows are handed to it
        (device upload) as soon as they are final, overlapping the
        remaining camera work. Returns a Future completing when this
        camera's packed rows are fully written (or the first failure).
        """
        height, width = stage1_size
        camera = Future()
        # This future owns the whole task chain, including decode and bands.
        # Cancelling only the wrapper cannot cancel or drain those jobs.
        camera.set_running_or_notify_cancel()
        lock = threading.Lock()

        def fail(error, pending=()):
            if pending:
                remaining = [len(pending)]

                def drained(_):
                    with lock:
                        remaining[0] -= 1
                        if remaining[0]:
                            return
                    fail(error)

                for job in pending:
                    job.add_done_callback(drained)
                return
            if not camera.done():
                camera.set_exception(error)

        def decoded(decode_job):
            if decode_job.cancelled():
                fail(RuntimeError("WALL-OSS-1.0-Nano image decode job was cancelled"))
                return
            error = decode_job.exception()
            if error is not None:
                fail(error)
                return
            bands = []
            try:
                image = decode_job.result()
                for start, end in self._stage1_bands(image, height):
                    bands.append(executor.submit(self._resize_band, image, width, height, start, end))
            except BaseException as error:
                fail(error, bands)
                return
            remaining = [len(bands)]

            def bands_done(_):
                with lock:
                    remaining[0] -= 1
                    if remaining[0]:
                        return
                try:
                    parts = [band.result() for band in bands]
                    job = executor.submit(
                        self._processor_resize_bands, parts, stage1_size, stage2_size)
                except BaseException as error:
                    fail(error)
                    return

                def resize_done(resize_job):
                    if resize_job.cancelled():
                        fail(RuntimeError("WALL-OSS-1.0-Nano image resize job was cancelled"))
                        return
                    error = resize_job.exception()
                    if error is not None:
                        fail(error)
                        return
                    packs = []
                    try:
                        array = resize_job.result()
                        half = self._pack_split(stage2_size)
                        if half is None:
                            packs.append(executor.submit(self._pack_image_rows, array, output, 0))
                        else:
                            packs.append(executor.submit(self._pack_image_rows, array[:half], output, 0))
                            packs.append(executor.submit(self._pack_image_rows, array[half:], output, half // 16))
                    except BaseException as caught:
                        fail(caught, packs)
                        return
                    pending = [len(packs)]

                    def pack_done(_):
                        with lock:
                            pending[0] -= 1
                            if pending[0]:
                                return
                        for pack in packs:
                            if pack.cancelled():
                                fail(RuntimeError("WALL-OSS-1.0-Nano image pack job was cancelled"))
                                return
                            error = pack.exception()
                            if error is not None:
                                fail(error)
                                return
                        if pixel_sink is not None:
                            try:
                                pixel_sink(camera_index, output)
                            except BaseException as error:
                                fail(error)
                                return
                        camera.set_result(None)

                    for pack in packs:
                        pack.add_done_callback(pack_done)

                job.add_done_callback(resize_done)

            for band in bands:
                band.add_done_callback(bands_done)

        decode = executor.submit(_decode_image, view)
        decode.add_done_callback(decoded)
        return camera

    def __call__(self, observation: Mapping[str, Any], *, text_hook=None, pixel_sink=None) -> PreparedRequest:
        if self._closed:
            raise RuntimeError("WALL-OSS-1.0-Nano input processor is closed")
        if (
            observation.get("use_rtc")
            or observation.get("rtc_prefix_actions") is not None
            or observation.get("prev_action_chunk") is not None
            or observation.get("inference_delay", 0) != 0
        ):
            raise ValueError("WALL-OSS-1.0-Nano first profile accepts non-RTC only")
        for key in ("num_flow_steps", "num_steps"):
            if key in observation and (isinstance(observation[key], bool) or observation[key] != 8):
                raise ValueError("WALL-OSS-1.0-Nano requires NFE=8")
        if (
            observation.get("dataset_key", "x2_normal") != "x2_normal"
            or str(observation.get("robot_id", "10070")) != "10070"
        ):
            raise ValueError("WALL-OSS-1.0-Nano requires x2_normal / robot 10070")
        for name in ("agent_pos_mask", "dof_mask"):
            if name in observation and not np.array_equal(np.asarray(observation[name]).reshape(-1), PROFILE_MASK):
                raise ValueError(f"{name} must match [1]*20+[0]*6")
        if ("state" in observation) == ("proprioception" in observation):
            raise ValueError("provide exactly one of state or proprioception")
        state = np.asarray(observation.get("state", observation.get("proprioception")), dtype=np.float32)
        normalized = normalize_state(state, self.normalizers["state_min"], self.normalizers["state_delta"])
        state = state.reshape(26).copy()
        views = observation.get("images", observation.get("views"))
        if not isinstance(views, Mapping) or any(name not in views for name in DEFAULT_CAMERAS):
            raise ValueError(f"images must provide the canonical cameras {DEFAULT_CAMERAS}")
        if self._pool and not self.audit_inputs:
            # Production path: header-only size probe on this thread; the
            # pool decodes while the prompt/text work below runs here.
            ordered = [views[name] for name in DEFAULT_CAMERAS]
            sizes = [_image_size(view) for view in ordered]
            images = None
        else:
            images = [_decode_image(views[name]) for name in DEFAULT_CAMERAS]
            sizes = [image.size for image in images]
            ordered = None
        sizes1 = [dataset_v2_size(size) for size in sizes]
        sizes2 = [processor_size(h, w, min_pixels=self._bounds[0], max_pixels=self._bounds[1]) for h, w in sizes1]
        grids = [(1, h // 16, w // 16) for h, w in sizes2]
        if tuple(grids) != IMAGE_GRIDS:
            raise ValueError("WALL-OSS-1.0-Nano FP16 TP8 camera grids must be 14x22,14x20,14x20 in canonical order")
        jobs = []
        cameras = []
        patches = None
        try:
            if self._pool and not self.audit_inputs:
                # Production path: banded two-stage pipeline, packed on the
                # pool while the prompt/text work below runs on this thread.
                patches = np.empty((868, 1536), dtype=np.float16)
                offset = 0
                for camera_index, (view, size1, size2) in enumerate(zip(ordered, sizes1, sizes2)):
                    count = size2[0] * size2[1] // 256
                    cameras.append(self._submit_camera(
                        self._pool, view, camera_index, size1, size2,
                        patches[offset:offset + count], pixel_sink))
                    offset += count
            elif self._pool:
                jobs = [self._pool.submit(self._camera, *args) for args in zip(images, sizes1, sizes2)]
            prefix, postfix = build_prompt(observation.get("instruction", ""), state_string=digitize_state(normalized))
            expanded = prefix
            for t, h, w in grids:
                expanded = expanded.replace("<|image_pad|>", "<|placeholder|>" * (t * h * w // 4), 1)
            expanded = expanded.replace("<|placeholder|>", "<|image_pad|>")
            full = self.tokenizer(
                [expanded + postfix], return_tensors="pt", padding=True, padding_side="right", truncation=False
            )
            ids, mask = full.input_ids, full.attention_mask
            # <|action|> is an added special token, so the fast tokenizer's
            # pre-tokenization splits at every action token: encoding
            # expanded+postfix equals encoding expanded followed by the
            # postfix tokens, and the prefix ids are exactly ids[:, :length].
            # The second encode pass is redundant; the suffix assertion below
            # re-validates the boundary on every request.
            action_id = self._action_id
            length = int(ids.shape[1]) - 16
            if length > 512 or length <= 0 or ids.shape != (1, length + 16) or not bool(mask.all()):
                raise ValueError("WALL-OSS-1.0-Nano requires an unpadded prefix <=512 plus exactly 16 action tokens")
            if not bool(ids[0, length:].eq(action_id).all()):
                raise ValueError("action suffix must contain exactly 16 atomic action tokens")
            image_id = self._image_id
            positions = mrope_position_ids(ids.numpy(), image_id, grids)
            action_positions = positions[..., length:].copy()
            action_positions += 10  # The non-RTC profile retains the checkpoint's action-position gap.
            noise, packed_noise, noise_info = prepare_noise(
                observation.get("initial_noise"), observation.get("noise_seed")
            )
            padding_action = (-self.normalizers["action_min"]) / self.normalizers["action_delta"]
            padding_action *= np.float32(2)
            padding_action -= np.float32(1)
            np.clip(padding_action, -1, 1, out=padding_action)
            padding_velocity = padding_action[None] - noise.numpy()
            # Text stage complete. On the deferred path, expose the prepared
            # text tensors now so the caller can run request-side text
            # binding (prefill ids/rope + action tables) while the camera
            # pool is still producing pixels; pixel_values is filled in
            # after the image join below.
            tensors = dict(
                input_ids=ids[:, :length].contiguous(),
                attention_mask=mask[:, :length].contiguous(),
                mm_token_type_ids=(ids[:, :length] == image_id).to(torch.int32),
                position_ids=torch.from_numpy(positions[..., :length].copy()),
                action_position_ids=torch.from_numpy(action_positions),
                pixel_values=None,
                image_grid_thw=self._constants["image_grid_thw"],
                normalized_state=torch.from_numpy(normalized),
                agent_pos_mask=self._constants["agent_pos_mask"],
                dof_mask=self._constants["dof_mask"],
                noise=noise,
                noise_fp16=packed_noise,
                padding_velocity=torch.from_numpy(padding_velocity),
                padding_velocity_fp16=torch.from_numpy(padding_velocity).half(),
                time_values=self._constants["time_values"],
                dt=self._constants["dt"],
                prefix_length=length,
                prefix_bucket=((length + 63) // 64) * 64,
                num_steps=8,
            )
            if text_hook is not None:
                # Run the caller's request-side text binding (prefix
                # input_ids/rope plus the action tables) while the camera
                # pool is still packing pixels. The partial shares this
                # tensors dict, so the pixel_values filled in below is what
                # the subsequent finish stage sees.
                text_hook(PreparedRequest(tensors=tensors, meta=dict(
                    state26=state,
                    noise=noise_info,
                    inputs={},
                    resized_images=None,
                    camera_order=list(DEFAULT_CAMERAS),
                    gt_prediction_frame_offset=1,
                )))
            if cameras:
                for camera in cameras:
                    camera.result()
                pixel_array = patches
                resized_images = [None] * len(DEFAULT_CAMERAS)
            elif jobs:
                parts = [job.result() for job in jobs]
                pixel_array = np.concatenate([part[0] for part in parts])
                resized_images = [part[1] for part in parts]
            else:
                parts = [self._camera(*args) for args in zip(images, sizes1, sizes2)]
                pixel_array = np.concatenate([part[0] for part in parts])
                resized_images = [part[1] for part in parts]
            if pixel_array.shape != (868, 1536):
                raise ValueError("invalid WALL-OSS-1.0-Nano FP16 TP8 patch tensor")
            if not cameras:
                # Banded output is a clipped gather from the _UINT8_TO_FP16
                # table (validated once at import), so the per-request
                # exponent scan is vacuous there; keep it on the other paths.
                if np.any((pixel_array.view(np.uint16) & 0x7c00) == 0x7c00):
                    raise ValueError("invalid WALL-OSS-1.0-Nano FP16 TP8 patch tensor")
            tensors["pixel_values"] = torch.from_numpy(pixel_array)
            manifests = (
                {key: array_manifest(value) for key, value in tensors.items() if isinstance(value, torch.Tensor)}
                if self.audit_inputs
                else {}
            )
            return PreparedRequest(
                tensors=tensors,
                meta=dict(
                    state26=state,
                    noise=noise_info,
                    inputs=manifests,
                    resized_images=resized_images,
                    camera_order=list(DEFAULT_CAMERAS),
                    gt_prediction_frame_offset=1,
                ),
            )
        finally:
            for job in jobs:
                job.cancel()
            if jobs:
                wait(jobs)  # callers may reuse image buffers once this call returns
            for camera in cameras:
                camera.cancel()
            if cameras:
                wait(cameras)

    def close(self):
        if not self._closed:
            self._closed = True
            if self._pool:
                self._pool.shutdown(wait=True, cancel_futures=True)


class WallOSS10NanoRPUOutputProcessor:
    """Independent FP32 physical relative26 plus the original absolute14 decoder."""

    def __init__(self, model_config=None, *, include_absolute14: bool = False):
        self.model_config = model_config
        self.include_absolute14 = bool(include_absolute14)

    def __call__(self, outputs: Mapping[str, Any], request: PreparedRequest) -> dict[str, Any]:
        raw = outputs["predict_action"]
        if isinstance(raw, torch.Tensor):
            if raw.device.type != "cpu" or raw.dtype != torch.float32:
                raise ValueError("predict_action must be CPU FP32 physical relative26")
            raw = raw.detach().contiguous().numpy()
        raw = np.asarray(raw)
        if raw.dtype != np.float32 or raw.shape != (1, 16, 26) or not np.isfinite(raw).all():
            raise ValueError("predict_action must be finite FP32 physical [1,16,26]")
        physical = raw.copy()
        result = dict(
            predict_action=physical,
            actions=physical.copy(),
            noise=request.meta.get("noise"),
            gt_prediction_frame_offset=1,
        )
        if self.include_absolute14:
            result["action_pred"] = relative26_to_absolute14(physical[0], request.meta["state26"])
        return result


def denormalize_actions(normalized: Any, minimum: Any, delta: Any) -> np.ndarray:
    """Keep the reference's separate FP32 output materialization operations."""
    action = np.asarray(normalized, dtype=np.float32)
    low, span = np.asarray(minimum, dtype=np.float32), np.asarray(delta, dtype=np.float32)
    if action.shape != (1, 16, 26) or low.shape != (16, 26) or span.shape != (16, 26):
        raise ValueError("actions [1,16,26], normalizers [16,26] required")
    if not np.isfinite(action).all() or not np.isfinite(low).all() or not np.isfinite(span).all() or (span <= 0).any():
        raise ValueError("finite actions/normalizers and positive delta required")
    result = action + np.float32(1)
    result *= np.float32(0.5)
    result *= span[None]
    result += low[None]
    return result.copy()
