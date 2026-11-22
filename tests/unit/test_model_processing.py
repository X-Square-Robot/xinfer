"""Shared input-processing helpers: concurrent per-view work and segment-cached tokenization."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from xinfer.engine.cudagraph import warmup_requests
from xinfer.models.interface import BucketSpec, PreparedRequest, VLAModel
from xinfer.models.processing import (
    text_length_range,
    SegmentTokenizer,
    check_image_preprocess_device,
    device_image,
    is_jpeg,
    jpeg_size,
    map_views,
    source_size,
    view_inputs,
)


def test_map_views_keeps_order_runs_concurrently_and_raises():
    barrier = threading.Barrier(3, timeout=10)  # passes only if three views run at the same time

    def view(i):
        barrier.wait()
        return i * i

    assert map_views(view, [3, 1, 2]) == [9, 1, 4]

    def fail(i):
        raise ValueError(f"view {i}")

    with pytest.raises(ValueError, match="view"):
        map_views(fail, [0, 1])


def test_concurrent_requests_get_threads_for_all_their_views():
    barrier = threading.Barrier(6, timeout=10)  # two requests of three views, all running at the same time

    def view(i):
        barrier.wait()
        return i

    with ThreadPoolExecutor(2) as requests:
        assert list(requests.map(lambda items: map_views(view, items), [[0, 1, 2], [3, 4, 5]])) == [[0, 1, 2],
                                                                                                    [3, 4, 5]]


def test_view_inputs_reads_consecutive_image_names():
    assert view_inputs({"image_0": "a", "image_1": "b", "image_3": "x", "other": 1}) == ["a", "b"]


def test_segment_tokenizer_matches_the_tokenizer():
    """Encoding the pieces between added tokens, with a bounded cache, gives the tokenizer's own ids."""
    tokenizers = pytest.importorskip("tokenizers")

    byte_level = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.pre_tokenizer = byte_level
    tokenizer.train_from_iterator(
        ["Observation: front view, left wrist view. Instruction: pass the package 12 345"] * 4,
        tokenizers.trainers.BpeTrainer(
            vocab_size=300, initial_alphabet=byte_level.alphabet(),
            special_tokens=["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]))
    tokenizer.add_tokens(["<|action|>"])
    segments = SegmentTokenizer(tokenizer, cache_size=2)
    start, end, pad = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
    for instruction in ("pass the package", "放下包裹 7", "12 345 6789"):
        text = (f"<|im_start|>system\nObservation: front view: {start}{pad * 5}{end} left view: "
                f"{start}{pad * 3}{end}\nInstruction: {instruction}<|im_end|>\n" + "<|action|>" * 4)
        assert segments(text) == tokenizer.encode(text, add_special_tokens=False).ids


def test_jpeg_size_reads_the_frame_header_without_decoding():
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((37, 53, 3), np.uint8)
    for flags in ([], [cv2.IMWRITE_JPEG_PROGRESSIVE, 1]):
        data = cv2.imencode(".jpg", image, flags)[1].tobytes()
        assert is_jpeg(data) and jpeg_size(data) == (37, 53) == source_size(data)
    with pytest.raises(ValueError, match="no frame header"):
        jpeg_size(data[:20])
    png = cv2.imencode(".png", image)[1].tobytes()
    assert not is_jpeg(png)
    with pytest.raises(ValueError, match="not a JPEG"):
        jpeg_size(png)
    assert source_size(image) == (37, 53)


def test_device_image_carries_the_bytes_or_the_frame_with_the_sizes_and_the_recipe():
    data = bytes(range(256))
    image = device_image(data, [(4, 6)], [("bicubic", False)])
    assert image.data.dtype == torch.uint8 and image.data.numpy().tobytes() == data
    assert image.sizes == ((4, 6),) and image.resize == (("bicubic", False),)
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    frame.flags.writeable = False  # as decoded from the wire
    image = device_image(frame, [(2, 3), (4, 4)], (("bicubic", True), ("bilinear", True)))
    assert torch.equal(image.data, torch.from_numpy(frame.copy())) and image.sizes == ((2, 3), (4, 4))


def test_image_preprocess_device_is_cpu_or_gpu():
    check_image_preprocess_device("cpu")
    check_image_preprocess_device("gpu")
    with pytest.raises(ValueError, match="image_preprocess_device"):
        check_image_preprocess_device("cuda")


def test_exact_and_capped_grid_widths_are_distinct():
    exact, grid = BucketSpec(), BucketSpec(16, max_length=200)
    assert [s.width(50) for s in (exact, grid)] == [50, 64]
    assert exact.widths(14, 17) == [17, 16, 15, 14]
    assert grid.widths(14, 17) == [32, 16]
    assert grid.widths(190, 200) == [200, 192]  # the cap need not be a multiple of 16


def test_width_lists_pad_to_the_smallest_width_and_reject_longer_lengths():
    spec = BucketSpec(sizes=(640, 768, 896, 1024))
    assert [spec.width(n) for n in (1, 640, 641, 1024)] == [640, 640, 768, 1024]
    with pytest.raises(ValueError, match="largest width 1024"):
        spec.width(1025)
    assert spec.widths(600, 700) == [768, 640]
    assert spec.widths(1000, 1100) == [1024]  # lengths past the last width are rejected, not warmed
    assert spec.pads and not BucketSpec().pads


@pytest.mark.parametrize("options", [
    {"granularity": -1}, {"granularity": None}, {"granularity": 1.5}, {"granularity": True},
    {"max_length": 0}, {"sizes": (768, 640)}, {"sizes": (640, 640)}, {"sizes": (0,)},
    {"sizes": (640,), "granularity": 16}, {"sizes": (640,), "max_length": 1024},
])
def test_invalid_policies_fail_at_construction(options):
    with pytest.raises(ValueError):
        BucketSpec(**options)


def test_wall_oss_0_5_uses_exact_width_when_bucketing_is_disabled():
    from xinfer.models.wall_oss_0_5.processing import ProcessingConfig

    assert ProcessingConfig(prompt_bucket=0).bucket_spec().width(50) == 50
    assert ProcessingConfig().bucket_spec().width(50) == 64


def test_text_range_keeps_separators_and_caps_truncated_prompts():
    # Three state values: 1, 23, 456. Only their six digit tokens vary; the other 20 tokens stay fixed.
    assert text_length_range(26, 6, 3) == (23, 29)
    assert text_length_range(26, 6, 3, max_tokens=4) == (23, 32)
    assert text_length_range(200, 42, 14, max_length=200) == (172, 200)


class Bucketed(VLAModel):
    pad_prompt = staticmethod(lambda tensors, width: dict(tensors, width=width))  # the default pad uses it

    def __init__(self, spec, variants=None):
        self.spec, self.variants = spec, variants

    def shape_buckets(self):
        return {"prompt": self.spec}

    def warmup_requests(self, prepared):
        return self.variants or [prepared]

    @classmethod
    def build_processors(cls, checkpoint, processing=None, **_ignored):
        return ("input", checkpoint, processing), ("output", checkpoint, processing), {}


def test_model_defaults_build_the_processors_and_pad_only_the_prompt():
    model = Bucketed(BucketSpec(8))
    model.checkpoint, model.processing = "checkpoint", {"prompt_bucket": 8}
    assert model.input_processor() == ("input", "checkpoint", {"prompt_bucket": 8})
    assert model.output_processor() == ("output", "checkpoint", {"prompt_bucket": 8})
    prepared = PreparedRequest({"width": 50}, {"robot": "a"})
    padded = model.pad(prepared, "prompt", 16)
    assert padded.tensors == {"width": 16} and padded.meta is prepared.meta
    with pytest.raises(NotImplementedError):
        model.pad(prepared, "history", 4)


def test_warmup_preserves_unbucketed_requests_and_expands_each_models_variant():
    plain = PreparedRequest({"width": 50})
    ranged = PreparedRequest({"width": 50}, {"prompt_len_range": (5, 10)})
    other = PreparedRequest({"width": 50}, {"prompt_len_range": (17, 25)})
    assert warmup_requests(Bucketed(BucketSpec(8)), plain) == [plain]
    assert warmup_requests(Bucketed(BucketSpec()), ranged) == [ranged]
    variants = warmup_requests(Bucketed(BucketSpec(8), [ranged, other]), plain)
    assert [r.tensors["width"] for r in variants] == [16, 8, 32, 24]
    assert all(r.meta is ranged.meta for r in variants[:2])
    assert all(r.meta is other.meta for r in variants[2:])
