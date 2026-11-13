"""Board-free numerical contracts; real HF pixel parity is asset opt-in."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import pytest
import torch
from PIL import Image

from xinfer.models.wallx2b.rpu import processing
from xinfer.models.interface import PreparedRequest
from xinfer.models.wallx2b.rpu.geometry import (
    absolute14_metrics,
    euler_zyx_to_matrix,
    ground_truth_window,
    relative26_to_absolute14,
)
from xinfer.models.wallx2b.rpu.processing import (
    DEFAULT_CAMERAS,
    WallX2BInputProcessor,
    WallX2BOutputProcessor,
    build_prompt,
    dataset_v2_size,
    denormalize_actions,
    digitize_state,
    mrope_position_ids,
    normalize_state,
    pack_image_array,
    prepare_noise,
    processor_size,
)


def test_two_resize_geometry_preserves_ces_grids():
    first = [dataset_v2_size(size) for size in ((1280, 720), (640, 480), (640, 480))]
    assert first == [(128, 224), (160, 224), (160, 224)]
    second = [processor_size(h, w, min_pixels=65536, max_pixels=16777216) for h, w in first]
    assert second == [(224, 352), (224, 320), (224, 320)]


def test_patch_content_order_and_temporal_duplication():
    image = np.arange(64 * 64 * 3, dtype=np.uint32).reshape(64, 64, 3).astype(np.uint8)
    actual = pack_image_array(image).reshape(-1, 3, 2, 16, 16)
    expected = []
    for merge_h in range(2):
        for merge_w in range(2):
            for a in range(2):
                for b in range(2):
                    patch = image[
                        (2 * merge_h + a) * 16 : (2 * merge_h + a + 1) * 16,
                        (2 * merge_w + b) * 16 : (2 * merge_w + b + 1) * 16,
                    ]
                    patch = ((patch.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)).astype(np.float16)
                    expected.append(np.repeat(patch.transpose(2, 0, 1)[:, None], 2, axis=1))
    np.testing.assert_array_equal(actual, np.asarray(expected))


def test_state_clips_before_float32_right_boundary_binning():
    state = np.zeros(26, dtype=np.float32)
    state[:4] = [-2, -1, -1 + 1 / 256, 2]
    normalized = normalize_state(state, np.full(26, -1, np.float32), np.full(26, 2, np.float32))
    assert digitize_state(normalized).split()[:4] == ["0", "0", "1", "511"]
    assert len(digitize_state(normalized).split()) == 20


def test_prompt_uses_task_detail_and_exact_checkpoint_embodiment():
    prefix, postfix = build_prompt({"task": " show   barcode ", "detail": " turn it "}, state_string="1 2")
    assert "Embodiment: X2 EX001 Desktop" in prefix
    assert "Camera Setup: front view, left wrist view, right wrist view," in prefix
    assert "Task: show barcode\nInstruction: turn it" in prefix
    assert prefix.count("<|image_pad|>") == 3
    assert postfix == "<|action|>" * 16


def test_multimodal_rope_is_independent_of_physical_prefix_length():
    ids = np.array([[1, 99, 99, 99, 99, 2, 3]], dtype=np.int64)
    positions = mrope_position_ids(ids, 99, [(1, 4, 4)])
    np.testing.assert_array_equal(
        positions[:, 0], [[0, 1, 1, 1, 1, 3, 4], [0, 1, 1, 2, 2, 3, 4], [0, 1, 2, 1, 2, 3, 4]]
    )
    assert positions[0, 0, -1] != ids.shape[1] - 1


def test_explicit_noise_keeps_bytes_and_owns_its_snapshot():
    original = np.linspace(-1, 1, 16 * 26, dtype=np.float32).reshape(1, 16, 26)
    snapshot = original.copy()
    noise, packed, manifest = prepare_noise(original)
    original.fill(12)
    np.testing.assert_array_equal(noise.numpy(), snapshot)
    np.testing.assert_array_equal(packed.numpy(), snapshot.astype(np.float16))
    assert manifest["source"] == "explicit"
    assert manifest["fp32"]["dtype"] == "<f4"
    assert manifest["fp32"]["sha256"] != manifest["fp16"]["sha256"]
    with pytest.raises(ValueError, match="either"):
        prepare_noise(snapshot, 0)


def test_default_seed_restarts_per_request_like_reference():
    a, _, _ = prepare_noise()
    prepare_noise(noise_seed=17)
    b, _, _ = prepare_noise()
    assert torch.equal(a, b)


def test_relative_rotation_composes_delta_on_left():
    state = np.zeros(26, np.float32)
    relative = np.zeros((1, 26), np.float32)
    state_rotation = euler_zyx_to_matrix([0.3, -0.2, 0.7])
    delta = euler_zyx_to_matrix([0.1, 0.2, -0.4])
    for pos, rot, grip in ((0, 3, 9), (10, 13, 19)):
        state[pos : pos + 3] = [1, 2, 3]
        state[rot : rot + 6] = state_rotation[:2].reshape(-1)
        relative[0, pos : pos + 3] = [0.1, 0.2, 0.3]
        relative[0, rot : rot + 6] = delta[:2].reshape(-1)
        relative[0, grip] = 0.6
    result = relative26_to_absolute14(relative, state)
    np.testing.assert_allclose(euler_zyx_to_matrix(result[0, 3:6]), delta @ state_rotation, atol=1e-7)
    np.testing.assert_allclose(result[0, :3], [1.1, 2.2, 3.3], atol=1e-6)
    assert result[0, 6] == np.float32(0.6)


def test_output_has_independent_fp32_storage():
    state = np.zeros(26, np.float32)
    state[[3, 7, 13, 17]] = 1
    raw = np.zeros((1, 16, 26), np.float32)
    raw[:, :, [3, 7, 13, 17]] = 1
    out = WallX2BOutputProcessor(include_absolute14=True)(
        {"predict_action": raw}, PreparedRequest({}, {"state26": state})
    )
    raw.fill(7)
    assert out["predict_action"][0, 0, 0] == 0
    assert out["action_pred"].shape == (16, 14)
    assert out["gt_prediction_frame_offset"] == 1


def test_gt_begins_after_anchor_and_metrics_keep_units_separate():
    trajectory = []
    for i in range(4):
        trajectory.append(
            {
                f"master_{s}_{k}": ([i] * 3 if k == "position" else [0] * 3 if k == "rotation" else i)
                for s in ("left", "right")
                for k in ("position", "rotation", "gripper")
            }
        )
    target = ground_truth_window(trajectory, 0, 2)
    assert target[:, 0].tolist() == [1, 2]
    prediction = target.copy()
    prediction[:, (0, 1, 2, 7, 8, 9)] += 0.25
    prediction[:, (6, 13)] += 0.5
    metrics = absolute14_metrics(prediction, target)
    assert metrics["position_mae"] == 0.25
    assert metrics["gripper_mae"] == 0.5
    assert metrics["rotation_geodesic_deg"] == 0


def test_denormalize_preserves_reference_float32_operation_order():
    x = np.linspace(-2, 2, 416, dtype=np.float32).reshape(1, 16, 26)
    low, span = np.full((16, 26), 0.17, np.float32), np.full((16, 26), 0.31, np.float32)
    expected = ((torch.from_numpy(x) + 1.0) * 0.5) * torch.from_numpy(span)[None] + torch.from_numpy(low)[None]
    np.testing.assert_array_equal(denormalize_actions(x, low, span), expected.numpy())


def test_cancelled_request_still_drains_failed_camera_decode(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def decode(_):
        started.set()
        assert release.wait(5)
        raise ValueError("injected camera decode failure")

    monkeypatch.setattr(processing, "_decode_image", decode)
    processor = object.__new__(WallX2BInputProcessor)
    with ThreadPoolExecutor(max_workers=1) as pool:
        camera = processor._submit_camera(pool, None, 0, (128, 224), (224, 352), None, None)
        try:
            assert started.wait(5)
            # This is the request's finally cleanup: cancellation must not
            # complete the wrapper while an actual decode still owns input.
            assert not camera.cancel()
            assert not camera.done()
        finally:
            release.set()
        done, pending = wait([camera], timeout=5)
        assert done == {camera} and not pending
        with pytest.raises(ValueError, match="injected camera decode failure"):
            camera.result()


@pytest.mark.parametrize("stage", ["bands", "packs"])
def test_partial_camera_submission_failure_drains_submitted_jobs(monkeypatch, stage):
    started, release, rejected = threading.Event(), threading.Event(), threading.Event()
    processor = object.__new__(WallX2BInputProcessor)
    processor._stage1_bands = lambda *args: [(0, 32), (32, 64)]
    processor._processor_resize_bands = lambda *args: np.zeros((64, 32, 3), dtype=np.uint8)
    processor._resize_band = lambda *args: None
    processor._pack_image_rows = lambda *args: None
    monkeypatch.setattr(processing, "_decode_image", lambda _: None)

    def blocked(*args):
        started.set()
        assert release.wait(5)

    target = "_resize_band" if stage == "bands" else "_pack_image_rows"
    setattr(processor, target, blocked)

    with ThreadPoolExecutor(max_workers=4) as pool:
        class RejectSecond:
            count = 0

            def submit(self, function, *args):
                if function is blocked:
                    self.count += 1
                    if self.count == 2:
                        rejected.set()
                        raise RuntimeError("injected executor shutdown during submission")
                return pool.submit(function, *args)

        camera = processor._submit_camera(
            RejectSecond(), None, 0, (64, 32), (64, 32), np.empty((8, 1536), np.float16), None)
        try:
            assert rejected.wait(5) and started.wait(5)
            assert not camera.done(), "a submitted child still owns the request buffers"
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="injected executor shutdown"):
            camera.result(timeout=5)


@pytest.mark.parametrize("source_size", [(1280, 720), (640, 480)])
def test_banded_camera_pixels_match_serial_two_resize_bytes(source_size):
    processor = object.__new__(WallX2BInputProcessor)
    processor._pack_scratch = threading.local()
    processor.audit_inputs = False
    width, height = source_size
    image = Image.fromarray(np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3).astype(np.uint8))
    size1 = dataset_v2_size(source_size)
    size2 = processor_size(*size1, min_pixels=65536, max_pixels=16777216)
    expected, _ = processor._camera(image, size1, size2)
    actual = np.empty_like(expected)
    delivered = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        camera = processor._submit_camera(
            pool, image, 1, size1, size2, actual,
            lambda index, output: delivered.append((index, output.copy())))
        camera.result(timeout=10)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert len(delivered) == 1 and delivered[0][0] == 1
    np.testing.assert_array_equal(delivered[0][1].view(np.uint16), expected.view(np.uint16))


@pytest.mark.assets
def test_checkpoint_processor_two_resize_pixels_and_parallel_requests():
    checkpoint = os.environ.get("XINFER_WALLX2B_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set XINFER_WALLX2B_CHECKPOINT to admitted external small assets")
    from PIL import Image

    images = {
        name: Image.fromarray((np.arange(h * w * 3, dtype=np.uint32).reshape(h, w, 3) + i * 31).astype(np.uint8))
        for i, (name, (h, w)) in enumerate(zip(DEFAULT_CAMERAS, ((720, 1280), (480, 640), (480, 640))))
    }
    processor = WallX2BInputProcessor(checkpoint, image_workers=6, audit_inputs=True)
    try:
        obs = dict(
            images=images,
            instruction="show the barcode",
            state=np.zeros(26, np.float32),
            initial_noise=np.zeros((1, 16, 26), np.float32),
        )
        actual = processor(obs)
        expected = processor.processor.image_processor(
            images=[
                image.resize(size, Image.Resampling.BICUBIC)
                for image, size in zip(images.values(), ((224, 128), (224, 160), (224, 160)))
            ],
            return_tensors="pt",
        )
        assert torch.equal(actual.tensors["pixel_values"], expected["pixel_values"].half())
        assert actual.tensors["image_grid_thw"].tolist() == [[1, 14, 22], [1, 14, 20], [1, 14, 20]]
        assert "pixel_values" in actual.meta["inputs"]
        assert all(image["stage1"]["sha256"] for image in actual.meta["resized_images"])
        repeat = processor(obs)
        for key in actual.tensors:
            if isinstance(actual.tensors[key], torch.Tensor):
                assert torch.equal(actual.tensors[key], repeat.tensors[key]), key
        raw_positions = mrope_position_ids(
            torch.cat((actual.tensors["input_ids"], torch.full((1, 16), 248078)), 1).numpy(),
            processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        )
        np.testing.assert_array_equal(actual.tensors["action_position_ids"].numpy(), raw_positions[..., -16:] + 10)
    finally:
        processor.close()
