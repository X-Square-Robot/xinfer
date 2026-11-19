"""Independent HF CPU position reference; no model/weights are constructed."""

from types import SimpleNamespace
import inspect
import os

import numpy as np
import pytest
import torch

from xinfer.models.wallx2b.rpu.processing import IMAGE_GRIDS, mrope_position_ids

IMAGE_TOKEN, VISION_START, VISION_END, ACTION_TOKEN = 248056, 248053, 248054, 248078


def _reference():
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    class Reference:
        compute_3d_position_ids = Qwen3_5Model.compute_3d_position_ids
        get_rope_index = Qwen3_5Model.get_rope_index

    if hasattr(Qwen3_5Model, "get_vision_position_ids"):
        Reference.get_vision_position_ids = Qwen3_5Model.get_vision_position_ids
    reference = Reference()
    reference.config = SimpleNamespace(
        image_token_id=IMAGE_TOKEN,
        video_token_id=248057,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    reference.rope_deltas = None
    return reference


def _request(prefix_length):
    prefix = [17] * 16
    for t, h, w in IMAGE_GRIDS:
        prefix += [VISION_START] + [IMAGE_TOKEN] * (t * h * w // 4) + [VISION_END] + [23] * 3
    assert len(prefix) <= prefix_length
    prefix += [31] * (prefix_length - len(prefix))
    return np.array([prefix + [ACTION_TOKEN] * 16], dtype=np.int64)


def _compute(reference, ids):
    kwargs = dict(
        input_ids=torch.from_numpy(ids),
        inputs_embeds=None,
        image_grid_thw=torch.tensor(IMAGE_GRIDS, dtype=torch.int64),
        video_grid_thw=None,
        attention_mask=torch.ones_like(torch.from_numpy(ids)),
        past_key_values=None,
    )
    if "mm_token_type_ids" in inspect.signature(reference.compute_3d_position_ids).parameters:
        kwargs["mm_token_type_ids"] = torch.from_numpy((ids == IMAGE_TOKEN).astype(np.int32))
    return reference.compute_3d_position_ids(**kwargs)


@pytest.mark.parametrize("prefix_length", [319, 320, 321, 383, 384, 385, 447, 448, 449, 511, 512])
def test_rpu_profile_positions_match_hf_across_physical_bucket_boundaries(prefix_length):
    ids = _request(prefix_length)
    reference = _reference()
    expected = _compute(reference, ids)
    actual = mrope_position_ids(ids, IMAGE_TOKEN)
    np.testing.assert_array_equal(actual, expected.numpy())
    assert int(reference.rope_deltas.item()) == expected.max().item() + 1 - ids.shape[1]
    assert actual[0, 0, prefix_length] != prefix_length


def test_request_a_b_a_does_not_reuse_an_old_rope_delta():
    reference = _reference()
    results = []
    for length in (383, 321, 383):
        ids = _request(length)
        expected = _compute(reference, ids)
        actual = mrope_position_ids(ids, IMAGE_TOKEN)
        np.testing.assert_array_equal(actual, expected.numpy())
        results.append(actual)
    np.testing.assert_array_equal(results[0], results[2])


@pytest.mark.assets
def test_actual_checkpoint_prompt_positions_and_non_rtc_gap():
    checkpoint = os.environ.get("XINFER_WALLX2B_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set XINFER_WALLX2B_CHECKPOINT for the actual prompt reference")
    from xinfer.models.wallx2b.rpu.processing import DEFAULT_CAMERAS, WallX2BInputProcessor

    images = {
        name: np.full((h, w, 3), index * 37, dtype=np.uint8)
        for index, (name, (h, w)) in enumerate(zip(DEFAULT_CAMERAS, ((720, 1280), (480, 640), (480, 640))))
    }
    processor = WallX2BInputProcessor(checkpoint, image_workers=0)
    reference = _reference()
    outputs = []
    try:
        for instruction in ("show the barcode", " ".join(["lift"] * 48), "show the barcode"):
            prepared = processor(
                dict(
                    images=images,
                    instruction=instruction,
                    state=np.zeros(26, np.float32),
                    initial_noise=np.zeros((1, 16, 26), np.float32),
                )
            )
            tensors = prepared.tensors
            prefix = tensors["input_ids"].numpy()
            ids = np.concatenate((prefix, np.full((1, 16), ACTION_TOKEN, dtype=np.int64)), axis=1)
            expected = _compute(reference, ids).numpy()
            np.testing.assert_array_equal(tensors["position_ids"].numpy(), expected[..., :-16])
            np.testing.assert_array_equal(tensors["action_position_ids"].numpy(), expected[..., -16:] + 10)
            outputs.append(tensors["position_ids"].numpy())
        np.testing.assert_array_equal(outputs[0], outputs[2])
    finally:
        processor.close()
