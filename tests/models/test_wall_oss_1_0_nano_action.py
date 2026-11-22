"""Hard masking and FP16 rounding contracts independent of a device graph."""

import numpy as np
import pytest
import torch

from xinfer.models.registry import available_models, get_model_class
from xinfer.models.wall_oss_1_0_nano.rpu.action import action_attention_mask, euler_fp16_reference, flow_schedule


def test_same_bucket_length_change_updates_softmax_denominator():
    first = action_attention_mask(405, 448)
    second = action_attention_mask(410, 448)
    assert first.shape == second.shape == (16, 464)
    assert np.isneginf(first[:, 405:448]).all()
    assert (second[:, 405:410] == 0).all()
    probability = torch.softmax(torch.from_numpy(first).float(), dim=-1).numpy()
    assert (probability[:, 405:448] == 0).all()
    # Exactly 405 prefix +16 action tokens participate, even when the
    # physical gap has zero K/V values.
    np.testing.assert_allclose(probability[:, 0], 1 / 421, rtol=1e-7)
    assert (first[:, 448:] == 0).all()


def test_euler_retains_mask_tail_and_returns_independent_storage():
    action = np.zeros((16, 64), dtype=np.float16)
    action[:, 26:52] = 1
    velocity = np.full_like(action, 2)
    keep = np.zeros(64, dtype=np.float16)
    keep[:20] = 1
    padding = np.zeros_like(action)
    padding[:, 20:26] = -2
    result = euler_fp16_reference(action, velocity, keep, padding, np.float16(0.125))
    assert (result[:, :20] == np.float16(0.25)).all()
    assert (result[:, 20:26] == np.float16(-0.25)).all()
    assert (result[:, 26:52] == 1).all()
    assert not np.shares_memory(result, action)
    assert not action[:, :26].any()


def test_serving_schedule_is_eight_updates_with_checkpoint_time_scale():
    times, dt = flow_schedule()
    assert times.shape == (9,)
    assert times[0] == 0 and times[-1] == torch.tensor(0.999, dtype=torch.float32)
    assert dt.dtype == np.dtype("float16")
    assert float(dt) == 0.1248779296875


def test_euler_rounds_product_before_adding_residual():
    action = np.full((16, 64), -1, dtype=np.float16)
    velocity = np.full_like(action, 7.5)
    result = euler_fp16_reference(
        action, velocity, np.ones(64, dtype=np.float16), np.zeros_like(action), flow_schedule()[1]
    )
    # Exact half arithmetic yields -65/1024. A fused FP32 multiply/add
    # rounded only at the output gives the adjacent bit pattern 0xac0f.
    assert (result.view(np.uint16) == 0xAC10).all()


def test_wall_oss_1_0_nano_and_wall_oss_0_5_discover_independently():
    assert {"wall-oss-1.0-nano", "wall-oss-0.5"} <= set(available_models())
    wall_oss_1_0_nano = get_model_class("wall-oss-1.0-nano")
    wall_oss_0_5 = get_model_class("wall-oss-0.5")
    assert get_model_class("WALL-OSS-1.0-Nano") is wall_oss_1_0_nano
    assert get_model_class("WALL-OSS-0.5") is wall_oss_0_5
    assert wall_oss_1_0_nano is not wall_oss_0_5
    from xinfer.models.wall_oss_1_0_nano.rpu.model import WallOSS10NanoRPU

    assert WallOSS10NanoRPU.stage_graph.order[0].name == "vision"
    with pytest.raises(ValueError, match="only FP16"):
        wall_oss_1_0_nano.from_checkpoint("unused", device="rpu:0", quantization="mxfp8")
