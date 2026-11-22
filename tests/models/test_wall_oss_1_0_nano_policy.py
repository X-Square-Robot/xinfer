"""End-to-end WALL-OSS-1.0-Nano inference using a supplied checkpoint and synthetic observations."""

import os

import numpy as np
import pytest
import torch

from xinfer import Policy

CHECKPOINT = os.environ.get("XINFER_WALL_OSS_1_0_NANO_CHECKPOINT")
pytestmark = [
    pytest.mark.assets,
    pytest.mark.gpu,
    pytest.mark.skipif(not (CHECKPOINT and torch.cuda.is_available()),
                       reason="needs CUDA and XINFER_WALL_OSS_1_0_NANO_CHECKPOINT"),
]


@pytest.mark.parametrize("numerics", ["reference", "optimized"])
def test_policy_infers_flow_and_pins_the_rtc_prefix(numerics):
    rng = np.random.default_rng(42)
    observation = {
        "views": {name: rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                  for name in ("camera_front", "camera_left", "camera_right")},
        "state": {"follow1_pos": [0.0] * 7, "follow2_pos": [0.0] * 7},
        "instruction": "Pick up the block.",
        "init_noise": rng.standard_normal((16, 26), dtype=np.float32),
    }
    policy = Policy("wall-oss-1.0-nano", CHECKPOINT, device="cuda", numerics=numerics, cudagraphs=False,
                    reference_ops=numerics == "reference", image_preprocess_device="cpu")
    try:
        flow = policy.infer(observation)
        assert flow["action_pred"].shape == (16, 14)
        assert flow["predict_action_norm"].shape == (1, 16, 26)
        assert np.isfinite(flow["action_pred"]).all() and np.isfinite(flow["predict_action_norm"]).all()
        poses = np.concatenate([flow["follow1_pos"], flow["follow2_pos"]], axis=1)
        np.testing.assert_array_equal(flow["action_pred"], poses)

        delay = 3
        rtc = dict(observation, use_rtc=True, prev_action_chunk=np.zeros((16, 14), np.float32),
                   inference_delay=delay, s_model=0)
        pinned = policy.prepare(rtc).tensors["rtc_actions"][0, :delay].numpy()
        response = policy.infer(rtc)
        np.testing.assert_array_equal(response["predict_action_norm"][0, :delay], pinned)
        assert np.isfinite(response["action_pred"]).all()
    finally:
        policy.engine.shutdown()
