"""WALL-OSS-1.0-Nano processing helpers on a synthetic checkpoint configuration (CPU, no model assets).

Covers the configuration loader, the state fields, the desktop RTC protocol and the relative/absolute pose
conversions.
"""

import numpy as np
import pytest

from xinfer.models.interface import PreparedRequest

from wall_oss_1_0_nano_checkpoint import D, H, write_checkpoint


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    from xinfer.models.wall_oss_1_0_nano.config import WallOSS10NanoConfig

    root = tmp_path_factory.mktemp("wall-oss-1.0-nano")
    write_checkpoint(root)
    return WallOSS10NanoConfig.load(root)


def test_config_reads_the_dataset_snapshot(config):
    # The vision processor's resolution replaces data.resolution, as in the reference service.
    assert config.resolution == {"face_view": 224}
    assert (config.resize_factor, config.resize_min_pixels, config.min_pixels) == (32, 4096, 65536)
    assert (config.action_dim, config.state_dim, config.num_steps) == (D, D, 10)
    assert (config.rtc_max_delay, config.rtc_rope_gap, config.state_bins) == (8, 10, 512)
    assert (config.embodiment, config.frequency_hz, config.relative_action) == ("Test desktop robot", 32, True)
    assert config.clamp_normalized and config.special_tokens == ("<point>",)


def test_arm_state_reads_canonical_fields_before_legacy_ones():
    from xinfer.models.wall_oss_1_0_nano.processing import arm_state

    legacy = {"follow1_pos": [1, 2, 3, 4, 5, 6, 7]}
    assert arm_state(legacy, "left", "follow1_pos").tolist() == [1, 2, 3, 4, 5, 6, 7]
    mixed = dict(legacy, follow_left_gripper=[0.5])
    assert arm_state(mixed, "left", "follow1_pos").tolist() == [1, 2, 3, 4, 5, 6, 0.5]
    with pytest.raises(ValueError, match="follow_right_position"):
        arm_state({"follow_right_position": [0, 0], "follow_right_rotation": [0] * 3, "follow_right_gripper": [0]},
                  "right", "follow2_pos")
    with pytest.raises(ValueError, match="follow1_pos"):
        arm_state({"follow1_pos": [0] * 6 + [np.nan]}, "left", "follow1_pos")


def test_optional_state_fields_are_masked_when_missing(config):
    from xinfer.models.wall_oss_1_0_nano.processing import build_state

    state = {"follow1_pos": [0.1] * 7, "follow2_pos": [0.2] * 7, "lift": 0.3, "head_pos": [1.0]}
    agent_pos, mask, _ = build_state(state, config)
    assert agent_pos.dtype == np.float64 and agent_pos.shape == mask.shape == (1, D)
    assert mask[0].tolist() == [1.0] * 20 + [0.0] * 3 + [1.0] + [0.0] * 2  # head_pos has the wrong size
    assert agent_pos[0, 23] == 0.3 and not agent_pos[0, 20:23].any() and not agent_pos[0, 24:].any()


def _rtc(rows, delay=3, shift=6, **extra):
    chunk = np.zeros((rows, 14), np.float32)
    chunk[:, 0] = np.arange(rows)  # row number in the left x position
    return dict(use_rtc=True, inference_delay=delay, s_model=shift, prev_action_chunk=chunk, **extra)


def test_parse_rtc_follows_the_desktop_protocol(config):
    from xinfer.models.wall_oss_1_0_nano.processing import parse_rtc

    assert parse_rtc({"use_rtc": False, "inference_delay": 3}, config) is None
    poses, delay = parse_rtc(_rtc(H), config)  # future poses only: rows 0 .. d-1
    assert delay == 3 and poses[:, 0].tolist() == [0, 1, 2]
    poses, delay = parse_rtc(_rtc(H + 1), config)  # anchor row first: rows 1 .. d
    assert delay == 3 and poses[:, 0].tolist() == [1, 2, 3]
    poses, delay = parse_rtc(_rtc(H, delay=6, shift=14), config)  # two valid rows left
    assert delay == 2 and poses[:, 0].tolist() == [0, 1]
    assert parse_rtc(_rtc(H, shift=H), config) is None  # executed chunk: a flow request
    for bad in (_rtc(H, delay=9), _rtc(H, delay=0), _rtc(H, delay=True), _rtc(H - 1), _rtc(H, shift=H + 1),
                _rtc(H, rtc_mode="guided")):
        with pytest.raises(ValueError):
            parse_rtc(bad, config)
    far = _rtc(H)
    far["prev_action_chunk"][1, 10] = 3.3  # right yaw beyond the angle limit, on a pinned row
    with pytest.raises(ValueError, match="exceed"):
        parse_rtc(far, config)
    far["prev_action_chunk"][1, 10] = 0
    far["prev_action_chunk"][5, 10] = 3.3  # not pinned at delay 3
    assert parse_rtc(far, config)[1] == 3


@pytest.mark.parametrize("column", [6, 13])
def test_rtc_gripper_limit_is_five_on_pinned_rows(config, column):
    from xinfer.models.wall_oss_1_0_nano.processing import parse_rtc

    observation = _rtc(H)
    observation["prev_action_chunk"][1, column] = 5.0
    assert parse_rtc(observation, config)[1] == 3
    observation["prev_action_chunk"][1, column] = np.nextafter(np.float32(5.0), np.float32(np.inf))
    with pytest.raises(ValueError, match="exceed 5.0"):
        parse_rtc(observation, config)
    observation["prev_action_chunk"][1, column] = 0
    observation["prev_action_chunk"][3, column] = 5.3
    assert parse_rtc(observation, config)[1] == 3


def test_rtc_prefix_and_response_compose_back_to_the_pinned_poses(config):
    """Absolute poses → relative actions → normalized → response poses gives the poses back."""
    from xinfer.models.wall_oss_1_0_nano.processing import WallOSS10NanoOutputProcessor, rtc_prefix

    rng = np.random.default_rng(0)
    d = 5

    def poses(n):
        out = np.concatenate([rng.uniform(-0.5, 0.5, (n, 3)), rng.uniform(-2.5, 2.5, (n, 1)),
                              rng.uniform(-1.2, 1.2, (n, 1)), rng.uniform(-2.5, 2.5, (n, 1)),
                              rng.uniform(0, 4, (n, 1))], axis=1)
        return out.astype(np.float32)

    arms = {"left": poses(1)[0].astype(np.float64), "right": poses(1)[0].astype(np.float64)}
    target = np.concatenate([poses(d), poses(d)], axis=1)
    relative = rtc_prefix(target, np.concatenate([arms["left"], arms["right"]]).astype(np.float32), config)
    assert relative.dtype == np.float32 and relative.shape == (d, D) and not relative[:, 20:].any()
    norm = config.action_norm
    normalized = np.zeros((1, H, D), np.float32)
    normalized[0, :d] = (relative - norm.min[:d]) / norm.delta[:d] * 2 - 1
    response = WallOSS10NanoOutputProcessor(config)({"predict_action_norm": normalized},
                                              PreparedRequest(tensors={}, meta={"arms": arms}))
    assert list(response) == ["follow1_pos", "follow2_pos", "predict_action_norm", "action_pred"]
    assert all(type(v) is float for row in response["follow1_pos"] for v in row)
    assert response["action_pred"].dtype == np.float32 and response["action_pred"].shape == (H, 14)
    np.testing.assert_array_equal(response["action_pred"][:, :7], np.asarray(response["follow1_pos"], np.float32))
    np.testing.assert_allclose(response["action_pred"][:d], target, atol=2e-5)
