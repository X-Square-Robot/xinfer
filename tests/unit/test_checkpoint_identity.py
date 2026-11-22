import json
from types import SimpleNamespace

import pytest

from xinfer.models.checkpoint_identity import validate_checkpoint_model


def test_checkpoint_without_sidecar_is_accepted(tmp_path):
    validate_checkpoint_model(tmp_path, "wall-oss-0.5")


@pytest.mark.parametrize(("declared", "expected"), [
    ("wall-oss-0.5", "wall-oss-0.5"),
    ("WALL-OSS-1.0-Nano", "wall-oss-1.0-nano"),
])
def test_checkpoint_accepts_canonical_identity(tmp_path, declared, expected):
    (tmp_path / "xinfer.json").write_text(
        json.dumps({"model_id": declared, "description": "Example export"}), encoding="utf-8"
    )
    validate_checkpoint_model(str(tmp_path), expected)


@pytest.mark.parametrize(("declared", "expected"), [
    ("wall-oss-0.5", "wall-oss-1.0-nano"),
    ("wallx2b", "wall-oss-1.0-nano"),
    ("wall_oss", "wall-oss-0.5"),
])
def test_checkpoint_rejects_wrong_or_legacy_identity(tmp_path, declared, expected):
    (tmp_path / "xinfer.json").write_text(json.dumps({"model_id": declared}), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        validate_checkpoint_model(tmp_path, expected)
    assert f"model_id is {declared!r}, expected {expected!r}" in str(error.value)


@pytest.mark.parametrize(("contents", "message"), [
    (b"{", "invalid UTF-8 JSON"),
    (b"\xff", "invalid UTF-8 JSON"),
    (b"[]", "expected a JSON object"),
    (b"{}", "model_id must be a non-empty string"),
    (b'{"model_id": 5}', "model_id must be a non-empty string"),
    (b'{"model_id": ""}', "model_id must be a non-empty string"),
])
def test_checkpoint_rejects_malformed_sidecar(tmp_path, contents, message):
    path = tmp_path / "xinfer.json"
    path.write_bytes(contents)
    with pytest.raises(ValueError) as error:
        validate_checkpoint_model(tmp_path, "wall-oss-0.5")
    assert str(path) in str(error.value)
    assert message in str(error.value)


def test_all_checkpoint_loaders_validate_the_model_identity(tmp_path):
    from xinfer.models.wall_oss_0_5.config import WallOSS05Config
    from xinfer.models.wall_oss_1_0_nano.config import WallOSS10NanoConfig
    from xinfer.models.wall_oss_1_0_nano.rpu.config import WallOSS10NanoRPUConfig

    (tmp_path / "xinfer.json").write_text(json.dumps({"model_id": "another-model"}), encoding="utf-8")
    for config, model_id in (
        (WallOSS05Config, "wall-oss-0.5"),
        (WallOSS10NanoConfig, "wall-oss-1.0-nano"),
        (WallOSS10NanoRPUConfig, "wall-oss-1.0-nano"),
    ):
        with pytest.raises(ValueError) as error:
            config.load(tmp_path)
        assert f"expected {model_id!r}" in str(error.value)


def test_wire_metadata_uses_the_released_model_ids():
    from xinfer.models.wall_oss_0_5.processing import wire_metadata as oss05_metadata
    from xinfer.models.wall_oss_1_0_nano.processing import wire_metadata as nano_metadata

    config = SimpleNamespace(action_horizon=16, action_dim=26, norm_key="default", rtc_rope_gap=10, num_steps=10)
    assert oss05_metadata(config)["model"] == "wall-oss-0.5"
    metadata = nano_metadata(config)
    assert metadata["model"] == "wall-oss-1.0-nano"
    assert metadata["model_family"] == "qwen3_5"


def test_rpu_metadata_uses_the_nano_model_id(tmp_path, monkeypatch):
    from xinfer.models.wall_oss_1_0_nano.rpu import processing
    from xinfer.models.wall_oss_1_0_nano.rpu.config import WallOSS10NanoRPUConfig
    from xinfer.models.wall_oss_1_0_nano.rpu.model import WallOSS10NanoRPU

    config = SimpleNamespace(name="wall-oss-1.0-nano-fp16-r1")
    monkeypatch.setattr(WallOSS10NanoRPUConfig, "load", lambda checkpoint: config)
    monkeypatch.setattr(processing, "WallOSS10NanoRPUInputProcessor", lambda *args, **kwargs: None)
    monkeypatch.setattr(processing, "WallOSS10NanoRPUOutputProcessor", lambda config: None)
    _, _, metadata = WallOSS10NanoRPU.build_processors(tmp_path)
    assert metadata["model"] == "wall-oss-1.0-nano"
    assert metadata["profile"] == config.name
