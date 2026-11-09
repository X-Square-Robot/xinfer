"""``Policy``'s ``image_preprocess_device`` argument and its platform default."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from xinfer.config import ConfigError, resolve_image_preprocess_device
from xinfer.platforms.detect import _cpu_platform


def _platform(device_images):
    platform = _cpu_platform()
    return replace(platform, name="test", capabilities=replace(platform.capabilities, device_images=device_images))


@pytest.mark.parametrize(("device_images", "explicit", "expected"), [
    (True, None, "gpu"), (False, None, "cpu"), (True, "cpu", "cpu"), (False, "cpu", "cpu"), (True, "gpu", "gpu"),
])
def test_the_default_follows_the_platform_and_explicit_values_are_kept(device_images, explicit, expected):
    assert resolve_image_preprocess_device(_platform(device_images), explicit) == expected


@pytest.mark.parametrize(("explicit", "message"), [("gpu", "cannot decode images on the device"),
                                                   ("cuda", "must be")])
def test_unsupported_values_are_rejected(explicit, message):
    with pytest.raises(ConfigError, match=message):
        resolve_image_preprocess_device(_platform(False), explicit)


def test_the_cpu_platform_has_no_device_images():
    assert not _cpu_platform().capabilities.device_images


@pytest.fixture
def policy(monkeypatch):
    """``Policy(**options)`` on a stub model; returns the ``processing`` option ``from_checkpoint`` received."""
    import xinfer.entrypoints.policy as policy_module
    from test_config import ThreeStage

    model = SimpleNamespace(stage_graph=ThreeStage.stage_graph, capabilities=ThreeStage().capabilities,
                            shape_buckets=lambda: {}, io_schema=lambda: None,
                            input_processor=lambda: None, output_processor=lambda: None)
    received = []

    def load(checkpoint, *, device, ops, **options):
        received.append(options.get("processing"))
        return model

    monkeypatch.setattr(policy_module, "get_model_class", lambda name: SimpleNamespace(from_checkpoint=load))
    monkeypatch.setattr(policy_module.ops.registry, "load_plugins", lambda: None)
    monkeypatch.setattr(policy_module, "Engine", lambda model, platform, config: None)

    def make(device_images=True, **options):
        monkeypatch.setattr(policy_module, "resolve_platform", lambda device: _platform(device_images))
        policy_module.Policy("test", "unused", **options)
        return received[-1]

    return make


def test_policy_sets_the_processing_field_only_for_the_device_path(policy):
    assert policy(processing={"noise_seed": 1}) == {"noise_seed": 1, "image_preprocess_device": "gpu"}
    assert policy() == {"image_preprocess_device": "gpu"}
    assert policy(image_preprocess_device="cpu", processing={"noise_seed": 1}) == {"noise_seed": 1}
    assert policy(device_images=False) is None


def test_policy_takes_the_switch_only_as_its_argument(policy):
    with pytest.raises(ConfigError, match="Policy argument"):
        policy(processing={"image_preprocess_device": "cpu"})
    with pytest.raises(ConfigError, match="cannot decode images"):
        policy(device_images=False, image_preprocess_device="gpu")


def test_wall_oss_processor_takes_the_device_path():
    from xinfer.models.wall_oss.processing import processing_config

    assert processing_config({"image_preprocess_device": "gpu"}).image_preprocess_device == "gpu"


def test_policy_prepares_through_one_connection_of_the_processor(monkeypatch):
    import xinfer.entrypoints.policy as policy_module
    from echo_model import EchoModel

    monkeypatch.setattr(policy_module, "get_model_class", lambda name: EchoModel)
    policy = policy_module.Policy("echo", "unused", device="cpu", cudagraphs=False)
    try:
        assert [policy.prepare({"x": [0.0]}).meta["connection_call"] for _ in range(3)] == [1, 2, 3]
    finally:
        policy.engine.shutdown()
