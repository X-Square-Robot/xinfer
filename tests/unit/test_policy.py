"""The ``image_preprocess_device`` model option: ``Policy``'s platform default, ``xinfer serve``'s ``"cpu"``."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from xinfer.config import ConfigError, apply_image_preprocess_device
from xinfer.platforms.cpu import cpu_platform


def _platform(device_images):
    platform = cpu_platform()
    return replace(platform, name="test", capabilities=replace(platform.capabilities, device_images=device_images))


def _device(device_images, explicit, default=None):
    """The resolved device: ``"gpu"`` when ``apply_image_preprocess_device`` sets the processing field."""
    options = apply_image_preprocess_device({"image_preprocess_device": explicit}, _platform(device_images), default)
    return options.get("processing", {}).get("image_preprocess_device", "cpu")


@pytest.mark.parametrize(("device_images", "explicit", "default", "expected"), [
    (True, None, None, "gpu"), (False, None, None, "cpu"),  # Policy: the platform's
    (True, None, "cpu", "cpu"),  # xinfer serve
    (True, "cpu", None, "cpu"), (False, "cpu", None, "cpu"), (True, "gpu", "cpu", "gpu"),
])
def test_the_default_comes_from_the_entry_point_or_the_platform_and_explicit_values_are_kept(
        device_images, explicit, default, expected):
    assert _device(device_images, explicit, default) == expected


@pytest.mark.parametrize(("explicit", "message"), [("gpu", "cannot decode images on the device"),
                                                   ("cuda", "must be")])
def test_unsupported_values_are_rejected(explicit, message):
    with pytest.raises(ConfigError, match=message):
        _device(False, explicit)


@pytest.mark.parametrize(("options", "expected"), [
    ({"quantization": "mxfp8", "image_preprocess_device": "gpu", "processing": {"noise_seed": 1}},
     {"quantization": "mxfp8", "processing": {"noise_seed": 1, "image_preprocess_device": "gpu"}}),
    ({"quantization": "mxfp8", "image_preprocess_device": "cpu"}, {"quantization": "mxfp8"}),
    ({"processing": {"noise_seed": 1}}, {"processing": {"noise_seed": 1}}),  # xinfer serve's default: "cpu"
])
def test_the_top_level_option_reaches_the_processors(options, expected):
    before = copy.deepcopy(options)
    assert apply_image_preprocess_device(options, _platform(True), default="cpu") == expected
    assert options == before


def test_the_option_is_not_a_processing_field():
    with pytest.raises(ConfigError, match="top-level model option"):
        apply_image_preprocess_device({"processing": {"image_preprocess_device": "gpu"}}, _platform(True), "cpu")


def test_the_cpu_platform_has_no_device_images():
    assert not cpu_platform().capabilities.device_images


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
    monkeypatch.setattr(policy_module._engine_factory, "create_engine", lambda platform, model, config: None)

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


def test_policy_takes_the_switch_only_at_the_top_level(policy):
    with pytest.raises(ConfigError, match="top-level model option"):
        policy(processing={"image_preprocess_device": "cpu"})
    with pytest.raises(ConfigError, match="cannot decode images"):
        policy(device_images=False, image_preprocess_device="gpu")


def test_wall_oss_0_5_processor_takes_the_device_path():
    from xinfer.models.wall_oss_0_5.processing import processing_config

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
