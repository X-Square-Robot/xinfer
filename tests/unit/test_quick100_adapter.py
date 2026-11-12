"""The precision adapter preserves the serving image option and both model compositions."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xinfer import ops
from xinfer.platforms.cpu import cpu_platform


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    import xinfer.engine.device_images as images
    import xinfer.models.wall_oss.model as wall_oss
    import xinfer.platforms as platforms

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "tools" / "precision"))
    from wall_oss_quick100_adapter import Adapter

    platform = cpu_platform()
    platform = replace(platform, capabilities=replace(platform.capabilities, device_images=True))
    monkeypatch.setattr(platforms, "resolve_platform", lambda device: platform)
    monkeypatch.setattr(ops.registry, "load_plugins", lambda: None)
    monkeypatch.setattr(torch, "set_grad_enabled", lambda enabled: None)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {
        "min.ex_normal": torch.zeros(20), "delta.ex_normal": torch.ones(20)})
    decoder = object()
    monkeypatch.setattr(images, "DeviceImageDecoder", lambda device: decoder)
    monkeypatch.setattr(Adapter, "_sources", lambda self: {})
    monkeypatch.delenv("XINFER_XCOMPUTE_LIBRARY", raising=False)
    # The adapter inserts the source path and sets these process options.
    monkeypatch.syspath_prepend(str(root))
    for key in ("NUMBA_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(key, "test")

    def load(checkpoint, *, device, numerics, processing):
        return SimpleNamespace(numerics=numerics, processing=processing,
                               input_processor=lambda: None, output_processor=lambda: None)

    monkeypatch.setattr(wall_oss.WallOSS, "from_checkpoint", load)

    def make(**options):
        config = dict(xinfer_src=str(root), checkpoint=str(tmp_path), xinfer_device="cpu", **options)
        return Adapter(config=config, checkpoint_hashes={}, role="candidate", device=0)

    return make


def test_adapter_resolves_the_top_level_image_option(adapter):
    make = adapter
    candidate = make(image_preprocess_device="gpu")
    assert candidate.model.processing.image_preprocess_device == "gpu"
    assert candidate.decode is not None
    assert make(image_preprocess_device="cpu").decode is None
    assert make().decode is None
    nested = make(processing={"image_preprocess_device": "gpu"})
    assert nested.model.processing.image_preprocess_device == "gpu"
    assert nested.decode is not None  # the harness nests the option under "processing"; the adapter lifts it


@pytest.mark.parametrize("numerics", ["optimized", "reference"])
def test_adapter_identity_accepts_both_compositions(adapter, numerics):
    make = adapter
    candidate = make(numerics=numerics)
    identity = candidate.identity()
    assert identity["numerics"] == numerics
    assert identity["ops"] == candidate.table.report()
