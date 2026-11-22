"""Unsupported RPU load profiling fails before plugins, models or request assets load."""

import pytest

from xinfer import ops
from xinfer.entrypoints import profile
from xinfer.frontend import protocol
from xinfer.models import registry


@pytest.mark.parametrize("device", ["rpu", "rpu:0"])
def test_load_rejects_rpu_before_plugin_or_asset_loading(tmp_path, monkeypatch, device):
    checkpoint = tmp_path / "missing-checkpoint"
    requests = tmp_path / "missing-requests.json"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unsupported load profiling must reject before plugins or asset loading")

    monkeypatch.setattr(ops.registry, "load_plugins", forbidden)
    monkeypatch.setattr(ops.registry, "build_table", forbidden)
    monkeypatch.setattr(registry, "get_model_class", forbidden)
    monkeypatch.setattr(protocol, "read_requests", forbidden)

    with pytest.raises(SystemExit, match="xinfer-profile load.*execution_model='cuda-lanes'"):
        profile.main(["load", "--model", "wall-oss-1.0-nano", "--checkpoint", str(checkpoint),
                      "--requests", str(requests), "--device", device, "--out", str(tmp_path / "out")])

    assert not calls
    assert not checkpoint.exists() and not requests.exists()
