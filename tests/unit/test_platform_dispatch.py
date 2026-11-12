"""The platform dispatch surface a new accelerator vendor must satisfy.

These tests are intentionally vendor-name-generic: they assert the capability-shaped dispatch,
not the presence/absence of any particular ``is_<vendor>`` predicate. When a new vendor wants to
add an accelerator, adding ``platforms/<kind>.py`` plus one lazy branch in ``detect.py`` and
declaring the right capabilities is meant to be the entire job (docs/design.md §13, §18.3).
"""

from __future__ import annotations

import pytest

from xinfer.entrypoints._factory import create_engine
from xinfer.platforms.devices import device_module_for, jpeg_decoder_for
from xinfer.platforms.interface import Platform, PlatformCapabilities


def _platform(device="cpu", *, execution_model="cuda-lanes", op_selection="table"):
    return Platform(f"test-{device}", device,
                    PlatformCapabilities(device_type=device.partition(":")[0], arch=None,
                                         cudagraphs=False, stream_priorities=False, unified_memory=True,
                                         multiprocess_service=False, sm_count=None, total_memory_bytes=None,
                                         execution_model=execution_model, op_selection=op_selection))


def test_device_module_for_does_not_resolve_non_torch_devices():
    # Non-torch-numbered platforms and CPU return None. The call must not import a vendor
    # SDK and must not return a torch-namespaced module by mistake.
    assert device_module_for("cpu") is None
    assert device_module_for("rpu:0") is None


def test_jpeg_decoder_for_rejects_devices_without_an_adapter():
    with pytest.raises(ValueError, match="need a CUDA or NPU device"):
        jpeg_decoder_for("cpu")
    with pytest.raises(ValueError, match="need a CUDA or NPU device"):
        jpeg_decoder_for("rpu:0")


def test_engine_factory_dispatches_on_capability_not_identity():
    # The factory itself must reject unknown cap values explicitly; vendor names are not part
    # of its branch shape. Identical input shape across calls selects deterministically once.
    with pytest.raises(ValueError, match="unknown execution_model"):
        create_engine(_platform(execution_model="not-a-real-mode"), model=None)


def test_engine_factory_rpu_land_on_rpu_engine():
    # Smoke: asking for the explicit-completion path constructs RpuEngine, which validates the cap and
    # refuses to run on a cuda-lanes platform. The vendor context is never constructed here.
    from xinfer.engine.rpu import RpuEngine

    model = type("M", (), {"stage_graph": ()})()
    # Reaching this far is evidence the factory took the RPU branch: the engine's capability guard
    # already matched, and the next failure is the model's missing RPU hook (not a wrong-engine
    # dispatch fallback).
    with pytest.raises(TypeError, match="RPU models must provide create_rpu_request"):
        create_engine(_platform("rpu:0", execution_model="explicit-completion", op_selection="external"), model,
                      config=None)

    # Negative: the wrong engine class for a given cap is refused by that engine's own guard,
    # so a vendor fork calling the wrong engine class is caught cheaply at the engine boundary.
    with pytest.raises(ValueError, match="explicit-completion"):
        RpuEngine(model, _platform("cpu"), None)
