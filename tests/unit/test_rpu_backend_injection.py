"""The entry point assembles operators and platform owners without reversing dependencies."""

import sys
from types import SimpleNamespace

import pytest

from xinfer.entrypoints import _factory
from xinfer.platforms import resolve_platform


def _model(**attributes):
    return SimpleNamespace(stage_graph=(), create_rpu_request=lambda prepared, context: None, **attributes)


def _provider(monkeypatch, *, runtime=True):
    from xinfer.ops import backends

    events = []

    class NativeContext:
        asset_provenance = {"directory": "test-assets", "manifest_sha256": "b" * 64}

        def __init__(self, **options):
            events.append(("create", options))

        def close(self):
            events.append(("close", None))

    provenance = {"library": "test-native-library", "abi_version": 11, "runtime_available": runtime}
    provider = SimpleNamespace(backend_factory=NativeContext, provenance=provenance)
    monkeypatch.setattr(backends, "rpu", SimpleNamespace(load_backend=lambda: provider), raising=False)
    return events, provenance


def test_factory_injects_provider_and_engine_owns_the_created_context(monkeypatch):
    events, provenance = _provider(monkeypatch)
    model = _model(rpu_allow_candidate=True, rpu_retained_segments=192)
    engine = _factory.create_engine(resolve_platform("rpu"), model)
    try:
        assert engine._owns_context
        reported = engine.context.describe()["provenance"]
        assert all(reported[name] == value for name, value in provenance.items())
        assert engine.context.provenance["assets"] == engine.context.native.asset_provenance
        assert engine.context.provenance["assets"] is not engine.context.native.asset_provenance
        assert events == [("create", {"allow_candidate": True, "retained_segments": 192})]
    finally:
        engine.shutdown()
    engine.shutdown()
    assert events[-1] == ("close", None) and len(events) == 2


def test_factory_rejects_host_only_provider_before_native_context_creation(monkeypatch):
    events, _ = _provider(monkeypatch, runtime=False)
    with pytest.raises(RuntimeError, match="host-only"):
        _factory.create_engine(resolve_platform("rpu"), _model())
    assert not events


def test_preparation_failure_closes_the_owned_injected_context(monkeypatch):
    events, _ = _provider(monkeypatch)
    primary = ValueError("model preparation failed")

    def prepare(context):
        assert context.provenance["abi_version"] == 11
        raise primary

    model = _model(prepare_rpu=prepare, close_rpu=lambda: events.append(("model_close", None)))
    with pytest.raises(ValueError) as caught:
        _factory.create_engine(resolve_platform("rpu"), model)
    assert caught.value is primary
    assert [kind for kind, _ in events] == ["create", "model_close", "close"]


def test_missing_operator_package_reports_the_required_installation(monkeypatch):
    from xinfer.ops import backends

    monkeypatch.delattr(backends, "rpu", raising=False)
    monkeypatch.setitem(sys.modules, "xinfer.ops.backends.rpu", None)
    with pytest.raises(RuntimeError, match="in-tree RPU operator backend is unavailable"):
        _factory.create_engine(resolve_platform("rpu"), _model())


def test_engine_refuses_ambiguous_context_ownership():
    from xinfer.engine.rpu import RpuEngine

    with pytest.raises(ValueError, match="either context or context_factory"):
        RpuEngine(_model(), resolve_platform("rpu"), context=object(), context_factory=lambda: None)


@pytest.mark.parametrize("overrides", [{"reference_ops": True}, {"op_backends": {"gemm": "reference"}}])
def test_model_bound_policy_rejects_tensor_table_overrides_before_model_loading(overrides):
    from xinfer import Policy

    with pytest.raises(ValueError, match="op_selection='model-bound'"):
        Policy("missing-model", "unused-checkpoint", device="rpu", **overrides)
