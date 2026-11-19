"""The entry point assembles operators and platform owners without reversing dependencies."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from xinfer.entrypoints import _factory
from xinfer.ops import Contract, OpRegistry, OpSpec, OpTable
from xinfer.platforms import resolve_platform


def _model(**attributes):
    return SimpleNamespace(stage_graph=(), create_rpu_request=lambda prepared, context: None,
                           ops=attributes.pop("ops", OpTable("rpu", {})), **attributes)


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
    def bind_ops(table):
        def prepare(context, op, **arguments):
            return getattr(table, "prepare_" + op)(context, **arguments)
        return prepare

    monkeypatch.setattr(backends, "rpu", SimpleNamespace(load_backend=lambda: provider, bind_ops=bind_ops),
                        raising=False)
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


def test_missing_operator_package_preserves_import_error(monkeypatch):
    from xinfer.ops import backends

    monkeypatch.delattr(backends, "rpu", raising=False)
    monkeypatch.setitem(sys.modules, "xinfer.ops.backends.rpu", None)
    with pytest.raises(ImportError, match="rpu"):
        _factory.create_engine(resolve_platform("rpu"), _model())


def test_engine_refuses_ambiguous_context_ownership():
    from xinfer.engine.rpu import RpuEngine

    with pytest.raises(ValueError, match="either context or context_factory"):
        RpuEngine(_model(), resolve_platform("rpu"), context=object(), context_factory=lambda: None)


@pytest.mark.parametrize("overrides", [{"reference_ops": True}, {"op_backends": {"gemm": "reference"}}])
def test_rpu_policy_passes_operator_options_to_the_shared_registry(monkeypatch, overrides):
    from xinfer import Policy
    from xinfer.entrypoints import policy as policy_module

    calls = []

    def build_table(platform, **kwargs):
        calls.append((platform, kwargs))
        raise RuntimeError("table construction reached")

    monkeypatch.setattr(policy_module.ops.registry, "load_plugins", lambda: None)
    monkeypatch.setattr(policy_module.ops.registry, "build_table", build_table)
    with pytest.raises(RuntimeError, match="table construction reached"):
        Policy("missing-model", "unused-checkpoint", device="rpu", **overrides)
    assert calls == [(resolve_platform("rpu"), {
        "backends": overrides.get("op_backends"), "reference_only": overrides.get("reference_ops", False),
    })]


@pytest.mark.parametrize("binding", ["test-double", "in-tree"])
def test_factory_uses_the_pinned_prepared_implementation(monkeypatch, binding):
    if binding == "in-tree":
        try:
            bind_ops = importlib.import_module("xinfer.ops.backends.rpu").bind_ops
        except ImportError:
            pytest.skip("requires the separate in-tree RPU operator change")
    events, _ = _provider(monkeypatch)
    if binding == "in-tree":
        from xinfer.ops.backends import rpu

        monkeypatch.setattr(rpu, "bind_ops", bind_ops)
    registry = OpRegistry()

    def reference(context, *, tensors, integers=None, scalars=None, **selectors):
        raise AssertionError("the prepared operator must use the pinned implementation")

    def selected(context, *, tensors, integers=None, scalars=None, **selectors):
        events.append(("prepare", (context, tensors, integers, scalars, selectors)))
        return SimpleNamespace(close=lambda: events.append(("operation_close", None)), entry="selected_fill")

    registry.define(OpSpec("prepare_fill", "prepare_fill(context, *, tensors, **options)", reference))
    registry.add("prepare_fill", selected, backend="rpu", platforms=("rpu",), library="selected",
                 contract=Contract(numerics="tolerance", graph_safe=False))
    platform = resolve_platform("rpu")
    table = registry.build_table(platform, backends={"prepare_fill": "rpu:selected"})

    def prepare(context):
        operation = context.prepare("fill", tensors={}, scalars={"value": 3}, mode="broadcast")
        assert operation.entry == "selected_fill"

    engine = _factory.create_engine(platform, _model(ops=table, prepare_rpu=prepare))
    try:
        assert events[1] == ("prepare", (engine.context.native, {}, None, {"value": 3}, {"mode": "broadcast"}))
        assert engine.context.stats["operation_preparations"] == 1
        assert [row["library"] for row in table.report()] == ["selected"]
    finally:
        engine.shutdown()
