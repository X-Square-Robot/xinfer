"""Model admission and resident stage integration, without device arithmetic."""

from types import SimpleNamespace
from collections import Counter

import numpy as np
import pytest
import torch

from xinfer.models.interface import PreparedRequest
from xinfer.models.wallx2b.rpu.model import WallX2B


@pytest.mark.parametrize("options,error", [
    ({"device": "cpu"}, "requires device"),
    ({"device": "rpu:0", "quantization": "bf16"}, "only FP16"),
    ({"device": "rpu:0"}, "allow_candidate"),
    ({"device": "rpu:0", "allow_candidate": True, "max_cached_signatures": 9}, "signatures"),
    ({"device": "rpu:0", "allow_candidate": True, "unknown": True}, "unsupported"),
    ({"device": "rpu:0", "allow_candidate": True, "prefill_collective_window": 16}, "window"),
    ({"device": "rpu:0", "allow_candidate": True, "prefill_collective_window": 0}, "window"),
    ({"device": "rpu:0", "allow_candidate": True, "prefill_collective_window": True}, "window"),
])
def test_unsupported_profile_rejects_before_checkpoint_reads(monkeypatch, options, error):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid execution profile reached checkpoint loading")

    monkeypatch.setattr("xinfer.models.wallx2b.rpu.model.preflight_checkpoint", unexpected)
    with pytest.raises(ValueError, match=error):
        WallX2B.from_checkpoint("unused", **options)


def test_resident_stage_plan_orders_embedding_scatter_and_materializes_physical_actions():
    model = WallX2B()
    context = object()
    model.context, model._prepared, model._invalid = context, True, False
    model.stats = Counter()
    events = []
    invalidated = []
    def state_owner():
        return SimpleNamespace(invalidate=lambda: invalidated.append(True))
    model.prefix_state = SimpleNamespace(recurrent=state_owner(), history=state_owner(), history_out=state_owner(),
                                         key_caches={i: state_owner() for i in range(6)},
                                         value_caches={i: state_owner() for i in range(6)})
    vision_executions, input_executions, text_executions, action_executions = (
        (object(), object()), (object(),), (object(),), (object(), object())
    )
    prefix = SimpleNamespace(input_executions=input_executions, executions=text_executions,
                             bind=lambda positions, input_ids: events.append("prefix_bind"))
    normalized = np.full((1, 16, 26), 0.25, np.float32)
    binding = SimpleNamespace(executions=action_executions, collect_normalized=lambda: normalized,
                              close=lambda: events.append("release"))
    model.vision = SimpleNamespace(executions=vision_executions, bind=lambda pixels: events.append("vision_bind"))
    model.action = SimpleNamespace(bind=lambda request: events.append("action_bind") or binding,
                                   destination_caches=[(state_owner(), state_owner()) for _ in range(6)])
    model._prefix_plan = lambda request: prefix
    model.normalizers = dict(action_min=np.full((16, 26), -2, np.float32),
                             action_delta=np.full((16, 26), 8, np.float32))
    prepared = PreparedRequest(dict(pixel_values=object(), position_ids=object(), input_ids=object()))
    plan = model.create_rpu_request(prepared, context)
    assert events == ["prefix_bind", "action_bind", "vision_bind"]
    assert len(invalidated) == 27
    assert plan.stages == {"vision": vision_executions,
                           "prefill": (*input_executions, *text_executions), "action": action_executions}
    first = plan.collect()["predict_action"]
    np.testing.assert_array_equal(first, np.full((1, 16, 26), 3, np.float32))
    normalized.fill(-1)
    np.testing.assert_array_equal(first, np.full((1, 16, 26), 3, np.float32))
    assert first.flags.owndata and first.dtype == np.float32
    plan.release()
    assert events[-1] == "release"
    plan.abort(RuntimeError("injected execution failure"))
    with pytest.raises(RuntimeError, match="recovered"):
        model.create_rpu_request(prepared, context)


def test_split_finish_failure_retains_release_owner_without_admitting_rebuild():
    model = WallX2B()
    model.context = SimpleNamespace(_resources=[])
    model._base_resources = frozenset()
    model._prepared, model._invalid = True, False
    state = SimpleNamespace(invalidate=lambda: None)
    model.stats = Counter()
    model.prefix_state = SimpleNamespace(recurrent=state, history=state, history_out=state,
                                         key_caches={}, value_caches={})
    released = []
    binding = SimpleNamespace(close=lambda: released.append(True))
    model._prefix_plan = lambda prepared: SimpleNamespace(bind=lambda *args, **kwargs: None)
    model.action = SimpleNamespace(bind=lambda prepared: binding, destination_caches=[])
    def reject(pixels):
        raise ValueError("bad pixels")
    model.vision = SimpleNamespace(bind=reject)
    def rebuild():
        assert model._invalid
        with pytest.raises(RuntimeError, match="recovered"):
            model.begin_rpu_request(prepared, model.context)
        setattr(model, "_invalid", False)
    model._build_plans = rebuild
    model._close_resources = lambda resources: []
    prepared = PreparedRequest(dict(pixel_values=object(), position_ids=object(), input_ids=object()))
    binding = model.begin_rpu_request(prepared, model.context)
    with pytest.raises(ValueError, match="bad pixels"):
        model.finish_rpu_request(binding, prepared)
    assert model._invalid and not released
    model.rollback_rpu_request(RuntimeError("request was not admitted"))
    assert released == [True] and not model._invalid

def test_early_action_binding_failure_rebuilds_owners_without_native_recovery():
    model = WallX2B()
    retired, retained, rebuilt = object(), object(), []
    model.context = SimpleNamespace(_resources=[retired, retained])
    model._base_resources = frozenset((retained,))
    model._prepared, model._invalid = True, False
    state = SimpleNamespace(invalidate=lambda: None)
    model.stats = Counter()
    model.prefix_state = SimpleNamespace(recurrent=state, history=state, history_out=state,
                                         key_caches={}, value_caches={})
    closed = []
    model._close_resources = lambda resources: closed.extend(resources)
    model._prefix_plan = lambda prepared: SimpleNamespace(bind=lambda *args, **kwargs: None)
    def reject(prepared):
        raise ValueError("bad action")
    model.action = SimpleNamespace(bind=reject, destination_caches=[])
    model._build_plans = lambda: (rebuilt.append(True), setattr(model, "_invalid", False))
    with pytest.raises(ValueError, match="bad action"):
        model.begin_rpu_request(PreparedRequest(dict(position_ids=object(), input_ids=object())), model.context)
    assert model._invalid and not rebuilt
    model._request_rollback(None, ValueError("bad action"))
    assert closed == [retired] and rebuilt == [True] and not model._invalid
    with pytest.raises(ValueError, match="bad action"):
        model.create_rpu_request(PreparedRequest(dict(position_ids=object(), input_ids=object())), model.context)
    model.rollback_rpu_request(RuntimeError("bad action"))
    assert model._invalid  # No Action binding exists to release or rebuild.
    assert closed == [retired] and rebuilt == [True]


def test_serving_dispatch_preserves_operator_table_and_device(monkeypatch):
    from xinfer.models.wallx2b import serving

    calls = []
    gpu = SimpleNamespace(from_checkpoint=lambda path, **kwargs: calls.append(("gpu", path, kwargs)))
    rpu = SimpleNamespace(from_checkpoint=lambda path, **kwargs: calls.append(("rpu", path, kwargs)))
    monkeypatch.setattr("xinfer.models.wallx2b.model.Wallx2b", gpu)
    monkeypatch.setattr("xinfer.models.wallx2b.rpu.model.WallX2B", rpu)
    table = object()
    serving.Wallx2bServing.from_checkpoint("gpu-checkpoint", device="cuda:0", ops=table)
    serving.Wallx2bServing.from_checkpoint("rpu-checkpoint", device="rpu:0", ops=table)
    assert calls == [
        ("gpu", "gpu-checkpoint", {"device": "cuda:0", "ops": table}),
        ("rpu", "rpu-checkpoint", {"device": "rpu:0", "ops": table}),
    ]


def test_image_scatter_signature_tracks_exact_canonical_runs():
    types = torch.zeros((1, 350), dtype=torch.int32)
    for begin, end in ((12, 89), (100, 170), (180, 250)):
        types[:, begin:end] = 1
    prepared = PreparedRequest(dict(mm_token_type_ids=types))
    assert WallX2B._image_starts(prepared) == (12, 100, 180)
    types[:, 249] = 0
    with pytest.raises(ValueError, match="canonical image runs"):
        WallX2B._image_starts(prepared)


def test_model_close_releases_weights_in_borrowed_context_without_closing_external_owners():
    class Owner:
        def __init__(self):
            self.closed = False
        def close(self):
            self.closed = True

    external, weight, workspace = Owner(), Owner(), Owner()
    model = WallX2B()
    model.context = SimpleNamespace(_resources=[external, weight, workspace])
    model._initial_resources = frozenset((external,))
    model._base_resources = frozenset((external, weight, workspace))
    model._signatures, model.weights, model._prepared = {}, {"weight": weight}, True
    model.close_rpu()
    assert not external.closed and weight.closed and workspace.closed
    assert not model.weights and not model._prepared


def test_partial_preparation_can_close_before_packed_weight_base_is_published():
    class Owner:
        closed = False
        def close(self):
            self.closed = True

    partial = Owner()
    model = WallX2B()
    model.context = SimpleNamespace(_resources=[partial])
    model._initial_resources = frozenset()
    model._signatures, model.weights = {}, {"partial": partial}
    model.close_rpu()
    assert partial.closed and not model.weights


@pytest.mark.parametrize("options,expected", [
    ({}, 1), *[({"prefill_collective_window": window}, window) for window in (2, 4, 8, 15)],
])
def test_prefill_window_options_are_retained_without_device_initialization(monkeypatch, options, expected):
    manifest = SimpleNamespace(config=SimpleNamespace())
    monkeypatch.setattr("xinfer.models.wallx2b.rpu.model.preflight_checkpoint", lambda path: manifest)
    monkeypatch.setattr(WallX2B, "build_processors", classmethod(lambda cls, path, **kw: (object(), object(), None)))
    model = WallX2B.from_checkpoint("unused", device="rpu:0", allow_candidate=True, **options)
    assert model.prefill_collective_window == expected and model.context is None
    assert not model._prepared and not model._attempted
