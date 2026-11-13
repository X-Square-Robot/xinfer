"""Static-step ownership checks; fake backend is not board evidence."""

import numpy as np
import pytest

from xinfer.platforms.rpu import RpuBoundExecution
from test_wallx2b_denoise import _inputs, model as _model_fixture


@pytest.fixture
def model():
    yield from _model_fixture.__wrapped__()


def test_each_static_step_retains_its_own_modulation_and_queue(model):
    binding = model.bind(_inputs())
    plan = binding.plan
    try:
        assert len(plan.steps) == 8 and len({id(step) for step in plan.steps}) == 8
        assert len({id(step.native) for step in plan.steps}) == 8
        for index, step in enumerate(plan.steps):
            assert not isinstance(step, RpuBoundExecution)
            assert len(step.operations) == 100
            assert step.operations[3].tensor_bindings["input"] is model._rows[index][0]
            assert step.operations[4].tensor_bindings["input"] is model._rows[index][1]
            assert float(step.operations[3].tensor_bindings["input"].native.array[1, 0]) == index / 16
        assert model.context.native.trace == []
        assert model.context.native.rebindings == 0
    finally:
        binding.close()
    assert not any(step.closed for step in plan.steps)
    assert all(not op.closed for op in plan.operations)


@pytest.mark.parametrize("fail_at", [5, 9])
def test_partial_static_build_failure_closes_earlier_steps_and_views(model, monkeypatch, fail_at):
    before = set(model.context._resources)
    before_spm = model.workspace.arena.native.array.copy()
    original = model.context.batch
    calls = 0

    def fail_batch(operations):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("injected static step build failure")
        return original(operations)

    monkeypatch.setattr(model.context, "batch", fail_batch)
    with pytest.raises(RuntimeError, match="injected static step build"):
        model.bind(_inputs())
    assert model._active is None and not model._plans
    assert set(model.context._resources) == before
    assert model.context.native.trace == []
    np.testing.assert_array_equal(model.workspace.arena.native.array, before_spm)


def test_failed_entry_build_closes_already_prepared_fused_steps(model, monkeypatch):
    before = set(model.context._resources)
    model.fused_steps = True
    # The CPU backend only models ownership; use its ordinary queue factory
    # for the fused queue and reject the subsequent plain entry construction.
    monkeypatch.setattr(model.context, "batch_fused", model.context.batch)

    def reject_entry(operations):
        raise RuntimeError("injected entry build failure")

    monkeypatch.setattr(model.context, "batch", reject_entry)
    with pytest.raises(RuntimeError, match="injected entry build"):
        model.bind(_inputs())
    assert model._active is None and not model._plans
    assert set(model.context._resources) == before
    assert model.context.native.trace == []


def test_two_buckets_own_sixteen_batches_and_evict_all_idle_steps(model):
    def bind(prefix):
        binding = model.bind(_inputs(prefix=prefix))
        plan = binding.plan
        binding.close()
        return plan

    first = bind(319)
    second = bind(383)
    assert model.describe()["retained_step_batches"] == 16
    assert model.context.stats["batch_preparations"] == 18
    assert sum(len(plan.operations) for plan in model._plans.values()) == 1600
    third = bind(447)
    assert all(step.closed for step in first.steps)
    assert all(op.closed for op in first.operations)
    assert not any(step.closed for step in (*second.steps, *third.steps))
