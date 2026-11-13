"""Board-free lifecycle tests; fake SDK work is not model/board qualification."""

import threading

import numpy as np
import pytest

from xinfer.config import ConfigError, ExplicitSettings, resolve
from xinfer.engine.core import AdmitError, Engine
from xinfer.engine.executor import Lane
from xinfer.engine.rpu import RpuEngine, RpuRequestPlan
from xinfer.models.interface import Capabilities, PreparedRequest, Stage, StageGraph
from xinfer.platforms import resolve_platform
from xinfer.platforms.rpu import RpuBusyError, RpuExecutionContext


class NativeResource:
    def close(self):
        self.closed = True


class NativeBuffer(NativeResource):
    def __init__(self, shape, **kwargs):
        self.shape = tuple(shape)
        self.dtype = kwargs.get("dtype", "float16")
        self.storage = kwargs.get("storage", "spm")
        self.layout = kwargs.get("layout", "row_major")
        self.distribution = kwargs.get("distribution", "local")
        self.state_kind = kwargs.get("state_kind", "none")
        self.immutable = kwargs.get("immutable", False)
        self.array = np.zeros((8, *shape) if self.storage == "spm" else shape, dtype=self.dtype)
        self.resets = 0

    def copy_from(self, host):
        self.array[...] = host

    def numpy(self):
        return self.array.copy()

    def reset(self):
        self.array.fill(0)
        self.resets += 1


class NativeExecution(NativeResource):
    def __init__(self, context, operations):
        self.context = context
        self.operations = operations
        self.error = None

    def execute(self, *, async_):
        assert async_ is True
        self.context.async_starts += 1

    def wait(self):
        self.context.wait_entered.set()
        assert self.context.wait_gate.wait(timeout=5)
        if self.context.fail_next:
            self.context.fail_next = False
            self.error = RuntimeError("injected device completion failure")
        if self.error:
            raise self.error
        for operation in self.operations:
            self.context.trace.append(operation.name)
            operation.tensors["out"].array.fill(operation.scalars.get("value", 0))


class NativeContext(NativeResource):
    def __init__(self, **_):
        self.trace = []
        self.async_starts = 0
        self.fail_next = False
        self.wait_entered = threading.Event()
        self.wait_gate = threading.Event()
        self.wait_gate.set()

    def empty(self, shape, **kwargs):
        return NativeBuffer(shape, **kwargs)

    def prepare_weight(self, host, *, layout):
        result = NativeBuffer(host.shape, storage="ddr", layout=layout,
                              distribution="shared", immutable=True)
        result.copy_from(host)
        return result

    def prepare(self, op, *, tensors, integers=None, scalars=None):
        result = NativeResource()
        result.name, result.tensors, result.scalars = op, tensors, scalars or {}
        def rebind(*, tensors=None, integers=None, scalars=None):
            result.tensors.update(tensors or {})
            result.scalars.update(scalars or {})
        result.rebind = rebind
        return result

    def batch(self, operations):
        return NativeExecution(self, operations)


@pytest.fixture
def context():
    with RpuExecutionContext(backend_factory=NativeContext) as result:
        yield result


class TwoStageModel:
    stage_graph = StageGraph(Stage("vision", "vision"), Stage("action", "action", "vision"))

    def __init__(self):
        self.result = np.zeros((1, 2), dtype=np.float32)
        self.aborted = 0
        self.resets = 0

    def capabilities(self):
        return Capabilities(batch_sizes={"vision": (1,), "action": (1, 2)}, num_steps=8)

    def shape_buckets(self):
        return {}

    def create_rpu_request(self, prepared, context):
        if not hasattr(self, "buffer"):
            self.buffer = context.empty((2,), state_kind="state")
        stages = {}
        for name in ("vision", "action"):
            operation = context.prepare(name, tensors={"out": self.buffer},
                                        scalars={"value": prepared.tensors["value"]})
            stages[name] = (context.batch((operation,)),)

        def collect():
            self.result[...] = self.buffer.numpy()[0]
            return {"actions": self.result}

        def abort(error):
            self.aborted += 1

        return RpuRequestPlan(stages, collect, abort=abort)

    def reset_rpu(self):
        self.resets += 1


def test_engine_invokes_model_rollback_when_plan_creation_fails(context):
    from types import SimpleNamespace

    def create(prepared, native):
        raise ValueError("bad admission")
    model = SimpleNamespace(
        stage_graph=StageGraph(Stage("vision", "vision")),
        create_rpu_request=create,
        rollback_rpu_request=lambda error: model.events.append(("rollback", str(error))),
        events=[],
    )
    engine = RpuEngine(model, resolve_platform("rpu"), context=context)
    with pytest.raises(ValueError, match="bad admission"):
        engine.run(PreparedRequest({}))
    assert model.events == [("rollback", "bad admission")]
    assert context.needs_reset
    engine.shutdown()


def test_wallx2b_vision_failure_rolls_back_with_real_engine(context):
    from collections import Counter
    from types import SimpleNamespace
    from xinfer.models.wallx2b.rpu.model import WallX2B

    model = WallX2B()
    model.context = context
    model._initial_resources, model._base_resources = frozenset(), frozenset(context._resources)
    model._signatures, model.weights = {}, {}
    model.prepare_rpu = lambda native: setattr(model, "_prepared", True)
    model._invalid = False
    model._base_resources = frozenset(context._resources)
    model.stats = Counter()
    state = SimpleNamespace(invalidate=lambda: None)
    model.prefix_state = SimpleNamespace(recurrent=state, history=state, history_out=state,
                                         key_caches={}, value_caches={})
    released, observations = [], []

    def rebuild():
        observations.append(model._invalid)
        with pytest.raises(RuntimeError, match="recovered"):
            model.begin_rpu_request(PreparedRequest({}), context)
        setattr(model, "vision", SimpleNamespace(bind=False))
        setattr(model, "action", SimpleNamespace())
        setattr(model, "prefix_state", SimpleNamespace())
        setattr(model, "_invalid", False)

    model._build_plans = rebuild
    model._prefix_plan = lambda prepared: SimpleNamespace(bind=lambda *args, **kwargs: None)
    model.action = SimpleNamespace(bind=lambda prepared: SimpleNamespace(
        close=lambda: released.append(True)), destination_caches=[])
    model.vision = SimpleNamespace(bind=lambda pixels: (_ for _ in ()).throw(ValueError("bad pixels")))
    engine = RpuEngine(model, resolve_platform("rpu"), context=context)
    prepared = PreparedRequest(dict(pixel_values=object(), position_ids=object(), input_ids=object()))
    with pytest.raises(ValueError, match="bad pixels"):
        engine.run(prepared)
    assert released == [True] and observations == [True]
    model.rollback_rpu_request(RuntimeError("duplicate rollback"))
    assert released == [True]
    assert context.needs_reset
    engine.shutdown()

def test_rpu_platform_does_not_use_torch_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "device", lambda *a, **k: pytest.fail("RPU is not a Torch device"))
    assert resolve_platform("rpu").device == "rpu:0"
    assert resolve_platform("rpu:0").is_rpu
    with pytest.raises(ValueError, match="rpu:0"):
        resolve_platform("rpu:1")


def test_native_context_requires_explicit_library_before_importing_xcompute(monkeypatch):
    import sys

    monkeypatch.delenv("XCOMPUTE_LIBRARY_PATH", raising=False)
    before = sys.modules.get("xcompute")
    with pytest.raises(RuntimeError, match="explicit XCOMPUTE_LIBRARY_PATH"):
        RpuExecutionContext()
    assert sys.modules.get("xcompute") is before


def test_rpu_resolution_limits_admission_before_building_a_device_context():
    platform, model = resolve_platform("rpu"), TwoStageModel()
    resolved = resolve(model, platform, model_name="test", cpus=32, tuned=None)
    assert resolved.value("max_running_requests") == 1
    assert resolved.value("max_queued_requests") == 1
    assert resolved.value("frontends") == 1
    assert all(resolved.value(f"lanes.{stage.name}") == 1 for stage in model.stage_graph)
    assert resolved.value("batch_sizes.action") == [1]
    with pytest.raises(ConfigError, match="requires lanes.action=1"):
        resolve(model, platform, model_name="test", explicit=ExplicitSettings(lanes={"action": 2}))
    with pytest.raises(ValueError, match="execution_model='cuda-lanes'"):
        Engine(model, platform)
    with pytest.raises(ValueError, match="cuda-lanes|explicit-completion"):
        Lane("vision", 0, platform)



def test_completion_owns_access_barrier_and_context_is_exclusive(context):
    with pytest.raises(RpuBusyError, match="one live"):
        RpuExecutionContext(backend_factory=NativeContext)
    output = context.empty((2,), distribution="replicated")
    operation = context.prepare("fill_f16", tensors={"out": output}, scalars={"value": 2})
    execution = context.batch((operation,))
    completion = execution.execute()
    assert not completion.done
    for access in (output.numpy, lambda: output.copy_from(np.zeros(2, np.float16)),
                   lambda: context.empty((2,)), execution.execute, output.close):
        with pytest.raises(RpuBusyError):
            access()
    completion.wait()
    assert completion.done
    np.testing.assert_array_equal(output.numpy(), np.full((8, 2), 2, dtype=np.float16))
    assert context.stats["logical_submissions"] == context.stats["completion_waits"] == 1


def test_completion_failure_is_sticky_and_recovery_invalidates_old_plans(context):
    state = context.empty((2,), state_kind="state")
    operation = context.prepare("fill_f16", tensors={"out": state})
    execution = context.batch((operation,))
    context.native.fail_next = True
    completion = execution.execute()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected device completion failure"):
            completion.wait()
    assert context.needs_reset
    context.recover()
    assert state.native.resets == 1
    assert not context.needs_reset
    with pytest.raises(RuntimeError, match="stale"):
        execution.execute()
    with pytest.raises(RuntimeError, match="stale"):
        context.batch((operation,))
    rebuilt = context.prepare("fill_f16", tensors={"out": state})
    context.batch((rebuilt,)).execute().wait()


def test_engine_resident_handoff_and_independent_output_storage(context):
    model = TwoStageModel()
    engine = RpuEngine(model, resolve_platform("rpu"), context=context)
    try:
        first = engine.run(PreparedRequest({"value": 1}))["actions"]
        second = engine.run(PreparedRequest({"value": 2}))["actions"]
        np.testing.assert_array_equal(first, np.ones((1, 2), np.float32))
        np.testing.assert_array_equal(second, np.full((1, 2), 2, np.float32))
        assert not np.shares_memory(first, model.result)
        assert context.native.trace == ["vision", "action", "vision", "action"]
        assert context.stats["allocations"] == 1
        assert context.stats["uploads"] == 0
        assert context.stats["downloads"] == 2
    finally:
        engine.shutdown()


def test_engine_blocks_second_request_and_reset_until_completion(context):
    model = TwoStageModel()
    engine = RpuEngine(model, resolve_platform("rpu"), context=context)
    context.native.wait_gate.clear()
    try:
        first = engine.submit(PreparedRequest({"value": 1}))
        assert context.native.wait_entered.wait(timeout=5)
        assert not first.done()
        assert not first.cancel()  # Device work drains; there is no native cancellation ABI.
        with pytest.raises(AdmitError, match="queue_full"):
            engine.submit(PreparedRequest({"value": 2})).result()
        with pytest.raises(RuntimeError, match="idle"):
            engine.reset()
        context.native.wait_gate.set()
        first.result(timeout=5)
    finally:
        context.native.wait_gate.set()
        engine.shutdown()
    with pytest.raises(AdmitError, match="stopped"):
        engine.submit(PreparedRequest({"value": 2})).result()


def test_engine_failure_requires_explicit_reset_then_accepts_valid_request(context):
    model = TwoStageModel()
    engine = RpuEngine(model, resolve_platform("rpu"), context=context)
    try:
        context.native.fail_next = True
        with pytest.raises(RuntimeError, match="injected"):
            engine.run(PreparedRequest({"value": 1}))
        assert model.aborted == 1
        with pytest.raises(RuntimeError, match="explicit reset"):
            engine.run(PreparedRequest({"value": 2}))
        engine.reset()
        result = engine.run(PreparedRequest({"value": 3}))
        assert model.resets == 1
        np.testing.assert_array_equal(result["actions"], np.full((1, 2), 3, np.float32))
    finally:
        engine.shutdown()


def test_bound_invocations_reuse_one_prepared_batch_and_release_independently(context):
    output = context.empty((16,), distribution="replicated")
    operation = context.prepare("fill_f16", tensors={"out": output}, scalars={"value": 0})
    execution = context.batch((operation,))
    first = execution.bind({operation: {"scalars": {"value": 1}}})
    second = execution.bind({operation: {"scalars": {"value": 2}}})
    for bound, expected in ((first, 1), (second, 2), (first, 1)):
        bound.execute().wait()
        np.testing.assert_array_equal(output.numpy(), np.full((8, 16), expected, np.float16))
    saved = output.numpy()
    first.close()
    assert not getattr(execution.native, "closed", False)
    second.execute().wait()
    np.testing.assert_array_equal(saved, np.full((8, 16), 1, np.float16))
    assert context.stats["batch_preparations"] == 1
    assert context.stats["operation_preparations"] == 1
    assert len(operation.tensors) == 1
    second.close()


def test_rejected_bound_patch_submits_no_work_and_requires_recovery(context):
    output = context.empty((16,), distribution="replicated")
    operation = context.prepare("fill_f16", tensors={"out": output})
    execution = context.batch((operation,))
    bound = execution.bind({operation: {"scalars": {"unknown": 1}}})
    def reject(**kwargs):
        raise ValueError("invalid mutable binding")
    operation.native.rebind = reject
    with pytest.raises(ValueError, match="invalid mutable"):
        bound.execute()
    assert context.needs_reset and context.native.async_starts == 0
    context.recover()
    with pytest.raises(RuntimeError, match="stale"):
        bound.execute()


def test_policy_shutdown_failure_still_closes_processor_and_preserves_primary_error():
    from types import SimpleNamespace
    from xinfer.entrypoints.policy import Policy

    closed = []
    def shutdown():
        raise RuntimeError("device completion failed")
    def close_processor():
        closed.append(True)
        raise ValueError("processor cleanup failed")
    policy = Policy.__new__(Policy)
    policy.engine = SimpleNamespace(shutdown=shutdown)
    policy._prepare = SimpleNamespace(close=close_processor)
    with pytest.raises(RuntimeError, match="device completion failed"):
        policy.close()
    assert closed == [True]
    with pytest.raises(KeyError, match="request failed"):
        with policy:
            raise KeyError("request failed")
    assert closed == [True, True]


def test_failed_release_retains_exclusive_ownership_until_close_is_retried(context):
    buffer = context.empty((2,))
    original_close = buffer.native.close
    attempts = []

    def close_once_fails():
        attempts.append(None)
        if len(attempts) == 1:
            raise RuntimeError("injected release failure")
        original_close()

    buffer.native.close = close_once_fails
    with pytest.raises(RuntimeError, match="injected release failure"):
        context.close()
    assert not context.closed
    with pytest.raises(RpuBusyError, match="one live"):
        RpuExecutionContext(backend_factory=NativeContext)
    context.close()
    assert context.closed
    with RpuExecutionContext(backend_factory=NativeContext):
        pass


def test_shutdown_preserves_model_error_when_context_cleanup_also_fails(context):
    model = TwoStageModel()
    engine = RpuEngine(model, resolve_platform("rpu"), context=context)
    engine._owns_context = True

    def fail_model_close():
        raise RuntimeError("primary model close failure")

    def fail_context_close():
        raise RuntimeError("secondary context close failure")

    model.close_rpu = fail_model_close
    original_close = context.close
    context.close = fail_context_close
    try:
        with pytest.raises(RuntimeError, match="primary model close failure"):
            engine.shutdown()
        assert not engine._closed
    finally:
        context.close = original_close
        del model.close_rpu
        engine.shutdown()


def test_released_native_wait_error_finishes_adapter_cleanup(context):
    buffer = context.empty((2,))
    buffer.native._id = 17

    def release_then_fail():
        buffer.native._id = 0
        raise RuntimeError("completion failed after native release")

    buffer.native.close = release_then_fail
    with pytest.raises(RuntimeError, match="completion failed after native release"):
        context.close()
    assert buffer.closed and context.closed and not context._resources
    with RpuExecutionContext(backend_factory=NativeContext):
        pass


def test_reset_rebuild_failure_keeps_engine_blocked_until_successful_reset(context):
    model = TwoStageModel()
    engine = RpuEngine(model, resolve_platform("rpu"), context=context)
    original_reset = model.reset_rpu

    def fail_rebuild():
        raise MemoryError("plan rebuild failed")

    try:
        model.reset_rpu = fail_rebuild
        with pytest.raises(MemoryError, match="plan rebuild failed"):
            engine.reset()
        assert context.needs_reset
        with pytest.raises(RuntimeError, match="explicit reset"):
            engine.run(PreparedRequest({"value": 2}))
        assert context.native.async_starts == 0
        model.reset_rpu = original_reset
        engine.reset()
        assert not context.needs_reset
        np.testing.assert_array_equal(engine.run(PreparedRequest({"value": 3}))["actions"],
                                      np.full((1, 2), 3, np.float32))
    finally:
        engine.shutdown()


def test_descriptor_budget_is_process_local_and_applied_before_backend(monkeypatch):
    import resource

    limits = [1024, 1048576]
    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda kind: tuple(limits))

    def set_limit(kind, value):
        assert kind == resource.RLIMIT_NOFILE
        calls.append(tuple(value))
        limits[:] = value

    def backend(**options):
        assert limits == [4096, 1048576]
        return NativeContext(**options)

    monkeypatch.setattr(resource, "setrlimit", set_limit)
    with RpuExecutionContext(file_descriptor_budget=4096, backend_factory=backend) as context:
        assert calls == [(4096, 1048576)]
        assert context.describe()["provenance"]["file_descriptors"] == dict(
            required=4096, soft_before=1024, soft_applied=4096, hard=1048576)
        context.require_file_descriptor_budget(2048)
        assert calls == [(4096, 1048576)]
        assert context.describe()["provenance"]["file_descriptors"] == dict(
            required=4096, soft_before=1024, soft_applied=4096, hard=1048576)


def test_insufficient_descriptor_budget_fails_before_backend(monkeypatch):
    import resource

    monkeypatch.setattr(resource, "getrlimit", lambda kind: (1024, 2048))
    monkeypatch.setattr(resource, "setrlimit", lambda *args: pytest.fail("must not raise hard limit"))
    with pytest.raises(RuntimeError, match="process hard limit is 2048"):
        RpuExecutionContext(file_descriptor_budget=4096,
                            backend_factory=lambda **kwargs: pytest.fail("SDK must not initialize"))
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            RpuExecutionContext(file_descriptor_budget=invalid,
                                backend_factory=lambda **kwargs: pytest.fail("SDK must not initialize"))


def test_explicit_AR_windows_require_candidate_and_backend_capability_without_default_fallback():
    class CalibratedBackend(NativeContext):
        maximum = 15
        def __init__(self, **kw):
            super().__init__(**kw)
            self.windows = []
            self.queries = 0
        def ar_collective_window_max(self):
            self.queries += 1
            return self.maximum
        def batch(self, operations, *, collective_window=1):
            self.windows.append(collective_window)
            return super().batch(operations)
    with RpuExecutionContext(allow_candidate=True, backend_factory=CalibratedBackend) as context:
        out = context.empty((1,1024), distribution="replicated")
        source = context.empty((1,1024), distribution="partial")
        operation = context.prepare("all_reduce_residual", tensors={"out":out,"input":source})
        context.batch((operation,))
        assert context.native.windows == [1] and context.native.queries == 0
        context.batch((operation,), collective_window=4)
        assert context.native.windows == [1,4] and context.native.queries == 1
        context.batch((operation,), collective_window=15)
        assert context.native.windows == [1,4,15] and context.native.queries == 2
        context.native.maximum = 3
        before = dict(context.stats)
        with pytest.raises(RuntimeError, match="backend lacks"):
            context.batch((operation,), collective_window=4)
        assert context.native.windows == [1,4,15] and dict(context.stats) == before
        for invalid in (0, 16, True):
            with pytest.raises(ValueError, match="integer in"):
                context.batch((operation,), collective_window=invalid)
    with RpuExecutionContext(backend_factory=CalibratedBackend) as context:
        with pytest.raises(RuntimeError, match="candidate admission"):
            context.require_collective_window(4)
        assert context.native.queries == 0
    with RpuExecutionContext(allow_candidate=True, backend_factory=NativeContext) as context:
        with pytest.raises(RuntimeError, match="backend lacks"):
            context.require_collective_window(4)


def test_M16_window_cap_is_independent_and_mixed_families_fail_before_batch():
    class Backend(NativeContext):
        ar_maximum, m16_maximum = 8, 4
        def __init__(self, **kw):
            super().__init__(**kw)
            self.windows, self.queries = [], []
        def ar_collective_window_max(self):
            self.queries.append("AR")
            return self.ar_maximum
        def action_m16_collective_window_max(self):
            self.queries.append("M16")
            return self.m16_maximum
        def batch(self, operations, *, collective_window=1):
            self.windows.append(collective_window)
            return super().batch(operations)
    with RpuExecutionContext(allow_candidate=True, backend_factory=Backend) as context:
        tensors = {}
        for name, shape, role, axis in (
            ("src", (16,256), "sharded", 1), ("partial", (16,1024), "partial", -1),
            ("residual", (16,1024), "replicated", -1), ("stream", (16,1024), "replicated", -1),
            ("norm", (16,1024), "replicated", -1)):
            tensors[name] = context.empty(shape, distribution=role, shard_axis=axis)
        tensors["weight"] = context.prepare_weight(np.zeros((8,1024,256), np.float16), layout="tp8_nk16")
        for name in ("gate_row", "next_row"):
            tensors[name] = context.empty((3,1024), storage="ddr", distribution="shared")
        m16 = context.prepare("down_allreduce_rmsnorm_m16n1024k256_acc32", tensors=tensors)
        ar = context.prepare("all_reduce_residual", tensors={
            "input": context.empty((1,1024), distribution="partial"),
            "out": context.empty((1,1024), distribution="replicated")})
        context.batch((m16,), collective_window=4)
        assert context.native.windows == [4] and context.native.queries == ["M16"]
        before = dict(context.stats)
        for operations in ((ar,m16), (m16,ar)):
            with pytest.raises(RuntimeError, match="cannot mix"):
                context.batch(operations, collective_window=4)
        context.native.m16_maximum = 3
        with pytest.raises(RuntimeError, match="backend lacks"):
            context.batch((m16,), collective_window=4)
        with pytest.raises(RuntimeError, match="backend lacks"):
            context.batch((m16,), collective_window=8)
        assert context.native.windows == [4] and dict(context.stats) == before
        context.native.queries.clear()
        context.batch((ar,m16))
        assert context.native.windows == [4,1] and context.native.queries == []
