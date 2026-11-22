"""CPU device-binding interpreter; these checks are not RPU numerical evidence.

Virtual sparse weights make all attention/MLP branches zero and the input and
output projections nonzero. An independent scalar reference checks identity
residuals, eight Euler steps, dynamic bindings and output ownership. Dense
attention, GEMM and native fusion numerics require the board suites.
"""

from collections import Counter

import numpy as np
import pytest

from xinfer.models.wall_oss_1_0_nano.rpu.action import ActionConditioning, flow_schedule
from xinfer.models.wall_oss_1_0_nano.rpu.config import FULL_ATTENTION_LAYERS
from xinfer.models.wall_oss_1_0_nano.rpu.denoise import (
    DenoiseWorkspace,
    PreparedDenoise,
    action_rope_tables,
    pack_action_request,
)
from xinfer.platforms.rpu import RpuBuffer, RpuExecutionContext

from wall_oss_1_0_nano_plan_helpers import catalog_call


def _half(value):
    return np.asarray(value, dtype=np.float16)


class _Resource:
    def close(self):
        self.closed = True


class _Tensor(_Resource):
    def __init__(self, shape, *, array=None, **description):
        self.shape = tuple(shape)
        self.dtype = description.get("dtype", "float16")
        self.storage = description.get("storage", "spm")
        self.layout = description.get("layout", "row_major")
        self.distribution = description.get("distribution", "local")
        self.shard_axis = description.get("shard_axis", -1)
        self.state_kind = description.get("state_kind", "none")
        self.immutable = description.get("immutable", False)
        self.spec = description.get("spec")
        self.extent = 512 if self.state_kind == "cache" else 0
        if self.state_kind == "cache" or self.layout in ("nk16", "tp8_nk16"):
            self.array = None
        elif array is not None:
            self.array = array
        else:
            self.array = np.empty((8, *self.shape) if self.storage == "spm" else self.shape, dtype=self.dtype)
            if self.storage == "spm" and self.distribution == "single":
                self.array[1:] = 0

    def copy_from(self, host):
        if self.storage == "spm" and self.distribution == "single":
            self.array[0] = host
            self.array[1:] = 0
        else:
            self.array[...] = host

    def numpy(self):
        return self.array.copy()

    def reset(self):
        self.extent = 0


def _norm(x, modulation):
    """Staged source FP16 arithmetic for the uniform sparse fixture."""
    mean = _half(np.sum(x.astype(np.float32) ** 2, axis=-1, keepdims=True) * np.float32(4))
    inverse = _half(1 / np.sqrt(_half(mean + np.float16(1e-6 / 16)).astype(np.float32)))
    value = _half(_half(x * inverse) * np.float16(64))
    return _half(_half(value * modulation[:, None, 0]) + modulation[:, None, 1])


class _Execution(_Resource):
    def __init__(self, backend, operations):
        self.backend = backend
        self.operations = tuple(operations)

    def execute(self, *, async_):
        assert async_

    def wait(self):
        for operation in self.operations:
            name, t = operation.name, operation.tensors
            a = {key: owner.array for key, owner in t.items()}
            self.backend.trace.append(name)
            if self.backend.fail_next == name:
                self.backend.fail_next = None
                raise RuntimeError("injected device-step failure")
            if name == "copy_f16":
                mode = operation.integers["mode"]
                if mode == 2:
                    if t["input"].storage == "spm":
                        a["out"][...] = a["input"][0]
                    else:
                        a["out"][0] = a["input"]
                else:
                    assert mode == 1
                    a["out"][...] = a["input"]
            elif name == "copy_physical_prefix_f16":
                prefix, bucket = operation.integers["prefix_len"], operation.integers["bucket_len"]
                for index in range(6):
                    for kind in ("k", "v"):
                        assert t[f"src_{kind}{index}"].extent >= prefix
                        target = t[f"dst_{kind}{index}"]
                        target.extent = bucket
                        target.real_prefix = prefix
            elif name == "gemm_ddr_bias_scale_m16n1024k64_acc32":
                value = _half(a["input"][0, :, :1].astype(np.float32) / np.float32(64))
                a["out"][0] = _half(value / np.float16(4))
            elif name == "adaptive_rmsnorm_m16h1024_f16":
                assert operation.scalars == {"epsilon": 1e-6 / 16, "normalization_scale": 64.0}
                a["out"][...] = _norm(a["x"].copy(), a["modulation"])
            elif name == "residual_gate_m16h1024_f16":
                a["out"][...] = _half(_half(a["branch"] * a["gate"][:, None]) + a["residual"])
            elif name == "qknorm_qkv_rope_m16k1024_acc32":
                assert operation.integers == {"position_offset": 0}
                a["planes"].fill(0)
                self.backend.weight_trace.append(t["weight"].spec)
            elif name in ("kcache_insert_d256_f16", "vcache_insert_d256_f16"):
                assert t["cache"].extent >= operation.integers["position"]
                t["cache"].extent = operation.integers["position"] + 16
            elif name == "sdpa_masked_m16d256_f16":
                cache = t["key_cache"]
                length = operation.integers["kv_length"]
                assert cache.extent >= length and t["value_cache"].extent >= length
                assert np.isneginf(a["mask"][:, :, cache.real_prefix : length - 16]).all()
                assert (a["mask"][:, :, : cache.real_prefix] == 0).all()
                assert (a["mask"][:, :, length - 16 :] == 0).all()
                a["out"].fill(0)
            elif name == "sigmoid_mul_m16d256_f16":
                a["out"].fill(0)
            elif name == "gate_up_silu_m16n256k1024_acc32":
                a["out"].fill(0)
                self.backend.weight_trace.extend((t["gate_weight"].spec, t["up_weight"].spec))
            elif name == "down_allreduce_rmsnorm_m16n1024k256_acc32":
                self.backend.weight_trace.append(t["weight"].spec)
                # Sparse fixture down/O weights are zero. The fused epilogue
                # still carries every identity residual and both Ada sites.
                a["partial"].fill(0)
                stream = a["residual"].copy()
                a["stream"][...] = stream
                row = np.broadcast_to(a["next_row"], (8, 3, 1024))
                normalized = _norm(stream, row)
                if "next2_row" in a:
                    mixed = _half(_half(normalized * row[:, None, 2]) + stream)
                    a["norm"][...] = mixed
                    row2 = np.broadcast_to(a["next2_row"], (8, 3, 1024))
                    a["w_out"][...] = _norm(mixed, row2)
                else:
                    a["norm"][...] = normalized
            elif name == "gemm_ddr_bias_m16n64k1024_acc32":
                a["out"][0].fill(0)
                a["out"][0, :, :26] = _half(a["input"][0, :, :26] * np.float16(1 / 8))
            elif name == "euler_update_m16d64_f16":
                masked = _half(a["velocity"] * a["keep_mask"][:, None])
                mixed = _half(masked + a["padding_velocity"])
                increment = _half(mixed * np.float16(operation.scalars["delta_t"]))
                a["action"][...] = _half(a["action"] + increment)
                a["velocity"][...] = increment
            else:
                raise AssertionError(name)


class _Operation(_Resource):
    def __init__(self, backend, name, tensors, integers, scalars, names=None):
        self.backend = backend
        self.name, self.tensors = name, tensors
        self.integers, self.scalars = integers, scalars
        # The native operation reports its resolved catalog entry; rebind
        # accepts the argument names prepare used (family or catalog).
        self.entry = name
        self._names = names or {key: key for key in tensors}

    def rebind(self, *, tensors=None, integers=None, scalars=None):
        tensors = {self._names[name]: value for name, value in (tensors or {}).items()}
        for name, value in tensors.items():
            previous = self.tensors[name]
            assert (value.shape, value.dtype, value.layout, value.distribution) == (
                previous.shape,
                previous.dtype,
                previous.layout,
                previous.distribution,
            )
            self.tensors[name] = value
        self.integers.update(integers or {})
        self.scalars.update(scalars or {})
        self.backend.rebindings += 1


class _Backend(_Resource):
    def __init__(self, **_):
        self.trace, self.weight_trace = [], []
        self.fail_next = None
        self.rebindings = 0

    def empty(self, shape, **description):
        return _Tensor(shape, **description)

    def view(self, parent, shape, *, offset_bytes=0, distribution=None, shard_axis=-1):
        description = dict(
            dtype=parent.dtype,
            storage=parent.storage,
            layout=parent.layout,
            distribution=parent.distribution if distribution is None else distribution,
            shard_axis=shard_axis,
            immutable=parent.immutable,
        )
        count, offset = int(np.prod(shape)), offset_bytes // 2
        if parent.storage == "spm":
            value = parent.array.reshape(8, -1)[:, offset : offset + count].reshape(8, *shape)
        else:
            value = parent.array.reshape(-1)[offset : offset + count].reshape(shape)
        return _Tensor(shape, array=value, **description)

    def prepare_weight(self, host, *, layout):
        result = _Tensor(host.shape, storage="ddr", layout=layout, distribution="shared", immutable=True)
        result.copy_from(host)
        return result

    def prepare(self, op, *, tensors, integers=None, scalars=None, **selectors):
        entry, catalog, resolved, names = catalog_call(op, tensors, integers, selectors,
                                                       with_names=True)
        return _Operation(self, entry, catalog, resolved, scalars or {}, names)

    def batch(self, operations):
        return _Execution(self, operations)


def _weights(context):
    shapes = {"action.input": ((1024, 64), "nk16"), "action.output": ((64, 1024), "nk16")}
    for layer in range(24):
        for name, shape in (("gate", (8, 256, 1024)), ("up", (8, 256, 1024)), ("down", (8, 1024, 256))):
            shapes[f"action.{layer}.{name}"] = (shape, "tp8_nk16")
        if layer in FULL_ATTENTION_LAYERS:
            shapes[f"action.{layer}.qkv"] = ((4, 8, 256, 1024), "tp8_nk16")
            shapes[f"action.{layer}.o"] = ((8, 1024, 256), "tp8_nk16")
            shapes[f"action.{layer}.qk_norm"] = ((2, 256), "row_major")
    return {
        name: RpuBuffer(
            context, _Tensor(shape, storage="ddr", layout=layout, distribution="shared", immutable=True, spec=name)
        )
        for name, (shape, layout) in shapes.items()
    }


def _inputs(prefix=321, noise=0.25, rotary=100):
    return dict(
        prefix_length=prefix,
        prefix_bucket=((prefix + 63) // 64) * 64,
        num_steps=8,
        noise_fp16=np.full((1, 16, 26), noise, np.float16),
        padding_velocity_fp16=np.full((1, 16, 26), -0.125, np.float16),
        dof_mask=np.array([1] * 20 + [0] * 6, np.float32).reshape(1, 1, 26),
        action_position_ids=np.broadcast_to(np.arange(rotary, rotary + 16, dtype=np.int64), (3, 1, 16)).copy(),
    )


@pytest.fixture
def model():
    with RpuExecutionContext(backend_factory=_Backend) as context:
        weights = _weights(context)
        sources = tuple(
            (
                context.empty(
                    (8, 784, 256), storage="ddr", layout="causal_k_cache", distribution="shared", state_kind="cache"
                ),
                context.empty(
                    (8, 784, 256), storage="ddr", layout="causal_v_cache", distribution="shared", state_kind="cache"
                ),
            )
            for _ in range(6)
        )
        modulation = np.zeros((8, 49, 3, 1024), np.float16)
        modulation[:, :, 0] = 1
        modulation[:, ::2, 2] = np.float16(1 / 256)
        modulation[:, 1::2, 2] = np.float16(4 / 256)
        for step in range(8):
            modulation[step, 0, 1] = np.float16(step / 16)
            modulation[step, 48, 1] = np.float16(step / 4)
        times, dt = flow_schedule()
        prepared = PreparedDenoise(
            context, weights, ActionConditioning(modulation, times.numpy(), dt), sources, max_cached_buckets=2
        )
        try:
            yield prepared
        finally:
            prepared.close()


def _run(model, inputs):
    binding = model.bind(inputs)
    try:
        for execution in binding.executions:
            execution.execute().wait()
        return binding.collect_normalized()
    finally:
        binding.close()


def _scalar_reference(inputs):
    action, keep, padding = pack_action_request(inputs)
    _, dt = flow_schedule()
    for step in range(8):
        # All hidden channels are equal for the fixture's input projection.
        value = _half(_half(action[:, 0].astype(np.float32) / np.float32(64)) / np.float16(4))
        for identity in range(18):
            mean = _half(value.astype(np.float32) ** 2 * np.float32(4096))
            inverse = _half(1 / np.sqrt(_half(mean + np.float16(1e-6 / 16)).astype(np.float32)))
            normalized = _half(_half(value * inverse) * np.float16(64))
            if identity == 0:
                normalized = _half(normalized + np.float16(step / 16))
            value = _half(value + _half(normalized * np.float16(1 / 256)))
        mean = _half(value.astype(np.float32) ** 2 * np.float32(4096))
        inverse = _half(1 / np.sqrt(_half(mean + np.float16(1e-6 / 16)).astype(np.float32)))
        normalized = _half(_half(value * inverse) * np.float16(64))
        normalized = _half(normalized + np.float16(step / 4))
        velocity = np.zeros((16, 64), np.float16)
        velocity[:, :26] = _half(normalized[:, None] * np.float16(1 / 8))
        action = _half(action + _half(_half(_half(velocity * keep) + padding) * dt))
    return action[None, :, :26].astype(np.float32)


def test_complete_sparse_decoder_and_eight_fp16_euler_steps(model):
    inputs = _inputs()
    model.workspace.arena.native.array.fill(7)  # prior phase owns the same arena
    actual = _run(model, inputs)
    np.testing.assert_array_equal(actual, _scalar_reference(inputs))
    counts = Counter(model.context.native.trace)
    assert counts["gate_up_silu_m16n256k1024_acc32"] == 24 * 8
    assert counts["qknorm_qkv_rope_m16k1024_acc32"] == 6 * 8
    assert counts["down_allreduce_rmsnorm_m16n1024k256_acc32"] == 30 * 8
    assert counts["euler_update_m16d64_f16"] == 8
    assert model.describe()["step_nodes"]["384"] == [100] * 8
    assert model.describe()["retained_step_nodes"]["384"] == 800
    assert model.describe()["retained_step_batches"] == 8
    assert model.context.stats["operation_preparations"] == 805
    assert model.context.native.rebindings == 1
    assert model.describe()["workspace_bytes_per_core"] < 320 * 1024


def test_a_b_a_updates_same_bucket_masks_rope_noise_and_keeps_old_output(model):
    a = _inputs(prefix=321, noise=0.25, rotary=100)
    b = _inputs(prefix=383, noise=0.5, rotary=127)
    first = _run(model, a)
    saved = first.copy()
    first_cosine = model.cosine.native.array.copy()
    other = _run(model, b)
    assert not np.array_equal(first_cosine, model.cosine.native.array)
    np.testing.assert_array_equal(other, _scalar_reference(b))
    again = _run(model, a)
    np.testing.assert_array_equal(first, saved)
    np.testing.assert_array_equal(again, saved)
    assert model.describe()["stats"]["bucket_builds"] == 1
    assert model.describe()["stats"]["bucket_replays"] == 2
    assert model.context.stats["operation_preparations"] == 805
    assert model.context.stats["batch_preparations"] == 9
    assert model.describe()["entry_rebuilt_per_request"] is False


def test_cross_bucket_plans_replay_without_rebuilding_weights(model):
    _run(model, _inputs(prefix=511))
    _run(model, _inputs(prefix=319))
    _run(model, _inputs(prefix=511))
    assert model.describe()["cached_buckets"] == [320, 512]
    assert model.describe()["stats"]["bucket_builds"] == 2
    assert model.describe()["stats"]["bucket_replays"] == 1
    assert model.context.stats["weight_preparations"] == 1  # the cached Ada table only


def test_three_bucket_lru_evicts_only_idle_plans_and_rebuilds_closed_signature(model):
    def bind_once(prefix):
        binding = model.bind(_inputs(prefix=prefix))
        plan = binding.plan
        binding.close()
        return plan

    a = bind_once(319)  # bucket320
    b = bind_once(383)  # bucket384
    assert bind_once(319).execution is a.execution
    c = bind_once(447)  # evicts B, because A was just used
    assert all(step.closed for step in b.steps) and all(op.closed for op in b.operations)
    assert not any(step.closed for step in (*a.steps, *c.steps))
    assert model.describe()["cached_buckets"] == [320, 448]
    rebuilt = bind_once(383)
    assert rebuilt.execution is not b.execution and all(step.closed for step in a.steps)
    assert model.describe()["cached_buckets"] == [384, 448]
    stats = model.describe()["stats"]
    assert stats["bucket_builds"] == 4 and stats["bucket_evictions"] == 2 and stats["bucket_replays"] == 1
    assert model.context.stats["weight_preparations"] == 1


def test_failed_new_bucket_keeps_existing_lru_plans_until_replacement_is_prepared(model):
    def bind_once(prefix):
        binding = model.bind(_inputs(prefix=prefix))
        plan = binding.plan
        binding.close()
        return plan

    a = bind_once(319)
    b = bind_once(383)
    original = model._prepare_step

    def fail_third_bucket(step, mask, bucket):
        if bucket == 448:
            raise RuntimeError("injected new bucket lowering failure")
        return original(step, mask, bucket)

    model._prepare_step = fail_third_bucket
    with pytest.raises(RuntimeError, match="injected new bucket lowering failure"):
        model.bind(_inputs(prefix=447))
    assert model._plans == {320: a, 384: b}
    assert not any(step.closed for step in (*a.steps, *b.steps))
    assert model.stats["bucket_evictions"] == 0
    model._prepare_step = original
    c = bind_once(447)
    assert model._plans == {384: b, 448: c}
    assert a.execution.closed and not b.execution.closed


    before = model.workspace.arena.native.array.copy()
    binding = model.bind(_inputs())
    try:
        assert model.context.native.trace == []
        np.testing.assert_array_equal(model.workspace.arena.native.array, before)
        with pytest.raises(RuntimeError, match="previous Action"):
            model.bind(_inputs())
    finally:
        binding.close()


def test_device_failure_blocks_new_request_until_lowering_is_rebuilt(model):
    model.context.native.fail_next = "euler_update_m16d64_f16"
    with pytest.raises(RuntimeError, match="injected"):
        _run(model, _inputs())
    with pytest.raises(RuntimeError, match="stale"):
        model.bind(_inputs())
    model.context.recover()
    with pytest.raises(RuntimeError, match="stale"):
        model.bind(_inputs())


def test_binding_close_drains_pending_failure_and_releases_request_owners(model):
    binding = model.bind(_inputs())
    binding.executions[0].execute().wait()
    model.context.native.fail_next = "euler_update_m16d64_f16"
    binding.executions[1].execute()
    with pytest.raises(RuntimeError, match="injected device-step"):
        binding.close()
    assert binding.closed and binding.entry.closed and not binding.prefix_operation.closed
    assert model._active is None and model.context.needs_reset


def test_wait_error_survives_a_second_cleanup_failure_and_close_can_retry(model):
    binding = model.bind(_inputs())
    binding.executions[0].execute().wait()
    model.context.native.fail_next = "euler_update_m16d64_f16"
    binding.executions[1].execute()
    original_close = binding.entry.close

    def reject_close():
        raise RuntimeError("injected cleanup rejection")

    binding.entry.close = reject_close
    try:
        with pytest.raises(RuntimeError, match="injected device-step"):
            binding.close()
        assert not binding.entry.closed and not binding.closed
    finally:
        binding.entry.close = original_close
        binding.close()
    assert binding.closed and model._active is None


def test_component_close_releases_all_owned_resources_after_wait_failure(model):
    borrowed = {
        *map(id, model.weights.buffers.values()),
        *(id(owner) for pair in model.source_caches for owner in pair),
    }
    binding = model.bind(_inputs())
    binding.executions[0].execute().wait()
    model.context.native.fail_next = "euler_update_m16d64_f16"
    binding.executions[1].execute()
    with pytest.raises(RuntimeError, match="injected device-step"):
        model.close()
    assert model.closed and model.workspace.closed and binding.closed
    assert {id(owner) for owner in model.context._resources} == borrowed


def test_rope_bakes_logical_positions_without_a_second_gap_or_bucket_offset():
    positions = np.broadcast_to(np.arange(113, 129, dtype=np.int64), (3, 1, 16)).copy()
    cosine, sine = action_rope_tables(positions)
    frequency = 1 / (10_000_000.0 ** (np.arange(0, 64, 2, dtype=np.float64) / 64))
    phase = np.arange(113, 129, dtype=np.float64)[:, None] * frequency
    np.testing.assert_array_equal(cosine, np.cos(phase).astype(np.float16))
    np.testing.assert_array_equal(sine, np.sin(phase).astype(np.float16))
    with pytest.raises(ValueError, match="identical"):
        bad = positions.copy()
        bad[1] += 1
        action_rope_tables(bad)


def test_action_packing_restores_mask_tail_and_masks_already_rounded_padding():
    inputs = _inputs()
    packed, keep, padding = pack_action_request(inputs)
    np.testing.assert_array_equal(packed[:, :26], inputs["noise_fp16"][0])
    np.testing.assert_array_equal(packed[:, 26:52], np.broadcast_to(inputs["dof_mask"][0], (16, 26)))
    assert not packed[:, 52:].any() and not keep[26:].any()
    assert not padding[:, :20].any() and not padding[:, 26:].any()
    assert DenoiseWorkspace.single_bytes < 41 * 1024


def test_explicit_M16_window4_keeps_static_step_owners_and_native_route():
    class WindowBackend(_Backend):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.windows = []
        def action_m16_collective_window_max(self):
            return 4
        def batch(self, operations, *, collective_window=1):
            self.windows.append(collective_window)
            if collective_window == 4:
                assert sum(op.name == "down_allreduce_rmsnorm_m16n1024k256_acc32"
                           for op in operations) == 30
            return super().batch(operations)
    with RpuExecutionContext(allow_candidate=True, backend_factory=WindowBackend) as context:
        weights = _weights(context)
        sources = tuple((context.empty((8,784,256), storage="ddr", layout="causal_k_cache",
                                      distribution="shared", state_kind="cache"),
                         context.empty((8,784,256), storage="ddr", layout="causal_v_cache",
                                      distribution="shared", state_kind="cache")) for _ in range(6))
        modulation = np.zeros((8,49,3,1024), np.float16)
        modulation[:,:,0] = 1
        times, dt = flow_schedule()
        prepared = PreparedDenoise(context, weights, ActionConditioning(modulation, times.numpy(), dt),
                                  sources, max_cached_buckets=2, collective_window=4)
        try:
            for prefix in (447,451):
                binding = prepared.bind(_inputs(prefix=prefix))
                assert binding.plan.node_counts == (100,)*8
                assert len({id(step.native) for step in binding.plan.steps}) == 8
                binding.close()
            assert context.native.windows.count(4) == 16 and context.native.windows.count(1) == 2
            assert prepared.describe()["collective_window"] == 4
            assert context.native.trace == []  # Planning evidence only, never boardmath.
        finally:
            prepared.close()
