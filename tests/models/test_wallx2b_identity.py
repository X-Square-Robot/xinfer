"""CPU binding simulation and dyadic oracle; these are not RPU board results."""

import numpy as np
import pytest

from xinfer.models.wallx2b.rpu.identity import (
    IdentityLayerWeights,
    IdentityModulation,
    IdentityWorkspace,
    prepare_identity_layer,
)
from xinfer.platforms.rpu import RpuBusyError, RpuExecutionContext
from wallx2b_plan_helpers import catalog_call


class _Owner:
    def close(self):
        self.closed = True


class _Tensor(_Owner):
    def __init__(self, shape, **description):
        self.shape = tuple(shape)
        self.dtype = description.get("dtype", "float16")
        self.storage = description.get("storage", "spm")
        self.layout = description.get("layout", "row_major")
        self.distribution = description.get("distribution", "local")
        self.shard_axis = description.get("shard_axis", -1)
        self.state_kind = description.get("state_kind", "none")
        self.immutable = description.get("immutable", False)
        self.array = np.empty((8, *self.shape) if self.storage == "spm" else self.shape, dtype=self.dtype)

    def copy_from(self, value):
        self.array[...] = value

    def numpy(self):
        return self.array.copy()


def _half(value):
    return np.asarray(value, dtype=np.float16)


def _sparse_gemm(src, weight):
    """Independent FP32 sum of sparse products; test weights have one term/output."""
    result = np.zeros((8, 16, weight.shape[1]), dtype=np.float32)
    for core in range(8):
        rows, cols = np.nonzero(weight[core])
        for n, k in zip(rows, cols):
            result[core, :, n] += src[core, :, k].astype(np.float32) * np.float32(weight[core, n, k])
    return _half(result)


class _Execution(_Owner):
    def __init__(self, context, operations):
        self.context, self.operations = context, operations

    def execute(self, *, async_):
        assert async_ is True
        self.context.starts += 1

    def wait(self):
        for operation in self.operations:
            op, t, s = operation.name, operation.tensors, operation.scalars
            a = {name: owner.array for name, owner in t.items()}
            if op == "adaptive_rmsnorm_m16h1024_f16":
                assert s == {"epsilon": 1e-6 / 16, "normalization_scale": 64.0}
                x = a["x"].copy()
                mean = _half(np.sum(x.astype(np.float32) ** 2, axis=-1, keepdims=True) * np.float32(4))
                # The dyadic fixture avoids the device's under/overflow recovery arm.
                assert np.isfinite(mean).all() and (mean > np.float16(2**-14)).all()
                inv = _half(1 / np.sqrt(_half(mean + np.float16(s["epsilon"])).astype(np.float32)))
                result = _half(_half(x * inv) * np.float16(64))
                result = _half(result * a["modulation"][:, None, 0, :])
                a["out"][...] = _half(result + a["modulation"][:, None, 1, :])
            elif op == "residual_gate_m16h1024_f16":
                product = _half(a["branch"] * a["gate"][:, None, :])
                a["out"][...] = _half(product + a["residual"])
            elif op == "gemm_ddr_col_m16n256k1024_acc32":
                a["out"][...] = _sparse_gemm(a["x"], a["weight"])
            elif op == "silu_mul_fp16":
                sigmoid = _half(1 / (1 + np.exp(-a["x"].astype(np.float32))))
                a["out"][...] = _half(_half(a["x"] * sigmoid) * a["y"])
            elif op == "gemm_ddr_row_m16n1024k256_acc32":
                a["out"][...] = _sparse_gemm(a["src"], a["weight"])
            elif op == "all_reduce_residual":
                assert operation.integers == {"with_residual": 0}
                value = a["input"].copy()
                while len(value) > 1:
                    value = _half(value[::2] + value[1::2])
                a["out"][...] = value[0]
            else:
                raise AssertionError(op)
            self.context.trace.append(op)


class _Backend(_Owner):
    def __init__(self, **_):
        self.starts, self.trace = 0, []

    def empty(self, shape, **description):
        return _Tensor(shape, **description)

    def prepare_weight(self, host, *, layout):
        result = _Tensor(host.shape, storage="ddr", distribution="shared", layout=layout, immutable=True)
        result.copy_from(host)
        return result

    def prepare(self, op, *, tensors, integers=None, scalars=None, **selectors):
        # These mock checks exercise our binding choices, not native ABI validation.
        op, tensors, integers = catalog_call(op, tensors, integers, selectors)
        if op == "gemm_ddr_col_m16n256k1024_acc32":
            assert tensors["x"].distribution == "replicated"
            assert (tensors["out"].distribution, tensors["out"].shard_axis) == ("sharded", 1)
        elif op == "gemm_ddr_row_m16n1024k256_acc32":
            assert (tensors["src"].distribution, tensors["src"].shard_axis) == ("sharded", 1)
            assert tensors["out"].distribution == "partial"
        elif op == "all_reduce_residual":
            assert tensors["input"].distribution == "partial"
            assert tensors["out"].distribution == "replicated"
            assert tensors["out"] is not tensors["input"]
        elif op == "residual_gate_m16h1024_f16":
            assert all(t.distribution == "replicated" for t in tensors.values())
            assert tensors["out"] is not tensors["residual"]
            assert tensors["branch"] is not tensors["residual"]
        result = _Owner()
        result.name, result.tensors = op, tensors
        result.integers, result.scalars = integers or {}, scalars or {}
        return result

    def batch(self, operations):
        return _Execution(self, operations)


@pytest.fixture
def prepared():
    with RpuExecutionContext(backend_factory=_Backend) as context:
        gate = np.zeros((8, 256, 1024), dtype=np.float16)
        up = np.zeros_like(gate)
        down = np.zeros((8, 1024, 256), dtype=np.float16)
        for core in range(8):
            gate[core, :, 0] = 16  # sigmoid(16) rounds to exactly 1 in FP16
            up[core, :, 0] = np.float16((core + 1) / 32)
            down[core, np.arange(1024), np.arange(1024) % 256] = np.float16(1 / 4)
        weights = IdentityLayerWeights.prepare(context, gate=gate, up=up, down=down)
        mixer = np.zeros((3, 1024), dtype=np.float16)
        mixer[0], mixer[2] = 1, np.float16(1 / 256)
        mlp = mixer.copy()
        mlp[2] = np.float16(4 / 256)  # source down /4 compensated in packed gate
        modulation = IdentityModulation.prepare(context, mixer=mixer, mlp=mlp)
        workspace = IdentityWorkspace(context)
        src = context.empty((16, 1024), distribution="replicated")
        src.copy_from(np.full((16, 1024), 1 / 256, dtype=np.float16))
        out = context.empty((16, 1024), distribution="replicated")
        plan = prepare_identity_layer(
            context, layer_index=0, src=src, out=out, weights=weights, modulation=modulation, workspace=workspace
        )
        yield context, plan


def test_mock_identity_chain_has_nonzero_mlp_and_exact_dyadic_result(prepared):
    context, plan = prepared
    before = dict(context.stats)
    completion = plan.execute()
    with pytest.raises(RpuBusyError):
        plan.out.numpy()
    completion.wait()
    # pre-norm=1; identity residual=1/256+1/256=1/128.
    # per-core down partial=(core+1)/8; TP8 sum=4.5;
    # final=1/128 + (4.5 * 4/256) = 0.078125 exactly.
    np.testing.assert_array_equal(plan.out.numpy(), np.full((8, 16, 1024), 0.078125, np.float16))
    assert plan.operation_count == len(context.native.trace) == 9
    assert context.native.trace.count("all_reduce_residual") == plan.collective_count == 1
    for key in ("allocations", "weight_preparations", "operation_preparations", "uploads"):
        assert context.stats[key] == before[key], key
    assert context.stats["downloads"] == before.get("downloads", 0) + 1


def test_serial_layers_share_scratch_and_preserve_previous_output(prepared):
    context, first = prepared
    first.execute().wait()
    previous = first.out.numpy()
    second_out = context.empty((16, 1024), distribution="replicated")
    second = prepare_identity_layer(
        context,
        layer_index=1,
        src=first.out,
        out=second_out,
        weights=first.weights,
        modulation=first.modulation,
        workspace=first.workspace,
    )
    allocations = context.stats["allocations"]
    second.execute().wait()
    np.testing.assert_array_equal(first.out.numpy(), previous)
    # First identity contribution 1/256, then the same nonzero MLP contribution.
    np.testing.assert_array_equal(second.out.numpy(), np.full((8, 16, 1024), 0.15234375, np.float16))
    first.execute().wait()
    np.testing.assert_array_equal(first.out.numpy(), previous)
    assert context.stats["allocations"] == allocations
    assert first.workspace.bytes_per_core == 144 * 1024


@pytest.mark.parametrize("layer", [3, 7, 11, 15, 19, 23, -1, 24, True])
def test_identity_component_rejects_full_attention_and_invalid_layers(prepared, layer):
    context, plan = prepared
    with pytest.raises(ValueError, match="18 CES identity"):
        prepare_identity_layer(
            context,
            layer_index=layer,
            src=plan.src,
            out=plan.out,
            weights=plan.weights,
            modulation=plan.modulation,
            workspace=plan.workspace,
        )


def test_identity_output_cannot_alias_input_or_shared_scratch(prepared):
    context, plan = prepared
    for out in (plan.src, plan.workspace.reduced):
        with pytest.raises(ValueError, match="must be distinct"):
            prepare_identity_layer(
                context,
                layer_index=2,
                src=plan.src,
                out=out,
                weights=plan.weights,
                modulation=plan.modulation,
                workspace=plan.workspace,
            )


def test_component_operations_can_enter_caller_batch_without_repreparing_weights(prepared):
    context, first = prepared
    count = context.stats["weight_preparations"]
    out = context.empty((16, 1024), distribution="replicated")
    second = prepare_identity_layer(
        context,
        layer_index=2,
        src=first.out,
        out=out,
        weights=first.weights,
        modulation=first.modulation,
        workspace=first.workspace,
        make_batch=False,
    )
    with pytest.raises(RuntimeError, match="caller-owned batch"):
        second.execute()
    context.batch((*first.operations, *second.operations)).execute().wait()
    np.testing.assert_array_equal(second.out.numpy(), np.full((8, 16, 1024), 0.15234375, np.float16))
    assert context.stats["weight_preparations"] == count
    assert len(context.native.trace) == 18


def test_modulation_uploads_preserve_once_packed_gate_bytes(prepared):
    _, plan = prepared
    np.testing.assert_array_equal(plan.modulation.mixer.numpy()[:, 2], plan.modulation.mixer_gate.numpy())
    np.testing.assert_array_equal(plan.modulation.mlp.numpy()[:, 2], plan.modulation.mlp_gate.numpy())
    assert plan.modulation.mlp_gate.numpy()[0, 0] == np.float16(4 / 256)
