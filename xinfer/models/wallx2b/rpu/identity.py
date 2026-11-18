"""Prepared Wall-X 2B FP16 TP8 action identity-mixer layer over public native TP operations.

This is one real B1/H16 component, not a full policy executor. All intermediate
activations remain resident. The native provider retains its 48-collective limit,
serial segment completion checks, and numerical guards. Shared scratch may be
reused by multiple prepared layers only in a serial execution order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xinfer.platforms.rpu import RpuBuffer, RpuExecutionContext
from .config import FULL_ATTENTION_LAYERS


def _host(value, shape, name):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float16
        or value.shape != shape
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} requires finite contiguous FP16 {shape}")
    return value


def _buffer(value, context, shape, distribution, name, *, weight=False):
    if not isinstance(value, RpuBuffer) or value.context is not context:
        raise ValueError(f"{name} must be an RPU buffer from this context")
    value._check()
    if (
        value.shape != shape
        or value.dtype != np.dtype("float16")
        or value.distribution != distribution
        or value.storage != ("ddr" if weight else "spm")
        or value.layout != ("tp8_nk16" if weight else "row_major")
        or value.immutable != weight
        or value.state_kind != "none"
    ):
        raise ValueError(f"{name} has incompatible shape/layout/distribution/ownership")
    return value


@dataclass(frozen=True)
class IdentityLayerWeights:
    """Model-prepared natural TP8 weights; down already contains source /4.

    ``prepare`` performs only native packing/upload. The caller must use
    ``WeightPreparer`` for source splitting and profile scaling; applying /4 here
    would silently double-scale already prepared checkpoint weights.
    """

    gate: RpuBuffer
    up: RpuBuffer
    down: RpuBuffer

    @classmethod
    def prepare(cls, context: RpuExecutionContext, *, gate, up, down):
        _host(gate, (8, 256, 1024), "gate weight")
        _host(up, (8, 256, 1024), "up weight")
        _host(down, (8, 1024, 256), "down weight (source /4)")
        owners = []
        try:
            for value in (gate, up, down):
                owners.append(context.prepare_weight(value, layout="tp8_nk16"))
            return cls(*owners)
        except BaseException:
            for owner in reversed(owners):
                owner.close()
            raise


@dataclass(frozen=True)
class IdentityModulation:
    """One layer/step's once-packed CPU Ada rows and independent gate owners.

    the native provider has no public tensor view into modulation row2. Therefore the
    same packed FP16 gate bytes are uploaded into a separate vector once at
    preparation, without any activation readback or invented alias/proof.
    The caller supplies gamma=1+scale, shift and gate/S (MLP gate also x4).
    """

    mixer: RpuBuffer
    mixer_gate: RpuBuffer
    mlp: RpuBuffer
    mlp_gate: RpuBuffer

    @classmethod
    def prepare(cls, context: RpuExecutionContext, *, mixer, mlp):
        _host(mixer, (3, 1024), "mixer modulation")
        _host(mlp, (3, 1024), "MLP modulation")
        owners = []
        try:
            for row in (mixer, mlp):
                for value in (row, np.ascontiguousarray(row[2])):
                    owner = context.empty(value.shape, distribution="replicated")
                    owners.append(owner)
                    owner.copy_from(value)
            return cls(*owners)
        except BaseException:
            for owner in reversed(owners):
                owner.close()
            raise


class IdentityWorkspace:
    """144 KiB per core, allocated once and shared across serial layer plans.

    mixer: pre-norm then post-identity residual, live through final residual;
    mlp_norm: post-norm, live through both column GEMMs;
    gate/up: column shards, gate is overwritten by fused SiLU-mul;
    partial/reduced: separate pre/post reduction owners. Output belongs to
    the caller, so preparing another layer cannot silently alias its result.
    """

    bytes_per_core = 144 * 1024

    def __init__(self, context: RpuExecutionContext):
        self.context = context
        owners = []
        try:
            for name, shape, distribution in (
                ("mixer", (16, 1024), "replicated"),
                ("mlp_norm", (16, 1024), "replicated"),
                ("gate", (16, 256), "sharded"),
                ("up", (16, 256), "sharded"),
                ("partial", (16, 1024), "partial"),
                ("reduced", (16, 1024), "replicated"),
            ):
                owner = context.empty(
                    shape, distribution=distribution, **({"shard_axis": 1} if distribution == "sharded" else {})
                )
                owners.append(owner)
                setattr(self, name, owner)
        except BaseException:
            for owner in reversed(owners):
                owner.close()
            raise


@dataclass
class PreparedIdentityLayer:
    """Nine prepared operations, one explicit collective and a reusable batch.

    ``operations`` may instead be composed by a model-owned larger batch.
    With current native semantics the standalone batch has three SDK segments
    (seven ordinary operations, reduction, final residual gate). It is a
    functional bring-up path with no claim of retained-Graph performance.
    Input/output, weights, modulation and shared scratch remain caller-owned.
    """

    context: RpuExecutionContext
    layer_index: int
    src: RpuBuffer
    out: RpuBuffer
    weights: IdentityLayerWeights
    modulation: IdentityModulation
    workspace: IdentityWorkspace
    operations: tuple
    execution: object | None

    operation_count = 9
    collective_count = 1

    def execute(self):
        if self.execution is None:
            raise RuntimeError("this component's operations were prepared for a caller-owned batch")
        return self.execution.execute()

    def close(self):
        """Release prepared bindings after completion; external buffers stay owned."""
        if self.execution is not None:
            self.execution.close()
        for operation in reversed(self.operations):
            operation.close()


def prepare_identity_layer(
    context: RpuExecutionContext,
    *,
    layer_index: int,
    src: RpuBuffer,
    out: RpuBuffer,
    weights: IdentityLayerWeights,
    modulation: IdentityModulation,
    workspace: IdentityWorkspace,
    make_batch: bool = True,
) -> PreparedIdentityLayer:
    """Lower one exact identity layer; no GDN and no MLP deletion.

    Both Ada norms use S/4=64 and epsilon=(checkpoint 1e-6)/16. The physical
    residual x/S and packed gates/S are supplied unchanged. Down's local
    FP16 partials are reduced exactly once *before* the MLP residual gate.
    """
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not 0 <= layer_index < 24
        or layer_index in FULL_ATTENTION_LAYERS
    ):
        raise ValueError("identity component accepts only the 18 Wall-X 2B identity-mixer layers")
    if not isinstance(make_batch, bool):
        raise TypeError("make_batch must be bool")
    _buffer(src, context, (16, 1024), "replicated", "src")
    _buffer(out, context, (16, 1024), "replicated", "out")
    if not isinstance(workspace, IdentityWorkspace) or workspace.context is not context:
        raise ValueError("workspace must belong to this RPU context")
    if not isinstance(weights, IdentityLayerWeights) or not isinstance(modulation, IdentityModulation):
        raise TypeError("explicit identity weight and modulation owners required")
    for name, shape in (("gate", (8, 256, 1024)), ("up", (8, 256, 1024)), ("down", (8, 1024, 256))):
        _buffer(getattr(weights, name), context, shape, "shared", name + " weight", weight=True)
    for name, shape in (("mixer", (3, 1024)), ("mlp", (3, 1024)), ("mixer_gate", (1024,)), ("mlp_gate", (1024,))):
        _buffer(getattr(modulation, name), context, shape, "replicated", name + " modulation")
    scratch = []
    for name, shape, role in (
        ("mixer", (16, 1024), "replicated"),
        ("mlp_norm", (16, 1024), "replicated"),
        ("gate", (16, 256), "sharded"),
        ("up", (16, 256), "sharded"),
        ("partial", (16, 1024), "partial"),
        ("reduced", (16, 1024), "replicated"),
    ):
        scratch.append(_buffer(getattr(workspace, name), context, shape, role, name + " scratch"))
    if len({id(b.native) for b in (src, out, *scratch)}) != 8:
        raise ValueError("input, independent output and six scratch owners must be distinct")
    norm_scalars = {"epsilon": 1.0e-6 / 16.0, "normalization_scale": 64.0}
    definitions = (
        (
            "rmsnorm",
            {"out": workspace.mixer, "x": src, "modulation": modulation.mixer},
            None,
            norm_scalars,
        ),
        (
            "residual_gate",
            {"out": workspace.mixer, "branch": workspace.mixer, "residual": src, "gate": modulation.mixer_gate},
            None,
            None,
        ),
        (
            "rmsnorm",
            {"out": workspace.mlp_norm, "x": workspace.mixer, "modulation": modulation.mlp},
            None,
            norm_scalars,
        ),
        (
            "gemm",
            {"out": workspace.gate, "x": workspace.mlp_norm, "w": weights.gate},
            None,
            None,
        ),
        (
            "gemm",
            {"out": workspace.up, "x": workspace.mlp_norm, "w": weights.up},
            None,
            None,
        ),
        ("silu_and_mul", {"out": workspace.gate, "x": workspace.gate, "y": workspace.up}, None, None),
        (
            "gemm",
            {"out": workspace.partial, "x": workspace.gate, "w": weights.down},
            None,
            None,
            {"parallel": "row"},
        ),
        ("all_reduce", {"out": workspace.reduced, "x": workspace.partial}, {"with_residual": 0}, None),
        (
            "residual_gate",
            {"out": out, "branch": workspace.reduced, "residual": workspace.mixer, "gate": modulation.mlp_gate},
            None,
            None,
        ),
    )
    operations = []
    try:
        for op, tensors, integers, scalars, *selectors in definitions:
            operations.append(context.prepare(op, tensors=tensors, integers=integers, scalars=scalars,
                                              **(selectors[0] if selectors else {})))
        execution = context.batch(operations) if make_batch else None
    except BaseException:
        for operation in reversed(operations):
            operation.close()
        raise
    return PreparedIdentityLayer(
        context, layer_index, src, out, weights, modulation, workspace, tuple(operations), execution
    )
