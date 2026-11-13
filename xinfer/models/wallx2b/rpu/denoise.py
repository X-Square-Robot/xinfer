"""Resident Wall-X 2B FP16 TP8 action lowering over xcompute's public TP8 interface.

The sparse decoder has eighteen identity token mixers and six attention
mixers; every layer retains its MLP. The fused down/reduce/gate/Ada boundary
matches the reference, including its identity-next epilogue. Euler writes
FP16 after mask, add, multiply and add. No activation crosses the host between
steps. Eight independent step batches are retained per physical bucket; each
captures its fixed immutable Ada-row inputs and replays without step rebinds.
whole_stage mode instead retains ONE fused queue per bucket: all 240
collectives pin static vendor pool ids 3..49 per occurrence (the original's
action_fusion_counter formula) with per-launch zeroing {0,1,own}.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch

from xinfer.models.interface import PreparedRequest
from xinfer.platforms.rpu import RpuBuffer, RpuExecutionContext
from .action import ActionConditioning, action_attention_mask, flow_schedule
from .config import FULL_ATTENTION_LAYERS


def _release_all(resources, first_error=None):
    failed = False
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as error:
            failed = True
            if first_error is None:
                first_error = error
            elif error is not first_error and hasattr(first_error, "add_note"):
                first_error.add_note(f"Action cleanup also failed: {error}")
    return first_error, failed


def _array(value, shape, dtype, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        value = value.detach().numpy()
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} requires {np.dtype(dtype)} {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(value)


def _buffer(value, context, shape, *, storage, layout="row_major", distribution="shared", immutable=None):
    if not isinstance(value, RpuBuffer) or value.context is not context:
        raise ValueError("Action buffers must belong to this RPU context")
    value._check()
    if (
        value.shape != shape
        or value.dtype != np.dtype("float16")
        or value.storage != storage
        or value.layout != layout
        or value.distribution != distribution
    ):
        raise ValueError(f"incompatible Action buffer; expected {storage}/{layout}/{distribution} FP16 {shape}")
    if immutable is not None and value.immutable != immutable:
        raise ValueError("Action buffer has incompatible immutability")
    return value


def action_rope_tables(position_ids) -> tuple[np.ndarray, np.ndarray]:
    """Bake the already shifted logical action positions into token-indexed tables.

    The processor applies the checkpoint's gap10. Physical KV bucket offsets
    never participate in RoPE. Float64 frequency construction and interleaving
    preserve the CPU reference's single FP16 table boundary.
    """
    if isinstance(position_ids, torch.Tensor):
        if position_ids.device.type != "cpu":
            raise ValueError("Action positions must be CPU metadata")
        position_ids = position_ids.detach().numpy()
    if not isinstance(position_ids, np.ndarray) or position_ids.dtype != np.int64:
        raise ValueError("Action positions require int64 [3,1,16] or [3,16]")
    if position_ids.shape == (3, 1, 16):
        position_ids = position_ids[:, 0]
    if position_ids.shape != (3, 16) or (position_ids < 0).any():
        raise ValueError("Action positions require nonnegative int64 [3,16]")
    if not np.array_equal(position_ids, np.broadcast_to(position_ids[0], (3, 16))):
        raise ValueError("the non-RTC action suffix requires identical T/H/W positions")
    if not np.array_equal(np.diff(position_ids, axis=1), np.ones((3, 15), np.int64)):
        raise ValueError("Action positions must be consecutive logical positions")
    inverse = 1.0 / (10_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    positions = torch.from_numpy(np.ascontiguousarray(position_ids)).to(torch.float64)
    frequencies = (inverse[None, None, :, None].expand(3, 1, -1, 1) @ positions[:, None, None, :]).transpose(2, 3)
    merged = frequencies[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        stop = (11, 11, 10)[axis] * 3
        merged[..., offset:stop:3] = frequencies[axis, ..., offset:stop:3]
    return tuple(table.squeeze(0).half().contiguous().numpy().copy() for table in (merged.cos(), merged.sin()))


def pack_action_request(tensors: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep noise and padding velocity's original, separate FP16 boundaries."""
    noise = _array(tensors["noise_fp16"], (1, 16, 26), "float16", "initial noise")
    padding = _array(tensors["padding_velocity_fp16"], (1, 16, 26), "float16", "padding velocity")
    mask = _array(tensors["dof_mask"], (1, 1, 26), "float32", "DOF mask")
    expected_mask = np.array([1] * 20 + [0] * 6, dtype=np.float32).reshape(1, 1, 26)
    if not np.array_equal(mask, expected_mask):
        raise ValueError("Action requires the admitted first20/last6 DOF mask")
    half_mask = mask.astype(np.float16)
    packed = np.zeros((16, 64), np.float16)
    packed[:, :26] = noise[0]
    packed[:, 26:52] = half_mask[0, 0]
    keep = np.zeros(64, np.float16)
    keep[:26] = half_mask[0, 0]
    padded_velocity = np.zeros((16, 64), np.float16)
    padded_velocity[:, :26] = padding[0] * (np.float16(1) - half_mask[0, 0])
    return packed, keep, padded_velocity


@dataclass(frozen=True)
class DenoiseWeights:
    """Borrowed xcompute owners from the one-time model weight loader.

    QK gamma is an immutable row-major DDR table, copied into shared scratch
    at each attention layer. All GEMM packing is owned by xcompute; down
    weights already contain source/4 and conditioning already restores gate*4.
    """

    buffers: Mapping[str, RpuBuffer]

    @classmethod
    def from_buffers(cls, context, buffers: Mapping[str, RpuBuffer]):
        expected = {"action.input": ((1024, 64), "nk16"), "action.output": ((64, 1024), "nk16")}
        for layer in range(24):
            for name, shape in (("gate", (8, 256, 1024)), ("up", (8, 256, 1024)), ("down", (8, 1024, 256))):
                expected[f"action.{layer}.{name}"] = (shape, "tp8_nk16")
            if layer in FULL_ATTENTION_LAYERS:
                expected[f"action.{layer}.qkv"] = ((4, 8, 256, 1024), "tp8_nk16")
                expected[f"action.{layer}.o"] = ((8, 1024, 256), "tp8_nk16")
                expected[f"action.{layer}.qk_norm"] = ((2, 256), "row_major")
        missing = set(expected) - set(buffers)
        if missing:
            raise ValueError(f"Action weight owners are missing {sorted(missing)}")
        result = {}
        for name, (shape, layout) in expected.items():
            result[name] = _buffer(buffers[name], context, shape, storage="ddr", layout=layout, immutable=True)
        return cls(result)

    def layer(self, layer, name):
        return self.buffers[f"action.{layer}.{name}"]


class DenoiseWorkspace:
    """A reusable Action slice plus independent core0 storage.

    The local arena may be shared with Vision/prefill because every live input
    is brought in by Action's device copies. Core0 buffers are separate native
    allocations, preserving zero nonowner cores across all eight Euler steps.
    No single/owner0 identity is inferred from arbitrary reused scratch.
    """

    _slices = (
        ("residual", (16, 1024), "replicated", -1),
        ("mixed", (16, 1024), "replicated", -1),
        ("normalized", (16, 1024), "replicated", -1),
        ("decoder_scratch", (16, 1024), "replicated", -1),
        ("partial", (16, 1024), "partial", -1),
        ("gate", (16, 256), "sharded", 1),
        ("planes", (4, 16, 256), "local", -1),
        ("attention", (16, 256), "sharded", 1),
        ("qk_norm", (2, 256), "replicated", -1),
        ("mod0", (3, 1024), "replicated", -1),
        ("mod1", (3, 1024), "replicated", -1),
        ("mask", (16 * 528,), "replicated", -1),
    )
    arena_bytes = sum(math.ceil(math.prod(shape) * 2 / 256) * 256 for _, shape, _, _ in _slices)
    single_bytes = 40 * 1024 + 512

    def __init__(self, context: RpuExecutionContext, *, arena: RpuBuffer | None = None):
        self.context = context
        self.closed = False
        self._owners = []
        try:
            if arena is None:
                arena = context.empty((self.arena_bytes // 2,), distribution="local")
                self._owners.append(arena)
            else:
                _buffer(arena, context, arena.shape, storage="spm", distribution="local", immutable=False)
                if math.prod(arena.shape) * 2 < self.arena_bytes:
                    raise ValueError("shared Action arena is too small")
            self.arena = arena
            offset = 0
            for name, shape, distribution, axis in self._slices:
                owner = context.view(arena, shape, offset_bytes=offset, distribution=distribution, shard_axis=axis)
                self._owners.append(owner)
                setattr(self, name, owner)
                offset += math.ceil(math.prod(shape) * 2 / 256) * 256
            for name, shape in (
                ("action", (16, 64)),
                ("padding_velocity", (16, 64)),
                ("keep_mask", (64,)),
                ("projection", (16, 1024)),
                ("velocity", (16, 64)),
                ("input_bias", (1024,)),
                ("output_bias", (64,)),
            ):
                owner = context.empty(shape, distribution="single")
                self._owners.append(owner)
                setattr(self, name, owner)
            for name, index in (("query", 0), ("key", 1), ("value", 2), ("attention_gate", 3)):
                owner = context.view(
                    self.planes, (16, 256), offset_bytes=index * 16 * 256 * 2, distribution="sharded", shard_axis=1
                )
                self._owners.append(owner)
                setattr(self, name, owner)
            self.mod0_gate = context.view(self.mod0, (1024,), offset_bytes=2048 * 2)
            self._owners.append(self.mod0_gate)
            # These two single owners never alias a stage's shared workspace.
            self.input_bias.copy_from(np.zeros(1024, np.float16))
            self.output_bias.copy_from(np.zeros(64, np.float16))
        except BaseException as error:
            _release_all((self,), error)
            raise

    def close(self):
        if self.closed:
            return
        error, failed = _release_all(reversed(self._owners))
        if not failed:
            self._owners.clear()
            self.closed = True
        if error is not None:
            raise error


@dataclass
class _BucketPlan:
    bucket: int
    mask_ddr: RpuBuffer
    mask_spm: RpuBuffer
    entry_copies: tuple
    prefix_operation: object
    entry_execution: object
    execution: object
    steps: tuple
    operations: tuple
    views: tuple
    node_counts: tuple[int, ...]

    def close(self):
        error, _ = _release_all(
            (
                *reversed(self.steps),
                self.entry_execution,
                self.prefix_operation,
                *reversed((*self.entry_copies, *self.operations)),
                *reversed(self.views),
            )
        )
        if error is not None:
            raise error


class DenoiseBinding:
    """One initialization and eight retained, independently prepared step batches."""

    def __init__(self, owner, plan, entry):
        self.owner = owner
        self.plan = plan
        self.entry = entry
        self.prefix_operation = plan.prefix_operation
        if plan.entry_execution is not None:
            self.executions = (entry, *plan.steps)
        else:
            # Entry fold: `entry` is the bound invocation of the single
            # fused queue (prefix copy inside); executing it IS the stage.
            self.executions = (entry,)
        self.closed = False

    def collect_normalized(self):
        """Read after all executions complete; return independently owned FP32."""
        if self.closed:
            raise RuntimeError("Action request binding is closed")
        self.owner.context._check_idle()
        if self.owner.context.needs_reset:
            raise RuntimeError("failed Action execution cannot publish an output")
        result = self.owner.workspace.action.numpy()[0, :, :26].astype(np.float32)[None]
        if result.shape != (1, 16, 26) or not np.isfinite(result).all():
            raise FloatingPointError("Action produced nonfinite or malformed normalized actions")
        return result.copy()

    def close(self):
        if self.closed:
            return
        error = None
        try:
            self.owner.context.wait()
        except BaseException as caught:
            error = caught
        error, failed = _release_all((self.entry,), error)
        if not failed:
            self.closed = True
            self.owner._active = None
        if error is not None:
            raise error


class PreparedDenoise:
    """Prepared complete Action profile, with all eight Euler steps on device.

    Step plans are cached by physical prefix bucket. Real prefix length is
    applied by a mutable binding on the retained prefix-copy operation; mask,
    initial action, padding velocity and RoPE update stable DDR owners. Entry
    and step queues replay without per-request operation preparation. Multiple
    synchronized batches remain distinct from one whole-stage device graph.
    """

    def __init__(
        self,
        context: RpuExecutionContext,
        weights: Mapping[str, RpuBuffer] | DenoiseWeights,
        conditioning: ActionConditioning,
        prefix_caches: Sequence[tuple[RpuBuffer, RpuBuffer]],
        *,
        workspace: DenoiseWorkspace | None = None,
        max_cached_buckets: int = 8,
        collective_window: int = 1,
        whole_stage: bool = False,
        fused_steps: bool = False,
    ):
        if (
            not isinstance(max_cached_buckets, int)
            or isinstance(max_cached_buckets, bool)
            or not 1 <= max_cached_buckets <= 8
        ):
            raise ValueError("Action cache requires 1..8 retained physical buckets")
        if (isinstance(collective_window, bool) or not isinstance(collective_window, int)
                or collective_window not in (1, 4)):
            raise ValueError("Action supports explicit collective_window1 or4")
        if not isinstance(whole_stage, bool):
            raise ValueError("Action whole_stage must be an explicit boolean")
        if not isinstance(fused_steps, bool):
            raise ValueError("Action fused_steps must be an explicit boolean")
        if whole_stage and fused_steps:
            raise ValueError("Action whole_stage already folds the entry; "
                             "fused_steps is the plain-entry/fused-steps mode")
        if whole_stage and collective_window != 1:
            raise ValueError("Action whole_stage replaces windowing; use collective_window1")
        if fused_steps and collective_window != 1:
            raise ValueError("Action fused_steps replaces windowing; use collective_window1")
        if collective_window != 1:
            context.require_collective_window(collective_window, family="M16")
        if whole_stage or fused_steps:
            context.require_batch_fused()
        self.collective_window = collective_window
        self.whole_stage = whole_stage
        self.fused_steps = fused_steps
        if len(prefix_caches) != 6:
            raise ValueError("Action needs exactly six resident full-attention prefix cache pairs")
        self.context = context
        self.weights = DenoiseWeights.from_buffers(
            context, weights.buffers if isinstance(weights, DenoiseWeights) else weights
        )
        self.source_caches = tuple(prefix_caches)
        for key, value in self.source_caches:
            for cache, layout in ((key, "causal_k_cache"), (value, "causal_v_cache")):
                _buffer(cache, context, (8, 784, 256), storage="ddr", layout=layout, immutable=False)
                if cache.state_kind != "cache":
                    raise ValueError("prefix handoffs must preserve native cache state owners")
        schedule, dt = flow_schedule()
        modulation = _array(conditioning.modulation, (8, 49, 3, 1024), "float16", "Action conditioning")
        if (
            not np.array_equal(conditioning.times, schedule.numpy())
            or np.float16(conditioning.delta_t).tobytes() != dt.tobytes()
        ):
            raise ValueError("Action conditioning must use the admitted NFE8 schedule")
        self.delta_t = float(dt)
        self.max_cached_buckets = max_cached_buckets
        self.stats = Counter()
        self._plans = OrderedDict()
        self._active = None
        self._owners = []
        self._epoch = context.epoch
        self.closed = False
        self._owns_workspace = workspace is None
        self.workspace = workspace
        try:
            if self.workspace is None:
                self.workspace = DenoiseWorkspace(context)
            if self.workspace.context is not context or self.workspace.closed:
                raise ValueError("Action workspace must belong to this context and remain open")
            self.modulation = self._own(context.prepare_weight(modulation, layout="row_major"))
            self._rows = []
            for step in range(8):
                rows = []
                for row in range(49):
                    offset = (step * 49 + row) * 3 * 1024 * 2
                    rows.append(self._own(context.view(self.modulation, (3, 1024), offset_bytes=offset)))
                self._rows.append(tuple(rows))
            self.cosine = self._own(context.empty((16, 32), storage="ddr", distribution="shared"))
            self.sine = self._own(context.empty((16, 32), storage="ddr", distribution="shared"))
            self.action_input = self._own(context.empty((16, 64), storage="ddr", distribution="shared"))
            self.padding_input = self._own(context.empty((16, 64), storage="ddr", distribution="shared"))
            self.keep_input = self._own(context.empty((64,), storage="ddr", distribution="shared"))
            self.hidden_staging = self._own(context.empty((16, 1024), storage="ddr", distribution="shared"))
            self.mask_storage = self._own(context.empty((16 * 528,), storage="ddr", distribution="shared"))
            self.destination_caches = tuple(
                (self._cache("causal_k_cache"), self._cache("causal_v_cache")) for _ in range(6)
            )
        except BaseException as error:
            _release_all((self,), error)
            raise

    def _own(self, owner):
        self._owners.append(owner)
        return owner

    def _cache(self, layout):
        return self._own(
            self.context.empty((8, 784, 256), storage="ddr", layout=layout, distribution="shared", state_kind="cache")
        )

    def _prepare_step(self, step, mask, bucket):
        context, w = self.context, self.workspace
        rows = self._rows[step]
        operations = []
        mutable_rows = []
        row_indexes = {id(row): index for index, row in enumerate(rows)}

        def op(name, tensors, *, integers=None, scalars=None, **selectors):
            value = context.prepare(name, tensors=tensors, integers=integers, scalars=scalars,
                                    **selectors)
            operations.append(value)
            return value

        def copy(src, dst, mode, *, row_index=None):
            operation = context.prepare_copy(src, dst, mode=mode)
            operations.append(operation)
            if row_index is not None:
                mutable_rows.append((operation, {"input": row_index}))

        norm_scalars = {"epsilon": 1.0e-6 / 16, "normalization_scale": 64.0}

        def fusion_counter(layer, kind):
            # The original's per-occurrence static pool assignment, read from
            # rpu_qwen3_5_model.cpp action_fusion_counter: id = 3 + (occ % 47)
            # with occurrence = 48*step + 2*layer + kind (kind0 = attention-o,
            # kind1 = MLP down). Consecutive emitting launches never share an
            # id; each id carries ~5 occurrences across the 8 NFE steps, and
            # reuse stays graph-ordered through the layer chain.
            occurrence = 48 * step + 2 * layer + kind
            return 3 + (occurrence % 47)

        def down(src, weight, residual, stream, norm, gate_row, next_row, *,
                 next2_row=None, w_out=None, layer=None, kind=None):
            tensors = dict(
                x=src,
                w=weight,
                partial=w.partial,
                residual=residual,
                stream=stream,
                norm=norm,
                gate_row=gate_row,
                next_row=next_row,
            )
            if next2_row is not None:
                tensors.update(next2_row=next2_row, w_out=w_out)
            integers = None
            if self.whole_stage or self.fused_steps:
                if kind is None or layer is None:
                    raise ValueError("fused action steps require the down occurrence (layer, kind)")
                integers = {"counter": fusion_counter(layer, kind)}
            operation = op(
                "down_allreduce_norm",
                tensors,
                integers=integers,
                scalars=norm_scalars,
            )
            mutable_rows.append(
                (
                    operation,
                    {
                        name: row_indexes[id(tensors[name])]
                        for name in ("gate_row", "next_row", "next2_row")
                        if name in tensors
                    },
                )
            )

        try:
            op(
                "gemm",
                dict(out=w.projection, x=w.action, bias=w.input_bias, w=self.weights.buffers["action.input"]),
                output_scale="quarter",
            )
            copy(w.projection, self.hidden_staging, "core0")
            copy(self.hidden_staging, w.residual, "broadcast")
            copy(rows[0], w.mod0, "broadcast", row_index=0)
            copy(rows[1], w.mod1, "broadcast", row_index=1)
            op(
                "rmsnorm",
                dict(out=w.normalized, x=w.residual, modulation=w.mod0),
                scalars=norm_scalars,
            )
            op(
                "residual_gate",
                dict(out=w.mixed, branch=w.normalized, residual=w.residual, gate=w.mod0_gate),
            )
            op(
                "rmsnorm",
                dict(out=w.normalized, x=w.mixed, modulation=w.mod1),
                scalars=norm_scalars,
            )
            for layer in range(24):
                if layer in FULL_ATTENTION_LAYERS:
                    copy(self.weights.layer(layer, "qk_norm"), w.qk_norm, "broadcast")
                    op(
                        "qkv_proj",
                        dict(
                            x=w.normalized,
                            planes=w.planes,
                            w=self.weights.layer(layer, "qkv"),
                            qk_weight=w.qk_norm,
                            cos=self.cosine,
                            sin=self.sine,
                        ),
                        integers={"position_offset": 0},
                        scalars={"epsilon": 1.0e-6},
                    )
                    cache_index = FULL_ATTENTION_LAYERS.index(layer)
                    key, value = self.destination_caches[cache_index]
                    op("kv_cache_insert", dict(input=w.key, cache=key), integers={"position": bucket})
                    op("kv_cache_insert", dict(input=w.value, cache=value), integers={"position": bucket})
                    op(
                        "attention",
                        dict(out=w.attention, query=w.query, key_cache=key, value_cache=value, mask=mask),
                        integers={"kv_length": bucket + 16},
                    )
                    op("sigmoid_mul", dict(out=w.attention, gate=w.attention_gate, value=w.attention))
                    down(
                        w.attention,
                        self.weights.layer(layer, "o"),
                        w.residual,
                        w.mixed,
                        w.normalized,
                        rows[2 * layer],
                        rows[2 * layer + 1],
                        layer=layer,
                        kind=0,
                    )
                op(
                    "gemm_glu",
                    dict(
                        x=w.normalized,
                        out=w.gate,
                        w_gate=self.weights.layer(layer, "gate"),
                        w_up=self.weights.layer(layer, "up"),
                    ),
                )
                next_layer = layer + 1
                if next_layer == 24:
                    down(
                        w.gate,
                        self.weights.layer(layer, "down"),
                        w.mixed,
                        w.decoder_scratch,
                        w.normalized,
                        rows[2 * layer + 1],
                        rows[48],
                        layer=layer,
                        kind=1,
                    )
                elif next_layer not in FULL_ATTENTION_LAYERS:
                    down(
                        w.gate,
                        self.weights.layer(layer, "down"),
                        w.mixed,
                        w.decoder_scratch,
                        w.mixed,
                        rows[2 * layer + 1],
                        rows[2 * next_layer],
                        next2_row=rows[2 * next_layer + 1],
                        w_out=w.normalized,
                        layer=layer,
                        kind=1,
                    )
                else:
                    down(
                        w.gate,
                        self.weights.layer(layer, "down"),
                        w.mixed,
                        w.residual,
                        w.normalized,
                        rows[2 * layer + 1],
                        rows[2 * next_layer],
                        layer=layer,
                        kind=1,
                    )
            op(
                "gemm",
                dict(out=w.velocity, x=w.normalized, bias=w.output_bias, w=self.weights.buffers["action.output"]),
            )
            op(
                "euler_update",
                dict(action=w.action, velocity=w.velocity, keep_mask=w.keep_mask, padding_velocity=w.padding_velocity),
                scalars={"delta_t": self.delta_t},
            )
            return operations, mutable_rows
        except BaseException as error:
            _release_all(reversed(operations), error)
            raise

    def _bucket(self, bucket):
        if bucket in self._plans:
            self.stats["bucket_replays"] += 1
            self._plans.move_to_end(bucket)
            return self._plans[bucket]
        evict = None
        if len(self._plans) >= self.max_cached_buckets:
            active_bucket = None if self._active is None else self._active.plan.bucket
            for previous, retained in tuple(self._plans.items()):
                if previous != active_bucket:
                    evict = previous, retained
                    break
            else:
                raise RuntimeError("all retained Action buckets belong to an active request")
        context, w = self.context, self.workspace
        views, copies, executions, operations = [], [], [], []
        prefix_operation = None
        try:
            shape = (16, bucket + 16)
            mask_ddr = context.view(self.mask_storage, shape)
            views.append(mask_ddr)
            mask_spm = context.view(w.mask, shape)
            views.append(mask_spm)
            for source, out, mode in (
                (self.action_input, w.action, "core0"),
                (self.padding_input, w.padding_velocity, "core0"),
                (self.keep_input, w.keep_mask, "core0"),
                (mask_ddr, mask_spm, "broadcast"),
            ):
                copies.append(context.prepare_copy(source, out, mode=mode))
            bindings = {}
            for index, (source, target) in enumerate(zip(self.source_caches, self.destination_caches)):
                for kind, src, dst in zip(("k", "v"), source, target):
                    bindings[f"src_{kind}{index}"] = src
                    bindings[f"dst_{kind}{index}"] = dst
            prefix_operation = context.prepare(
                "copy_prefix",
                tensors=bindings,
                integers={"prefix_len": bucket, "bucket_len": bucket, "capacity": 784},
            )
            node_counts = []
            step_ops_list = []
            for step in range(8):
                step_ops, _ = self._prepare_step(step, mask_spm, bucket)
                step_ops_list.append(step_ops)
                operations.extend(step_ops)
                node_counts.append(len(step_ops))
            entry = None
            if self.whole_stage:
                # Whole-stage vendor discipline with the entry fold:
                # the eight NFE steps AND the entry nodes — the physical
                # prefix copy plus the four staging copies at the queue
                # head, in their entry order — form ONE retained fused
                # queue; the original's action stage is a single SDK
                # segment (721C/1758D) with the prefix copy inside.
                steps = (context.batch_fused((prefix_operation, *copies, *operations)),)
                executions.extend(steps)
            elif self.fused_steps:
                # Keep the entry nodes as a plain
                # batch (physical prefix copy + staging copies) and pack
                # the eight NFE steps into ONE retained fused queue.
                steps = (context.batch_fused((*operations,)),)
                executions.extend(steps)
                entry = context.batch((prefix_operation, *copies))
                executions.append(entry)
            else:
                for step_ops in step_ops_list:
                    execution = (context.batch(step_ops) if self.collective_window == 1 else
                                 context.batch(step_ops, collective_window=self.collective_window))
                    executions.append(execution)
                steps = tuple(executions)
                entry = context.batch((prefix_operation, *copies))
                executions.append(entry)
            plan = _BucketPlan(
                bucket=bucket,
                mask_ddr=mask_ddr,
                mask_spm=mask_spm,
                entry_copies=tuple(copies),
                prefix_operation=prefix_operation,
                entry_execution=entry,
                execution=steps[0],
                steps=steps,
                operations=tuple(operations),
                views=tuple(views),
                node_counts=tuple(node_counts),
            )
            if evict is not None:
                previous, retired = evict
                # Prepare the replacement before committing LRU eviction.
                # If a new bucket fails lowering, the idle old plans survive;
                # only native/binding ownership still requires model rollback.
                retired.close()
                del self._plans[previous]
                self.stats["bucket_evictions"] += 1
            self._plans[bucket] = plan
            self.stats["bucket_builds"] += 1
            return plan
        except BaseException as error:
            if prefix_operation is not None:
                operations.append(prefix_operation)
            _release_all(
                (
                    *reversed(executions),
                    *reversed(operations),
                    *reversed(copies),
                    *reversed(views),
                ),
                error,
            )
            raise

    def bind(self, prepared: PreparedRequest | Mapping) -> DenoiseBinding:
        """Snapshot dynamic host inputs while idle; never advance device state."""
        self.context._check_idle()
        if self.closed or self.context.needs_reset or self.context.epoch != self._epoch:
            raise RuntimeError("Action lowering is closed or stale; rebuild it after recovery")
        if self._active is not None:
            raise RuntimeError("release the previous Action request binding before admission")
        tensors = prepared.tensors if isinstance(prepared, PreparedRequest) else prepared
        if tensors.get("num_steps", 8) != 8:
            raise ValueError("Action supports exactly NFE8")
        prefix = tensors["prefix_length"]
        bucket = tensors["prefix_bucket"]
        if (
            isinstance(prefix, bool)
            or not isinstance(prefix, int)
            or not 1 <= prefix <= 512
            or bucket != ((prefix + 63) // 64) * 64
        ):
            raise ValueError("Action requires real prefix1..512 and its ceil64 physical bucket")
        packed, keep, padding = pack_action_request(tensors)
        cosine, sine = action_rope_tables(tensors["action_position_ids"])
        mask = action_attention_mask(prefix, bucket)
        plan = self._bucket(bucket)
        self.action_input.copy_from(packed)
        self.padding_input.copy_from(padding)
        self.keep_input.copy_from(keep)
        self.cosine.copy_from(cosine)
        self.sine.copy_from(sine)
        plan.mask_ddr.copy_from(mask)
        if plan.entry_execution is not None:
            entry = plan.entry_execution.bind({plan.prefix_operation: {"integers": {"prefix_len": prefix}}})
        else:
            # M5 entry fold: rebind the prefix length on the fused queue that
            # now carries the prefix copy; the bound invocation is the stage.
            entry = plan.execution.bind({plan.prefix_operation: {"integers": {"prefix_len": prefix}}})
        self._active = DenoiseBinding(self, plan, entry)
        self.stats["requests_bound"] += 1
        return self._active

    def describe(self):
        return dict(
            num_steps=8,
            identity_layers=18,
            attention_layers=list(FULL_ATTENTION_LAYERS),
            collectives_per_step=30,
            collective_window=self.collective_window,
            whole_stage=self.whole_stage,
            collectives_per_request=240,
            cached_buckets=sorted(self._plans),
            max_cached_buckets=self.max_cached_buckets,
            step_nodes={str(bucket): list(plan.node_counts) for bucket, plan in self._plans.items()},
            retained_step_nodes={str(bucket): len(plan.operations) for bucket, plan in self._plans.items()},
            retained_step_batches=sum(len(plan.steps) for plan in self._plans.values()),
            workspace_bytes_per_core=self.workspace.arena_bytes + self.workspace.single_bytes,
            entry_rebuilt_per_request=False,
            stats=dict(self.stats),
        )

    def close(self):
        if self.closed:
            return
        error = None
        try:
            self.context.wait()
        except BaseException as caught:
            error = caught
        resources = []
        if self._active is not None:
            resources.append(self._active)
        resources.extend(reversed(tuple(self._plans.values())))
        resources.extend(reversed(self._owners))
        if self._owns_workspace and self.workspace is not None:
            resources.append(self.workspace)
        error, failed = _release_all(resources, error)
        if not failed:
            self._plans.clear()
            self._owners.clear()
            self.closed = True
        if error is not None:
            raise error
