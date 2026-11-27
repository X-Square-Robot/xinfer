# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Fixed TP8 resident tensors and explicit, reusable ordered batches.

The calling framework selects operations, tensor placement and dependency
order. This module owns the device memory and SDK execution resources for
that supplied sequence. All calls use the existing AOT xinfer_rpu.so TVM FFI.
"""


import operator

from ._native import lib, require_runtime_assets
from .rpu import _Resource, _writable_dlpack_input


def _dims(shape):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if not 1 <= len(shape) <= 6 or any(isinstance(x, bool) for x in shape):
        raise ValueError("TP shape must have rank 1..6 and integer dimensions")
    result = tuple(operator.index(x) for x in shape)
    if any(x <= 0 for x in result):
        raise ValueError("TP dimensions must be positive")
    return result


# Family entry points of RpuTpContext.prepare() and their string selectors.
# The native resolver (src/common/rpu/tp_family.cc) reads the integer codes;
# catalog entry names (the compatibility names) take no selectors.
FAMILIES = (
    "gemm",
    "qkv_proj",
    "gemm_glu",
    "bmm",
    "rmsnorm",
    "layernorm",
    "l2norm",
    "qk_norm_rope",
    "silu_and_mul",
    "sigmoid_mul",
    "activation",
    "rope",
    "attention",
    "kv_cache_insert",
    "copy_prefix",
    "expand_kv",
    "all_reduce",
    "all_gather",
    "down_allreduce_norm",
    "residual_gate",
    "euler_update",
    "binary",
    "broadcast",
    "scale",
    "fill",
    "embedding",
    "causal_conv1d",
    "gated_delta_rule_prefill",
    "cumsum",
    "unit_tril_inverse",
    "slice_last",
    "transpose",
)
_SELECTORS = {
    "parallel": {"column": 0, "row": 1},
    "activation": {"none": 0, "gelu": 1, "silu": 2},
    "mode": {"default": 0, "coefficient": 1, "gated": 2},
    "function": {
        "gelu_tanh": 0,
        "gelu_erf": 1,
        "sigmoid": 2,
        "exp": 3,
        "silu": 4,
        "softplus": 5,
    },
    "operation": {"add": 0, "sub": 1, "mul": 2, "max": 3},
    # Core0 input projection epilogue: round_fp16(round_fp16(x @ w.T + b) / 4).
    "output_scale": {"none": 0, "quarter": 1},
}
# Which family accepts which string selector (and whether it is required).
_FAMILY_SELECTORS = {
    "gemm": {"parallel": False, "activation": False, "output_scale": False},
    "gemm_glu": {"activation": False},
    "rmsnorm": {"mode": False},
    "activation": {"function": True},
    "binary": {"operation": True},
}


def _family_integers(op, selectors, integers):
    """Encode string selectors of a family call into native integer selectors."""
    accepted = _FAMILY_SELECTORS.get(op, {})
    for name in selectors:
        if name not in accepted:
            raise TypeError(f"{op}() got an unexpected selector {name!r}")
    for name, required in accepted.items():
        if required and name not in selectors:
            raise TypeError(f"{op}() requires selector {name!r}")
    encoded = dict(integers)
    for name, value in selectors.items():
        table = _SELECTORS[name]
        if value not in table:
            raise ValueError(
                f"{op}(): {name} must be one of {', '.join(table)}, got {value!r}"
            )
        if name in encoded:
            raise ValueError(f"{op}(): {name} given both as selector and integer")
        encoded[name] = table[value]
    return encoded


def resolve(op, tensors, integers=None, **selectors):
    """Resolve a family call to its catalog entry without a device.

    ``tensors`` maps argument names to tensor descriptions: any object with
    ``shape``, ``layout`` and ``distribution`` attributes (an RpuTpTensor or a
    framework's planning stand-in). Returns ``(entry, names, integers)``:
    the catalog entry, a dict from each given argument name to the catalog
    argument name, and the final integer arguments the device receives
    (selectors consumed). A catalog name resolves to itself. This is the
    TP8 counterpart of the CUDA families' ``selected_backend``; it needs only
    the host resolver, which both host-only and runtime builds export.
    """
    integers = {} if integers is None else dict(integers)
    if op in FAMILIES:
        integers = _family_integers(op, selectors, integers)
    elif selectors:
        raise TypeError(
            f"selectors {sorted(selectors)} apply to family entry points only; "
            f"{op!r} is a catalog entry"
        )
    names = list(tensors)
    values = [
        str(v)
        for v in lib.rpu_tp_resolve(
            op,
            names,
            [[operator.index(d) for d in tensors[n].shape] for n in names],
            [str(tensors[n].layout) for n in names],
            [str(tensors[n].distribution) for n in names],
            list(integers),
            [operator.index(v) for v in integers.values()],
        )
    ]
    entry, count = values[0], int(values[1])
    renamed = dict(zip(names, values[2 : 2 + count]))
    rest = values[2 + count :]
    pairs = int(rest[0])
    resolved = {rest[1 + 2 * i]: int(rest[2 + 2 * i]) for i in range(pairs)}
    return entry, renamed, resolved


class RpuTpTensor(_Resource):
    """Owned logical tensor with explicit storage, layout and TP identity.

    SPM shape describes each core; numpy() returns [8,*shape]. DDR shape
    describes the complete shared allocation. Cache transfers convert between
    the logical [8,physical_capacity,channels] shape and device tile layout.
    Packed tp8_nk16/nk16 weights are execution-only: shape records the logical
    preparation contract, and copy_to()/numpy() reject host downloads.
    Immutable row_major tables remain downloadable.
    reset() zeroes a state owner and invalidates plans capturing its generation.
    """

    def __init__(
        self,
        handle,
        context,
        shape,
        dtype,
        storage,
        layout,
        distribution,
        shard_axis=-1,
        state_kind="none",
        immutable=False,
    ):
        super().__init__(handle)

        self.context = context
        self.shape = tuple(shape)
        self.dtype = dtype
        self.storage = storage
        self.layout = layout
        self.distribution = distribution
        self.shard_axis = shard_axis
        self.state_kind = state_kind
        self.immutable = immutable

    def copy_from(self, host):
        lib.rpu_tp_upload(self._handle, _writable_dlpack_input(host))
        return self

    def _require_downloadable(self):
        if self.layout in {"tp8_nk16", "nk16"}:
            raise ValueError(
                f"{self.layout} weights are execution-only; "
                "packed download is unsupported"
            )

    def copy_to(self, host):
        self._require_downloadable()
        lib.rpu_tp_download(host, self._handle)
        return host

    def numpy(self):
        self._require_downloadable()
        import numpy as np

        dims = (8, *self.shape) if self.storage == "spm" else self.shape
        return self.copy_to(np.empty(dims, dtype=self.dtype))

    def reset(self):
        lib.rpu_tp_reset(self._handle)
        return self

    def invalidate(self):
        """Discard logical state validity, retaining storage and plan generation.

        No bytes are cleared. A subsequent initial producer must overwrite
        every interval before any reader. Explicit reset still clears bytes
        and invalidates prepared generations after an error or user reset.
        """
        lib.rpu_tp_invalidate(self._handle)
        return self


class RpuTpOperation(_Resource):
    """Validated operator binding; append to context.batch in caller order.

    ``op`` is the name passed to prepare(). ``kernel`` is the device kernel
    that will run and ``admission`` its catalog admission; ``entry`` is the
    catalog entry the request resolved to and ``family`` the family entry
    point (``None`` for a catalog name). This is the TP8 counterpart of the
    CUDA families' ``selected_backend``.
    """

    def __init__(self, handle, context, op):
        super().__init__(handle)
        self.context = context
        self.op = op
        kernel, entry, family, admission = (
            str(v) for v in lib.rpu_tp_selection(handle)
        )
        self.kernel = kernel
        self.entry = entry or op
        self.family = family or None
        self.admission = admission or "default"

    def rebind(self, *, tensors=None, integers=None, scalars=None):
        """Atomically validate a parameter patch without rebuilding SDK queues.

        Entry, grid, write structure and captured state owners stay fixed.
        The next execution synchronizes mutable kernel/DMA parameters while
        the device is idle. A rejected patch leaves this operation unchanged.
        """
        tensors, integers, scalars = tensors or {}, integers or {}, scalars or {}
        for name, tensor in tensors.items():
            if (
                not isinstance(tensor, RpuTpTensor)
                or tensor.context is not self.context
            ):
                raise ValueError(f"{name} must belong to this TP context")
        lib.rpu_tp_rebind(
            self._handle,
            list(tensors),
            [t._handle for t in tensors.values()],
            list(integers),
            [operator.index(i) for i in integers.values()],
            list(scalars),
            [float(s) for s in scalars.values()],
        )
        return self


class RpuTpExecution(_Resource):
    """One logical execution retaining its prepared SDK segments and owners."""

    def execute(self, *, async_=False):
        if not isinstance(async_, bool):
            raise TypeError("async_ must be bool")
        lib.rpu_tp_execute(self._handle, async_)
        return self

    def wait(self):
        lib.rpu_tp_wait(self._handle)
        return self

    def describe(self):
        names = ("nodes", "sdk_kernels", "segments", "dmas", "barriers", "collectives")
        return dict(zip(names, lib.rpu_tp_batch_stats(self._handle)))

    def close(self):
        if self._id:
            try:
                self.wait()
            finally:
                super().close()


class RpuTpContext(_Resource):
    """Fixed physical cores 0..7 with joint SPM allocation and explicit batches.

    A context reserves a 7 MiB arena per core on its first SPM allocation.
    The calling framework remains responsible for the model graph and TP
    shard contents.
    Unsupported parameters fail in native validation, including direct FFI.
    """

    def __init__(self, *, allow_candidate=False, retained_segments=128):
        if not isinstance(allow_candidate, bool):
            raise TypeError("allow_candidate must be bool")
        if isinstance(retained_segments, bool):
            raise TypeError("retained_segments must be an integer")
        retained_segments = operator.index(retained_segments)
        if (
            retained_segments < 128
            or retained_segments > 8192
            or retained_segments % 64
        ):
            raise ValueError("retained_segments must be a multiple of 64 in [128,8192]")
        assets = require_runtime_assets()
        handle = (
            lib.rpu_tp_context_create(allow_candidate)
            if retained_segments == 128
            else lib.rpu_tp_context_create_with_pool(
                allow_candidate, retained_segments
            )
        )
        super().__init__(handle)
        self._asset_provenance = dict(assets)

    @property
    def asset_provenance(self):
        """Identity of the external asset manifest validated before context creation."""
        return dict(self._asset_provenance)

    def memory_stats(self):
        names = (
            "spm_arena_bytes_per_core",
            "spm_live_bytes_per_core",
            "spm_peak_bytes_per_core",
            "spm_largest_free_bytes_per_core",
        )
        return dict(
            zip(names, map(int, lib.rpu_tp_context_stats(self._handle)))
        )

    def pool_stats(self):
        names = (
            "capacity",
            "current",
            "peak",
            "graph_retained",
            "ephemeral",
            "capacity_bytes",
            "current_bytes",
            "peak_bytes",
            "graph_retained_bytes",
            "ephemeral_bytes",
        )
        return dict(zip(names, map(int, lib.rpu_tp_pool_stats(self._handle))))

    def empty(
        self,
        shape,
        *,
        dtype="float16",
        storage="spm",
        layout="row_major",
        distribution="local",
        shard_axis=-1,
        state_kind="none",
    ):
        shape = _dims(shape)
        shard_axis = operator.index(shard_axis)
        handle = lib.rpu_tp_empty(
            self._handle,
            shape,
            dtype,
            storage,
            layout,
            distribution,
            shard_axis,
            state_kind,
        )
        return RpuTpTensor(
            handle,
            self,
            shape,
            dtype,
            storage,
            layout,
            distribution,
            shard_axis,
            state_kind,
        )

    def prepare_weight(self, host, *, layout="tp8_nk16"):
        """Snapshot FP16 logical weights and pack once; never repack on replay.

        tp8_nk16 takes [8,N,K] or [planes,8,N,K]; nk16 takes [N,K].
        These packed weights are execution-only and reject copy_to()/numpy();
        their shape preserves the logical preparation contract.
        row_major creates downloadable immutable tables without GEMM packing.
        """
        handle = lib.rpu_tp_weight(
            self._handle, _writable_dlpack_input(host), layout
        )
        return RpuTpTensor(
            handle, self, host.shape, "float16", "ddr", layout, "shared", immutable=True
        )

    def view(self, parent, shape, *, offset_bytes=0, distribution=None, shard_axis=-1):
        """Retain a contiguous row-major region without allocating device memory.

        Aliases share native upload/device-write proof invalidation. Packed
        weights and state/cache owners cannot be viewed. A local scratch arena
        may be assigned explicit model-owned TP roles; other roles are preserved.
        """
        if not isinstance(parent, RpuTpTensor) or parent.context is not self:
            raise ValueError("view parent must belong to this TP context")
        shape = _dims(shape)
        distribution = parent.distribution if distribution is None else distribution
        offset_bytes, shard_axis = operator.index(offset_bytes), operator.index(
            shard_axis
        )
        handle = lib.rpu_tp_view(
            parent._handle, shape, offset_bytes, distribution, shard_axis
        )
        view = RpuTpTensor(
            handle,
            self,
            shape,
            parent.dtype,
            parent.storage,
            "row_major",
            distribution,
            shard_axis,
            immutable=parent.immutable,
        )
        view._owner = parent
        return view

    def prepare_copy(self, input, out, *, mode="corewise"):
        """Prepare a device DMA with explicit per-core/broadcast/core0 semantics."""
        modes = {"corewise": 0, "broadcast": 1, "core0": 2, "shared": 3}
        if mode not in modes:
            raise ValueError(f"unsupported RPU copy mode: {mode}")
        return self.prepare(
            "copy_f16",
            tensors={"input": input, "out": out},
            integers={"mode": modes[mode]},
        )

    def prepare(self, op, *, tensors, integers=None, scalars=None, **selectors):
        """Validate and bind one operation; returns an RpuTpOperation.

        ``op`` is a family entry point (see ``FAMILIES``: ``gemm``,
        ``qkv_proj``, ``rmsnorm``, ``attention`` ...) with semantic argument
        names, or a catalog entry name (compatibility). A family call selects
        the kernel from tensor shapes and explicit selectors, e.g.
        ``prepare("gemm", tensors={"x": x, "w": w, "out": y}, parallel="row")``.
        Selectors that change numerics are never inferred. The selected kernel
        is reported by ``operation.kernel`` and its admission is checked here.
        """
        integers = {} if integers is None else dict(integers)
        scalars = {} if scalars is None else dict(scalars)
        if op in FAMILIES:
            integers = _family_integers(op, selectors, integers)
        elif selectors:
            raise TypeError(
                f"selectors {sorted(selectors)} apply to family entry points only; "
                f"{op!r} is a catalog entry"
            )
        names, handles = [], []
        for name, tensor in tensors.items():
            if not isinstance(tensor, RpuTpTensor) or tensor.context is not self:
                raise ValueError(f"{name} must be a tensor from this TP context")
            names.append(name)
            handles.append(tensor._handle)
        int_values = [operator.index(x) for x in integers.values()]
        scalar_values = [float(x) for x in scalars.values()]
        handle = lib.rpu_tp_prepare(
            self._handle,
            op,
            names,
            handles,
            list(integers),
            int_values,
            list(scalars),
            scalar_values,
        )
        return RpuTpOperation(handle, self, op)

    @staticmethod
    def families():
        """Family entry points accepted by prepare(), with their selectors."""
        return {
            family: {
                name: tuple(_SELECTORS[name])
                for name in _FAMILY_SELECTORS.get(family, {})
            }
            for family in FAMILIES
        }

    def ar_collective_window_max(self):
        """Query calibrated AR protocol; does not initialize a new device."""
        _ = self._handle
        query = lib.get_function("rpu_tp_ar_window_max")
        if query is None:
            raise RuntimeError(
                "backend lacks calibrated AR collective-window capability"
            )
        return int(query())

    def down_norm_collective_window_max(self):
        """Query the calibrated M16 DownNorm window; no device initialization."""
        _ = self._handle
        query = lib.get_function("rpu_tp_down_norm_window_max")
        if query is None:
            # Compatibility with libraries built before the semantic rename.
            query = lib.get_function("rpu_tp_action_m16_window_max")
        if query is None:
            raise RuntimeError(
                "backend lacks calibrated M16 DownNorm collective-window capability"
            )
        return int(query())

    action_m16_collective_window_max = down_norm_collective_window_max

    def batch(self, operations, *, collective_window=1):
        """Retain explicit operations; experimental windows require candidate admission.

        The default keeps a separate completed segment for each collective.
        AR windows2..15 (M1 single-window boundary) or M16 DownNorm windows2..4
        use separate counter owners and a completed initialization segment.
        A batch cannot mix these two families; other collective families retain
        hard boundaries.
        """
        operations = list(operations)
        if any(
            not isinstance(op, RpuTpOperation) or op.context is not self
            for op in operations
        ):
            raise ValueError("all operations must belong to this TP context")
        collective_window = operator.index(collective_window)
        if not 1 <= collective_window <= 15:
            raise ValueError("collective_window must be in [1,15]")
        if collective_window > 1:
            entries = {op.entry for op in operations}
            ar = "all_reduce_residual" in entries
            m16 = "down_allreduce_rmsnorm_m16n1024k256_acc32" in entries
            if ar and m16:
                raise RuntimeError(
                    "collective windows cannot mix AR and M16 DownNorm in one batch"
                )
            if ar and self.ar_collective_window_max() < collective_window:
                raise RuntimeError(
                    "backend lacks requested calibrated AR collective-window capability"
                )
            if m16 and self.down_norm_collective_window_max() < collective_window:
                raise RuntimeError(
                    "backend lacks requested calibrated M16 DownNorm collective-window capability"
                )
        handles = [op._handle for op in operations]
        handle = (
            lib.rpu_tp_batch(self._handle, handles)
            if collective_window == 1
            else lib.rpu_tp_batch_with_window(
                self._handle, handles, collective_window
            )
        )
        return RpuTpExecution(handle)

    def batch_fused(self, operations):
        """Retain explicit operations as ONE whole-stage explicitly-ordered queue.

        Counted collectives support M16 DownNorm (request counter 3..49) and
        AR/ARR (counter 3..50), with distinct consecutive ids and graph ordering.
        All other counter writes are limited to transient 0/1. Counterless
        owner-push all_gather is admitted only immediately before its DMA burst.
        No isolation segments or global counter initialization are inserted.
        """
        operations = list(operations)
        if any(
            not isinstance(op, RpuTpOperation) or op.context is not self
            for op in operations
        ):
            raise ValueError("all operations must belong to this TP context")
        fused = lib.get_function("rpu_tp_batch_fused")
        if fused is None:
            raise RuntimeError("backend lacks the fused whole-stage batch capability")
        handles = [op._handle for op in operations]
        return RpuTpExecution(fused(self._handle, handles))
