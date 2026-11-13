"""The RPU device's platform: fixed TP8 execution contract and completion barriers.

RPU addresses are owned by the vendor SDK, not torch's device allocator: resolving ``rpu[:0]`` is
board-free and never imports torch. Buffer and completion ownership over xcompute's public TP ABI
live here; the engine never sees vendor primitives. The per-device execution model
(``execution_model="explicit-completion"``) routes ownership of request admission and completion
waits to ``xinfer/engine/rpu.py`` through :meth:`xinfer.platforms.Platform.create_engine`.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import weakref
from collections import Counter
from pathlib import Path

import numpy as np


def rpu_platform(device: str):
    """Describe the fixed single-device TP8 execution contract, without loading SDKs.

    Resolution is also usable for board-free admission. Native availability and
    candidate ABI admission are checked when the execution context is created.
    An architecture match alone is not an operator support claim.

    RPU execution carries its own kernel bindings (no torch op table) and waits
    on explicit submission completion (no torch stream/event), declared through
    the platform capabilities.
    """
    from xinfer.platforms.interface import Platform, PlatformCapabilities

    if device not in {"rpu", "rpu:0"}:
        raise ValueError("the first RPU profile supports only device rpu:0")
    return Platform(
        name="rpu-tp8", device="rpu:0",
        capabilities=PlatformCapabilities(
            device_type="rpu", arch="rpu-tp8", cudagraphs=False,
            stream_priorities=False, unified_memory=False,
            multiprocess_service=False, sm_count=None, total_memory_bytes=None,
            execution_model="explicit-completion", op_selection="external",
        ),
    )


class RpuBusyError(RuntimeError):
    """Device access attempted before an outstanding completion was waited."""


class RpuCompletion:
    """One submission, retaining its execution and all buffer owners until wait."""

    def __init__(self, execution):
        self.execution = execution
        self.done = False
        self.error = None

    def wait(self):
        context = self.execution.context
        with context._lock:
            if not self.done:
                try:
                    self.execution.native.wait()
                except BaseException as error:
                    self.error = error
                    context.needs_reset = True
                finally:
                    self.done = True
                    context._pending = None
                    context.stats["completion_waits"] += 1
            if self.error is not None:
                raise self.error
        return self


class _Resource:
    def __init__(self, context, native):
        self.context = context
        self.native = native
        self.closed = False
        context._resources.append(self)

    def _check(self):
        self.context._check_idle()
        if self.closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def close(self):
        with self.context._lock:
            if self.closed:
                return
            self._check()
            try:
                self.native.close()
            except BaseException:
                if getattr(self.native, "_id", None) == 0:
                    self.closed = True
                    self.context._resources.remove(self)
                raise
            self.closed = True
            self.context._resources.remove(self)

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()


class RpuBuffer(_Resource):
    """Stable native owner, or an explicit context-created contiguous view."""

    def __init__(self, context, native, spm_bytes=0):
        super().__init__(context, native)
        self.shape = tuple(native.shape)
        self.dtype = np.dtype(native.dtype)
        self.storage = native.storage
        self.layout = native.layout
        self.distribution = native.distribution
        self.shard_axis = getattr(native, "shard_axis", -1)
        self.state_kind = native.state_kind
        self.immutable = native.immutable
        self._spm_bytes = spm_bytes

    def copy_from(self, host):
        with self.context._lock:
            self._check()
            if self.immutable:
                raise ValueError("prepared weights are immutable")
            if not isinstance(host, np.ndarray) or not host.flags.c_contiguous:
                raise TypeError("RPU upload requires a contiguous NumPy array")
            shapes = {self.shape}
            if self.storage == "spm":
                shapes.add((8, *self.shape))
            if host.dtype != self.dtype or host.shape not in shapes:
                raise ValueError(f"upload requires dtype {self.dtype} and shape in {shapes}")
            self.native.copy_from(host)
            self.context.stats["uploads"] += 1
            self.context.stats["upload_bytes"] += host.nbytes
        return self

    def numpy(self):
        with self.context._lock:
            self._check()
            result = self.native.numpy()
            self.context.stats["downloads"] += 1
            self.context.stats["download_bytes"] += result.nbytes
            return result

    def reset(self):
        with self.context._lock:
            self._check()
            if self.state_kind == "none":
                raise ValueError("reset requires an explicit cache/history/state owner")
            self.native.reset()
            self.context.stats["state_resets"] += 1
        return self

    def invalidate(self):
        with self.context._lock:
            self._check()
            if self.state_kind == "none":
                raise ValueError("invalidate requires a state/cache/history owner")
            self.native.invalidate()
            self.context.stats["state_invalidations"] += 1
        return self


class RpuOperation(_Resource):
    def __init__(self, context, native, tensors, logical_op=None):
        super().__init__(context, native)
        self.logical_op = logical_op
        # The catalog entry the operation resolved to. Family calls ("gemm",
        # "all_reduce", ...) resolve natively; backends without the selection
        # query keep the name the operation was prepared with.
        self.entry = getattr(native, "entry", None) or logical_op
        self.kernel = getattr(native, "kernel", None) or self.entry
        self.tensor_bindings = dict(tensors)
        self.tensors = tuple(self.tensor_bindings.values())
        self.epoch = context.epoch

    def rebind(self, *, tensors=None, integers=None, scalars=None):
        with self.context._lock:
            self._check()
            if self.context.needs_reset or self.epoch != self.context.epoch:
                raise RuntimeError("RPU operation is stale; rebuild after recovery")
            for name, tensor in (tensors or {}).items():
                if not isinstance(tensor, RpuBuffer) or tensor.context is not self.context:
                    raise ValueError(f"{name} must belong to this RPU context")
                tensor._check()
            self.native.rebind(tensors={k: v.native for k, v in (tensors or {}).items()},
                               integers=integers, scalars=scalars)
            self.tensor_bindings.update(tensors or {})
            self.tensors = tuple(self.tensor_bindings.values())
            self.context.stats["operation_rebindings"] += 1
        return self


class RpuExecution(_Resource):
    """Prepared batch. Every replay returns a distinct real completion owner."""

    def __init__(self, context, native, operations):
        super().__init__(context, native)
        self.operations = tuple(operations)
        self.epoch = context.epoch

    def execute(self):
        context = self.context
        with context._lock:
            self._check()
            if context.needs_reset or self.epoch != context.epoch:
                raise RuntimeError("RPU execution is stale; recover and rebuild its bindings")
            completion = RpuCompletion(self)
            try:
                # This is the public native worker/future path, not a CPU
                # event pretending device completion has already happened.
                self.native.execute(async_=True)
            except BaseException:
                context.needs_reset = True
                raise
            context._pending = completion
            context.stats["logical_submissions"] += 1
            return completion

    def bind(self, updates):
        with self.context._lock:
            self._check()
            for operation in updates:
                if not isinstance(operation, RpuOperation) or operation not in self.operations:
                    raise ValueError("bindings must target operations retained by this execution")
                operation._check()
            return RpuBoundExecution(self, updates)


class RpuBoundExecution(RpuExecution):
    """An invocation retaining a shared batch and fixed parameter updates.

    It allocates no SDK queue. Closing it releases this invocation, while the
    prepared execution remains owned by its plan. Updates occur under the same
    device lock as submission; a failed patch submits no work and requires
    explicit recovery before the partially rebound plan can be reused.
    """

    def __init__(self, execution, updates):
        _Resource.__init__(self, execution.context, execution.native)
        self.owner = execution
        self.operations, self.epoch = execution.operations, execution.epoch
        self.updates = {op: {name: dict(values) for name, values in patch.items()}
                        for op, patch in updates.items()}

    def execute(self):
        with self.context._lock:
            self._check()
            self.owner._check()
            try:
                for operation, patch in self.updates.items():
                    operation.rebind(**patch)
                return self.owner.execute()
            except BaseException:
                self.context.needs_reset = True
                raise

    def close(self):
        with self.context._lock:
            if self.closed:
                return
            self._check()
            self.closed = True
            self.context._resources.remove(self)


class RpuExecutionContext:
    """One process-owned device context for physical cores 0..7.

    ``recover`` is an explicit failure/reset boundary: it resets state owners,
    advances an epoch, and requires model plans to be rebuilt. It is not called
    between normal requests. SPM reservations are conservatively retained until
    context close: a native execution can retain a closed buffer handle. Native
    accounting includes retained completion markers and mathematical tables;
    SDK queue DDR is reported separately.
    """

    _owner_lock = threading.Lock()
    _owner = None

    def __init__(self, *, allow_candidate=False, retained_segments=128,
                 file_descriptor_budget=None, backend_factory=None):
        if not isinstance(allow_candidate, bool):
            raise TypeError("allow_candidate must be bool")
        self._lock = threading.RLock()
        self._resources = []
        self._pending = None
        self.closed = False
        self.needs_reset = False
        self.epoch = 0
        self.stats = Counter()
        if isinstance(retained_segments, bool) or not isinstance(retained_segments, int):
            raise TypeError("retained_segments must be an integer")
        if not 128 <= retained_segments <= 8192 or retained_segments % 64:
            raise ValueError("retained_segments must be a multiple of 64 in [128,8192]")
        self.provenance = {"adapter": "xinfer-rpu-tp-v1", "allow_candidate": allow_candidate,
                           "retained_segments": retained_segments}
        with self._owner_lock:
            owner = None if self._owner is None else self._owner()
            if owner is not None and not owner.closed:
                raise RpuBusyError("one live RPU execution context is permitted per process")
            if file_descriptor_budget is not None:
                self.require_file_descriptor_budget(file_descriptor_budget)
            if backend_factory is None:
                selected = os.environ.get("XCOMPUTE_LIBRARY_PATH")
                if not selected:
                    raise RuntimeError("RPU execution requires explicit XCOMPUTE_LIBRARY_PATH; "
                                       "build/select the AOT library before starting xinfer")
                selected = Path(selected).resolve(strict=True)
                import xcompute
                from xcompute.rpu import RpuTpContext

                loaded = Path(xcompute.KERNEL_LIB_PATH).resolve(strict=True)
                if loaded != selected:
                    raise RuntimeError(f"xcompute already loaded {loaded}, expected {selected}")
                if "rpu" not in xcompute.compiled_backends():
                    raise RuntimeError(f"selected xcompute library does not contain RPU: {loaded}")
                for symbol in ("rpu_tp_context_create", "rpu_tp_empty", "rpu_tp_weight", "rpu_tp_reset",
                               "rpu_tp_upload", "rpu_tp_download", "rpu_tp_prepare", "rpu_tp_batch",
                               "rpu_tp_execute", "rpu_tp_wait", "rpu_tp_view",
                               "rpu_tp_context_create_with_pool", "rpu_tp_batch_stats",
                               "rpu_tp_context_stats", "rpu_tp_pool_stats", "rpu_tp_rebind",
                               "rpu_tp_invalidate", "rpu_release"):
                    if xcompute.xcompute_lib.get_function(symbol) is None:
                        raise RuntimeError(f"selected xcompute library is missing {symbol}")
                digest = hashlib.sha256()
                with loaded.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                self.provenance.update(library=str(loaded), library_sha256=digest.hexdigest(),
                                       compiled_backends=list(xcompute.compiled_backends()))
                backend_factory = RpuTpContext
            options = dict(allow_candidate=allow_candidate)
            if retained_segments != 128:
                options["retained_segments"] = retained_segments
            self.native = backend_factory(**options)
            type(self)._owner = weakref.ref(self)

    def require_file_descriptor_budget(self, minimum):
        """Admit a finite process-local owner budget before SDK allocations.

        SDK buffers and retained graph arenas own DMA-BUF descriptors. Raising
        the soft limit within the existing hard limit changes this process,
        never the machine configuration. The applied limits are recorded.
        """
        import resource

        with self._lock:
            self._check_idle()
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
                raise ValueError("file descriptor budget must be a positive integer")
            before, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if hard != resource.RLIM_INFINITY and minimum > hard:
                raise RuntimeError(f"RPU needs {minimum} file descriptors; process hard limit is {hard}")
            applied = before if before == resource.RLIM_INFINITY else max(before, minimum)
            if applied != before:
                resource.setrlimit(resource.RLIMIT_NOFILE, (applied, hard))
            previous = self.provenance.get("file_descriptors", {})
            self.provenance["file_descriptors"] = dict(
                required=max(minimum, previous.get("required", 0)),
                soft_before=previous.get("soft_before", before), soft_applied=applied, hard=hard)

    def _check_idle(self):
        if self.closed:
            raise RuntimeError("RPU execution context is closed")
        if self._pending is not None:
            raise RpuBusyError("wait for the RPU completion before device access")

    def describe(self):
        pool = None
        with self._lock:
            if not self.closed and self._pending is None:
                self._refresh_spm()
                pool_stats = getattr(self.native, "pool_stats", None)
                if callable(pool_stats):
                    pool = pool_stats()
        return dict(
            provenance=dict(self.provenance), stats=dict(self.stats), epoch=self.epoch,
            closed=self.closed, needs_reset=self.needs_reset,
            limits=dict(tp=8, batch_size=1, max_inflight=1, max_ops_per_batch=1024,
                        max_collectives_per_batch=48, spm_arena_bytes_per_core=7 * 1024 * 1024,
                        nonblocking_completion_query=False, mutable_bindings=True),
            spm_accounting="native allocator includes completion markers and immutable mathematical tables",
            sdk_graph_arena_pool=pool,
        )

    def empty(self, shape, **description):
        with self._lock:
            self._check_idle()
            self._refresh_spm()
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            dtype = np.dtype(description.get("dtype", "float16"))
            spm_bytes = 0
            if description.get("storage", "spm") == "spm":
                spm_bytes = math.ceil(math.prod(shape) * dtype.itemsize / 256) * 256
                if self.stats["spm_live_bytes_per_core"] + spm_bytes > 7 * 1024 * 1024:
                    raise MemoryError("RPU per-core 7 MiB SPM budget exceeded")
            native = self.native.empty(shape, **description)
            buffer = RpuBuffer(self, native, spm_bytes)
            self.stats["allocations"] += 1
            self.stats["spm_live_bytes_per_core"] += spm_bytes
            self.stats["spm_peak_bytes_per_core"] = max(
                self.stats["spm_peak_bytes_per_core"], self.stats["spm_live_bytes_per_core"])
            self._refresh_spm()
            return buffer

    def _refresh_spm(self):
        memory_stats = getattr(self.native, "memory_stats", None)
        if callable(memory_stats):
            values = memory_stats()
            for key in ("spm_live_bytes_per_core", "spm_peak_bytes_per_core",
                        "sigmoid_lut_bytes_per_core", "rsqrt_lut_bytes_per_core"):
                if key in values:
                    self.stats[key] = values[key]

    def prepare_weight(self, host, *, layout="tp8_nk16"):
        with self._lock:
            self._check_idle()
            if not isinstance(host, np.ndarray) or host.dtype != np.float16 or not host.flags.c_contiguous:
                raise TypeError("weight preparation requires contiguous FP16 NumPy data")
            buffer = RpuBuffer(self, self.native.prepare_weight(host, layout=layout))
            self.stats["weight_preparations"] += 1
            self.stats["weight_host_bytes"] += host.nbytes
            return buffer

    def view(self, parent, shape, *, offset_bytes=0, distribution=None, shard_axis=-1):
        with self._lock:
            self._check_idle()
            if not isinstance(parent, RpuBuffer) or parent.context is not self:
                raise ValueError("view parent must belong to this context")
            parent._check()
            native = self.native.view(parent.native, shape, offset_bytes=offset_bytes,
                                      distribution=distribution, shard_axis=shard_axis)
            buffer = RpuBuffer(self, native)
            buffer._owner = parent
            self.stats["views"] += 1
            return buffer

    def prepare_copy(self, input, out, *, mode="corewise"):
        modes = {"corewise": 0, "broadcast": 1, "core0": 2, "shared": 3}
        if mode not in modes:
            raise ValueError(f"unsupported RPU copy mode: {mode}")
        return self.prepare("copy_f16", tensors={"input": input, "out": out},
                            integers={"mode": modes[mode]})

    def prepare(self, op, *, tensors, integers=None, scalars=None, **selectors):
        """Prepare a family entry point (gemm, rmsnorm, ...) or a catalog entry.

        Selectors (parallel=, activation=, mode=, function=, operation=) pass
        through to xcompute's family resolution unchanged.
        """
        with self._lock:
            self._check_idle()
            for name, buffer in tensors.items():
                if not isinstance(buffer, RpuBuffer) or buffer.context is not self:
                    raise ValueError(f"{name} must be an RPU buffer from this context")
                buffer._check()
            native = self.native.prepare(op, tensors={k: v.native for k, v in tensors.items()},
                                         integers=integers, scalars=scalars, **selectors)
            self.stats["operation_preparations"] += 1
            return RpuOperation(self, native, tensors, logical_op=op)

    def require_collective_window(self, collective_window, *, family="AR"):
        """Admit one calibrated family before model-plan allocations."""
        with self._lock:
            self._check_idle()
            if (isinstance(collective_window, bool) or not isinstance(collective_window, int)
                    or not 1 <= collective_window <= 15):
                raise ValueError("collective_window must be an integer in [1,15]")
            if collective_window == 1:
                return 1
            if not self.provenance["allow_candidate"]:
                raise RuntimeError("collective windows require explicit candidate admission")
            families = {
                "AR": ("ar_collective_window_max", 15, "all_reduce_residual"),
                "M16": ("action_m16_collective_window_max", 4,
                        "down_allreduce_rmsnorm_m16n1024k256_acc32"),
            }
            if family not in families:
                raise ValueError("unqualified collective-window family")
            method, maximum, eligible = families[family]
            query = getattr(self.native, method, None)
            if collective_window > maximum or not callable(query) or query() < collective_window:
                raise RuntimeError(f"backend lacks the requested calibrated {family} window protocol")
            self.provenance[family + "_collective_window"] = dict(
                maximum=maximum, eligible=eligible, terminal_release=True)
            return collective_window

    def batch(self, operations, *, collective_window=1):
        with self._lock:
            self._check_idle()
            operations = tuple(operations)
            if not 1 <= len(operations) <= 1024:
                raise ValueError("an RPU batch requires 1..1024 operations")
            for operation in operations:
                if not isinstance(operation, RpuOperation) or operation.context is not self:
                    raise ValueError("all operations must belong to this RPU context")
                operation._check()
                if operation.epoch != self.epoch:
                    raise RuntimeError("RPU operation is stale after reset; prepare it again")
            # Collective admission and segment/counter budgets remain native
            # checks. Do not infer counts from names or raise the 48 limit.
            window = collective_window
            if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 15:
                raise ValueError("collective_window must be an integer in [1,15]")
            if window != 1:
                entries = {op.entry for op in operations}
                ar = "all_reduce_residual" in entries
                m16 = "down_allreduce_rmsnorm_m16n1024k256_acc32" in entries
                if ar and m16:
                    raise RuntimeError("collective windows cannot mix AR and M16 DownNorm in one batch")
                if ar:
                    self.require_collective_window(window, family="AR")
                if m16:
                    self.require_collective_window(window, family="M16")
            native = (self.native.batch([op.native for op in operations]) if window == 1 else
                      self.native.batch([op.native for op in operations], collective_window=window))
            self._refresh_spm()
            self.stats["batch_preparations"] += 1
            return RpuExecution(self, native, operations)

    def require_batch_fused(self):
        """Admit the whole-stage fused queue before model-plan allocations."""
        with self._lock:
            self._check_idle()
            if not self.provenance["allow_candidate"]:
                raise RuntimeError("fused whole-stage batches require explicit candidate admission")
            if not callable(getattr(self.native, "batch_fused", None)):
                raise RuntimeError("backend lacks the fused whole-stage batch capability")
            self.provenance["fused_whole_stage"] = dict(
                segments=1, per_launch_zeroing=True,
                static_pool="3..49 action DownNorm / 3..50 vision AR+ARR")
            return True

    def batch_fused(self, operations):
        """Retain one whole-stage vendor-discipline queue.

        Every counted collective must carry its own request-pinned counter
        id — M16 DownNorm within 3..49, vision AR/ARR within 3..50 (the
        merger re-arms 50,49 after the tower); the native layer enforces
        the per-family pool discipline, consecutive-distinct ids and
        graph-ordered reuse, and builds a single retained segment without
        isolation boundaries or initialization sweeps. Counter/segment
        budgets remain native checks.
        """
        with self._lock:
            self._check_idle()
            operations = tuple(operations)
            if not 1 <= len(operations) <= 1024:
                raise ValueError("an RPU batch requires 1..1024 operations")
            for operation in operations:
                if not isinstance(operation, RpuOperation) or operation.context is not self:
                    raise ValueError("all operations must belong to this RPU context")
                operation._check()
                if operation.epoch != self.epoch:
                    raise RuntimeError("RPU operation is stale after reset; prepare it again")
            self.require_batch_fused()
            native = self.native.batch_fused([op.native for op in operations])
            self._refresh_spm()
            self.stats["batch_preparations"] += 1
            return RpuExecution(self, native, operations)

    def wait(self):
        with self._lock:
            pending = self._pending
        if pending is not None:
            pending.wait()

    def recover(self):
        with self._lock:
            self._check_idle()
            self.needs_reset = True
            self.epoch += 1
            for resource in self._resources:
                if isinstance(resource, RpuBuffer) and not resource.closed and resource.state_kind != "none":
                    resource.reset()
            self.needs_reset = False
            self.stats["recoveries"] += 1

    def close(self):
        wait_error = None
        try:
            self.wait()
        except BaseException as caught:
            wait_error = caught
        with self._lock:
            if self.closed:
                return
            close_error = None
            native_released = False
            for resource in reversed(tuple(self._resources)):
                try:
                    resource.close()
                except BaseException as caught:
                    close_error = close_error or caught
            if not self._resources:
                try:
                    self.native.close()
                    native_released = True
                except BaseException as caught:
                    close_error = close_error or caught
                    native_released = getattr(self.native, "_id", None) == 0
            if not self._resources and native_released:
                self.closed = True
                self._resources.clear()
                self.stats["spm_live_bytes_per_core"] = 0
                self.stats["sigmoid_lut_bytes_per_core"] = 0
                self.stats["rsqrt_lut_bytes_per_core"] = 0
            else:
                # Retain failed owners and exclusivity. Explicit close can be
                # retried; a failed release must not advertise a free device.
                self.needs_reset = True
        error = wait_error or close_error
        if wait_error is not None and close_error is not None and hasattr(wait_error, "add_note"):
            wait_error.add_note(f"RPU release cleanup failed: {close_error}")
        if error is not None:
            raise error

    def __enter__(self):
        self._check_idle()
        return self

    def __exit__(self, *_):
        self.close()
