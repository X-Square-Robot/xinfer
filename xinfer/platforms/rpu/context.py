"""Process-owned RPU context, SDK admission and prepared-resource lifecycle."""

from __future__ import annotations

import math
import threading
import weakref
from collections import Counter

import numpy as np

from .execution import RpuExecution
from .resources import RpuBuffer, RpuBusyError, RpuOperation


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
                raise RuntimeError("RPU execution requires an injected native backend; "
                                   "use the entry-point engine factory or pass backend_factory")
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
        through to the native operator provider's family resolution unchanged.
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
