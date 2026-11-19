"""Native RPU resource, buffer and prepared-operation ownership."""

from __future__ import annotations

import numpy as np


class RpuBusyError(RuntimeError):
    """Device access attempted before an outstanding completion was waited."""


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
