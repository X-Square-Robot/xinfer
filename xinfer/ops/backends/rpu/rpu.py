# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Resident FP16 RPU buffers and reusable execution plans.

All native operations use the TVM FFI exports of the loaded xinfer_rpu.so. SDK
objects, physical addresses, SPM offsets and weight packing stay in the backend.
Uploads/downloads are explicit synchronous transfers. Prepare once, execute
many times; an execution retains its native resources until completion.
"""


import operator
from contextlib import suppress

from ._native import lib, require_runtime_assets

# The RpuTp* names resolve through the module __getattr__ below (lazy, so
# rpu_tp can import this module's resource base first without a cycle).
__all__ = [  # noqa: F822
    "RpuBuffer",
    "RpuContext",
    "RpuExecution",
    "RpuTpContext",
    "RpuTpExecution",
    "RpuTpOperation",
    "RpuTpTensor",
    "RpuWeight",
]


def _shape(shape):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if not 1 <= len(shape) <= 2:
        raise ValueError("RPU buffer shape must have rank 1 or 2")
    try:
        dims = tuple(operator.index(d) for d in shape)
    except TypeError as error:
        raise ValueError("RPU buffer dimensions must be integers") from error
    if any(isinstance(d, bool) for d in shape) or any(d <= 0 for d in dims):
        raise ValueError("RPU buffer dimensions must be positive integers")
    return dims


def _required_function(name):
    function = lib.get_function(name)
    if function is None:
        raise RuntimeError(
            f"backend lacks {name}; build xinfer_rpu.so with the required RPU runtime bindings"
        )
    return function


def _writable_dlpack_input(host):
    """Stage readonly NumPy inputs for the legacy DLPack FFI consumer.

    Upload and weight preparation finish synchronously, so this private copy
    only needs to survive the FFI call. Writable inputs are passed unchanged;
    no dtype conversion or repair of unsupported strides is performed.
    Inspecting the array interface keeps Torch/DLPack-only callers independent
    of NumPy, and never changes the caller's writeability flags.
    """
    interface = getattr(host, "__array_interface__", None)
    if isinstance(interface, dict) and interface.get("data", (None, False))[1]:
        flags = getattr(host, "flags", None)
        if flags is None or not flags.c_contiguous:
            raise ValueError("RPU readonly host input must be C-contiguous")
        return host.copy(order="C")
    return host


class _Resource:
    def __init__(self, handle):
        self._id = handle

    @property
    def _handle(self):
        if not self._id:
            raise RuntimeError(f"{type(self).__name__} is closed")
        return self._id

    def close(self):
        """Release this handle; native executions retain dependencies as needed."""
        if self._id:
            lib.rpu_release(self._id)
            self._id = 0

    def __enter__(self):
        _ = self._handle
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        # Explicit close()/wait() reports errors; Python finalizers cannot.
        with suppress(Exception):
            self.close()


class RpuBuffer(_Resource):
    """Owned, contiguous FP16 LocalSPM storage; no pointer reinterpretation.

    Allocate through RpuContext.empty/upload. Views and partial aliases are not
    supported. In-place SiLU can reuse the same buffer object exactly.
    """

    def __init__(self, handle, context, shape):
        super().__init__(handle)
        self.context = context
        self.shape = tuple(shape)

    @property
    def dtype(self):
        return "float16"

    @property
    def device(self):
        return f"rpu:{self.context.core}"

    @property
    def ndim(self):
        return len(self.shape)

    def copy_from(self, host):
        """Synchronously upload contiguous CPU FP16 NumPy/Torch/DLPack data."""
        if tuple(host.shape) != self.shape:
            raise ValueError(f"host shape {tuple(host.shape)} != buffer {self.shape}")
        lib.rpu_buffer_upload(self._handle, _writable_dlpack_input(host))
        return self

    def copy_to(self, host):
        """Synchronously download into caller-owned contiguous CPU FP16 data."""
        if tuple(host.shape) != self.shape:
            raise ValueError(f"host shape {tuple(host.shape)} != buffer {self.shape}")
        lib.rpu_buffer_download(host, self._handle)
        return host

    def numpy(self):
        """Explicitly allocate a CPU NumPy array and download into it."""
        import numpy as np

        return self.copy_to(np.empty(self.shape, dtype=np.float16))


class RpuWeight(_Resource):
    """Immutable prepared FP16 [N,K] weight, packed/uploaded once into DDR."""

    def __init__(self, handle, context, shape):
        super().__init__(handle)
        self.context = context
        self.shape = tuple(shape)


class RpuExecution(_Resource):
    """Prepared SDK queue retaining program, kernel, buffers and packed weight.

    execute(async_=True) requires wait() before accessing buffers or submitting
    another plan on the device. close() waits for outstanding work. Repeated
    execution reuses the queue; no SDK Graph or caller graph ownership is implied.
    """

    def execute(self, *, async_=False):
        if not isinstance(async_, bool):
            raise TypeError("async_ must be bool")
        lib.rpu_execute(self._handle, async_)
        return self

    def wait(self):
        lib.rpu_wait(self._handle)
        return self


class RpuContext(_Resource):
    """A single RPU core and its allocator; the caller owns TP/model orchestration.

    Candidate GEMM and SiLU kernels require explicit allow_candidate=True. No silent
    conversion, padding, host fallback, collective or model graph is inserted.
    """

    def __init__(self, core=0, *, allow_candidate=False):
        if isinstance(core, bool):
            raise TypeError("core must be an integer")
        core = operator.index(core)
        if not isinstance(allow_candidate, bool):
            raise TypeError("allow_candidate must be bool")
        assets = require_runtime_assets()
        super().__init__(lib.rpu_context_create(core, allow_candidate))
        self._asset_provenance = dict(assets)
        self.core = core

    @property
    def asset_provenance(self):
        """Identity of the external asset manifest validated before context creation."""
        return dict(self._asset_provenance)

    def empty(self, shape):
        shape = _shape(shape)
        rows, cols = (1, shape[0]) if len(shape) == 1 else shape
        return RpuBuffer(
            lib.rpu_buffer_empty(self._handle, rows, cols), self, shape
        )

    def upload(self, host):
        buffer = self.empty(host.shape)
        try:
            return buffer.copy_from(host)
        except Exception:
            buffer.close()
            raise

    def prepare_weight(self, host):
        """Pack row-major FP16 W[N,K] into [N/16,K/16,16N,16K] once."""
        return RpuWeight(
            _required_function("rpu_prepare_weight")(
                self._handle, _writable_dlpack_input(host)
            ),
            self,
            host.shape,
        )

    def _check(self, *resources):
        _ = self._handle
        for resource in resources:
            if not isinstance(resource, (RpuBuffer, RpuWeight)):
                raise TypeError("expected RpuBuffer or RpuWeight")
            if resource.context is not self:
                raise ValueError("all RPU resources must belong to the same context")
            _ = resource._handle

    def prepare_gemm(self, out, a, weight, bias):
        """Prepare out[M,N] = a[M,K] @ weight[N,K].T + bias[N]."""
        self._check(out, a, weight, bias)
        if not isinstance(weight, RpuWeight):
            raise TypeError("weight must be prepared by context.prepare_weight")
        prepare = _required_function("rpu_prepare_gemm")
        return RpuExecution(
            prepare(out._handle, a._handle, weight._handle, bias._handle)
        )

    def prepare_layernorm(self, out, x, weight, bias=None, epsilon=1e-5):
        """Prepare row-wise affine LayerNorm; epsilon is encoded as FP16."""
        self._check(out, x, weight, *(() if bias is None else (bias,)))
        prepare = _required_function("rpu_prepare_layernorm")
        return RpuExecution(
            prepare(
                out._handle,
                x._handle,
                weight._handle,
                0 if bias is None else bias._handle,
                epsilon,
            )
        )

    def prepare_silu_mul(self, out, x, y):
        """Prepare staged FP16 SiLU-mul; exact out==x/y alias is supported."""
        self._check(out, x, y)
        if out.shape != x.shape or x.shape != y.shape:
            raise ValueError("SiLU out, x and y shapes must match")
        prepare = _required_function("rpu_prepare_silu_mul")
        return RpuExecution(prepare(out._handle, x._handle, y._handle))


_TP_EXPORTS = ("RpuTpContext", "RpuTpExecution", "RpuTpOperation", "RpuTpTensor")


def __getattr__(name):
    # rpu_tp imports this module's resource base; resolve the TP8 classes on
    # first use so either module can be imported first without a cycle.
    if name in _TP_EXPORTS:
        from . import rpu_tp

        return getattr(rpu_tp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
