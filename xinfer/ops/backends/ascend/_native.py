"""The native kernel library of the Ascend backend and the memory its kernels need.

``xinfer_kernels_ascend.so`` is built from ``src/`` with CMake (src/README.md) and loaded through TVM-FFI:
``lib`` is the loaded module, or None when it was not built, ``tvm_ffi`` is not installed or
``XINFER_ASCEND_KERNELS=0``; the operators registered with ``NATIVE`` then run on other implementations, at
the end their torch references. ``XINFER_ASCEND_KERNELS_LIBRARY`` names a library built elsewhere from the
same sources. The external xcompute path (``backends/_xcompute.py``) keeps serving prebuilt external builds
alongside this one: a library built here loads only through ``XINFER_ASCEND_KERNELS_LIBRARY`` or the package
directory, and ``XINFER_XCOMPUTE_LIBRARY`` pointed at it still fails the external loader's paired-package
check, so the two loaders cannot serve each other's library.

SoCs: the library reports the SoCs it was built for (``BUILD_SOCS``, e.g. ``("ascend950dt9582",)`` — the
vendor's device name lowercased to alphanumerics, the string ``platforms/npu.py`` reports as the platform's
arch) and registers its operators on exactly those platforms. A library of another device family (it reports
no SoC) registers nothing here.

Buffers: scratch memory per (device, stream) and tensors derived once from a weight. An NPU graph (TorchNPU's
``NPUGraph``) replays the device-side work it captured, so a buffer a captured graph launches against keeps
its address for the graph's life: growing one while its stream captures is an error (run each shape once
before capturing — the engine's warmup and capture discipline does), and an outgrown buffer that a graph used
is kept alive. CANN's operator framework workspaces (the workspace some aclnn operators want besides their
arguments) flow through the same scratch ledger, per (name, device, stream), so a captured graph keeps
pointing at the workspace it was captured with.

This regulation is written after the CUDA-graph rules of the CUDA backend's ``_native.py`` and the NPUGraph
discipline the Ascend deployment validated (launches resolve the current NPU stream per call; every shape is
warmed eagerly before its capture; device buffers live through the queued work that reads them). TODO(R1):
verify on hardware, then keep or amend each point:

- whether ``torch.npu.is_current_stream_capturing()`` covers the engine's capture stream on the pinned
  TorchNPU (2.10.0.post4) — the engine captures through ``torch.npu.graph(auto_dispatch_capture=True)``;
- whether TorchNPU's capture-time allocations go through a graph memory pool the growth guard can see at
  all, or the warmup rule alone must prevent them — the FIA-FIX lesson stands either way: a first launch of
  a shape on a capturing stream is not trusted, because AOT MixedKernel launches were seen to drop silently
  (a device slab rewritten in place per request interacted with NPUGraph capture);
- that ``derived`` copies keyed by address stay valid under TorchNPU's caching allocator, whose reuse of a
  freed address must never hand a later weight an earlier weight's copy (the CUDA tests assert this; the
  Ascend kernel tests must too).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import types
import weakref
from pathlib import Path

import torch

try:  # the NPU stream and device APIs live on the TorchNPU extension, as the other Ascend modules import it
    import torch_npu  # noqa: F401 - registers the npu device and torch.npu
except ImportError:
    torch_npu = None

logger = logging.getLogger(__name__)

ENV = "XINFER_ASCEND_KERNELS_LIBRARY"


def _load():
    """``(module, path)`` of the kernel library, or ``(None, None)`` (the reason is logged).

    Raises:
        FileNotFoundError: ``$XINFER_ASCEND_KERNELS_LIBRARY`` does not exist.
    """
    if os.environ.get("XINFER_ASCEND_KERNELS", "1") == "0":
        return None, None
    explicit = os.environ.get(ENV)
    path = Path(explicit) if explicit else Path(__file__).with_name("xinfer_kernels_ascend.so")
    if not path.exists():
        if explicit:
            raise FileNotFoundError(f"{ENV}={explicit} does not exist")
        logger.info("the Ascend kernel library is not built (%s): xinfer/ops/backends/ascend/src/README.md", path)
        return None, None
    try:
        import tvm_ffi
    except ImportError:
        logger.info("the Ascend kernel library needs the apache-tvm-ffi package: src/README.md")
        return None, None
    return tvm_ffi.load_module(str(path)), str(path.resolve())


lib, LIBRARY_PATH = _load()
# SoCs the library was built for, e.g. ("ascend950dt9582",): kernels of other SoCs are not in it. A library of
# another device family (it reports no SoC — a CUDA library answers a compute capability, an external xcompute
# build answers no SoC at all) registers nothing here.
_SOC = re.compile(r"ascend[a-z0-9]*\d")
BUILD_SOCS = () if lib is None else tuple(s for s in str(lib.build_target_archs()).split(",") if _SOC.fullmatch(s))
if not BUILD_SOCS:
    lib = None
# ``register`` arguments of every operator that calls the library. ``platforms`` are the SoC strings the
# platform reports as its arch (``ascend950dt9582``), the per-SoC granularity the CUDA side reaches with
# ``sm120``: ``Implementation.supports`` matches them against {platform name, device type, arch}.
NATIVE = dict(requires=None, platforms=(), version=None)
if lib is not None:
    _digest = hashlib.sha256(Path(LIBRARY_PATH).read_bytes()).hexdigest()
    NATIVE = dict(requires=types.SimpleNamespace(__name__="kernels"), platforms=BUILD_SOCS,
                  version=f"xinfer_kernels_ascend.so sha256:{_digest[:12]}")

_soc_of = {}


def soc(device) -> str:
    """The SoC of ``device`` (``Ascend950DT9582`` -> ``ascend950dt9582``), the platform's arch string."""
    index = torch.npu.current_device() if device.index is None else device.index
    if index not in _soc_of:
        name = str(getattr(torch.npu.get_device_properties(index), "name", "") or "Ascend")
        _soc_of[index] = re.sub(r"[^a-z0-9]", "", name.lower()) or "ascend"
    return _soc_of[index]


# GEMM kernels of a SoC: operation -> classes with ``soc``, ``takes(call)`` and ``run(call, out)``, asked in
# order (the gemm dispatcher; the classes are in gemm_<soc>.py).
KERNELS = {"gemm": [], "gemm_glu": [], "qkv_proj_rope": []}


def kernel(op, soc):
    """Class decorator: list a kernel of ``op`` for NPUs of SoC ``soc`` (e.g. ``ascend950dt9582``)."""

    def add(cls):
        cls.soc = soc
        if soc in BUILD_SOCS:
            KERNELS[op].append(cls)
        return cls

    return add


_scratch = {}  # (name, device index, stream) -> uint8 tensor
_retired = []  # outgrown buffers a captured NPU graph may still point at


def capturing(device) -> bool:
    """Whether the current stream of ``device`` is capturing an NPU graph."""
    if device.index is None or device.index == torch.npu.current_device():
        return torch.npu.is_current_stream_capturing()
    with torch.npu.device(device):
        return torch.npu.is_current_stream_capturing()


def scratch(name, nbytes, device, zeroed=False):
    """A uint8 buffer of at least ``nbytes`` for the current stream of ``device``; grows, never shrinks.

    ``zeroed``: cleared when allocated, for kernels that leave their flags zeroed.
    """
    if nbytes == 0:
        return None
    key = (name, device.index, torch.npu.current_stream(device).npu_stream)
    buf = _scratch.get(key)
    if buf is None or buf.numel() < nbytes:
        if capturing(device):
            raise RuntimeError(f"{name} needs {nbytes} scratch bytes on a stream that is capturing an NPU graph; "
                               "run each shape once before capture")
        if buf is not None and getattr(buf, "_in_graph", False):
            _retired.append(buf)
        buf = (torch.zeros if zeroed else torch.empty)(max(int(nbytes), 4096), dtype=torch.uint8, device=device)
        _scratch[key] = buf
    if not getattr(buf, "_in_graph", False) and capturing(device):
        buf._in_graph = True
    return buf


class Carver:
    """Hands out 256-byte aligned slices of one scratch buffer in order."""

    ALIGN = 256

    def __init__(self, buf):
        self.buf, self.off = buf, 0

    @classmethod
    def size(cls, *nbytes):
        return sum((b + cls.ALIGN - 1) // cls.ALIGN * cls.ALIGN for b in nbytes)

    def take(self, nbytes):
        piece = self.buf[self.off : self.off + nbytes]
        self.off += self.size(nbytes)
        return piece

    def tensor(self, shape, dtype):
        count = 1
        for dim in shape:
            count *= dim
        return self.take(count * torch.empty((), dtype=dtype).element_size()).view(dtype).view(shape)


_derived = {}  # (what, data pointer, shape, strides) -> tensor derived from a weight


def derived(what, weight, make):
    """``make()``, computed once per weight and dropped when the tensor that owns the weight's memory is freed
    (a later tensor at the same address never gets another weight's result)."""
    key = (what, weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()))
    if key not in _derived:
        if capturing(weight.device):
            raise RuntimeError(f"the {what} form of a weight is built on its first call, which must not be inside "
                               "an NPU graph capture")
        _derived[key] = make()
        owner = weight if weight._base is None else weight._base
        weakref.finalize(owner, _derived.pop, key, None)
    return _derived[key]
