"""The native kernel library of the CUDA backend and the memory its kernels need.

``xinfer_kernels.so`` is built from ``src/`` with CMake (docs/kernels.md) and loaded through TVM-FFI: ``lib`` is the
loaded module, or None when it was not built, ``tvm_ffi`` is not installed, it does not load (logged as a warning)
or ``XINFER_KERNELS=0``; the operators registered with ``NATIVE`` then run on other implementations, at the end their
torch references. ``XINFER_KERNELS_LIBRARY`` names a library built elsewhere from the same sources; a library given
that way that does not load is an error. A library older than a file of ``src/`` is loaded with a warning to rebuild
it. The kernels run on torch's current stream, which TVM-FFI passes with the tensors (``apache-tvm-ffi`` 0.1.9 to
0.1.x).

Buffers: scratch memory per (device, stream) and tensors derived once from a weight. A CUDA graph keeps the
addresses it captured, so a buffer never moves once a graph may point at it: growing one while its stream captures
is an error (run each shape once before capturing), and an outgrown buffer that a graph used is kept alive.
"""

from __future__ import annotations

import hashlib
import logging
import os
import types
import weakref
from pathlib import Path

import torch

from xinfer.ops.layout import mxfp8 as mxfp8_layout
from xinfer.ops.layout import nvfp4 as nvfp4_layout

logger = logging.getLogger(__name__)

ENV = "XINFER_KERNELS_LIBRARY"


SOURCES = Path(__file__).with_name("src")


def _stale(path):
    """Whether a file of ``src/`` (when the sources are there) is newer than the library at ``path``."""
    if not SOURCES.is_dir():
        return False
    built = path.stat().st_mtime
    return any(f.is_file() and f.stat().st_mtime > built for f in SOURCES.rglob("*"))


def _load():
    """``(module, path, architectures)`` of the kernel library, or ``(None, None, ())`` (the reason is logged).

    Raises:
        FileNotFoundError: ``$XINFER_KERNELS_LIBRARY`` does not exist.
        RuntimeError: the library ``$XINFER_KERNELS_LIBRARY`` names does not load.
    """
    if os.environ.get("XINFER_KERNELS", "1") == "0":
        return None, None, ()
    explicit = os.environ.get(ENV)
    path = Path(explicit) if explicit else Path(__file__).with_name("xinfer_kernels.so")
    if not path.exists():
        if explicit:
            raise FileNotFoundError(f"{ENV}={explicit} does not exist")
        logger.info("the CUDA kernel library is not built (%s): docs/kernels.md", path)
        return None, None, ()
    try:
        import tvm_ffi
    except ImportError:
        if explicit:
            raise
        logger.info("the CUDA kernel library needs the apache-tvm-ffi package: docs/kernels.md")
        return None, None, ()
    try:
        module = tvm_ffi.load_module(str(path))
        archs = tuple(int(a) for a in str(module.build_target_archs()).split(",") if a.isdigit())
    except Exception as error:  # a library of another toolkit, driver or source revision
        if explicit:
            raise RuntimeError(f"{ENV}={explicit} does not load: {error}") from error
        logger.warning("the CUDA kernel library %s does not load (%s); the CUDA operators run on their torch "
                       "references. Rebuild it: docs/kernels.md", path, error)
        return None, None, ()
    if not explicit and _stale(path):
        logger.warning("the CUDA kernel library %s is older than its sources (%s): rebuild it, docs/kernels.md",
                       path, SOURCES)
    return module, str(path.resolve()), archs


# Architectures the library was built for, e.g. (120,): kernels of other architectures are not in it. A library of
# another device family (it reports no compute capability) registers nothing here.
lib, LIBRARY_PATH, BUILD_ARCHS = _load()
if not BUILD_ARCHS:
    lib = None
# ``register`` arguments of every operator that calls the library
NATIVE = dict(requires=None, platforms=(), version=None)
if lib is not None:
    _digest = hashlib.sha256(Path(LIBRARY_PATH).read_bytes()).hexdigest()
    NATIVE = dict(requires=types.SimpleNamespace(__name__="kernels"), platforms=tuple(f"sm{a}" for a in BUILD_ARCHS),
                  version=f"xinfer_kernels.so sha256:{_digest[:12]}")

_arch = {}


def arch(device) -> int:
    """Compute capability of ``device`` as an integer (12.0 -> 120)."""
    index = torch.cuda.current_device() if device.index is None else device.index
    if index not in _arch:
        major, minor = torch.cuda.get_device_capability(index)
        _arch[index] = major * 10 + minor
    return _arch[index]


# GEMM kernels of an architecture: operation -> classes with ``arch``, ``takes(call)`` and ``run(call, out)``, asked
# in order (gemm.py; the classes are in gemm_sm<arch>.py).
KERNELS = {"gemm": [], "gemm_glu": [], "qkv_proj_rope": [], "fused_mlp": []}


def kernel(op, arch):
    """Class decorator: list a kernel of ``op`` for GPUs of compute capability ``arch``."""

    def add(cls):
        cls.arch = arch
        if arch in BUILD_ARCHS:
            KERNELS[op].append(cls)
        return cls

    return add


def empty_mxfp8(rows, columns, device):
    """An uninitialized MXFP8 pair ``(E4M3 values [rows, columns], uint8 scales)``."""
    return (torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=device),
            torch.empty(mxfp8_layout.scale_elements(rows, columns), dtype=torch.uint8, device=device))


def empty_nvfp4(rows, columns, device):
    """An uninitialized NVFP4 pair ``(packed E2M1 uint8 [rows, columns / 2], E4M3 scales)``."""
    return (torch.empty((rows, columns // 2), dtype=torch.uint8, device=device),
            torch.empty(nvfp4_layout.scale_elements(rows, columns), dtype=torch.float8_e4m3fn, device=device))


_scratch = {}  # (name, device index, stream) -> uint8 tensor
_retired = []  # outgrown buffers a captured graph may still point at


def capturing(device) -> bool:
    """Whether the current stream of ``device`` is capturing a CUDA graph."""
    if device.index is None or device.index == torch.cuda.current_device():
        return torch.cuda.is_current_stream_capturing()
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def scratch(name, nbytes, device, zeroed=False):
    """A uint8 buffer of at least ``nbytes`` for the current stream of ``device``; grows, never shrinks.

    ``zeroed``: cleared when allocated, for kernels that leave their flags zeroed.
    """
    if nbytes == 0:
        return None
    key = (name, device.index, torch.cuda.current_stream(device).cuda_stream)
    buf = _scratch.get(key)
    if buf is None or buf.numel() < nbytes:
        if capturing(device):
            raise RuntimeError(f"{name} needs {nbytes} scratch bytes on a stream that is capturing a CUDA graph; run "
                               "each shape once before capture")
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
            raise RuntimeError(f"the {what} form of a weight is built on its first call, which must not be inside a "
                               "CUDA graph capture")
        _derived[key] = make()
        owner = weight if weight._base is None else weight._base
        weakref.finalize(owner, _derived.pop, key, None)
    return _derived[key]
