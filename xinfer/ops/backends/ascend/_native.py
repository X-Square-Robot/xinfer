"""Ascend native library, launch adapters and graph-owned buffers.

Build ``xinfer_kernels_ascend.so`` from ``src/`` with CMake. The package-local
library is optional; ``XINFER_ASCEND_KERNELS_LIBRARY`` selects an explicit build
and ``XINFER_ASCEND_KERNELS=0`` disables it. Explicit load errors propagate.
TVM-FFI faces and existing C-ABI adapters use this one library and its fingerprint.

Scratch buffers belong to a device and stream. Warm shapes before capture;
buffers referenced by captured graphs retain their addresses and stay alive.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import re
import types
import weakref
from pathlib import Path
from threading import Lock

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

# The existing C-ABI adapters and typed TVM-FFI faces share the same library.
_raw_library = None
_WARMED: set = set()


def core_library():
    """C-ABI view of the loaded native library, or None when it is unavailable."""
    global _raw_library
    if lib is not None and _raw_library is None:
        _raw_library = ctypes.CDLL(LIBRARY_PATH)
    return _raw_library


def enabled() -> bool:
    return lib is not None


def has_symbol(lib_: ctypes.CDLL, name: str) -> bool:
    try:
        getattr(lib_, name)
    except AttributeError:
        return False
    return True


def stream_of(tensor):
    """The current NPU stream as a C pointer (per-call: NPUGraph capture records it)."""
    return ctypes.c_void_p(_torch_current_stream(tensor))


def _torch_current_stream(tensor):
    return torch.npu.current_stream(tensor.device).npu_stream


def aot_entry(lib_: ctypes.CDLL, name: str, pointers: int, scalars: int):
    """Bind ``xinfer_aot_<name>_{call,get_last_error}`` on ``lib_``.

    The launcher ABI is the packed-launch contract: ``pointers`` void* tensor operands, ``scalars``
    int64 values, then the stream. Returns ``(call, get_last_error)`` or ``None`` when the library
    predates the kernel.
    """
    try:
        call = getattr(lib_, f"xinfer_aot_{name}_call")
        error = getattr(lib_, f"xinfer_aot_{name}_get_last_error")
    except AttributeError:
        return None
    call.restype = ctypes.c_int32
    call.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int64] * scalars + [ctypes.c_void_p]
    error.restype = ctypes.c_char_p
    return call, error


def warmup(lib_key: str) -> bool:
    """Whether this library still needs its first eager launch on this stream.

    The AOT first-call loss: a kernel launched for the first time inside an NPU graph capture can
    silently not execute. Glue routes call ``warmup`` before their first launch per (library,
    stream) and refuse to run cold inside a capture, exactly like the fia_fuse/gqa-glue knives did.
    """
    return lib_key in _WARMED


def mark_warm(lib_key: str) -> None:
    _WARMED.add(lib_key)


def launch(aot_pair, tensors, scalars, stream=None):
    """Run one bound launcher; raises with the library's error text on failure."""
    call, get_error = aot_pair
    if stream is None:
        stream = _torch_current_stream(tensors[0])
    status = call(*[ctypes.c_void_p(t.data_ptr()) for t in tensors],
                 *[ctypes.c_int64(int(s)) for s in scalars], ctypes.c_void_p(stream))
    if status:
        message = get_error()
        raise RuntimeError("xinfer_aot launcher failed: "
                           f"{message.decode('utf-8', 'replace') if message else 'unknown failure'}")


_FFI_FACES: dict = {}  # kernel name -> its typed function, cached on first sight (the hot dose)


def launch_kernel(name, tensors, scalars=()):
    """Launch one AOT kernel through the single library's TVM-FFI face; ``False`` when it cannot.

    The unified boundary of the per-kernel glue faces (the ``aot_glue.cc`` files of ``src/``): the
    face takes the same packed-launch ABI the ctypes bindings drive - one tensor per pointer, one
    int64 per scalar - and resolves the current NPU stream itself, so the python envelopes call it
    with exactly the tensors and scalars they hold. ``False`` (the caller keeps its ctypes path)
    when the single library is not loaded or carries no face of ``name``: a family-library
    deployment or a library built before a kernel then serves unchanged. The face lookup is cached
    per name: the serving stacks launch thousands of kernels per request, and the module attribute
    walk is measurable next to a ctypes call at that dose.
    """
    face = _FFI_FACES.get(name)
    if face is None:
        if lib is None:
            return False
        try:
            face = getattr(lib, name)
        except AttributeError:
            return False
        _FFI_FACES[name] = face
    face(*tensors, *[int(s) for s in scalars])
    return True


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

    The buffer is per stream (the key carries the NPU stream handle) and is never shared across streams:
    concurrent streams reading one buffer corrupted results before. ``zeroed``: cleared when allocated, for
    kernels that leave their flags zeroed.
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


_derived = {}  # weight identity and layout -> (weak owner, derived tensor)
_derived_lock = Lock()


def derived(what, weight, make):
    """Cache an independent derived tensor for the lifetime of its source tensor owner.

    Ordinary views share their ``_base`` owner. Inference views without ``_base`` must themselves
    stay alive across warmup and capture: a storage address does not identify a logical weight.
    Neither the cache nor its cleanup callback keeps the source weight alive.
    """
    owner = weight if weight._base is None else weight._base
    version = None if owner.is_inference() else owner._version
    key = (what, weight.device, weight.dtype, id(owner), weight.data_ptr(),
           tuple(weight.shape), weight.stride(), version)
    entry = _derived.get(key)
    if entry is not None and entry[0]() is owner:
        return entry[1]
    with _derived_lock:
        entry = _derived.get(key)
        if entry is not None and entry[0]() is owner:
            return entry[1]
        if capturing(weight.device):
            raise RuntimeError(f"the {what} form of a weight is built on its first call, which must not be inside "
                               "an NPU graph capture; retain its owner through warmup and capture")
        result = make()

        def discard(reference):
            if _derived.get(key, (None,))[0] is reference:
                _derived.pop(key, None)

        _derived[key] = weakref.ref(owner, discard), result
        return result


# ---- the C-ABI composition bridges of the core family library (src/common/) ------------------
class RawBridge:
    """The C-ABI composition bridges of the core library (lane-3 statics).

    The v22 v22 library kept aclnn compositions (FIA probes, ACLNN Mm, the
    GDN vendor chunk, GEMM finish launchers) as internal primitives with raw
    former ``C`` symbols; the glue bound them through ctypes. The unified
    core library carries the same launchers under ``xinfer_*`` names
    (src/common/) and this class exposes the python callables with the
    bridges' documented signatures.
    """

    def __init__(self):
        self._lib = None

    def library(self):
        if self._lib is None:
            self._lib = core_library()
        return self._lib

    def _pair(self, query_name, launch_name, error_name):
        lib = self.library()
        if lib is None:
            return None
        try:
            query = getattr(lib, query_name)
            launch = getattr(lib, launch_name)
            error = getattr(lib, error_name)
        except AttributeError:
            return None
        return query, launch, error

    # ---- ACLNN Mm (cann_mm_fp32) -------------------------------------
    def mm_pair(self):
        """``(query(out, x, w, math_type), launch(out, ws, x, w, math_type))``.

        Three-dimensional BNSD ``[16, rows, 128]`` o-projection inputs ride the
        core library's Cube kernels instead of the ACLNN Mm bridge (the v22
        bridge's SmallmBnsdRoute/BnsdM768Route, ported to the python side):
        ``[16, rows in {16, 32, 64}, 128]`` over the ``[1024, 2048]`` weight
        takes ``smallm_bnsd_o_proj{,_m32,_m64}`` with an FP32 ``[rows, 1024]``
        out, and the prefill ``[16, 768, 128]`` form over the ``[2048, 2048]``
        weight takes ``bnsd_o_proj_m768_bf16`` with a BF16 ``[768, 2048]``
        out; both routes are workspace-free.
        """

        pair = self._pair("xinfer_cann_mm_fp32_workspace_bytes",
                          "xinfer_cann_mm_fp32_launch",
                          "xinfer_cann_mm_fp32_get_last_error")
        if pair is None:
            return None
        query, launch, error = pair
        error.restype = ctypes.c_char_p
        query.restype = ctypes.c_int64
        # (device, bf16, in_dims*, w_dims*, out_dims*, in, w, out, weight_kn)
        query.argtypes = [ctypes.c_int32, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
                          ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        launch.restype = ctypes.c_int
        # (device, bf16, in_dims*, w_dims*, out_dims*, in, w, out, ws, ws_bytes, weight_kn)
        launch.argtypes = [ctypes.c_int32, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
                           ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_int64, ctypes.c_int64]

        bnsd_entries = {}
        for name in ("smallm_bnsd_o_proj", "smallm_bnsd_o_proj_m32",
                     "smallm_bnsd_o_proj_m64", "bnsd_o_proj_m768_bf16"):
            entry = self._pair(f"xinfer_aot_{name}_call",
                               f"xinfer_aot_{name}_get_last_error",
                               f"xinfer_aot_{name}_get_last_error")
            if entry is None:
                continue
            call, _, kern_error = entry
            call.restype = ctypes.c_int32
            call.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_void_p]
            kern_error.restype = ctypes.c_char_p
            bnsd_entries[name] = (call, kern_error)

        def dims(x, w, out):
            m, k = x.shape
            n = w.shape[0]
            return ((ctypes.c_int64 * 2)(int(m), int(k)),
                    (ctypes.c_int64 * 2)(int(n), int(k)),
                    (ctypes.c_int64 * 2)(int(m), int(n)))

        def bnsd_form(x, w, math_type):
            """(kernel name, out dtype string) for a BNSD input, else None."""
            if (math_type != 0 or x.ndim != 3 or not x.is_contiguous()
                    or x.shape[0] != 16 or x.shape[2] != 128
                    or w.ndim != 2 or not w.is_contiguous()
                    or x.dtype != w.dtype or str(x.dtype) != "torch.bfloat16"
                    or x.device.type != "npu" or w.device != x.device):
                return None
            rows = int(x.shape[1])
            if rows in (16, 32, 64) and tuple(w.shape) == (1024, 2048):
                return ("smallm_bnsd_o_proj" if rows == 16 else f"smallm_bnsd_o_proj_m{rows}",
                        "torch.float32")
            if rows == 768 and tuple(w.shape) == (2048, 2048):
                return ("bnsd_o_proj_m768_bf16", "torch.bfloat16")
            return None

        def bnsd_validate(out, x, w, dtype_name):
            import torch

            rows = int(x.shape[1])
            expected = (768, 2048) if dtype_name == "torch.bfloat16" else (rows, 1024)
            dtype = getattr(torch, dtype_name.split(".")[-1])
            if (out.dtype != dtype or tuple(out.shape) != expected or not out.is_contiguous()
                    or out.device != x.device):
                raise ValueError(
                    f"BNSD o-projection out must be a contiguous {dtype} "
                    f"{tuple(expected)} on the input NPU")

        def mm_query(out, x, w, math_type=0):
            form = bnsd_form(x, w, math_type)
            if form is not None:
                if form[0] not in bnsd_entries:
                    raise ValueError(f"this core library does not carry the {form[0]} BNSD kernel")
                bnsd_validate(out, x, w, form[1])
                return 0
            in_d, w_d, out_d = dims(x, w, out)
            size = query(x.get_device(), 1 if math_type == 0 else 0, in_d, w_d, out_d,
                         ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(w.data_ptr()),
                         ctypes.c_void_p(out.data_ptr()), 0)
            if size < 0:
                raise RuntimeError(f"cann_mm_fp32 workspace: {error().decode('utf-8', 'replace')}")
            return int(size)

        def mm_launch(out, workspace, x, w, math_type=0):
            form = bnsd_form(x, w, math_type)
            if form is not None:
                name = form[0]
                if name not in bnsd_entries:
                    raise ValueError(f"this core library does not carry the {name} BNSD kernel")
                bnsd_validate(out, x, w, form[1])
                # the single library's TVM-FFI face serves first (the unified launch
                # boundary); the ctypes launcher of the family library is the fallback
                if launch_kernel(name, (x, w, out)):
                    return
                call, kern_error = bnsd_entries[name]
                import torch

                stream = torch.npu.current_stream(x.device).npu_stream
                status = call(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(w.data_ptr()),
                              ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(stream))
                if status:
                    message = kern_error()
                    raise RuntimeError(
                        f"{name} failed: "
                        f"{message.decode('utf-8', 'replace') if message else 'unknown'}")
                return
            in_d, w_d, out_d = dims(x, w, out)
            status = launch(x.get_device(), 1 if math_type == 0 else 0, in_d, w_d, out_d,
                            ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(w.data_ptr()),
                            ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(workspace.data_ptr()),
                            workspace.numel(), 0)
            if status:
                raise RuntimeError(f"cann_mm_fp32 launch: {error().decode('utf-8', 'replace')}")

        return mm_query, mm_launch

    def typed_workspace_size(self):
        """``size(m, n, k, kn, element_bytes)`` — the typed Ascend cube scratch query."""
        lib = self.library()
        if lib is None:
            return None
        try:
            query = lib.xinfer_npu_typed_workspace_size
            error = lib.xinfer_npu_typed_workspace_size_get_last_error
        except AttributeError:
            return None
        error.restype = ctypes.c_char_p
        query.restype = ctypes.c_int64
        query.argtypes = [ctypes.c_int64] * 5

        def typed_size(m, n, k, kn, element_bytes):
            size = query(int(m), int(n), int(k), int(bool(kn)), int(element_bytes))
            if size < 0:
                raise RuntimeError(f"npu_typed_workspace: {error().decode('utf-8', 'replace')}")
            return int(size)

        return typed_size

    # ---- FIA probes (cann_attention) ---------------------------------
    def bnsd_pair(self):
        """``(query, launch)`` over the masked BNSD FIA V5 composition."""
        pair = self._pair("xinfer_cann_fia_v5_probe_workspace_size",
                          "xinfer_cann_fia_v5_probe_launch",
                          "xinfer_cann_fia_v5_get_last_error")
        if pair is None:
            return None
        return _fia_pair(pair, bnsd=True)

    def tnd_pair(self):
        """``(query, launch)`` over the packed TND FIA vision composition."""
        pair = self._pair("xinfer_cann_fia_vision_probe_workspace_size",
                          "xinfer_cann_fia_vision_probe_launch",
                          "xinfer_cann_fia_v5_get_last_error")
        if pair is None:
            return None
        return _fia_pair(pair, bnsd=False)

    # ---- GDN vendor chunk (cann_gated_delta_net) ---------------------
    def gdr_chunk_pair(self):
        """``(query, launch)`` over the vendor ChunkGatedDeltaRule bridge.

        The TensorView-signature callables the glue expects:
        ``query(out, final_state, q, k, v, beta, state, lengths, g, scale)``
        and ``launch(out, final_state, workspace, q, k, v, beta, state,
        lengths, g, scale)`` over the raw C-ABI of the static bridge.
        """
        lib = self.library()
        if lib is None:
            return None
        try:
            query = lib.xinfer_gdr_cann_chunk_workspace_size
            launch = lib.xinfer_gdr_cann_chunk_launch
            error = lib.xinfer_gdr_cann_chunk_get_last_error
        except AttributeError:
            return None
        error.restype = ctypes.c_char_p
        query.restype = ctypes.c_int64
        query.argtypes = ([ctypes.c_int32] + [ctypes.c_void_p] * 9
                          + [ctypes.c_int64] * 3 + [ctypes.c_double])
        launch.restype = ctypes.c_int
        launch.argtypes = ([ctypes.c_int32] + [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 3
                           + [ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64])

        def chunk_query(out, final_state, q, k, v, beta, state, lengths, g, scale):
            t, nk, nv = int(q.shape[0]), int(q.shape[1]), int(v.shape[1])
            size = query(q.get_device(),
                         ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                         ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(beta.data_ptr()),
                         ctypes.c_void_p(g.data_ptr()) if g is not None else None,
                         ctypes.c_void_p(state.data_ptr()), ctypes.c_void_p(final_state.data_ptr()),
                         ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(lengths.data_ptr()),
                         t, nk, nv, float(scale))
            if size < 0:
                raise RuntimeError(f"gdr chunk: {error().decode('utf-8', 'replace')}")
            return int(size)

        def chunk_launch(out, final_state, workspace, q, k, v, beta, state, lengths, g, scale):
            t, nk, nv = int(q.shape[0]), int(q.shape[1]), int(v.shape[1])
            stream = _torch_current_stream(q)
            status = launch(q.get_device(),
                            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                            ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(beta.data_ptr()),
                            ctypes.c_void_p(g.data_ptr()) if g is not None else None,
                            ctypes.c_void_p(state.data_ptr()), ctypes.c_void_p(final_state.data_ptr()),
                            ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(lengths.data_ptr()),
                            t, nk, nv, float(scale), ctypes.c_void_p(stream),
                            ctypes.c_void_p(workspace.data_ptr()), workspace.numel())
            if status:
                raise RuntimeError(f"gdr chunk: {error().decode('utf-8', 'replace')}")

        return chunk_query, chunk_launch

    # ---- GEMM finish launchers ---------------------------------------
    def gemm_glu_fp32_launch(self):
        """``launch(out, ws, x, w, bias, m, k, n2, paired, kn)`` (gemm_glu_fp32)."""
        lib = self.library()
        if lib is None:
            return None
        try:
            launch = lib.xinfer_gemm_glu_fp32_launch
            error = lib.xinfer_gemm_glu_fp32_get_last_error
        except AttributeError:
            return None
        error.restype = ctypes.c_char_p
        launch.restype = ctypes.c_int
        launch.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                           ctypes.c_int64, ctypes.c_int64]

        def run(out, workspace, x, w, bias, m, k, n2, paired, kn):
            status = launch(
                x.get_device(),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(workspace.data_ptr()),
                workspace.numel(),
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(w.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()) if bias is not None else None,
                int(m), int(k), int(n2), int(bool(paired)), int(bool(kn)),
            )
            if status:
                raise RuntimeError(f"gemm_glu_fp32: {error().decode('utf-8', 'replace')}")

        return run


def _fia_pair(pair, *, bnsd: bool):
    query, launch, error = pair
    error.restype = ctypes.c_char_p

    if bnsd:
        query.restype = ctypes.c_int64
        query.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                          ctypes.c_double, ctypes.c_int64]
        launch.restype = ctypes.c_int
        launch.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_int64), ctypes.c_double, ctypes.c_int64]

        def bnsd_query(out, q, k, v, drop_mask, scale, inner_precise=2, sparse_mode=0):
            b, h, m, d = q.shape
            hkv, length = k.shape[1], k.shape[2]
            dims = (ctypes.c_int64 * 6)(int(b), int(h), int(m), int(d), int(hkv), int(length))
            # The retired FFI face's trailing pair is (inner_precise,
            # sparse_mode): the production glue always passes (2, 0). The
            # static bridge's raw C-ABI takes inner_precise and has no sparse
            # mode, so sparse_mode is accepted and dropped.
            size = query(q.get_device(), ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(q.data_ptr()),
                         ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()),
                         ctypes.c_void_p(drop_mask.data_ptr()), dims, float(scale), int(inner_precise))
            if size < 0:
                raise RuntimeError(f"fia v5 workspace: {error().decode('utf-8', 'replace')}")
            return int(size)

        def bnsd_launch(out, workspace, q, k, v, drop_mask, scale, inner_precise=2, sparse_mode=0):
            b, h, m, d = q.shape
            hkv, length = k.shape[1], k.shape[2]
            dims = (ctypes.c_int64 * 6)(int(b), int(h), int(m), int(d), int(hkv), int(length))
            status = launch(q.get_device(), ctypes.c_void_p(out.data_ptr()),
                            ctypes.c_void_p(workspace.data_ptr()), workspace.numel(),
                            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                            ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(drop_mask.data_ptr()),
                            dims, float(scale), int(inner_precise))
            if status:
                raise RuntimeError(f"fia v5 launch: {error().decode('utf-8', 'replace')}")

        return bnsd_query, bnsd_launch

    query.restype = ctypes.c_int64
    query.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
                      ctypes.POINTER(ctypes.c_int64), ctypes.c_double]
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
                       ctypes.POINTER(ctypes.c_int64), ctypes.c_double]

    def tnd_query(out, q, k, v, endpoints, scale):
        tokens, heads, dim = q.shape
        bounds = tuple(int(end) for end in endpoints)
        dims = (ctypes.c_int64 * 3)(int(tokens), int(heads), int(dim))
        ends = (ctypes.c_int64 * len(bounds))(*bounds)
        size = query(q.get_device(), ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(q.data_ptr()),
                     ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()), ends, len(ends),
                     dims, float(scale))
        if size < 0:
            raise RuntimeError(f"fia vision workspace: {error().decode('utf-8', 'replace')}")
        return int(size)

    def tnd_launch(out, workspace, q, k, v, endpoints, scale):
        tokens, heads, dim = q.shape
        bounds = tuple(int(end) for end in endpoints)
        dims = (ctypes.c_int64 * 3)(int(tokens), int(heads), int(dim))
        ends = (ctypes.c_int64 * len(bounds))(*bounds)
        status = launch(q.get_device(), ctypes.c_void_p(out.data_ptr()),
                        ctypes.c_void_p(workspace.data_ptr()), workspace.numel(),
                        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                        ctypes.c_void_p(v.data_ptr()), ends, len(ends), dims, float(scale))
        if status:
            raise RuntimeError(f"fia vision launch: {error().decode('utf-8', 'replace')}")

    return tnd_query, tnd_launch


BRIDGE = RawBridge()
