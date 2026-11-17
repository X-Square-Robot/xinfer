"""The native kernel library of the Ascend backend and the memory its kernels need.

``xinfer_kernels_ascend.so`` is built from ``src/`` with CMake (src/README.md) and loaded through TVM-FFI:
``lib`` is the loaded module, or None when it was not built, ``tvm_ffi`` is not installed or
``XINFER_ASCEND_KERNELS=0``; the operators registered with ``NATIVE`` then run on other implementations, at
the end their torch references. ``XINFER_ASCEND_KERNELS_LIBRARY`` names a library built elsewhere from the
same sources. The external xcompute path (``backends/_xcompute.py``) keeps serving prebuilt external builds
alongside this one: a library built here loads only through ``XINFER_ASCEND_KERNELS_LIBRARY`` or the package
directory, and ``XINFER_XCOMPUTE_LIBRARY`` pointed at it still fails the external loader's paired-package
check, so the two loaders cannot serve each other's library.

Transition (R1a -> R1b): the per-kernel TVM-FFI glue faces have landed (R1b — the ``aot_glue.cc`` files of
``src/``, one typed function per AOT kernel), and this module's ``launch_kernel`` is the unified boundary
the backend's envelopes call: the single library's face serves a launch when it is loaded, and the family
libraries of the delivered stacks remain the fallback — one core library plus one knife library per routed
family (``src/FAMILIES.md``, built by ``src/build-libraries.sh``), loaded through ctypes under the
deployment's ``XINFER_<FAMILY>_LIB`` names, with the AOT launcher binding, the first-call warmup rule and
the C-ABI composition bridges of the core library (the loading surface of the retired ``_provider.py``,
same names and semantics, re-homed so the backend has one loading module). The master gate is shared:
``XINFER_ASCEND_KERNELS=0`` disables both forms. The faces retire the ctypes bindings module by module as
each envelope's verification lands; the family form retires when every operator runs through ``lib``.

SoCs: the library reports the SoCs it was built for (``BUILD_SOCS``, e.g. ``("ascend950dt9582",)`` — the
vendor's device name lowercased to alphanumerics, the string ``platforms/npu.py`` reports as the platform's
arch) and registers its operators on exactly those platforms. A library of another device family (it reports
no SoC) registers nothing here.

The external loader and this one must never serve each other's library. ``XINFER_XCOMPUTE_LIBRARY`` does not
name this library: the external loader's ``is_ascend`` heuristic reads a soc string as "an Ascend build", and
what refuses the library there is the paired-package check (an in-tree build has no xcompute Python package
whose library-path attribute matches), so the misdirection ends in a refusal, not a double registration — the
acceptance test that points ``XINFER_XCOMPUTE_LIBRARY`` at ``xinfer_kernels_ascend.so`` and expects ``xc is
None`` must keep passing. During the transition an operator may exist twice (``ascend:xcompute`` and
``ascend:kernels``); which one wins is decided explicitly — priority or pin — when IN-TREE-R1 moves the first
operator onto this library, never by registration order.

Buffers: scratch memory per (device, stream) and tensors derived once from a weight. **The scratch key carries
the NPU stream**: a scratch buffer is never shared across streams, because concurrent streams reading one
buffer through this backend corrupted results before (the Wall-X 2B concurrency fix) — a per-stream buffer is
a correctness rule, not an optimization. An NPU graph (TorchNPU's ``NPUGraph``) replays the device-side work
it captured, so a buffer a captured graph launches against keeps its address for the graph's life: growing
one while its stream captures is an error (run each shape once before capturing — the engine's warmup and
capture discipline does), and an outgrown buffer that a graph used is kept alive. CANN's operator framework
workspaces (the workspace some aclnn operators want besides their arguments) flow through the same scratch
ledger, per (name, device, stream), so a captured graph keeps pointing at the workspace it was captured with.

Calls go through TVM-FFI typed functions, which forward the current torch stream with the tensor arguments;
under TorchNPU that stream is the NPU stream the caller is on.

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
- that TVM-FFI's implicit stream forwarding follows the caller's NPU stream under TorchNPU, per operator
  moved in (the CUDA side relies on the same mechanism);
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

# ---- the transition family libraries (the retired provider surface) -------------------------
#
# Until the per-kernel TVM-FFI glue faces land (R1b), the backend's wrappers dispatch through the
# delivered family libraries: one core library plus one knife library per routed family
# (``src/FAMILIES.md``), loaded through ctypes with the deployment's ``XINFER_<FAMILY>_LIB`` names
# preserved verbatim from the v22-era fastpaths. This is the loading surface of the retired
# ``_provider.py`` — the same names, the same semantics, re-homed on the native loader so the tree has
# one loading module; R1b retires the block when every operator runs through ``lib`` above.
import ctypes
from threading import Lock

_MASTER = "XINFER_ASCEND_KERNELS"
_CORE_ENV = "XINFER_ASCEND_CORE_LIB"
_CORE_SHA_ENV = "XINFER_ASCEND_CORE_SHA256"
_CORE_REQUIRED_ENV = "XINFER_ASCEND_CORE_REQUIRED"

_LIB_DIR = Path(__file__).with_name("lib")  # src/build-libraries.sh install's into this directory

_LIBRARIES: dict = {}
_LIBRARY_PATHS: dict = {}  # name -> the path the family library was loaded from
_WARMED: set = set()
_LOCK = Lock()


def _default_library_path(name: str) -> str:
    """The ``lib/<name>`` install position (build-libraries.sh's target)."""
    return str(_LIB_DIR / name)


def _verify_sha256(path: str, expected: str) -> None:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected.strip().lower():
        raise OSError(f"{path}: sha256 {digest} does not match the pinned {expected.strip()!r}")


def library(name: str, *, env: str | None = None, default: str | None = None,
            sha_env: str | None = None, required: bool = False):
    """Load one family library once per process; ``None`` when it is absent.

    ``env`` names the deployment's explicit path (``XINFER_<FAMILY>_LIB``); ``default`` the fallback
    file name inside ``lib/``. An unset or unloadable library degrades the calling family to "not
    registered" — deployments that must not silently fall back pass ``required=True`` (directly or
    via the core's ``XINFER_ASCEND_CORE_REQUIRED``).
    """
    with _LOCK:
        loaded = _LIBRARIES.get(name)
        if loaded is not None:
            return loaded
        path = None
        if env is not None:
            path = os.environ.get(env) or None
        if path is None:
            candidate = _default_library_path(default or name)
            if Path(candidate).exists():
                path = candidate
        if path is None:
            if required or (name == "xinfer_ascend_core"
                            and os.environ.get(_CORE_REQUIRED_ENV) == "1"):
                raise OSError(f"{name}: no library found (set {env or _CORE_ENV} or run "
                              "src/build-libraries.sh install)")
            return None
        if sha_env is not None:
            pinned = os.environ.get(sha_env)
            if pinned:
                _verify_sha256(path, pinned)
        try:
            loaded = ctypes.CDLL(path)
        except OSError:
            if required:
                raise
            return None
        _LIBRARIES[name] = loaded
        _LIBRARY_PATHS[name] = str(path)
        return loaded


def core_library():
    """The core family library (the class-entry kernels and the C-ABI bridges)."""
    if not enabled():
        return None
    return library("xinfer_ascend_core", env=_CORE_ENV, default="libxinfer_ascend_core.so",
                   sha_env=_CORE_SHA_ENV)


def knife_library(family: str, *, env: str, default: str, sha_env: str | None = None):
    """A routed family library (soup2/soupform/fia_fuse/...)."""
    if not enabled():
        return None
    return library(family, env=env, default=default, sha_env=sha_env)


def enabled() -> bool:
    """``XINFER_ASCEND_KERNELS`` (default on); ``0/off`` unregisters everything."""
    return os.environ.get(_MASTER, "on").strip().lower() not in ("", "0", "off", "false", "no")


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


def launch_kernel(name, tensors, scalars=()):
    """Launch one AOT kernel through the single library's TVM-FFI face; ``False`` when it cannot.

    The unified boundary of the per-kernel glue faces (the ``aot_glue.cc`` files of ``src/``): the
    face takes the same packed-launch ABI the ctypes bindings drive - one tensor per pointer, one
    int64 per scalar - and resolves the current NPU stream itself, so the python envelopes call it
    with exactly the tensors and scalars they hold. ``False`` (the caller keeps its ctypes path)
    when the single library is not loaded or carries no face of ``name``: a family-library
    deployment or a library built before a kernel then serves unchanged.
    """
    if lib is None:
        return False
    try:
        face = getattr(lib, name)
    except AttributeError:
        return False
    face(*tensors, *[int(s) for s in scalars])
    return True


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
