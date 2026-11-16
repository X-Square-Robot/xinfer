"""The unified Ascend kernel provider: pure-ctypes loading, zero the former kernel package.

The ascend backend of this branch carries its own operator sources
(``kernels/``) and their python dispatch (``glue/``); this module is the only
library-loading surface. It replaces the former former paired loader:

* one core library (``libxinfer_ascend_core.so``) for the always-on kernel
  surface behind the glue's class-style entries, discovered through
  ``XINFER_ASCEND_CORE_LIB`` (optional ``XINFER_ASCEND_CORE_SHA256`` pin and
  ``XINFER_ASCEND_CORE_REQUIRED`` fail-fast, the v22 ``REQUIRED`` semantics);
* one knife library per routed family, through the family's own
  ``XINFER_<FAMILY>_LIB`` — every environment name of the fastpaths/v22
  deployment is preserved verbatim;
* the default search path ``ops/backends/ascend/kernels/lib/<name>`` (the
  ``build.sh install`` target) for libraries whose env var is unset;
* AOT launchers bind as ``xinfer_aot_<kernel>_{call,get_last_error}`` with
  per-call current-stream resolution (the NPUGraph capture contract) and a
  per-library first-call warmup (the AOT first-launch loss pitfall).

A missing library never raises by itself: the family simply does not register
and the backend keeps the reference composition. ``XINFER_ASCEND_KERNELS=0``
disables everything (the ``XINFER_XCOMPUTE=0`` semantics, renamed).
``XINFER_ASCEND_CORE_REQUIRED=1`` turns a missing core library into an error
for deployments that pin their stack (the delivery smoke scripts).

No former-package import, no tvm_ffi, no torch linkage: ``ctypes.CDLL`` plus
CANN's runtime libraries (see kernels/README.md for the build contract).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from threading import Lock

_MASTER = "XINFER_ASCEND_KERNELS"
_CORE_ENV = "XINFER_ASCEND_CORE_LIB"
_CORE_SHA_ENV = "XINFER_ASCEND_CORE_SHA256"
_CORE_REQUIRED_ENV = "XINFER_ASCEND_CORE_REQUIRED"

_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "lib")

_LIBRARIES: dict[str, ctypes.CDLL] = {}
_WARMED: set[str] = set()
_LOCK = Lock()


def _default_library_path(name: str) -> str:
    """The ``kernels/lib/<name>`` install position (build.sh's target)."""
    return os.path.join(_LIB_DIR, name)


def _verify_sha256(path: str, expected: str) -> None:
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != expected.strip().lower():
        raise OSError(
            f"{path}: sha256 {digest} does not match the pinned {expected.strip()!r}"
        )


def library(name: str, *, env: str | None = None, default: str | None = None,
            sha_env: str | None = None, required: bool = False) -> ctypes.CDLL | None:
    """Load one shared library once per process; ``None`` when it is absent.

    ``env`` names the deployment's explicit path (``XINFER_<FAMILY>_LIB``);
    ``default`` the fallback file name inside ``kernels/lib/``. An unset or
    unloadable library degrades the calling family to "not registered" —
    deployments that must not silently fall back pass ``required=True``
    (directly or via the core's ``XINFER_ASCEND_CORE_REQUIRED``).
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
            if os.path.exists(candidate):
                path = candidate
        if path is None:
            if required or (name == "xinfer_ascend_core"
                            and os.environ.get(_CORE_REQUIRED_ENV) == "1"):
                raise OSError(
                    f"{name}: no library found (set {env or _CORE_ENV} or build "
                    "kernels/ with build.sh install)"
                )
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
        return loaded


def core_library() -> ctypes.CDLL | None:
    """The core library (class-entry kernels and the C-ABI bridges)."""
    if not enabled():
        return None
    return library(
        "xinfer_ascend_core",
        env=_CORE_ENV,
        default="libxinfer_ascend_core.so",
        sha_env=_CORE_SHA_ENV,
    )


def knife_library(family: str, *, env: str, default: str,
                  sha_env: str | None = None) -> ctypes.CDLL | None:
    """A routed family library (soup2/soupform/fia_fuse/...)."""
    if not enabled():
        return None
    return library(family, env=env, default=default, sha_env=sha_env)


def enabled() -> bool:
    """``XINFER_ASCEND_KERNELS`` (default on); ``0/off`` unregisters everything."""
    return os.environ.get(_MASTER, "on").strip().lower() not in ("", "0", "off", "false", "no")


def has_symbol(lib: ctypes.CDLL, name: str) -> bool:
    try:
        getattr(lib, name)
    except AttributeError:
        return False
    return True


def stream_of(tensor) -> ctypes.c_void_p:
    """The current NPU stream as a C pointer (per-call: NPUGraph capture records it)."""
    return ctypes.c_void_p(_torch_current_stream(tensor))


def _torch_current_stream(tensor):
    import torch

    return torch.npu.current_stream(tensor.device).npu_stream


def aot_entry(lib: ctypes.CDLL, name: str, pointers: int, scalars: int):
    """Bind ``xinfer_aot_<name>_{call,get_last_error}`` on ``lib``.

    The launcher ABI is the packed-launch contract: ``pointers`` void* tensor
    operands, ``scalars`` int64 values, then the stream (fia_fuse/soup2
    precedent). Returns ``(call, get_last_error)`` or ``None`` when the
    library predates the kernel.
    """
    try:
        call = getattr(lib, f"xinfer_aot_{name}_call")
        error = getattr(lib, f"xinfer_aot_{name}_get_last_error")
    except AttributeError:
        return None
    call.restype = ctypes.c_int32
    call.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int64] * scalars + [ctypes.c_void_p]
    error.restype = ctypes.c_char_p
    return call, error


def warmup(lib_key: str) -> bool:
    """Whether this library still needs its first eager launch on this stream.

    The AOT first-call loss: a kernel launched for the first time inside an
    NPU graph capture can silently not execute. Glue routes call ``warmup``
    before their first launch per (library, stream) and refuse to run cold
    inside a capture, exactly like the fia_fuse/gqa-glue knives did.
    """
    return lib_key in _WARMED


def mark_warm(lib_key: str) -> None:
    _WARMED.add(lib_key)


def launch(aot_pair, tensors, scalars, stream=None):
    """Run one bound launcher; raises with the library's error text on failure."""
    call, get_error = aot_pair
    if stream is None:
        stream = _torch_current_stream(tensors[0])
    status = call(
        *[ctypes.c_void_p(t.data_ptr()) for t in tensors],
        *[ctypes.c_int64(int(s)) for s in scalars],
        ctypes.c_void_p(stream),
    )
    if status:
        message = get_error()
        raise RuntimeError(
            f"xinfer_aot launcher failed: "
            f"{message.decode('utf-8', 'replace') if message else 'unknown failure'}"
        )


class RawBridge:
    """The C-ABI composition bridges of the core library (lane-3 statics).

    The v22 v22 library kept aclnn compositions (FIA probes, ACLNN Mm, the
    GDN vendor chunk, GEMM finish launchers) as internal primitives with raw
    former ``C`` symbols; the glue bound them through ctypes. The unified
    core library carries the same launchers under ``xinfer_*`` names
    (kernels/static/) and this class exposes the python callables with the
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
            launch = lib.xinfer_gdr_cann_chunk
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
