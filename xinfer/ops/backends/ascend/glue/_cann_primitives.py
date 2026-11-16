"""Absorbed CANN bridge primitives: the in-library raw C-ABI launchers.

The cann_* TVM-FFI exports are option-gated benchmark arms; the production
the v22 library kept the same ACLNN cores as internal composition primitives
(the GDR mode, npu/docs/ffi_surface_alignment_v22.md C class). The CANN
bridges bind these ctypes launchers when the paired library exports no
cann_* FFI name: identical cores, layouts and parameters, so the composed
operators keep their numerics while the FFI surface stays at the NV subset.
"""

from __future__ import annotations

import ctypes

from xinfer.ops.backends.ascend import _provider
from xinfer.ops.backends.ascend.glue.core import CORE

_library = None


def _load():
    """The unified core library through the provider.

    tvm_ffi has already loaded this object; the second dlopen only increments
    its reference count so the internal launchers resolve in-process.
    """
    global _library
    if _library is None:
        _library = _provider.core_library()
    return _library


def _symbol(library, name):
    try:
        return getattr(library, name)
    except AttributeError:
        return None


def _failure(library, getter, api):
    message = getter()
    text = message.decode("utf-8", "replace") if message else "unknown failure"
    return RuntimeError(f"{api}: {text}")


_DENSE_PAIR = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
               ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


def mm_pair():
    """``(query, launch)`` over the in-library ACLNN Mm core, or ``None``.

    The callables mirror the option-gated FFI signatures: ``query(out, x, w,
    math_type)`` returns the workspace bytes and ``launch(out, workspace, x,
    w, math_type)`` projects BF16 (math type 0) or FP32 (math type 3) rows by
    an NK weight into the FP32 ``out``. Three-dimensional BNSD ``[16, rows,
    128]`` inputs against the o-projection weights ride the in-library Cube
    kernels instead (the v22 bridge's SmallmBnsdRoute/BnsdM768Route, ported
    to the python side): the [16, rows in {16, 32, 64}, 128] forms over the
    [1024, 2048] weight take ``smallm_bnsd_o_proj{,_m32,_m64}`` with an FP32
    ``[rows, 1024]`` out, and the prefill [16, 768, 128] form over the
    [2048, 2048] weight takes ``bnsd_o_proj_m768_bf16`` with a BF16
    ``[768, 2048]`` out; both routes are workspace-free.
    """
    library = _load()
    query = _symbol(library, "xinfer_cann_mm_fp32_workspace_bytes")
    launch = _symbol(library, "xinfer_cann_mm_fp32_launch")
    if query is None or launch is None:
        return None
    error = library.xinfer_cann_mm_fp32_get_last_error
    error.restype = ctypes.c_char_p
    query.restype = ctypes.c_int64
    query.argtypes = [ctypes.c_int32, *_DENSE_PAIR, ctypes.c_int64]
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_int32, *_DENSE_PAIR, ctypes.c_void_p, ctypes.c_int64,
                       ctypes.c_int64]

    bnsd_entries = {}
    for name in ("smallm_bnsd_o_proj", "smallm_bnsd_o_proj_m32",
                 "smallm_bnsd_o_proj_m64", "bnsd_o_proj_m768_bf16"):
        call = _symbol(library, f"xinfer_aot_{name}_call")
        kern_error = _symbol(library, f"xinfer_aot_{name}_get_last_error")
        if call is None or kern_error is None:
            continue
        call.restype = ctypes.c_int32
        call.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_void_p]
        kern_error.restype = ctypes.c_char_p
        bnsd_entries[name] = (call, kern_error)

    def _bnsd_form(x, w, math_type):
        """(kernel name, out dtype) for a BNSD input this build serves, else None."""
        import torch

        if (math_type != 0 or x.ndim != 3 or not x.is_contiguous()
                or x.shape[0] != 16 or x.shape[2] != 128
                or w.ndim != 2 or not w.is_contiguous()
                or x.dtype != w.dtype or str(x.dtype) != "torch.bfloat16"
                or x.device.type != "npu" or w.device != x.device):
            return None
        rows = int(x.shape[1])
        if rows in (16, 32, 64) and tuple(w.shape) == (1024, 2048):
            return ("smallm_bnsd_o_proj" if rows == 16 else f"smallm_bnsd_o_proj_m{rows}",
                    torch.float32)
        if rows == 768 and tuple(w.shape) == (2048, 2048):
            return ("bnsd_o_proj_m768_bf16", torch.bfloat16)
        return None

    def _bnsd_validate(out, x, w, dtype):
        import torch

        rows = int(x.shape[1])
        expected = (768, 2048) if dtype == torch.bfloat16 else (rows, 1024)
        if (out.dtype != dtype or tuple(out.shape) != expected or not out.is_contiguous()
                or out.device != x.device):
            raise ValueError(
                f"BNSD o-projection out must be a contiguous {dtype} "
                f"{tuple(expected)} on the input NPU")

    def mm_query(out, x, w, math_type=0):
        form = _bnsd_form(x, w, math_type)
        if form is not None:
            if form[0] not in bnsd_entries:
                raise ValueError(f"this core library does not carry the {form[0]} BNSD kernel")
            _bnsd_validate(out, x, w, form[1])
            return 0
        m, k = x.shape
        n = w.shape[0]
        in_dims = (ctypes.c_int64 * 2)(int(m), int(k))
        w_dims = (ctypes.c_int64 * 2)(int(n), int(k))
        out_dims = (ctypes.c_int64 * 2)(int(m), int(n))
        size = query(x.get_device(), 1 if math_type == 0 else 0, in_dims, w_dims, out_dims,
                     ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(w.data_ptr()),
                     ctypes.c_void_p(out.data_ptr()), 0)
        if size < 0:
            raise _failure(library, error, "cann_mm_fp32 workspace")
        return int(size)

    def mm_launch(out, workspace, x, w, math_type=0):
        form = _bnsd_form(x, w, math_type)
        if form is not None:
            name = form[0]
            if name not in bnsd_entries:
                raise ValueError(f"this core library does not carry the {name} BNSD kernel")
            _bnsd_validate(out, x, w, form[1])
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
        m, k = x.shape
        n = w.shape[0]
        in_dims = (ctypes.c_int64 * 2)(int(m), int(k))
        w_dims = (ctypes.c_int64 * 2)(int(n), int(k))
        out_dims = (ctypes.c_int64 * 2)(int(m), int(n))
        status = launch(x.get_device(), 1 if math_type == 0 else 0, in_dims, w_dims, out_dims,
                        ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(w.data_ptr()),
                        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(workspace.data_ptr()),
                        workspace.numel(), 0)
        if status:
            raise _failure(library, error, "cann_mm_fp32 launch")

    return mm_query, mm_launch


def glu_fp32_launch():
    """``launch(out, workspace, x, w, bias, m, k, n2, paired, kn)`` or ``None``.

    The in-library FP32-through SwiGLU core: the former gemm_glu_fp32_npu FFI
    entry (ACLNN Mm projection, in-kernel FP32 bias add, segment-staged
    finish with the single final BF16 rounding), kept as an internal
    composition primitive and bound here the way the CANN bridges bind their
    cores.
    """
    library = _load()
    launch = _symbol(library, "xinfer_gemm_glu_fp32_launch")
    if launch is None:
        return None
    error = library.xinfer_gemm_glu_fp32_get_last_error
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
            raise _failure(library, error, "gemm_glu_fp32")

    return run


def compat_cube_pair():
    """``(workspace_size, launch)`` over the pinned compatibility cube host forms.

    The severed gemm_compat_npu / gemm_compat_workspace_size cores
    (CompatibilityCube kinds 4 and 5): ``workspace_size(kind, m, k, n)``
    returns the cube scratch bytes and ``launch(out, workspace, x, w, bias,
    table, kind, m)`` serves the HD80 pairs4 (kind 4, rotary 80) and
    std-wide raw-NK (kind 5, rotary 64) pinned forms.
    """
    library = _load()
    query = _symbol(library, "xinfer_gemm_compat_cube_workspace_size")
    launch = _symbol(library, "xinfer_gemm_compat_cube_launch")
    if query is None or launch is None:
        return None
    error = library.xinfer_gemm_compat_cube_get_last_error
    error.restype = ctypes.c_char_p
    query.restype = ctypes.c_int64
    query.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64]

    def cube_workspace(kind, m, k, n):
        size = query(int(kind), int(m), int(k), int(n))
        if size < 0:
            raise _failure(library, error, "gemm_compat_workspace")
        return int(size)

    def cube_launch(out, workspace, x, w, bias, table, kind, m):
        status = launch(
            x.get_device(),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(w.data_ptr()),
            ctypes.c_void_p(bias.data_ptr()) if bias is not None else None,
            ctypes.c_void_p(table.data_ptr()),
            int(kind), int(m),
        )
        if status:
            raise _failure(library, error, "gemm_compat_cube")

    return cube_workspace, cube_launch


def npu_typed_workspace_size():
    """``size(m, n, k, kn, element_bytes)`` or ``None``: the typed Ascend cube
    GEMM scratch query (the former gemm_generic_npu_workspace_size_for_dtype
    FFI entry, kept as an internal composition primitive)."""
    library = _load()
    query = _symbol(library, "xinfer_npu_typed_workspace_size")
    if query is None:
        return None
    error = library.xinfer_npu_typed_workspace_size_get_last_error
    error.restype = ctypes.c_char_p
    query.restype = ctypes.c_int64
    query.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                      ctypes.c_int64, ctypes.c_int64]

    def typed_size(m, n, k, kn, element_bytes):
        size = query(int(m), int(n), int(k), int(bool(kn)), int(element_bytes))
        if size < 0:
            raise _failure(library, error, "npu_typed_workspace")
        return int(size)

    return typed_size


def mcast_glu_cube_launch():
    """``launch(out, workspace, x, w)`` or ``None``.

    The former mcast_glu_cube_npu FFI core (McastGluLaunch with its explicit
    workspace arm: ACLNN Mm projection of the persistent block-ordered
    [32768,2048] rows plus the GeGLU finish), kept as an internal composition
    primitive for the Ascend arm of the mcast_bf16_sm110 GLU backend.
    """
    library = _load()
    launch = _symbol(library, "xinfer_mcast_glu_cube_launch")
    if launch is None:
        return None
    error = library.xinfer_mcast_glu_cube_get_last_error
    error.restype = ctypes.c_char_p
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

    def run(out, workspace, x, w):
        status = launch(
            x.get_device(),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(w.data_ptr()),
            int(x.shape[0]),
        )
        if status:
            raise _failure(library, error, "mcast_glu_cube")

    return run


def mcast_qkv_cube_launch():
    """``launch(out, workspace, x, w, cos, sin, k_cache, v_cache, start)`` or ``None``.

    The former mcast_qkv_cube_npu FFI core (McastQkvLaunch with its explicit
    workspace arm: Cube projection plus the fp32 cache-rotate finishes),
    kept as an internal composition primitive for the Ascend arm of the
    mcast_bf16_sm110 QKV backend.
    """
    library = _load()
    launch = _symbol(library, "xinfer_mcast_qkv_cube_launch")
    if launch is None:
        return None
    error = library.xinfer_mcast_qkv_cube_get_last_error
    error.restype = ctypes.c_char_p
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                       ctypes.c_int64]

    def run(out, workspace, x, w, cos, sin, k_cache, v_cache, start):
        status = launch(
            x.get_device(),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(w.data_ptr()),
            ctypes.c_void_p(cos.data_ptr()),
            ctypes.c_void_p(sin.data_ptr()),
            ctypes.c_void_p(k_cache.data_ptr()),
            ctypes.c_void_p(v_cache.data_ptr()),
            int(k_cache.shape[0]),
            int(start),
            int(x.shape[0]),
        )
        if status:
            raise _failure(library, error, "mcast_qkv_cube")

    return run


def bnsd_pair():
    """``(query, launch)`` over the in-library masked BNSD FIA V5 core, or ``None``.

    The callables mirror the option-gated FFI signatures: ``query(out, q, k,
    v, drop_mask, scale, inner_precise)`` returns the workspace bytes and
    ``launch(out, workspace, q, k, v, drop_mask, scale, inner_precise)``
    attends over grouped K/V with the bool drop mask, both on BNSD tensors.
    """
    library = _load()
    query = _symbol(library, "xinfer_cann_fia_v5_probe_workspace_size")
    launch = _symbol(library, "xinfer_cann_fia_v5_probe_launch")
    if query is None or launch is None:
        return None
    error = library.xinfer_cann_fia_v5_get_last_error
    error.restype = ctypes.c_char_p
    query.restype = ctypes.c_int64
    query.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                      ctypes.c_double, ctypes.c_int64]
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_int64), ctypes.c_double, ctypes.c_int64]

    def bnsd_query(out, q, k, v, drop_mask, scale, inner_precise):
        b, h, m, d = q.shape
        hkv, l = k.shape[1], k.shape[2]
        dims = (ctypes.c_int64 * 6)(int(b), int(h), int(m), int(d), int(hkv), int(l))
        size = query(q.get_device(), ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(q.data_ptr()),
                     ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()),
                     ctypes.c_void_p(drop_mask.data_ptr()), dims, float(scale), int(inner_precise))
        if size < 0:
            raise _failure(library, error, "cann_fia_v5_probe workspace")
        return int(size)

    def bnsd_launch(out, workspace, q, k, v, drop_mask, scale, inner_precise):
        b, h, m, d = q.shape
        hkv, l = k.shape[1], k.shape[2]
        dims = (ctypes.c_int64 * 6)(int(b), int(h), int(m), int(d), int(hkv), int(l))
        status = launch(q.get_device(), ctypes.c_void_p(out.data_ptr()),
                        ctypes.c_void_p(workspace.data_ptr()), workspace.numel(),
                        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                        ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(drop_mask.data_ptr()),
                        dims, float(scale), int(inner_precise))
        if status:
            raise _failure(library, error, "cann_fia_v5_probe launch")

    return bnsd_query, bnsd_launch


def tnd_pair():
    """``(query, launch)`` over the in-library packed TND FIA V5 core, or ``None``.

    The callables mirror the option-gated FFI signatures: ``query(out, q, k,
    v, endpoints, scale)`` returns the workspace bytes and ``launch(out,
    workspace, q, k, v, endpoints, scale)`` attends within the static packed
    segments; ``endpoints`` are the positive cumulative offsets.
    """
    library = _load()
    query = _symbol(library, "xinfer_cann_fia_vision_probe_workspace_size")
    launch = _symbol(library, "xinfer_cann_fia_vision_probe_launch")
    if query is None or launch is None:
        return None
    error = library.xinfer_cann_fia_v5_get_last_error
    error.restype = ctypes.c_char_p
    query.restype = ctypes.c_int64
    query.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
                      ctypes.POINTER(ctypes.c_int64), ctypes.c_double]
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
                       ctypes.POINTER(ctypes.c_int64), ctypes.c_double]

    def _pack(q, endpoints):
        tokens, heads, dim = q.shape
        bounds = tuple(int(end) for end in endpoints)
        dims = (ctypes.c_int64 * 3)(int(tokens), int(heads), int(dim))
        ends = (ctypes.c_int64 * len(bounds))(*bounds)
        return dims, ends

    def tnd_query(out, q, k, v, endpoints, scale):
        dims, ends = _pack(q, endpoints)
        size = query(q.get_device(), ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(q.data_ptr()),
                     ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()), ends, len(ends),
                     dims, float(scale))
        if size < 0:
            raise _failure(library, error, "cann_fia_vision_probe workspace")
        return int(size)

    def tnd_launch(out, workspace, q, k, v, endpoints, scale):
        dims, ends = _pack(q, endpoints)
        status = launch(q.get_device(), ctypes.c_void_p(out.data_ptr()),
                        ctypes.c_void_p(workspace.data_ptr()), workspace.numel(),
                        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                        ctypes.c_void_p(v.data_ptr()), ends, len(ends), dims, float(scale))
        if status:
            raise _failure(library, error, "cann_fia_vision_probe launch")

    return tnd_query, tnd_launch
