"""Internal Ascend ACLNN Mm projection with the FP32 epilogue boundary."""

from __future__ import annotations
from xinfer.ops.backends.ascend import _native

import atexit
import os

from contextlib import contextmanager
from threading import Lock

import torch
import torch.nn.functional as F
from xinfer.ops.backends.ascend.core import CORE
from xinfer.ops.backends.ascend.rmsnorm import add_rmsnorm as _add_rmsnorm
try:  # r6a-m16epi A1 (XINFER_SMALLM_R3=gateup, default off)
    from xinfer.ops.backends.ascend import smallm_r3
except Exception:  # pragma: no cover - optional mount
    smallm_r3 = None
try:  # glu-epilogue K-B (XINFER_GLUEPI=prefillglu, default off)
    from xinfer.ops.backends.ascend import glu_epi
except Exception:  # pragma: no cover - optional mount
    glu_epi = None
try:  # finish-bf16 (XINFER_FINISHBF16, default off)
    from xinfer.ops.backends.ascend import finishbf16
except Exception:  # pragma: no cover - optional mount
    finishbf16 = None
try:  # copykill-r2 integration flags (XINFER_COPYKILL_R2, default off)
    from xinfer.ops.backends.ascend.copykill_r2 import enabled as _copykill_enabled
except Exception:  # pragma: no cover - optional mount
    def _copykill_enabled(_family):
        return False
try:  # INT-WOSS g2112-r3 flag-isolated routes (XINFER_G2112_*, default off)
    from xinfer.ops.backends.ascend.g2112_tilelang import (
        G2112_FLAGS, glu_route, mm_fp32_route, down_route, accumulator_route, samu_route,
    )
except Exception:  # pragma: no cover - optional mount
    G2112_FLAGS = {name: False for name in ("GLU", "VITQKV", "DOWN", "PREFQKV", "TORCH_A", "TORCH_C")}
    glu_route = mm_fp32_route = down_route = accumulator_route = samu_route = None

_ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda value: F.gelu(value, approximate="tanh"),
}
_FLOAT_DTYPES = (torch.bfloat16, torch.float32)
_MAX_I32 = 2**31 - 1

# INT-WOSS COPYKILL-r2 K4: memoized FP32 bias copies (one cast per bias
# storage identity, 64-entry LRU clear).
_BIAS_FP32_MEMO = {}


def _bias_fp32(bias):
    key = (bias.data_ptr(), tuple(bias.shape), bias.dtype, bias.device.index)
    entry = _BIAS_FP32_MEMO.get(key)
    if entry is not None:
        return entry[1]
    value = bias.float()
    if len(_BIAS_FP32_MEMO) >= 64:
        _BIAS_FP32_MEMO.clear()
    _BIAS_FP32_MEMO[key] = (bias, value)
    return value


# mx8-r3: opt-in routes for the wall-oss prefill down projection family.
_MX8_DOWN_MODE = os.environ.get("XINFER_MX8_DOWN_MODE", "off")
if _MX8_DOWN_MODE == "off":
    # splitk outranks prepack when both are exported (the final-consolidate
    # arm sets both; splitk is the form the 38.82 stack actually ran -- its
    # down_bmm 3.1ms shows in the msprof ledger).
    if os.environ.get("XINFER_MX8_DOWN_SPLITK", "0") not in ("", "0", "off"):
        _MX8_DOWN_MODE = "splitk"
    elif os.environ.get("XINFER_MX8_DOWN_PREPACK", "0") not in ("", "0", "off"):
        _MX8_DOWN_MODE = "prepack"
_MX8_DOWN_SPLITK = 4

_GAP_UBF_READY = None


def _gap_ubf_routes(m, n):
    """GAP-WOSS-NONGE k2: compiled+tileable check for libubfinish_r5."""
    global _GAP_UBF_READY
    if _GAP_UBF_READY is None:
        try:
            from xinfer.ops.backends.ascend import ubfinish_r5

            _GAP_UBF_READY = ubfinish_r5.routes if ubfinish_r5.enabled() else (lambda rows, cols: False)
        except Exception:  # noqa: BLE001 - any miss keeps the released chain
            _GAP_UBF_READY = lambda rows, cols: False
    return _GAP_UBF_READY(m, n)

# gemm-audit K1 (XINFER_GEMM_BF16DIRECT=on, default off): see the route below.
_BF16DIRECT_ON = os.environ.get("XINFER_GEMM_BF16DIRECT", "") == "on"
_BF16DIRECT_STATS = os.environ.get("XINFER_GEMM_BF16DIRECT_STATS")
_BF16DIRECT_COUNT = {"routed": 0}


def _bf16direct_dump():
    if _BF16DIRECT_STATS:
        import json

        try:
            with open(f"{_BF16DIRECT_STATS}-pid{os.getpid()}", "w") as fh:
                fh.write(json.dumps(_BF16DIRECT_COUNT))
        except Exception:  # noqa: BLE001 - a stats receipt must never break serving
            pass



atexit.register(_bf16direct_dump)



def _geometry(x, weight):
    # r4: the attention output buffer in BNSD [16, 16, 128] head-major order
    # against the [1024, 2048] o-projection weight. cann_mm_fp32 routes it to
    # the TileLang BNSD Cube kernel (the out-finish transpose folds into the
    # A-load; K order matches the ACLNN Mm bitwise on this shape).
    # qkv-batch r1: the B2/B4 request-batch rows widen the accepted A to
    # [16, rows, 128] with rows in {16, 32, 64}; the 32/64-row kernel forms
    # keep the same per-row K order (16 sequential head steps).
    if x.ndim == 3 and x.shape[0] == 16 and x.shape[2] == 128 and x.shape[1] in (16, 32, 64):
        if (weight.ndim == 2 and tuple(weight.shape) == (1024, 2048)
                and x.device.type == "npu" and weight.device == x.device
                and x.dtype == weight.dtype == torch.bfloat16 and x.is_contiguous()):
            return int(x.shape[1]), 1024, 2048
        raise ValueError("BNSD input requires the [1024, 2048] KN o-projection weight on the same NPU")
    if (
        x.ndim != 2
        or weight.ndim != 2
        or x.shape[1] != weight.shape[1]
        or x.device.type != "npu"
        or weight.device != x.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
    ):
        raise ValueError("CANN projection requires BF16 [M,K]/[N,K] on the same NPU")
    m, k = x.shape
    n = weight.shape[0]
    if not all(0 < dim <= _MAX_I32 for dim in (m, n, k)):
        raise ValueError("CANN projection dimensions must be positive int32")
    return m, n, k


# SOUP2 finplain: the plain-finish prefill GEMM envelope. Every (k, n) below
# is probed bitwise (matmul-bf16 vs Mm-fp32+cast, zero flips across the
# production M values 16..868; SOUP2 probe2 receipt). The m16 pair stays in
# the original arm so the released behavior is unchanged when SOUP2 is off.
_SOUP2_FINPLAIN_SHAPES = {(2048, 2048), (2048, 5120), (1024, 1536), (1024, 4096), (2048, 1024)}


def _soup2_family_ok(family):
    try:
        from xinfer.ops.backends.ascend import soup2

        return soup2.enabled(family)
    except Exception:  # noqa: BLE001 - any miss keeps the released chain
        return False


_soup2_finplain_ok = lambda: _soup2_family_ok("finplain")


def _check_tensor(tensor, name, device, shape, dtypes):
    if tensor is None or tensor.device != device or tensor.shape != shape or tensor.dtype not in dtypes:
        raise ValueError(f"{name} must have shape {shape}, dtype in {dtypes}, and device {device}")


def _finish(y, bias, activation, residual, residual_gate, row_class, row_scale):
    # SOUPKILL knife A (opt-in, XINFER_SOUPKILL=finish): the plain bias+residual
    # finish of a 1024-wide FP32 accumulator is one SIMD pass over the
    # accumulator -- same two FP32 adds in the same order, one final RNE. Any
    # envelope miss, or a deployment without the knife library, returns None and
    # the released chain below runs unchanged.
    if (
        bias is not None
        and residual is not None
        and activation is None
        and residual_gate is None
        and row_class is None
        and row_scale is None
        and y.dtype == torch.float32
        and y.dim() == 2
        and y.shape[1] == 1024
    ):
        from xinfer.ops.backends.ascend import soupkill

        fused = soupkill.finish_bias_res(
            y, bias, residual, y.new_empty(tuple(y.shape), dtype=torch.bfloat16)
        )
        if fused is not None:
            return fused
    # SOUP2 fc1gelu (opt-in, XINFER_SOUP2=fc1gelu): ONE kernel does the bias
    # add (same fp32 vadd, bitwise) + tanh-form gelu + the single bf16 RNE,
    # replacing the aclnnAdd + aclnnGeluV2 + aclnnCast tail. NOT bitwise vs
    # aclnnGeluV2 (vendor transcendental); the action gate adjudicates.
    if (
        bias is not None and activation == "gelu_tanh" and residual is None
        and residual_gate is None and row_class is None and row_scale is None
        and y.dtype == torch.float32 and y.dim() == 2 and y.shape[1] == 4096
        and y.is_contiguous() and y.device.type == "npu"
        and _soup2_family_ok("fc1gelu")
    ):
        from xinfer.ops.backends.ascend import soup2

        fused = soup2.fc1gelu(y, bias.float(), y.new_empty(tuple(y.shape), dtype=torch.bfloat16))
        if fused is not None:
            return fused
    # INT-WOSS COPYKILL-r2 K5: single-kernel biased BF16 round. Off keeps the
    # walloss two-kernel chain byte-for-byte; K4 independently controls memo.
    if (_copykill_enabled("k5") and bias is not None and activation is None
            and residual is None and residual_gate is None and row_scale is None
            and y.is_contiguous()):
        out = torch.empty(y.shape, dtype=torch.bfloat16, device=y.device)
        torch.add(y, _bias_fp32(bias), out=out)
        return out
    if bias is not None:
        y = y + (_bias_fp32(bias) if _copykill_enabled("k4") else bias.float())
    if row_scale is not None:
        y = y * row_scale[:, None]
    if activation is not None:
        y = _ACTIVATIONS[activation](y)
    if residual is None:
        return y.to(torch.bfloat16)
    if residual_gate is not None:
        # Only the gated branch rounds here. The residual add rounds again.
        y = (y * residual_gate.index_select(0, row_class).float()).to(torch.bfloat16)
        if residual.dtype == torch.bfloat16:
            return residual + y
        y = y.float()
    return (residual.float() + y).to(torch.bfloat16)


def _aot_ctypes_entry(name):
    """The internal AOT launcher ``xinfer_aot_<name>_call`` of ``name``, or None.

    The +2-pending FFI rule of the v22 face: new walloss kernels land as internal
    launchers, not __tvm_ffi_ exports. The AOT build exports plain C symbols
    (``..._call`` taking the kernel's parameters in order plus the stream, then
    ``..._get_last_error``); ctypes reaches them without touching the FFI face.
    Returns None when the symbols are absent (older libs): callers fall back.
    """
    import ctypes

    try:

        lib = _native.core_library()
        call = getattr(lib, f"xinfer_aot_{name}_call", None)
        error = getattr(lib, f"xinfer_aot_{name}_get_last_error", None)
    except (OSError, AttributeError, ImportError):
        return None
    if call is None or error is None:
        return None
    call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
    call.restype = ctypes.c_int32
    error.restype = ctypes.c_char_p

    def launch(x, weight, bias, cos, sin, out, rows, columns, reduction, stream):
        status = call(x.data_ptr(), weight.data_ptr(), bias.data_ptr(), cos.data_ptr(),
                      sin.data_ptr(), out.data_ptr(), rows, columns, reduction, stream)
        if status:
            raise RuntimeError(f"{name} failed: {error().decode('utf-8', 'replace')}")

    return launch


class CannProjection:
    """BF16-input, FP32-output ACLNN Mm and the Wall-style BF16 epilogue.

    The provider owns temporary FP32 output and workspace per stream and
    shape. Warm up each on its actual stream before capture, and keep this
    instance alive until its graphs and pending work are retired. Returned
    BF16 tensors do not alias the provider's buffers.
    """

    def __init__(self):
        self._query = getattr(CORE, "cann_mm_fp32_workspace_size", None)
        self._launch = getattr(CORE, "cann_mm_fp32", None)
        if not (callable(self._query) and callable(self._launch)):
            # Absorbed route (ffi_surface_alignment_v22.md C class): the
            # production library keeps the ACLNN Mm core as an internal
            # composition primitive without the cann_* FFI export, so bind
            # its raw C-ABI launcher through ctypes. Older builds without
            # either entry raise AttributeError and the CANN arms degrade
            # as before.
            from xinfer.ops.backends.ascend import _cann_primitives

            pair = _cann_primitives.mm_pair()
            if pair is None:
                raise AttributeError(
                    "the unified core library has neither the cann_mm pair "
                    "nor the in-library ACLNN Mm core"
                )
            self._query, self._launch = pair
        # r3: the fused FP32 SwiGLU finish (gemm_glu_fp32_finish_split_2048 via
        # the gemm_epilogue_glu FFI entry) for the action-stage [M, 4096] split
        # halves; the four-op torch epilogue stays for every other shape.
        # glu-r1: the prefill [768, 22016] fused finish rides the existing
        # gemm_epilogue_glu entry; the vision [2112, 6912] bias-folded chain
        # rides gemm_glu_fp32_npu (its own shared ACLNN Mm + bias-staged
        # split_3456 finish). Both are reused FFI names -- the surface is
        # unchanged.
        self._glu_split_fp32_finish = getattr(CORE, "gemm_epilogue_glu", None)
        self._glu_fp32_deferred = getattr(CORE, "gemm_glu_fp32_npu", None)
        # r5: the wall_oss vision QKV fused finish (one UB pass over the
        # borrowed FP32 accumulator; the torch rope chain measured ~113 us on
        # the [2112, 3840] rows, the kernel ~10.6 us, probe-r5).
        self._vision_rope_finish = getattr(CORE, "gemm_qkv_vision_rope_finish", None)
        # r10: the fused vision QKV Cube (gemm + bias + rotate-half RoPE + one RNE in
        # one launch; the FP32 accumulator never round-trips GM). Internal mount via
        # the lib's AOT launcher symbols -- no new FFI export.
        self._vision_rope_cube = _aot_ctypes_entry("gemm_qkv_vision_rope_cube")
        self._vision_cube_warmed = False
        self._buffers = {}
        self._m768_warmed = False
        # 2b-m16: the wallx2b action feeds o_t/down_t as fresh transposed
        # [N, K] views of persistent [K, N] parameters; materializing the
        # row-major weight per call costs a 4 MB device transpose inside
        # every captured replay (probe card0: 12.4 -> 5.0 us for the M16
        # o-projection once the copy is cached). Keyed by the storage
        # identity of the source view and pinned to the source's lifetime.
        self._contig_weights = {}
        # mx8-r3: flag-isolated routes for the wall-oss prefill down
        # projection family ([M, 11008] x [2048, 11008]).
        self._mx8_kn = {}
        self._mx8_split = {}
        self._lock = Lock()

    def _stable_contiguous(self, weight):
        if weight.is_contiguous():
            return weight
        key = (
            weight.data_ptr(),
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.dtype,
        )
        entry = self._contig_weights.get(key)
        if entry is None:
            copy = weight.contiguous()
            # Keep the source view (and its storage) alive so the identity
            # cannot be recycled by a later allocation.
            self._contig_weights[key] = (weight, copy)
            return copy
        return entry[1]

    def _mx8_down(self, x, weight):
        """mx8-r3 flag routes for the [M, 11008] x [2048, 11008] down projection.

        Off by default (XINFER_MX8_DOWN_MODE=off keeps the released shared
        FP32-accumulator chain). Prepack: one-time [K, N] copy per storage
        identity and a single-round BF16 torch.mm -- bitwise with
        mm(x, W.t()) (op-precheck card 0) and 147.6 -> 136.1 us on M=768.
        Splitk: one-time [4, 2752, 2048] chunk stack, per-call strided-view
        bmm + BF16 sum, 110.9 us -- the partials round 4x in BF16 (relerr
        5.6e-3 vs the chain's 2.4e-3), so adoption rides the quick100 gate.
        """
        x = x.contiguous()
        if _MX8_DOWN_MODE == "prepack":
            key = (weight.data_ptr(), tuple(weight.shape), weight.dtype)
            entry = self._mx8_kn.get(key)
            if entry is None:
                if torch.npu.is_current_stream_capturing():
                    raise RuntimeError("warm up the mx8 down prepack route before graph capture")
                copy = weight.t().contiguous()
                self._mx8_kn[key] = (weight, copy)
                entry = self._mx8_kn[key]
            return torch.mm(x, entry[1])
        if _MX8_DOWN_MODE == "splitk":
            key = (weight.data_ptr(), tuple(weight.shape), weight.dtype)
            entry = self._mx8_split.get(key)
            if entry is None:
                if torch.npu.is_current_stream_capturing():
                    raise RuntimeError("warm up the mx8 down splitk route before graph capture")
                stacked = weight.t().reshape(_MX8_DOWN_SPLITK, -1, weight.shape[0]).contiguous()
                self._mx8_split[key] = (weight, stacked)
                entry = self._mx8_split[key]
            rows = x.shape[0]
            view = x.reshape(rows, _MX8_DOWN_SPLITK, -1).permute(1, 0, 2)
            return torch.bmm(view, entry[1]).sum(0)
        raise RuntimeError(f"unknown XINFER_MX8_DOWN_MODE {_MX8_DOWN_MODE!r}")

    @contextmanager
    def _project(self, x, weight, shape):
        m, n, k = shape
        device = x.device
        x = x.contiguous()
        weight = self._stable_contiguous(weight)
        stream = torch.npu.current_stream(device).npu_stream
        # The BNSD layout carries its own (workspace-free) route; keep its
        # buffers apart from the plain [M, K] shape of the same geometry.
        key = (device, stream, m, n, k, x.ndim)
        # Hold the lock until the epilogue has enqueued its consumers of the
        # temporary output. Calls sharing a stream can then safely reuse it.
        with self._lock:
            if key not in self._buffers:
                if torch.npu.is_current_stream_capturing():
                    raise RuntimeError("warm up CANN projection on this stream and shape before graph capture")
                output = torch.empty((m, n), dtype=torch.float32, device=device)
                size = self._query(output, x, weight, 0)
                workspace = torch.empty(size, dtype=torch.uint8, device=device)
                self._buffers[key] = output, workspace
            output, workspace = self._buffers[key]
            self._launch(output, workspace, x, weight, 0)
            yield output

    def accumulator(self, x, weight):
        """Borrow FP32 `[M,N]` inside a context, scheduling every consumer before exit."""
        # INT-WOSS G2112-r3 PREFQKV, default off.
        if x.ndim == 2 and weight.ndim == 2:
            m, k = x.shape
            if (G2112_FLAGS["PREFQKV"] and (m, k) == (768, 2048)
                    and tuple(weight.shape) == (2560, 2048)
                    and x.dtype == weight.dtype == torch.bfloat16
                    and x.is_contiguous() and weight.is_contiguous()
                    and x.device.type == "npu" and weight.device == x.device):
                return accumulator_route(x, weight, "prefqkv")
        return self._project(x, weight, _geometry(x, weight))

    def gemm_add_rms_norm(self, x, weight, residual, norm_weight, epsilon):
        """r3 (agent H): bias-free projection + the f32in add+rms fast plan.

        The FP32 accumulator feeds the kernel directly; the RNE round to BF16
        happens inside it -- bit-identical to the released chain
        ``gemm_epilogue`` (whose ``_finish`` casts) followed by ``add_rmsnorm``,
        with the separate cast kernel removed. The output pair are fresh BF16
        tensors; the accumulator borrow is stream-ordered under the provider
        lock. Envelope: bias-free [M,K]x[N,K] BF16 operands, BF16 residual,
        FP32/BF16 norm weight; the kernel itself enforces M16/H1024 (any
        other geometry raises and the caller falls back to the reference).
        """
        with self._project(x, weight, _geometry(x, weight)) as y:
            if y.shape != residual.shape:
                if y.numel() != residual.numel():
                    raise ValueError(
                        "gemm_add_rms_norm projection ["
                        + "x".join(str(d) for d in y.shape)
                        + "] does not match residual ["
                        + "x".join(str(d) for d in residual.shape) + "]")
                y = y.reshape(residual.shape)
            return _add_rmsnorm(residual, y, norm_weight, epsilon, inference=True)

    def copykill_f32in_available(self):
        """INT-WOSS COPYKILL-r2 K2 capability: the core library's f32in AOT pair.

        The v22 delivery's probe resolved the pair from its combined library;
        the unified core carries the same kernel (rms_consistent_generic_h1024_f32in,
        the leaf battery's bitwise-verified entry), so the probe resolves it there.
        """
        lib = _native.core_library()
        return (lib is not None
                and _native.has_symbol(lib, "xinfer_aot_rms_consistent_generic_h1024_f32in_call"))

    def _bnsd_m768(self, x, weight):
        """The r8 prefill M768 BNSD route: BF16 out, zero workspace, no shared buffers.

        Any M<=768 (the flow file is 752; other recorded widths differ) pads
        into a fresh [16, 768, 128] buffer: the garbage tail rows only ever
        compute into discarded output rows, so every valid row stays bitwise
        vs the packed chain (output rows are independent).
        """
        rows = int(x.shape[1])
        if torch.npu.is_current_stream_capturing() and not self._m768_warmed:
            raise RuntimeError("warm up the BNSD M768 o-projection on this stream before graph capture")
        a = x
        if rows != 768:
            a = x.new_empty((16, 768, 128), dtype=torch.bfloat16)
            a[:, :rows].copy_(x)
        out = a.new_empty((768, 2048), dtype=torch.bfloat16)
        empty = a.new_empty((0,), dtype=torch.uint8)
        if not self._m768_warmed:
            size = self._query(out, a, weight, 0)
            if size:
                raise RuntimeError("the BNSD M768 o-projection route must not need workspace")
            self._m768_warmed = True
        self._launch(out, empty, a, weight, 0)
        return out if rows == 768 else out[:rows]

    def _glu_deferred_workspace(self, x, weight, shape):
        """Per-stream uint8 workspace for the gemm_glu_fp32_npu entry."""
        m, n2, k = shape
        device = x.device
        stream = torch.npu.current_stream(device).npu_stream
        key = (device, stream, m, n2, k, "glu-deferred")
        with self._lock:
            ws = self._buffers.get(key)
            if ws is None:
                if torch.npu.is_current_stream_capturing():
                    raise RuntimeError(
                        "warm up the deferred GLU projection on this stream and shape before graph capture"
                    )
                size = int(
                    CORE.gemm_generic_npu_workspace_size_for_dtype(m, n2, k, False, 2)
                )
                ws = torch.empty(size, dtype=torch.uint8, device=device)
                self._buffers[key] = ws
            return ws

    def gemm_epilogue(
        self,
        x,
        weight,
        bias=None,
        *,
        activation=None,
        residual=None,
        residual_gate=None,
        row_class=None,
        row_scale=None,
    ):
        """Project `[M,K]` by NK weights, apply the FP32 epilogue, and return BF16."""
        # INT-WOSS G2112-r3 TORCH_C/DOWN, disjoint from MX8/SMALLM shapes.
        if ((G2112_FLAGS["DOWN"] or G2112_FLAGS["TORCH_C"])
                and bias is None and activation is None and residual is None
                and residual_gate is None and row_class is None and row_scale is None
                and x.ndim == 2 and weight.ndim == 2
                and tuple(weight.shape) == (1280, 3456) and x.shape[0] == 2112
                and x.dtype == weight.dtype == torch.bfloat16
                and x.device.type == "npu" and weight.device == x.device):
            if G2112_FLAGS["DOWN"]:
                return down_route(x, weight)
            return torch.mm(x, weight.t())
        # mx8-r3: flag-isolated routes for the prefill e0 down projection
        # ([M, 11008] x [2048, 11008], trivial epilogue only). Default off.
        if (
            _MX8_DOWN_MODE != "off"
            and bias is None and activation is None and residual is None
            and residual_gate is None and row_class is None and row_scale is None
            and x.ndim == 2 and weight.ndim == 2
            and tuple(weight.shape) == (2048, 11008)
            and x.dtype == weight.dtype == torch.bfloat16
            and x.device.type == "npu" and weight.device == x.device
        ):
            return self._mx8_down(x, weight)
        # 2b-m16: the wallx2b action o/down projection -- BF16 [16, 2048] rows
        # against the [1024, 2048] weight with a trivial epilogue (the finish
        # would only cast) -- is a single aclnnMatmul BF16 store: one launch
        # replacing the ACLNN Mm FP32 accumulator plus the cast kernel
        # (probe card0: 3.92 vs 5.03 us), bitwise with the released chain on
        # this shape (the aclnn kernel accumulates FP32 and rounds once in
        # its final store, the same RNE the cast applies to the FP32 result).
        if (
            bias is None and activation is None and residual is None
            and residual_gate is None and row_class is None and row_scale is None
            and x.ndim == 2 and weight.ndim == 2
            and x.dtype == weight.dtype == torch.bfloat16
            and x.device.type == "npu" and weight.device == x.device
            and x.is_contiguous()
            and (
                (tuple(x.shape) == (16, 2048) and tuple(weight.shape) == (1024, 2048))
                or (
                    _soup2_finplain_ok()
                    and 1 <= x.shape[0] <= 8192
                    and (x.shape[1], weight.shape[0]) in _SOUP2_FINPLAIN_SHAPES
                )
            )
        ):
            # EPILOGUE-R2 blade A (XINFER_SMALLM_R3=down, default off): the
            # m16 down/o_proj [16,2048]x[1024,2048] plain store rides the
            # bitwise smallm_r3_down Cube kernel (smallm-r3 receipt: 0 flips
            # vs the aclnn chain on this exact shape). Flag off keeps the
            # released torch.matmul verbatim. CAPACITY-PREVIEW: the explicit
            # m16 guard keeps the blade inside its audited envelope now that
            # the widened branch also admits the SOUP2 finplain shapes.
            if (
                smallm_r3 is not None and smallm_r3.enabled("down")
                and tuple(x.shape) == (16, 2048) and tuple(weight.shape) == (1024, 2048)
            ):
                return smallm_r3.down(x, self._stable_contiguous(weight))
            if tuple(x.shape) == (16, 2048) and tuple(weight.shape) == (1024, 2048):
                return torch.matmul(x, self._stable_contiguous(weight).t())
            from xinfer.ops.backends.ascend import soup2

            return soup2.finplain(x, self._stable_contiguous(weight).t())
        if (
            _BF16DIRECT_ON
            and bias is None and activation is None and residual is None
            and residual_gate is None and row_class is None and row_scale is None
            and x.ndim == 2 and weight.ndim == 2
            and x.dtype == weight.dtype == torch.bfloat16
            and x.device.type == "npu" and weight.device == x.device
            and x.is_contiguous()
            and tuple(x.shape) == (464, 2048)
            and tuple(weight.shape) in ((2048, 2048), (5120, 2048))
        ):
            # gemm-audit K1: trivial-epilogue prefill shapes route to one
            # aclnnMatmul BF16 store (FP32 accumulate, single RNE, bitwise vs
            # the Mm FP32 accumulator + Cast pair on both audited shapes; iso
            # v2 card 5: -2.7/-5.3 us per call, hot==rot4).
            _BF16DIRECT_COUNT["routed"] += 1
            return torch.matmul(x, self._stable_contiguous(weight).t())
        # smallm-r3: opt-in 32-block single-shot BNSD o-projection kernel
        # (the shipped smallm_bnsd_o_proj runs 16 blocks; this one also folds
        # the follow-up FP32->BF16 cast slot). Bitwise vs the shipped kernel
        # + cast chain on the twin-fp32 reference (aot-verify, card 0).
        if (smallm_r3 is not None and smallm_r3.enabled("oproj")
                and bias is None and activation is None and residual is None
                and residual_gate is None and row_class is None and row_scale is None
                and x.ndim == 3 and tuple(x.shape) == (16, 16, 128)
                and weight.ndim == 2 and tuple(weight.shape) == (1024, 2048)
                and x.dtype == weight.dtype == torch.bfloat16
                and x.is_contiguous() and weight.is_contiguous()
                and x.device.type == "npu" and weight.device == x.device):
            return smallm_r3.o_proj(x, weight)
        # r8: the prefill causal out arrives BNSD [16, 768, 128] head-major.
        # The dedicated Cube kernel folds the packed reshape copy and the
        # finish round (bitwise vs the composite chain, probe r8); only the
        # trivial epilogue qualifies -- every other finish keeps the FP32
        # accumulator route. Unknown 3-D layouts still fall through to
        # _geometry's ValueError.
        if (x.ndim == 3 and bias is None and activation is None and residual is None
                and residual_gate is None and row_class is None and row_scale is None
                and x.dtype == weight.dtype == torch.bfloat16
                and x.is_contiguous() and weight.is_contiguous()
                and x.device.type == "npu" and weight.device == x.device):
            # r8b/r8d: the flow prefill is M=752 and other recorded widths
            # differ (the rtc file crashed e2e-r8c at a third width) -- any
            # M<=768 pads into the M768 kernel inside the route, M=768 feeds
            # it directly (output rows are independent, so the bitwise-vs-
            # packed-chain argument holds for every row count). The [16, 16,
            # 128] action input keeps its r4 BNSD case inside _geometry; any
            # other 3-D shape flattens to the exact pre-r8 packed view
            # (fallback beats loud error: two e2e crashes taught it).
            if (tuple(weight.shape) == (2048, 2048)
                    and x.shape[0] == 16 and x.shape[2] == 128
                    and 0 < x.shape[1] <= 768):
                return self._bnsd_m768(x, weight)
            if tuple(x.shape) == (16, 16, 128):
                pass  # the r4 action BNSD case routes inside _geometry
            elif (x.ndim == 3 and x.shape[0] == 16 and x.shape[2] == 128
                    and x.shape[1] in (32, 64)
                    and tuple(weight.shape) == (1024, 2048)):
                # Union fix (integration r2): the qkv-batch M32/64 action
                # rows must reach _geometry's 3-D case (the smallm m32/m64
                # fast plans). The r8 pass-list (16, 16, 128) predates them,
                # so the merge let the else-branch transpose these rows back
                # to the pre-r4 composite (9.42us vs the 6.58us receipt) and
                # stranded the m32/m64 kernels.
                pass
            else:
                x = x.transpose(0, 1).reshape(x.shape[1], -1)
        # FINISH-BF16 (XINFER_FINISHBF16, default off): the vision bias+
        # residual (finres) and bias+gelu_tanh (gelu) finishes read one
        # aclnnMatmul BF16 store (FP32 accumulate, single RNE -- bitwise the
        # value the FP32 chain rounded, GEMM-AUDIT iso v2) instead of the
        # aclnnMm FP32 accumulator, halving the finish's accumulator read.
        if (
            finishbf16 is not None and finishbf16.enabled("finres")
            and activation is None and bias is not None and residual is not None
            and residual_gate is None and row_class is None and row_scale is None
            and x.ndim == 2 and weight.ndim == 2
            and x.dtype == weight.dtype == torch.bfloat16
            and x.device.type == "npu" and weight.device == x.device
            and x.is_contiguous() and weight.is_contiguous()
            and bias.is_contiguous() and bias.dtype == torch.bfloat16
            and residual.is_contiguous() and residual.dtype == torch.bfloat16
            and tuple(x.shape) == (868, weight.shape[1])
            and tuple(weight.shape) in ((1024, 1024), (1024, 4096))
            and tuple(bias.shape) == (1024,)
            and tuple(residual.shape) == (868, 1024)
        ):
            fused = finishbf16.finish_br_bf16(x, weight, bias, residual)
            if fused is not None:
                return fused
        if (
            finishbf16 is not None and finishbf16.enabled("gelu")
            and activation == "gelu_tanh" and bias is not None and residual is None
            and residual_gate is None and row_class is None and row_scale is None
            and x.ndim == 2 and weight.ndim == 2
            and x.dtype == weight.dtype == torch.bfloat16
            and x.device.type == "npu" and weight.device == x.device
            and x.is_contiguous() and weight.is_contiguous()
            and bias.is_contiguous() and bias.dtype == torch.bfloat16
            and tuple(x.shape) == (868, weight.shape[1])
            and tuple(weight.shape) == (4096, 1024)
            and tuple(bias.shape) == (4096,)
        ):
            fused = finishbf16.gelu_bf16(x, weight, bias)
            if fused is not None:
                return fused
        shape = _geometry(x, weight)
        m, n, _ = shape
        if activation is not None and activation not in _ACTIVATIONS:
            raise ValueError("activation must be None, silu, gelu, or gelu_tanh")
        device = x.device
        if bias is not None:
            _check_tensor(bias, "bias", device, (n,), _FLOAT_DTYPES)
        if row_scale is not None:
            _check_tensor(row_scale, "row_scale", device, (m,), (torch.float32,))
        if residual is not None:
            _check_tensor(residual, "residual", device, (m, n), _FLOAT_DTYPES)
        if residual_gate is not None:
            if residual is None or row_class is None or residual_gate.ndim != 2 or not residual_gate.shape[0]:
                raise ValueError("residual_gate needs a nonempty [R,N] table, residual, and row_class")
            _check_tensor(residual_gate, "residual_gate", device, (residual_gate.shape[0], n), _FLOAT_DTYPES)
            _check_tensor(row_class, "row_class", device, (m,), (torch.int32, torch.int64))
        elif row_class is not None:
            raise ValueError("row_class requires residual_gate")

        # smallm-r3: opt-in action down-projection kernel ([16,2048]x
        # [1024,2048], trivial epilogue): 32-block single-shot W stream, the
        # BF16 store folds the cast round (bitwise on the twin reference).
        if (smallm_r3 is not None and smallm_r3.enabled("down")
                and bias is None and activation is None and residual is None
                and residual_gate is None and row_class is None and row_scale is None
                and x.ndim == 2 and weight.ndim == 2
                and tuple(x.shape) == (16, 2048)
                and tuple(weight.shape) == (1024, 2048)
                and x.dtype == weight.dtype == torch.bfloat16
                and x.is_contiguous() and weight.is_contiguous()
                and x.device.type == "npu" and weight.device == x.device):
            return smallm_r3.down(x, weight)
        # layout-oproj K1: the decode attention hands back its output
        # token-major [16, 2048], so the o-projection now sees the same
        # contiguous A as the down site (same GEMM body, same k order, same
        # RNE store round -> bitwise). Kept after the released down branch:
        # the two sites are shape-identical, so in the combined arm the down
        # branch still owns both and this only serves arms with down off.
        if (smallm_r3 is not None and smallm_r3.enabled("oproj")
                and bias is None and activation is None and residual is None
                and residual_gate is None and row_class is None and row_scale is None
                and x.ndim == 2 and weight.ndim == 2
                and tuple(x.shape) == (16, 2048)
                and tuple(weight.shape) == (1024, 2048)
                and x.dtype == weight.dtype == torch.bfloat16
                and x.is_contiguous() and weight.is_contiguous()
                and x.device.type == "npu" and weight.device == x.device):
            return smallm_r3.o_proj_row(x, weight)
        # GAP-WOSS-NONGE k2: ubfinish-r5 rowpass-SIMD bias finish (witnessed
        # bitwise vs the torch tail; graph 6.65us vs the 2-kernel chain
        # 10.30us on [2112,1280]). Opt-in; unregistered shapes fall through.
        if (
            os.environ.get("XINFER_UBFINISH_R5") == "1"
            and bias is not None
            and activation is None
            and residual is None
            and residual_gate is None
            and row_class is None
            and row_scale is None
            and bias.dtype == torch.bfloat16
            and bias.is_contiguous()
            and _gap_ubf_routes(m, n)
        ):
            from xinfer.ops.backends.ascend import ubfinish_r5 as _ub5

            with self._project(x, weight, shape) as output:
                out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
                _ub5.finish(out, output, bias, m, n)
                return out

        with self._project(x, weight, shape) as output:
            return _finish(output, bias, activation, residual, residual_gate, row_class, row_scale)

    def gemm_glu_epilogue(self, x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
        """Apply SiLU to the FP32 gate, multiply by the FP32 up projection and round once."""
        if modulation is not None or row_class is not None:
            raise ValueError("CANN GLU projection requires unmodulated input")
        shape = _geometry(x, weight)
        _, n2, _ = shape
        if n2 % 2:
            raise ValueError("CANN GLU projection requires [gate | up] weights with even row count")
        if bias is not None:
            _check_tensor(bias, "bias", x.device, (n2,), _FLOAT_DTYPES)
        n = n2 // 2
        # INT-WOSS G2112-r3 TORCH_A/GLU, default off.
        if ((G2112_FLAGS["GLU"] or G2112_FLAGS["TORCH_A"])
                and bias is not None and n2 == 6912 and shape[0] == 2112
                and bias.dtype == torch.bfloat16 and bias.is_contiguous()
                and x.dtype == weight.dtype == torch.bfloat16
                and x.is_contiguous() and weight.is_contiguous()
                and x.device.type == "npu" and weight.device == x.device):
            if G2112_FLAGS["GLU"]:
                return glu_route(x, weight, bias)
            return samu_route(torch.addmm(bias, x, weight.t()))
        if (bias is not None and n2 == 6912 and bias.dtype == torch.bfloat16
                and bias.is_contiguous() and callable(self._glu_fp32_deferred)):
            # glu-r1: the vision shape [2112, 1280] x [6912, 1280] rides the
            # gemm_glu_fp32_npu entry end to end: the same shared ACLNN Mm
            # projects into the entry's FP32 accumulator and the split_3456
            # finish folds the BF16 bias into its single staged pass. The old
            # chain spent a full [2112, 6912] FP32 read+write round on
            # `output += bias.float()` before the finish.
            out = x.new_empty((shape[0], n), dtype=torch.bfloat16)
            ws = self._glu_deferred_workspace(x, weight, shape)
            self._glu_fp32_deferred(out, ws, x, weight, bias, False, False)
            return out
        # r6a-m16epi blade A1 (XINFER_SMALLM_R3=gateup, default off): fused
        # action gate_up Cube+GLU single-shot kernel.
        if (smallm_r3 is not None and smallm_r3.enabled("gateup")
                and bias is None
                and x.ndim == 2 and weight.ndim == 2
                and tuple(x.shape) == (16, 1024)
                and tuple(weight.shape) == (4096, 1024)
                and x.dtype == weight.dtype == torch.bfloat16
                and x.is_contiguous() and weight.is_contiguous()
                and x.device.type == "npu" and weight.device == x.device):
            return smallm_r3.gateup(x, weight)
        if (
            finishbf16 is not None and finishbf16.enabled("gateup")
            and bias is None and n2 == 12288
            and x.ndim == 2 and weight.ndim == 2
            and x.dtype == weight.dtype == torch.bfloat16
            and x.device.type == "npu" and weight.device == x.device
            and x.is_contiguous() and weight.is_contiguous()
            and tuple(weight.shape) == (12288, 2048)
        ):
            # FINISH-BF16 gateup: one aclnnMatmul BF16 store feeds the
            # bf16-reading split_6144 finish; the borrowed FP32 accumulator
            # and its 2x-wide read disappear (GEMM-AUDIT R6 gateup row).
            return finishbf16.glu_split_bf16(x, weight)
        with self._project(x, weight, shape) as output:
            if bias is None and n2 == 4096 and callable(self._glu_split_fp32_finish):
                # r3: the action shape [M, 1024] x [4096, 1024] finishes through
                # the fused split_2048 kernel (one silu*up + BF16 round) instead
                # of the silu-slice + swish + mul + cast torch chain; enqueued
                # inside the projection lock like every other consumer of the
                # borrowed FP32 buffer. The dispatch is quick100-adjudicated.
                out = x.new_empty((shape[0], n), dtype=torch.bfloat16)
                self._glu_split_fp32_finish(out, output, 0, n)
                return out
            if (bias is not None and n2 == 6912 and bias.dtype == torch.bfloat16
                    and bias.is_contiguous() and callable(self._glu_split_fp32_finish)):
                # r4: the vision shape [rows, 1280] x [6848, 1280]. The BF16 bias
                # is pre-added into the borrowed FP32 accumulator -- elementwise
                # the torch chain's y = output + bias.float() -- and the fused
                # split_3456 finish (has_bias=0) does silu*up + one BF16 round in
                # a single pass over the accumulator.
                output += bias.float()
                out = x.new_empty((shape[0], n), dtype=torch.bfloat16)
                self._glu_split_fp32_finish(out, output, 0, n)
                return out
            if bias is None and n2 == 22016 and callable(self._glu_split_fp32_finish):
                # glu-r1: the prefill shape [768, 2048] x [22016, 2048]: one
                # fused split_11008 pass over the borrowed FP32 accumulator
                # replaces the three-op torch tail (silu + mul + cast).
                out = x.new_empty((shape[0], n), dtype=torch.bfloat16)
                self._glu_split_fp32_finish(out, output, 0, n)
                return out
            if (bias is None and n2 == 12288 and glu_epi is not None
                    and glu_epi.enabled("prefillglu")):
                # glu-epilogue K-B (XINFER_GLUEPI=prefillglu, default off):
                # the wallx2b prefill SwiGLU [M, 2048] x [12288, 2048] finishes
                # through the shipped AOT split_6144 kernel -- one staged
                # silu*up + BF16 round pass over the borrowed FP32 accumulator
                # instead of the torch tail (silu_slice + silu + mul + cast,
                # four aclnn launches per site).
                out = x.new_empty((shape[0], n), dtype=torch.bfloat16)
                glu_epi.split_6144_finish(out, output, shape[0])
                return out
            y = output if bias is None else output + bias.float()
            # SOUP-FORM glumulcast: keep aclnnSilu on the gate half (the bits
            # are the reference), fold the strided mul + BF16 round into one
            # pass over the borrowed accumulator. Parked: the kernel lost its
            # race with the aclnn chain at the call sites (45.9 vs 38.4 us per
            # site), so the family is out of the default arm; the hook stays
            # for the record and for the envelope audit.
            if bias is None:
                try:
                    from xinfer.ops.backends.ascend import soupform

                    if soupform.enabled("glumulcast"):
                        gate = F.silu(y[:, :n])
                        fused = soupform.mulcast(gate, y, n)
                        if fused is not None:
                            return fused
                except Exception:  # noqa: BLE001 - envelope miss falls back
                    pass
            return (F.silu(y[:, :n]) * y[:, n:]).to(torch.bfloat16)

    def _vision_rope_cube_call(self, x, weight, bias, cos, sin):
        """The r10 fused vision QKV Cube route: [2112, 1280] x [3840, 1280] NK, one launch.

        M=2112 pads into a [2176, 1280] activation (the 64 tail rows only compute
        into output rows the slice discards; valid rows are independent, the r8
        M768 o-projection argument), the kernel writes [2176, 3840] BF16 and the
        route returns the [:2112] view. Warm up once per stream before capture,
        exactly like the BNSD M768 route.
        """
        rows, columns, reduction = 2112, 3840, 1280
        if torch.npu.is_current_stream_capturing() and not self._vision_cube_warmed:
            raise RuntimeError("warm up the vision rope cube on this stream before graph capture")
        activation = x
        if x.shape[0] % 128:
            activation = x.new_empty((2176, reduction), dtype=torch.bfloat16)
            activation[:rows].copy_(x)
        out = activation.new_empty((2176, columns), dtype=torch.bfloat16)
        stream = torch.npu.current_stream(x.device).npu_stream
        self._vision_rope_cube(activation, weight, bias, cos, sin, out,
                               2176, columns, reduction, stream)
        self._vision_cube_warmed = True
        return out[:rows]

    def qkv_rope_epilogue(self, x, weight, bias, cos, sin, num_heads, head_dim):
        """Apply FP32 half-split RoPE to Q/K and return standard-order BF16 QKV."""
        shape = _geometry(x, weight)
        m, n, _ = shape
        if num_heads < 1 or head_dim < 1 or n != 3 * num_heads * head_dim:
            raise ValueError("CANN QKV RoPE weight must have 3 * num_heads * head_dim rows")
        if bias is None:
            raise ValueError("CANN QKV RoPE requires bias")
        _check_tensor(bias, "bias", x.device, (n,), _FLOAT_DTYPES)
        if cos is None or cos.ndim != 2 or not 0 < 2 * cos.shape[1] <= head_dim:
            raise ValueError("CANN QKV RoPE requires a nonempty rotary prefix fitting each head")
        half = cos.shape[1]
        _check_tensor(cos, "cos", x.device, (m, half), (torch.float32,))
        _check_tensor(sin, "sin", x.device, (m, half), (torch.float32,))
        if (m == 2112 and n == 3840 and num_heads == 16 and head_dim == 80
                and half == 40 and bias.dtype == torch.bfloat16
                and bias.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
                and x.is_contiguous() and weight.is_contiguous()
                and x.dtype == weight.dtype == torch.bfloat16 and x.device == weight.device
                and callable(self._vision_rope_cube)
                and os.environ.get("XINFER_VISION_ROPE_CUBE") == "1"):
            # r10: the fused Cube route. Numerics landed BITWISE vs the ACLNN Mm +
            # gemm_qkv_vision_rope_finish pair (probe-rope-r10: max_abs 0.0, mismatch
            # 0.0), but its first tiling (bm=128 x bn=160, 408 blocks, the SimtVF rope
            # on both AIVs) measured 176.15 us vs the pair 104.45 -- a 72% regression,
            # so the route stays opt-in (XINFER_VISION_ROPE_CUBE=1) until the gemm
            # tile lane re-tils it (bn=240 with an M-split epilogue keeps the +/-40
            # partner in-tile too); the kernel and the ctypes mount are in the lib.
            return self._vision_rope_cube_call(x, weight, bias, cos, sin)
        with self._project(x, weight, shape) as output:
            if (m == 2112 and n == 3840 and num_heads == 16 and head_dim == 80
                    and half == 40 and bias.dtype == torch.bfloat16
                    and bias.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
                    and callable(self._vision_rope_finish)):
                # r5: the fused vision finish (gemm_qkv_vision_rope_finish) --
                # one pass: bias add + standard-order rotate-half RoPE on the
                # Q/K heads + one BF16 round, oracle-anchored (probe-r5).
                out = x.new_empty((m, n), dtype=torch.bfloat16)
                self._vision_rope_finish(out, output, bias, cos, sin)
                return out
            y = (output + bias.float()).view(m, 3, num_heads, head_dim)
            c, s = cos[:, None, None, :], sin[:, None, None, :]
            qk = y[:, :2]
            a, b = qk[..., :half], qk[..., half:2 * half]
            rotated = torch.cat((a * c - b * s, b * c + a * s, qk[..., 2 * half:]), dim=-1)
            return torch.cat((rotated, y[:, 2:]), dim=1).reshape(m, n).to(torch.bfloat16)
