"""Internal Ascend action projections and vector epilogues for inference."""

from __future__ import annotations

import contextlib
import math
import os
from numbers import Integral, Real
from threading import Lock

import torch
import torch.nn.functional as F
from xinfer.ops.backends.ascend import _native
from xinfer.ops.backends.ascend.core import CORE
from xinfer.ops.backends.ascend._cann_action_scope import memo_action_metadata
from xinfer.ops.backends.ascend import smallm_r3


_QKV_2B_CACHE_FINISH = None


def _qkv_2b_cache_finish_pair():
    """(call, get_error) for the fused wallx2b QKV tail, or None (ctypes C-ABI)."""
    global _QKV_2B_CACHE_FINISH
    if _QKV_2B_CACHE_FINISH is None:
        import ctypes

        from xinfer.ops.backends.ascend import _native

        try:
            library = _native.core_library()
            call = getattr(library, "xinfer_aot_gemm_qkv_2b_norm_gate_cache_bnsdq_call")
            get_error = getattr(
                library, "xinfer_aot_gemm_qkv_2b_norm_gate_cache_bnsdq_get_last_error"
            )
        except (OSError, AttributeError):
            _QKV_2B_CACHE_FINISH = False
            return None
        call.restype = ctypes.c_int
        call.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
        get_error.restype = ctypes.c_char_p
        _QKV_2B_CACHE_FINISH = (call, get_error)
    return _QKV_2B_CACHE_FINISH or None


_FLOAT_DTYPES = (torch.bfloat16, torch.float32)
_INDEX_DTYPES = (torch.int32, torch.int64)
_FP32_GLU_FINISH_ENV = "XINFER_ASCEND_ACTION_FP32_GLU_FINISH"


def _check_tensor(tensor, name, device, shape, dtypes):
    if (
        tensor is None
        or tensor.device != device
        or tensor.shape != shape
        or tensor.dtype not in dtypes
    ):
        raise ValueError(
            f"{name} must have shape {shape}, dtype in {dtypes}, and device {device}"
        )


def _finish_fp32_glu(projected):
    from xinfer.ops.backends.ascend import _native

    output = torch.empty((projected.shape[0], 2560), dtype=torch.bfloat16, device=projected.device)
    CORE.gemm_epilogue_glu(output, projected, 0, 2560)
    return output


class CannAction:
    """Compose BF16-input CANN Matmul, FP32 norm/RoPE and BF16 epilogues.

    Pass ``CannProjection.accumulator`` or an equivalent context manager
    factory with signature ``(activation_mk, weight_nk)``. It yields the
    FP32 projection and must retain its storage through epilogue enqueue.
    Warm up each shape on the graph's stream and retain both providers while
    their graphs are live.

    ``bf16-round`` rounds the AdaRMS activation once. ``bf16-split2`` uses
    one Matmul over stacked high and residual BF16 components and adds the
    FP32 results. Both are explicit precision choices for FP32 activations.
    """

    def __init__(
        self,
        accumulator,
        *,
        precision,
        use_ada_rms=True,
        use_qk_rms=True,
        use_rope=True,
    ):
        if precision not in ("bf16-round", "bf16-split2"):
            raise ValueError("action precision must be bf16-round or bf16-split2")
        if not callable(accumulator):
            raise TypeError("accumulator must be a callable context manager factory")
        self._accumulator = accumulator
        self._precision = precision
        self._use_fp32_glu_finish = os.environ.get(_FP32_GLU_FINISH_ENV) == "1"
        self._use_ada_rms = use_ada_rms
        self._use_qk_rms = use_qk_rms
        self._rms_norm = self._rotary = None
        # walloss series bindings: the prefill fused finish ships on the FFI
        # surface; the m16 symbol is retired (getattr -> None degrades to the
        # NV-named action_qkv_gemm_norm_rope route below).
        self._qkv_bias_rope_m16 = getattr(CORE, "gemm_qkv_cache_bias_rope_m16", None)
        self._qkv_bias_rope_prefill = getattr(CORE, "gemm_qkv_cache_bias_rope_prefill", None)
        if use_ada_rms or use_qk_rms or use_rope:
            import torch_npu

            if use_ada_rms or use_qk_rms:
                self._rms_norm = torch_npu.npu_rms_norm
            if use_rope:
                self._rotary = torch_npu.npu_apply_rotary_pos_emb
        self._rms_weights = {}
        # 2b-m16: the wallx2b action passes persistent per-layer Q/K norm
        # weight tensors; ``1 + weight.float()`` was recomputed as two kernels
        # on every call (probe: the torch tail is ~52 us of the 57.9 us op).
        # The memoized factors hold the same FP32 values, so every consumer
        # stays bitwise with the recomputed chain.
        self._one_plus_cache = {}
        self._qkv_2b_cache_warmed = False
        self._lock = Lock()

    def _one_plus(self, weight):
        if weight is None:
            return None
        key = (weight.data_ptr(), tuple(weight.shape), weight.dtype)
        entry = self._one_plus_cache.get(key)
        if entry is None:
            factor = 1 + weight.float()
            # Keep the source alive so its storage identity cannot be reused.
            self._one_plus_cache[key] = (weight, factor)
            return factor
        return entry[1]

    def _rms(self, x, eps, use_cann):
        if not use_cann:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        key = (x.device, torch.npu.current_stream(x.device).npu_stream, x.shape[-1])
        with self._lock:
            if key not in self._rms_weights:
                if torch.npu.is_current_stream_capturing():
                    raise RuntimeError(
                        "warm up CANN action RMSNorm on this stream and width before graph capture"
                    )
                self._rms_weights[key] = x.new_ones(x.shape[-1])
            gamma = self._rms_weights[key]
        normed, _ = self._rms_norm(x.contiguous(), gamma, epsilon=eps)
        _check_tensor(normed, "RMSNorm result", x.device, x.shape, (torch.float32,))
        return normed

    def _modulation_rows(self, x, modulation, row_class, eps, *, gate=False):
        if (
            x.ndim != 2
            or x.device.type != "npu"
            or x.dtype != torch.bfloat16
            or not all(x.shape)
        ):
            raise ValueError("action activation must be nonempty BF16 [M,K] on an NPU")
        if not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite host number")
        m, k = x.shape
        if modulation is None or modulation.ndim != 2 or not modulation.shape[0]:
            raise ValueError(
                "action modulation must be a nonempty two-dimensional table"
            )
        columns = modulation.shape[1]
        if (gate and columns != 3 * k) or columns < 2 * k:
            raise ValueError(
                "action modulation needs K scale, K shift and, for a gated residual, K gate columns"
            )
        _check_tensor(
            modulation, "modulation", x.device, modulation.shape, (torch.bfloat16,)
        )
        _check_tensor(row_class, "row_class", x.device, (m,), _INDEX_DTYPES)
        return modulation.index_select(0, row_class)

    def _adarms(self, x, rows, eps):
        k = x.shape[1]
        rows_fp32 = rows.float()
        scale, shift = rows_fp32[:, :k], rows_fp32[:, k : 2 * k]
        return self._rms(x.float(), eps, self._use_ada_rms) * (1 + scale) + shift

    def gated_input_residual(self, x, modulation, row_class, eps):
        """Return BF16 ``x + BF16(BF16(AdaRMS(x)) * gate)``."""
        rows = self._modulation_rows(x, modulation, row_class, eps, gate=True)
        normed = self._adarms(x, rows, eps).to(torch.bfloat16)
        return x + normed * rows[:, 2 * x.shape[1] :]

    def _matmul_input(self, x, weight, modulation, row_class, eps):
        rows = self._modulation_rows(x, modulation, row_class, eps)
        _check_tensor(
            weight, "weight", x.device, (weight.shape[0], x.shape[1]), (torch.bfloat16,)
        )
        normalized = self._adarms(x, rows, eps)
        high = normalized.to(torch.bfloat16)
        if self._precision == "bf16-round":
            return high
        low = (normalized - high.float()).to(torch.bfloat16)
        return torch.cat((high, low), dim=0)

    def _finish_accumulation(self, projected, activation, m, n):
        _check_tensor(
            projected,
            "accumulator result",
            activation.device,
            (activation.shape[0], n),
            (torch.float32,),
        )
        if self._precision == "bf16-round":
            return projected
        high, low = projected.split(m, dim=0)
        return high + low

    def gemm_glu_epilogue(
        self, x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6
    ):
        """Project modulated rows, apply FP32 bias and SiLU-GLU, and return BF16."""
        if weight.ndim != 2 or not weight.shape[0] or weight.shape[0] % 2:
            raise ValueError(
                "GLU weight must pack equally sized gate and up projections"
            )
        n = weight.shape[0]
        if bias is not None:
            _check_tensor(bias, "bias", x.device, (n,), _FLOAT_DTYPES)
        activation = self._matmul_input(x, weight, modulation, row_class, eps)
        with self._accumulator(activation, weight) as projected:
            y = self._finish_accumulation(projected, activation, x.shape[0], n)
            if (self._use_fp32_glu_finish and self._precision == "bf16-round" and bias is None
                    and x.shape in ((68, 1280), (76, 1280)) and weight.shape == (5120, 1280)
                    and modulation.shape == (3, 3840) and row_class.dtype == torch.int32 and eps == 1e-6
                    and y.is_contiguous() and y.data_ptr() % 32 == 0):
                return _finish_fp32_glu(y)
            if bias is not None:
                y = y + bias.float()
            gate, up = y.chunk(2, dim=1)
            return (F.silu(gate) * up).to(torch.bfloat16)

    def _norm_qk(self, qk, q_weight, k_weight, heads, kv_heads, eps):
        if self._use_qk_rms and q_weight is not None and k_weight is not None:
            qk = self._rms(qk, eps, True)
            q, k = qk.split((heads, kv_heads), dim=1)
            return q * self._one_plus(q_weight), k * self._one_plus(k_weight)

        def apply(tensor, weight):
            if weight is None:
                return tensor
            return self._rms(tensor, eps, self._use_qk_rms) * self._one_plus(weight)

        q, k = qk.split((heads, kv_heads), dim=1)
        return apply(q, q_weight), apply(k, k_weight)

    def _rope(self, q, k, cos, sin):
        half = cos.shape[-1]
        width = 2 * half
        if self._rotary is not None:
            # The vendor op can mutate its operands. These are private copies,
            # including when the entire head rotates and Q/K alias the Matmul.
            q_prefix = q[..., :width].clone(memory_format=torch.contiguous_format)
            k_prefix = k[..., :width].clone(memory_format=torch.contiguous_format)

            def expand_tables():
                return (
                    torch.cat((cos, cos), dim=-1).unsqueeze(1),
                    torch.cat((sin, sin), dim=-1).unsqueeze(1),
                )

            cos_full, sin_full = memo_action_metadata(
                ("rope_tables", id(self)), (cos, sin), expand_tables
            )
            q_rot, k_rot = self._rotary(
                q_prefix, k_prefix, cos_full, sin_full, layout="TND", rotary_mode="half"
            )
            _check_tensor(
                q_rot, "Q RoPE result", q.device, q_prefix.shape, (torch.float32,)
            )
            _check_tensor(
                k_rot, "K RoPE result", k.device, k_prefix.shape, (torch.float32,)
            )
            if width == q.shape[-1]:
                return q_rot, k_rot
            return torch.cat((q_rot, q[..., width:]), dim=-1), torch.cat(
                (k_rot, k[..., width:]), dim=-1
            )

        c, s = cos[:, None, :], sin[:, None, :]

        def apply(tensor):
            a, b = tensor[..., :half], tensor[..., half:width]
            return torch.cat(
                (a * c - b * s, b * c + a * s, tensor[..., width:]), dim=-1
            )

        return apply(q), apply(k)

    def _check_qkv(
        self,
        x,
        weight,
        q_weight,
        k_weight,
        cos,
        sin,
        k_cache,
        v_cache,
        start,
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
        *,
        allow_gate,
    ):
        heads, kv_heads, d = num_heads, num_kv_heads, head_dim
        if not all(
            isinstance(dim, Integral) and dim > 0 for dim in (heads, kv_heads, d)
        ):
            raise ValueError(
                "Q/KV head counts and head dimension must be positive integers"
            )
        channels = (heads + 2 * kv_heads) * d
        if (
            weight.ndim != 2
            or weight.shape[0] < channels
            or (not allow_gate and weight.shape[0] != channels)
        ):
            raise ValueError(
                "weight must pack [Q | K | V] with extra rows only for the gate"
            )
        n = weight.shape[0]
        if (
            x.ndim != 2
            or x.device.type != "npu"
            or x.dtype != torch.bfloat16
            or not all(x.shape)
        ):
            raise ValueError("QKV activation must be nonempty BF16 [M,K] on an NPU")
        m = x.shape[0]
        _check_tensor(weight, "weight", x.device, (n, x.shape[1]), (torch.bfloat16,))
        if cos.ndim != 2 or not 0 < 2 * cos.shape[1] <= d:
            raise ValueError(
                "compact rotary table width must be positive and fit in a head"
            )
        if self._rotary is not None and 2 * cos.shape[1] > 1024:
            raise ValueError("CANN rotary prefix must fit within 1024 channels")
        _check_tensor(cos, "cos", x.device, (m, cos.shape[1]), (torch.float32,))
        _check_tensor(sin, "sin", x.device, cos.shape, (torch.float32,))
        for name, norm in (("q_weight", q_weight), ("k_weight", k_weight)):
            if norm is not None:
                _check_tensor(norm, name, x.device, (d,), _FLOAT_DTYPES)
        if (q_weight is not None or k_weight is not None) and (
            not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0
        ):
            raise ValueError("Q/K RMSNorm eps must be a positive finite host number")
        if (
            k_cache.ndim != 3
            or not isinstance(start, Integral)
            or not 0 <= start <= k_cache.shape[1] - m
        ):
            raise ValueError("cache append interval is out of bounds")
        cache_shape = (kv_heads, k_cache.shape[1], d)
        _check_tensor(k_cache, "k_cache", x.device, cache_shape, (torch.bfloat16,))
        _check_tensor(v_cache, "v_cache", x.device, cache_shape, (torch.bfloat16,))
        return m, n

    def _write_qkv(
        self,
        y,
        q_weight,
        k_weight,
        cos,
        sin,
        k_cache,
        v_cache,
        start,
        heads,
        kv_heads,
        d,
        eps,
    ):
        m, n = y.shape
        qd, kvd = heads * d, kv_heads * d
        gate_width = n - qd - 2 * kvd
        qk, v, gate = y.split((qd + kvd, kvd, gate_width), dim=1)
        qk = qk.view(m, heads + kv_heads, d)
        q, k = self._norm_qk(qk, q_weight, k_weight, heads, kv_heads, eps)
        q, k = self._rope(q, k, cos, sin)
        # 2b-m16: one casting copy per cache slice. ``copy_`` converts FP32 to
        # BF16 with the same RNE as the previous ``.to`` round-trip, so the
        # written cache bytes are unchanged while the separate cast kernels
        # disappear (two fewer launches per call).
        k_cache[:, start : start + m].copy_(k.transpose(0, 1))
        v_cache[:, start : start + m].copy_(v.view(m, kv_heads, d).transpose(0, 1))
        # 2b-m16: assemble the packed [Q | gate] return in place with two
        # casting slice copies instead of two standalone casts plus a concat.
        q_out = torch.empty((m, qd + gate_width), dtype=torch.bfloat16, device=y.device)
        q_out[:, :qd].copy_(q.reshape(m, qd))
        if gate_width:
            q_out[:, qd:].copy_(gate)
        return q_out

    def adarms_qkv_kvcache(
        self,
        x,
        weight,
        modulation,
        row_class,
        q_weight,
        k_weight,
        cos,
        sin,
        k_cache,
        v_cache,
        start,
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
        bias=None,
    ):
        """Project AdaRMS rows, append BF16 K/V, and return packed BF16 Q/gate.

        ``bias`` (BF16 or FP32 ``[N]``) is added to the FP32 projection before the q/k
        norms and the rotation, matching ``qkv_rope_kvcache``.
        """
        m, n = self._check_qkv(
            x,
            weight,
            q_weight,
            k_weight,
            cos,
            sin,
            k_cache,
            v_cache,
            start,
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            allow_gate=True,
        )
        if bias is not None:
            _check_tensor(bias, "bias", x.device, (n,), _FLOAT_DTYPES)

        # Wall-OSS action fast path: one fused per-row AdaRMS kernel, the CANN
        # accumulator and the fused M16 bias+RoPE+append finish; the input norm,
        # the QKV projection, the bias, the rotation and the K/V append are one
        # operator call.
        if (
            callable(self._qkv_bias_rope_m16)
            and (m, n, x.shape[1], num_heads, num_kv_heads, head_dim)
            == (16, 2560, 1024, 16, 2, 128)
            and q_weight is None
            and k_weight is None
            and bias is not None
            and bias.dtype == torch.bfloat16
            and cos.shape == (16, 64)
            and modulation.dtype == torch.bfloat16
            and modulation.ndim == 2
            and modulation.shape[1] == 3 * x.shape[1]
            and modulation.is_contiguous()
            and x.is_contiguous()
            and row_class is not None
            and row_class.dtype == torch.int32
            and row_class.numel() == 16
            and row_class.is_contiguous()
            and all(
                tensor.is_contiguous()
                for tensor in (bias, cos, sin, k_cache, v_cache)
            )
        ):
            from xinfer.ops.backends.ascend.adarmsnorm import Adarmsnorm

            activation, _ = Adarmsnorm()(
                x, modulation, float(eps), True, row_class=row_class
            )
            with self._accumulator(activation, weight) as projected:
                _check_tensor(
                    projected,
                    "accumulator result",
                    activation.device,
                    (activation.shape[0], n),
                    (torch.float32,),
                )
                # r3: BNSD [H, M, D] -- the kernel writes the head-major layout
                # the ACLNN FIA query consumes, so the attention transpose is a view.
                query = x.new_empty((num_heads, 16, head_dim))
                self._qkv_bias_rope_m16(
                    query, k_cache, v_cache, projected, bias, cos, sin, int(start)
                )
                return query

        activation = self._matmul_input(x, weight, modulation, row_class, eps)
        with self._accumulator(activation, weight) as projected:
            y = self._finish_accumulation(projected, activation, m, n)
            if bias is not None:
                y = y + bias.float()
            return self._write_qkv(
                y,
                q_weight,
                k_weight,
                cos,
                sin,
                k_cache,
                v_cache,
                start,
                num_heads,
                num_kv_heads,
                head_dim,
                eps,
            )

    def qkv_norm_rope_kvcache(
        self,
        x,
        weight,
        q_weight,
        k_weight,
        cos,
        sin,
        k_cache,
        v_cache,
        start,
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
    ):
        """Project BF16 rows into FP32, normalize and rotate Q/K, and append BF16 K/V."""
        m, n = self._check_qkv(
            x,
            weight,
            q_weight,
            k_weight,
            cos,
            sin,
            k_cache,
            v_cache,
            start,
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            allow_gate=True,
        )
        # smallm-r3 qkv blade (XINFER_SMALLM_R3=qkv, default off): the m16
        # action QKV projection rides the bitwise Cube kernel; the route is
        # lock-free with no shared buffers (the WALLOSS-QKV capacity win).
        if (smallm_r3.enabled("qkv")
                and (m, n, x.shape[1]) == (16, 2560, 1024)
                and x.dtype == weight.dtype == torch.bfloat16
                and x.is_contiguous() and weight.is_contiguous()
                and x.device.type == "npu"):
            projected_ctx = contextlib.nullcontext(smallm_r3.qkv_fp32(x, weight))
        else:
            projected_ctx = self._accumulator(x, weight)
        with projected_ctx as y:
            _check_tensor(y, "accumulator result", x.device, (m, n), (torch.float32,))
            # 2b-qkv / R6B-BRIDGE B1 (env XINFER_R6B_BRIDGE=1, the v22 20.62
            # stack's run flag): the wallx2b action QKV tail fused into one
            # AOT kernel -- per-head RMS norm on the raw FP32 accumulator,
            # the (1 + w) factors, the 64-dim partial rotary, the head-major
            # K/V append, the Q half stored head-major (BNSD-contiguous) and
            # the gate in its own dense buffer, both by the same job plan and
            # FP32 arithmetic as the released packed kernel, only the store
            # indices move. Returns (q_bnsd [1, H, M, D], gate [1, M, H*D]) --
            # the head-major query is what the ACLNN FIA dense descriptor
            # reads, so the action attention takes the FIA-FUSE pair without
            # the token-major materialization. K/V caches and the gate half
            # are bitwise with the chain; Q/K values carry an ulp-level rstd
            # residue (CANN npu_rms_norm approximate rsqrt, ~7e-7 rel; probe:
            # 1 BF16 flip in 65536 on a synthetic batch). Envelope misses (or
            # the flag off) keep the released _write_qkv chain below.
            finish = _qkv_2b_cache_finish_pair()
            if (
                finish is not None
                and os.environ.get("XINFER_R6B_BRIDGE", "0") == "1"
                and (m, n, x.shape[1], num_heads, num_kv_heads, head_dim)
                == (16, 5120, 1024, 8, 2, 256)
                and float(eps) == 1e-6
                and q_weight is not None
                and k_weight is not None
                and tuple(q_weight.shape) == (head_dim,)
                and tuple(k_weight.shape) == (head_dim,)
                and cos.shape == (m, 32)
                and sin.shape == (m, 32)
                and v_cache.shape == k_cache.shape
                and k_cache.shape[0] == num_kv_heads
                and k_cache.shape[2] == head_dim
                and all(
                    tensor.is_contiguous()
                    for tensor in (
                        x, weight, q_weight, k_weight, cos, sin, k_cache, v_cache,
                    )
                )
            ):
                import ctypes

                if (
                    torch.npu.is_current_stream_capturing()
                    and not self._qkv_2b_cache_warmed
                ):
                    raise RuntimeError(
                        "warm up the fused 2B QKV tail on this stream before graph capture"
                    )
                q_bnsd = x.new_empty(
                    (1, num_heads, m, head_dim), dtype=torch.bfloat16
                )
                gate = x.new_empty(
                    (1, m, num_heads * head_dim), dtype=torch.bfloat16
                )
                q_factor = self._one_plus(q_weight)
                k_factor = self._one_plus(k_weight)
                # the single library's TVM-FFI face serves first (the unified launch
                # boundary); the ctypes launcher of the family library is the fallback
                # below
                if _native.launch_kernel(
                    "gemm_qkv_2b_norm_gate_cache_bnsdq",
                    (q_bnsd, gate, k_cache, v_cache, y, q_factor, k_factor, cos, sin),
                    (m, k_cache.shape[1], int(start)),
                ):
                    self._qkv_2b_cache_warmed = True
                    return q_bnsd, gate
                call, get_error = finish
                stream = ctypes.c_void_p(
                    torch.npu.current_stream(x.device).npu_stream
                )
                rc = call(
                    ctypes.c_void_p(q_bnsd.data_ptr()),
                    ctypes.c_void_p(gate.data_ptr()),
                    ctypes.c_void_p(k_cache.data_ptr()),
                    ctypes.c_void_p(v_cache.data_ptr()),
                    ctypes.c_void_p(y.data_ptr()),
                    ctypes.c_void_p(q_factor.data_ptr()),
                    ctypes.c_void_p(k_factor.data_ptr()),
                    ctypes.c_void_p(cos.data_ptr()),
                    ctypes.c_void_p(sin.data_ptr()),
                    m,
                    k_cache.shape[1],
                    int(start),
                    stream,
                )
                self._qkv_2b_cache_warmed = True
                if rc != 0:
                    message = get_error()
                    text = (
                        message.decode("utf-8", "replace") if message else "unknown failure"
                    )
                    raise RuntimeError(f"gemm_qkv_2b_norm_gate_cache_bnsdq: {text}")
                return q_bnsd, gate
            return self._write_qkv(
                y,
                q_weight,
                k_weight,
                cos,
                sin,
                k_cache,
                v_cache,
                start,
                num_heads,
                num_kv_heads,
                head_dim,
                eps,
            )

    def qkv_rope_kvcache(
        self,
        x,
        weight,
        bias,
        cos,
        sin,
        k_cache,
        v_cache,
        start,
        num_heads,
        num_kv_heads,
        head_dim,
    ):
        """Project BF16 rows, apply FP32 bias and RoPE, and append BF16 K/V."""
        m, n = self._check_qkv(
            x,
            weight,
            None,
            None,
            cos,
            sin,
            k_cache,
            v_cache,
            start,
            num_heads,
            num_kv_heads,
            head_dim,
            0.0,
            allow_gate=False,
        )
        if bias is not None:
            _check_tensor(bias, "bias", x.device, (n,), _FLOAT_DTYPES)
        with self._accumulator(x, weight) as projected:
            _check_tensor(
                projected, "accumulator result", x.device, (m, n), (torch.float32,)
            )
            if (
                callable(self._qkv_bias_rope_prefill)
                and n == 2560 and x.shape[1] in (1024, 2048)
                and num_heads == 16 and num_kv_heads == 2 and head_dim == 128
                and bias is not None and bias.dtype == torch.bfloat16 and cos.shape == (m, 64)
                and m % 4 == 0 and m > 16
                and all(t.is_contiguous() for t in (projected, bias, cos, sin, k_cache, v_cache))
            ):
                # r6: the prefill fused finish (gemm_qkv_cache_bias_rope_prefill)
                # -- one ub-staged pass: bias + RoPE on the 18 Q/K heads, the
                # head-major K/V append and the packed Q write (the m16
                # kernel's structure at multi-row tiles; probe-r6). r7: Q
                # lands BNSD [H, M, D] -- the head-major layout the prefill
                # causal attention consumes, so the per-call transpose copy
                # disappears (the m16 kernel's r3 pattern).
                # qkv-batch r1: the K gate widens to the action family
                # (K=1024), so the B2/B4 batch calls (M=32/64, cache rows
                # [start, start + M)) ride the same fused finish instead of
                # the torch fallback chain (37.7/41.9 -> 10.1/11.1 us,
                # bench_qkv_batch_baseline). M=16 still rides the m16
                # epilogue below (7.8 vs 9.5 us).
                query = x.new_empty((num_heads, m, head_dim))
                self._qkv_bias_rope_prefill(query, k_cache, v_cache, projected, bias, cos, sin, int(start))
                return query
            if (
                (m, n, x.shape[1], num_heads, num_kv_heads, head_dim, start)
                == (16, 2560, 1024, 16, 2, 128, 768)
                and bias is not None and bias.dtype == torch.bfloat16 and cos.shape == (16, 64)
                and all(t.is_contiguous() for t in (projected, bias, cos, sin, k_cache, v_cache))
            ):
                # The fused M16 epilogue rides the NV-named entry's cache-append
                # form: input=FP32 accumulator, q/k_weight=K/V caches,
                # rotary=start row, epsilon=0 (weight is not re-read).
                query = x.new_empty((16, 2048))
                CORE.action_qkv_gemm_norm_rope(
                    query, projected, weight, bias, k_cache, v_cache, cos, sin,
                    int(start), 0.0,
                )
                return query
            if (
                callable(self._qkv_bias_rope_m16)
                and (m, n, x.shape[1], num_heads, num_kv_heads, head_dim)
                == (16, 2560, 1024, 16, 2, 128)
                and bias is not None and bias.dtype == torch.bfloat16 and cos.shape == (16, 64)
                and all(t.is_contiguous() for t in (projected, bias, cos, sin, k_cache, v_cache))
            ):
                # r3: BNSD [H, M, D] -- the kernel writes the head-major layout
                # the ACLNN FIA query consumes, so the attention transpose is a view.
                query = x.new_empty((num_heads, m, head_dim))
                self._qkv_bias_rope_m16(query, k_cache, v_cache, projected, bias, cos, sin, int(start))
                return query
            y = projected if bias is None else projected + bias.float()
            return self._write_qkv(
                y,
                None,
                None,
                cos,
                sin,
                k_cache,
                v_cache,
                start,
                num_heads,
                num_kv_heads,
                head_dim,
                0.0,
            )
