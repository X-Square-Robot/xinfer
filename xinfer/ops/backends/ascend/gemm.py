"""Fused Ascend projections over the in-tree glue package.

Explicit glue exports with the OpSpec names take precedence. Otherwise CANN bridges handle supported
projection and AdaRMS calls in eager execution and graph capture. Builds
without these bridges use the references. Other layouts/options also run the reference, with support cached
per call layout; legacy eager-only raw entries are not registered.
Stream-K reductions may differ at rounding level across runs.

``gemm_add_rms_norm`` and ``mlp_add_rms_norm`` call this backend's ``gemm_epilogue``, ``gemm_glu_epilogue`` and
``add_rms_norm`` in turn (the default selections of those operators), so their results are those calls'.
Contracts against the torch references: the delivered kernels bitwise on their audited
served shapes (the production gates), the generic aclnn routes by their 12/12 bitwise witness.
"""

import os

import torch

from xinfer.ops.backends.ascend import _native
from xinfer.ops.backends.ascend._support import PLATFORMS, PRIORITY, VERSION, bridge, canonical, partial, xc
from xinfer.ops.backends.ascend.norm import add_rms_norm, norm_library
from xinfer.ops.reference import gemm as reference
from xinfer.ops.registry import register


def _projection():
    module = bridge("cann_projection")
    factory = getattr(module, "CannProjection", None)
    try:
        return factory() if callable(factory) else None
    except getattr(xc, "MissingBackendError", ()):
        return None  # the constructor binds exports missing from builds without the CANN bridge


def _action(projection):
    if not callable(getattr(projection, "accumulator", None)):
        return None
    factory = getattr(bridge("cann_action"), "CannAction", None)
    if not callable(factory):
        return None
    # The action-precision run knob (the retired backend flag, renamed with
    # the XINFER_ prefix like its siblings): bf16-round is the single-RNE form
    # the 20.62 stack ran; bf16-split2 (the default) splits the normalized
    # activation into high/low BF16 halves for a compensated accumulator.
    precision = os.environ.get("XINFER_ASCEND_ACTION_PRECISION", "bf16-split2")
    return factory(projection.accumulator, precision=precision)


canonical_gemm = canonical("gemm_epilogue", reference.gemm_epilogue)
canonical_glu = canonical("gemm_glu_epilogue", reference.gemm_glu_epilogue)
canonical_qkv = canonical("qkv_rope_epilogue", reference.qkv_rope_epilogue)
canonical_cache = canonical("qkv_rope_kvcache", reference.qkv_rope_kvcache)
cann_projection = (_projection() if any(function is None for function in
                                       (canonical_gemm, canonical_glu, canonical_qkv, canonical_cache)) else None)
cann_action = _action(cann_projection) if canonical_glu is None or canonical_cache is None else None
native_gemm = canonical_gemm or getattr(cann_projection, "gemm_epilogue", None)
# The bridge's projection reads a BNSD ``[H, M, D]`` attention output with the head transpose folded into its
# Cube A-load (and pads any narrower flow width into its M768 kernel); the canonical entry takes 2-D rows only.
bnsd_gemm = getattr(cann_projection, "gemm_epilogue", None)
cann_glu = getattr(cann_projection, "gemm_glu_epilogue", None)
native_qkv = canonical_qkv or getattr(cann_projection, "qkv_rope_epilogue", None)
native_cache = canonical_cache or getattr(cann_action, "qkv_rope_kvcache", None)
# The wallx2b action QKV takes the CANN action bridge first as well: its 2b-qkv tail-fused kernel (one
# AOT launch for the per-head RMS norm, the (1 + w) factors, the 64-dim partial rotary, the head-major
# K/V append and the packed [Q | gate] return) and its memoized torch chain for other shapes are the
# lineage's fast paths - the raw [N, K] weight the unified projection entry would take runs its generic
# route there. The unified entry stays the fallback for builds without the bridge.
native_2b_cache = getattr(cann_action, "qkv_norm_rope_kvcache", None)


@register("gemm_epilogue", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if callable(native_gemm) else None,
          version=VERSION)
@partial(reference.gemm_epilogue)
def gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, residual_gate=None,
                  row_class=None, row_scale=None):
    if x.ndim == 3:
        # A head-major attention output feeds the projection directly (the lineage r4 action / r8 prefill
        # form). The bridge folds the head transpose into its Cube A-load and pads any narrower flow width
        # into its M768 kernel; a layout its BNSD routes do not take falls through to the packed view every
        # 2-D caller (and the reference below) reads, so other models keep their 2-D paths unchanged.
        if callable(bnsd_gemm):
            try:
                return bnsd_gemm(x, weight, bias, activation=activation, residual=residual,
                                 residual_gate=residual_gate, row_class=row_class, row_scale=row_scale)
            except ValueError:
                pass
        x = x.transpose(0, 1).reshape(x.shape[1], -1)
    return native_gemm(x, weight, bias, activation=activation, residual=residual, residual_gate=residual_gate,
                       row_class=row_class, row_scale=row_scale)


def _glu_supported(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
    return canonical_glu is not None or modulation is None or callable(getattr(cann_action, "gemm_glu_epilogue", None))


@register("gemm_glu_epilogue", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if canonical_glu is not None or callable(cann_glu) else None,
          version=VERSION)
@partial(reference.gemm_glu_epilogue, check=_glu_supported)
def gemm_glu_epilogue(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
    if canonical_glu is not None:
        return canonical_glu(x, weight, bias=bias, modulation=modulation, row_class=row_class, eps=eps)
    if modulation is None:
        return cann_glu(x, weight, bias=bias, row_class=row_class, eps=eps)
    return cann_action.gemm_glu_epilogue(x, weight, bias=bias, modulation=modulation,
                                        row_class=row_class, eps=eps)


@register("gemm_add_rms_norm", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if callable(native_gemm) else None, version=VERSION)
def gemm_add_rms_norm(x, projection, residual, weight, eps, bias=None):
    if cann_projection is not None:
        projection = cann_projection._stable_contiguous(projection)
    normalize = add_rms_norm if norm_library is not None else reference.add_rms_norm
    return normalize(gemm_epilogue(x, projection, bias).view(residual.shape), residual, weight, eps)


@register("mlp_add_rms_norm", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if (callable(native_gemm) and norm_library is not None
                          and (canonical_glu is not None or callable(cann_glu))) else None, version=VERSION)
def mlp_add_rms_norm(x, gate_up, down, residual, weight, eps):
    return gemm_add_rms_norm(gemm_glu_epilogue(x, gate_up), down, residual, weight, eps)


@register("qkv_rope_epilogue", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if callable(native_qkv) else None,
          version=VERSION)
@partial(reference.qkv_rope_epilogue)
def qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim, *, qk_order="standard"):
    return native_qkv(x, weight, bias, cos, sin, num_heads, head_dim)  # the standard order serves "any" too


def _qkv_rope_split_fallback(x, weight, bias, cos, sin, num_heads, head_dim, *, qk_order="standard"):
    out = qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim, qk_order=qk_order)
    from xinfer.ops.backends.ascend.copykill_r2 import enabled

    if enabled("k3"):
        return out.view(x.shape[0], 3, num_heads, head_dim).permute(1, 0, 2, 3).contiguous().unbind(0)
    return out.view(x.shape[0], 3, num_heads, head_dim).unbind(1)


@register("qkv_rope_split", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if callable(native_qkv) else None, version=VERSION)
@partial(_qkv_rope_split_fallback)
def qkv_rope_split(x, weight, bias, cos, sin, num_heads, head_dim, *, qk_order="standard"):
    from xinfer.ops.backends.ascend import _projection_layout

    split = _projection_layout.qkv_rope_split(x, weight, bias, cos, sin, num_heads, head_dim, cann_projection)
    if split is not None:
        return split
    return _qkv_rope_split_fallback(x, weight, bias, cos, sin, num_heads, head_dim, qk_order=qk_order)


def _gemm_split_fallback(x, weight, split_size, *, row_scale=None):
    out = gemm_epilogue(x, weight, row_scale=row_scale)
    return out[:, :split_size], out[:, split_size:]


@register("gemm_split", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if callable(native_gemm) else None, version=VERSION)
@partial(_gemm_split_fallback)
def gemm_split(x, weight, split_size, *, row_scale=None):
    from xinfer.ops.backends.ascend import _projection_layout

    split = _projection_layout.gemm_split(x, weight, split_size, row_scale, cann_projection)
    if split is not None:
        return split
    return _gemm_split_fallback(x, weight, split_size, row_scale=row_scale)


_fused_projection = getattr(xc, "qkv_proj_rope", None) if xc is not None else None
_qk_norm = getattr(xc, "QkNorm", None) if xc is not None else None


# ``qkv_rope_kvcache`` takes the CANN action bridge first: its fused prefill finish (one staged pass for bias,
# RoPE, the head-major K/V append and the Q write) and the M16 action epilogue are the wall-oss fast kernels,
# and its Q returns in whichever layout the kernel writes — BNSD ``[H, M, D]`` for the fused routes, packed
# ``[M, H·D]`` for compositions. The M16 route that rides the merged ``action_qkv_gemm_norm_rope`` entry
# allocates a 2-D ``[16, 2048]`` buffer and writes it head-major (``H == M == 16`` makes the shape alone
# ambiguous), so the form is disambiguated once per process against the reference composition and normalized
# to the unambiguous 3-D BNSD view. The unified projection entry stays the fallback for builds and shapes
# without the bridge (it serves the same call bit-exactly through its generic route, as the
# reference-composed quick100 gate validated end to end).
_m16_q_is_bnsd: bool | None = None


def _normalize_m16_q(result, args):
    """The M16 Q in its unambiguous form: 3-D BNSD ``[H, M, D]``, or the packed ``[M, H·D]`` rows as returned."""
    global _m16_q_is_bnsd
    x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads, head_dim = args
    if result.ndim != 2 or result.shape != (num_heads, num_heads * head_dim) or bias is None:
        return result  # only the square M == H form is ambiguous; 3-D returns are already BNSD
    if _m16_q_is_bnsd is None:
        rows = result.shape[0]
        # A synthetic probe (the live inputs could be degenerate enough to be permutation-invariant): the
        # same M/N/K/cache layout the call came with, random values, its own cache clones.
        generator = torch.Generator(device="cpu").manual_seed(20261113)
        probe_x = torch.randn(rows, x.shape[1], generator=generator).to(x.dtype).to(x.device) * 0.5
        probe_w = torch.randn(weight.shape, generator=generator).to(weight.dtype).to(weight.device) * 0.02
        probe_bias = torch.randn(weight.shape[0], generator=generator).to(bias.dtype).to(bias.device) * 0.02
        angles = torch.randn(rows, cos.shape[-1], generator=generator).to(cos.device) * 3
        probe_cos, probe_sin = angles.cos().to(cos.dtype), angles.sin().to(sin.dtype)
        probe_k, probe_v = k_cache.clone(), v_cache.clone()
        native = native_cache(probe_x, probe_w, probe_bias, probe_cos, probe_sin, probe_k, probe_v, start,
                              num_heads, num_kv_heads, head_dim)
        probe_k2, probe_v2 = k_cache.clone(), v_cache.clone()
        packed = reference.qkv_rope_kvcache(probe_x, probe_w, probe_bias, probe_cos, probe_sin, probe_k2, probe_v2,
                                            start, num_heads, num_kv_heads, head_dim)
        bnsd_read = native.view(num_heads, rows, head_dim).transpose(0, 1).reshape(rows, -1)
        _m16_q_is_bnsd = bool((bnsd_read.float() - packed.float()).abs().max().item()
                              <= (native.reshape(rows, -1).float() - packed.float()).abs().max().item())
    if _m16_q_is_bnsd:
        return result.view(num_heads, result.shape[0], head_dim)
    return result


@register("qkv_rope_kvcache", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if callable(_fused_projection) or callable(native_cache) else None,
          version=VERSION, deterministic=False)
@partial(reference.qkv_rope_kvcache)
def qkv_rope_kvcache(x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads, head_dim):
    if callable(native_cache):
        result = native_cache(x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads, head_dim)
        return _normalize_m16_q(result, (x, weight, bias, cos, sin, k_cache, v_cache, int(start), num_heads,
                                         num_kv_heads, head_dim))
    return _fused_projection(x.contiguous(), weight, bias, cos=cos, sin=sin, rotary_dim=2 * cos.shape[-1],
                             num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                             kv_cache=(k_cache, v_cache, int(start)), backend="auto")


# The prompt stream's plain projection (``F.linear``'s contract) and its gated Q/K split. The GEMM family's
# unified entry serves both on Ascend: the provider's gemm entry runs the plain arithmetic (bit for bit the torch
# reference at the served shapes), and ``qkv_norm_rope_kvcache`` below is the same family's fused projection.
_plain_gemm = getattr(xc, "gemm", None) if xc is not None else None


def _linear_supported(x, weight, bias=None):
    """Rows with a 2-D weight of the same dtype and width, as the unified entry takes them."""
    return (x.device.type == "npu" and x.ndim >= 2 and weight.device == x.device and weight.ndim == 2
            and weight.shape[-1] == x.shape[-1]
            and x.dtype == weight.dtype and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and (bias is None or (bias.device == x.device and bias.dtype == x.dtype
                                  and bias.shape == (weight.shape[0],))))


# The unified entry serves a row count that is a multiple of 64 with its compact kernel and any other row
# count by composing a zero-padded FP32 plan, a Cube pass and a finish pass (measured at the prompt stream's
# [M, 2048] x [2048, 6144] shape: 33 us versus 363 us at M=464, of which 149 us is the pad pass alone). The
# two plans agree bit for bit on every valid row -- output rows are independent -- so rounding the row count up
# to the next multiple of 64 is a pure dispatch change: same operator, same weights, same values. The padded
# rows read whatever the buffer held before (zero on first use, then the previous call's activations), and the
# measured garbage-row case is identical, so their content only has to be finite.
_pad_buffers: dict = {}       # (device, rows, K, dtype) -> zero-initialised [rows, K]
_PAD_ALIGN = 64


def _padded_rows(x, weight, bias):
    """``_plain_gemm`` on ``x`` after rounding its row count up to the compact kernel's alignment."""
    rows, k = x.shape
    aligned = (rows + _PAD_ALIGN - 1) // _PAD_ALIGN * _PAD_ALIGN
    if aligned == rows:
        return _plain_gemm(x, weight, bias)
    key = (x.device, aligned, k, x.dtype)
    buffer = _pad_buffers.get(key)
    if buffer is None:
        if torch.npu.is_current_stream_capturing():
            # No buffer for this geometry: keep the unpadded plan (identical values, only slower) rather than
            # allocating graph-pool memory that would outlive the capture. The eager pre-run every capture
            # starts with fills this, so a captured graph always finds its buffer.
            return _plain_gemm(x, weight, bias)
        buffer = torch.zeros((aligned, k), dtype=x.dtype, device=x.device)
        _pad_buffers[key] = buffer
    buffer[:rows].copy_(x)
    return _plain_gemm(buffer, weight, bias)[:rows]


@register("linear", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if callable(_plain_gemm) else None,
          version=VERSION)
@partial(reference.linear, check=_linear_supported)
def linear(x, weight, bias=None):
    # The entry takes rank-matching operands, so flatten the leading dimensions of a higher-rank input and
    # restore them (the same shape handling the official-operator alternative performs). The row count is
    # rounded up to the unified entry's compact-kernel alignment first (see _padded_rows: bitwise).
    lead = x.shape[:-1]
    out = _padded_rows(x.reshape(-1, x.shape[-1]), weight, bias)
    return out.reshape(*lead, weight.shape[0]) if lead else out


_no_action_rows = {}  # (batch, device) -> zero action-row starts/ends [B] and offsets [B + 1] (int64)


try:
    from xinfer.ops.backends.ascend.cann_action import _qkv_2b_cache_finish_pair as _qkv_2b_pair
except Exception:  # builds without the package's CANN bridge module
    _qkv_2b_pair = None

_qkv_2b_tail = None      # (call, get_error) of the fused wallx2b QKV tail, or False
_qkv_2b_tables = {}      # device -> (ones [2048, 32], zeros [2048, 32]) FP32 identity rotation
_qkv_2b_factors = {}     # (weight ptr, shape, dtype) -> (source, 1 + weight FP32)
_qkv_2b_warmed = False


def _qkv_2b_tail_pair():
    global _qkv_2b_tail
    if _qkv_2b_tail is None:
        _qkv_2b_tail = (_qkv_2b_pair() or False) if callable(_qkv_2b_pair) else False
    return _qkv_2b_tail or None


def _qkv_2b_factor(weight):
    entry = _qkv_2b_factors.get((weight.data_ptr(), tuple(weight.shape), weight.dtype))
    if entry is None:
        factor = 1 + weight.float()
        _qkv_2b_factors[(weight.data_ptr(), tuple(weight.shape), weight.dtype)] = (weight, factor)
        return factor
    return entry[1]


def _qkv_2b_identity_tables(device):
    pair = _qkv_2b_tables.get(device)
    if pair is None:
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("warm up the fused 2B QKV split on this stream before graph capture")
        pair = (torch.ones(2048, 32, dtype=torch.float32, device=device),
                torch.zeros(2048, 32, dtype=torch.float32, device=device))
        _qkv_2b_tables[device] = pair
    return pair


def _qkv_2b_split(packed, q_weight, k_weight, tail):
    """The wallx2b prefill split through the 2b-qkv fused tail (one AOT launch).

    The tail kernel takes its rows as a runtime parameter and fuses the per-head RMS
    norm, the (1 + w) factors, the 64-dim partial rotary, the head-major K/V append and
    the packed [Q | gate] return. The prefill chunk order -- each head's [q_h | gate_h]
    side by side, then K, then V -- is permuted into the accumulator layout the kernel
    reads, and identity rotation tables (cos = 1, sin = 0) neutralize its in-kernel
    rotary: ``z * 1 - partner * 0`` is exact, so the split returns unrotated values for
    the model's own ``rope_full``. Measured ~202us per call against ~516us for the
    replace kernel, with the outputs a single rounding apart (2 flips of 1.19M).
    """
    global _qkv_2b_warmed
    import ctypes

    rows = int(packed.shape[1])
    if not 0 < rows <= 2048:
        return None
    call, get_error = tail
    if torch.npu.is_current_stream_capturing() and not _qkv_2b_warmed:
        raise RuntimeError("warm up the fused 2B QKV split on this stream before graph capture")
    ones, zeros = _qkv_2b_identity_tables(packed.device)
    q_factor = _qkv_2b_factor(q_weight)
    k_factor = _qkv_2b_factor(k_weight)
    accumulator = torch.empty((rows, 5120), dtype=torch.float32, device=packed.device)
    head = accumulator.view(rows, 20, 256)
    source = packed.view(rows, 20, 256)
    head[:, 0:8] = source[:, 0:16:2]
    head[:, 8:10] = source[:, 16:18]
    head[:, 10:12] = source[:, 18:20]
    head[:, 12:20] = source[:, 1:16:2]
    q_out = torch.empty((rows, 4096), dtype=torch.bfloat16, device=packed.device)
    k_cache = torch.empty((2, rows, 256), dtype=torch.bfloat16, device=packed.device)
    v_cache = torch.empty((2, rows, 256), dtype=torch.bfloat16, device=packed.device)
    # the single library's TVM-FFI face serves first (the unified launch boundary); the
    # ctypes launcher of the family library is the fallback below
    if _native.launch_kernel("gemm_qkv_2b_norm_gate_cache_bnsdq",
                             (q_out, k_cache, v_cache, accumulator, q_factor, k_factor,
                              ones, zeros), (rows, rows, 0)):
        _qkv_2b_warmed = True
        q = q_out[:, :2048].view(rows, 8, 256).unsqueeze(0)
        k = k_cache.transpose(0, 1).unsqueeze(0).contiguous()
        v = v_cache.transpose(0, 1).unsqueeze(0).contiguous()
        gate = q_out[:, 2048:].reshape(1, rows, 2048)
        return q, k, v, gate
    stream = ctypes.c_void_p(torch.npu.current_stream(packed.device).npu_stream)
    status = call(ctypes.c_void_p(q_out.data_ptr()), ctypes.c_void_p(k_cache.data_ptr()),
                  ctypes.c_void_p(v_cache.data_ptr()), ctypes.c_void_p(accumulator.data_ptr()),
                  ctypes.c_void_p(q_factor.data_ptr()), ctypes.c_void_p(k_factor.data_ptr()),
                  ctypes.c_void_p(ones.data_ptr()), ctypes.c_void_p(zeros.data_ptr()),
                  rows, rows, 0, stream)
    _qkv_2b_warmed = True
    if status != 0:
        message = get_error()
        detail = message.decode("utf-8", "replace") if message else "unknown failure"
        raise RuntimeError(f"gemm_qkv_2b_norm_gate_cache: {detail}")
    q = q_out[:, :2048].view(rows, 8, 256).unsqueeze(0)
    k = k_cache.transpose(0, 1).unsqueeze(0).contiguous()
    v = v_cache.transpose(0, 1).unsqueeze(0).contiguous()
    gate = q_out[:, 2048:].reshape(1, rows, 2048)
    return q, k, v, gate


def _ft_qkv_factor_pair(q_weight, k_weight):
    """Keep each norm factor only while its source weight lives."""
    return (_native.derived("qkv_norm_factor", q_weight, lambda: q_weight.float() + 1),
            _native.derived("qkv_norm_factor", k_weight, lambda: k_weight.float() + 1))


def _gqa_bhsd():
    import os

    return "bhsd" in os.environ.get("XINFER_GQA_GLUE", "")


@register("qkv_norm_split", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if callable(_qkv_2b_pair) else None,
          version=VERSION)
@partial(reference.qkv_norm_split)
def qkv_norm_split(packed, q_weight, k_weight, num_heads, num_kv_heads, head_dim, eps):
    tail = _qkv_2b_tail_pair()
    if (tail is not None
            and packed.dim() == 3 and packed.shape[0] == 1
            and num_heads == 8 and num_kv_heads == 2 and head_dim == 256
            and float(eps) == 1e-6 and packed.shape[2] == 5120
            and packed.dtype == torch.bfloat16 and packed.is_contiguous()
            and q_weight is not None and k_weight is not None
            and tuple(q_weight.shape) == (256,) and tuple(k_weight.shape) == (256,)
            and q_weight.dtype == torch.float32 and k_weight.dtype == torch.float32
            and q_weight.is_contiguous() and k_weight.is_contiguous()):
        split = _qkv_2b_split(packed, q_weight, k_weight, tail)
        if split is not None:
            return split
    # 2B-COVERAGE: the vectorized split/apply kernel rides the AOT launcher (ctypes; the internal
    # xinfer_aot_* launcher precedent) with the torch-chain FP32 statistics and the (weight + 1)
    # factors mirrored verbatim from the reference's _zero_centered_heads chain — the library's
    # scalar qkv_split_rstd measured slower than the reference tail at the prefill shapes (149 us
    # per call at 464 rows), and this kernel is bitwise to the composition on identical statistics.
    # The module is absent without the delivered package and the guard falls through below.
    try:
        from xinfer.ops.backends.ascend.qkv_split_vec import qkv_split_vec as _qkv_vec
    except (ImportError, OSError):
        _qkv_vec = None
    try:
        from xinfer.ops.backends.ascend.qkv_split_bhsd import qkv_split_bhsd as _qkv_bhsd
    except (ImportError, OSError):
        _qkv_bhsd = None
    if (_qkv_vec is not None
            and packed.dim() == 3 and packed.shape[0] == 1 and packed.shape[2] == 5120
            and num_heads == 8 and num_kv_heads == 2 and head_dim == 256
            and float(eps) == 1e-6
            and packed.dtype == torch.bfloat16 and packed.is_contiguous()
            and packed.device.type == "npu"
            and torch.is_tensor(q_weight) and tuple(q_weight.shape) == (256,)
            and q_weight.device == packed.device
            and q_weight.dtype in (torch.float32, torch.bfloat16)
            and torch.is_tensor(k_weight) and tuple(k_weight.shape) == (256,)
            and k_weight.device == packed.device
            and k_weight.dtype in (torch.float32, torch.bfloat16)):
        try:
            b, n = packed.shape[0], packed.shape[1]
            query = packed[..., :4096].reshape(b, n, 8, 2, 256)
            kview = packed[..., 4096:4608].reshape(b, n, 2, 256)
            q32 = query[..., 0, :].float()
            k32 = kview.float()
            rq = torch.rsqrt(q32.pow(2).mean(-1, keepdim=True) + float(eps)).reshape(-1)
            rk = torch.rsqrt(k32.pow(2).mean(-1, keepdim=True) + float(eps)).reshape(-1)
            qf, kf = _ft_qkv_factor_pair(q_weight, k_weight)
            if (_gqa_bhsd() and _qkv_bhsd is not None and (b * n) % 8 == 0):
                # GQA-GLUE knife 1: head-major BHSD q/k/v direct
                # production (bitwise-equal values; layout permutation).
                return _qkv_bhsd(packed, rq, rk, qf, kf)
            return _qkv_vec(packed, rq, rk, qf, kf)
        except (ValueError, RuntimeError):
            pass
    raise ValueError("QKV layout is not supported by the native projection")


# ``qkv_norm_rope_kvcache`` is the unified projection entry with the norm, RoPE and cache writes as epilogues.
# The recorded reason for keeping it on the reference (a workspace the caller had to size, so it could not be
# captured) no longer holds: this build sizes its workspace per stream, the lineage binding of this same entry
# runs captured, and the fused path is far ahead of the composition (about 60 us against more than 300 per call
# at the action shape). The raw ``[N, K]`` weight takes the unified entry's generic route; prepacking at load
# time is the model-side follow-up that would take the fast kernels.


@register("qkv_norm_rope_kvcache", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if (callable(_fused_projection) and callable(_qk_norm)) or callable(native_2b_cache)
          else None,
          version=VERSION, deterministic=False)
@partial(reference.qkv_norm_rope_kvcache)
def qkv_norm_rope_kvcache(x, weight, q_weight, k_weight, cos, sin, k_cache, v_cache, start, num_heads,
                          num_kv_heads, head_dim, eps):
    if callable(native_2b_cache):
        return native_2b_cache(x, weight, q_weight, k_weight, cos, sin, k_cache, v_cache, start, num_heads,
                               num_kv_heads, head_dim, eps)
    return _fused_projection(
        x.contiguous(), weight, cos=cos, sin=sin, rotary_dim=2 * cos.shape[-1], num_heads=num_heads,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        extra_cols=weight.shape[0] - (num_heads + 2 * num_kv_heads) * head_dim,
        qk_norm=_qk_norm(q_weight, k_weight, float(eps), zero_centered=True),
        kv_cache=(k_cache, v_cache, int(start)), backend="auto")
