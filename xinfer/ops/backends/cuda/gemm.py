"""GEMM operators with fused epilogues on CUDA.

| operator | function here |
|---|---|
| ``gemm_epilogue`` | ``gemm`` |
| ``gemm_glu_epilogue`` | ``gemm_glu`` |
| ``qkv_rope_epilogue``, ``qkv_rope_kvcache`` | ``qkv_proj_rope`` |
| ``qkv_norm_rope_kvcache`` | ``qkv_proj_rope`` (``qk_norm``) |
| ``gemm_add_rms_norm`` | ``gemm``, then ``add_rms_norm`` (integrated GPUs: ``gemm`` with the residual, ``rmsnorm``) |
| ``mlp_add_rms_norm`` | the fused MLP where a kernel takes the call, else ``gemm_glu`` and ``gemm_add_rms_norm`` |
| ``prefetch_weights``, ``join_prefetch`` | L2 prefetches on a side stream (integrated GPUs only) |
| ``mxfp8_quantize`` | ``quantize_mxfp8`` |
| ``mxfp8_gemm_epilogue``, ``mxfp8_gemm_glu_quantize``, ``mxfp8_qkv_rope_epilogue`` | the three with MXFP8 operands |

    gemm(x, weight, bias, activation=,           out = act(row_scale * (x @ weight^T + bias)) + residual
         residual=, row_scale=, quantize=)
    gemm_glu(x, weight, bias, activation=,       out = act(gate) * up, [gate | up] = x @ weight^T + bias
             quantize=)
    qkv_proj_rope(x, weight, bias, ...)          Q | K | V = x @ weight^T + bias, RoPE on Q and K, K / V optionally
                                                 written to a KV cache

The operands are BF16, ``x`` ``[M, K]`` and ``weight`` ``[N, K]`` as the model stores it (contiguous, or the
transposed view of a contiguous ``[K, N]`` tensor), or both are MXFP8 pairs ``(values, scales)``
(``xinfer.ops.layout.mxfp8``); the bias and the residual are BF16 either way. Results are BF16, or with
``quantize=True`` an MXFP8 pair with one scale per 128 columns.

Every function runs through the generic path: one cuBLASLt GEMM, then the epilogue as a separate elementwise kernel
(BF16 operands on any CUDA GPU, MXFP8 operands where cuBLASLt has block-scaled FP8 GEMMs: SM120). An architecture
module (``gemm_sm120``, ``gemm_sm110``) adds kernels that fuse the epilogue into the GEMM; each kernel states the
calls it takes (``takes``), and a function runs the first kernel of its architecture that takes the call, the generic
path otherwise. A kernel that needs the weight in another memory order builds that copy on the weight's first call and
keeps it while the weight lives.

An operator call with an option the functions do not take runs on the torch reference. Some kernels differ at
rounding level across runs (``deterministic=False``). ``qkv_rope_epilogue`` builds the packed RoPE table once per
``cos`` / ``sin`` pair, which the layers of a forward share.

On integrated GPUs (Jetson Thor: DRAM shared with the CPU, a small L2) ``gemm_add_rms_norm`` adds the residual in the
GEMM (the sum rounded once, from the FP32 product) and normalizes the sum, where the two-tensor ``add_rms_norm``
would run at DRAM bandwidth, and ``prefetch_weights`` prefetches weights into L2 from a side stream (an RTX 5090
would lose time on it). Elsewhere ``gemm_add_rms_norm`` runs ``gemm_epilogue`` and ``add_rms_norm``, with their bits,
and ``prefetch_weights`` does nothing. ``mlp_add_rms_norm`` runs the fused MLP where a kernel takes the call (its
result is added into ``residual`` in place, which it returns).
"""

from __future__ import annotations

import dataclasses
import weakref
from typing import NamedTuple, Optional

import torch

from xinfer.ops.backends.cuda._native import KERNELS, NATIVE, Carver, arch, derived, empty_mxfp8, lib, scratch
from xinfer.ops.backends.cuda.norm import add_rms_norm, rmsnorm
from xinfer.ops.layout import mxfp8 as layout
from xinfer.ops.reference import gemm as reference
from xinfer.ops.registry import register


def _is_pair(t):
    return isinstance(t, (tuple, list)) and len(t) == 2


ACTIVATIONS = {"silu": 0, "gelu_tanh": 1, "gelu": 2}  # codes of the epilogue kernels


@dataclasses.dataclass
class Call:
    """One GEMM call as the kernels see it."""

    x: torch.Tensor  # [M, K], contiguous: BF16, or the E4M3 values of an MXFP8 pair
    weight: torch.Tensor  # [N, K] (gemm_glu: [2 N, K]) as passed by the caller: BF16, or E4M3 values
    bias: Optional[torch.Tensor]
    activation: Optional[str]
    m: int
    n: int  # output features (gemm_glu: per half)
    k: int
    residual: Optional[torch.Tensor] = None  # BF16 [M, N]
    row_scale: Optional[torch.Tensor] = None  # FP32 [M]
    x_scale: Optional[torch.Tensor] = None  # the scales of MXFP8 operands
    weight_scale: Optional[torch.Tensor] = None
    quantize: bool = False  # the result is an MXFP8 pair
    rope: Optional[tuple] = None  # qkv_proj_rope: (heads, KV heads, head dim, rotary dim, packed table or None)
    qk_order: str = "standard"  # qkv_proj_rope: "any" lets a kernel write Q and K in rotary-pair order

    @property
    def device(self):
        return self.x.device

    @property
    def mxfp8(self):
        return self.x_scale is not None

    @property
    def features(self):
        """What the call asks for beyond the BF16 product: a subset of {"mxfp8", "bias", "residual", "row_scale",
        <activation>, "quantize"}."""
        parts = {name for name in ("bias", "residual", "row_scale") if getattr(self, name) is not None}
        parts |= {name for name in ("mxfp8", "quantize") if getattr(self, name)}
        return frozenset(parts | ({self.activation} if self.activation else set()))


def weight_nk(weight):
    """``weight`` as a contiguous ``[N, K]`` tensor: itself, or a copy made once."""
    if weight.is_contiguous():
        return weight
    return derived("nk", weight, weight.contiguous)


def weight_kn(weight):
    """``weight`` as a contiguous ``[K, N]`` tensor: its transposed view where that is contiguous, else a copy."""
    if weight.t().is_contiguous():
        return weight.t()
    return derived("kn", weight, lambda: weight.t().contiguous())


def _kernel_for(op, call):
    device_arch = arch(call.device)
    for cls in KERNELS[op]:
        if cls.arch == device_arch and cls.takes(call):
            return cls
    return None


def kernel_name(op, x, weight, bias=None, activation=None, residual=None, row_scale=None, quantize=False, **rope):
    """Name of the kernel that runs ``op(x, weight, bias, ...)`` on this GPU, or ``"generic"``. For
    ``qkv_proj_rope`` pass its ``num_heads``, ``head_dim`` and table arguments too."""
    if op == "qkv_proj_rope":
        c = _qkv_call(x, weight, bias, **rope)[0]
        fused = rope.get("qk_norm") is None and rope.get("kv_cache") is None
        cls = _kernel_for(op, c) if fused else None
    else:
        cls = _kernel_for(op, _call(op, x, weight, bias, activation, residual, row_scale, quantize))
    return "generic" if cls is None else cls.__name__


# ---------------------------------------------------------------------------
# Generic path
# ---------------------------------------------------------------------------
_CUBLAS_BYTES = 0 if lib is None else int(lib.gemm_generic_workspace_size())


def _cublas(out, c, workspace, residual=None):
    """out (BF16) = x @ weight^T (+ bias) (+ residual) through cuBLASLt. BF16 operands: on the weight's memory as
    it is where it is dense, the residual added to the FP32 product. MXFP8 operands: the product is rounded to
    BF16 before the residual is added."""
    if c.mxfp8:
        lib.gemm_generic_mxfp8(out, c.x, c.x_scale, c.weight, c.weight_scale, c.bias, workspace, False)
        if residual is not None:
            lib.gemm_epilogue_residual(out, residual, None, 1.0)
        return
    kn = not c.weight.is_contiguous() and c.weight.t().is_contiguous()
    w = c.weight.t() if kn else weight_nk(c.weight)
    lib.gemm_generic(out, c.x, w, c.bias, residual, 1.0, 1.0, workspace, kn)


def _generic_gemm(c, out):
    if c.quantize:  # the BF16 result goes to scratch memory and is quantized from there
        carve = Carver(scratch("gemm", Carver.size(_CUBLAS_BYTES, c.m * c.n * 2), c.device))
        workspace, y = carve.take(_CUBLAS_BYTES), carve.tensor((c.m, c.n), torch.bfloat16)
    else:
        workspace, y = scratch("gemm", _CUBLAS_BYTES, c.device), out
    _cublas(y, c, workspace, c.residual)
    if c.row_scale is not None:
        lib.gemm_epilogue_row_scale(y, c.row_scale)
    if c.activation is not None:
        lib.gemm_epilogue_act(y, ACTIVATIONS[c.activation])
    if c.quantize:
        lib.quantize_mxfp8_into(y, out[0], out[1], True)


def _generic_glu(c, out):
    carve = Carver(scratch("gemm", Carver.size(_CUBLAS_BYTES, c.m * 2 * c.n * 2), c.device))
    workspace = carve.take(_CUBLAS_BYTES)
    y = carve.tensor((c.m, 2 * c.n), torch.bfloat16)  # [gate | up]
    _cublas(y, c, workspace)
    if c.quantize:
        lib.gemm_epilogue_glu_quant(out[0], out[1], y, ACTIVATIONS[c.activation], c.n, True)
    else:
        lib.gemm_epilogue_glu(out, y, ACTIVATIONS[c.activation], c.n)


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------
def _call(op, x, weight, bias, activation, residual=None, row_scale=None, quantize=False):
    x_scale = weight_scale = None
    if _is_pair(x) or _is_pair(weight):
        if not (_is_pair(x) and _is_pair(weight)):
            raise ValueError(f"{op}: x and weight must both be BF16 tensors or both be MXFP8 pairs")
        (x, x_scale), (weight, weight_scale) = x, weight
    dtypes = (torch.float8_e4m3fn, torch.uint8) if x_scale is not None else (torch.bfloat16,)
    if x.dim() != 2 or weight.dim() != 2 or x.dtype not in dtypes or weight.dtype not in dtypes:
        raise ValueError(f"{op}: x must be [M, K] and weight [N, K], both BF16 or both MXFP8 pairs")
    if not x.is_contiguous():
        raise ValueError(f"{op}: x must be contiguous")
    m, k = x.shape
    rows = weight.shape[0]
    if weight.shape[1] != k or weight.device != x.device:
        raise ValueError(f"{op}: weight {tuple(weight.shape)} does not match x {tuple(x.shape)} on {x.device}")
    if x_scale is not None:
        if k % 32 or not weight.is_contiguous():
            raise ValueError(f"{op}: MXFP8 operands need K a multiple of 32 and a contiguous weight")
        if x_scale.numel() < layout.scale_elements(m, k) or weight_scale.numel() < layout.scale_elements(rows, k):
            raise ValueError(f"{op}: the scales of an MXFP8 operand are shorter than layout.mxfp8.scale_elements")
    if bias is not None and (bias.dtype != torch.bfloat16 or bias.numel() != rows or not bias.is_contiguous()):
        raise ValueError(f"{op}: bias must be a contiguous BF16 tensor of {rows} elements")
    if op == "gemm_glu":
        if rows % 2 or activation not in ("silu", "gelu_tanh"):
            raise ValueError("gemm_glu: weight must be [2 N, K] ([gate; up]) and activation 'silu' or 'gelu_tanh'")
        rows //= 2
    elif activation is not None and activation not in ACTIVATIONS:
        raise ValueError(f"gemm: activation must be one of {sorted(ACTIVATIONS)}")
    if residual is not None:
        if residual.dtype != torch.bfloat16 or tuple(residual.shape) != (m, rows) or not residual.is_contiguous():
            raise ValueError(f"{op}: residual must be a contiguous BF16 tensor of shape {(m, rows)}")
        if activation is not None or row_scale is not None:
            raise ValueError(f"{op}: residual does not combine with an activation or row_scale")
    if row_scale is not None and (row_scale.dtype != torch.float32 or row_scale.numel() != m
                                  or not row_scale.is_contiguous()):
        raise ValueError(f"{op}: row_scale must be a contiguous FP32 tensor of {m} elements")
    if quantize and rows % 32:
        raise ValueError(f"{op}: quantize needs an output width that is a multiple of 32")
    return Call(x, weight, bias, activation, m, rows, k, residual, row_scale, x_scale, weight_scale, bool(quantize))


def _out(out, shape, c):
    if c.quantize:
        if out is None:
            return empty_mxfp8(*shape, c.device)
        if not _is_pair(out) or tuple(out[0].shape) != tuple(shape) or out[1].numel() < layout.scale_elements(*shape):
            raise ValueError(f"out must be an MXFP8 pair for shape {tuple(shape)}")
        return out
    if out is None:
        return torch.empty(shape, dtype=torch.bfloat16, device=c.device)
    if tuple(out.shape) != tuple(shape) or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous BF16 tensor of shape {tuple(shape)}")
    if c.mxfp8 and c.residual is not None and out.data_ptr() == c.residual.data_ptr():
        raise ValueError("with MXFP8 operands the residual cannot be the output")
    return out


def gemm(x, weight, bias=None, *, activation=None, residual=None, row_scale=None, quantize=False, out=None):
    """``act(row_scale * (x @ weight^T + bias)) + residual``: x ``[M, K]`` and weight ``[N, K]`` BF16 or MXFP8
    pairs, bias BF16 ``[N]``; BF16 ``[M, N]``, or an MXFP8 pair with ``quantize=True``.

    ``activation``: None, ``"silu"``, ``"gelu"`` (erf) or ``"gelu_tanh"``. ``residual``: BF16 ``[M, N]`` added to
    the product (with BF16 operands it may be ``out``; with MXFP8 operands the product is rounded to BF16
    first); not with an activation or ``row_scale``. ``row_scale``: FP32 ``[M]``, one factor per row (a 0 / 1
    token mask, for example).
    """
    c = _call("gemm", x, weight, bias, activation, residual, row_scale, quantize)
    out = _out(out, (c.m, c.n), c)
    cls = _kernel_for("gemm", c)
    if cls is None:
        _generic_gemm(c, out)
    else:
        cls.run(c, out)
    return out


def gemm_glu(x, weight, bias=None, *, activation="silu", quantize=False, out=None):
    """``act(gate) * up`` with ``[gate | up] = x @ weight^T + bias``: weight ``[2 N, K]`` holds the gate rows, then
    the up rows; BF16 ``[M, N]``, or an MXFP8 pair with ``quantize=True``."""
    c = _call("gemm_glu", x, weight, bias, activation, quantize=quantize)
    out = _out(out, (c.m, c.n), c)
    cls = _kernel_for("gemm_glu", c)
    if cls is None:
        _generic_glu(c, out)
    else:
        cls.run(c, out)
    return out


class QkNorm(NamedTuple):
    """Per-head RMSNorm of the Q and K heads before RoPE: ``t * rsqrt(mean(t^2) + eps) * w``, with ``w`` the BF16
    ``[head_dim]`` weight of Q or K, or ``1 + w`` when ``zero_centered``."""

    q_weight: torch.Tensor
    k_weight: torch.Tensor
    eps: float = 1e-6
    zero_centered: bool = False


def _qkv_call(x, weight, bias=None, *, num_heads, head_dim, num_kv_heads=None, cos=None, sin=None,
              rope_table=None, rotary_dim=None, qk_norm=None, kv_cache=None, qk_order="standard", out=None):
    """The checked call of qkv_proj_rope and its (extra rows, table flag)."""
    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    c = _call("qkv_proj_rope", x, weight, bias, None)
    extra = c.n - (num_heads + 2 * num_kv_heads) * head_dim
    if extra < 0:
        raise ValueError(f"qkv_proj_rope: weight has {c.n} rows, the heads need at least "
                         f"{(num_heads + 2 * num_kv_heads) * head_dim}")
    if (rope_table is None) == (cos is None) or (cos is None) != (sin is None):
        raise ValueError("qkv_proj_rope: pass cos and sin, or rope_table")
    table = rope_table is not None
    if table and (rope_table.dtype != torch.float32 or tuple(rope_table.shape) != (rotary_dim // 4, c.m, 4)):
        raise ValueError("qkv_proj_rope: rope_table must be layout.mxfp8.rope_table of these M rows")
    c.rope = (num_heads, num_kv_heads, head_dim, rotary_dim, rope_table)
    c.qk_order = qk_order
    return c, extra


def qkv_proj_rope(x, weight, bias=None, *, num_heads, head_dim, num_kv_heads=None, cos=None, sin=None,
                  rope_table=None, rotary_dim=None, qk_norm=None, kv_cache=None, qk_order="standard", out=None):
    """The attention input projection with rotate-half RoPE on the Q and K heads.

    x ``[M, K]`` and ``weight`` are BF16 or MXFP8 pairs. ``weight`` ``[(H + 2 H_kv) * D + E, K]`` holds the Q, K
    and V rows in that order, then ``E >= 0`` extra rows whose outputs pass through unchanged (an output gate,
    for example). RoPE tables: FP32 ``cos`` and ``sin`` ``[M, rotary_dim / 2]``, or ``rope_table`` from
    ``xinfer.ops.layout.mxfp8.rope_table``; ``rotary_dim`` defaults to ``head_dim``. ``qk_norm``
    (:class:`QkNorm`) normalizes each Q and K head before the rotation.

    Returns BF16 ``[M, (H + 2 H_kv) * D + E]`` (Q | K | V | extra). With ``kv_cache=(k_cache, v_cache, start)``,
    K (after RoPE) and V go to rows ``start .. start + M`` of the BF16 caches instead, row-major
    ``[rows, H_kv * D]`` or head-major ``[H_kv, rows, D]`` (a strided view with a unit last stride), and the
    result is ``[M, H * D + E]`` (Q | extra).

    ``qk_order="any"``: where a kernel that writes them so takes the call, the channels of every Q and K head come
    in rotary-pair order (``j`` and ``j + rotary_dim / 2`` adjacent), which leaves attention over them unchanged.
    """
    c, extra = _qkv_call(x, weight, bias, num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads,
                         cos=cos, sin=sin, rope_table=rope_table, rotary_dim=rotary_dim, qk_order=qk_order)
    num_heads, num_kv_heads, head_dim, rotary_dim, _ = c.rope
    table = rope_table is not None
    out = _out(out, (c.m, num_heads * head_dim + extra if kv_cache is not None else c.n), c)
    cos = rope_table if table else cos
    if qk_norm is None and kv_cache is None:
        cls = _kernel_for("qkv_proj_rope", c)
        if cls is not None:
            cls.run(c, out)
            return out

    def normalize(y):
        if qk_norm is not None:
            lib.gemm_epilogue_qk_norm(y, qk_norm.q_weight, qk_norm.k_weight, num_heads, num_kv_heads, head_dim,
                                      float(qk_norm.eps), bool(qk_norm.zero_centered))

    if kv_cache is None:
        y = out  # norm and RoPE in place on the GEMM result
        _cublas(y, c, scratch("gemm", _CUBLAS_BYTES, c.device))
        normalize(y)
        lib.gemm_epilogue_rope(out, y, cos, sin, num_heads + num_kv_heads, head_dim, rotary_dim, False, False,
                               table)
        return out
    k_cache, v_cache, start = kv_cache
    carve = Carver(scratch("gemm", Carver.size(_CUBLAS_BYTES, c.m * c.n * 2), c.device))
    workspace = carve.take(_CUBLAS_BYTES)
    y = carve.tensor((c.m, c.n), torch.bfloat16)
    _cublas(y, c, workspace)
    normalize(y)
    lib.gemm_epilogue_rope_kvcache(out, y, cos, sin, k_cache, v_cache, int(start), num_heads, num_kv_heads,
                                   head_dim, rotary_dim, False, False, table)
    return out


def quantize_mxfp8(x, *, scale_block=128, out=None):
    """BF16 ``x`` ``[M, K]`` (K a multiple of 32) as an MXFP8 pair: per row and block, the scale is the smallest
    power of two that keeps the block's largest magnitude in E4M3 range, and the values are rounded to nearest.

    ``scale_block``: 32 gives every 32 columns its own scale; 128 (the default) shares one scale between the
    four blocks of 128 columns, which the fused kernels write and read faster.
    """
    if x.dim() != 2 or x.dtype != torch.bfloat16 or x.shape[1] % 32 or not x.is_contiguous():
        raise ValueError("quantize_mxfp8: x must be a contiguous BF16 [M, K] tensor with K a multiple of 32")
    if scale_block not in (32, 128):
        raise ValueError("quantize_mxfp8: scale_block must be 32 or 128")
    values, scales = empty_mxfp8(*x.shape, x.device) if out is None else out
    lib.quantize_mxfp8_into(x, values, scales, scale_block == 128)
    return values, scales


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

@register("gemm_epilogue", backend="cuda", deterministic=False, **NATIVE)
def gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, residual_gate=None, row_class=None,
                  row_scale=None):
    if residual_gate is not None or row_class is not None or (
            residual is not None and (activation is not None or row_scale is not None)):
        return reference.gemm_epilogue(x, weight, bias, activation=activation, residual=residual,
                                       residual_gate=residual_gate, row_class=row_class, row_scale=row_scale)
    return gemm(x.contiguous(), weight, bias, activation=activation, residual=residual, row_scale=row_scale)


@register("gemm_glu_epilogue", backend="cuda", **NATIVE)
def gemm_glu_epilogue(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
    if modulation is not None:
        return reference.gemm_glu_epilogue(x, weight, bias=bias, modulation=modulation, row_class=row_class, eps=eps)
    return gemm_glu(x.contiguous(), weight, bias, activation="silu")


_integrated = {}  # device index -> whether the GPU shares its memory with the CPU


def _integrated_gpu(device) -> bool:
    if device.index not in _integrated:
        _integrated[device.index] = bool(torch.cuda.get_device_properties(device).is_integrated)
    return _integrated[device.index]


@register("gemm_add_rms_norm", backend="cuda", deterministic=False, **NATIVE)
def gemm_add_rms_norm(x, projection, residual, weight, eps, bias=None):
    if _integrated_gpu(x.device) and residual.is_contiguous():
        x = x.contiguous()
        out = gemm(x, projection, bias, residual=residual.view(x.shape[0], -1)).view(residual.shape)
        return out, rmsnorm(out, weight, eps)
    return add_rms_norm(gemm_epilogue(x, projection, bias).view(residual.shape), residual, weight, eps)


@register("mlp_add_rms_norm", backend="cuda", deterministic=False, **NATIVE)
def mlp_add_rms_norm(x, gate_up, down, residual, weight, eps):
    x = x.contiguous()
    c = _call("gemm_glu", x, gate_up, None, "silu")
    cls = _kernel_for("fused_mlp", c)
    if cls is not None and tuple(down.shape) == (c.k, c.n) and residual.is_contiguous():
        cls.run(c, down, residual.view(c.m, -1))  # adds into the residual in place
        return residual, rmsnorm(residual, weight, eps)
    return gemm_add_rms_norm(gemm_glu_epilogue(x, gate_up), down, residual, weight, eps)


PREFETCH_GRID = 2  # CTAs issuing the L2 prefetches: a few keep DRAM busy without taking SMs from the compute stream
_prefetch_streams = {}  # (device, caller's stream) -> side stream
_prefetching = set()  # keys whose side stream has work the caller's stream has not waited for


@register("prefetch_weights", backend="cuda", **NATIVE)
def prefetch_weights(weights):
    """On integrated GPUs, L2 prefetches of ``weights`` on a side stream forked from the caller's stream (one per
    caller's stream, so lanes capturing graphs on different streams do not share it); the caller's stream waits
    for them at ``join_prefetch``. On other GPUs nothing."""
    stream = torch.cuda.current_stream()
    if not _integrated_gpu(stream.device):
        return
    key = (stream.device, stream.cuda_stream)
    side = _prefetch_streams.setdefault(key, torch.cuda.Stream(device=stream.device))
    side.wait_stream(stream)
    with torch.cuda.stream(side):
        for weight in weights:
            lib.l2_prefetch(weight, 0, weight.numel() * weight.element_size(), PREFETCH_GRID, 256)
    _prefetching.add(key)


@register("join_prefetch", backend="cuda", **NATIVE)
def join_prefetch():
    stream = torch.cuda.current_stream()
    key = (stream.device, stream.cuda_stream)
    if key in _prefetching:
        stream.wait_stream(_prefetch_streams[key])
        _prefetching.discard(key)


_rope_tables = {}  # (cos, sin) -> their packed RoPE table


def _rope_table(cos, sin):
    """The packed RoPE table of ``cos`` and ``sin``, built once while they live (they are not modified in
    place): every layer of a forward passes the same tensors."""
    key = (cos.data_ptr(), tuple(cos.shape), sin.data_ptr())
    if key not in _rope_tables:
        _rope_tables[key] = layout.rope_table(cos, sin)
        for t in (cos, sin):
            weakref.finalize(t if t._base is None else t._base, _rope_tables.pop, key, None)
    return _rope_tables[key]


def _qkv_rope_packed(x, weight, bias, cos, sin, num_heads, head_dim, *, qk_order):
    return qkv_proj_rope(x.contiguous(), weight, bias, rope_table=_rope_table(cos, sin), num_heads=num_heads,
                         head_dim=head_dim, qk_order=qk_order)


@register("qkv_rope_epilogue", backend="cuda", deterministic=False, **NATIVE)
def qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim, *, qk_order="standard"):
    return _qkv_rope_packed(x, weight, bias, cos, sin, num_heads, head_dim, qk_order=qk_order)


@register("qkv_rope_split", backend="cuda", deterministic=False, **NATIVE)
def qkv_rope_split(x, weight, bias, cos, sin, num_heads, head_dim, *, qk_order="standard"):
    out = _qkv_rope_packed(x, weight, bias, cos, sin, num_heads, head_dim, qk_order=qk_order)
    return out.view(x.shape[0], 3, num_heads, head_dim).unbind(1)


@register("gemm_split", backend="cuda", deterministic=False, **NATIVE)
def gemm_split(x, weight, split_size, *, row_scale=None):
    out = gemm_epilogue(x, weight, row_scale=row_scale)
    return out[:, :split_size], out[:, split_size:]


@register("qkv_norm_rope_kvcache", backend="cuda", deterministic=False, **NATIVE)
def qkv_norm_rope_kvcache(x, weight, q_weight, k_weight, cos, sin, k_cache, v_cache, start, num_heads,
                          num_kv_heads, head_dim, eps):
    return qkv_proj_rope(x.contiguous(), weight, cos=cos, sin=sin, rotary_dim=2 * cos.shape[-1],
                            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                            qk_norm=QkNorm(q_weight, k_weight, float(eps), zero_centered=True),
                            kv_cache=(k_cache, v_cache, int(start)))


@register("qkv_rope_kvcache", backend="cuda", deterministic=False, **NATIVE)
def qkv_rope_kvcache(x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads, head_dim):
    return qkv_proj_rope(x.contiguous(), weight, bias, cos=cos, sin=sin, rotary_dim=2 * cos.shape[-1],
                            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                            kv_cache=(k_cache, v_cache, int(start)))


qkv_rope_kvcache.packed_q_only = True


@register("mxfp8_quantize", backend="cuda", **NATIVE)
def mxfp8_quantize(x, *, share_scale_128=True):
    return quantize_mxfp8(x.contiguous(), scale_block=128 if share_scale_128 else 32)


@register("mxfp8_gemm_epilogue", backend="cuda", **NATIVE)
def mxfp8_gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, row_mask=None, out_quant=False):
    q, scales = x
    if row_mask is not None:  # the rows of x with a 0 mask become FP8 zeros
        q = (q.view(torch.uint8) * row_mask.to(torch.uint8)[:, None]).view(torch.float8_e4m3fn)
    return gemm((q, scales), weight, bias, activation=activation, residual=residual, quantize=out_quant)


@register("mxfp8_gemm_glu_quantize", backend="cuda", **NATIVE)
def mxfp8_gemm_glu_quantize(x, weight):
    return gemm_glu(x, weight, activation="silu", quantize=True)


@register("mxfp8_qkv_rope_epilogue", backend="cuda", **NATIVE)
def mxfp8_qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim):
    return qkv_proj_rope(x, weight, bias, rope_table=_rope_table(cos, sin), num_heads=num_heads,
                            head_dim=head_dim)


from xinfer.ops.backends.cuda import gemm_sm110, gemm_sm120  # noqa: E402, F401 - list the kernels of each architecture
