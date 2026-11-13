"""GEMM operators with fused epilogues on CUDA, through xcompute (``_xcompute``).

Stream-K GEMMs may differ at rounding level across runs (``deterministic=False``).

| operator | xcompute entry |
|---|---|
| ``gemm_add_rms_norm`` | ``gemm`` (``residual``), then ``rmsnorm``, on integrated GPUs |
| ``mlp_add_rms_norm`` | ``fused_mlp`` (residual), then ``rmsnorm`` |
| ``prefetch_weights``, ``join_prefetch`` | ``l2_prefetch`` on a side stream (integrated GPUs only) |

``qkv_rope_epilogue(..., qk_order="any")`` packs the weight in rotary-pair row order once and, where a kernel writing
that order is listed for the call, returns Q and K in pair order; its packed RoPE table is built once per ``cos`` /
``sin`` pair, which the layers of a forward share. ``gemm_add_rms_norm`` adds the residual in the GEMM (the sum rounded
once, from the FP32 product) and normalizes the sum on integrated GPUs, where the two-tensor ``add_rms_norm`` runs at
DRAM bandwidth; elsewhere it runs ``gemm_epilogue`` and ``add_rms_norm``, with their bits. ``mlp_add_rms_norm`` runs
the fused MLP where xcompute lists a kernel for the call, with the intermediate zero-padded to a multiple of
``FUSED_MLP_PAD``, adding into ``residual`` in place (the sum is returned in its memory); elsewhere it runs
``gemm_glu_epilogue`` and ``gemm_add_rms_norm`` (on discrete GPUs ``gemm_epilogue`` and ``add_rms_norm``, with their
bits).
"""

from __future__ import annotations

import weakref

import torch

from xinfer.ops.backends._xcompute import XCOMPUTE, xc, xcompute_with
from xinfer.ops.backends.cuda.norm import add_rms_norm
from xinfer.ops.registry import register

FUSED_MLP_PAD = 1024  # intermediate multiple xcompute's fused MLP kernels take in whole chunks (padding adds zeros)

_supports_kn = True
_packed = {}  # (op, weight key) -> packed weight, or None when packing does not help
_fused_mlp = {}  # (gate/up key, down key) -> (packed gate/up, packed down), or None when no fused kernel is listed


def _key(weight):
    return weight.data_ptr(), tuple(weight.shape), tuple(weight.stride())


def _cached(cache, key, weights, make):
    """``cache[key]``, made on the first call. The entry is dropped when the tensor owning any of ``weights``' storage
    is freed, so a later weight allocated at the same address never gets another weight's packing."""
    if key not in cache:
        cache[key] = make()
        for weight in weights:
            weakref.finalize(weight if weight._base is None else weight._base, cache.pop, key, None)
    return cache[key]


def _listed_weight(op, x, weight, **options):
    """``weight``, or its xcompute packing when the call would run the generic path on the raw ``[N, K]`` weight
    and runs a listed kernel on the packed one.

    A weight is packed once, on its first call that would run the generic path (eager: the engine runs every new
    graph key before capturing it), and the packing is kept only if that call runs a listed kernel with it.
    """
    if xc.selected_backend(op, x, weight, **options) != "generic":
        return weight

    def pack():
        packed = xc.prepack_weight(weight, op=op)
        return packed if xc.selected_backend(op, x, packed, **options) != "generic" else None

    packed = _cached(_packed, (op, *_key(weight)), (weight,), pack)
    return weight if packed is None or xc.selected_backend(op, x, packed, **options) == "generic" else packed


def _kn(weight):
    """Share the transposed weight when xcompute supports ``kn``. Older builds need a dense ``[N, K]`` weight."""
    global _supports_kn
    if not _supports_kn:
        return weight.contiguous()
    try:
        packed = xc.prepack_weight(weight, op="gemm", layout="kn")
    except ValueError as exc:
        if "layout= only takes 'nk_k64blk'" not in str(exc):
            raise
        _supports_kn = False
        return weight.contiguous()
    if packed.tensors["w"].data_ptr() != weight.data_ptr():
        raise ValueError("xcompute packed the GEMM weight into a copy")
    return packed


@register("gemm_epilogue", backend="cuda", deterministic=False, **XCOMPUTE)
def gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, residual_gate=None, row_class=None,
                  row_scale=None):
    x = x.contiguous()
    options = dict(bias=bias, activation=activation, residual=residual, residual_gate=residual_gate,
                   row_class=row_class, row_scale=row_scale)
    if not weight.is_contiguous() and weight.t().is_contiguous():
        weight = _kn(weight)
    else:
        weight = _listed_weight("gemm", x, weight, **options)
    return xc.gemm(x, weight, **options, backend="auto")


@register("gemm_glu_epilogue", backend="cuda", **XCOMPUTE)
def gemm_glu_epilogue(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
    k = x.shape[-1]
    prenorm = None if modulation is None else xc.AdaRms(scale=modulation[:, :k], shift=modulation[:, k:2 * k],
                                                        row_class=row_class, eps=float(eps))
    x, options = x.contiguous(), dict(bias=bias, activation="silu", prenorm=prenorm)
    return xc.gemm_glu(x, _listed_weight("gemm_glu", x, weight, **options), **options, backend="auto")


_integrated = {}  # device index -> whether the GPU shares memory with the CPU


def _integrated_gpu(device) -> bool:
    """Whether ``device`` shares memory with the CPU (low DRAM bandwidth, small L2: Jetson). There gemm_add_rms_norm
    adds the residual in the GEMM, and prefetch_weights prefetches (Jetson Thor −4.6 ms per Wall-OSS request; on an
    RTX 5090 it would cost +5.3 ms)."""
    if device.index not in _integrated:
        _integrated[device.index] = bool(torch.cuda.get_device_properties(device).is_integrated)
    return _integrated[device.index]


@register("gemm_add_rms_norm", backend="cuda", deterministic=False, **XCOMPUTE)
def gemm_add_rms_norm(x, projection, residual, weight, eps, bias=None):
    if _integrated_gpu(x.device):
        x = x.contiguous()
        options = dict(bias=bias, residual=residual.view(x.shape[0], -1))
        out = xc.gemm(x, _listed_weight("gemm", x, projection, **options), **options, backend="auto")
        out = out.view(residual.shape)
        return out, xc.rmsnorm(out, weight, float(eps), True)
    return add_rms_norm(gemm_epilogue(x, projection, bias).view(residual.shape), residual, weight, eps)


_pair_weights = {}  # (weight, bias key, heads) -> the weight and bias packed in rotary-pair row order, or None
_rope_tables = {}  # (cos key, sin key) -> xcompute's packed RoPE table of them


def _rope_pairs_weight(weight, bias, num_heads, head_dim):
    """``weight`` and ``bias`` packed with Q/K rows in rotary-pair order, or None when xcompute's packing picks
    another layout for them on this GPU (no kernel listed for that order). Packed once per weight."""
    def pack():
        packed = xc.prepack_weight(weight, op="qkv_proj_rope", bias=bias, num_heads=num_heads, head_dim=head_dim)
        return packed if packed.layout.endswith("rope_pairs") else None

    key = (*_key(weight), bias.data_ptr(), num_heads, head_dim)
    return _cached(_pair_weights, key, (weight, bias), pack)


def _rope_table(cos, sin):
    """xcompute's packed RoPE table of ``cos`` and ``sin``, built once while they live (they are not modified in
    place): every layer of a forward passes the same tensors."""
    return _cached(_rope_tables, (*_key(cos), *_key(sin)), (cos, sin), lambda: xc.pack_rope_table(cos, sin))


@register("qkv_rope_epilogue", backend="cuda", deterministic=False, **XCOMPUTE)
def qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim, *, qk_order="standard"):
    x, table = x.contiguous(), _rope_table(cos, sin)
    if qk_order == "any":  # Q / K in rotary-pair order where xcompute lists a kernel writing that order
        packed = _rope_pairs_weight(weight, bias, num_heads, head_dim)
        options = dict(rope_table=table, out_layout="rope_pairs")
        if packed is not None and xc.selected_backend("qkv_proj_rope", x, packed, **options) != "generic":
            return xc.qkv_proj_rope(x, packed, **options, backend="auto")
    return xc.qkv_proj_rope(x, weight, bias, rope_table=table, num_heads=num_heads, num_kv_heads=num_heads,
                            head_dim=head_dim, backend="auto")


@register("qkv_norm_rope_kvcache", backend="cuda", deterministic=False, **xcompute_with("QkNorm"))
def qkv_norm_rope_kvcache(x, weight, q_weight, k_weight, cos, sin, k_cache, v_cache, start, num_heads,
                          num_kv_heads, head_dim, eps):
    return xc.qkv_proj_rope(
        x.contiguous(), weight, cos=cos, sin=sin, rotary_dim=2 * cos.shape[-1], num_heads=num_heads,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        extra_cols=weight.shape[0] - (num_heads + 2 * num_kv_heads) * head_dim,
        qk_norm=xc.QkNorm(q_weight, k_weight, float(eps), zero_centered=True),
        kv_cache=(k_cache, v_cache, int(start)), backend="auto")


@register("qkv_rope_kvcache", backend="cuda", deterministic=False, **XCOMPUTE)
def qkv_rope_kvcache(x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads, head_dim):
    return xc.qkv_proj_rope(x.contiguous(), weight, bias, cos=cos, sin=sin,
                            rotary_dim=2 * cos.shape[-1], num_heads=num_heads,
                            num_kv_heads=num_kv_heads, head_dim=head_dim,
                            kv_cache=(k_cache, v_cache, int(start)), backend="auto")


def _fused_mlp_weights(gate_up, down):
    """The fused MLP's packed weights, or None when xcompute lists no fused kernel for them on this GPU (its packing
    then picks the generic ``nk`` layout). Packed once per weight pair, on the first (eager) call."""

    def pack():
        packed = xc.prepack_weight(gate_up, op="fused_mlp", pad_n=FUSED_MLP_PAD)
        if packed.layout == "nk":
            return None
        layout = dict(layout="nk_k64blk") if packed.layout.endswith("_k64") else dict(op="gemm")  # K-blocked pairs
        return packed, xc.prepack_weight(down, pad_k=FUSED_MLP_PAD, **layout)

    return _cached(_fused_mlp, (*_key(gate_up), *_key(down)), (gate_up, down), pack)


@register("mlp_add_rms_norm", backend="cuda", deterministic=False, **XCOMPUTE)
def mlp_add_rms_norm(x, gate_up, down, residual, weight, eps):
    x = x.contiguous()
    fused = _fused_mlp_weights(gate_up, down)
    if fused is not None:
        rows = residual.view(x.shape[0], -1)
        options = dict(activation="silu", residual=rows, out=rows)  # added in place: no copy of the residual
        if xc.selected_backend("fused_mlp", x, fused[0], w_down=fused[1], **options) != "generic":
            xc.fused_mlp(x, fused[0], fused[1], backend="auto", **options)
            return residual, xc.rmsnorm(residual, weight, float(eps), True)
    return gemm_add_rms_norm(gemm_glu_epilogue(x, gate_up), down, residual, weight, eps)


PREFETCH_GRID = 2  # CTAs issuing L2 prefetches: a few keep DRAM busy without taking SMs from the compute stream
_prefetch_streams = {}  # (device, caller's stream) -> side stream
_prefetching = set()  # keys whose side stream has work the caller's stream has not waited for


@register("prefetch_weights", backend="cuda", **XCOMPUTE)
def prefetch_weights(weights):
    """On integrated GPUs, L2 prefetches of ``weights`` on a side stream forked from the caller's stream (one per
    caller's stream, so lanes capturing graphs on different streams do not share it); the caller's stream does not
    wait for them until ``join_prefetch``. On discrete GPUs nothing."""
    stream = torch.cuda.current_stream()
    if not _integrated_gpu(stream.device):
        return
    key = (stream.device, stream.cuda_stream)
    side = _prefetch_streams.setdefault(key, torch.cuda.Stream(device=stream.device))
    side.wait_stream(stream)
    with torch.cuda.stream(side):
        for weight in weights:
            xc.l2_prefetch(weight, grid=PREFETCH_GRID)
    _prefetching.add(key)


@register("join_prefetch", backend="cuda", **XCOMPUTE)
def join_prefetch():
    stream = torch.cuda.current_stream()
    key = (stream.device, stream.cuda_stream)
    if key in _prefetching:
        stream.wait_stream(_prefetch_streams[key])
        _prefetching.discard(key)
