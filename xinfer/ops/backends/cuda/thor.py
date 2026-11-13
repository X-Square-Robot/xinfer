"""SM110 BF16 operators used by Wall-OSS vision and prefill."""

from functools import lru_cache
from weakref import WeakKeyDictionary

import torch

from xinfer.ops.backends._xcompute import xc, xcompute_with
from xinfer.ops.backends.cuda.gemm import gemm_epilogue, gemm_glu_epilogue
from xinfer.ops.registry import register


def sm110_with(*entries, functions=()):
    """Limit registration to a build targeting SM110 and exporting the required entries."""
    options = xcompute_with(*entries, functions=functions)
    if "sm110" not in options["platforms"]:
        options["requires"] = None
    return dict(options, platforms=("sm110",))


_packings = WeakKeyDictionary()


def _packed_pair(left, right, kind, make):
    """Share the packing while both source storages live. Distinguish slices and layouts of the same storage."""
    right_storages = _packings.setdefault(left.untyped_storage(), WeakKeyDictionary())
    entries = right_storages.setdefault(right.untyped_storage(), {})
    key = (kind, left.storage_offset(), left.shape, left.stride(), left.dtype,
           right.storage_offset(), right.shape, right.stride(), right.dtype)
    if key not in entries:
        entries[key] = make()
    return entries[key]


@register("qkv_rope_for_attention", backend="cuda", deterministic=False,
          **sm110_with(functions=("bf16_qkv_rope_sm110",)))
def qkv_rope_for_attention(x, weight, bias, cos, sin, num_heads, head_dim, *, table_cache=None):
    """Allow paired Q/K channels only when the selected kernel writes that order. V stays in checkpoint order."""
    x = x.contiguous()
    rotary_dim = 2 * cos.shape[-1]
    if rotary_dim != head_dim or head_dim % 4:
        return xc.qkv_proj_rope(x, weight, bias, cos=cos, sin=sin, rotary_dim=rotary_dim, num_heads=num_heads,
                                num_kv_heads=num_heads, head_dim=head_dim, backend="auto")
    cached = table_cache.get("rope") if table_cache is not None else None
    if cached is None or cached[0] is not cos or cached[1] is not sin:
        cached = (cos, sin, xc.pack_rope_table(cos, sin))
        if table_cache is not None:
            table_cache["rope"] = cached
    table = cached[2]

    def pack():
        packed = xc.prepack_weight(weight, op="qkv_proj_rope", bias=bias, num_heads=num_heads, head_dim=head_dim)
        return packed if packed.layout == "nk_rope_pairs" else None

    packed = _packed_pair(weight, bias, ("qkv", num_heads, head_dim), pack)
    options = dict(rope_table=table, out_layout="rope_pairs")
    if packed is not None and xc.selected_backend("qkv_proj_rope", x, packed, **options) != "generic":
        return xc.qkv_proj_rope(x, packed, **options, backend="auto")
    return xc.qkv_proj_rope(x, weight, bias, rope_table=table, num_heads=num_heads, num_kv_heads=num_heads,
                            head_dim=head_dim, backend="auto")


@register("gemm_add_rms_norm", backend="cuda", deterministic=False, **sm110_with())
def gemm_add_rms_norm(x, weight, residual, gamma, eps, *, bias=None):
    """Add the residual in the GEMM epilogue, then normalize the rounded sum."""
    total = gemm_epilogue(x, weight, bias, residual=residual.view(x.shape[0], -1)).view_as(residual)
    return total, xc.rmsnorm(total, gamma, float(eps), True)


@register("mlp_add_rms_norm", backend="cuda", deterministic=False,
          **sm110_with("fused_mlp", functions=("mcast_fused_mlp_sm110", "mcast_fused_mlp_sm110_flags_bytes",
                                               "mcast_fused_mlp_sm110_hbuf_bytes")))
def mlp_add_rms_norm(x, gate_up, down, residual, gamma, eps):
    """Use the fused gated MLP when the kernel covers this shape. The residual may be updated in place."""
    def pack():
        # The fused kernel consumes whole 1024-channel intermediate tiles. Both halves use zero padding.
        if down.shape[0] % 512 or down.shape[1] % 64:
            return None
        gate = xc.prepack_weight(gate_up, op="fused_mlp", pad_n=1024)
        if gate.layout != "nk_glu_blocked128_k64":
            return None
        return gate, xc.prepack_weight(down, layout="nk_k64blk", pad_k=1024)

    x = x.contiguous()
    weights = _packed_pair(gate_up, down, "mlp", pack)
    if weights is not None:
        gate, projection = weights
        rows = residual.view(x.shape[0], -1)
        options = dict(activation="silu", residual=rows, out=rows)
        if xc.selected_backend("fused_mlp", x, gate, w_down=projection, **options) != "generic":
            xc.fused_mlp(x, gate, projection, backend="auto", **options)
            return residual, xc.rmsnorm(residual, gamma, float(eps), True)
    return gemm_add_rms_norm(gemm_glu_epilogue(x, gate_up), down, residual, gamma, eps)


@lru_cache(maxsize=None)
def _prefetch_stream(caller):
    """Use a separate side stream for each capture or execution lane."""
    return torch.cuda.Stream(device=caller.device)


@lru_cache(maxsize=None)
def _integrated_gpu(device):
    return getattr(torch.cuda.get_device_properties(device), "is_integrated", False)


_PREFETCH = sm110_with("l2_prefetch", functions=("l2_prefetch",))


@register("prefetch_weights", backend="cuda", **_PREFETCH)
def prefetch_weights(weights):
    """Schedule L2 reads on an integrated GPU. Return the side stream to rejoin before the stage ends."""
    if not weights or not _integrated_gpu(weights[0].device):
        return None
    caller = torch.cuda.current_stream(weights[0].device)
    side = _prefetch_stream(caller)
    side.wait_stream(caller)
    try:
        with torch.cuda.stream(side):
            for weight in weights:
                xc.l2_prefetch(weight, grid=2)
    except BaseException:
        caller.wait_stream(side)
        raise
    return side


@register("join_prefetch", backend="cuda", **_PREFETCH)
def join_prefetch(side):
    """Join the side stream back to the current stream on its device, including during graph capture."""
    if side is not None:
        torch.cuda.current_stream(side.device).wait_stream(side)
