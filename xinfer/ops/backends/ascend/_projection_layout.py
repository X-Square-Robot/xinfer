"""Split projection layouts, reusing the existing Ascend launchers unchanged."""

import weakref

import torch

from xinfer.ops.backends.ascend import finishbf16, projection_norm_epilogue, fp32_accumulator_epilogue

_rope_tables = {}


def _expanded_rope(cos, sin):
    """Expand the immutable tables once while their owners live, as CUDA caches its packed table."""
    def key(tensor):
        version = None if tensor.is_inference() else tensor._version
        return tensor.device, tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), version

    identity = key(cos), key(sin)
    pair = _rope_tables.get(identity)
    if pair is None:
        pair = torch.cat((cos, cos), dim=-1).contiguous(), torch.cat((sin, sin), dim=-1).contiguous()
        _rope_tables[identity] = pair
        for tensor in (cos, sin):
            owner = tensor if tensor._base is None else tensor._base
            weakref.finalize(owner, _rope_tables.pop, identity, None)
    return pair


def qkv_rope_split(x, weight, bias, cos, sin, num_heads, head_dim, projection):
    """Separate Q/K/V planes when one of the existing fused projections accepts the call."""
    # The existing fused routes consume the full rotate-half table and are enabled together.
    if head_dim != 64 or not fp32_accumulator_epilogue.enabled("vrope") or cos.dim() != 2:
        return None
    cos2, sin2 = _expanded_rope(cos, sin)
    if x.dim() != 2 or cos2.shape[-1] != head_dim:
        return None
    if finishbf16.enabled("vrope"):
        packed = finishbf16.vrope_bf16(x, weight, bias, cos2, sin2, num_heads, head_dim)
        if packed is not None:
            return packed.unbind(0)
    if projection is None:
        return None
    if projection_norm_epilogue.enabled("vropepack"):
        with projection.accumulator(x, weight) as acc:
            packed = projection_norm_epilogue.vision_qkv_rope_packed(acc, bias, cos2, sin2, num_heads, head_dim)
        if packed is not None:
            return packed.unbind(0)
    with projection.accumulator(x, weight) as acc:
        out = x.new_empty((x.shape[0], 3 * num_heads * head_dim), dtype=torch.bfloat16)
        fused = fp32_accumulator_epilogue.vision_qkv_rope(acc, bias, cos2, sin2, out, num_heads, head_dim)
    if fused is not None:
        return fused.view(x.shape[0], 3, num_heads, head_dim).unbind(1)
    return None


def gemm_split(x, weight, split_size, row_scale, projection):
    """The existing fused projection/row-mask/split route, or None for an unsupported call."""
    if x.dim() != 2 or row_scale is None:
        return None
    if finishbf16.enabled("zba"):
        split = finishbf16.zbasplit_bf16(x, weight, row_scale, split_size)
        if split is not None:
            return split
    if projection_norm_epilogue.enabled("zbasplit") and projection is not None:
        with projection.accumulator(x, weight) as acc:
            return projection_norm_epilogue.zbasplit(acc, row_scale, split_size)
    return None
