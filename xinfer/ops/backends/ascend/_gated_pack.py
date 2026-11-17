"""Launch the unchanged v22 R6B gated attention pack from the unified library."""
import torch

from xinfer.ops.backends.ascend import _native

_library = _native.core_library()
_PAIR = (_native.aot_entry(_library, "gated_attn_bnsd_pack_pow2", 3, 7)
         if _library is not None else None)


def _applies(out, gate):
    if _PAIR is None or out.ndim != 4 or out.device.type != "npu" or out.dtype != torch.bfloat16:
        return False
    batch, heads, rows, dim = out.shape
    return (1 <= batch <= 4 and (heads, rows, dim) == (8, 16, 256) and out.is_contiguous()
            and gate.shape == (batch, rows, heads * dim) and gate.dtype == out.dtype and gate.device == out.device
            and gate.stride(-1) == 1 and gate.stride(1) >= heads * dim
            and (batch == 1 or gate.stride(0) == rows * gate.stride(1)))


def gated_pack(out, gate):
    """Packed gated rows, or None before launch when the existing program does not apply."""
    if not _applies(out, gate):
        return None
    batch, heads, rows, dim = out.shape
    packed = torch.empty((batch, rows, heads * dim), dtype=out.dtype, device=out.device)
    pitch = gate.stride(1)
    scalars = (out.numel(), (batch * rows - 1) * pitch + heads * dim, heads * dim, rows, dim, pitch, 0)
    try:
        if not _native.launch_kernel("gated_attn_bnsd_pack_pow2", (packed, out, gate), scalars):
            _native.launch(_PAIR, (packed, out, gate), scalars)
    except ValueError as exc:
        raise RuntimeError(f"gated attention pack launch failed: {exc}") from exc
    return packed
