"""Ascend rotary position embedding for WALL-OSS-1.0-Nano.

``rope_full`` rotates the rotary prefix of every head and passes the rest through, which is what the entry's
``Rope.full`` does with ``interleave=False`` (the half-split layout the reference uses).
Contracts against the torch references: bitwise for the d64 launchers (their only bit-level
deviations are signed zeros), tolerance for the composed forms.
"""

import torch

from xinfer.ops.backends.ascend._support import PLATFORMS, PRIORITY, VERSION, partial, xc
from xinfer.ops.reference.rope import rope_full as reference_rope_full
from xinfer.ops.registry import register

_rope = getattr(xc, "Rope", None) if xc is not None else None
_full = _rope().full if callable(_rope) else None


def _gqa_bhsd():
    import os

    return "bhsd" in os.environ.get("XINFER_GQA_GLUE", "")


@register("rope_full", backend="ascend", priority=PRIORITY, requires=xc if callable(_full) else None,
          version=VERSION, platforms=PLATFORMS)
@partial(reference_rope_full,
         check=lambda q, k, cos, sin, rotary_dim:
         q.device.type == "npu" and q.ndim == 4 and k.ndim == 4
         and all(t.dtype == torch.bfloat16 and t.device == q.device for t in (q, k))
         and cos.dtype == torch.float32 and sin.dtype == torch.float32
         and cos.device == q.device and sin.device == q.device
         and 0 < rotary_dim <= q.shape[-1] and cos.shape[-1] * 2 >= rotary_dim)
def rope_full(q, k, cos, sin, rotary_dim):
    # 2B-COVERAGE: the rotary_dim-64 partial rotation rides the d64 AOT launcher (ctypes; the
    # internal xinfer_aot_* launcher precedent) on the exact 2B prefill envelope — q [1, P, 8, 256]
    # and k [1, P, 2, 256] BF16 contiguous with compact FP32 [1, P, 32] tables and P a multiple of 4.
    # Bitwise to the torch composition (the only bit-level deviations are signed zeros, +0 vs -0);
    # the angle tables are duplicated to 64 columns with a fresh concat per call, so a graph capture
    # always bakes the duplication into the graph. Every miss keeps the paths below.
    try:
        from xinfer.ops.backends.ascend.rope_d64 import rope_half_d64 as _rope_d64
    except (ImportError, OSError):
        _rope_d64 = None
    try:
        from xinfer.ops.backends.ascend.rope_d64_bhsd import rope_d64_bhsd as _rope_bhsd
    except (ImportError, OSError):
        _rope_bhsd = None
    # The public tensors are token-major views of the native head-major storage.
    if (_rope_bhsd is not None and _gqa_bhsd()
            and int(rotary_dim) == 64
            and q.ndim == 4 and k.ndim == 4
            and q.shape[2] == 8 and q.shape[-1] == 256
            and (q.shape[0] * q.shape[1]) % 8 == 0
            and tuple(k.shape) == (q.shape[0], q.shape[1], 2, 256)
            and q.dtype == k.dtype == torch.bfloat16
            and q.device.type == "npu" and k.device == q.device
            and q.transpose(1, 2).is_contiguous() and k.transpose(1, 2).is_contiguous()
            and torch.is_tensor(cos) and cos.dtype == torch.float32
            and cos.dim() == 3 and cos.shape[-1] == 32
            and cos.shape[-2] == q.shape[0] * q.shape[1]
            and cos.is_contiguous() and cos.device == q.device
            and torch.is_tensor(sin) and sin.dtype == torch.float32
            and sin.shape == cos.shape and sin.is_contiguous()):
        rotated_q, rotated_k = _rope_bhsd(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        return rotated_q.transpose(1, 2), rotated_k.transpose(1, 2)
    if (_rope_d64 is not None
            and int(rotary_dim) == 64
            and q.ndim == 4 and k.ndim == 4
            and q.shape[0] * q.shape[1] > 0 and (q.shape[0] * q.shape[1]) % 4 == 0
            and q.shape[2] == 8 and q.shape[-1] == 256
            and tuple(k.shape) == (q.shape[0], q.shape[1], 2, 256)
            and q.dtype == k.dtype == torch.bfloat16
            and q.device.type == "npu" and k.device == q.device
            and q.is_contiguous() and k.is_contiguous()
            and torch.is_tensor(cos) and cos.dtype == torch.float32
            and cos.dim() == 3 and cos.shape[-1] == 32
            and cos.shape[-2] == q.shape[1] and cos.is_contiguous()
            and cos.device == q.device
            and torch.is_tensor(sin) and sin.dtype == torch.float32
            and sin.shape == cos.shape and sin.is_contiguous()):
        return _rope_d64(q, k, cos, sin)
    # The 2B prefill heads are 256 wide with a 64-dim rotary prefix: the affine fast
    # kernels take full rotations of 64/128-wide heads only, so this shape falls to the
    # generic scalar pair program (~264us per call). Gather the rotary prefix into a
    # contiguous head, rotate it with the affine kernel (its width equals the rotary
    # dim there) and splice the pass-through suffix back: bit for bit the generic
    # program's values at ~27us per call. Other shapes keep the direct entry.
    if (q.ndim == 4 and k.ndim == 4 and rotary_dim in (64, 128)
            and 0 < int(rotary_dim) < q.shape[-1] and q.shape[-1] == k.shape[-1]
            and q.shape[:-2] == k.shape[:-2]
            and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
            and cos.dtype == torch.float32 and sin.dtype == torch.float32
            and cos.dim() == 3 and cos.shape[0] == 1 and sin.shape == cos.shape
            and cos.is_contiguous() and sin.is_contiguous()
            and cos.shape[-1] * 2 >= rotary_dim):
        rot = int(rotary_dim)
        prefix_q = q[..., :rot].contiguous()
        prefix_k = k[..., :rot].contiguous()
        rotated_q, rotated_k = _full(prefix_q, prefix_k, cos, sin, False, rot)
        return (torch.cat((rotated_q, q[..., rot:]), dim=-1),
                torch.cat((rotated_k, k[..., rot:]), dim=-1))
    return _full(q, k, cos, sin, False, rotary_dim)
