"""Ascend operators served by TorchNPU's official CANN operators.

``torch_npu.npu_linear`` is ``x @ weight.T + bias`` bit for bit, and ``torch_npu.npu_rms_norm(x, weight, eps)`` is
``x · rsqrt(mean(x²) + eps) · weight``, so the zero-centered form this checkpoint stores is that operator with
``1 + w`` as the weight. Nothing here is a hand-written kernel: the split is views, the arithmetic is the official
operator, and the only extra step is the elementwise ``1 + w`` on a ``[D]`` vector.

These entries are **not** the default: they sit below the provider ones by priority (the unified GEMM entry now
serves ``linear``, the norm-replace entry ``qkv_norm_split``), so a deployment runs on the provider entries unless
it opts in with ``--op-backends`` naming this library, e.g. ``--op-backends linear=ascend:torch_npu``.
``npu_linear`` takes a 2-D input, so ``linear`` flattens the leading dimensions and restores them.
Contracts against the torch references: tolerance (the vendor operators' own precision).
"""

import torch

from xinfer.ops.backends.ascend._support import VERSION, partial
from xinfer.ops.reference.gemm import linear as reference_linear
from xinfer.ops.reference.gemm import qkv_norm_split as reference_qkv_norm_split
from xinfer.ops.registry import register

try:  # the official operators live on the TorchNPU extension module, not on torch.npu
    import torch_npu
except ImportError:  # a CPU-only install still imports this module, it just registers nothing
    torch_npu = None

_npu_linear = getattr(torch_npu, "npu_linear", None) if torch_npu is not None else None
_npu_rms_norm = getattr(torch_npu, "npu_rms_norm", None) if torch_npu is not None else None

_PRIORITY = -5  # below the provider entries: nothing here is ever selected without an explicit pin


def _linear_supported(x, weight, bias=None) -> bool:
    """``npu_linear`` takes dense 2-D ``[..., K]`` rows with an ``[N, K]`` weight of the same dtype."""
    return (x.device.type == "npu" and x.ndim >= 2 and x.dtype == weight.dtype == torch.bfloat16
            and weight.device == x.device and weight.ndim == 2 and weight.shape[-1] == x.shape[-1]
            and (bias is None or (bias.device == x.device and bias.dtype == x.dtype
                                  and bias.shape == (weight.shape[0],))))


@register("linear", backend="ascend", requires=torch_npu if callable(_npu_linear) else None,
          version=VERSION, priority=_PRIORITY)
@partial(reference_linear, check=_linear_supported)
def linear(x, weight, bias=None):
    lead = x.shape[:-1]
    out = _npu_linear(x.reshape(-1, x.shape[-1]), weight, bias)
    return out.reshape(*lead, weight.shape[0])


def _qkv_norm_split_supported(packed, q_weight, k_weight, num_heads, num_kv_heads, head_dim, eps) -> bool:
    """The packed row layout and the raw zero-centered weights the checkpoint stores."""
    rows = 2 * num_heads * head_dim + 2 * num_kv_heads * head_dim
    return (packed.device.type == "npu" and packed.ndim == 3 and packed.dtype == torch.bfloat16
            and packed.shape[-1] == rows
            and q_weight.shape == k_weight.shape == (head_dim,)
            and q_weight.device == k_weight.device == packed.device
            and q_weight.dtype == k_weight.dtype and 0 < eps <= torch.finfo(torch.float32).max)


@register("qkv_norm_split", backend="ascend", requires=torch_npu if callable(_npu_rms_norm) else None,
          version=VERSION, priority=_PRIORITY)
@partial(reference_qkv_norm_split, check=_qkv_norm_split_supported)
def qkv_norm_split(packed, q_weight, k_weight, num_heads, num_kv_heads, head_dim, eps):
    # The zero-centered form is CANN's RMSNorm with ``1 + w`` as the weight; q and k are normalized, v and the
    # gate are copied, which is the reference's own split.
    b, n = packed.shape[:2]
    q_end, k_end = 2 * num_heads * head_dim, (2 * num_heads + num_kv_heads) * head_dim
    query = packed[..., :q_end].reshape(b, n, num_heads, 2, head_dim)
    k = packed[..., q_end:k_end].reshape(b, n, num_kv_heads, head_dim)
    v = packed[..., k_end:k_end + num_kv_heads * head_dim].reshape(b, n, num_kv_heads, head_dim)
    return (_npu_rms_norm(query[..., 0, :], q_weight.float() + 1, float(eps))[0].contiguous(),
            _npu_rms_norm(k, k_weight.float() + 1, float(eps))[0].contiguous(),
            v.contiguous(),
            query[..., 1, :].reshape(b, n, num_heads * head_dim))
