"""Wall-X 2B operators on CANN's own operator library, through TorchNPU's official entries.

The reference composition casts the weight to FP32 on every call: ``x.float() @ weight.float().t()``. At the shapes
this model uses that is an 8 MB conversion per call, and the action stage alone makes three hundred of them, so the
cast dominates the arithmetic. These implementations call the official CANN operators instead, which take the BF16
weight as it is and accumulate in FP32, and keep the epilogue in FP32 exactly where the reference rounds.

Nothing here is a hand-written kernel: every entry is ``torch_npu``'s own operator, and the rest is the elementwise
arithmetic the reference already did.

These entries are **not** the default. They sit below the provider ones by priority, so a deployment runs on the
provider kernels unless it opts in with ``--op-backends`` naming this library, e.g.
``--op-backends gemm_epilogue=ascend:npu_linear``. Measured end to end they were no faster than the provider (370.4 ms
against 374.4 ms for one Wall-X 2B request, inside run-to-run noise), so the default stays where the documented
numbers were taken. The ``partial`` checks keep every layout the official operator cannot take on the reference.
Contracts against the torch references: tolerance (the CANN operator compositions).
"""

import torch
import torch.nn.functional as F

from xinfer.ops.backends.ascend._support import VERSION, partial
from xinfer.ops.reference.gemm import ACTIVATIONS
from xinfer.ops.reference.gemm import gemm_epilogue as reference_gemm_epilogue
from xinfer.ops.reference.gemm import gemm_glu_epilogue as reference_gemm_glu_epilogue
from xinfer.ops.reference.gemm import linear as reference_linear
from xinfer.ops.reference.gemm import _qk_norm_rope_kvcache
from xinfer.ops.reference.gemm import qkv_norm_rope_kvcache as reference_qkv_norm_rope_kvcache
from xinfer.ops.reference.norm import _adarms
from xinfer.ops.reference.norm import layer_norm_cast as reference_layer_norm_cast
from xinfer.ops.reference.norm import zero_centered_rms_norm as reference_zero_centered_rms_norm
from xinfer.ops.registry import register

try:
    import torch_npu
except ImportError:  # a CPU-only install imports this module and registers nothing
    torch_npu = None

_npu_linear = getattr(torch_npu, "npu_linear", None) if torch_npu is not None else None
_npu_rms_norm = getattr(torch_npu, "npu_rms_norm", None) if torch_npu is not None else None

_PRIORITY = -5  # below the reference, so nothing here is ever selected without an explicit pin


def _on_device(*tensors) -> bool:
    return all(torch.is_tensor(t) and t.device.type == "npu" for t in tensors if t is not None)


def _rows(tensor: torch.Tensor) -> torch.Tensor:
    """``[..., K]`` to the 2-D ``[M, K]`` rows the official linear takes."""
    return tensor.reshape(-1, tensor.shape[-1])


def _linear_supported(x, weight, bias=None, **_) -> bool:
    """``npu_linear`` takes BF16 rows and an ``[N, K]`` BF16 weight of the same width."""
    return (_on_device(x, weight) and x.dtype == weight.dtype == torch.bfloat16
            and x.shape[-1] == weight.shape[-1]
            and (bias is None or bias.dtype in (torch.bfloat16, torch.float32)))


@register("linear", backend="ascend", requires=_npu_linear, version=VERSION, priority=_PRIORITY)
@partial(reference_linear, check=_linear_supported)
def linear(x, weight, bias=None):
    """``x @ weight.T + bias`` through the official linear, restoring the leading dimensions."""
    leading = x.shape[:-1]
    y = _npu_linear(_rows(x), weight, None if bias is None else bias.to(x.dtype))
    return y.reshape(*leading, weight.shape[0])


@register("gemm_epilogue", backend="ascend", requires=_npu_linear, version=VERSION, priority=_PRIORITY)
@partial(reference_gemm_epilogue, check=_linear_supported)
def gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, residual_gate=None,
                  row_class=None, row_scale=None):
    """``gemm_epilogue`` with the official linear in place of the FP32 matmul, its epilogue kept in FP32."""
    y = _npu_linear(x, weight, None if bias is None else bias.to(x.dtype)).float()
    if row_scale is not None:
        y = y * row_scale.float()[:, None]
    if activation is not None:
        y = ACTIVATIONS[activation](y)
    if residual is None:
        return y.to(x.dtype)
    if residual_gate is not None:
        y = (y * residual_gate[row_class].float()).to(x.dtype).float()
    return (residual.float() + y).to(x.dtype)


@register("gemm_glu_epilogue", backend="ascend", requires=_npu_linear, version=VERSION, priority=_PRIORITY)
@partial(reference_gemm_glu_epilogue, check=_linear_supported)
def gemm_glu_epilogue(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
    """``silu(h·W_gateᵀ)·(h·W_upᵀ)`` with the official linear and the reference's AdaRMS input norm."""
    h = x if modulation is None else _adarms(x, modulation, row_class, eps)
    y = _npu_linear(h.to(weight.dtype), weight, None if bias is None else bias.to(x.dtype)).float()
    half = weight.shape[0] // 2
    return (F.silu(y[:, :half]) * y[:, half:]).to(x.dtype)


@register("zero_centered_rms_norm", backend="ascend", requires=_npu_rms_norm, version=VERSION,
          priority=_PRIORITY)
@partial(reference_zero_centered_rms_norm, check=lambda x, weight, eps, **_: _on_device(x, weight))
def zero_centered_rms_norm(x, weight, eps):
    """``npu_rms_norm`` with ``1 + w`` as its weight; the checkpoint stores the raw weight.

    The official operator returns ``(normalized, reciprocal standard deviation)``; the contract is the first.
    """
    return _npu_rms_norm(x, (weight.float() + 1).to(x.dtype), eps)[0]


@register("layer_norm_cast", backend="ascend", version=VERSION, priority=_PRIORITY)
@partial(reference_layer_norm_cast, check=lambda x, weight, bias, eps, **_: _on_device(x, weight, bias))
def layer_norm_cast(x, weight, bias, eps):
    """LayerNorm over the last dimension: one ACLNN call, no explicit FP32 prologue."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def _qkv_supported(x, weight, q_weight, k_weight, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads,
                   head_dim, eps, **_) -> bool:
    """The official linear takes the rows; the norm and rope step is the reference's own."""
    return (_on_device(x, weight, q_weight, k_weight, k_cache, v_cache)
            and x.dtype == weight.dtype == torch.bfloat16 and x.shape[-1] == weight.shape[-1])


@register("qkv_norm_rope_kvcache", backend="ascend", requires=_npu_linear, version=VERSION,
          priority=_PRIORITY)
@partial(reference_qkv_norm_rope_kvcache, check=_qkv_supported)
def qkv_norm_rope_kvcache(x, weight, q_weight, k_weight, cos, sin, k_cache, v_cache, start, num_heads,
                          num_kv_heads, head_dim, eps):
    """The action projection with the official linear in place of the FP32 matmul, same norm and rope step.

    The reference casts the whole weight to FP32 on every call, which at `[5120, 1024]` is a 21 MB conversion inside
    the action stage's denoise loop; the official linear takes the BF16 weight as stored and accumulates in FP32.
    """
    projected = _npu_linear(x, weight).float()
    return _qk_norm_rope_kvcache(projected, x.dtype, q_weight, k_weight, cos, sin, k_cache, v_cache, start,
                                 num_heads, num_kv_heads, head_dim, eps)
