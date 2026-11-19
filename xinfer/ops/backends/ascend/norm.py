"""Ascend normalization operators for the two model families.

| Operator | Implementation |
|---|---|
| ``rms_norm``, ``zero_centered_rms_norm`` | RMSNorm with plain or zero-centered weights |
| ``add_rms_norm``, ``add_zero_centered_rms_norm`` | Residual add and RMSNorm |
| ``layer_norm_cast`` | LayerNorm on an owned output buffer |
| ``modulated_rms_norm``, ``gated_residual_modulated_rms_norm`` | AdaRMSNorm with optional gated residual |
| ``identity_layer_norms_`` | Fused pair of AdaRMSNorm operations with a gated residual |
"""

from functools import wraps

import torch

from xinfer.ops.backends.ascend._support import (
    PLATFORMS, PRIORITY, VERSION, canonical, entry, has_functions, partial, xc,
)
from xinfer.ops.reference.norm import (
    add_rms_norm as reference_add_rms_norm,
    add_zero_centered_rms_norm as reference_add_zero_centered_rms_norm,
    gated_residual_modulated_rms_norm as reference_gated_residual_modulated_rms_norm,
    identity_layer_norms_ as reference_identity_layer_norms_,
    layer_norm_cast as reference_layer_norm_cast,
    modulated_rms_norm as reference_modulated_rms_norm,
    rms_norm as reference_rms_norm,
    zero_centered_rms_norm as reference_zero_centered_rms_norm,
)
from xinfer.ops.registry import register

canonical_norm = canonical("add_rms_norm", reference_add_rms_norm)
norm_library = xc if canonical_norm is not None else entry("add_rmsnorm") if has_functions("add_rmsnorm") else None


def _supported(x, residual, weight, eps):
    if canonical_norm is not None:
        return True
    # The package entry serves every row count and promotes a BF16 weight; the legacy raw entry behind it
    # validated only the action scope, which left the vision and prompt norms on the reference.
    return (x.device.type == "npu" and x.shape == residual.shape and residual.device == x.device
            and weight.shape == (x.shape[-1],) and weight.device == x.device
            and weight.dtype in _WEIGHT_DTYPES
            and x.dtype == residual.dtype and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and 0 < eps <= torch.finfo(torch.float32).max)


@register("add_rms_norm", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=norm_library,
          version=VERSION)
@partial(reference_add_rms_norm, check=_supported)
def add_rms_norm(x, residual, weight, eps):
    # These residual add and RMSNorm kernels use double-buffered stores and FP32 weights.
    try:
        from xinfer.ops.backends.ascend import buffered_add_rmsnorm
        if (buffered_add_rmsnorm.enabled("v1280")
                and x.dim() == 2 and x.shape[-1] == 1280
                and x.dtype == residual.dtype == torch.bfloat16
                and residual.shape == x.shape and residual.device == x.device
                and x.is_contiguous() and residual.is_contiguous()
                and weight.dtype in (torch.float32, torch.bfloat16)
                and weight.device == x.device
                and tuple(weight.shape) == (1280,) and 0 < eps <= torch.finfo(torch.float32).max):
            return buffered_add_rmsnorm.add_rms_v1280(x, residual, weight, eps)
        if (buffered_add_rmsnorm.enabled("p2048")
                and x.dim() == 3 and x.shape[0] == 1 and x.shape[-1] == 2048
                and x.shape[1] % 4 == 0
                and x.dtype == residual.dtype == torch.bfloat16
                and residual.shape == x.shape and residual.device == x.device
                and x.is_contiguous() and residual.is_contiguous()
                and weight.dtype in (torch.float32, torch.bfloat16)
                and weight.device == x.device
                and tuple(weight.shape) == (2048,) and 0 < eps <= torch.finfo(torch.float32).max):
            return buffered_add_rmsnorm.add_rms_p2048(x, residual, weight, eps)
    except (ImportError, OSError):
        pass
    # Use the shape-specific fused add-RMSNorm kernels for action and prompt rows.
    try:
        from xinfer.ops.backends.ascend import fused_rmsnorm
        if (fused_rmsnorm.enabled("m16")
                and tuple(getattr(x, "shape", ())) == (1, 16, 1024)
                and x.dtype == residual.dtype == torch.bfloat16
                and residual.shape == x.shape and residual.device == x.device
                and x.is_contiguous() and residual.is_contiguous()
                and weight.dtype == torch.float32 and weight.device == x.device
                and tuple(weight.shape) == (1024,) and 0 < eps <= torch.finfo(torch.float32).max):
            return fused_rmsnorm.add_rms_m16(x, residual, weight, eps)
        if (fused_rmsnorm.enabled("h2048w")
                and x.dim() == 3 and x.shape[0] == 1 and x.shape[-1] == 2048
                and x.shape[1] % 4 == 0
                and x.dtype == residual.dtype == torch.bfloat16
                and residual.shape == x.shape and residual.device == x.device
                and x.is_contiguous() and residual.is_contiguous()
                and weight.dtype == torch.float32 and weight.device == x.device
                and tuple(weight.shape) == (2048,) and 0 < eps <= torch.finfo(torch.float32).max):
            return fused_rmsnorm.add_rms_h2048w(x, residual, weight, eps)
    except (ImportError, OSError):
        pass
    if canonical_norm is not None:
        return canonical_norm(x, residual, weight, eps)
    # The legacy raw entry takes (residual, x) — the reverse of the OpSpec's (x, residual). The sum is commutative
    # so the values are right, but the tuple it returns is (sum, normalized) in that entry's order.
    return xc.add_rmsnorm(residual, x, weight.contiguous(), float(eps), inference=True)


# ---------------------------------------------------------------- Wall-X 2B normalization family
# The glue entries below are classes with a stateful call, or functions that overwrite their first tensors.
_rmsnorm = getattr(xc, "Rmsnorm", None) if xc is not None else None
_add_rmsnorm = getattr(xc, "AddRmsnorm", None) if xc is not None else None
_adarmsnorm = getattr(xc, "Adarmsnorm", None) if xc is not None else None
_layernorm_inplace = getattr(xc, "layernorm_inplace", None) if xc is not None else None
_double_adarmsnorm = getattr(xc, "double_adarmsnorm_gated_residual_inplace", None) if xc is not None else None

_rms = _rmsnorm() if callable(_rmsnorm) else None
_add_rms = _add_rmsnorm() if callable(_add_rmsnorm) else None
_adarms = _adarmsnorm() if callable(_adarmsnorm) else None


def _plain_supported(x, weight, eps):
    """The entry's own layout: NPU rows with a BF16 / FP32 weight, the dtypes its kernels take."""
    return (x.device.type == "npu" and weight.device == x.device and weight.dtype in _WEIGHT_DTYPES
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32) and weight.shape == (x.shape[-1],))


@register("rms_norm", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if _rms is not None else None,
          version=VERSION)
@partial(reference_rms_norm, check=_plain_supported)
def rms_norm(x, weight, eps):
    # The 16-row action norm and the vision and prefill norms have distinct native shapes.
    try:
        from xinfer.ops.backends.ascend import fused_rmsnorm

        if (fused_rmsnorm.enabled("m16")
                and tuple(getattr(x, "shape", ())) == (1, 16, 1024)
                and x.dtype == torch.bfloat16 and x.is_contiguous()
                and weight.dtype == torch.float32 and weight.device == x.device
                and tuple(weight.shape) == (1024,)
                and 0 < eps <= torch.finfo(torch.float32).max):
            return fused_rmsnorm.rms_m16(x, weight, eps)
    except (ImportError, OSError):
        pass
    try:
        from xinfer.ops.backends.ascend import rms_plans

        plan = rms_plans.plain_rms(x, weight, eps)
        if plan is not None:
            return plan
    except (ImportError, OSError):
        pass
    # The entry's inference path with the plain (not zero-centered) weight, as the CUDA backend registers it.
    return _rms(x, weight, float(eps), True, False)


def _npu_same_device(*tensors) -> bool:
    """Every tensor sits on one NPU device. The entries take the mixed layout the 2B checkpoint stores: BF16
    activations with FP32 norm weights and modulation tables."""
    first = tensors[0]
    return first.device.type == "npu" and all(t.device == first.device for t in tensors)


# FP32 is what the 2B checkpoint stores for the norm weights; a BF16 weight is promoted by the entry.
_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)


def _layer_norm_supported(x, weight, bias, eps):
    return (_npu_same_device(x, weight, bias)
            and weight.dtype in _WEIGHT_DTYPES and bias.dtype in _WEIGHT_DTYPES)


@register("layer_norm_cast", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc,
          version=VERSION)
@partial(reference_layer_norm_cast, check=_layer_norm_supported)
def layer_norm_cast(x, weight, bias, eps):
    from xinfer.ops.backends.ascend import fused_norm_reduction

    if fused_norm_reduction.enabled("visionln"):
        result = fused_norm_reduction.layernorm_c1024(x, weight, bias, eps)
        if result is not None:
            return result
    if not callable(_layernorm_inplace):
        return reference_layer_norm_cast(x, weight, bias, eps)
    # The entry normalizes in place and returns its first tensor, while this operator's contract is functional:
    # the caller keeps using ``x`` as its residual. Clone so the entry overwrites the copy (the e2e action gate
    # isolates this exact mismatch: with the input passed through, the residual stream is polluted and the action
    # difference is 2.0; with the clone it is in the 0.03 band).
    return _layernorm_inplace(x.clone(), weight, bias, float(eps))


def _zero_centered_supported(x, weight, eps):
    return _npu_same_device(x, weight) and weight.dtype in _WEIGHT_DTYPES


@register("zero_centered_rms_norm", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if _rms is not None else None,
          version=VERSION)
@partial(reference_zero_centered_rms_norm, check=_zero_centered_supported)
def zero_centered_rms_norm(x, weight, eps):
    # ``zero_centered`` makes the entry scale by ``1 + weight``, the checkpoint's raw VoT weight.
    return _rms(x, weight, float(eps), True, True)


def _add_zero_centered_supported(x, residual, weight, eps):
    return (_npu_same_device(x, residual, weight) and x.dtype == residual.dtype
            and weight.dtype in _WEIGHT_DTYPES)


@register("add_zero_centered_rms_norm", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc,
          version=VERSION)
@partial(reference_add_zero_centered_rms_norm, check=_add_zero_centered_supported)
def add_zero_centered_rms_norm(x, residual, weight, eps):
    if _add_rms is None:
        return reference_add_zero_centered_rms_norm(x, residual, weight, eps)
    # (residual, input) order, as the entry documents, and the tuple is (sum, normalized).
    # Round the residual sum before squaring, then apply the zero-centered scale.
    try:
        from xinfer.ops.backends.ascend import projection_norm_epilogue
    except (ImportError, OSError):
        pass
    else:
        if projection_norm_epilogue.enabled("normprep"):
            fused = projection_norm_epilogue.normprep_norm(x, residual, weight, eps)
            if fused is not None:
                return fused
    return _add_rms(residual, x, weight, float(eps), True, True)


def _bind_add_zero_centered_rms_norm(*, layer_index):
    from xinfer.ops.backends.ascend import fused_norm_reduction

    if not fused_norm_reduction.enabled("rmsfull") or layer_index < fused_norm_reduction.from_layer():
        return add_zero_centered_rms_norm

    @wraps(add_zero_centered_rms_norm)
    def fused(x, residual, weight, eps):
        result = fused_norm_reduction.rmsnorm_full(x, residual, weight, eps)
        return add_zero_centered_rms_norm(x, residual, weight, eps) if result is None else result

    return fused


add_zero_centered_rms_norm.bind = _bind_add_zero_centered_rms_norm


def _modulated_supported(x, modulation, eps, pinned_modulation=None, pinned_rows=None):
    """The check receives the operator's own parameter names, so a keyword call resolves here too."""
    return (_npu_same_device(x, modulation) and modulation.shape[0] == x.shape[0]
            and modulation.shape[-1] == 3 * x.shape[-1])


_identity_operands = {}  # (device, pinned) -> (zero residual [1, 16, 1024], one branch gate)


def _modulated_identity_operands(x, pinned):
    """The 2B action shape's identity operands, cached per device and pinned state.

    The plain Ada call has no fused plan of its own: at ``[1, 16, 1024]`` with an FP32
    ``[1, 3072]`` modulation the generic statistics + pair programs measure ~424us while
    the gated-residual entry dispatches the fused action kernel (~8us) for exactly this
    shape. A zero residual and a one branch gate reduce that entry to the plain
    semantics -- ``residual_output = 0 + x * 1`` is exact in BF16 -- and the probe shows
    the outputs and gate bit for bit equal to the plain programs. With a pinned
    modulation the entry takes the branch gate per token (gate rows = the input rows),
    so the one buffer follows the pinned state.
    """
    key = (x.device, x.shape[0], bool(pinned))
    pair = _identity_operands.get(key)
    if pair is None:
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("warm up the identity operands on this stream before graph capture")
        gate = (torch.ones(x.shape[0], 16, 1024, dtype=torch.bfloat16, device=x.device) if pinned
                else torch.ones(x.shape[0], 1, 1024, dtype=torch.bfloat16, device=x.device))
        pair = (torch.zeros(x.shape[0], 16, 1024, dtype=torch.bfloat16, device=x.device), gate)
        _identity_operands[key] = pair
    return pair


@register("modulated_rms_norm", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if _adarms is not None else None,
          version=VERSION)
@partial(reference_modulated_rms_norm, check=_modulated_supported)
def modulated_rms_norm(x, modulation, eps, pinned_modulation=None, pinned_rows=None):
    if (x.dim() == 3 and modulation.dim() == 2
            and x.shape[0] in (1, 2, 4) and tuple(x.shape[1:]) == (16, 1024)
            and tuple(modulation.shape) == (x.shape[0], 3072)
            and x.dtype == torch.bfloat16 and modulation.dtype == torch.float32
            and x.is_contiguous() and modulation.is_contiguous()):
        zeros, ones = _modulated_identity_operands(
            x, pinned_modulation is not None or pinned_rows is not None)
        out, gate, _ = _adarms(x, modulation, float(eps), True,
                               pinned_modulation=pinned_modulation, pinned_rows=pinned_rows,
                               residual=zeros, branch_gate=ones)
        return out, gate
    return _adarms(x, modulation, float(eps), True,
                   pinned_modulation=pinned_modulation, pinned_rows=pinned_rows)


@register("gated_residual_modulated_rms_norm", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if _adarms is not None else None,
          version=VERSION)
@partial(reference_gated_residual_modulated_rms_norm,
         check=lambda residual, branch, branch_gate, modulation, eps, pinned_modulation=None, pinned_rows=None:
         _npu_same_device(residual, branch, branch_gate, modulation)
         and modulation.shape[0] == residual.shape[0] and modulation.shape[-1] == 3 * residual.shape[-1])
def gated_residual_modulated_rms_norm(residual, branch, branch_gate, modulation, eps,
                                      pinned_modulation=None, pinned_rows=None):
    # The same entry as modulated_rms_norm: with residual and branch_gate it normalizes the gated sum and returns
    # (output, gate, residual_output), which is this operator's contract — the entry's own gated-residual path.
    return _adarms(branch, modulation, float(eps), True,
                   pinned_modulation=pinned_modulation, pinned_rows=pinned_rows,
                   residual=residual, branch_gate=branch_gate)


@register("identity_layer_norms_", backend="ascend", priority=PRIORITY, platforms=PLATFORMS,
          requires=xc if callable(_double_adarmsnorm) else None,
          version=VERSION, deterministic=True)
@partial(reference_identity_layer_norms_,
         check=lambda residual, branch, branch_gate, input_modulation, post_modulation, eps,
         pinned_input_modulation=None, pinned_post_modulation=None, pinned_rows=None:
         _npu_same_device(residual, branch, branch_gate, input_modulation, post_modulation)
         and input_modulation.shape == post_modulation.shape
         and input_modulation.shape[-1] == 3 * residual.shape[-1])
def identity_layer_norms_(residual, branch, branch_gate, input_modulation, post_modulation, eps,
                          pinned_input_modulation=None, pinned_post_modulation=None, pinned_rows=None):
    gate = torch.empty_like(branch_gate)
    _double_adarmsnorm(residual, branch, gate, branch_gate, input_modulation, post_modulation, float(eps),
                       pinned_input_modulation=pinned_input_modulation,
                       pinned_post_modulation=pinned_post_modulation, pinned_rows=pinned_rows)
    return gate
