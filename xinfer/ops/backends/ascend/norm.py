"""Ascend normalization operators for the two model families.

Wall-OSS needs ``add_rms_norm`` and ``rms_norm`` (the vision, prompt and action streams' input norms); Wall-X 2B
also needs the zero-centered VoT norms, the ViT's layer norm and the AdaRMS family (``modulated_rms_norm``,
``gated_residual_modulated_rms_norm``, ``identity_layer_norms_``). Each registration names one OpSpec and delegates
to the xcompute entry that implements that semantics; entries reject unsupported layouts with ``ValueError`` (or
fail the ``check`` predicate) and the reference runs instead.
"""

import torch

from xinfer.ops.backends.ascend._support import VERSION, canonical, entry, has_functions, partial, xc
from xinfer.ops.backends.copykill_r2 import enabled as _copykill_enabled
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


@register("add_rms_norm", backend="ascend", requires=norm_library,
          version=VERSION)
@partial(reference_add_rms_norm, check=_supported)
def add_rms_norm(x, residual, weight, eps):
    # RMSFUSE-r4 RF1 (the [1, 16, 1024] action norm and the wide [1, T, 2048] prompt norm) rides the
    # delivered package's AOT kernels; the module is absent without that delivery and these guards
    # fall through to the entries below, so the composition is unchanged.
    try:
        from xcompute.kernel import rmsfuse_r4
        if (rmsfuse_r4.enabled("m16")
                and tuple(getattr(x, "shape", ())) == (1, 16, 1024)
                and x.dtype == residual.dtype == torch.bfloat16
                and residual.shape == x.shape and residual.device == x.device
                and x.is_contiguous() and residual.is_contiguous()
                and weight.dtype == torch.float32 and weight.device == x.device
                and tuple(weight.shape) == (1024,) and 0 < eps <= torch.finfo(torch.float32).max):
            return rmsfuse_r4.add_rms_m16(x, residual, weight, eps)
        if (rmsfuse_r4.enabled("h2048w")
                and x.dim() == 3 and x.shape[0] == 1 and x.shape[-1] == 2048
                and x.shape[1] % 4 == 0
                and x.dtype == residual.dtype == torch.bfloat16
                and residual.shape == x.shape and residual.device == x.device
                and x.is_contiguous() and residual.is_contiguous()
                and weight.dtype == torch.float32 and weight.device == x.device
                and tuple(weight.shape) == (2048,) and 0 < eps <= torch.finfo(torch.float32).max):
            return rmsfuse_r4.add_rms_h2048w(x, residual, weight, eps)
    except (ImportError, OSError):
        pass
    if canonical_norm is not None:
        return canonical_norm(x, residual, weight, eps)
    # The legacy raw entry takes (residual, x) — the reverse of the OpSpec's (x, residual). The sum is commutative
    # so the values are right, but the tuple it returns is (sum, normalized) in that entry's order.
    return xc.add_rmsnorm(residual, x, weight.contiguous(), float(eps), inference=True)


# The fused gemm+add_rms entries hang off this binding for the Wall-OSS action loop's getattr:
# RMSFUSE-r4 RF3 (gemm_add_rms_r4) and COPYKILL K2 (gemm_add_rms, behind its flag and the
# projection's f32-in capability probe). Both need the delivered python package and its AOT
# artifacts plus the CANN projection bridge; without them the attributes stay absent and the
# action loop keeps the operator composition above.
try:
    from xcompute.kernel import rmsfuse_r4
    if rmsfuse_r4.enabled("f32in"):
        from xinfer.ops.backends.ascend import gemm as _gemm_backend
        _projection = getattr(_gemm_backend, "cann_projection", None)
        if _projection is not None:
            def _gemm_add_rms_r4(x, weight, residual, norm_weight, epsilon, _proj=_projection):
                return rmsfuse_r4.gemm_add_rms_r4(x, weight, residual, norm_weight, epsilon, _proj)
            add_rms_norm.gemm_add_rms_r4 = _gemm_add_rms_r4
except (ImportError, OSError):
    pass
if _copykill_enabled("k2"):
    from xinfer.ops.backends.ascend import gemm as _gemm_backend
    _projection = getattr(_gemm_backend, "cann_projection", None)
    _fused = getattr(_projection, "gemm_add_rms_norm", None) if _projection is not None else None
    _capable = (_projection.copykill_f32in_available()
                if _projection is not None and hasattr(_projection, "copykill_f32in_available") else False)
    if callable(_fused) and _capable:
        add_rms_norm.gemm_add_rms = _fused


# ---------------------------------------------------------------- Wall-X 2B normalization family
# The xcompute entries below are classes with a stateful call, or functions that overwrite their first tensors.
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


@register("rms_norm", backend="ascend", requires=xc if _rms is not None else None,
          version=VERSION)
@partial(reference_rms_norm, check=_plain_supported)
def rms_norm(x, weight, eps):
    # Wall-OSS's vision, prompt and action input norms; the entry's inference path with the plain (not
    # zero-centered) weight, as the CUDA backend registers it.
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


@register("layer_norm_cast", backend="ascend", requires=xc if callable(_layernorm_inplace) else None,
          version=VERSION)
@partial(reference_layer_norm_cast, check=_layer_norm_supported)
def layer_norm_cast(x, weight, bias, eps):
    # The entry normalizes in place and returns its first tensor, while this operator's contract is functional:
    # the caller keeps using ``x`` as its residual. Clone so the entry overwrites the copy (the e2e action gate
    # isolates this exact mismatch: with the input passed through, the residual stream is polluted and the action
    # difference is 2.0; with the clone it is in the 0.03 band).
    return _layernorm_inplace(x.clone(), weight, bias, float(eps))


def _zero_centered_supported(x, weight, eps):
    return _npu_same_device(x, weight) and weight.dtype in _WEIGHT_DTYPES


@register("zero_centered_rms_norm", backend="ascend", requires=xc if _rms is not None else None,
          version=VERSION)
@partial(reference_zero_centered_rms_norm, check=_zero_centered_supported)
def zero_centered_rms_norm(x, weight, eps):
    # ``zero_centered`` makes the entry scale by ``1 + weight``, the checkpoint's raw VoT weight.
    return _rms(x, weight, float(eps), True, True)


def _add_zero_centered_supported(x, residual, weight, eps):
    return (_npu_same_device(x, residual, weight) and x.dtype == residual.dtype
            and weight.dtype in _WEIGHT_DTYPES)


@register("add_zero_centered_rms_norm", backend="ascend", requires=xc if _add_rms is not None else None,
          version=VERSION)
@partial(reference_add_zero_centered_rms_norm, check=_add_zero_centered_supported)
def add_zero_centered_rms_norm(x, residual, weight, eps):
    # (residual, input) order, as the entry documents, and the tuple is (sum, normalized).
    return _add_rms(residual, x, weight, float(eps), True, True)


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
    key = (x.device, bool(pinned))
    pair = _identity_operands.get(key)
    if pair is None:
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("warm up the identity operands on this stream before graph capture")
        gate = (torch.ones(1, 16, 1024, dtype=torch.bfloat16, device=x.device) if pinned
                else torch.ones(1, 1, 1024, dtype=torch.bfloat16, device=x.device))
        pair = (torch.zeros(1, 16, 1024, dtype=torch.bfloat16, device=x.device), gate)
        _identity_operands[key] = pair
    return pair


@register("modulated_rms_norm", backend="ascend", requires=xc if _adarms is not None else None,
          version=VERSION)
@partial(reference_modulated_rms_norm, check=_modulated_supported)
def modulated_rms_norm(x, modulation, eps, pinned_modulation=None, pinned_rows=None):
    if (x.dim() == 3 and modulation.dim() == 2
            and tuple(x.shape) == (1, 16, 1024) and tuple(modulation.shape) == (1, 3072)
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


@register("gated_residual_modulated_rms_norm", backend="ascend", requires=xc if _adarms is not None else None,
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


@register("identity_layer_norms_", backend="ascend", requires=xc if callable(_double_adarmsnorm) else None,
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
