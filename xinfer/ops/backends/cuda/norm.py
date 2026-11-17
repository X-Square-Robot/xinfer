"""Normalization operators on CUDA, one pass over each row.

| operator | function here |
|---|---|
| ``rms_norm``, ``add_rms_norm`` | ``rmsnorm``, ``add_rmsnorm`` |
| ``layer_norm_cast`` | ``layernorm`` |
| ``zero_centered_rms_norm``, ``add_zero_centered_rms_norm`` | the two with ``zero_centered`` (the sum is bitwise) |
| ``modulated_rms_norm``, ``gated_residual_modulated_rms_norm`` | ``adarmsnorm`` (with ``residual``) |
| ``identity_layer_norms_`` | ``double_adarmsnorm_`` |
| ``qkv_norm_split`` | itself (reduction order) |
| ``mxfp8_layer_norm_quantize`` | ``layernorm`` (``quantize``) |
| ``mxfp8_zero_centered_rms_norm_quantize`` | ``rmsnorm`` (``zero_centered``, ``quantize``) |
| ``mxfp8_add_zero_centered_rms_norm_quantize`` | ``add_rmsnorm`` (``zero_centered``, ``quantize``) |
| ``nvfp4_layer_norm_quantize`` | ``layernorm`` (``quantize="nvfp4"``) |
| ``nvfp4_add_zero_centered_rms_norm_quantize`` | ``add_rmsnorm`` (``zero_centered``, ``quantize="nvfp4"``) |

    rmsnorm(x, weight, eps, quantize=)          x * rsqrt(mean(x^2) + eps) * weight
    add_rmsnorm(residual, x, weight, eps, ...)  (residual + x, rmsnorm(residual + x))
    layernorm(x, weight, bias, eps, quantize=)  (x - mean) * rsqrt(var + eps) * weight + bias
    adarmsnorm(x, modulation, eps)              x * rsqrt(mean(x^2) + eps) * (1 + scale) + shift, and the gate
    double_adarmsnorm_(residual, branch, ...)   two gated-residual adarmsnorm steps in place

The kernels run on any CUDA GPU. Sums are taken in FP32 and every result is rounded once to the input dtype;
``zero_centered=True`` scales by ``1 + weight`` instead of ``weight``. With ``quantize=True`` the BF16 result is
written as an MXFP8 pair (``xinfer.ops.layout.mxfp8``) with one scale per 128 columns, with ``quantize="nvfp4"`` as
an NVFP4 pair (``xinfer.ops.layout.nvfp4``, amax scales), by the same kernel where it has that form: the bits of the
norm followed by a separate quantization.

Contracts against the torch references: tolerance. ``qkv_norm_split`` creates constant index tensors on the first
call for a batch size, which must therefore be eager (before a capture).
"""

from __future__ import annotations

import torch

from xinfer.ops.backends.cuda._native import NATIVE, empty_mxfp8, empty_nvfp4, lib
from xinfer.ops.registry import register


def _rows(t):
    """``t`` in the layout the kernels index: dense last dimension, one row stride over all leading dimensions."""
    stride, dense = (t.stride(-2) if t.dim() > 1 else 0), t.dim() < 1 or t.stride(-1) == 1
    for dim in range(t.dim() - 2, -1, -1):
        dense = dense and stride >= t.shape[-1] and t.stride(dim) == stride
        stride *= t.shape[dim]
    return t if dense else t.reshape(-1, t.shape[-1]).contiguous().reshape(t.shape)


def _rmsnorm_quantized(x, residual, weight, eps, zero_centered):
    """The MXFP8 pair of the BF16 norm of ``x`` (+ ``residual``), and the BF16 sum when there is a residual."""
    k = x.shape[-1]
    if x.dtype != torch.bfloat16 or k % 32:
        raise ValueError("rmsnorm: quantize needs BF16 rows of a multiple of 32 columns")
    rows = x.reshape(-1, k)
    values, scales = empty_mxfp8(rows.shape[0], k, x.device)
    other = rows if residual is None else residual.reshape(-1, k)
    total = rows if residual is None else torch.empty_like(rows)
    lib.rmsnorm_quant_mxfp8_into(rows, other, total, weight, values, scales, k, float(eps), residual is not None,
                                 True, True, bool(zero_centered), 0)
    values = values.reshape(*x.shape[:-1], k)
    return (values, scales) if residual is None else (total.reshape(x.shape), (values, scales))


def _add_rmsnorm_nvfp4(residual, x, weight, eps):
    """The NVFP4 pair of the BF16 zero-centered norm of ``residual + x``, and the BF16 sum: the wide add + RMSNorm
    kernel quantizing what it writes where it takes the rows, else ``add_rmsnorm`` and the quantization."""
    k = x.shape[-1]
    if x.dtype != torch.bfloat16 or k % 16:
        raise ValueError("add_rmsnorm: quantize='nvfp4' needs BF16 rows of a multiple of 16 columns")
    rows, residual = x.reshape(-1, k), residual.reshape(-1, k)
    weight = weight.contiguous()
    data, scales = empty_nvfp4(rows.shape[0], k, x.device)
    if (k % 64 == 0 and k <= 4096 * 8 and weight.dim() == 1 and weight.dtype in (torch.float32, torch.bfloat16)
            and rows.is_contiguous() and residual.is_contiguous()
            and all(t.data_ptr() % 16 == 0 for t in (rows, residual, weight))):
        total = torch.empty_like(rows)
        lib.add_rmsnorm_zero_centered_nvfp4(total, None, data, scales, residual, rows, weight, float(eps))
    else:
        total, normed = add_rmsnorm(residual, rows, weight, eps, zero_centered=True)
        lib.quantize_nvfp4_into(data, scales, normed.contiguous(), 1.0, False)
    return total.reshape(x.shape), (data, scales)


def rmsnorm(x, weight, eps, *, zero_centered=False, quantize=False):
    """``x * rsqrt(mean(x^2, -1) + eps) * weight``: x ``[..., K]``, weight ``[K]``; a new tensor like ``x``, or
    its MXFP8 pair with ``quantize=True``."""
    if quantize:
        return _rmsnorm_quantized(x, None, weight, eps, zero_centered)
    x = _rows(x)
    out = torch.empty_like(x)
    (lib.rmsnorm_zero_centered if zero_centered else lib.rmsnorm)(out, x, weight, float(eps), True)
    return out


def add_rmsnorm(residual, x, weight, eps, *, zero_centered=False, quantize=False):
    """``(residual + x, rmsnorm(residual + x))`` in one pass over the two inputs; the sum is the BF16 sum. With
    ``quantize=True`` the norm is returned as its MXFP8 pair, with ``quantize="nvfp4"`` (``zero_centered``) as its
    NVFP4 pair."""
    if residual.shape != x.shape:
        raise ValueError(f"add_rmsnorm: residual {tuple(residual.shape)} does not match x {tuple(x.shape)}")
    if quantize == "nvfp4":
        if not zero_centered:
            raise ValueError("add_rmsnorm: quantize='nvfp4' is zero-centered")
        return _add_rmsnorm_nvfp4(residual, x, weight, eps)
    if quantize:
        return _rmsnorm_quantized(x, residual, weight, eps, zero_centered)
    residual, x = _rows(residual), _rows(x)
    total, normed = torch.empty_like(x), torch.empty_like(x)
    (lib.add_rmsnorm_zero_centered if zero_centered else lib.add_rmsnorm)(total, normed, residual, x, weight,
                                                                          float(eps))
    return total, normed


def layernorm(x, weight, bias, eps, *, quantize=False):
    """``torch.nn.functional.layer_norm`` over the last dimension: x ``[..., K]``, weight and bias ``[K]``; a new
    tensor like ``x``, or its MXFP8 pair with ``quantize=True``, its NVFP4 pair with ``quantize="nvfp4"``."""
    if not quantize:
        out = torch.empty_like(x)
        lib.layernorm(out, x, weight, bias, float(eps))
        return out
    k = x.shape[-1]
    if quantize == "nvfp4":
        if x.dtype != torch.bfloat16 or k % 16:
            raise ValueError("layernorm: quantize='nvfp4' needs BF16 rows of a multiple of 16 columns")
        rows = x.reshape(-1, k)
        data, scales = empty_nvfp4(rows.shape[0], k, x.device)
        if (k % 128 == 0 and k <= 2048 and rows.is_contiguous() and weight.dtype == bias.dtype == torch.bfloat16
                and all(t.data_ptr() % 16 == 0 for t in (rows, weight, bias))):  # one kernel for both
            lib.layernorm_quant_nvfp4(data, scales, None, rows, weight, bias, float(eps))
        else:
            normed = torch.empty_like(rows)
            lib.layernorm(normed, rows, weight, bias, float(eps))
            lib.quantize_nvfp4_into(data, scales, normed, 1.0, False)
        return data, scales
    if x.dtype != torch.bfloat16 or k % 32:
        raise ValueError("layernorm: quantize needs BF16 rows of a multiple of 32 columns")
    rows = x.reshape(-1, k)
    values, scales = empty_mxfp8(rows.shape[0], k, x.device)
    if lib.layernorm_quant_mxfp8_fused_selected(rows, weight, bias):  # the widths one kernel does both for
        lib.layernorm_quant_mxfp8(values, scales, rows, weight, bias, float(eps))
    else:
        normed = torch.empty_like(rows)
        lib.layernorm(normed, rows, weight, bias, float(eps))
        lib.quantize_mxfp8_into(normed, values, scales, True)
    return values.reshape(*x.shape[:-1], k), scales


def adarmsnorm(x, modulation, eps, *, residual=None, branch_gate=None, pinned_modulation=None, pinned_rows=None):
    """RMSNorm with a per-request scale and shift: ``x * rsqrt(mean(x^2) + eps) * (1 + scale) + shift``.

    x ``[B, M, K]``; ``modulation`` ``[B, 3 K]`` holds ``[scale | shift | gate]``. Returns ``(normed, gate)`` with
    the gate ``[B, 1, K]`` in ``x.dtype``.

    With ``residual`` and ``branch_gate`` (``[B, 1 | M, K]``), ``x`` is a branch output: the kernel forms
    ``s = residual + x * branch_gate`` (rounded to BF16), normalizes ``s`` and returns ``(normed, gate, s)``.

    With ``pinned_modulation`` (``[B, 3 K]``) and ``pinned_rows`` (int32 ``[B]``, on the device), the first
    ``pinned_rows[b]`` rows of request ``b`` take the pinned modulation, and the gate is per row, ``[B, M, K]``.
    """
    if (residual is None) != (branch_gate is None):
        raise ValueError("adarmsnorm: residual and branch_gate go together")
    if (pinned_modulation is None) != (pinned_rows is None):
        raise ValueError("adarmsnorm: pinned_modulation and pinned_rows go together")
    x, modulation = _rows(x), _rows(modulation)
    normed = torch.empty_like(x)
    gate = torch.empty(x.shape if pinned_rows is not None else (*x.shape[:-2], 1, x.shape[-1]), dtype=x.dtype,
                       device=x.device)
    if residual is None:
        lib.adarmsnorm(normed, gate, x, modulation, pinned_modulation, pinned_rows, float(eps))
        return normed, gate
    total = torch.empty_like(x)
    lib.gated_residual_adarmsnorm(normed, gate, total, residual, x, branch_gate, modulation, pinned_modulation,
                                  pinned_rows, float(eps))
    return normed, gate, total


def double_adarmsnorm_(residual, branch, branch_gate, input_modulation, post_modulation, eps, *,
                       pinned_input_modulation=None, pinned_post_modulation=None, pinned_rows=None):
    """Two chained ``adarmsnorm(..., residual=, branch_gate=)`` steps in place, for a layer whose mixer is the
    identity::

        normed, gate, s = adarmsnorm(branch, input_modulation, eps, residual=residual, branch_gate=branch_gate)
        normed, gate, s = adarmsnorm(normed, post_modulation, eps, residual=s, branch_gate=gate)

    ``residual`` becomes ``s`` and ``branch`` becomes ``normed``; returns ``gate``. BF16 ``[B, M, 1024]`` rows run
    in one kernel, other widths as the two steps.
    """
    pinned = dict(pinned_rows=pinned_rows)
    if residual.shape[-1] != 1024 or residual.dtype != torch.bfloat16:
        normed, gate, total = adarmsnorm(branch, input_modulation, eps, residual=residual, branch_gate=branch_gate,
                                         pinned_modulation=pinned_input_modulation, **pinned)
        normed, gate, total = adarmsnorm(normed, post_modulation, eps, residual=total, branch_gate=gate,
                                         pinned_modulation=pinned_post_modulation, **pinned)
        residual.copy_(total)
        branch.copy_(normed)
        return gate
    gate = torch.empty(residual.shape if pinned_rows is not None else (*residual.shape[:-2], 1, residual.shape[-1]),
                       dtype=residual.dtype, device=residual.device)
    lib.gated_residual_double_adarmsnorm_inplace(residual, branch, gate, branch_gate, input_modulation,
                                                 post_modulation, pinned_input_modulation, pinned_post_modulation,
                                                 pinned_rows, float(eps))
    return gate


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
@register("rms_norm", backend="cuda", **NATIVE)
def rms_norm(x, weight, eps):
    return rmsnorm(x, weight, eps)


@register("add_rms_norm", backend="cuda", **NATIVE)
def add_rms_norm(x, residual, weight, eps):
    return add_rmsnorm(residual, x, weight, eps)


@register("layer_norm_cast", backend="cuda", **NATIVE)
def layer_norm_cast(x, weight, bias, eps):
    return layernorm(x, weight, bias, eps)


@register("zero_centered_rms_norm", backend="cuda", **NATIVE)
def zero_centered_rms_norm(x, weight, eps):
    return rmsnorm(x, weight, eps, zero_centered=True)


@register("add_zero_centered_rms_norm", backend="cuda", **NATIVE)
def add_zero_centered_rms_norm(x, residual, weight, eps):
    return add_rmsnorm(residual, x, weight, eps, zero_centered=True)


@register("modulated_rms_norm", backend="cuda", **NATIVE)
def modulated_rms_norm(x, modulation, eps, pinned_modulation=None, pinned_rows=None):
    return adarmsnorm(x, modulation, eps, pinned_modulation=pinned_modulation, pinned_rows=pinned_rows)


@register("gated_residual_modulated_rms_norm", backend="cuda", **NATIVE)
def gated_residual_modulated_rms_norm(residual, branch, branch_gate, modulation, eps, pinned_modulation=None,
                                      pinned_rows=None):
    return adarmsnorm(branch, modulation, eps, residual=residual, branch_gate=branch_gate,
                         pinned_modulation=pinned_modulation, pinned_rows=pinned_rows)


@register("identity_layer_norms_", backend="cuda", **NATIVE)
def identity_layer_norms_(residual, branch, branch_gate, input_modulation, post_modulation, eps,
                          pinned_input_modulation=None, pinned_post_modulation=None, pinned_rows=None):
    return double_adarmsnorm_(residual, branch, branch_gate, input_modulation, post_modulation, eps,
                                 pinned_input_modulation=pinned_input_modulation,
                                 pinned_post_modulation=pinned_post_modulation, pinned_rows=pinned_rows)


@register("mxfp8_layer_norm_quantize", backend="cuda", **NATIVE)
def mxfp8_layer_norm_quantize(x, weight, bias, eps):
    return layernorm(x.reshape(-1, x.shape[-1]), weight, bias, eps, quantize=True)


@register("mxfp8_zero_centered_rms_norm_quantize", backend="cuda", **NATIVE)
def mxfp8_zero_centered_rms_norm_quantize(x, weight, eps):
    return rmsnorm(x.reshape(-1, x.shape[-1]), weight, eps, zero_centered=True, quantize=True)


@register("mxfp8_add_zero_centered_rms_norm_quantize", backend="cuda", **NATIVE)
def mxfp8_add_zero_centered_rms_norm_quantize(x, residual, weight, eps):
    k = x.shape[-1]
    summed, quantized = add_rmsnorm(residual.reshape(-1, k), x.reshape(-1, k), weight, eps, zero_centered=True,
                                       quantize=True)
    return summed.view(x.shape), quantized


@register("nvfp4_layer_norm_quantize", backend="cuda", **NATIVE)
def nvfp4_layer_norm_quantize(x, weight, bias, eps):
    return layernorm(x.reshape(-1, x.shape[-1]), weight, bias, eps, quantize="nvfp4")


@register("nvfp4_add_zero_centered_rms_norm_quantize", backend="cuda", **NATIVE)
def nvfp4_add_zero_centered_rms_norm_quantize(x, residual, weight, eps):
    return add_rmsnorm(residual, x, weight, eps, zero_centered=True, quantize="nvfp4")


_no_rows = {}  # (batch, device) -> (int64 zeros [B], int64 zeros [B + 1])


@register("qkv_norm_split", backend="cuda", **NATIVE)
def qkv_norm_split(packed, q_weight, k_weight, num_heads, num_kv_heads, head_dim, eps):
    """Split the projection of a gated attention, ``[B, M, 2 H * D + 2 H_kv * D]`` holding each query head next
    to its output gate, ``[q_0 | gate_0 | q_1 | gate_1 | ...]``, then K, then V, with a zero-centered RMSNorm on
    every Q and K head: ``t * rsqrt(mean(t^2) + eps) * (1 + w)``, ``w`` BF16 ``[D]``. V and the gate are copied.

    Returns ``(q [B, M, H, D], k [B, M, H_kv, D], v [B, M, H_kv, D], gate [B, M, H * D])``. The constant index
    tensors of a batch size are created on its first call, which must not be inside a CUDA graph capture.
    """
    b, m = packed.shape[:2]
    key = (b, packed.device)
    if key not in _no_rows:  # the kernel also takes a second projection for a row range per request: none here
        _no_rows[key] = (torch.zeros(b, dtype=torch.int64, device=packed.device),
                         torch.zeros(b + 1, dtype=torch.int64, device=packed.device))
    rows, offsets = _no_rows[key]
    q = torch.empty((b, m, num_heads, head_dim), dtype=packed.dtype, device=packed.device)
    k = torch.empty((b, m, num_kv_heads, head_dim), dtype=packed.dtype, device=packed.device)
    v = torch.empty_like(k)
    gate = torch.empty((b, m, num_heads * head_dim), dtype=packed.dtype, device=packed.device)
    lib.qkv_norm_split(q, k, v, gate, packed, packed, rows, rows, offsets, q_weight, k_weight, q_weight, k_weight,
                       int(num_heads), int(num_kv_heads), int(head_dim), float(eps))
    return q, k, v, gate
