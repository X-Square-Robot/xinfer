"""Normalization operators on CUDA, through xcompute (``_xcompute``).

| operator | xcompute entry or library function |
|---|---|
| ``rms_norm``, ``add_rms_norm`` | ``rmsnorm``, ``add_rmsnorm`` |
| ``layer_norm_cast`` | ``layernorm`` |
| ``zero_centered_rms_norm`` | ``rmsnorm_zero_centered`` |
| ``add_zero_centered_rms_norm`` | ``add_rmsnorm_zero_centered`` (the sum is bitwise) |
| ``modulated_rms_norm``, ``gated_residual_modulated_rms_norm`` | ``adarmsnorm`` (with ``residual``) |
| ``identity_layer_norms_`` | ``double_adarmsnorm_gated_residual_inplace`` |
| ``qkv_norm_split`` | ``qwen35_vla_qkv_norm_replace_forward`` (no action rows; reduction order) |

Contracts against the torch references: tolerance. The empty action-row tensors of ``qkv_norm_split`` are created on
the first call for a batch size, which must therefore be eager (before a capture).
"""

from __future__ import annotations

import torch

from xinfer.ops.backends._xcompute import XCOMPUTE, library, xc, xcompute_with
from xinfer.ops.registry import register


@register("rms_norm", backend="cuda", **XCOMPUTE)
def rms_norm(x, weight, eps):
    return xc.rmsnorm(x, weight, float(eps), True)


@register("add_rms_norm", backend="cuda", **XCOMPUTE)
def add_rms_norm(x, residual, weight, eps):
    return xc.add_rmsnorm(residual, x, weight, float(eps), inference=True)


@register("layer_norm_cast", backend="cuda", **xcompute_with(functions=("layernorm",)))
def layer_norm_cast(x, weight, bias, eps):
    out = torch.empty_like(x)
    library.call("layernorm", out, x, weight, bias, float(eps))
    return out


@register("zero_centered_rms_norm", backend="cuda", **xcompute_with(functions=("rmsnorm_zero_centered",)))
def zero_centered_rms_norm(x, weight, eps):
    out = torch.empty_like(x)
    library.call("rmsnorm_zero_centered", out, x, weight, float(eps), True)
    return out


@register("add_zero_centered_rms_norm", backend="cuda", **xcompute_with(functions=("add_rmsnorm_zero_centered",)))
def add_zero_centered_rms_norm(x, residual, weight, eps):
    total, normed = torch.empty_like(x), torch.empty_like(x)
    library.call("add_rmsnorm_zero_centered", total, normed, residual, x, weight, float(eps))
    return total, normed


@register("modulated_rms_norm", backend="cuda", **xcompute_with("adarmsnorm"))
def modulated_rms_norm(x, modulation, eps, pinned_modulation=None, pinned_rows=None):
    return xc.adarmsnorm(x, modulation, float(eps), True, pinned_modulation=pinned_modulation,
                         pinned_rows=pinned_rows)


@register("gated_residual_modulated_rms_norm", backend="cuda", **xcompute_with("adarmsnorm"))
def gated_residual_modulated_rms_norm(residual, branch, branch_gate, modulation, eps, pinned_modulation=None,
                                      pinned_rows=None):
    return xc.adarmsnorm(branch, modulation, float(eps), True, residual=residual, branch_gate=branch_gate,
                         pinned_modulation=pinned_modulation, pinned_rows=pinned_rows)


@register("identity_layer_norms_", backend="cuda", **xcompute_with("double_adarmsnorm_gated_residual_inplace"))
def identity_layer_norms_(residual, branch, branch_gate, input_modulation, post_modulation, eps,
                          pinned_input_modulation=None, pinned_post_modulation=None, pinned_rows=None):
    gate = torch.empty(residual.shape if pinned_rows is not None else (*residual.shape[:-2], 1, residual.shape[-1]),
                       dtype=residual.dtype, device=residual.device)
    xc.double_adarmsnorm_gated_residual_inplace(residual, branch, gate, branch_gate, input_modulation,
                                                post_modulation, float(eps),
                                                pinned_input_modulation=pinned_input_modulation,
                                                pinned_post_modulation=pinned_post_modulation,
                                                pinned_rows=pinned_rows)
    return gate


_no_action_rows = {}  # (batch, device) -> zero action-row starts/ends [B] and offsets [B + 1] (int64)


@register("qkv_norm_split", backend="cuda", **xcompute_with(functions=("qwen35_vla_qkv_norm_replace_forward",)))
def qkv_norm_split(packed, q_weight, k_weight, num_heads, num_kv_heads, head_dim, eps):
    b, n = packed.shape[:2]
    key = (b, packed.device)
    if key not in _no_action_rows:  # every row is a text row
        _no_action_rows[key] = (torch.zeros(b, dtype=torch.int64, device=packed.device),
                                torch.zeros(b + 1, dtype=torch.int64, device=packed.device))
    rows, offsets = _no_action_rows[key]
    q = torch.empty((b, n, num_heads, head_dim), dtype=packed.dtype, device=packed.device)
    k = torch.empty((b, n, num_kv_heads, head_dim), dtype=packed.dtype, device=packed.device)
    v = torch.empty_like(k)
    gate = torch.empty((b, n, num_heads * head_dim), dtype=packed.dtype, device=packed.device)
    # The kernel takes a text and an action projection with their norm weights; with no action rows it reads the text
    # ones only.
    library.call("qwen35_vla_qkv_norm_replace_forward", q, k, v, gate, packed, packed, rows, rows, offsets, q_weight,
                 k_weight, q_weight, k_weight, int(num_heads), int(num_kv_heads), int(head_dim), float(eps))
    return q, k, v, gate
