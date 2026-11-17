"""Two AdaRMSNorm state transitions in one call.

The CUDA fused entry points reproduce -- bit for bit -- what two separate generic
``gated_residual_adarmsnorm`` launches would produce; they exist only to keep
the intermediate tensors in registers.  ``initial`` starts a sequence (no
residual to add to), ``gated_residual_inplace`` continues it.

The Ascend continuation specializes unpinned BF16 ``[1, 16, 1024]`` with FP32
modulation. It keeps the BF16 materialization boundaries and uses a per-row
warp reduction. Pinned rows and other layouts use the composed transitions.

The fused kernels cover BF16 at hidden 1024; any
other hidden size or dtype runs the two generic launches the entry points are
defined as, so both accept every shape the generic ``gated_residual_adarmsnorm``
does (with temporaries for the intermediate state).

Fused shapes: activations are contiguous BF16 ``[B, S, 1024]``, gates are ``[B, 1024]``
and both modulations are same-dtype FP32 or BF16 ``[B, 3 * 1024]``.
BF16 modulation is widened in registers; no temporary FP32 tensor is needed.

Pinned split (RTC denoise): ``pinned_input_modulation``,
``pinned_post_modulation`` and ``pinned_rows`` (int32 ``[B]``, read on device)
are given together; the first ``pinned_rows[b]`` tokens of batch row ``b`` use
the pinned modulations, and gates become per token, ``[B, S, 1024]``.  Each
row still equals the generic pair with the modulation it selected.
"""

import torch

from xinfer.ops.backends.ascend.core import CORE



def _fused_shape(tensor):
    return tensor.size(-1) == 1024 and tensor.dtype == torch.bfloat16


def _gate_like(tensor, pinned):
    shape = tensor.shape if pinned else (*tensor.shape[:-2], 1, tensor.shape[-1])
    return torch.empty(shape, dtype=tensor.dtype, device=tensor.device)


def double_adarmsnorm_initial_into(
    residual,
    normalized,
    next_gate,
    input,
    input_modulation,
    post_modulation,
    epsilon,
    *,
    pinned_input_modulation=None,
    pinned_post_modulation=None,
    pinned_rows=None,
):
    """First transition into caller-owned ``residual``/``normalized``/``next_gate``.

    Performs no allocation, so the buffers can be reused across calls and CUDA
    graph captures.
    """
    if not _fused_shape(input):
        pinned = pinned_rows is not None
        first_out, first_gate = torch.empty_like(input), _gate_like(input, pinned)
        CORE.adarmsnorm(
            first_out,
            first_gate,
            input,
            input_modulation,
            pinned_input_modulation,
            pinned_rows,
            float(epsilon),
        )
        CORE.gated_residual_adarmsnorm(
            normalized,
            next_gate,
            residual,
            input,
            first_out,
            first_gate,
            post_modulation,
            pinned_post_modulation,
            pinned_rows,
            float(epsilon),
        )
        return
    CORE.initial_identity_double_adarmsnorm_bf16(
        residual,
        normalized,
        next_gate,
        input,
        input_modulation,
        post_modulation,
        pinned_input_modulation,
        pinned_post_modulation,
        pinned_rows,
        float(epsilon),
    )


def double_adarmsnorm(
    input,
    input_modulation,
    post_modulation,
    epsilon,
    *,
    pinned_input_modulation=None,
    pinned_post_modulation=None,
    pinned_rows=None,
):
    """Allocating form of :func:`double_adarmsnorm_initial_into`.

    Returns ``(residual, normalized, next_gate)``.
    """
    residual = torch.empty_like(input)
    normalized = torch.empty_like(input)
    next_gate = torch.empty(
        (
            input.shape
            if pinned_rows is not None
            else (*input.shape[:-2], 1, input.shape[-1])
        ),
        dtype=input.dtype,
        device=input.device,
    )
    double_adarmsnorm_initial_into(
        residual,
        normalized,
        next_gate,
        input,
        input_modulation,
        post_modulation,
        epsilon,
        pinned_input_modulation=pinned_input_modulation,
        pinned_post_modulation=pinned_post_modulation,
        pinned_rows=pinned_rows,
    )
    return residual, normalized, next_gate


def double_adarmsnorm_gated_residual_inplace(
    residual,
    branch,
    next_gate,
    previous_gate,
    input_modulation,
    post_modulation,
    epsilon,
    *,
    pinned_input_modulation=None,
    pinned_post_modulation=None,
    pinned_rows=None,
):
    """Second and later transitions.  Overwrites ``residual`` (with the new
    sum), ``branch`` (with the normalized activation) and ``next_gate``."""
    if not _fused_shape(residual):
        pinned = pinned_rows is not None
        first_out, first_sum = torch.empty_like(residual), torch.empty_like(residual)
        first_gate = _gate_like(residual, pinned)
        CORE.gated_residual_adarmsnorm(
            first_out,
            first_gate,
            first_sum,
            residual,
            branch,
            previous_gate,
            input_modulation,
            pinned_input_modulation,
            pinned_rows,
            float(epsilon),
        )
        CORE.gated_residual_adarmsnorm(
            branch,
            next_gate,
            residual,
            first_sum,
            first_out,
            first_gate,
            post_modulation,
            pinned_post_modulation,
            pinned_rows,
            float(epsilon),
        )
        return
    CORE.gated_residual_identity_double_adarmsnorm_bf16_inplace(
        residual,
        branch,
        next_gate,
        previous_gate,
        input_modulation,
        post_modulation,
        pinned_input_modulation,
        pinned_post_modulation,
        pinned_rows,
        float(epsilon),
    )
