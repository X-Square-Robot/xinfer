from xinfer.ops.backends.ascend import _provider
from xinfer.ops.backends.ascend.glue.core import CORE
import torch

from xinfer.ops.backends.ascend.glue.rmsnorm import _force_kernel_layout


def _get_gate_shape(input, modulation):
    hidden_size = input.shape[-1]
    if modulation.shape[-1] != hidden_size * 3:
        raise ValueError(
            f"Expected modulation last dimension {hidden_size * 3}, "
            f"got {modulation.shape[-1]}"
        )

    if modulation.dim() == input.dim() and modulation.shape[:-1] == input.shape[:-1]:
        return (*input.shape[:-1], hidden_size)

    if input.dim() == 3 and modulation.dim() == 2:
        if modulation.shape[0] != input.shape[0]:
            raise ValueError(
                f"Expected modulation batch {input.shape[0]}, got {modulation.shape[0]}"
            )
        return (input.shape[0], 1, hidden_size)

    if input.dim() == 3 and modulation.dim() == 3:
        if modulation.shape[0] == input.shape[0] and modulation.shape[1] == 1:
            return (input.shape[0], 1, hidden_size)

    raise ValueError(
        "Unsupported AdaRMSNorm shapes: input must be (..., H) and modulation "
        "must be matching (..., 3H), or input (B, S, H) with modulation (B, 3H)"
    )


def _rows_fast_path(input, modulation, row_class):
    """Shapes of the fused per-row kernel (adarmsnorm_class_rows)."""
    if (
        input.dim() != 2
        or modulation.dim() != 2
        or input.device.type not in ("cuda", "npu")
    ):
        return False
    m, h = input.shape
    if h < 256 or h % 256 or h > 8192 or modulation.shape[1] != 3 * h:
        return False
    if input.dtype != torch.bfloat16 or modulation.dtype != torch.bfloat16:
        return False
    if not (input.is_contiguous() and modulation.is_contiguous()):
        return False
    if row_class is None:
        return modulation.shape[0] == m
    return (
        row_class.dtype == torch.int32
        and row_class.numel() == m
        and row_class.is_contiguous()
        and row_class.device == input.device
    )


class _AdarmsnormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, modulation, epsilon):
        # The kernel requires fully-contiguous row layout (see GetRMSNormShape).
        # Sequence-parallel sharding yields non-contiguous views, so normalize
        # here — matching the guard already used by rmsnorm/gated_rmsnorm.
        input = _force_kernel_layout(input)
        modulation = _force_kernel_layout(modulation)
        output = torch.empty_like(input)
        gate = torch.empty(
            _get_gate_shape(input, modulation), dtype=input.dtype, device=input.device
        )
        CORE.adarmsnorm(output, gate, input, modulation, None, None, epsilon)
        ctx.save_for_backward(input, modulation)
        ctx.epsilon = epsilon
        ctx.gate_shape = gate.shape
        return output, gate

    @staticmethod
    def backward(ctx, grad_output, grad_gate):
        input, modulation = ctx.saved_tensors
        epsilon = ctx.epsilon

        # Defensive: forward now saves contiguous tensors, but guard anyway so a
        # non-kernel-layout saved tensor never reaches adarmsnorm_backward.
        input = _force_kernel_layout(input)
        modulation = _force_kernel_layout(modulation)

        dx = torch.empty_like(input)
        dmod_float = torch.zeros(
            modulation.shape, dtype=torch.float32, device=modulation.device
        )

        if grad_gate is None:
            grad_gate = torch.zeros(
                ctx.gate_shape, dtype=input.dtype, device=input.device
            )
        else:
            grad_gate = grad_gate.contiguous()

        CORE.adarmsnorm_backward(
            dx,
            dmod_float,
            grad_output.contiguous(),
            grad_gate,
            input,
            modulation,
            epsilon,
        )

        dmod = dmod_float.to(modulation.dtype)
        return dx, dmod, None


class Adarmsnorm:
    def __init__(self):
        self.lib = CORE

    def __call__(
        self,
        input,
        modulation,
        epsilon,
        inference=False,
        *,
        pinned_modulation=None,
        pinned_rows=None,
        residual=None,
        branch_gate=None,
        row_class=None,
        gated_input_residual=False,
        backend="auto",
    ):
        """AdaRMSNorm, the single entry point of the AdaRMSNorm semantics.

        Returns ``(output, gate)``. ``modulation`` holds (scale, shift, gate)
        blocks and the effective scale is ``1 + scale``.

        ``pinned_modulation``/``pinned_rows`` (inference only) split a shared
        ``[B, 3H]`` modulation by row: in batch row ``b`` the first
        ``pinned_rows[b]`` tokens (int32 ``[B]``, read on device) use
        ``pinned_modulation`` instead, and the gate becomes per token.

        ``residual`` with ``branch_gate`` (inference only) normalizes the gated
        residual sum ``residual + input * branch_gate`` instead of ``input``
        and returns ``(output, gate, residual_output)`` -- the
        ``gated_residual_adarmsnorm`` semantics.

        ``row_class`` (inference only, int32 ``[M]`` for ``input`` ``[M, H]``)
        picks for token ``i`` the row ``row_class[i]`` of a ``[R, 3H]``
        modulation table (values in ``[0, R)``); ``gate`` is then per token.
        ``gated_input_residual=True`` also returns
        ``input_residual = bf16(input + bf16(output * gate))`` (an identity
        mixer with a gated residual) as a third value.

        With ``row_class`` or ``gated_input_residual``, BF16 ``input [M, H]``
        (``H % 256 == 0``, ``H <= 8192``) and a contiguous ``[R, 3H]``
        (``[M, 3H]`` without ``row_class``) modulation always run the fused
        per-row kernel (``backend="auto"``; one CTA per row, fixed reduction
        order, so a row's result does not depend on M):
        ``output = bf16(x * rstd * (1 + scale) + shift)`` with
        ``rstd = rsqrt(sum(x^2) / H + eps)``, and the gate rows as given.
        ``backend="generic"`` runs the AdaRMSNorm kernel on the selected rows
        plus elementwise ops instead (the same formula, but its own reduction
        order: results may differ by one BF16 rounding).
        """
        if row_class is not None or gated_input_residual:
            if not inference or residual is not None or pinned_modulation is not None:
                raise ValueError(
                    "row_class / gated_input_residual: inference only, without residual or pinned_modulation"
                )
            if backend not in ("auto", "generic"):
                raise ValueError("adarmsnorm: backend must be 'auto' or 'generic'")
            if backend == "auto" and _rows_fast_path(input, modulation, row_class):
                out = torch.empty_like(input)
                gate = torch.empty_like(input)
                x1 = torch.empty_like(input) if gated_input_residual else None
                self.lib.adarmsnorm_class_rows(
                    out, gate, x1, input, modulation, row_class, float(epsilon)
                )
                return (out, gate) if x1 is None else (out, gate, x1)
            if row_class is not None:
                if (
                    input.dim() != 2
                    or modulation.dim() != 2
                    or row_class.dtype != torch.int32
                    or row_class.numel() != input.shape[0]
                ):
                    raise ValueError(
                        "row_class needs input [M, H], modulation [R, 3H] and int32 row_class [M]"
                    )
                modulation = modulation.index_select(0, row_class.long())
            if input.device.type == "npu":
                # Share the existing row-layout normalization with the Ada
                # call and the identity residual; no second materialization.
                input = _force_kernel_layout(input)
            output, gate = self(input, modulation, epsilon, inference=True)
            if not gated_input_residual:
                return output, gate
            return (
                output,
                gate,
                input + output * gate,
            )  # BF16 tensors: each op rounds once

        if residual is not None or branch_gate is not None:
            if residual is None or branch_gate is None:
                raise ValueError("residual and branch_gate must be given together")
            from xinfer.ops.backends.ascend.glue.gated_residual_adarmsnorm import (
                gated_residual_adarmsnorm,
            )

            return gated_residual_adarmsnorm(
                residual,
                input,
                branch_gate,
                modulation,
                epsilon,
                pinned_modulation=pinned_modulation,
                pinned_rows=pinned_rows,
            )
        pinned = pinned_modulation is not None or pinned_rows is not None
        if pinned and not inference:
            raise ValueError("pinned_modulation is supported for inference only")
        if inference:
            input = _force_kernel_layout(input)
            modulation = _force_kernel_layout(modulation)
            output = torch.empty_like(input)
            gate = torch.empty(
                input.shape if pinned else _get_gate_shape(input, modulation),
                dtype=input.dtype,
                device=input.device,
            )
            self.lib.adarmsnorm(
                output,
                gate,
                input,
                modulation,
                pinned_modulation,
                pinned_rows,
                epsilon,
            )
            return output, gate
        return _AdarmsnormFunction.apply(input, modulation, epsilon)
