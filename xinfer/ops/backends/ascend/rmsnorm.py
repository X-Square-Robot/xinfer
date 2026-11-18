import torch
from xinfer.ops.backends.ascend.core import CORE




def _get_sm_count():
    """Cache SM count for the current GPU."""
    if not hasattr(_get_sm_count, "_count"):
        _get_sm_count._count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    return _get_sm_count._count


def _get_plain_rmsnorm_backward_blocks(num_tokens: int, device=None) -> int:
    """Persistent-grid size for the plain rmsnorm backward (the C++ side sizes
    its grid from partial_dw.shape[0], so this knob only lives here).

    ~3 resident blocks/SM is the empirical sweet spot for the wide backward
    kernel (register-bound residency); the grid is then shrunk so the row
    chunks (the kernel processes rows in pairs) split evenly across it,
    avoiding a ragged last wave."""
    chunks = max((num_tokens + 1) // 2, 1)
    if device is not None and device.type == "npu":
        # The generic AOT reduction uses caller-owned partition rows. It does
        # not query CUDA geometry or depend on simultaneous block residency.
        return min(chunks, 64)
    limit = _get_sm_count() * 3
    if chunks <= limit:
        return chunks
    waves = -(-chunks // limit)
    return -(-chunks // waves)


def _cast_weight_gradient(gradient, weight):
    return gradient.to(weight.dtype)


def _is_rmsnorm_kernel_layout(tensor: torch.Tensor) -> bool:
    if tensor.dim() < 1 or tensor.stride(-1) != 1:
        return False
    if tensor.dim() == 1:
        return True

    hidden_size = tensor.size(-1)
    row_stride = tensor.stride(-2)
    if row_stride < hidden_size:
        return False

    expected_stride = row_stride
    for dim in range(tensor.dim() - 2, -1, -1):
        if tensor.stride(dim) != expected_stride:
            return False
        expected_stride *= tensor.size(dim)
    return True


def _force_kernel_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor the RMSNorm kernel accepts (fully-packed row layout).

    ``.contiguous()`` is NOT sufficient: a leading size-1 dim (e.g. a
    ``[1, S/sp, H]`` sequence-parallel shard of a ``[1, S, H]`` tensor) keeps a
    stale outer stride yet torch still reports ``is_contiguous()==True``, so
    ``.contiguous()`` is a no-op while the kernel's strict
    ``stride(i) == expected_stride`` check still fails. Flatten the leading dims
    and re-expand to force genuine packed strides."""
    if _is_rmsnorm_kernel_layout(tensor):
        return tensor
    flat = tensor.reshape(-1, tensor.shape[-1]).contiguous()
    return flat.reshape(tensor.shape)




def _prepare_add_rmsnorm_inputs(residual, input):
    """Shared prologue for the autograd and inference add_rmsnorm paths:
    validate shapes, force kernel layout, allocate the two outputs."""
    if residual.shape != input.shape:
        raise ValueError(
            f"add_rmsnorm residual shape {tuple(residual.shape)} "
            f"does not match input shape {tuple(input.shape)}"
        )
    residual = _force_kernel_layout(residual)
    input = _force_kernel_layout(input)
    return residual, input, torch.empty_like(input), torch.empty_like(input)


class _AddRmsnormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, input, weight, epsilon, zero_centered):
        residual, input, sum_output, norm_output = _prepare_add_rmsnorm_inputs(
            residual, input
        )
        hidden_size = weight.numel()
        num_tokens = input.numel() // hidden_size
        rstd = torch.empty(num_tokens, dtype=torch.float32, device=input.device)
        if zero_centered:
            CORE.add_rmsnorm_zero_centered_with_rstd(
                sum_output, norm_output, rstd, residual, input, weight, epsilon
            )
        else:
            CORE.add_rmsnorm_with_rstd(
                sum_output, norm_output, rstd, residual, input, weight, epsilon
            )
        ctx.save_for_backward(sum_output, weight, rstd)
        ctx.epsilon = epsilon
        ctx.zero_centered = zero_centered
        return sum_output, norm_output

    @staticmethod
    def backward(ctx, grad_sum_output, grad_norm_output):
        sum_output, weight, rstd = ctx.saved_tensors
        epsilon = ctx.epsilon
        zero_centered = ctx.zero_centered
        hidden_size = weight.numel()

        if grad_norm_output is None:
            d_rms = None
            dw = None
        else:
            grad_norm_output = _force_kernel_layout(grad_norm_output)

            num_tokens = sum_output.numel() // hidden_size
            dw_float = torch.empty(
                hidden_size, dtype=torch.float32, device=sum_output.device
            )

            use_rstd_backward = grad_sum_output is not None and (
                hidden_size <= 4096 or sum_output.device.type == "npu"
            )
            # Same grid helper as the plain rmsnorm backward: the wide kernels
            # share the row partition, which keeps the fused dw bit-identical
            # to the materialized (add + rmsnorm) path.
            num_blocks = _get_plain_rmsnorm_backward_blocks(
                num_tokens, sum_output.device
            )
            partial_dw = torch.empty(
                num_blocks, hidden_size, dtype=torch.float32, device=sum_output.device
            )

            if use_rstd_backward:
                grad_sum_output = _force_kernel_layout(grad_sum_output)
                d_add = torch.empty_like(sum_output)
                if zero_centered:
                    CORE.add_rmsnorm_zero_centered_backward_with_rstd(
                        d_add,
                        dw_float,
                        partial_dw,
                        grad_sum_output,
                        grad_norm_output,
                        sum_output,
                        weight,
                        rstd,
                    )
                else:
                    CORE.add_rmsnorm_backward_with_rstd(
                        d_add,
                        dw_float,
                        partial_dw,
                        grad_sum_output,
                        grad_norm_output,
                        sum_output,
                        weight,
                        rstd,
                    )
                return d_add, d_add, _cast_weight_gradient(dw_float, weight), None, None

            d_rms = torch.empty_like(sum_output)
            if sum_output.device.type == "npu":
                backward = (
                    CORE.rmsnorm_zero_centered_backward_with_rstd
                    if zero_centered
                    else CORE.rmsnorm_backward_with_rstd
                )
                backward(
                    d_rms,
                    dw_float,
                    partial_dw,
                    grad_norm_output,
                    sum_output,
                    weight,
                    rstd,
                    epsilon,
                )
            elif zero_centered:
                CORE.rmsnorm_zero_centered_backward(
                    d_rms,
                    dw_float,
                    partial_dw,
                    grad_norm_output,
                    sum_output,
                    weight,
                    epsilon,
                )
            else:
                CORE.rmsnorm_backward(
                    d_rms,
                    dw_float,
                    partial_dw,
                    grad_norm_output,
                    sum_output,
                    weight,
                    epsilon,
                )
            dw = _cast_weight_gradient(dw_float, weight)

        if grad_sum_output is None:
            d_add = d_rms
        elif d_rms is None:
            d_add = grad_sum_output
        else:
            d_add = grad_sum_output + d_rms

        return d_add, d_add, dw, None, None








class AddRmsnorm:
    def __init__(self):
        self.lib = CORE

    def __call__(
        self, residual, input, weight, epsilon, inference=False, zero_centered=False
    ):
        if inference or not torch.is_grad_enabled():
            if (
                input.dtype == torch.float32
                and residual.dtype == torch.bfloat16
                and not zero_centered
            ):
                # r3 (agent H): the K1 f32in fast plan -- the FP32 mm output
                # feeds the kernel directly and the RNE round to BF16 happens
                # inside it, bit-identical to casting first while saving the
                # separate cast kernel launch.
                if residual.shape != input.shape:
                    raise ValueError(
                        f"add_rmsnorm residual shape {tuple(residual.shape)} "
                        f"does not match input shape {tuple(input.shape)}"
                    )
                residual = _force_kernel_layout(residual)
                input = _force_kernel_layout(input)
                sum_output = torch.empty(
                    input.shape, dtype=torch.bfloat16, device=input.device
                )
                norm_output = torch.empty(
                    input.shape, dtype=torch.bfloat16, device=input.device
                )
                self.lib.add_rmsnorm(
                    sum_output, norm_output, residual, input, weight, epsilon
                )
                return sum_output, norm_output
            residual, input, sum_output, norm_output = _prepare_add_rmsnorm_inputs(
                residual, input
            )
            if zero_centered:
                self.lib.add_rmsnorm_zero_centered(
                    sum_output, norm_output, residual, input, weight, epsilon
                )
            else:
                self.lib.add_rmsnorm(
                    sum_output, norm_output, residual, input, weight, epsilon
                )
            return sum_output, norm_output

        return _AddRmsnormFunction.apply(
            residual, input, weight, epsilon, zero_centered
        )

    def inplace(self, residual, input, weight, epsilon):
        """Zero-centred ``residual + input`` then RMSNorm, writing both results
        back into the caller's buffers: ``residual`` receives the sum and
        ``input`` receives the normalized activation.  No allocation, no
        autograd -- this is the inference fast path."""
        self.lib.add_rmsnorm_zero_centered_inplace(
            residual, input, weight, float(epsilon)
        )


add_rmsnorm = AddRmsnorm()
