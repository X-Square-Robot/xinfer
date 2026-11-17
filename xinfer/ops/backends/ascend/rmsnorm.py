import torch
from xinfer.ops.backends.ascend import _native
from xinfer.ops.backends.ascend.core import CORE
from xinfer.ops.backends.ascend.core import CORE


_MXFP8_SCALE_BLOCKS = (32, 128)


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


class _RmsnormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, epsilon, zero_centered, use_wide):
        input = _force_kernel_layout(input)
        output = torch.empty_like(input)
        hidden_size = weight.numel()
        num_tokens = input.numel() // hidden_size
        # Materialize per-row rstd (1/rms) so the backward can skip the
        # sum(x^2) reduction; it is also bit-identical to the rstd the forward
        # normalized with.
        rstd = torch.empty(num_tokens, dtype=torch.float32, device=input.device)
        if zero_centered:
            CORE.rmsnorm_zero_centered_with_rstd(
                output, rstd, input, weight, epsilon, use_wide
            )
        else:
            CORE.rmsnorm_with_rstd(
                output, rstd, input, weight, epsilon, use_wide
            )
        ctx.save_for_backward(input, weight, rstd)
        ctx.epsilon = epsilon
        ctx.zero_centered = zero_centered
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, rstd = ctx.saved_tensors
        epsilon = ctx.epsilon
        zero_centered = ctx.zero_centered
        grad_output = _force_kernel_layout(grad_output)
        hidden_size = weight.numel()
        num_tokens = input.numel() // hidden_size

        dx = torch.empty_like(input)
        dw_float = torch.empty(hidden_size, dtype=torch.float32, device=input.device)

        # Pre-allocate partial_dw buffer: [num_blocks, hidden_size]
        num_blocks = _get_plain_rmsnorm_backward_blocks(num_tokens, input.device)
        partial_dw = torch.empty(
            num_blocks, hidden_size, dtype=torch.float32, device=input.device
        )

        if zero_centered:
            CORE.rmsnorm_zero_centered_backward_with_rstd(
                dx, dw_float, partial_dw, grad_output, input, weight, rstd, epsilon
            )
        else:
            CORE.rmsnorm_backward_with_rstd(
                dx, dw_float, partial_dw, grad_output, input, weight, rstd, epsilon
            )

        dw = _cast_weight_gradient(dw_float, weight)
        return dx, dw, None, None, None


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


def _rmsnorm_bf16(input, weight, epsilon, zero_centered, residual, quant, valid_rows):
    if residual is not None or quant is not None or zero_centered:
        raise ValueError(
            "weight=None / valid_rows support plain BF16 RMSNorm only "
            "(no residual, quant or zero_centered)"
        )
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (input, weight)
    ):
        raise RuntimeError("weight=None / valid_rows RMSNorm is inference only")
    if not isinstance(input, torch.Tensor) or input.dim() < 1:
        raise ValueError("input must be a tensor with at least one dimension")
    hidden = input.size(-1)
    flat = input.reshape(-1, hidden)
    output = torch.empty_like(flat)
    CORE.rmsnorm_bf16_into(output, flat, weight, valid_rows, float(epsilon), 0)
    return output.reshape(input.shape)


def _rmsnorm_quantized(
    input, weight, epsilon, zero_centered, residual, quant, scale_block, round_to_bf16
):
    if quant is None:
        raise ValueError("scale_block / round_to_bf16 only apply with quant='mxfp8'")
    if quant != "mxfp8":
        raise ValueError(f"quant must be None or 'mxfp8', got {quant!r}")
    if zero_centered:
        raise ValueError("quantized RMSNorm does not support zero_centered")
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (input, weight, residual)
    ):
        raise RuntimeError("quantized RMSNorm is inference only (no autograd)")
    if scale_block not in _MXFP8_SCALE_BLOCKS:
        raise ValueError(
            f"scale_block must be one of {_MXFP8_SCALE_BLOCKS}, got {scale_block}"
        )
    if not isinstance(input, torch.Tensor) or input.dim() < 1:
        raise ValueError("input must be a tensor with at least one dimension")
    hidden = input.size(-1)
    rows = input.numel() // hidden if hidden > 0 else 0
    padded = -(-hidden // 32) * 32
    flat = input.reshape(rows, hidden)
    quantized = torch.empty(
        (rows, padded), device=input.device, dtype=torch.float8_e4m3fn
    )
    scales = torch.empty(
        (scale_storage_elements(rows, padded),), device=input.device, dtype=torch.uint8
    )
    if residual is None:
        residual_flat = residual_sum = flat
    else:
        if residual.shape != input.shape:
            raise ValueError("residual must have the same shape as input")
        residual_flat = residual.reshape(rows, hidden)
        residual_sum = torch.empty_like(flat)
    CORE.rmsnorm_quant_mxfp8_into(
        flat,
        residual_flat,
        residual_sum,
        weight,
        quantized,
        scales,
        hidden,
        float(epsilon),
        residual is not None,
        scale_block == 128,
        bool(round_to_bf16),
        0,
    )
    quantized = quantized.reshape(*input.shape[:-1], padded)
    if residual is None:
        return quantized, scales
    return residual_sum.reshape(input.shape), quantized, scales


class Rmsnorm:
    def __init__(self):
        self.lib = CORE

    def __call__(
        self,
        input,
        weight,
        epsilon,
        inference=False,
        zero_centered=False,
        use_wide=True,
        *,
        residual=None,
        quant=None,
        scale_block=128,
        round_to_bf16=False,
        valid_rows=None,
    ):
        """RMSNorm, the single entry point of the RMSNorm semantics.

        ``y = (x [+ residual]) * rsqrt(mean((x [+ residual])^2) + epsilon) * w``
        (``w`` replaced by ``1 + w`` when ``zero_centered``).

        Output format, chosen by keyword arguments:

        * ``quant=None`` (default): BF16 ``y``. ``use_wide=True`` preserves the
          existing behavior: the forward pass tries the wide kernel and falls
          back to the generic kernel when needed; ``False`` bypasses it (no
          effect on backward). With ``residual`` this is ``add_rmsnorm`` and
          returns ``(x + residual, y)``.
        * ``quant="mxfp8"``: inference only. Returns ``(quantized, scales)``,
          or ``(x + residual, quantized, scales)`` with ``residual``.
          ``quantized`` is E4M3 with the last dimension padded to a multiple
          of 32; ``scales`` is the flat E8M0 storage in the SWIZZLE_32_4_4
          layout (one byte per 32 columns, rows padded to 128).
          ``scale_block=128`` gives the four 32-column slots of each
          128-column tile one shared scale;
          ``scale_block=32`` gives every 32 columns its own scale (OCP MX).
          ``round_to_bf16=True`` rounds ``y`` to BF16 before quantizing, i.e.
          quantizes what the BF16 RMSNorm would output; ``False`` quantizes
          the FP32 ``y`` directly.
          ``weight`` may be ``[K]`` or ``[R, K]`` with ``R`` identical rows
          (``R`` a power of two <= 64, prepared once, e.g.
          ``weight.repeat(16, 1)``): at large row counts every block reads
          the whole weight, and replicas spread those reads over L2.

        ``weight=None`` normalizes without a scale (for models that fold the
        norm weight into the next linear layer). ``valid_rows`` (an int32
        CUDA tensor with one element, read on the GPU so CUDA graphs can
        replay with a new count) writes zeros to every row at or past it.
        Both are inference only, BF16, without residual or quantization.
        """
        if weight is None or valid_rows is not None:
            return _rmsnorm_bf16(
                input, weight, epsilon, zero_centered, residual, quant, valid_rows
            )
        if quant is not None or scale_block != 128 or round_to_bf16:
            return _rmsnorm_quantized(
                input,
                weight,
                epsilon,
                zero_centered,
                residual,
                quant,
                scale_block,
                round_to_bf16,
            )
        if residual is not None:
            return add_rmsnorm(
                residual,
                input,
                weight,
                epsilon,
                inference=inference,
                zero_centered=zero_centered,
            )
        # Under no_grad the Function path would still materialize rstd for a
        # backward that can never run; route to the inference entry instead.
        if inference or not torch.is_grad_enabled():
            input = _force_kernel_layout(input)
            output = torch.empty_like(input)
            if zero_centered:
                self.lib.rmsnorm_zero_centered(output, input, weight, epsilon, use_wide)
            else:
                self.lib.rmsnorm(output, input, weight, epsilon, use_wide)
            return output
        return _RmsnormFunction.apply(input, weight, epsilon, zero_centered, use_wide)


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


def rmsnorm_apply_rstd(input, rstd, weight, zero_centered=True):
    """Bit-aligned apply-only RMSNorm (rmsnorm-r3).

    ``y = bf16((x32 * rstd) * (1 + w))`` with the per-row FP32 ``rstd``
    supplied by the caller. The statistics chain must be produced by the
    torch reference ops (``rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)``):
    the in-kernel reduction order and approximate rsqrt of the fused plans
    are not bit-compatible with the aclnn patterns, which compounds over the
    prefill norm chain to action-gate failures. This apply keeps the
    reference multiplication association and a single RNE round, so the
    output is bitwise-identical to the torch composition on identical
    inputs (probe-r1 F/G receipts).

    ``input`` is BF16 with last-dim 2048 and a row count divisible by 4;
    ``rstd`` is a contiguous FP32 vector with one element per row;
    ``weight`` is the raw ``[2048]`` FP32/BF16 scale.
    """
    input = _force_kernel_layout(input)
    output = torch.empty_like(input)
    CORE.rmsnorm_apply_rstd(output, rstd, input, weight, bool(zero_centered))
    return output


def qkv_split_rstd(packed, rstd_q, rstd_k, q_factor, k_factor):
    """Bit-aligned prefill QKV split + norm apply (rmsnorm-r4).

    ``packed`` is the checkpoint layout ``[b, n, 5120]`` (``[q_h | gate_h]``
    interleaved, then K, then V; H=8, KV=2, D=256). The per-head FP32
    ``rstd_q``/``rstd_k`` come from the torch reference statistics chain and
    ``q_factor``/``k_factor`` are ``(weight.float() + 1)``; the kernel applies
    ``((x32 * factor) * rstd)`` with one RNE round (the ``_zero_centered_heads``
    association) and copies v and the gate. Returns
    ``(q [b, n, 8, 256], k [b, n, 2, 256], v [b, n, 2, 256], gate [b, n, 2048])``.
    """
    b, n = packed.shape[0], packed.shape[1]
    q = torch.empty((b, n, 8, 256), dtype=packed.dtype, device=packed.device)
    k = torch.empty((b, n, 2, 256), dtype=packed.dtype, device=packed.device)
    v = torch.empty_like(k)
    gate = torch.empty((b, n, 2048), dtype=packed.dtype, device=packed.device)
    CORE.qkv_split_rstd(q, k, v, gate, packed, rstd_q, rstd_k,
                                q_factor, k_factor)
    return q, k, v, gate


def rope_half(q, k, cos, sin):
    """Bit-aligned full-head half-split rope over the normed prefill q/k
    (rmsnorm-r4): separately rounded multiplies (no FMA), one BF16 round.
    ``rotary_dim == head_dim == 256``; ``cos``/``sin`` are FP32 ``[.., n, 128]``.
    Returns ``(q_rot, k_rot)`` shaped like the inputs.
    """
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    CORE.rope_half(q_out, k_out, q, k, cos, sin)
    return q_out, k_out
