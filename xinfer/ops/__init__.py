"""Operator contracts, reference implementations and backend selection.

Layers call operators through the op table that is active while they are
constructed (``use_ops``), so the backend for every operator is fixed at
model build time. Backend packages register through the ``xinfer.ops`` entry
point group (``register(registry)``).
"""

import logging

import torch

from xinfer.ops import fused, mxfp8, nvfp4, reference
from xinfer.ops.registry import (
    REFERENCE_BACKEND,
    Implementation,
    OpRegistry,
    OpTable,
    active_ops,
    use_ops,
)
from xinfer.ops.spec import Contract, OpSpec

__all__ = [
    "Contract",
    "Implementation",
    "OpRegistry",
    "OpSpec",
    "OpTable",
    "active_ops",
    "registry",
    "use_ops",
]
logger = logging.getLogger(__name__)

registry = OpRegistry()

for _spec in (
    OpSpec("linear", "linear(x, weight, bias=None) -> Tensor", reference.linear),
    OpSpec("rms_norm", "rms_norm(x, weight, eps) -> Tensor", reference.rms_norm),
    OpSpec("silu_and_mul", "silu_and_mul(gate, up) -> Tensor", reference.silu_and_mul),
    OpSpec("silu", "silu(x) -> Tensor", reference.silu),
    OpSpec("gelu", "gelu(x) -> Tensor  # exact erf", reference.gelu),
    OpSpec("gelu_tanh", "gelu_tanh(x) -> Tensor  # tanh approximation", reference.gelu_tanh),
    OpSpec("gelu_tanh_and_mul",
           "gelu_tanh_and_mul(gate, up) -> Tensor",
           reference.gelu_tanh_and_mul),
    OpSpec("gated_residual", "gated_residual(x, y, gate) -> x + y * gate  # product rounded first",
           reference.gated_residual),
    OpSpec("layer_norm", "layer_norm(x, weight, bias, eps) -> fp32 Tensor  # in FP32", reference.layer_norm),
    OpSpec("ada_rms_norm", "ada_rms_norm(x[B, L, K], scale[B, 1, K], shift[B, 1, K], eps) -> Tensor"
           "  # x·rsqrt(mean(x²)+eps)·(1+scale)+shift in FP32, x.dtype out", reference.ada_rms_norm),
    OpSpec("conv2d", "conv2d(x[B, C, H, W], weight[O, C, kH, kW], bias, stride) -> Tensor  # no padding",
           reference.conv2d),
    OpSpec("causal_conv1d", "causal_conv1d(x[B, C, L], weight[C, 1, K]) -> silu(depthwise causal conv) [B, C, L]",
           reference.causal_conv1d),
    OpSpec("chunk_gated_delta_rule",
           "chunk_gated_delta_rule(q[B, L, H, Dk], k, v[B, L, H, Dv], g[B, L, H] fp32, beta[B, L, H]) -> [B, L, H, Dv]"
           "  # zero initial state, q/k L2-normalized, FP32 chunks of 64",
           reference.chunk_gated_delta_rule),
    OpSpec("gated_rms_norm", "gated_rms_norm(x[..., D], gate[..., D], weight[D], eps) -> rms_norm(x)·weight·silu(gate)",
           reference.gated_rms_norm),
    OpSpec("rotary_embedding",
           "rotary_embedding(x[..., H, D], cos[..., 1, D] fp32, sin) -> Tensor  # in FP32, x.dtype out",
           reference.rotary_embedding),
    OpSpec("rotary_embedding_rounded",
           "rotary_embedding_rounded(x[..., H, D], cos[..., 1, D], sin) -> Tensor  # in x.dtype, products rounded",
           reference.rotary_embedding_rounded),
    OpSpec(
        "causal_attention",
        "causal_attention(q[B, L, H, D], k[B, L, KV, D], v, *, scale=None) -> [B, L, H, D]",
        reference.causal_attention,
    ),
    OpSpec(
        "action_attention",
        "action_attention(q[B, L, H, D], k[B, L, KV, D], v, k_prompt[B, KV, P, D], v_prompt, num_pinned[B],"
        " prompt_lengths[B]=None, *, scale=None) -> [B, L, H, D]"
        "  # RTC joint attention: rows < num_pinned see own keys < num_pinned; prompts left-padded to P",
        reference.action_attention,
    ),
    OpSpec(
        "masked_joint_attention",
        "masked_joint_attention(q[B, L, H, D], k[B, L, KV, D], v, k_prompt[B, KV, P, D], v_prompt,"
        " mask[B, 1, L, P + L] bool, *, scale=None) -> [B, L, H, D]",
        reference.masked_joint_attention,
    ),
    OpSpec(
        "kvcache_attention",
        "kvcache_attention(q[B, L, H, D], k[B, L, KV, D], v, k_cache[B, S, KV, D], v_cache, cache_start[B],"
        " cache_valid[B], *, scale=None) -> [B, L, H, D]"
        "  # writes k/v into the caches after each row's cached keys, then attends to cached + own keys",
        reference.kvcache_attention,
    ),
    OpSpec(
        "prefix_attention",
        "prefix_attention(q[B, S, H, D], k[B, S, KV, D], v, valid[B], *, scale=None) -> [B, S, H, D]"
        "  # bidirectional; keys are each row's first valid positions (right padding)",
        reference.prefix_attention,
    ),
    OpSpec(
        "varlen_attention",
        # cu_seqlens is a host sequence (part of the CUDA graph key), which keeps
        # the reference capturable.
        "varlen_attention(q, k, v, cu_seqlens, *, scale=None) -> Tensor  # packed [T, H, D]",
        reference.varlen_attention,
    ),
    OpSpec(
        "mxfp8_quantize",
        "mxfp8_quantize(x_bf16[M, K], *, share_scale_128=True) -> (e4m3[M, K], e8m0 scales uint8)",
        mxfp8.quantize,
    ),
    OpSpec(
        "mxfp8_gemm",
        "mxfp8_gemm(a_q, a_scales, w_q[N, K], w_scales, bias=None, *, fast_accum=False) -> bf16[M, N]",
        mxfp8.gemm,
    ),
    # Fused forms (xinfer/ops/mxfp8.py): FP32 intermediates are not rounded to BF16.
    OpSpec(
        "mxfp8_rmsnorm_quantize",
        "mxfp8_rmsnorm_quantize(x_bf16[M, K], weight[K], eps, *, share_scale_128=True) -> (e4m3, scales)",
        mxfp8.rmsnorm_quantize,
    ),
    OpSpec(
        "mxfp8_silu_mul_quantize",
        "mxfp8_silu_mul_quantize(gate_up_bf16[M, 2K], *, share_scale_128=True) -> (e4m3[M, K], scales)",
        mxfp8.silu_mul_quantize,
    ),
    OpSpec(
        "mxfp8_gemm_rope",
        "mxfp8_gemm_rope(a_q, a_scales, w_q, w_scales, bias, table[D/4, M, 4], q_dim, k_dim, head_dim)"
        " -> bf16[M, N]  # w rows in rope_pair_permutation order",
        mxfp8.gemm_rope,
    ),
    OpSpec(
        "mxfp8_gemm_silu_mul",
        "mxfp8_gemm_silu_mul(a_q, a_scales, w_q, w_scales, bias=None) -> bf16[M, N/2]"
        "  # w rows in gate_up_interleave order",
        mxfp8.gemm_silu_mul,
    ),
    OpSpec(
        "mxfp8_gemm_residual",
        "mxfp8_gemm_residual(a_q, a_scales, w_q, w_scales, residual_bf16[M, N], bias=None) -> bf16[M, N]",
        mxfp8.gemm_residual,
    ),
    # Fused BF16 operators used by Wall-OSS.
    OpSpec("gemm_epilogue",
           "gemm_epilogue(x[M, K], weight[N, K], bias[N]=None, *, activation=None, residual=None, residual_gate[R, N]"
           "=None, row_class[M]=None, row_scale[M]=None) -> [M, N]  # residual + bf16(act(y)·gate[row_class])",
           fused.gemm_epilogue),
    OpSpec("gemm_glu_epilogue",
           "gemm_glu_epilogue(x[M, K], weight[2N, K], *, bias[2N]=None, modulation[R, 3K]=None, row_class[M]=None,"
           " eps)"
           " -> silu(x̂·W_gateᵀ)·(x̂·W_upᵀ)  # x̂ = AdaRMS(x), FP32",
           fused.gemm_glu_epilogue),
    OpSpec("qkv_rope_epilogue",
           "qkv_rope_epilogue(x[M, K], weight[3HD, K], bias, cos[M, R/2], sin, num_heads, head_dim)"
           " -> [Q | K | V]  # RoPE on the FP32 accumulator", fused.qkv_rope_epilogue),
    OpSpec("prefix_kvcache_attention",
           "prefix_kvcache_attention(q[B, M, H, D], k_cache[B, L, KV, D], v_cache, prefix_len[B], new_pos, *,"
           " flash_mask=None, scale=None) -> [B, M, HD]  # keys [0, prefix) ∪ [new_pos, new_pos + M)",
           fused.prefix_kvcache_attention),
    OpSpec("qkv_rope_kvcache",
           "qkv_rope_kvcache(x[M, K], weight[(H + 2KV)D, K], bias, cos[M, R/2], sin, k_cache[KV, rows, D], v_cache,"
           " start, H, KV, D) -> [M, HD]  # K/V written in place", fused.qkv_rope_kvcache),
    OpSpec("rms_norm_cast", "rms_norm_cast(x[..., K], weight[K], eps) -> x.dtype  # x̂·w, one rounding",
           fused.rms_norm_cast),
    OpSpec("add_rms_norm", "add_rms_norm(x, residual, weight[K], eps) -> (residual + x, rms_norm_cast(sum))",
           fused.add_rms_norm),
    OpSpec("flow_velocity_euler",
           "flow_velocity_euler(x[B, M, A] fp32, h[B, M, K] (normed), weight[A, K], dt, dof_mask=None,"
           " padding_velocity=None, pinned=None, pinned_values=None) -> fp32[B, M, A]  # flow velocity",
           fused.flow_velocity_euler),
    # NVFP4 (xinfer/ops/nvfp4.py): packed E2M1 uint8 [M, K/2] with E4M3 scales per 16 columns.
    OpSpec(
        "nvfp4_quantize",
        "nvfp4_quantize(x_bf16[M, K], *, global_scale=1.0, mse=False) -> (e2m1x2 uint8[M, K/2], e4m3 scales)",
        nvfp4.quantize,
    ),
    OpSpec(
        "nvfp4_gemm",
        "nvfp4_gemm(a_q, a_scales, w_q[N, K/2], w_scales, bias=None, *, alpha=1.0) -> bf16[M, N]",
        nvfp4.gemm,
    ),
    OpSpec(
        "nvfp4_rmsnorm_quantize",
        "nvfp4_rmsnorm_quantize(x_bf16[M, K], weight[K], eps) -> (e2m1x2, scales)  # normalized value rounded to BF16",
        nvfp4.rmsnorm_quantize,
    ),
    OpSpec(
        "nvfp4_silu_mul_quantize",
        "nvfp4_silu_mul_quantize(gate_up_bf16[M, 2K]) -> (e2m1x2[M, K/2], scales)",
        nvfp4.silu_mul_quantize,
    ),
):
    registry.define(_spec)
del _spec


def warn_slow_fallbacks(model: str, table: OpTable, stage_ops, device) -> list[str]:
    """Warn when, on a CUDA device, operators a model calls have no backend kernel and run on a fused reference.

    A fused operator's reference (``xinfer/ops/fused.py``) composes torch operations in place of a backend kernel,
    so the model runs much slower on it. ``Policy`` and ``xinfer-serve`` call this once the model is built.

    Args:
        model: the model name, for the message.
        table: the op table the model was built with.
        stage_ops: ``{stage: operator names}``, the model's ``Capabilities.stage_ops``.
        device: the model's device.

    Returns:
        The operators on a fused reference, sorted (none on other devices).
    """
    if torch.device(device).type != "cuda":
        return []
    names = sorted({name for names in stage_ops.values() for name in names
                    if table.selection(name).backend == REFERENCE_BACKEND
                    and registry.spec(name).reference.__module__ == fused.__name__})
    if names:
        logger.warning("%s: %d operators have no fast kernel loaded and fall back to slow torch code (%s). Install "
                       "xcompute built for this GPU, or set XINFER_XCOMPUTE_LIBRARY to such a build: docs/kernels.md",
                       model, len(names), ", ".join(names))
    return names
