"""Operator contracts, reference implementations and backend selection.

Layers call operators through the op table that is active while they are
constructed (``use_ops``), so the backend for every operator is fixed at
model build time. Backend modules register their implementations with
``register`` when they are imported; packages outside XInfer name theirs in
the ``xinfer.ops`` entry point group.
"""

import logging

import torch

from xinfer.ops import reference
from xinfer.ops.reference import attn, flow, gemm, mxfp8, norm, nvfp4, rope
from xinfer.ops.registry import (
    REFERENCE_BACKEND,
    STRICT_ENV,
    Implementation,
    OpRegistry,
    OpTable,
    active_ops,
    register,
    registry,
    strict,
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
    "register",
    "registry",
    "use_ops",
]
logger = logging.getLogger(__name__)

# The operators whose references compose torch code in place of a backend kernel (the fused families of
# xinfer/ops/reference/); ``warn_slow_fallbacks`` reports these when they run on the reference.
FUSED_OPS = frozenset({"gemm_epilogue", "gemm_glu_epilogue", "qkv_rope_epilogue", "qkv_rope_kvcache",
                       "prefix_kvcache_attention", "add_rms_norm", "flow_velocity_euler"})

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
    # Fused forms (xinfer/ops/reference/mxfp8.py): FP32 intermediates are not rounded to BF16.
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
    # Fused vision, language and action operators for Wall-X 2B.
    OpSpec("layer_norm_cast", "layer_norm_cast(x, weight, bias, eps) -> x.dtype  # FP32 inside",
           norm.layer_norm_cast),
    OpSpec("zero_centered_rms_norm", "zero_centered_rms_norm(x, w, eps) -> x̂·(1 + w)  # FP32, one rounding",
           norm.zero_centered_rms_norm),
    OpSpec("add_zero_centered_rms_norm",
           "add_zero_centered_rms_norm(x, residual, w, eps) -> (residual + x, zero_centered_rms_norm(·))",
           norm.add_zero_centered_rms_norm),
    OpSpec("gated_delta_net",
           "gated_delta_net(qkv[B, L, C], zba[B, L, HvDv + 2Hv], conv_weight[C, Kc], A_log, dt_bias, norm_weight,"
           " num_k_heads, num_v_heads, head_k_dim, head_v_dim, eps) -> [B, L, HvDv]  # zero initial state",
           attn.gated_delta_net),
    OpSpec("pos_embed_interp", "pos_embed_interp(weight[G², C], grids, merge_size) -> [N, C]  # bilinear, one rounding",
           rope.pos_embed_interp),
    OpSpec("rot_pos", "rot_pos(inv_freq[F], grids, merge_size) -> fp32 [N, 2F]  # [row·f | col·f]",
           rope.rot_pos),
    OpSpec("rope_full", "rope_full(q[B, L, H, D], k[B, L, KV, D], cos[B|1, L, R/2], sin, rotary_dim) -> (q, k)",
           rope.rope_full),
    OpSpec("qkv_norm_split",
           "qkv_norm_split(packed[B, L, 2HD + 2KVD], q_weight, k_weight, num_heads, num_kv_heads, head_dim, eps)"
           " -> (q, k, v, gate)  # [q_h | gate_h] per head, then K, V",
           gemm.qkv_norm_split),
    OpSpec("segment_attention",
           "segment_attention(q[T, H, D], k, v, cu_seqlens, *, scale=None) -> [T, H, D]"
           "  # varlen_attention's function on xcompute's attention_varlen",
           attn.segment_attention),
    OpSpec("gated_causal_attention",
           "gated_causal_attention(q[T, H, D], k[T, KV, D], v, gate[T, HD], *, scale=None) -> attention·sigmoid(gate)",
           attn.gated_causal_attention),
    OpSpec("gated_kvcache_attention",
           "gated_kvcache_attention(q[B, M, H, D], k_cache[B, L, KV, D], v_cache, prefix_len[B], new_pos,"
           " gate[B, M, HD], *, flash_mask=None, scale=None) -> attention·sigmoid(gate)"
           "  # keys [0, prefix) ∪ [new_pos, new_pos + M)",
           attn.gated_kvcache_attention),
    OpSpec("flow_input_embed",
           "flow_input_embed(x[B, M, A] fp32, weight[heads, N, 2A] fp32, head_index=None, extra=None,"
           " input_mask=None, addends=(), override=None, override_mask=None) -> bf16[B, M, N]",
           flow.flow_input_embed),
    OpSpec("modulated_rms_norm",
           "modulated_rms_norm(x[B, L, K], modulation[B, 3K], eps, pinned_modulation=None, pinned_rows=None)"
           " -> (normed, gate[B, 1 or L, K])", norm.modulated_rms_norm),
    OpSpec("gated_residual_modulated_rms_norm",
           "gated_residual_modulated_rms_norm(residual[B, L, K], branch, branch_gate, modulation[B, 3K], eps,"
           " pinned_modulation=None, pinned_rows=None) -> (normed, gate, residual + branch·branch_gate)",
           norm.gated_residual_modulated_rms_norm),
    OpSpec("identity_layer_norms_",
           "identity_layer_norms_(residual[B, L, K], branch, branch_gate, input_modulation[B, 3K],"
           " post_modulation[B, 3K], eps, pinned_input_modulation=None, pinned_post_modulation=None,"
           " pinned_rows=None) -> gate  # residual and branch updated in place", norm.identity_layer_norms_),
    OpSpec("qkv_norm_rope_kvcache",
           "qkv_norm_rope_kvcache(x[M, K], weight[(H + 2KV)D + G, K], q_weight[D], k_weight[D], cos[M, R/2],"
           " sin, k_cache[KV, rows, D], v_cache, start, H, KV, D, eps) -> [M, HD + G]  # K/V written in place",
           gemm.qkv_norm_rope_kvcache),
    # Fused BF16 operators used by Wall-OSS.
    OpSpec("gemm_epilogue",
           "gemm_epilogue(x[M, K], weight[N, K], bias[N]=None, *, activation=None, residual=None, residual_gate[R, N]"
           "=None, row_class[M]=None, row_scale[M]=None) -> [M, N]  # residual + bf16(act(y)·gate[row_class])",
           gemm.gemm_epilogue),
    OpSpec("gemm_glu_epilogue",
           "gemm_glu_epilogue(x[M, K], weight[2N, K], *, bias[2N]=None, modulation[R, 3K]=None, row_class[M]=None,"
           " eps)"
           " -> silu(x̂·W_gateᵀ)·(x̂·W_upᵀ)  # x̂ = AdaRMS(x), FP32",
           gemm.gemm_glu_epilogue),
    OpSpec("qkv_rope_epilogue",
           "qkv_rope_epilogue(x[M, K], weight[3HD, K], bias, cos[M, R/2], sin, num_heads, head_dim)"
           " -> [Q | K | V]  # RoPE on the FP32 accumulator", gemm.qkv_rope_epilogue),
    OpSpec("prefix_kvcache_attention",
           "prefix_kvcache_attention(q[B, M, H, D], k_cache[B, L, KV, D], v_cache, prefix_len[B], new_pos, *,"
           " flash_mask=None, scale=None) -> [B, M, HD]  # keys [0, prefix) ∪ [new_pos, new_pos + M)",
           attn.prefix_kvcache_attention),
    OpSpec("qkv_rope_kvcache",
           "qkv_rope_kvcache(x[M, K], weight[(H + 2KV)D, K], bias, cos[M, R/2], sin, k_cache[KV, rows, D], v_cache,"
           " start, H, KV, D) -> [M, HD]  # K/V written in place", gemm.qkv_rope_kvcache),
    OpSpec("add_rms_norm", "add_rms_norm(x, residual, weight[K], eps) -> (residual + x, rms_norm(sum))",
           norm.add_rms_norm),
    OpSpec("flow_velocity_euler",
           "flow_velocity_euler(x[B, M, A] fp32, h[B, M, K] (normed), weight[A, K], dt, dof_mask=None,"
           " padding_velocity=None, pinned=None, pinned_values=None) -> fp32[B, M, A]  # flow velocity",
           flow.flow_velocity_euler),
    # NVFP4 (xinfer/ops/reference/nvfp4.py): packed E2M1 uint8 [M, K/2] with E4M3 scales per 16 columns.
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


def _is_slow_fallback(implementation) -> bool:
    """Whether this selection only composes torch code, whatever backend name it registered under.

    The reference is one case; the Ascend backend's graph-safe attention is the other — it registers as
    ``ascend:torch`` so it composes torch primitives in place of a kernel, exactly as slow as the reference.
    """
    return implementation.backend == REFERENCE_BACKEND or implementation.library == "torch"


def warn_slow_fallbacks(model: str, table: OpTable, stage_ops, device) -> list[str]:
    """Warn when, on an accelerator, operators a model calls have no backend kernel and run on composed torch code.

    Such a reference (the ``gemm``, ``norm``, ``attn`` and ``flow`` modules of ``xinfer/ops/reference/``, or the
    Ascend backend's graph-safe attention) composes torch operations in place of a backend kernel, so the model
    runs much slower on it. ``Policy`` and ``xinfer-serve`` call this once the model is built.

    Args:
        model: the model name, for the message.
        table: the op table the model was built with.
        stage_ops: ``{stage: operator names}``, the model's ``Capabilities.stage_ops``.
        device: the model's device.

    Returns:
        The operators on composed torch code, sorted (none on other devices).

    Raises:
        RuntimeError: there are such operators and ``XINFER_OPS_STRICT`` is set.
    """
    kind = torch.device(device).type
    if kind not in ("cuda", "npu"):
        return []
    names = sorted({name for names in stage_ops.values() for name in names
                    if _is_slow_fallback(table.selection(name)) and name in FUSED_OPS})
    if names:
        hint = ("Install xcompute built for this GPU, or set XINFER_XCOMPUTE_LIBRARY to such a build: "
                "docs/kernels.md" if kind == "cuda" else
                "Install the Ascend xcompute build, or set XINFER_XCOMPUTE_LIBRARY to it: docs/ascend950dt.md")
        message = (f"{model}: {len(names)} operators have no fast kernel loaded and fall back to slow torch code "
                   f"({', '.join(names)}). {hint}")
        if strict():
            raise RuntimeError(f"{message} ({STRICT_ENV} is set)")
        logger.warning("%s", message)
    return names
