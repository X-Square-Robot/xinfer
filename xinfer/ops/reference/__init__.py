"""Torch reference implementations of the operators, one category module per backend family.

These define the semantics every backend is checked against, architecture-agnostically: the same functions run
on every platform, and every backend — CUDA or NPU — is validated against them the same way. They follow the
numerics of the Hugging Face / Wall-X modules they replace (for example, RMSNorm normalizes and applies the
weight in FP32 and rounds once, as Wall-X does) so that a model built on reference operators can be compared
with its source model. Backends with the same semantics register through ``xinfer.ops.registry``.

The layout mirrors the hardware backends (``xinfer/ops/backends/<backend>/``): ``attn``, ``gemm``, ``norm`` and
``flow`` correspond one-to-one with the backend modules of the same name; ``rope``, ``elementwise`` and ``conv``
hold the operators no backend registers yet; ``mxfp8`` and ``nvfp4`` hold the low-precision ones. This package
re-exports every reference function, so ``xinfer.ops.reference.<operator>`` resolves regardless of its module.
"""

from xinfer.ops.reference.attn import (
    action_attention,
    attention,
    causal_attention,
    chunk_gated_delta_rule,
    gate_output,
    gated_causal_attention,
    gated_delta_net,
    gated_kvcache_attention,
    kvcache_attention,
    masked_joint_attention,
    prefix_attention,
    prefix_kvcache_attention,
    rtc_pinned_mask,
    segment_attention,
    varlen_attention,
)
from xinfer.ops.reference.conv import causal_conv1d, conv2d
from xinfer.ops.reference.elementwise import (
    gated_residual,
    gelu,
    gelu_tanh,
    gelu_tanh_and_mul,
    silu,
    silu_and_mul,
)
from xinfer.ops.reference.flow import flow_input_embed, flow_velocity_euler
from xinfer.ops.reference.gemm import (
    gemm_add_rms_norm,
    gemm_epilogue,
    gemm_glu_epilogue,
    linear,
    mlp_add_rms_norm,
    qkv_norm_rope_kvcache,
    qkv_norm_split,
    qkv_rope_epilogue,
    qkv_rope_for_attention,
    qkv_rope_kvcache,
)
from xinfer.ops.reference.norm import (
    add_zero_centered_rms_norm,
    ada_rms_norm,
    add_rms_norm,
    gated_residual_modulated_rms_norm,
    gated_rms_norm,
    identity_layer_norms_,
    l2_normalize,
    layer_norm,
    layer_norm_cast,
    modulated_rms_norm,
    rms_norm,
    zero_centered_rms_norm,
)
from xinfer.ops.reference.rope import (
    apply_rotary,
    pos_embed_interp,
    rotary_embedding,
    rotary_embedding_rounded,
    rotate_half,
    rope_full,
    rot_pos,
)


def prefetch_weights(weights):
    """The portable weight-prefetch hint schedules no work and returns an empty handle."""
    return None


def join_prefetch(side):
    """The portable hint has no outstanding work to join."""


__all__ = (
    "action_attention", "ada_rms_norm", "add_rms_norm", "add_zero_centered_rms_norm", "apply_rotary",
    "attention", "causal_attention", "causal_conv1d", "chunk_gated_delta_rule", "conv2d", "flow_input_embed",
    "flow_velocity_euler", "gate_output", "gated_causal_attention", "gated_delta_net",
    "gated_kvcache_attention", "gated_residual", "gated_residual_modulated_rms_norm", "gated_rms_norm",
    "gelu", "gelu_tanh", "gelu_tanh_and_mul", "gemm_epilogue", "gemm_glu_epilogue", "identity_layer_norms_",
    "gemm_add_rms_norm", "mlp_add_rms_norm", "qkv_rope_for_attention",
    "kvcache_attention", "l2_normalize", "layer_norm", "layer_norm_cast", "linear", "masked_joint_attention",
    "modulated_rms_norm", "pos_embed_interp", "prefix_attention", "prefix_kvcache_attention",
    "prefetch_weights", "join_prefetch",
    "qkv_norm_rope_kvcache", "qkv_norm_split", "qkv_rope_epilogue", "qkv_rope_kvcache", "rms_norm",
    "rope_full", "rot_pos", "rotary_embedding", "rotary_embedding_rounded", "rotate_half", "rtc_pinned_mask",
    "segment_attention", "silu", "silu_and_mul", "varlen_attention", "zero_centered_rms_norm"
)
