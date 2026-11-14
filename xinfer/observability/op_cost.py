"""Operator cost: the FLOPs, bytes read and written, and compute pipe of one operator call (docs/profiling.md).

Profiling uses it for an operator call's roofline bound; the inference path does not. ``op_cost(name, args, kwargs,
outputs)`` gives a call's ``OpCost``. Operators that do matrix products have an explicit formula in ``FORMULAS``,
called with the operator's arguments under the reference implementation's parameter names; every other operator moves
data only. Costs read shapes, dtypes and host-side values, never tensor data, so they also run on ``meta`` tensors.

Conventions:

- **FLOPs** count matrix products only, a multiply-add as two. Epilogues, fused norms, softmax and elementwise work
  are not counted.
- **Bytes** are those of the tensors as stored (packed E2M1, E4M3 / E8M0 scales with their swizzle padding, BF16 or
  FP32), each input read once and each output written once (of a cache, the rows read or written); intermediates are
  not counted.
- **Attention** whose valid keys are given by device tensors (lengths, masks) counts every key of the padded layout,
  as a dense kernel computes: reading the lengths would synchronize with the device.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from xinfer.ops.layout.mxfp8 import scale_elements

_PIPES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


@dataclass(frozen=True)
class OpCost:
    """The FLOPs, bytes read and written, and compute pipe of one operator call.

    Attributes:
        flops: floating-point operations on ``pipe``, a multiply-add counting two; 0 when ``pipe`` is ``"memory"``.
        read_bytes: bytes read: weights and their scales, activations, caches, tables.
        write_bytes: bytes written.
        pipe: the compute pipe ``flops`` run on: ``"bf16"``, ``"fp16"`` or ``"fp32"`` (by input dtype), ``"mxfp8"``,
            ``"nvfp4"``, or ``"memory"`` for data movement only.
    """

    flops: int
    read_bytes: int
    write_bytes: int
    pipe: str

    @property
    def bytes(self) -> int:
        """All bytes moved."""
        return self.read_bytes + self.write_bytes


def _bytes(*values) -> int:
    """Stored bytes of the tensors among ``values``, also inside tuples and lists; anything else counts 0."""
    total = 0
    for value in values:
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
        elif isinstance(value, (tuple, list)):
            total += _bytes(*value)
    return total


# ------------------------------------------------------------ dense products
def linear(x, weight, bias=None) -> OpCost:
    """``2·M·N·K`` with ``M`` the rows of ``x`` on the pipe of ``x.dtype``; writes ``[M, N]`` in that dtype."""
    n, k = weight.shape
    m = x.numel() // k
    return OpCost(2 * m * n * k, _bytes(x, weight, bias), m * n * x.element_size(), _PIPES[x.dtype])


def conv2d(x, weight, bias, stride) -> OpCost:
    """``2·B·H'·W'·O·C·kH·kW`` (no padding, one group) on the pipe of ``x.dtype``."""
    b, c, h, w = x.shape
    o, _, kh, kw = weight.shape
    pixels = b * ((h - kh) // stride + 1) * ((w - kw) // stride + 1)
    return OpCost(2 * pixels * o * c * kh * kw, _bytes(x, weight, bias), pixels * o * x.element_size(), _PIPES[x.dtype])


def chunk_gated_delta_rule(q, k, v, g, beta, *, chunk_size=64) -> OpCost:
    """The chunked form's products over the length padded to ``C = chunk_size``, on the pipe of ``q.dtype``.

    Per chunk and head: ``2C²(3Dk + 2Dv)`` within the chunk, ``6C·Dk·Dv`` with the state and ``(C−1)C(2C−1)/3``
    for the triangular solve.
    """
    b, length, h, dk = q.shape
    dv, c = v.shape[-1], chunk_size
    per_chunk = 2 * c * c * (3 * dk + 2 * dv) + 6 * c * dk * dv + (c - 1) * c * (2 * c - 1) // 3
    chunks = b * h * -(-length // c)
    return OpCost(chunks * per_chunk, _bytes(q, k, v, g, beta), b * length * h * dv * q.element_size(), _PIPES[q.dtype])


# ------------------------------------------------------------ attention
def _attention(q, pairs: int, read: int, written: int = 0) -> OpCost:
    """``4·D`` FLOPs per (query, key) pair and head (``QKᵀ`` and ``PV``); the output has ``q``'s shape."""
    return OpCost(4 * q.shape[-1] * pairs, read, _bytes(q) + written, _PIPES[q.dtype])


def causal_attention(q, k, v, *, scale=None) -> OpCost:
    """Exact causal work: ``L(L+1)/2`` pairs per row and head."""
    b, n, h, _ = q.shape
    return _attention(q, b * h * n * (n + 1) // 2, _bytes(q, k, v))


def action_attention(q, k, v, k_prompt, v_prompt, num_pinned, prompt_lengths=None, *, scale=None) -> OpCost:
    """Padded layout: every query sees all ``P + L`` keys (pinned rows and prompt lengths are device data)."""
    b, n, h, _ = q.shape
    read = _bytes(q, k, v, k_prompt, v_prompt, num_pinned, prompt_lengths)
    return _attention(q, b * h * n * (k_prompt.shape[2] + n), read)


def masked_joint_attention(q, k, v, k_prompt, v_prompt, mask, *, scale=None) -> OpCost:
    """Padded layout: every query sees all ``P + L`` keys; the mask is read."""
    b, n, h, _ = q.shape
    return _attention(q, b * h * n * (k_prompt.shape[2] + n), _bytes(q, k, v, k_prompt, v_prompt, mask))


def kvcache_attention(q, k, v, k_cache, v_cache, cache_start, cache_valid, *, scale=None) -> OpCost:
    """Padded layout: every query sees all ``S - L`` cache columns and the ``L`` own keys.

    Reads those cache columns and writes the own keys and values into both caches.
    """
    b, n, h, _ = q.shape
    s = k_cache.shape[1]
    cached = _bytes(k_cache, v_cache) * (s - n) // s
    return _attention(q, b * h * n * s, _bytes(q, k, v, cache_start, cache_valid) + cached, _bytes(k, v))


def prefix_attention(q, k, v, valid, *, scale=None) -> OpCost:
    """Padded layout: every query sees all ``S`` keys (the valid lengths are device data)."""
    b, s, h, _ = q.shape
    return _attention(q, b * h * s * s, _bytes(q, k, v, valid))


def varlen_attention(q, k, v, cu_seqlens, *, scale=None) -> OpCost:
    """Exact work: ``l²`` pairs per segment and head, from the host-side segment bounds."""
    bounds = cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else list(cu_seqlens)
    pairs = sum((end - start) ** 2 for start, end in zip(bounds[:-1], bounds[1:]))
    return _attention(q, q.shape[1] * pairs, _bytes(q, k, v))


# ------------------------------------------------------------ MXFP8 and NVFP4
def _gemm(pipe, a, a_scales, w, w_scales, bias, k, n_out, reads=()) -> OpCost:
    """``2·M·N·K`` of ``a [M, ·]`` by ``w [N, ·]ᵀ`` (block-scaled with scales, BF16 with None) writing BF16
    ``[M, n_out]``."""
    m, n = a.shape[0], w.shape[0]
    return OpCost(2 * m * n * k, _bytes(a, a_scales, w, w_scales, bias, *reads), 2 * m * n_out, pipe)


def mxfp8_gemm(a_q, a_scales, w_q, w_scales, bias=None, *, fast_accum=False) -> OpCost:
    """``2·M·N·K`` on the MXFP8 pipe; reads E4M3 operands and their E8M0 scales, writes BF16 ``[M, N]``."""
    return _gemm("mxfp8", a_q, a_scales, w_q, w_scales, bias, a_q.shape[1], w_q.shape[0])


def mxfp8_gemm_rope(a_q, a_scales, w_q, w_scales, bias, table, q_dim, k_dim, head_dim) -> OpCost:
    """As ``mxfp8_gemm``, also reading the rope table; the rotation in the epilogue is not counted."""
    return _gemm("mxfp8", a_q, a_scales, w_q, w_scales, bias, a_q.shape[1], w_q.shape[0], (table,))


def mxfp8_gemm_silu_mul(a_q, a_scales, w_q, w_scales, bias=None) -> OpCost:
    """As ``mxfp8_gemm``, writing BF16 ``[M, N/2]``."""
    return _gemm("mxfp8", a_q, a_scales, w_q, w_scales, bias, a_q.shape[1], w_q.shape[0] // 2)


def mxfp8_gemm_residual(a_q, a_scales, w_q, w_scales, residual, bias=None) -> OpCost:
    """As ``mxfp8_gemm``, also reading the residual."""
    return _gemm("mxfp8", a_q, a_scales, w_q, w_scales, bias, a_q.shape[1], w_q.shape[0], (residual,))


def mxfp8_gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, row_mask=None,
                        out_quant=False) -> OpCost:
    """MXFP8 product with a BF16 output or E4M3 values and padded E8M0 scales."""
    m, n = x[0].shape[0], weight[0].shape[0]
    cost = _gemm("mxfp8", *x, *weight, bias, x[0].shape[1], n, (residual, row_mask))
    return OpCost(cost.flops, cost.read_bytes, m * n + scale_elements(m, n), "mxfp8") if out_quant else cost


def mxfp8_gemm_glu_quantize(x, weight) -> OpCost:
    """Count both gate and up products and write one quantized output pair of half the width."""
    m, n = x[0].shape[0], weight[0].shape[0] // 2
    cost = _gemm("mxfp8", *x, *weight, None, x[0].shape[1], n)
    return OpCost(cost.flops, cost.read_bytes, m * n + scale_elements(m, n), "mxfp8")


def mxfp8_qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim) -> OpCost:
    """``2·M·N·K`` on the MXFP8 pipe, also reading the RoPE tables."""
    return _gemm("mxfp8", *x, *weight, bias, x[0].shape[1], weight[0].shape[0], (cos, sin))


def nvfp4_gemm(a_q, a_scales, w_q, w_scales, bias=None, *, alpha=1.0) -> OpCost:
    """``2·M·N·K`` with ``K = 2·a_q.shape[1]`` on the NVFP4 pipe; reads packed E2M1 operands and E4M3 scales."""
    return _gemm("nvfp4", a_q, a_scales, w_q, w_scales, bias, 2 * a_q.shape[1], w_q.shape[0])


# ------------------------------------------------------------ fused operators (xinfer/ops/reference/)
def gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, residual_gate=None, row_class=None,
                  row_scale=None) -> OpCost:
    """``2·M·N·K`` on the pipe of ``x.dtype``, also reading the residual, gate rows, row classes and row scales."""
    return _gemm(_PIPES[x.dtype], x, None, weight, None, bias, x.shape[1], weight.shape[0],
                 (residual, residual_gate, row_class, row_scale))


def gemm_glu_epilogue(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6) -> OpCost:
    """``2·M·N·K`` over the gate and up rows on the pipe of ``x.dtype``, writing ``[M, N/2]``; also reads the input
    norm's modulation and row classes."""
    return _gemm(_PIPES[x.dtype], x, None, weight, None, bias, x.shape[1], weight.shape[0] // 2,
                 (modulation, row_class))


def gemm_add_rms_norm(x, projection, residual, weight, eps, bias=None) -> OpCost:
    """``2·M·N·K`` on the pipe of ``x.dtype``; reads ``x``, the projection, the bias, the residual and the norm weight,
    writes the sum and its norm."""
    m, n = x.shape[0], projection.shape[0]
    return OpCost(2 * m * n * x.shape[1], _bytes(x, projection, bias, residual, weight), 2 * 2 * m * n,
                  _PIPES[x.dtype])


def mlp_add_rms_norm(x, gate_up, down, residual, weight, eps) -> OpCost:
    """``2·M·K·2I`` (gate and up) plus ``2·M·I·N`` (down) on the pipe of ``x.dtype``; reads the weights, ``x`` and
    the residual, writes the sum and its norm. The intermediate is not counted as traffic (fused: in L2)."""
    m, i, n = x.shape[0], down.shape[1], down.shape[0]
    flops = 2 * m * x.shape[1] * gate_up.shape[0] + 2 * m * i * n
    return OpCost(flops, _bytes(x, gate_up, down, residual, weight), 2 * 2 * m * n, _PIPES[x.dtype])


def qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim, *, qk_order="standard") -> OpCost:
    """``2·M·N·K`` on the pipe of ``x.dtype``, also reading the RoPE tables."""
    return _gemm(_PIPES[x.dtype], x, None, weight, None, bias, x.shape[1], weight.shape[0], (cos, sin))


def qkv_rope_kvcache(x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads,
                     head_dim) -> OpCost:
    """As ``qkv_rope_epilogue``: of the ``N`` columns written per row, K and V go to the new cache rows (the caches
    are not read)."""
    return _gemm(_PIPES[x.dtype], x, None, weight, None, bias, x.shape[1], weight.shape[0], (cos, sin))


def prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, *, flash_mask=None, scale=None) -> OpCost:
    """Padded layout: every query sees cache rows ``[0, new_pos + M)``, the prefix padded to ``new_pos`` and the own
    keys (prefix lengths and ``flash_mask`` are device data); reads those rows of both caches."""
    b, m, h, _ = q.shape
    rows = int(new_pos) + m
    cached = _bytes(k_cache, v_cache) * rows // k_cache.shape[1]
    return _attention(q, b * h * m * rows, _bytes(q, prefix_len, flash_mask) + cached)


def flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None,
                        pinned_values=None) -> OpCost:
    """Read the action stream and its projection weights, and write one FP32 velocity step."""
    read = _bytes(x, h, weight, dof_mask, padding_velocity, pinned, pinned_values)
    return OpCost(2 * h.numel() * weight.shape[0], read, _bytes(x), _PIPES[torch.promote_types(h.dtype, weight.dtype)])


# Operator name -> formula, for the operators that do matrix products.
FORMULAS: dict[str, Callable[..., OpCost]] = {formula.__name__: formula for formula in (
    linear, conv2d, chunk_gated_delta_rule, causal_attention, action_attention, masked_joint_attention,
    kvcache_attention, prefix_attention, varlen_attention, mxfp8_gemm, mxfp8_gemm_rope, mxfp8_gemm_silu_mul,
    mxfp8_gemm_residual, mxfp8_gemm_epilogue, mxfp8_gemm_glu_quantize, mxfp8_qkv_rope_epilogue,
    nvfp4_gemm, gemm_epilogue, gemm_glu_epilogue, gemm_add_rms_norm, mlp_add_rms_norm, qkv_rope_epilogue,
    qkv_rope_kvcache, prefix_kvcache_attention, flow_velocity_euler)}


def op_cost(name: str, args: tuple, kwargs: Mapping[str, object], outputs) -> OpCost:
    """The ``OpCost`` of one call of operator ``name``.

    Args:
        name: the operator name.
        args: the call's positional arguments.
        kwargs: the call's keyword arguments.
        outputs: what the call returned, a tensor or a tuple of tensors; only the generic cost reads it.

    Returns:
        ``FORMULAS[name](*args, **kwargs)`` for an operator with a formula. Otherwise data movement: 0 FLOPs on the
        ``"memory"`` pipe, the stored bytes of every tensor argument read and of every output tensor written.
    """
    formula = FORMULAS.get(name)
    if formula is not None:
        return formula(*args, **kwargs)
    return OpCost(0, _bytes(*args, *kwargs.values()), _bytes(outputs), "memory")
