"""MXFP8 linear layers: E4M3 weights and activations with E8M0 block scales.

Weights are quantized once after loading, from their BF16 values, with the
same operator that quantizes activations; the BF16 weight is then dropped.
``share_scale_128`` uses one scale per 128 input columns ("1x128 blocks"), as
the Wall-X MXFP8 service did.

The fusable forms follow the Wall-X 4B MXFP8 service, so that a model built
on them reproduces it:

| layer call | activation quantization | GEMM |
|---|---|---|
| ``linear(x)`` | ``mxfp8_quantize`` | ``mxfp8_gemm`` (bias in the epilogue) |
| ``linear(x, norm=n)`` | ``mxfp8_rmsnorm_quantize`` (no BF16 rounding of ``n(x)``) | as above |
| ``linear(gate_up, input_act="silu_and_mul")`` | ``mxfp8_silu_mul_quantize`` | as above |
| ``linear(x, residual=r)`` | as above | ``mxfp8_gemm_residual`` (``r`` added to the FP32 accumulator) |
| ``QKVLinear(x, rope)`` | as above | ``mxfp8_gemm_rope`` (bias and rotary on the accumulator) |
| ``GatedLinear(x)`` | as above | ``mxfp8_gemm_silu_mul`` (``silu(gate) * up`` on the accumulator) |

``QKVLinear`` and ``GatedLinear`` weights are stored with their rows in the
order those epilogues read (``rope_pair_permutation``, ``gate_up_interleave``).

``fast_accum`` (``mxfp8_gemm`` only): ``"auto"`` enables the GEMM's fast FP8
accumulation for ``rows <= 16 or out_features >= 8192`` (the Wall-X service
rule, wall-x ``mxfp8_1d1d_128bs.py`` ``select_fast_accum``); ``True``/``False``
force it.
"""

from __future__ import annotations

import torch

from xinfer.ops import active_ops
from xinfer.ops import mxfp8 as fmt
from xinfer.quantization.base import METHODS, GatedLayout, QKVLayout, QuantMethod


class MXFP8(QuantMethod):
    """MXFP8 linear method: E4M3 weights and activations with E8M0 block scales, BF16 output.

    Binds the seven ``mxfp8_*`` operators of the active op table at construction. After
    ``process_weights_after_loading`` a layer holds ``weight_q``/``weight_scales`` buffers instead of
    ``weight``, and ``row_order`` (the permutation applied to its rows for a fused epilogue, or None).

    Args:
        share_scale_128: one scale per 128 input columns (the Wall-X "1x128" recipe) instead of per 32.
        fast_accum: ``"auto"`` (fast FP8 accumulation for ``rows <= 16 or out_features >= 8192``), True or
            False; applies to the plain GEMM only.

    Attributes:
        share_scale_128: as passed.
        fast_accum: as passed.

    Raises:
        ValueError: ``fast_accum`` is not True, False or ``"auto"``.
    """

    name = "mxfp8"

    def __init__(self, share_scale_128: bool = True, fast_accum: bool | str = "auto"):
        super().__init__()
        if fast_accum not in (True, False, "auto"):
            raise ValueError("fast_accum must be True, False or 'auto'")
        self.share_scale_128, self.fast_accum = share_scale_128, fast_accum
        ops = active_ops()
        self._quantize, self._gemm = ops.mxfp8_quantize, ops.mxfp8_gemm
        self._rmsnorm_quantize, self._silu_mul_quantize = ops.mxfp8_rmsnorm_quantize, ops.mxfp8_silu_mul_quantize
        self._gemm_rope, self._gemm_silu_mul = ops.mxfp8_gemm_rope, ops.mxfp8_gemm_silu_mul
        self._gemm_residual = ops.mxfp8_gemm_residual

    def supports(self, in_features: int, out_features: int) -> bool:
        """True when ``in_features`` is a multiple of 32 (the scale block)."""
        return in_features % fmt.BLOCK == 0

    # ------------------------------------------------------------ weights
    def process_weights_after_loading(self, layer) -> None:
        """Quantize the layer's BF16 weight to MXFP8 and drop the BF16 weight.

        ``QKVLinear`` and ``GatedLinear`` weights (and biases) are first reordered to ``rope_pair_permutation``
        / ``gate_up_interleave`` order. Registers non-persistent buffers ``weight_q`` (``float8_e4m3fn
        [out_features, in_features]``) and ``weight_scales`` (swizzled uint8), removes the ``weight``
        parameter and sets ``layer.row_order`` (int64 permutation or None). The bias stays BF16.

        Args:
            layer: the linear layer with its BF16 checkpoint weight loaded.

        Raises:
            ValueError: the weight or the bias is not BF16.
        """
        weight, bias = layer.weight, layer.bias
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"MXFP8 weights are quantized from BF16, got {weight.dtype}")
        if bias is not None and bias.dtype != torch.bfloat16:
            raise ValueError("MXFP8 GEMM bias must be BF16")
        order = self._row_order(layer)
        if order is not None:  # rows in the order the fused epilogue reads them
            order = order.to(weight.device)
            weight = weight[order]
            if bias is not None:
                layer._parameters["bias"] = torch.nn.Parameter(bias[order].contiguous(), requires_grad=False)
        layer.row_order = order
        q, scales = self._quantize(weight.contiguous(), share_scale_128=self.share_scale_128)
        del layer._parameters["weight"]
        layer.register_buffer("weight_q", q, persistent=False)
        layer.register_buffer("weight_scales", scales, persistent=False)

    @staticmethod
    def _row_order(layer):
        """The row permutation the layer's fused epilogue needs (int64 ``[out_features]``), or None."""
        layout = getattr(layer, "layout", None)
        if isinstance(layout, QKVLayout):
            return fmt.rope_pair_permutation(layout.q_dim, layout.k_dim, layout.head_dim, layer.out_features)
        if isinstance(layout, GatedLayout):
            return fmt.gate_up_interleave(layout.intermediate)
        return None

    # ------------------------------------------------------------ forward
    def _input(self, x, norm=None, input_act=None):
        """Quantize the (normalized / activated) 2D input: ``(q, scales)``.

        Args:
            x: BF16 ``[..., K]`` (``[..., 2K]`` with ``input_act``), flattened to 2D rows.
            norm: optional RMSNorm module fused into the quantization (``mxfp8_rmsnorm_quantize``).
            input_act: None or ``"silu_and_mul"`` (``mxfp8_silu_mul_quantize``).

        Returns:
            ``(q, scales)``: ``float8_e4m3fn [M, K]`` with ``M`` the product of the leading dims, and the
            swizzled scales.

        Raises:
            ValueError: both ``norm`` and ``input_act`` are given, ``x`` is not BF16, or ``input_act`` is
                unknown.
        """
        if norm is not None and input_act is not None:
            raise ValueError("norm and input_act cannot both be fused into one quantization")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"MXFP8 linear expects BF16 activations, got {x.dtype}")
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        share = self.share_scale_128
        if norm is not None:
            return self._rmsnorm_quantize(x2, norm.weight, norm.eps, share_scale_128=share)
        if input_act is not None:
            if input_act != "silu_and_mul":
                raise ValueError(f"unknown input activation {input_act!r}")
            return self._silu_mul_quantize(x2, share_scale_128=share)
        return self._quantize(x2, share_scale_128=share)

    def _plain_gemm(self, layer, q, scales):
        """``mxfp8_gemm`` of quantized activations with the layer's weight and bias.

        Resolves ``fast_accum="auto"`` from the row count and ``out_features``.

        Args:
            layer: the processed linear layer.
            q: ``float8_e4m3fn [M, K]`` activations.
            scales: their swizzled scales.

        Returns:
            BF16 ``[M, out_features]``.

        Raises:
            RuntimeError: the layer's weight rows are reordered for a fused epilogue.
        """
        if layer.row_order is not None:
            raise RuntimeError("this layer's weight rows are reordered for a fused epilogue; "
                               "call it through its fused form")
        fast = self.fast_accum
        if fast == "auto":  # rows is a shape, so this is fixed per captured graph
            fast = q.shape[0] <= 16 or layer.out_features >= 8192
        return self._gemm(q, scales, layer.weight_q, layer.weight_scales, layer.bias, fast_accum=fast)

    def apply(self, layer, x):
        """``mxfp8_quantize`` then ``mxfp8_gemm``: BF16 ``[..., in_features]`` → BF16 ``[..., out_features]``."""
        return self.apply_fused(layer, x)

    def apply_fused(self, layer, x, *, norm=None, input_act=None, residual=None):
        """``residual + linear(input_act(norm(x)))`` with the norm or activation fused into the quantization.

        Without ``residual`` the GEMM is ``mxfp8_gemm``; with it, ``mxfp8_gemm_residual`` adds the residual to
        the FP32 accumulator.

        Args:
            layer: the processed linear layer.
            x: BF16 ``[..., in_features]`` (``[..., 2 * in_features]`` with ``input_act="silu_and_mul"``).
            norm: optional RMSNorm module; cannot be combined with ``input_act``.
            input_act: None or ``"silu_and_mul"``.
            residual: optional tensor with ``M * out_features`` elements (``[..., out_features]``).

        Returns:
            BF16 ``[..., out_features]``.

        Raises:
            ValueError: see ``_input``.
            RuntimeError: the layer's weight rows are reordered for a fused epilogue.
        """
        lead = x.shape[:-1]
        q, scales = self._input(x, norm, input_act)
        if residual is None:
            out = self._plain_gemm(layer, q, scales)
        else:
            if layer.row_order is not None:
                raise RuntimeError("fused residual on a reordered weight")
            r = residual.reshape(-1, layer.out_features)
            out = self._gemm_residual(q, scales, layer.weight_q, layer.weight_scales, r, layer.bias)
        return out.reshape(*lead, out.shape[-1])

    def apply_qkv(self, layer, x, rope, *, norm=None):
        """Fused QKV projection with bias and rotary embedding on the FP32 accumulator (``mxfp8_gemm_rope``).

        Uses ``rope.packed()``, the unrounded FP32 cos/sin, unlike the eager rotation.

        Args:
            layer: a processed ``QKVLinear``.
            x: BF16 ``[..., hidden]`` input.
            rope: the ``RopeTables`` of this forward, one table row per row of ``x``.
            norm: optional RMSNorm module fused into the quantization.

        Returns:
            ``(q, k, v)`` BF16: rotated ``q [..., H, D]``, rotated ``k [..., KV, D]`` and ``v [..., KV, D]``.
        """
        layout = layer.layout
        lead = x.shape[:-1]
        q, scales = self._input(x, norm)
        out = self._gemm_rope(q, scales, layer.weight_q, layer.weight_scales, layer.bias, rope.packed(),
                              layout.q_dim, layout.k_dim, layout.head_dim)
        q_states, k_states, v_states = out.split([layout.q_dim, layout.k_dim, layout.k_dim], dim=-1)
        return (q_states.reshape(*lead, layout.num_heads, layout.head_dim),
                k_states.reshape(*lead, layout.num_kv_heads, layout.head_dim),
                v_states.reshape(*lead, layout.num_kv_heads, layout.head_dim))

    def apply_gated(self, layer, x, *, norm=None):
        """``silu(gate) * up`` on the FP32 accumulator of the fused gate/up GEMM (``mxfp8_gemm_silu_mul``).

        Args:
            layer: a processed ``GatedLinear``.
            x: BF16 ``[..., hidden]`` input.
            norm: optional RMSNorm module fused into the quantization.

        Returns:
            BF16 ``[..., intermediate]``.
        """
        lead = x.shape[:-1]
        q, scales = self._input(x, norm)
        out = self._gemm_silu_mul(q, scales, layer.weight_q, layer.weight_scales, layer.bias)
        return out.reshape(*lead, out.shape[-1])


METHODS["mxfp8"] = MXFP8
