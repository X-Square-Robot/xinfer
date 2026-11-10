"""NVFP4 linear layers: E2M1 weights and activations with E4M3 block scales.

Two methods, the two Wall-X NVFP4 recipes of the embodied-serving NVFP4
service:

- ``nvfp4`` (``nvfp4_1d1d_16bs``): weight block scales from the block amax,
  as for activations; the bias is added in the GEMM epilogue.
- ``nvfp4_mse`` (``nvfp4_mse_1d1d_16bs``): each weight block's scale is chosen
  by an MSE search under a per-tensor global scale, which the GEMM applies to
  the FP32 accumulator; the bias is then added to the BF16 output, as the
  service did.

They are two methods rather than one with an option because ``QuantConfig``
options apply per method, and the Wall-X recipe uses both in one model (ViT
``nvfp4``, expert 0 ``nvfp4_mse``).

Activations are always quantized with amax scales. The fusable forms follow
the service:

| layer call | activation quantization |
|---|---|
| ``linear(x)`` | ``nvfp4_quantize`` |
| ``linear(x, norm=n)`` | ``nvfp4_rmsnorm_quantize`` (``n(x)`` rounded to BF16 first, as the Wall-X kernel) |
| ``linear(gate_up, input_act="silu_and_mul")`` | ``nvfp4_silu_mul_quantize`` (FP32, no BF16 rounding) |

Every other form runs the plain composition on the BF16 GEMM output: the
residual add, the rotary embedding of ``QKVLinear`` and ``silu_and_mul`` of
``GatedLinear``. Weight rows keep their checkpoint order.
"""

from __future__ import annotations

import torch

from xinfer.ops import active_ops
from xinfer.ops.layout import nvfp4 as fmt
from xinfer.ops.reference.nvfp4 import E2M1_MAX, E4M3_MAX
from xinfer.models.layers.quantization.base import METHODS, QuantMethod



def global_scale(weight: torch.Tensor) -> float:
    """Per-tensor scale of a weight with MSE block scales: ``amax / (448 · 6)`` rounded to FP32, 1.0 if all zero.

    The quotient is taken in double precision and then rounded to FP32, as the Wall-X recipe computes it on the
    host. Reads the amax back to the host.
    """
    amax = float(weight.detach().float().abs().amax())
    if amax == 0.0:
        return 1.0
    return float(torch.tensor(amax / (E4M3_MAX * E2M1_MAX), dtype=torch.float32))


class NVFP4(QuantMethod):
    """NVFP4 linear method with amax weight scales: E2M1 weights and activations, BF16 output.

    Binds the four ``nvfp4_*`` operators of the active op table at construction. After
    ``process_weights_after_loading`` a layer holds ``weight_q``/``weight_scales`` buffers instead of ``weight``,
    and ``weight_global_scale`` (a float).

    Attributes:
        mse_weight_scales: weight block scales by MSE search with a global scale (``nvfp4_mse``) instead of amax.
    """

    name = "nvfp4"
    mse_weight_scales = False

    def __init__(self):
        super().__init__()
        ops = active_ops()
        self._quantize, self._gemm = ops.nvfp4_quantize, ops.nvfp4_gemm
        self._rmsnorm_quantize, self._silu_mul_quantize = ops.nvfp4_rmsnorm_quantize, ops.nvfp4_silu_mul_quantize

    def supports(self, in_features: int, out_features: int) -> bool:
        """True when both widths are multiples of 16 (the scale block, and the Wall-X GEMMs' N granularity)."""
        return in_features % fmt.BLOCK == 0 and out_features % fmt.BLOCK == 0

    def process_weights_after_loading(self, layer) -> None:
        """Quantize the layer's BF16 weight to NVFP4 and drop the BF16 weight.

        Registers non-persistent buffers ``weight_q`` (uint8 ``[out_features, in_features / 2]``) and
        ``weight_scales`` (swizzled ``float8_e4m3fn``), removes the ``weight`` parameter and sets
        ``layer.weight_global_scale`` (1.0, or ``global_scale`` with MSE scales, which reads the weight's
        amax back to the host). The bias stays BF16.

        Args:
            layer: the linear layer with its BF16 checkpoint weight loaded.

        Raises:
            ValueError: the weight or the bias is not BF16.
        """
        weight, bias = layer.weight, layer.bias
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"NVFP4 weights are quantized from BF16, got {weight.dtype}")
        if bias is not None and bias.dtype != torch.bfloat16:
            raise ValueError("NVFP4 linear bias must be BF16")
        scale = global_scale(weight) if self.mse_weight_scales else 1.0
        q, scales = self._quantize(weight.contiguous(), global_scale=scale, mse=self.mse_weight_scales)
        del layer._parameters["weight"]
        layer.register_buffer("weight_q", q, persistent=False)
        layer.register_buffer("weight_scales", scales, persistent=False)
        layer.weight_global_scale = scale

    def _input(self, x, norm=None, input_act=None):
        """Quantize the (normalized / activated) 2D input: ``(q, scales)``.

        Args:
            x: BF16 ``[..., K]`` (``[..., 2K]`` with ``input_act``), flattened to 2D rows.
            norm: optional RMSNorm module fused into the quantization (``nvfp4_rmsnorm_quantize``).
            input_act: None or ``"silu_and_mul"`` (``nvfp4_silu_mul_quantize``).

        Returns:
            ``(q, scales)``: packed E2M1 uint8 ``[M, K/2]`` with ``M`` the product of the leading dims, and the
            swizzled scales.

        Raises:
            ValueError: both ``norm`` and ``input_act`` are given, ``x`` is not BF16, or ``input_act`` is
                unknown.
        """
        if norm is not None and input_act is not None:
            raise ValueError("norm and input_act cannot both be fused into one quantization")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"NVFP4 linear expects BF16 activations, got {x.dtype}")
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        if norm is not None:
            return self._rmsnorm_quantize(x2, norm.weight, norm.eps)
        if input_act is not None:
            if input_act != "silu_and_mul":
                raise ValueError(f"unknown input activation {input_act!r}")
            return self._silu_mul_quantize(x2)
        return self._quantize(x2)

    def apply(self, layer, x):
        """``nvfp4_quantize`` then ``nvfp4_gemm``: BF16 ``[..., in_features]`` → BF16 ``[..., out_features]``."""
        return self.apply_fused(layer, x)

    def apply_fused(self, layer, x, *, norm=None, input_act=None, residual=None):
        """``residual + linear(input_act(norm(x)))`` with the norm or activation fused into the quantization.

        The GEMM applies the weight's global scale to the FP32 accumulator. The bias is added there too with amax
        weight scales, and to the BF16 GEMM output with MSE weight scales; the residual is added to the BF16
        output.

        Args:
            layer: the processed linear layer.
            x: BF16 ``[..., in_features]`` (``[..., 2 * in_features]`` with ``input_act="silu_and_mul"``).
            norm: optional RMSNorm module; cannot be combined with ``input_act``.
            input_act: None or ``"silu_and_mul"``.
            residual: optional ``[..., out_features]``.

        Returns:
            BF16 ``[..., out_features]``.

        Raises:
            ValueError: see ``_input``.
        """
        lead = x.shape[:-1]
        q, scales = self._input(x, norm, input_act)
        epilogue_bias = None if self.mse_weight_scales else layer.bias
        out = self._gemm(q, scales, layer.weight_q, layer.weight_scales, epilogue_bias,
                         alpha=layer.weight_global_scale)
        if layer.bias is not None and epilogue_bias is None:
            out = out + layer.bias
        out = out.reshape(*lead, out.shape[-1])
        return out if residual is None else residual + out


class NVFP4MSE(NVFP4):
    """NVFP4 with MSE weight block scales under a per-tensor global scale (``nvfp4_mse``); see ``NVFP4``."""

    name = "nvfp4_mse"
    mse_weight_scales = True


METHODS["nvfp4"] = NVFP4
METHODS["nvfp4_mse"] = NVFP4MSE
