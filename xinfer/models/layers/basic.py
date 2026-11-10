"""Linear and normalization layers bound to the active op table.

Parameters are created on the "meta" device; a model's weight mapping
replaces them with checkpoint tensors in their inference dtype. The operator
each layer calls is fixed when the layer is constructed.
"""

from __future__ import annotations

import torch
from torch import nn

from xinfer.ops import active_ops
from xinfer.models.layers.quantization import GatedLayout, QKVLayout, active_quant, make_method


def _meta_param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, device="meta"), requires_grad=False)


class Linear(nn.Module):
    """A linear layer whose method (unquantized, MXFP8, ...) comes from the active QuantConfig.

    ``group`` names the module group the config matches against (for example
    ``"vision.blocks"``); layers without a group use the config default.
    ``layout`` describes the structure of the output rows for fused forms
    (``QKVLinear``, ``GatedLinear``).

    ``forward(x, norm=..., input_act=..., residual=...)`` computes
    ``residual + linear(input_act(norm(x)))``. Unquantized layers run exactly
    that composition; a quantized method may fuse the norm or activation into
    its input quantization and the residual add into the GEMM epilogue. The
    model chooses which forms to request.

    Args:
        in_features: Input width ``K``.
        out_features: Output width ``N``.
        bias: Whether the layer has a bias ``[N]``.
        group: Module group matched by the active ``QuantConfig`` (``None``: the config default).
        layout: ``QKVLayout`` / ``GatedLayout`` of fused outputs, or ``None``.

    Attributes:
        weight: ``[N, K]``, on the "meta" device until the weight mapping loads it (a quantized method may
            replace it in ``process_weights``).
        bias: ``[N]`` or ``None``.
        quant: The ``QuantMethod`` chosen for ``group`` at construction.

    Raises:
        ValueError: The chosen method does not support a ``K → N`` layer (e.g. MXFP8 with ``K`` not a
            multiple of 32).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, group: str | None = None,
                 layout=None):
        super().__init__()
        self.in_features, self.out_features, self.group, self.layout = in_features, out_features, group, layout
        self.weight = _meta_param(out_features, in_features)
        self.bias = _meta_param(out_features) if bias else None
        config = active_quant()
        self.quant = make_method(config.method_for(group), config)
        if not self.quant.supports(in_features, out_features):
            raise ValueError(f"{self.quant.name} does not support a {in_features}->{out_features} linear "
                             f"(group {group!r})")

    def forward(self, x: torch.Tensor, *, norm: "RMSNorm | None" = None, input_act: str | None = None,
                residual: torch.Tensor | None = None) -> torch.Tensor:
        """``residual + linear(input_act(norm(x)))``; without keywords, the plain ``x W^T + b``.

        Args:
            x: ``[..., K]``, or ``[..., 2K]`` (``[gate | up]``) with ``input_act="silu_and_mul"``.
            norm: ``RMSNorm`` over ``x`` applied first; a quantized method may fuse it into the input
                quantization.
            input_act: ``"silu_and_mul"`` computes ``silu(gate) * up`` of the two halves of ``x``; a quantized
                method may fuse it into the input quantization.
            residual: ``[..., N]`` added to the output; a quantized method may add it in the GEMM epilogue.

        Returns:
            ``[..., N]``.
        """
        if norm is None and input_act is None and residual is None:
            return self.quant.apply(self, x)
        return self.quant.apply_fused(self, x, norm=norm, input_act=input_act, residual=residual)


class QKVLinear(Linear):
    """Fused q/k/v projection that returns the rotated ``q`` and ``k`` heads and the ``v`` heads.

    The weight is ``[(H + 2·KV)·D, hidden]`` with output rows ``[q | k | v]`` (a fused method may reorder
    them for its epilogue; such a layer cannot be called through the plain form).

    Args:
        hidden: Input width.
        num_heads: Query heads ``H``.
        num_kv_heads: K/V heads ``KV``.
        head_dim: Head dimension ``D``.
        bias: Whether the projection has a bias.
        group: Quantization group.
    """

    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int, head_dim: int, bias: bool = True,
                 group: str | None = None):
        layout = QKVLayout(num_heads, num_kv_heads, head_dim)
        super().__init__(hidden, layout.q_dim + 2 * layout.k_dim, bias=bias, group=group, layout=layout)

    def forward(self, x, rope, *, norm: "RMSNorm | None" = None):
        """``x [..., hidden]`` → ``q [..., H, D]``, ``k``/``v [..., KV, D]``; ``rope`` is a ``RopeTables``.

        Args:
            x: ``[..., hidden]``; the leading dims are those ``rope`` was built for (``[B, L]``).
            rope: ``RopeTables`` of these rows; ``q`` and ``k`` are rotated with it (``v`` is not).
            norm: ``RMSNorm`` applied to ``x`` first; a quantized method may fuse it into the input
                quantization.

        Returns:
            ``(q, k, v)``: ``q`` ``[..., H, D]``, ``k`` and ``v`` ``[..., KV, D]``.
        """
        return self.quant.apply_qkv(self, x, rope, norm=norm)


class GatedLinear(Linear):
    """Fused ``[gate | up]`` projection that returns ``silu(gate) * up``.

    Args:
        hidden: Input width.
        intermediate: Width of each half; the weight is ``[2·intermediate, hidden]``.
        bias: Whether the projection has a bias.
        group: Quantization group.
    """

    def __init__(self, hidden: int, intermediate: int, bias: bool = False, group: str | None = None):
        super().__init__(hidden, 2 * intermediate, bias=bias, group=group, layout=GatedLayout(intermediate))

    def forward(self, x, *, norm: "RMSNorm | None" = None):
        """Return ``silu(gate) * up`` of ``[gate | up] = norm(x) W^T + b``.

        Args:
            x: ``[..., hidden]``.
            norm: ``RMSNorm`` applied to ``x`` first; a quantized method may fuse it into the input
                quantization (and the activation into the GEMM epilogue).

        Returns:
            ``[..., intermediate]``.
        """
        return self.quant.apply_gated(self, x, norm=norm)


class RMSNorm(nn.Module):
    """RMSNorm through the active op table's ``rms_norm`` (reference: normalize and scale in FP32, round once).

    Args:
        dim: Normalized (last) dimension.
        eps: Variance epsilon.

    Attributes:
        weight: ``[dim]``; its dtype comes from the weight mapping (FP32 or BF16).
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = _meta_param(dim)
        self._rms_norm = active_ops().rms_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x [..., dim]``; returns the same shape and dtype."""
        return self._rms_norm(x, self.weight, self.eps)


class GatedMLP(nn.Module):
    """``down(silu(gate(x)) * up(x))`` with gate and up fused in one GEMM.

    ``intermediate`` is the width of the fused weight's gate (and up) half;
    checkpoints that pad it for GEMM alignment keep the padded width here so
    the GEMM shapes match the source model.

    ``act_fused_into`` says where a quantized method may compute the gated
    activation: in the gate/up GEMM's epilogue (``"gate_up"``) or in the
    quantization of the down projection's input (``"down"``). Both are the
    same composition when unquantized.

    Args:
        hidden: Input and output width.
        intermediate: Width of the gated activation (one half of the fused weight).
        bias: Whether the projections have biases.
        group: Quantization group of both projections.
        act_fused_into: ``"gate_up"`` or ``"down"`` (see above).

    Attributes:
        gate_up: ``GatedLinear`` (``"gate_up"``) or plain ``Linear`` (``"down"``), ``hidden → 2·intermediate``.
        down: ``Linear`` ``intermediate → hidden``.

    Raises:
        ValueError: ``act_fused_into`` is neither ``"gate_up"`` nor ``"down"``.
    """

    def __init__(self, hidden: int, intermediate: int, bias: bool = False, group: str | None = None,
                 act_fused_into: str = "gate_up"):
        super().__init__()
        if act_fused_into not in ("gate_up", "down"):
            raise ValueError("act_fused_into must be 'gate_up' or 'down'")
        self.intermediate, self.act_fused_into = intermediate, act_fused_into
        if act_fused_into == "gate_up":
            self.gate_up = GatedLinear(hidden, intermediate, bias=bias, group=group)
        else:
            self.gate_up = Linear(hidden, 2 * intermediate, bias=bias, group=group)
        self.down = Linear(intermediate, hidden, bias=bias, group=group)

    def forward(self, x: torch.Tensor, *, norm: RMSNorm | None = None,
                residual: torch.Tensor | None = None) -> torch.Tensor:
        """``residual + down(act(gate_up(norm(x))))``.

        Args:
            x: ``[..., hidden]``.
            norm: ``RMSNorm`` applied to ``x`` first (fusable into the gate/up input quantization).
            residual: ``[..., hidden]`` added to the output (fusable into the down GEMM epilogue).

        Returns:
            ``[..., hidden]``.
        """
        if self.act_fused_into == "gate_up":
            return self.down(self.gate_up(x, norm=norm), residual=residual)
        return self.down(self.gate_up(x, norm=norm), input_act="silu_and_mul", residual=residual)
