"""Quantization config, method interface and the unquantized method."""

from __future__ import annotations

import contextlib
import contextvars
import importlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from xinfer.ops import active_ops


@dataclass(frozen=True)
class QuantConfig:
    """Method per module group; a group inherits from its dot-separated parents.

    ``QuantConfig(groups={"vision.blocks": "mxfp8", "prefill": "mxfp8"})``
    quantizes those groups and leaves everything else to ``default``.

    Attributes:
        groups: module group → method name (``"bf16"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp4_mse"``).
        default: method for layers whose group (and its parents) is not in ``groups``.
        options: method name → keyword arguments for that method's constructor.
    """

    groups: Mapping[str, str] = field(default_factory=dict)
    default: str = "bf16"
    options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # method -> kwargs

    def method_for(self, group: str | None) -> str:
        """The method name for a layer group.

        Args:
            group: dot-separated group name (``"vision.blocks.0"``), or None.

        Returns:
            The method of the longest dot-separated prefix of ``group`` found in ``groups``, else ``default``.
        """
        name = group or ""
        while name:
            if name in self.groups:
                return self.groups[name]
            name = name.rpartition(".")[0]
        return self.default

    def describe(self) -> dict:
        """The config as plain dicts (``groups``, ``default``, ``options``), for metadata and logs."""
        return dict(groups=dict(self.groups), default=self.default, options={k: dict(v) for k, v in
                                                                              self.options.items()})


@dataclass(frozen=True)
class QKVLayout:
    """Output rows of a fused QKV projection: ``[q heads | k heads | v heads]``.

    Attributes:
        num_heads: query heads ``H``.
        num_kv_heads: key/value heads ``KV``.
        head_dim: head dimension ``D``.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int

    @property
    def q_dim(self) -> int:
        """Width of the q rows, ``num_heads * head_dim``."""
        return self.num_heads * self.head_dim

    @property
    def k_dim(self) -> int:
        """Width of the k rows (and of the v rows), ``num_kv_heads * head_dim``."""
        return self.num_kv_heads * self.head_dim


@dataclass(frozen=True)
class GatedLayout:
    """Output rows of a fused gated projection: ``[gate | up]``, each ``intermediate`` wide.

    Attributes:
        intermediate: width of each half.
    """

    intermediate: int


class QuantMethod:
    """How one linear layer stores its weight and computes ``x @ W.T + b``.

    ``apply`` is the plain product. The other entry points are the fusable
    forms layers call; their defaults compose unfused operators, which is the
    eager numerics. A method may override them to fuse the input transform
    into its activation quantization and the output transform into the GEMM
    epilogue, which skips intermediate roundings (see the mxfp8 method).

    ``layer`` arguments are ``xinfer.models.layers.basic.Linear`` modules (``weight``, ``bias``, ``out_features``,
    ``layout``). Methods bind their operators from the active op table when constructed, so they must be
    created inside ``xinfer.ops.use_ops``.

    Attributes:
        name: method name used in ``QuantConfig`` and in the ``process_weights`` counts.
    """

    name = "abstract"

    def __init__(self):
        self._silu_and_mul = active_ops().silu_and_mul

    def supports(self, in_features: int, out_features: int) -> bool:
        """Whether this method can serve an ``in_features`` → ``out_features`` linear layer (base: always)."""
        return True

    def process_weights_after_loading(self, layer) -> None:
        """Convert the loaded weights in place (runs once, before sharing or capture).

        The base implementation keeps the weights as loaded.

        Args:
            layer: the linear layer, with its checkpoint weights loaded.
        """

    def apply(self, layer, x: torch.Tensor) -> torch.Tensor:
        """The plain product ``x @ W.T + b``.

        Args:
            layer: the linear layer.
            x: ``[..., in_features]`` input.

        Returns:
            ``[..., out_features]``.

        Raises:
            NotImplementedError: in the abstract base; subclasses implement it.
        """
        raise NotImplementedError

    def apply_fused(self, layer, x, *, norm=None, input_act: str | None = None, residual=None) -> torch.Tensor:
        """``residual + linear(input_act(norm(x)))``.

        ``norm`` is an RMSNorm module; ``input_act="silu_and_mul"`` takes
        ``x = [gate | up]`` and multiplies ``silu(gate) * up`` first.

        The base implementation runs the unfused composition (``norm``, the ``silu_and_mul`` operator,
        ``apply``, then the residual add), which is the eager numerics.

        Args:
            layer: the linear layer.
            x: ``[..., in_features]`` input, or ``[..., 2 * in_features]`` with ``input_act="silu_and_mul"``.
            norm: optional RMSNorm module applied to ``x`` first.
            input_act: None or ``"silu_and_mul"``.
            residual: optional ``[..., out_features]`` added to the output.

        Returns:
            ``[..., out_features]``.

        Raises:
            ValueError: ``input_act`` is not None or ``"silu_and_mul"``.
        """
        if norm is not None:
            x = norm(x)
        if input_act is not None:
            if input_act != "silu_and_mul":
                raise ValueError(f"unknown input activation {input_act!r}")
            half = x.shape[-1] // 2
            x = self._silu_and_mul(x[..., :half], x[..., half:])
        y = self.apply(layer, x)
        return y if residual is None else residual + y

    def apply_qkv(self, layer, x, rope, *, norm=None):
        """Fused QKV projection (``layer.layout`` is a ``QKVLayout``) with the rotary embedding.

        Returns ``q [..., H, D]``, ``k`` and ``v [..., KV, D]``; ``rope`` is a
        ``RopeTables`` for the rows of ``x``.

        The base implementation projects with ``apply_fused``, splits ``[q | k | v]`` and rotates q and k with
        ``rope.rotate`` (the eager rotation).

        Args:
            layer: a ``QKVLinear``.
            x: ``[..., hidden]`` input.
            rope: the ``RopeTables`` of this forward.
            norm: optional RMSNorm module applied to ``x`` first.

        Returns:
            ``(q, k, v)``: rotated ``q [..., H, D]``, rotated ``k [..., KV, D]`` and ``v [..., KV, D]``.
        """
        layout = layer.layout
        qkv = self.apply_fused(layer, x, norm=norm)
        q, k, v = qkv.split([layout.q_dim, layout.k_dim, layout.k_dim], dim=-1)
        lead = x.shape[:-1]
        q = rope.rotate(q.reshape(*lead, layout.num_heads, layout.head_dim))
        k = rope.rotate(k.reshape(*lead, layout.num_kv_heads, layout.head_dim))
        return q, k, v.reshape(*lead, layout.num_kv_heads, layout.head_dim)

    def apply_gated(self, layer, x, *, norm=None) -> torch.Tensor:
        """``silu(gate) * up`` of a fused ``[gate | up]`` projection (``layer.layout`` is a ``GatedLayout``).

        The base implementation projects with ``apply_fused`` and applies the ``silu_and_mul`` operator.

        Args:
            layer: a ``GatedLinear``.
            x: ``[..., hidden]`` input.
            norm: optional RMSNorm module applied to ``x`` first.

        Returns:
            ``[..., intermediate]``.
        """
        gate_up = self.apply_fused(layer, x, norm=norm)
        n = layer.layout.intermediate
        return self._silu_and_mul(gate_up[..., :n], gate_up[..., n:])


class Unquantized(QuantMethod):
    """Weights in their loaded dtype; the ``linear`` operator."""

    name = "bf16"

    def __init__(self):
        super().__init__()
        self._linear = active_ops().linear

    def apply(self, layer, x):
        """``x @ layer.weight.T + layer.bias`` with the op table's ``linear`` operator: ``[..., out_features]``."""
        return self._linear(x, layer.weight, layer.bias)


METHODS: dict[str, type] = {"bf16": Unquantized}
# Modules imported on first use of their methods; they add themselves to METHODS.
_MODULES = {"mxfp8": "mxfp8", "nvfp4": "nvfp4", "nvfp4_mse": "nvfp4"}


def make_method(name: str, config: QuantConfig | None = None) -> QuantMethod:
    """Construct quantization method ``name`` with its options from ``config``.

    ``"mxfp8"``, ``"nvfp4"`` and ``"nvfp4_mse"`` are imported on first use. Must run inside ``xinfer.ops.use_ops``.

    Args:
        name: method name, a key of ``METHODS``.
        config: supplies ``options[name]`` as constructor keyword arguments; None for no options.

    Returns:
        A new ``QuantMethod`` instance.

    Raises:
        KeyError: ``name`` is not a known method.
    """
    if name not in METHODS and name in _MODULES:
        importlib.import_module(f"xinfer.models.layers.quantization.{_MODULES[name]}")
    if name not in METHODS:
        raise KeyError(f"unknown quantization method {name!r}; known: {sorted(METHODS.keys() | _MODULES.keys())}")
    options = dict((config.options if config else {}).get(name, {}))
    return METHODS[name](**options)


_active: contextvars.ContextVar[QuantConfig | None] = contextvars.ContextVar("xinfer_quant", default=None)


@contextlib.contextmanager
def use_quant(config: QuantConfig | None) -> Iterator[QuantConfig | None]:
    """Make ``config`` the quantization config that linear layers read while they are constructed.

    The previous config is restored on exit; None means the default ``QuantConfig()`` (all ``bf16``).

    Args:
        config: the config to activate, or None.

    Yields:
        ``config``.
    """
    token = _active.set(config)
    try:
        yield config
    finally:
        _active.reset(token)


def active_quant() -> QuantConfig:
    """The config activated by the innermost ``use_quant``, or a default ``QuantConfig()`` (all ``bf16``)."""
    return _active.get() or QuantConfig()


def process_weights(module: torch.nn.Module) -> dict[str, int]:
    """Run each layer's ``process_weights_after_loading``; return layer counts per method.

    Args:
        module: the model; every submodule with a ``quant`` attribute that is a ``QuantMethod`` is processed.

    Returns:
        Method name → number of layers processed.
    """
    counts: dict[str, int] = {}
    for submodule in module.modules():
        method = getattr(submodule, "quant", None)
        if isinstance(method, QuantMethod):
            method.process_weights_after_loading(submodule)
            counts[method.name] = counts.get(method.name, 0) + 1
    return counts
