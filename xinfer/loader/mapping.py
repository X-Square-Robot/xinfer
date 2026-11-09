"""Assign checkpoint tensors to module parameters through an explicit mapping.

A model's weight mapping lists, for every parameter and buffer it owns, the
checkpoint key and the dtype the tensor must have at inference. Loading is
strict: every target must be mapped, every mapped key must exist with the
target's shape, and checkpoint keys the model does not read are reported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from xinfer.loader.safetensors import SafetensorsCheckpoint


@dataclass(frozen=True)
class Source:
    """Where one target tensor comes from.

    ``keys`` are read in order; ``combine`` builds the target from them (for
    example concatenating separate q/k/v projections). With one key and no
    ``combine`` the tensor is used as is.

    Attributes:
        keys: checkpoint keys to read, in order.
        dtype: dtype the target has at inference; the (combined) tensor is cast to it.
        combine: builds the target from the CPU tensors of ``keys``, in the checkpoint's dtype; ``None``
            uses the first tensor.
    """

    keys: tuple[str, ...]
    dtype: torch.dtype
    combine: Callable[[list[torch.Tensor]], torch.Tensor] | None = None

    @staticmethod
    def one(key: str, dtype: torch.dtype) -> Source:
        """A target read from one checkpoint ``key`` and cast to ``dtype``."""
        return Source((key,), dtype)


@dataclass
class LoadReport:
    """Result of ``load_mapped``.

    Attributes:
        loaded: number of model tensors assigned (the size of the mapping).
        unused_keys: checkpoint keys no ``Source`` read, sorted.
    """

    loaded: int
    unused_keys: tuple[str, ...]


def load_mapped(
    module: torch.nn.Module,
    mapping: Mapping[str, Source],
    checkpoint: SafetensorsCheckpoint,
    *,
    device: str,
) -> LoadReport:
    """Load ``checkpoint`` into ``module`` strictly, following ``mapping``.

    Every parameter and persistent buffer must be mapped (non-persistent
    buffers, derived at construction, may be left out); every mapped name
    must be a parameter or buffer of ``module``; every key must exist; and
    each built tensor, after the cast to ``Source.dtype``, must have the
    target's shape. Tensors are read on the CPU, moved to ``device`` and
    assigned by replacing the module's tensor objects, so ``module`` may be
    built on the ``meta`` device. Parameters become ``requires_grad=False``.

    Args:
        module: the model, with every target registered (possibly on ``meta``).
        mapping: dotted parameter or buffer name to its ``Source``.
        checkpoint: the open checkpoint.
        device: torch device string of the loaded tensors.

    Returns:
        A ``LoadReport`` with the number of loaded tensors and the unused checkpoint keys.

    Raises:
        KeyError: a parameter or persistent buffer is unmapped, the mapping names an unknown tensor, or
            the checkpoint lacks a key.
        ValueError: a built tensor's shape differs from the target's.
    """
    targets = dict(module.named_parameters())
    targets.update(dict(module.named_buffers()))
    # Buffers that are derived at construction (e.g. rotary inverse
    # frequencies) are excluded by not registering them as persistent.
    persistent = {n for n, _ in module.named_parameters()}
    for prefix, submodule in module.named_modules():
        for name in submodule._buffers:
            if name not in submodule._non_persistent_buffers_set and submodule._buffers[name] is not None:
                persistent.add(f"{prefix}.{name}" if prefix else name)
    unmapped = sorted(persistent - set(mapping))
    if unmapped:
        raise KeyError(f"{len(unmapped)} model tensors have no checkpoint source, e.g. {unmapped[:5]}")
    extra = sorted(set(mapping) - set(targets))
    if extra:
        raise KeyError(f"mapping names unknown model tensors, e.g. {extra[:5]}")

    available = set(checkpoint.keys())
    used = set()
    for name, source in mapping.items():
        missing = [k for k in source.keys if k not in available]
        if missing:
            raise KeyError(f"{name}: checkpoint lacks {missing}")
        tensors = [checkpoint.get(k) for k in source.keys]
        used.update(source.keys)
        value = source.combine(tensors) if source.combine else tensors[0]
        value = value.to(source.dtype)
        target = targets[name]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"{name}: checkpoint shape {tuple(value.shape)}, model {tuple(target.shape)}")
        _assign(module, name, value.to(device), parameter=isinstance(target, torch.nn.Parameter))
    return LoadReport(loaded=len(mapping), unused_keys=tuple(sorted(available - used)))


def _assign(module: torch.nn.Module, dotted: str, value: torch.Tensor, *, parameter: bool) -> None:
    """Set the parameter or buffer at ``dotted`` (e.g. ``"layers.0.qkv.weight"``) to ``value``.

    Args:
        module: the root module.
        dotted: name relative to ``module``.
        value: the tensor to install, already on the target device and in the target dtype.
        parameter: install as a ``Parameter`` (``requires_grad=False``) rather than a buffer.
    """
    # Replace the tensor object instead of copying into it, so modules may be
    # constructed on the "meta" device without allocating memory first.
    owner_name, _, attr = dotted.rpartition(".")
    owner = module.get_submodule(owner_name) if owner_name else module
    if parameter:
        owner._parameters[attr] = torch.nn.Parameter(value, requires_grad=False)
    else:
        owner._buffers[attr] = value
