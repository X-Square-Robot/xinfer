"""Merging requests into one stage batch (design §6.3, §7.3).

Tensors are concatenated along their batch axis. A handoff tensor whose
``TensorSpec`` declares a ``seq_axis`` may differ in length along that axis
between requests: the batch left-pads every row to the longest (zeros) and the
stage also receives ``<name>_lengths`` (``int32 [B]``, each row's own length).
The lengths are passed for one request too, so a stage sees one input layout.
Left padding keeps every row's last token at the same index, which is what
attention kernels with a per-row start offset (flash-attn ``cache_leftpad``)
expect.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch

LENGTHS_SUFFIX = "_lengths"


def ragged_inputs(model, stage: str) -> dict[str, tuple[int, int]]:
    """Handoff tensors ``stage`` receives with a variable-length axis: ``{name: (batch_axis, seq_axis)}``.

    Args:
        model: the model whose ``handoffs`` are read.
        stage: consumer stage name.

    Returns:
        ``{name: (batch_axis, seq_axis)}`` for every handoff tensor into ``stage`` whose ``TensorSpec`` has a
        ``seq_axis``; empty if there is none.
    """
    out = {}
    for handoff in model.handoffs:
        if handoff.consumer == stage:
            for name, spec in handoff.tensors.items():
                if spec.seq_axis is not None:
                    out[name] = (spec.batch_axis, spec.seq_axis)
    return out


def compat_shape(value: torch.Tensor, batch_axis: int, seq_axis: int | None = None) -> tuple:
    """The shape requests must share to be batched: without the batch (and ragged) axis.

    Args:
        value: one request's tensor.
        batch_axis: its batch axis (negative values count from the end).
        seq_axis: its variable-length axis, for a ragged handoff tensor; None otherwise.

    Returns:
        ``value.shape`` without those axes, as a tuple; ``()`` for a 0-d tensor.
    """
    if value.ndim == 0:
        return ()
    drop = {batch_axis % value.ndim} | ({seq_axis % value.ndim} if seq_axis is not None else set())
    return tuple(s for i, s in enumerate(value.shape) if i not in drop)


def merge(views: Sequence[Mapping[str, Any]], batch_axes: Mapping[str, int],
          ragged: Mapping[str, tuple[int, int]]) -> Mapping[str, Any]:
    """Merge per-request inputs into one batch (see the module docstring).

    ``batch_axes`` gives the batch axis of non-ragged tensors (default 0).
    One request is returned as is (plus the ragged lengths). Several are
    merged lazily, one input at a time when the stage reads it, so inputs the
    stage never reads cost nothing. Inputs that cannot be batched (shapes
    beyond the batch axis or host values that differ between requests) are
    absent from the batch: the engine only groups requests whose inputs the
    stage reads agree, so these are inputs the stage does not read (for
    example the prompt ids at the action stage).

    Args:
        views: per-request input mappings (device tensors and host values), in batch order.
        batch_axes: batch axis of each non-ragged tensor name (default 0).
        ragged: ``{name: (batch_axis, seq_axis)}`` of handoff tensors whose length may differ (``ragged_inputs``).

    Returns:
        For one request a new dict with its inputs; for several a lazily merged ``MergedInputs``. Either way
        every ragged tensor comes with ``<name>_lengths`` (``int32 [B]``, on the tensor's device). A
        ``MergedInputs`` merges each value on the stream that is current when the value is first read.
    """
    if len(views) == 1:
        merged = dict(views[0])
        for name, (_, seq_axis) in ragged.items():
            if name in merged:
                value = merged[name]
                merged[name + LENGTHS_SUFFIX] = torch.full((1,), value.shape[seq_axis], dtype=torch.int32,
                                                           device=value.device)
        return merged
    return MergedInputs(views, batch_axes, ragged)


class MergedInputs(Mapping):
    """Several requests' inputs, merged on access (see ``merge``).

    A read-only mapping whose keys are fixed at construction, in the first request's order: every non-ragged input
    that can be merged (``_mergeable``), and every ragged input present in all requests together with its
    ``<name>_lengths``. Ragged inputs are not checked for compatible shapes or dtypes; the engine only batches
    requests whose inputs the stage reads agree. Each value is merged on its first lookup and cached.
    """

    def __init__(self, views: Sequence[Mapping[str, Any]], batch_axes: Mapping[str, int],
                 ragged: Mapping[str, tuple[int, int]]):
        """Collect the mergeable input names; no tensor is merged yet.

        Args:
            views: per-request input mappings, in batch order (at least one).
            batch_axes: batch axis of each non-ragged tensor name (default 0).
            ragged: ``{name: (batch_axis, seq_axis)}`` of handoff tensors whose length may differ.
        """
        self._views, self._axes, self._ragged = list(views), dict(batch_axes), dict(ragged)
        self._cache: dict[str, Any] = {}
        names = []
        for name, value in self._views[0].items():
            if name in self._ragged:
                if all(name in view for view in self._views[1:]):
                    names += [name, name + LENGTHS_SUFFIX]
            elif self._mergeable(name, value):
                names.append(name)
        self._names = dict.fromkeys(names)

    def _mergeable(self, name: str, value: Any) -> bool:
        """Whether non-ragged input ``name`` can be merged across all requests.

        Args:
            name: input name.
            value: the first request's value.

        Returns:
            For a tensor: every other request has a tensor of the same dtype and the same shape apart from the batch
            axis. For a host value: every other request has an equal value (``_same``).
        """
        others = [view.get(name, _MISSING) for view in self._views[1:]]
        if torch.is_tensor(value):
            axis = self._axes.get(name, 0)
            key = compat_shape(value, axis)
            return all(torch.is_tensor(other) and other.dtype == value.dtype and compat_shape(other, axis) == key
                       for other in others)
        return all(other is not _MISSING and _same(other, value) for other in others)

    def _merge(self, name: str) -> Any:
        """Build the batch value of ``name`` (one of the keys).

        Args:
            name: input name.

        Returns:
            For ``<base>_lengths`` of a ragged ``base``: each request's length along the sequence axis, ``int32 [B]``
            on ``base``'s device. For a ragged tensor: a zero tensor with ``B`` along the batch axis and the longest
            length along the sequence axis, into which each request is copied right-aligned (left padding). For
            other tensors: ``torch.cat`` along the batch axis. For a host value: the first request's value.
        """
        if name.endswith(LENGTHS_SUFFIX) and name[: -len(LENGTHS_SUFFIX)] in self._ragged:
            base = name[: -len(LENGTHS_SUFFIX)]
            seq_axis = self._ragged[base][1]
            device = self._views[0][base].device
            host = torch.tensor([view[base].shape[seq_axis] for view in self._views], dtype=torch.int32,
                                pin_memory=device.type == "cuda")
            return host.to(device, non_blocking=True)
        if name in self._ragged:
            batch_axis, seq_axis = self._ragged[name]
            parts = [view[name] for view in self._views]
            lengths = [tensor.shape[seq_axis] for tensor in parts]
            longest = max(lengths)
            shape = list(parts[0].shape)
            shape[batch_axis], shape[seq_axis] = len(parts), longest
            out = torch.zeros(shape, dtype=parts[0].dtype, device=parts[0].device)
            for i, (tensor, length) in enumerate(zip(parts, lengths)):
                out.narrow(batch_axis, i, 1).narrow(seq_axis, longest - length, length).copy_(tensor)
            return out
        value = self._views[0][name]
        if torch.is_tensor(value):
            return torch.cat([view[name] for view in self._views], dim=self._axes.get(name, 0))
        return value

    def __getitem__(self, name: str) -> Any:
        """Return the merged value of ``name``, merging it on first access; ``KeyError`` if it is not a key."""
        if name not in self._names:
            raise KeyError(name)
        if name not in self._cache:
            self._cache[name] = self._merge(name)
        return self._cache[name]

    def __contains__(self, name: object) -> bool:
        return name in self._names

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


_MISSING = object()


def _same(a: Any, b: Any) -> bool:
    """Whether two host values are equal: ``bool(a == b)``, or equal ``repr`` when that comparison raises."""
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - e.g. arrays: compare their representation
        return repr(a) == repr(b)
