"""Operator specifications and numerical contracts (docs/design.md §14.1).

Every operator has a torch reference implementation. A backend implementation
declares how its numerics relate to the reference and which runtime
properties it guarantees; the startup selection table reports them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

NUMERICS = ("reference", "bitwise", "tolerance")


@dataclass(frozen=True)
class Contract:
    """Properties an implementation guarantees.

    Backends declare a contract when they register; the registry reports it in the startup selection table.

    Attributes:
        numerics: ``"reference"`` (this is the reference), ``"bitwise"`` (equal to the reference on the same
            platform) or ``"tolerance"`` (documented error bound vs the reference). Backend implementations
            must declare ``"bitwise"`` or ``"tolerance"``.
        deterministic: the same inputs give the same bits on every call.
        graph_safe: may be captured in a CUDA graph (no host sync, no allocation in capture).

    Raises:
        ValueError: ``numerics`` is not one of ``NUMERICS``.
    """

    # "reference": this is the reference; "bitwise": equal to the reference on
    # the same platform; "tolerance": documented error bound vs the reference.
    numerics: str = "reference"
    deterministic: bool = True
    # May be captured in a CUDA graph (no host sync, no allocation in capture).
    graph_safe: bool = True

    def __post_init__(self):
        if self.numerics not in NUMERICS:
            raise ValueError(f"numerics must be one of {NUMERICS}, got {self.numerics!r}")


@dataclass(frozen=True)
class OpSpec:
    """An operator: its name, call signature and reference implementation.

    Attributes:
        name: operator name; layers access the selected implementation as ``table.<name>``.
        signature: human-readable call signature with shapes and dtypes (documentation only, not parsed).
        reference: the torch reference implementation that defines the semantics.
    """

    name: str
    signature: str
    reference: Callable
