"""Model-owned request contracts for the RPU prepared-execution backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .execution import RpuExecution


@dataclass(frozen=True)
class RpuRequestPlan:
    """A model-owned request binding over resident RPU stage executions.

    ``create_rpu_request(prepared, context)`` binds every dynamic input while
    the context is idle and returns exactly the model's StageGraph stages.
    ``collect`` runs after final completion and downloads only outputs.
    ``abort`` invalidates cached model plans/state after a failure; a native
    reset changes captured state generations. ``release`` closes request-local
    owners after completion, retaining persistent weights and cached batches.
    Constructing a plan must not execute a state-advancing warmup.
    """

    stages: Mapping[str, Sequence[RpuExecution]]
    collect: Callable[[], Mapping[str, Any]]
    abort: Callable[[BaseException], None] | None = None
    release: Callable[[], None] | None = None
