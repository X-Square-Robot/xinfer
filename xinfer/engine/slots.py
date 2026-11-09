"""Admission slots with generation checks (docs/design.md §8.1).

The engine admits a request by allocating a slot (at most
``max_running_requests`` at once) and releases it when the request finishes. A
handle names a slot and the generation it was allocated in; after the slot is
released the generation advances, so a stale handle is detected instead of
reaching another request's state. A slot can hold request-owned tensors
(``tensors()``); the engine currently keeps handoff tensors on the request.
Release is deferred until the slot's last reader has finished (an event-like
object with ``query()``/``synchronize()``).
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass, field
from typing import Any


class SlotState(enum.Enum):
    """Lifecycle of a slot: FREE -> ACTIVE (``allocate``) -> RELEASING (``release``) -> FREE (reclaimed).

    Reclaiming waits for the release's ``after`` event and advances the slot's generation.
    """

    FREE = "free"
    ACTIVE = "active"  # allocated; stages write and read handoff tensors
    RELEASING = "releasing"  # released by the owner, waiting for the last reader


class StaleHandle(RuntimeError):
    """Raised for a handle whose slot was reclaimed since, or whose slot is not active (e.g. released twice)."""

    pass


class PoolExhausted(RuntimeError):
    """Raised by ``SlotPool.allocate`` when no slot is free (all active or still releasing)."""

    pass


@dataclass(frozen=True)
class SlotHandle:
    """Names one allocation of a slot.

    Attributes:
        index: the slot's index in the pool.
        generation: the slot's generation at allocation; it no longer matches once the slot is reclaimed.
    """

    index: int
    generation: int


@dataclass
class _Slot:
    """State of one slot, guarded by the pool's lock.

    Attributes:
        index: position in the pool.
        generation: incremented each time the slot is reclaimed.
        state: the ``SlotState``.
        tensors: request-owned tensors of the current allocation; cleared on reclaim.
        release_after: event-like object (``query()``) that must complete before reuse, or None.
    """

    index: int
    generation: int = 0
    state: SlotState = SlotState.FREE
    tensors: dict[str, Any] = field(default_factory=dict)
    release_after: Any = None  # event that must complete before reuse


class SlotPool:
    """Fixed-size pool of admission slots with generation-checked handles.

    Thread-safe: every method takes the pool's lock. Releases whose ``after`` event has completed are reclaimed
    lazily, by ``allocate``, ``release`` and ``in_use``.
    """

    def __init__(self, size: int):
        """Create ``size`` free slots.

        Args:
            size: number of slots.

        Raises:
            ValueError: ``size`` is less than 1.
        """
        if size < 1:
            raise ValueError("slot pool size must be positive")
        self._slots = [_Slot(i) for i in range(size)]
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._slots)

    def allocate(self) -> SlotHandle:
        """Allocate a free slot in one step, or raise ``PoolExhausted``.

        Completed releases are reclaimed first; the free slot with the lowest index becomes ACTIVE.

        Returns:
            A handle with the slot's index and current generation.

        Raises:
            PoolExhausted: every slot is active or still waiting for its release event.
        """
        with self._lock:
            self._reclaim_locked()
            for slot in self._slots:
                if slot.state is SlotState.FREE:
                    slot.state = SlotState.ACTIVE
                    return SlotHandle(slot.index, slot.generation)
        raise PoolExhausted(f"all {len(self._slots)} slots are in use")

    def tensors(self, handle: SlotHandle) -> dict[str, Any]:
        """The handoff tensors of an active slot.

        Args:
            handle: handle from ``allocate``.

        Returns:
            The slot's own ``{name: tensor}`` dict (not a copy); the caller may add entries. It is cleared when the
            slot is reclaimed.

        Raises:
            StaleHandle: the slot was reclaimed since the handle was issued, or it is not active.
        """
        with self._lock:
            slot = self._check_locked(handle)
            if slot.state is not SlotState.ACTIVE:
                raise StaleHandle(f"slot {handle.index} is {slot.state.value}")
            return slot.tensors

    def release(self, handle: SlotHandle, after: Any = None) -> None:
        """Release a slot; it becomes reusable once ``after`` (if given) completes.

        The slot becomes RELEASING. It is reclaimed (tensors cleared, generation advanced, FREE) in this call if
        ``after`` is None or has completed, else by a later pool call that finds ``after.query()`` true.

        Args:
            handle: handle from ``allocate``.
            after: event-like object with ``query()`` marking the slot's last reader, or None.

        Raises:
            StaleHandle: the slot was reclaimed since the handle was issued, or it was already released.
        """
        with self._lock:
            slot = self._check_locked(handle)
            if slot.state is not SlotState.ACTIVE:
                raise StaleHandle(f"slot {handle.index} released twice")
            slot.release_after = after
            slot.state = SlotState.RELEASING
            self._reclaim_locked()

    def in_use(self) -> int:
        """Number of slots not free (active or still releasing), after reclaiming completed releases."""
        with self._lock:
            self._reclaim_locked()
            return sum(s.state is not SlotState.FREE for s in self._slots)

    def _check_locked(self, handle: SlotHandle) -> _Slot:
        """Return the slot ``handle`` names (caller holds the lock).

        Raises:
            StaleHandle: the handle's generation is not the slot's current generation.
        """
        slot = self._slots[handle.index]
        if slot.generation != handle.generation:
            raise StaleHandle(
                f"slot {handle.index}: handle generation {handle.generation}, "
                f"current {slot.generation}"
            )
        return slot

    def _reclaim_locked(self) -> None:
        """Free every RELEASING slot whose ``release_after`` is None or done (caller holds the lock).

        A reclaimed slot's tensors are cleared and its generation advances, invalidating outstanding handles.
        """
        for slot in self._slots:
            if slot.state is not SlotState.RELEASING:
                continue
            event = slot.release_after
            if event is not None and not event.query():
                continue
            slot.tensors.clear()
            slot.release_after = None
            slot.generation += 1
            slot.state = SlotState.FREE
