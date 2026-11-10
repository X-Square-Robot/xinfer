"""Operator registry and build-time selection (docs/design.md §14.2).

Implementations are selected once, when a model is built, not per call. The
selection is recorded so that a service can report which backend, version and
contract each operator uses.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace

from xinfer.ops.spec import Contract, OpSpec
from xinfer.platforms.interface import Platform

logger = logging.getLogger(__name__)

REFERENCE_BACKEND = "reference"
PLUGIN_GROUP = "xinfer.ops"


@dataclass(frozen=True)
class Implementation:
    """One registered implementation of an operator.

    Attributes:
        op: operator name.
        backend: backend name (``"reference"`` for the torch reference).
        function: the callable layers bind to; same call signature as the operator's reference.
        platforms: platform names (``"cuda-sm120"``), device types (``"cuda"``) or arch names (``"sm120"``)
            this implementation runs on; ``"*"`` matches any platform.
        contract: the numerical and runtime properties it guarantees.
        priority: selection rank; the highest-priority supported implementation wins (reference: ``-1``).
        version: backend version string for the selection report, or None.
    """

    op: str
    backend: str
    function: Callable
    platforms: tuple[str, ...]  # platform names or arch names; "*" matches any
    contract: Contract
    priority: int = 0
    version: str | None = None

    def supports(self, platform: Platform) -> bool:
        """Whether this implementation may run on ``platform``.

        Args:
            platform: the resolved execution platform.

        Returns:
            True if ``platforms`` contains ``"*"``, the platform name, its device type or its arch.
        """
        if "*" in self.platforms:
            return True
        keys = {platform.name, platform.capabilities.device_type}
        if platform.capabilities.arch:
            keys.add(platform.capabilities.arch)
        return bool(keys.intersection(self.platforms))


class OpTable:
    """The operators selected for one platform; attribute access returns callables.

    Built by ``OpRegistry.build_table``.

    Args:
        platform_name: name of the platform the selections were made for.
        selections: the selected implementation per operator name.

    Attributes:
        platform_name: name of the platform the selections were made for.
    """

    def __init__(self, platform_name: str, selections: Mapping[str, Implementation]):
        self.platform_name = platform_name
        self._selections = dict(selections)

    def wrapped(self, wrap: Callable[[Implementation], Callable]) -> "OpTable":
        """A copy of this table whose operators call ``wrap(implementation)`` instead, for instrumentation.

        Selections, contracts and ``report()`` are unchanged; only ``Implementation.function`` differs. This table
        is not modified.

        Args:
            wrap: given each selected ``Implementation``, returns the callable that layers bind.

        Returns:
            The wrapped ``OpTable``.
        """
        selections = {name: replace(i, function=wrap(i)) for name, i in self._selections.items()}
        return OpTable(self.platform_name, selections)

    def __getattr__(self, name: str) -> Callable:
        """The selected implementation's callable for operator ``name``.

        Raises:
            AttributeError: ``name`` is not an operator of this table.
        """
        try:
            return self._selections[name].function
        except KeyError:
            raise AttributeError(f"operator {name!r} is not in this op table") from None

    def selection(self, name: str) -> Implementation:
        """The ``Implementation`` selected for operator ``name`` (KeyError if it is not in the table)."""
        return self._selections[name]

    def report(self) -> list[dict]:
        """One row per operator, for the startup selection table.

        Returns:
            Rows sorted by operator name, each a dict with keys ``op``, ``backend``, ``version``, ``numerics``,
            ``deterministic`` and ``graph_safe``.
        """
        rows = []
        for name in sorted(self._selections):
            implementation = self._selections[name]
            rows.append(
                dict(
                    op=name,
                    backend=implementation.backend,
                    version=implementation.version,
                    numerics=implementation.contract.numerics,
                    deterministic=implementation.contract.deterministic,
                    graph_safe=implementation.contract.graph_safe,
                )
            )
        return rows


class OpRegistry:
    """Operator definitions and their registered implementations.

    Operators are defined once (``define``) with their torch reference; backends add implementations
    (``register``, usually through ``load_plugins``); ``build_table`` selects one implementation per operator
    for a platform. The process-wide instance is ``xinfer.ops.registry``.
    """

    def __init__(self):
        self._specs: dict[str, OpSpec] = {}
        self._impls: dict[str, list[Implementation]] = {}
        self._plugins_loaded = False

    # -- definition -------------------------------------------------------
    def define(self, spec: OpSpec) -> OpSpec:
        """Define an operator and register its reference as the ``"reference"`` backend.

        The reference supports every platform (``"*"``) with priority ``-1`` and the default ``Contract()``.

        Args:
            spec: the operator specification.

        Returns:
            ``spec``.

        Raises:
            ValueError: an operator with the same name is already defined.
        """
        if spec.name in self._specs:
            raise ValueError(f"operator {spec.name!r} is already defined")
        self._specs[spec.name] = spec
        self._impls[spec.name] = [
            Implementation(
                op=spec.name,
                backend=REFERENCE_BACKEND,
                function=spec.reference,
                platforms=("*",),
                contract=Contract(),
                priority=-1,
            )
        ]
        return spec

    def spec(self, name: str) -> OpSpec:
        """The ``OpSpec`` of operator ``name`` (KeyError if it is not defined)."""
        return self._specs[name]

    def names(self) -> tuple[str, ...]:
        """Names of all defined operators, sorted."""
        return tuple(sorted(self._specs))

    def register(
        self,
        op: str,
        *,
        backend: str,
        contract: Contract,
        platforms: tuple[str, ...] = ("*",),
        priority: int = 0,
        version: str | None = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator registering a backend implementation of a defined operator.

        Args:
            op: name of a defined operator.
            backend: backend name; at most one implementation per backend and operator.
            contract: the implementation's contract; ``numerics`` must be ``"bitwise"`` or ``"tolerance"``.
            platforms: platform names, device types or arch names it supports (``"*"``: any).
            priority: selection rank; the highest-priority supported implementation is the default.
            version: backend version string for the selection report.

        Returns:
            A decorator that records the decorated function as the implementation and returns it unchanged.

        Raises:
            ValueError: ``backend`` is ``"reference"`` or ``contract.numerics`` is ``"reference"`` (immediately),
                or ``backend`` is already registered for ``op`` (when decorating).
            KeyError: ``op`` is not defined (when decorating).
        """
        if backend == REFERENCE_BACKEND:
            raise ValueError("the reference backend is defined by OpSpec")
        if contract.numerics == "reference":
            raise ValueError("backend implementations must declare bitwise or tolerance numerics")

        def decorate(function: Callable) -> Callable:
            if op not in self._specs:
                raise KeyError(f"operator {op!r} is not defined")
            implementations = self._impls[op]
            if any(i.backend == backend for i in implementations):
                raise ValueError(f"backend {backend!r} already registered for {op!r}")
            implementations.append(
                Implementation(
                    op=op,
                    backend=backend,
                    function=function,
                    platforms=tuple(platforms),
                    contract=contract,
                    priority=priority,
                    version=version,
                )
            )
            return function

        return decorate

    def implementer(self, backend: str, *, platforms: tuple[str, ...], version: str | None = None,
                    priority: int = 10, **contract) -> Callable[..., Callable[[Callable], Callable]]:
        """The registration decorator of one backend: ``implementer(...)(op, **contract_overrides)``.

        Fixes the backend's name, platforms, priority, version and contract fields (``numerics`` defaults to
        ``"tolerance"``); an operator may override contract fields, for example ``deterministic=False``.

        Args:
            backend: backend name.
            platforms: platform names, device types or arch names its implementations support.
            version: backend version string for the selection report.
            priority: selection rank of its implementations.
            **contract: ``Contract`` fields shared by its implementations.

        Returns:
            ``implements(op, **contract_overrides)``, a decorator as ``register`` returns.
        """
        shared = {"numerics": "tolerance", **contract}

        def implements(op: str, **overrides) -> Callable[[Callable], Callable]:
            return self.register(op, backend=backend, contract=Contract(**{**shared, **overrides}),
                                 platforms=platforms, priority=priority, version=version)

        return implements

    def implementations(self, op: str) -> tuple[Implementation, ...]:
        """All implementations of ``op`` in registration order, the reference first."""
        return tuple(self._impls[op])

    def backend_ops(self, backend: str) -> list[str]:
        """Names of the operators ``backend`` implements, sorted."""
        return [op for op in self.names() if any(i.backend == backend for i in self._impls[op])]

    # -- plugins ------------------------------------------------------------
    def load_plugins(self) -> list[str]:
        """Register the built-in backends, then every ``xinfer.ops`` entry point, through one protocol.

        A backend is a module (or entry point) with a ``register(registry)`` function that registers what the
        environment provides and returns the operator names it registered (empty when its library is missing).
        Built-in: ``xcompute`` (``XINFER_XCOMPUTE_LIBRARY`` or the installed ``xcompute`` package) and ``flash_attn``
        (the ``flash_attn`` package). ``XINFER_<NAME>=0`` skips the backend or entry point ``<name>`` (upper case,
        characters other than letters, digits and ``_`` as ``_``; for example ``XINFER_FLASH_ATTN=0``).

        Runs once per registry; later calls do nothing. Errors raised by a backend's ``register`` (for
        example a missing ``XINFER_XCOMPUTE_LIBRARY`` file) propagate.

        Returns:
            Names of the backends and entry points that registered operators in this call (empty after the first
            call).
        """
        if self._plugins_loaded:
            return []
        self._plugins_loaded = True
        import os
        from importlib.metadata import entry_points

        from xinfer.ops.backends import flash_attn, xcompute

        plugins = [(module.BACKEND, lambda module=module: module.register) for module in (xcompute, flash_attn)]
        plugins += [(entry_point.name, entry_point.load) for entry_point in entry_points(group=PLUGIN_GROUP)]
        loaded = []
        for name, load in plugins:
            if os.environ.get("XINFER_" + re.sub(r"\W", "_", name.upper()), "1") != "0" and load()(self):
                loaded.append(name)
        return loaded

    # -- selection ----------------------------------------------------------
    def build_table(
        self,
        platform: Platform,
        *,
        backends: Mapping[str, str] | None = None,
        reference_only: bool = False,
    ) -> OpTable:
        """Select one implementation per operator for ``platform``.

        ``backends`` pins a backend per operator; a pinned backend that does not
        support the platform is an error. Otherwise the highest-priority
        supported implementation wins and the reference is the fallback.

        Args:
            platform: the platform to select for.
            backends: operator name → backend name to pin; takes precedence over ``reference_only``.
            reference_only: select the torch reference for every operator that is not pinned.

        Returns:
            The ``OpTable`` for ``platform``.

        Raises:
            KeyError: ``backends`` names an operator that is not defined.
            RuntimeError: a pinned backend is not registered for the operator or does not support ``platform``.
        """
        backends = dict(backends or {})
        unknown = set(backends) - set(self._specs)
        if unknown:
            raise KeyError(f"unknown operators in backend selection: {sorted(unknown)}")
        selections = {}
        for name in self._specs:
            candidates = [i for i in self._impls[name] if i.supports(platform)]
            if name in backends:
                pinned = [i for i in candidates if i.backend == backends[name]]
                if not pinned:
                    raise RuntimeError(
                        f"backend {backends[name]!r} for {name!r} does not support {platform.name}"
                    )
                selections[name] = pinned[0]
            elif reference_only:
                selections[name] = next(i for i in candidates if i.backend == REFERENCE_BACKEND)
            else:
                selections[name] = max(candidates, key=lambda i: i.priority)
        return OpTable(platform.name, selections)


_active: contextvars.ContextVar[OpTable | None] = contextvars.ContextVar("xinfer_ops", default=None)


@contextlib.contextmanager
def use_ops(table: OpTable) -> Iterator[OpTable]:
    """Make ``table`` the op table that layers bind to while they are constructed.

    The previous active table (if any) is restored on exit. The binding is a context variable, so it is local
    to the current thread or task.

    Args:
        table: the op table to activate.

    Yields:
        ``table``.
    """
    token = _active.set(table)
    try:
        yield table
    finally:
        _active.reset(token)


def active_ops() -> OpTable:
    """The op table activated by the innermost ``use_ops``.

    Returns:
        The active ``OpTable``.

    Raises:
        RuntimeError: no op table is active.
    """
    table = _active.get()
    if table is None:
        raise RuntimeError(
            "no active op table: construct layers inside `with xinfer.ops.use_ops(table):`"
        )
    return table
