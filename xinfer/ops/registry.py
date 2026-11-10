"""Operator registry and build-time selection (docs/design.md §14.2).

A backend is the hardware an implementation runs on (``"cuda"``); the torch reference (``"reference"``) runs on every
platform. Backend modules (``xinfer/ops/backends/<backend>/<category>.py``) import the kernel libraries they call
(xcompute, flash-attn) and declare their implementations with ``register``; an implementation whose library did not
import is not registered, so the operator runs on its reference.

Implementations are selected once, when a model is built, not per call. The selection is recorded so that a service
can report which backend, library, version and contract each operator uses.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import inspect
import logging
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace

from xinfer.ops.spec import Contract, OpSpec
from xinfer.platforms.interface import Platform

logger = logging.getLogger(__name__)

REFERENCE_BACKEND = "reference"
PLUGIN_GROUP = "xinfer.ops"
# Built-in backend modules, loaded like ``xinfer.ops`` entry points but without relying on installed package metadata.
BUILTIN_MODULES = ("xinfer.ops.backends.cuda.attn", "xinfer.ops.backends.cuda.gemm", "xinfer.ops.backends.cuda.norm",
                   "xinfer.ops.backends.cuda.flow")
STRICT_ENV = "XINFER_OPS_STRICT"


def strict() -> bool:
    """Whether ``XINFER_OPS_STRICT`` (not ``0``) turns slow fallbacks into errors (CI and benchmarks)."""
    return os.environ.get(STRICT_ENV, "0") != "0"


def _skipped(name: str) -> bool:
    """Whether ``XINFER_<NAME>=0`` skips the entry point ``name``."""
    return os.environ.get("XINFER_" + re.sub(r"\W", "_", name.upper()), "1") == "0"


@dataclass(frozen=True)
class Implementation:
    """One registered implementation of an operator.

    Attributes:
        op: operator name.
        backend: the hardware it runs on (``"cuda"``), or ``"reference"`` for the torch reference.
        function: the callable layers bind to; same call signature as the operator's reference.
        platforms: platform names (``"cuda-sm120"``), device types (``"cuda"``) or arch names (``"sm120"``)
            this implementation runs on; ``"*"`` matches any platform.
        contract: the numerical and runtime properties it guarantees.
        priority: selection rank; the highest-priority supported implementation wins (reference: ``-1``).
        version: library version string for the selection report, or None.
        library: name of the kernel library it calls (``"xcompute"``), or None.
    """

    op: str
    backend: str
    function: Callable
    platforms: tuple[str, ...]  # platform names or arch names; "*" matches any
    contract: Contract
    priority: int = 0
    version: str | None = None
    library: str | None = None

    @property
    def name(self) -> str:
        """``backend:library`` (``"cuda:xcompute"``), or the backend for an implementation without a library."""
        return self.backend if self.library is None else f"{self.backend}:{self.library}"

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


_NO_LIBRARY = object()


def register(op: str, *, backend: str, requires: object = _NO_LIBRARY, platforms: Sequence[str] | None = None,
             version: str | None = None, priority: int = 10, **contract) -> Callable[[Callable], Callable]:
    """Decorator registering the decorated function as an implementation of ``op`` on ``backend``.

    Backend modules call it at module level, so importing a module (``OpRegistry.load_plugins``) registers its
    implementations into the process-wide ``registry``.

    Args:
        op: name of an operator defined in ``registry``.
        backend: the hardware it runs on (``"cuda"``).
        requires: the imported library it calls (a module such as ``flash_attn``), or None when that library did not
            import: then nothing is registered and the operator runs on another implementation, at the end its
            reference. Omitted for an implementation that calls no library.
        platforms: platform, device type or arch names it supports; None: ``(backend,)``.
        version: version string for the selection report; None: ``"<library> <library.__version__>"`` when the
            library has a version.
        priority: selection rank.
        **contract: ``Contract`` fields; ``numerics`` defaults to ``"tolerance"``.

    Returns:
        A decorator that registers the function (unless ``requires`` is None) and returns it unchanged.

    Raises:
        ValueError, KeyError, TypeError: as ``OpRegistry.add``, when decorating.
    """
    contract = Contract(**{"numerics": "tolerance", **contract})
    library = None if requires is _NO_LIBRARY or requires is None else requires.__name__
    if version is None and library is not None and hasattr(requires, "__version__"):
        version = f"{library} {requires.__version__}"

    def decorate(function: Callable) -> Callable:
        if requires is not None:
            registry.add(op, function, backend=backend, contract=contract, platforms=tuple(platforms or (backend,)),
                         priority=priority, version=version, library=library)
        return function

    return decorate


def _origin(function: Callable) -> tuple:
    """Where ``function`` is defined (module, qualified name, first line): the same again after a reload."""
    code = getattr(function, "__code__", None)
    return (getattr(function, "__module__", None), getattr(function, "__qualname__", None),
            code.co_firstlineno if code else id(function))


def _parameters(function: Callable) -> tuple | None:
    """``(name, kind)`` of each parameter of ``function``, or None when it has no Python signature."""
    try:
        return tuple((p.name, p.kind) for p in inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return None


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
            return self.__dict__["_selections"][name].function
        except KeyError:
            raise AttributeError(f"operator {name!r} is not in this op table") from None

    def selection(self, name: str) -> Implementation:
        """The ``Implementation`` selected for operator ``name`` (KeyError if it is not in the table)."""
        return self._selections[name]

    def report(self) -> list[dict]:
        """One row per operator, for the startup selection table.

        Returns:
            Rows sorted by operator name, each a dict with keys ``op``, ``backend``, ``library``, ``version``,
            ``numerics``, ``deterministic`` and ``graph_safe``.
        """
        rows = []
        for name in sorted(self._selections):
            implementation = self._selections[name]
            rows.append(
                dict(
                    op=name,
                    backend=implementation.backend,
                    library=implementation.library,
                    version=implementation.version,
                    numerics=implementation.contract.numerics,
                    deterministic=implementation.contract.deterministic,
                    graph_safe=implementation.contract.graph_safe,
                )
            )
        return rows


class OpRegistry:
    """Operator definitions and their registered implementations.

    Operators are defined once (``define``) with their torch reference; implementations are added with ``add``
    (backend modules use ``register``, which adds to the process-wide instance ``registry``); ``build_table`` selects
    the implementations of every operator for a platform.
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

    def add(
        self,
        op: str,
        function: Callable,
        *,
        backend: str,
        contract: Contract,
        platforms: tuple[str, ...] = ("*",),
        priority: int = 0,
        version: str | None = None,
        library: str | None = None,
    ) -> Implementation:
        """Add an implementation of a defined operator.

        Adding a function of the same module and name again (a reloaded backend module) replaces it.

        Args:
            op: name of a defined operator.
            function: the implementation; same parameters (names and kinds) as the operator's reference.
            backend: backend name; at most one implementation per backend, library and operator.
            contract: the implementation's contract; ``numerics`` must be ``"bitwise"`` or ``"tolerance"``.
            platforms: platform names, device types or arch names it supports (``"*"``: any).
            priority: selection rank; the highest-priority supported implementation is the default.
            version: library version string for the selection report.
            library: name of the kernel library it calls, or None.

        Returns:
            The added ``Implementation``.

        Raises:
            ValueError: ``backend`` is ``"reference"``, ``contract.numerics`` is ``"reference"``, or another function
                is registered for the same backend and library.
            KeyError: ``op`` is not defined.
            TypeError: the function's parameters differ from the reference's.
        """
        if backend == REFERENCE_BACKEND:
            raise ValueError("the reference backend is defined by OpSpec")
        if contract.numerics == "reference":
            raise ValueError("backend implementations must declare bitwise or tolerance numerics")
        if op not in self._specs:
            raise KeyError(f"operator {op!r} is not defined")
        implementation = Implementation(op=op, backend=backend, function=function, platforms=tuple(platforms),
                                        contract=contract, priority=priority, version=version, library=library)
        expected, got = _parameters(self._specs[op].reference), _parameters(function)
        if expected is not None and got is not None and expected != got:
            raise TypeError(f"{implementation.name} implements {op!r} with parameters {[n for n, _ in got]}, "
                            f"the reference takes {[n for n, _ in expected]}")
        implementations = self._impls[op]
        for index, existing in enumerate(implementations):
            if existing.name == implementation.name:
                if _origin(existing.function) != _origin(function):
                    raise ValueError(f"{implementation.name!r} already registered for {op!r}")
                implementations[index] = implementation  # the module was imported again
                return implementation
        implementations.append(implementation)
        return implementation

    def implementations(self, op: str) -> tuple[Implementation, ...]:
        """All implementations of ``op`` in registration order, the reference first."""
        return tuple(self._impls[op])

    def backend_ops(self, name: str) -> list[str]:
        """Names of the operators implemented by ``name`` (a backend, library or ``backend:library``), sorted."""
        return [op for op in self.names() if any(name in (i.backend, i.library, i.name) for i in self._impls[op])]

    # -- plugins ------------------------------------------------------------
    def load_plugins(self) -> list[str]:
        """Import the built-in backend modules, then every module of the ``xinfer.ops`` entry point group.

        Importing a backend module registers its implementations (``register``) into the process-wide ``registry``.
        Built-in: ``xinfer/ops/backends/cuda/`` (``attn``, ``gemm``, ``norm``, ``flow``), on xcompute
        (``XINFER_XCOMPUTE_LIBRARY`` or the installed ``xcompute`` package) and flash-attn (the ``flash_attn``
        package) when they import; ``XINFER_XCOMPUTE=0`` and ``XINFER_FLASH_ATTN=0`` leave them out.
        ``XINFER_<NAME>=0`` skips the entry point ``<name>`` without importing it (upper case, characters other than
        letters, digits and ``_`` as ``_``).

        Runs once per registry; later calls do nothing. Errors raised while importing a module (for example a missing
        ``XINFER_XCOMPUTE_LIBRARY`` file) propagate.

        Returns:
            The names (``"cuda:xcompute"``) of the implementations this call added to this registry, in order, without
            repeats (empty after the first call).
        """
        if self._plugins_loaded:
            return []
        self._plugins_loaded = True
        from importlib.metadata import entry_points

        before = {(op, i.name) for op, implementations in self._impls.items() for i in implementations}
        for module in BUILTIN_MODULES:
            importlib.import_module(module)
        for entry_point in entry_points(group=PLUGIN_GROUP):
            if not _skipped(entry_point.name):
                entry_point.load()
        added = []
        for op, implementations in self._impls.items():
            for i in implementations:
                if (op, i.name) not in before and i.name not in added:
                    added.append(i.name)
        return added

    # -- selection ----------------------------------------------------------
    def build_table(
        self,
        platform: Platform,
        *,
        backends: Mapping[str, str] | None = None,
        reference_only: bool = False,
    ) -> OpTable:
        """Select the implementations of every operator for ``platform``.

        ``backends`` pins an implementation per operator, by backend (``"cuda"``), library (``"xcompute"``) or both
        (``"cuda:xcompute"``); the highest-priority match wins, and a pin that matches no implementation supporting
        the platform is an error. Otherwise the highest-priority supported implementation is selected and the
        reference is the fallback.

        Args:
            platform: the platform to select for.
            backends: operator name → pin; takes precedence over ``reference_only``.
            reference_only: select the torch reference for every operator that is not pinned.

        Returns:
            The ``OpTable`` for ``platform``.

        Raises:
            KeyError: ``backends`` names an operator that is not defined.
            RuntimeError: a pin matches no implementation of the operator that supports ``platform``.
        """
        if self is registry:  # the backend modules register into the process-wide registry
            self.load_plugins()
        backends = dict(backends or {})
        unknown = set(backends) - set(self._specs)
        if unknown:
            raise KeyError(f"unknown operators in backend selection: {sorted(unknown)}")
        selections = {}
        for name in self._specs:
            # stable sort: among equal priorities the first registered comes first
            candidates = sorted((i for i in self._impls[name] if i.supports(platform)), key=lambda i: -i.priority)
            if name in backends:
                pin = backends[name]
                pinned = [i for i in candidates if pin in (i.name, i.backend, i.library)]
                if not pinned:
                    raise RuntimeError(f"backend {pin!r} for {name!r} does not support {platform.name}")
                selections[name] = pinned[0]
            elif reference_only:
                selections[name] = next(i for i in candidates if i.backend == REFERENCE_BACKEND)
            else:
                selections[name] = candidates[0]
        return OpTable(platform.name, selections)


registry = OpRegistry()  # process-wide; ``xinfer.ops`` defines the operators

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
