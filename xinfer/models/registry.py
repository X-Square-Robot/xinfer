"""Model registry.

Built-in models register by import path, so listing models does not import
torch or model code. Out-of-tree models register through the
``xinfer.models`` entry point group; each entry point loads to the model class.
"""

from __future__ import annotations

import importlib

PLUGIN_GROUP = "xinfer.models"

WALL_OSS_0_5_ID = "wall-oss-0.5"
WALL_OSS_1_0_NANO_ID = "wall-oss-1.0-nano"

_BUILTIN = {
    WALL_OSS_0_5_ID: "xinfer.models.wall_oss_0_5.model:WallOSS05",
    WALL_OSS_1_0_NANO_ID: "xinfer.models.wall_oss_1_0_nano.model:WallOSS10Nano",
}


def canonical_model_name(name: str) -> str:
    """Normalize the capitalization of the two built-in model IDs.

    Entry-point names and import paths keep their original spelling.
    """
    normalized = name.casefold()
    return normalized if normalized in (WALL_OSS_0_5_ID, WALL_OSS_1_0_NANO_ID) else name


def register_model(name: str, target: str) -> None:
    """Register a model class by import path ``"package.module:Class"`` (tests, embedding apps).

    The class is imported only when ``get_model_class`` resolves ``name``. A
    registered name takes precedence over an entry point of the same name.

    Args:
        name: model name used by ``--model`` and ``Policy``.
        target: import path ``"package.module:Class"``.
    """
    _BUILTIN[name] = target


def available_models() -> tuple[str, ...]:
    """List the model names that can be served, without importing model code.

    Returns:
        Registered names and ``xinfer.models`` entry point names, sorted.
    """
    from importlib.metadata import entry_points

    names = set(_BUILTIN)
    names.update(entry_point.name for entry_point in entry_points(group=PLUGIN_GROUP))
    return tuple(sorted(names))


def get_model_class(name: str):
    """Resolve a registered name, an entry point, or an import path ``"module:Class"``.

    Lookup order: registered names, then ``name`` as an import path when it
    contains ``":"``, then the ``xinfer.models`` entry points.

    Args:
        name: a model name or an import path ``"module:Class"``.

    Returns:
        The model class (a ``VLAModel`` subclass).

    Raises:
        KeyError: ``name`` is neither registered, an import path, nor an entry point.
    """
    target = _BUILTIN.get(name) or _BUILTIN.get(canonical_model_name(name))
    if target is None and ":" in name:
        target = name
    if target is not None:
        module, _, attr = target.partition(":")
        return getattr(importlib.import_module(module), attr)
    from importlib.metadata import entry_points

    for entry_point in entry_points(group=PLUGIN_GROUP):
        if entry_point.name == name:
            return entry_point.load()
    raise KeyError(f"unknown model {name!r}; available: {', '.join(available_models())}")
