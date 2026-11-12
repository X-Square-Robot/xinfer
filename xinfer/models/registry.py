"""Model registry.

Built-in models register by import path, so listing models does not import
torch or model code. Out-of-tree models register through the
``xinfer.models`` entry point group; each entry point loads to the model class.
"""

from __future__ import annotations

import importlib

PLUGIN_GROUP = "xinfer.models"

_BUILTIN = {
    "wall_oss": "xinfer.models.wall_oss.model:WallOSS",
    "wallx2b": "xinfer.models.wallx2b.serving:Wallx2bServing",
}


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
        The registered names and the ``xinfer.models`` entry point names, sorted.
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
    target = _BUILTIN.get(name, name if ":" in name else None)
    if target is not None:
        module, _, attr = target.partition(":")
        return getattr(importlib.import_module(module), attr)
    from importlib.metadata import entry_points

    for entry_point in entry_points(group=PLUGIN_GROUP):
        if entry_point.name == name:
            return entry_point.load()
    raise KeyError(f"unknown model {name!r}; available: {', '.join(available_models())}")
