"""Tuned tables: measured runtime settings per (model, architecture, quantization).

Files live in ``xinfer/tuned/<model>/<arch>-<quantization>.json`` and ship
with the package. Each records the values, the commit and date they were
measured at, the method and the evidence; the resolver reports them as
``tuned:<model>/<arch>-<quantization>@<commit>``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TUNED_ROOT = Path(__file__).resolve().parents[1] / "tuned"
REQUIRED = ("model", "arch", "quantization", "measured")


@dataclass(frozen=True)
class TunedTable:
    """Measured settings for one (model, architecture, quantization).

    Attributes:
        name: ``"<model>/<arch>-<quantization>"``.
        lanes: measured best lanes per stage name.
        measured: provenance: at least ``commit``, ``date``, ``method`` and ``evidence``.
        max_batch: measured batch-size limits per stage name.
    """

    name: str  # "<model>/<arch>-<quantization>"
    lanes: Mapping[str, int] = field(default_factory=dict)
    measured: Mapping[str, Any] = field(default_factory=dict)
    max_batch: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.max_batch, Mapping) or any(
            type(limit) is not int or limit < 1 for limit in self.max_batch.values()
        ):
            raise ValueError("tuned max_batch must map stage names to positive integers")

    @property
    def source(self) -> str:
        """Source label of tuned settings: ``tuned:<name>@<commit>`` (``?`` without a commit)."""
        return f"tuned:{self.name}@{self.measured.get('commit', '?')}"


def load_tuned(path: str | Path) -> TunedTable:
    """Load and validate one tuned-table JSON file.

    The file must have ``model``, ``arch``, ``quantization`` and ``measured``
    (with ``commit``, ``date``, ``method`` and ``evidence``); it may have
    ``lanes``, ``max_batch`` and ``notes``, and nothing else.

    Args:
        path: the JSON file (``--tuned <path>`` or a bundled table).

    Returns:
        The ``TunedTable``, with lanes converted to ``int``.

    Raises:
        ValueError: a required key is missing, a key is unknown, or the JSON is invalid.
        OSError: the file cannot be read.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ValueError(f"tuned table {path} lacks {missing}")
    known = set(REQUIRED) | {"lanes", "max_batch", "notes"}
    extra = set(data) - known
    if extra:
        raise ValueError(f"tuned table {path} has unknown keys {sorted(extra)}")
    for key in ("commit", "date", "method", "evidence"):
        if key not in data["measured"]:
            raise ValueError(f"tuned table {path}: measured.{key} is required")
    return TunedTable(name=f"{data['model']}/{data['arch']}-{data['quantization']}",
                      lanes={k: int(v) for k, v in data.get("lanes", {}).items()}, measured=data["measured"],
                      max_batch=data.get("max_batch", {}))


def find_tuned(model: str, arch: str | None, quantization: str | None, root: Path = TUNED_ROOT) -> TunedTable | None:
    """Look up the table ``<root>/<model>/<arch>-<quantization>.json``.

    Args:
        model: registry name of the model.
        arch: platform architecture, e.g. ``"sm120"``; ``None`` on CPU.
        quantization: quantization name, e.g. ``"mxfp8"``.
        root: directory of the tables; defaults to the tables bundled with the package.

    Returns:
        The loaded table, or ``None`` when ``arch`` or ``quantization`` is empty or no file exists.

    Raises:
        ValueError: the file exists but is not a valid tuned table.
    """
    if not arch or not quantization:
        return None
    path = root / model / f"{arch}-{quantization}.json"
    return load_tuned(path) if path.is_file() else None
