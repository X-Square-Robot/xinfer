"""Read tensors from single-file or sharded safetensors checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"


class SafetensorsCheckpoint:
    """Lazy key-addressed access to a checkpoint directory or file.

    Files are opened on first use (``safe_open`` on the CPU) and kept open;
    tensors are read one at a time.
    """

    def __init__(self, path: str | Path):
        """Locate the checkpoint files and list its keys.

        Args:
            path: a directory with ``model.safetensors.index.json`` (sharded) or ``model.safetensors``,
                or the path of one safetensors file.

        Raises:
            FileNotFoundError: a directory holds neither file.
        """
        path = Path(path)
        if path.is_dir():
            if (path / INDEX_FILE).exists():
                index = json.loads((path / INDEX_FILE).read_text())
                self._files = {k: path / v for k, v in index["weight_map"].items()}
            elif (path / SINGLE_FILE).exists():
                self._files = None
                self._single = path / SINGLE_FILE
            else:
                raise FileNotFoundError(f"no {SINGLE_FILE} or {INDEX_FILE} in {path}")
        else:
            self._files = None
            self._single = path
        self._handles: dict[Path, object] = {}
        if self._files is None:
            self._keys = tuple(self._handle(self._single).keys())
        else:
            self._keys = tuple(self._files)

    def _handle(self, file: Path):
        if file not in self._handles:
            self._handles[file] = safe_open(str(file), framework="pt", device="cpu")
        return self._handles[file]

    def _file(self, key: str) -> Path:
        if self._files is None:
            return self._single
        return self._files[key]

    def keys(self) -> tuple[str, ...]:
        """All tensor keys: the index's ``weight_map`` keys when sharded, else the file's keys."""
        return self._keys

    def __contains__(self, key: str) -> bool:
        return key in set(self._keys)

    def get(self, key: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Return a CPU tensor, cast to ``dtype`` when given.

        Args:
            key: tensor key.
            dtype: target dtype; ``None`` keeps the stored dtype.

        Returns:
            The tensor on the CPU.

        Raises:
            KeyError: ``key`` is not in a sharded checkpoint's index.
        """
        tensor = self._handle(self._file(key)).get_tensor(key)
        return tensor if dtype is None else tensor.to(dtype)

    def shape(self, key: str) -> tuple[int, ...]:
        """Return the stored shape of ``key`` without reading the tensor data."""
        return tuple(self._handle(self._file(key)).get_slice(key).get_shape())
