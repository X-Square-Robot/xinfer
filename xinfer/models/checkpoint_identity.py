"""Validate the optional XInfer model identity alongside a checkpoint."""

from __future__ import annotations

import json
from pathlib import Path


def validate_checkpoint_model(checkpoint: str | Path, expected_model_id: str) -> None:
    """If xinfer.json is present, require its model_id to match the canonical ID."""
    path = Path(checkpoint) / "xinfer.json"
    try:
        with path.open(encoding="utf-8") as source:
            identity = json.load(source)
    except FileNotFoundError:
        return
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{path}: invalid UTF-8 JSON: {error}") from error

    if not isinstance(identity, dict):
        raise ValueError(f"{path}: expected a JSON object with a model_id field")
    model_id = identity.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"{path}: model_id must be a non-empty string")
    if model_id.lower() != expected_model_id.lower():
        raise ValueError(f"{path}: model_id is {model_id!r}, expected {expected_model_id!r}")
