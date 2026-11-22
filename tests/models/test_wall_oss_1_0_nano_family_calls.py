"""Model plans prepare TP8 operations through in-tree RPU family entry points.

Catalog entry names remain accepted by the native provider for compatibility, so a call
site left on an old name still plans correctly and other tests cannot see it.
This test keeps the model code on the family API (gemm, rmsnorm, attention, ...).
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES = sorted((ROOT / "xinfer/models/wall_oss_1_0_nano/rpu").glob("*.py"))
# The adapter's DMA copy is not a catalog kernel and has no family form.
ALLOWED = {"copy_f16"}


def _prepared_names(path):
    """String literals passed as the operation name to prepare/_op/op calls."""
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if called not in {"prepare", "_op", "op"}:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.append((first.value, node.lineno))
    return names


def test_model_plans_use_family_entry_points():
    try:
        from xinfer.ops.backends.rpu import rpu_tp
    except ImportError as error:
        pytest.skip(f"in-tree RPU family table unavailable: {error}")
    families = set(rpu_tp.FAMILIES)
    offenders = [
        f"{path.relative_to(ROOT)}:{line}: {name}"
        for path in MODEL_FILES
        for name, line in _prepared_names(path)
        if name not in families and name not in ALLOWED
    ]
    assert not offenders, "use the RPU family entry point instead of:\n" + "\n".join(offenders)
    assert any(_prepared_names(path) for path in MODEL_FILES), "no prepare calls found"
