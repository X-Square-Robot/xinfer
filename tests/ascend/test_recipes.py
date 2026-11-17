"""The measurement recipe's explicit families retain the frozen all-selector behavior."""

import importlib.util
from pathlib import Path
import shlex

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _exported(recipe, name):
    for line in recipe.read_text(encoding="utf-8").splitlines():
        fields = shlex.split(line)
        if len(fields) == 2 and fields[0] == "export":
            key, _, value = fields[1].partition("=")
            if key == name:
                return value
    raise AssertionError(f"{name} is missing from {recipe}")


@pytest.mark.parametrize(("module_name", "variable"), [
    ("copykill_r2", "XINFER_COPYKILL_R2"), ("rmsfuse_r4", "XINFER_RMSFUSE_R4"),
])
def test_explicit_recipe_families_match_the_measured_all_selector(monkeypatch, module_name, variable):
    def selected(value):
        monkeypatch.setenv(variable, value)
        spec = importlib.util.spec_from_file_location(
            f"_recipe_{module_name}", ROOT / "xinfer/ops/backends/ascend" / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {family for family in module._FAMILIES if module.enabled(family)}

    explicit = _exported(ROOT / "examples/ascend/wall_oss.env", variable)
    assert explicit != "all"
    assert selected(explicit) == selected("all")
