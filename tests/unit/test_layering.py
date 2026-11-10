"""Dependency direction between packages, and the torch-free interfaces (docs/design.md §4).

The torch and device boundaries of §4 are targets and are not checked yet.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "xinfer"

# package -> xinfer packages it must not import
FORBIDDEN = {
    "platforms": {"ops", "models", "engine", "loader", "entrypoints", "observability"},
    "ops": {"models", "engine", "loader", "entrypoints", "observability"},
    "loader": {"models", "engine", "entrypoints"},
    "engine": {"models.layers", "loader", "entrypoints", "config"},
    "config": {"models.layers", "loader", "entrypoints", "frontend"},
    "frontend": {"engine", "models.layers", "loader", "ops", "config", "entrypoints"},
    # Profiling sits beside the runtime: it may use ops and platforms, never a model, a layer or the engine.
    "observability": {"engine", "models", "frontend", "loader", "config", "entrypoints"},
}
# The layers shared by the models may use ops and platforms, never a concrete model or the runtime.
LAYERS = "models/layers"
LAYERS_ALLOWED = ("xinfer.models.layers", "xinfer.ops", "xinfer.platforms")
# Interfaces that must stay free of torch, so that other runtimes can implement them.
TORCH_FREE = ("platforms/interface.py", "models/interface.py")
# The engine may use the model interface, never a concrete model, and the profiling spans and the request statistics,
# which import no other XInfer package.
ENGINE_ALLOWED = {"models": {"xinfer.models.interface"},
                  "observability": {"xinfer.observability.annotate", "xinfer.observability.stats"}}


def _imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_dependency_direction():
    violations = []
    for package, forbidden in FORBIDDEN.items():
        for path in (ROOT / package).rglob("*.py"):
            for name in _imports(path):
                parts = name.split(".")
                if parts[0] != "xinfer" or len(parts) < 2:
                    continue
                if any(name == f"xinfer.{f}" or name.startswith(f"xinfer.{f}.") for f in forbidden):
                    violations.append(f"{path.relative_to(ROOT)} imports {name}")
                if package == "engine" and parts[1] in ENGINE_ALLOWED and name not in ENGINE_ALLOWED[parts[1]]:
                    violations.append(f"{path.relative_to(ROOT)} imports {name}")
    for path in (ROOT / LAYERS).rglob("*.py"):
        violations += [f"{path.relative_to(ROOT)} imports {name}" for name in _imports(path)
                       if name.split(".")[0] == "xinfer" and not name.startswith(LAYERS_ALLOWED)]
    for module in ("annotate", "stats"):
        violations += [f"observability/{module}.py imports {name}"
                       for name in _imports(ROOT / f"observability/{module}.py") if name.split(".")[0] == "xinfer"]
    assert not violations, "\n".join(violations)


def test_concrete_models_do_not_import_runtime_internals():
    violations = []
    for path in (ROOT / "models").rglob("*.py"):
        for name in _imports(path):
            if name.startswith(("xinfer.engine", "xinfer.entrypoints")):
                violations.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not violations, "\n".join(violations)


def test_interfaces_do_not_import_torch():
    violations = [f"{module} imports {name}" for module in TORCH_FREE for name in _imports(ROOT / module)
                  if name.split(".")[0] == "torch"]
    assert not violations, "\n".join(violations)

