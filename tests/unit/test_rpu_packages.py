"""Public RPU imports remain compatible while implementation modules stay separate."""

import importlib
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_legacy_engine_imports_share_the_rpu_request_contract():
    from xinfer.engine.rpu import RpuEngine, RpuRequestPlan
    from xinfer.engine.rpu.engine import RpuEngine as EngineImplementation
    from xinfer.platforms.rpu.plan import RpuRequestPlan as BackendRequestPlan

    assert RpuEngine is EngineImplementation
    assert RpuRequestPlan is BackendRequestPlan


@pytest.mark.parametrize(("name", "module"), [
    ("rpu_platform", "platform"),
    ("RpuBusyError", "resources"),
    ("RpuBuffer", "resources"),
    ("RpuOperation", "resources"),
    ("RpuCompletion", "execution"),
    ("RpuExecution", "execution"),
    ("RpuBoundExecution", "execution"),
    ("RpuExecutionContext", "context"),
])
def test_legacy_platform_imports_keep_the_same_public_types(name, module):
    facade = importlib.import_module("xinfer.platforms.rpu")
    implementation = importlib.import_module(f"xinfer.platforms.rpu.{module}")

    assert getattr(facade, name) is getattr(implementation, name)


def test_rpu_platform_resolution_does_not_load_device_runtimes(tmp_path):
    import xinfer

    # Use the same package as the parent test, including when testing an installed wheel.
    package_root = str(Path(xinfer.__file__).resolve().parent.parent)
    script = f"import sys\nsys.path.insert(0, {package_root!r})\n" + textwrap.dedent("""
        import importlib.abc

        forbidden = {"torch", "torch_npu", "tvm_ffi", "xcompute", "rpu_backend", "rhino_launch", "acl", "pyacl"}
        attempts = []

        class DeviceRuntimeGuard(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in forbidden:
                    attempts.append(fullname)
                    raise AssertionError(f"RPU platform resolution tried to import {fullname}")

        sys.meta_path.insert(0, DeviceRuntimeGuard())

        from xinfer.platforms import resolve_platform
        from xinfer.platforms.rpu import RpuExecutionContext, rpu_platform

        assert callable(RpuExecutionContext)
        for device in ("rpu", "rpu:0"):
            platform = resolve_platform(device)
            assert platform == rpu_platform(device)
            assert platform.device == "rpu:0"
            assert platform.capabilities.execution_model == "explicit-completion"

        assert not attempts, attempts
        loaded = sorted(name for name in sys.modules if name.partition(".")[0] in forbidden)
        assert not loaded, loaded
    """)
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)

    assert result.returncode == 0, result.stdout + result.stderr
