"""Ownership of the optional compiled preparation helper's resized NumPy state output."""

from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import numpy as np
import pytest

preparefast = pytest.importorskip("xinfer._preparefast", reason="optional host helper is not built")


@pytest.fixture(scope="module")
def moving_allocator(tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip("the allocator fault-injection module is built on Linux")
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("allocator fault injection requires a C compiler")
    directory = tmp_path_factory.mktemp("preparefast-allocator")
    output = directory / ("host_allocator_probe" + sysconfig.get_config_var("EXT_SUFFIX"))
    source = Path(__file__).parents[1] / "fixtures" / "preparefast_allocator.c"
    subprocess.run([compiler, "-shared", "-fPIC", "-O0", "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
                    "-I" + sysconfig.get_path("include"), "-I" + np.get_include(), str(source), "-o", str(output)],
                   check=True, capture_output=True, text=True)
    return directory


@pytest.mark.parametrize("allocation_fails", [False, True])
def test_masked_state_resize_obeys_numpy_ownership(moving_allocator, allocation_fails):
    # A subprocess keeps the allocator override out of the suite and the serving runtime.
    script = r'''
import importlib.util
import sys
import numpy as np
sys.path.insert(0, sys.argv[2])
import host_allocator_probe as allocator
spec = importlib.util.spec_from_file_location("_preparefast", sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
positions = np.full((1, 26), 0.5, dtype=np.float64)
mask = np.ones_like(positions)
mask[0, 4:] = 0
minimum, delta = np.zeros_like(positions), np.ones_like(positions)
bins = np.linspace(-1, 1, 512)
fails = sys.argv[3] == "True"
allocator.install(fails)
try:
    if fails:
        try:
            helper.state_text(positions, mask, minimum, delta, bins, True)
        except MemoryError:
            pass
        else:
            raise AssertionError("failed resize must propagate MemoryError")
    else:
        text, values, count, digits = helper.state_text(positions, mask, minimum, delta, bins, True)
        assert values.tolist() == [255] * 4
        assert text == "255 255 255 255"
        assert (count, digits) == (4, 12)
finally:
    moved = allocator.reset()
    assert fails or moved == 1
'''
    result = subprocess.run([sys.executable, "-c", script, preparefast.__file__, str(moving_allocator),
                             str(allocation_fails)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("visible", [0, 4, 20, 26])
def test_state_text_matches_numpy_for_masked_states(visible):
    positions = np.linspace(-0.5, 1.5, 26, dtype=np.float64)[None]
    mask = np.zeros_like(positions)
    mask[0, :visible] = 1
    minimum, delta = np.zeros_like(positions), np.ones_like(positions)
    bins = np.linspace(-1, 1, 512)
    text, values, count, digits = preparefast.state_text(positions, mask, minimum, delta, bins, True)
    expected = (np.digitize(np.clip(positions * 2 - 1, -1, 1), bins) - 1)[0, :visible]
    assert values.tolist() == expected.tolist()
    assert text == " ".join(map(str, expected))
    assert (count, digits) == (visible, sum(len(str(abs(value))) for value in expected))


@pytest.mark.parametrize(("value", "range_value"), [(float("nan"), 1.0), (0.0, 0.0)])
def test_nan_normalization_declines_to_the_existing_numpy_path(value, range_value):
    positions = np.array([[value]], dtype=np.float64)
    mask, minimum = np.ones_like(positions), np.zeros_like(positions)
    delta = np.full_like(positions, range_value)
    bins = np.linspace(-1, 1, 512)
    assert preparefast.state_text(positions, mask, minimum, delta, bins, True) is None
    with np.errstate(invalid="ignore"):
        normalized = np.clip(((positions - minimum) / delta) * 2 - 1, -1, 1)
    assert (np.digitize(normalized, bins) - 1).item() == 511


def test_masked_nan_does_not_disable_a_finite_state_fast_path():
    positions = np.array([[0.5, np.nan]], dtype=np.float64)
    mask = np.array([[1.0, 0.0]])
    result = preparefast.state_text(positions, mask, np.zeros_like(positions), np.ones_like(positions),
                                    np.linspace(-1, 1, 512), True)
    assert result[0] == "255" and result[1].tolist() == [255]
