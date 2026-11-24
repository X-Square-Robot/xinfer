"""Kernels of the CUDA backend against FP32 torch: shared fixtures and helpers."""
from pathlib import Path

import pytest
import torch

from xinfer.ops.backends.cuda import _native
from xinfer.ops.layout import mxfp8 as layout
from xinfer.ops.reference import nvfp4

DEVICE = torch.device("cuda")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available() and _native.lib is not None:
        return
    here = Path(__file__).parent
    skip = pytest.mark.skip(reason="needs a CUDA GPU and the kernel library (docs/kernels.md)")
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(skip)


@pytest.fixture
def rnd():
    """Seeded normal tensors on the GPU: ``rnd(*shape, dtype=, scale=)``."""
    generator = torch.Generator().manual_seed(0)

    def make(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=generator) * scale).to(dtype).to(DEVICE)

    return make


def sm(*archs):
    """Mark a test as needing a GPU of one of the compute capabilities ``archs`` and a library built for it."""
    here = torch.cuda.is_available() and _native.arch(DEVICE) in archs and _native.arch(DEVICE) in _native.BUILD_ARCHS
    return pytest.mark.skipif(not here, reason=f"needs an SM{' or SM'.join(map(str, archs))} GPU and build")


def close(actual, expected, tolerance=5e-3):
    """BF16 results against an FP32 reference: relative error of the whole tensor."""
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item() < tolerance


def dequantize(pair):
    """FP32 values of an MXFP8 pair, or of an NVFP4 pair or weight triple ``(values, scales, global_scale)``."""
    values, scales = pair[0], pair[1]
    if values.dtype == torch.uint8:  # NVFP4
        return nvfp4.dequantize(values, scales) * (float(pair[2]) if len(pair) == 3 else 1.0)
    rows, k = values.shape
    exponent = scales[layout.scale_offsets(rows, k, values.device)][:, : k // 32].to(torch.int32)
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent - 127)
    return (values.to(torch.float32).view(rows, k // 32, 32) * scale[..., None]).view(rows, k)


MXFP8_TOLERANCE = 6e-2  # relative error of a tensor rounded to E4M3 (a 3-bit mantissa: at most 1/16 an element)
NVFP4_TOLERANCE = 0.15  # relative error of a tensor rounded to E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6 times a block scale)
