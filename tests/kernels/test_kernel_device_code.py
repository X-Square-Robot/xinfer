"""Device code of the kernel library: no kernel calls the driver fallback for bulk / TMA copies.

On SM120 ptxas compiles a cp.async.bulk{.tensor} with a .shared::cluster destination to an address check and a call
to __cuda_syscall_cp_async_bulk*. The first launch of such a kernel makes the driver raise the context's stack limit
(about 14 KB per thread, 3.3 GiB of device memory on an RTX 5090) for the life of the context. Kernels launched
without a cluster copy into their own shared memory with the .shared::cta destination.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from xinfer.ops.backends.cuda import _native


def _cuobjdump():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    found = shutil.which("cuobjdump") or str(Path(cuda_home) / "bin" / "cuobjdump")
    if not Path(found).is_file():
        pytest.skip("needs cuobjdump (CUDA toolkit) to read the device code")
    return found


def test_no_kernel_calls_the_bulk_copy_fallback():
    symbols = subprocess.run([_cuobjdump(), "-symbols", _native.LIBRARY_PATH], capture_output=True, text=True,
                             check=True).stdout
    offenders = []
    for cubin in symbols.split("Fatbin elf code:")[1:]:
        if re.search(r"\b__cuda_syscall_cp_async_bulk\w*", cubin):
            arch = re.search(r"arch = (\w+)", cubin)
            offenders.append((arch.group(1) if arch else "?", re.findall(r"STO_ENTRY\s+(\S+)", cubin)))
    assert not offenders, offenders
