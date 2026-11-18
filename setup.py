"""Tag wheels containing the locally built Ascend library for this Python and platform."""

from pathlib import Path

from setuptools import Distribution, setup


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        library = Path(__file__).parent / "xinfer/ops/backends/ascend/xinfer_kernels_ascend.so"
        return library.is_file()


setup(distclass=NativeDistribution)
