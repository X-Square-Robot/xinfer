"""Tag wheels containing locally built native libraries for this Python/platform."""

from pathlib import Path

from setuptools import Distribution, setup


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        package = Path(__file__).parent / "xinfer"
        return (any(package.glob("_*.so"))
                or any((package / "ops/backends").glob("*/xinfer_kernels*.so")))


setup(distclass=NativeDistribution)
