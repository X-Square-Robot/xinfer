"""Tag wheels containing packaged native kernels for this Python and platform."""

from pathlib import Path

from setuptools import Distribution, setup


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        package = Path(__file__).parent / "xinfer"
        return any(
            library.is_file()
            for pattern in self.package_data.get("xinfer", ())
            if pattern.endswith(".so")
            for library in package.glob(pattern)
        )


setup(distclass=NativeDistribution)
