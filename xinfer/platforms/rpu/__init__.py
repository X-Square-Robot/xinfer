"""RPU platform capabilities and explicit-completion runtime ownership.

Resolving ``rpu[:0]`` is board-free and never imports torch or the vendor SDK.
The explicit-completion capability selects ``xinfer.engine.rpu`` through
``xinfer.entrypoints._factory.create_engine``. Native buffers, prepared
operations and completion ownership remain in this platform package.
"""

from .context import RpuExecutionContext
from .execution import RpuBoundExecution, RpuCompletion, RpuExecution
from .platform import rpu_platform
from .resources import RpuBuffer, RpuBusyError, RpuOperation
from .resources import _Resource as _Resource

__all__ = [
    "rpu_platform",
    "RpuBusyError",
    "RpuCompletion",
    "RpuBuffer",
    "RpuOperation",
    "RpuExecution",
    "RpuBoundExecution",
    "RpuExecutionContext",
]
