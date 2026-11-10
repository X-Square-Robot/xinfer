"""Built-in operator backends: one package per backend (hardware), one module per operator category.

``cuda/``: ``attn``, ``gemm``, ``norm`` and ``flow``. Each module imports the kernel libraries it calls and declares
its implementations at module level with ``xinfer.ops.register(..., requires=<library>)``; an implementation whose
library did not import is not registered. Transitional backends that bind existing libraries live here; the kernel
package (xinfer-kernels) will register through the ``xinfer.ops`` entry point group.
"""
