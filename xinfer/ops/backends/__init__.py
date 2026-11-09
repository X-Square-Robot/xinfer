"""Built-in operator backends.

Backends register implementations into ``xinfer.ops.registry``. Transitional
backends that bind existing libraries live here; the kernel package
(xinfer-kernels) will register through the ``xinfer.ops`` entry point group.
"""
