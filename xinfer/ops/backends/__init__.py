"""Built-in operator backends: one package per backend (hardware), one module per operator category.

``cuda/``: ``attn``, ``gemm``, ``norm``, ``flow`` and ``position``, with the kernels they call in ``cuda/src``
(built into ``xinfer_kernels.so``, loaded by ``cuda/_native.py``) and, per GPU architecture, the list of GEMM kernels
and the shapes each one takes (``cuda/gemm_sm120.py``). Modules declare their implementations at module level with
``xinfer.ops.register(..., requires=<library>)``; an implementation whose library did not load is not registered.
``ascend/``: the same shape in-tree — the kernels in ``ascend/src`` (built into ``xinfer_kernels_ascend.so``,
loaded by ``ascend/_native.py``, with the per-SoC GEMM kernel list ``ascend/gemm_950dt.py``) — and, until the
per-kernel glue faces land, the transition family libraries of the delivered stacks
(``src/build-libraries.sh``, loaded by the same ``_native.py``). Packages outside XInfer register through the
``xinfer.ops`` entry point group.
"""
