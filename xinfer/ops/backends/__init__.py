"""Built-in operator backends: one package per backend (hardware), one module per operator category.

``cuda/``: ``attn``, ``gemm``, ``norm``, ``flow`` and ``position``, with the kernels they call in ``cuda/src``
(built into ``xinfer_kernels.so``, loaded by ``cuda/_native.py``) and, per GPU architecture, the list of GEMM kernels
and the shapes each one takes (``cuda/gemm_sm120.py``). Modules declare their implementations at module level with
``xinfer.ops.register(..., requires=<library>)``; an implementation whose library did not load is not registered.
``ascend/``: the kernels in ``ascend/src`` build into ``xinfer_kernels_ascend.so`` and load through
``ascend/_native.py``. Category wrappers select the existing launch adapters for supported layouts and use
reference implementations for other layouts. Packages outside XInfer register through the ``xinfer.ops``
entry point group.
"""
