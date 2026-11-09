"""Transitional backend: xcompute kernels through TVM-FFI (design §14, §18.1).

Enabled by ``register(registry, library=...)`` or the ``XINFER_XCOMPUTE_LIBRARY``
environment variable (path to an ``xcompute.so``). The library is loaded as
is; no build is triggered. Kernels run on the caller's current torch stream.

SM120 (RTX 5090) uses xcompute's ``wallx_sm120`` module (xcompute
``0.1.0+wallx2``): the MXFP8 kernels the Wall-X 4B service ran, kept bit for
bit.

| operator | xcompute function | contract vs the torch reference |
|---|---|---|
| ``mxfp8_quantize`` | ``wallx_sm120_quantize_mxfp8`` | tolerance: built with fast-math, the scale exponent uses an approximate ``amax / 448``, so blocks at a power-of-two boundary may get the neighbouring scale |
| ``mxfp8_rmsnorm_quantize`` | ``wallx_sm120_rmsnorm_quant_mxfp8`` | tolerance: fast-math ``rsqrtf`` and scale division; BF16 weight only |
| ``mxfp8_silu_mul_quantize`` | ``wallx_sm120_silu_mul_quant_mxfp8`` | tolerance: fast-math ``__expf`` and division |
| ``mxfp8_gemm`` | ``wallx_sm120_mxfp8_gemm_bf16`` | tolerance: cuBLASLt block-scaled GEMM, accumulation order |
| ``mxfp8_gemm_rope`` | ``wallx_sm120_mxfp8_qkv_rope`` mode 0 | tolerance: SM120 CUTLASS 128x128x128, accumulation order, FMA in the epilogue |
| ``mxfp8_gemm_silu_mul`` | ``wallx_sm120_mxfp8_qkv_rope`` mode 1 | tolerance: as above |
| ``mxfp8_gemm_residual`` | ``wallx_sm120_oproj_resid_add_bf16`` | tolerance: as above; no bias |

The three CUTLASS GEMMs (Expert 0 of the Wall-X service) use fixed tiles
without split-K and need no workspace: each output row depends only on its
own input row (batch invariant), and they capture in graphs at any time.
Scale buffers are allocated with ``torch.empty``, as the service did: the
kernels leave the padding rows unwritten and the GEMMs never store the
outputs those rows produce.

The GEMM runner allocates an 8 MiB workspace per stream and autotunes each
shape on its first eager call (shared by all streams of the process); both
must happen before a stream captures a graph, which the engine guarantees by
running every new graph key eagerly first. Autotuning picks the fastest of
the cuBLASLt candidates, so the algorithm, and therefore the low bits, can
differ between processes; within a process every lane uses the same one.

xcompute builds without the ``wallx_sm120`` module can provide SM120 BF16 kernels, registered with the
unified entries of the build's Python package by ``xcompute_bf16``. They also export SM110 (Thor) MXFP8 kernels;
those are not registered until validated on Thor hardware.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from xinfer.ops import mxfp8
from xinfer.ops.backends import xcompute_bf16
from xinfer.ops.spec import Contract

BACKEND = "xcompute"
ENV = "XINFER_XCOMPUTE_LIBRARY"
WALLX_SM120 = ("wallx_sm120_quantize_mxfp8", "wallx_sm120_silu_mul_quant_mxfp8", "wallx_sm120_rmsnorm_quant_mxfp8",
               "wallx_sm120_mxfp8_gemm_bf16", "wallx_sm120_mxfp8_qkv_rope", "wallx_sm120_oproj_resid_add_bf16",
               "wallx_sm120_build_info")


class _Library:
    """A loaded ``xcompute.so`` TVM-FFI module with cached function handles.

    Args:
        path: path to the library; resolved, hashed and loaded as is.

    Attributes:
        path: resolved absolute path.
        sha256: hex SHA-256 of the library file.
        registered: operator names ``register`` registered from this library (set by ``register``).
        build_info: parsed ``wallx_sm120_build_info`` JSON (set by ``register`` when the module is present).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ImportError: ``tvm_ffi`` is not installed.
    """

    def __init__(self, path: str):
        import tvm_ffi

        self.path = str(Path(path).resolve(strict=True))
        self.sha256 = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        self._tvm_ffi = tvm_ffi
        self._module = tvm_ffi.load_module(self.path)
        self._fns = {}

    def has_function(self, name: str) -> bool:
        """Whether the library exports function ``name``."""
        try:
            self.function(name)
            return True
        except Exception:  # noqa: BLE001 - missing symbol
            return False

    def function(self, name):
        """The TVM-FFI function ``name`` (looked up once, then cached)."""
        if name not in self._fns:
            self._fns[name] = self._module.get_function(name)
        return self._fns[name]

    def call(self, name, *args):
        """Call function ``name`` with ``args`` on the current torch CUDA stream and return its result."""
        with self._tvm_ffi.use_torch_stream():
            return self.function(name)(*args)


def register(registry, library: str | None = None) -> _Library | None:
    """Register the xcompute implementations the library provides; returns it, or None.

    When the library exports every ``WALLX_SM120`` function, the seven MXFP8 operators are registered for
    ``sm120`` at priority 10 with tolerance contracts (all batch invariant except ``mxfp8_gemm``), versioned
    by the build-info arch and the library hash. A matching Python package with the unified BF16 entries also
    registers the fused Wall-OSS operators.

    Args:
        registry: the ``OpRegistry`` to register into.
        library: path to ``xcompute.so``; defaults to ``$XINFER_XCOMPUTE_LIBRARY``.

    Returns:
        The loaded ``_Library`` (``registered`` lists the registered operators, possibly none), or None when no
        library path is given.
    """
    path = library or os.environ.get(ENV)
    if not path:
        return None
    xcompute = _Library(path)
    xcompute.registered = []
    if all(xcompute.has_function(name) for name in WALLX_SM120):
        info = json.loads(str(xcompute.call("wallx_sm120_build_info")))
        version = f"wallx_sm120 {info.get('arch')} xcompute.so sha256:{xcompute.sha256[:12]}"
        xcompute.build_info = info

        def contract(**kwargs):
            """A tolerance ``Contract`` with the given runtime properties."""
            return Contract(numerics="tolerance", **kwargs)

        def sm120(op, **kwargs):
            """Registration decorator for ``op`` on ``sm120`` with this library's version and contract ``kwargs``."""
            return registry.register(op, backend=BACKEND, platforms=("sm120",), priority=10, version=version,
                                     contract=contract(**kwargs))

        def quantized(rows, k, device):
            """Uninitialized outputs of a quantizer: ``float8_e4m3fn [rows, k]`` and uint8 swizzled scales."""
            return (torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=device),
                    torch.empty(mxfp8.scale_elements(rows, k), dtype=torch.uint8, device=device))

        @sm120("mxfp8_quantize", batch_invariant=True)
        def _quantize(x, *, share_scale_128=True):
            """``mxfp8_quantize`` via ``wallx_sm120_quantize_mxfp8``.

            Args:
                x: BF16 ``[M, K]`` on CUDA (made contiguous).
                share_scale_128: one scale per 128 columns instead of per 32.

            Returns:
                ``(q, scales)``: ``float8_e4m3fn [M, K]`` and uint8 ``[scale_elements(M, K)]`` in the
                SWIZZLE_32_4_4 layout; the scales of padding rows are left unwritten.
            """
            q, scales = quantized(*x.shape, x.device)
            xcompute.call("wallx_sm120_quantize_mxfp8", q, scales, x.contiguous(), bool(share_scale_128))
            return q, scales

        @sm120("mxfp8_rmsnorm_quantize", batch_invariant=True)
        def _rmsnorm_quantize(x, weight, eps, *, share_scale_128=True):
            """``mxfp8_rmsnorm_quantize`` via ``wallx_sm120_rmsnorm_quant_mxfp8`` without the residual add.

            Args:
                x: BF16 ``[M, K]`` on CUDA (made contiguous).
                weight: BF16 ``[K]`` norm weight.
                eps: added to the mean square.
                share_scale_128: one scale per 128 columns instead of per 32.

            Returns:
                ``(q, scales)`` as in ``_quantize``.

            Raises:
                ValueError: ``weight`` is not BF16.
            """
            if weight.dtype != torch.bfloat16:
                raise ValueError(f"wallx_sm120 rmsnorm_quant takes a BF16 weight, got {weight.dtype}")
            x = x.contiguous()
            rows, k = x.shape
            q, scales = quantized(rows, k, x.device)
            # (quantized, scales, residual_output, input, residual, weight, logical_k, eps, add_residual, share)
            xcompute.call("wallx_sm120_rmsnorm_quant_mxfp8", q, scales, x, x, x, weight.contiguous(), k, float(eps),
                          False, bool(share_scale_128))
            return q, scales

        @sm120("mxfp8_silu_mul_quantize", batch_invariant=True)
        def _silu_mul_quantize(gate_up, *, share_scale_128=True):
            """``mxfp8_silu_mul_quantize`` via ``wallx_sm120_silu_mul_quant_mxfp8``.

            Args:
                gate_up: BF16 ``[M, 2K]`` ``[gate | up]`` on CUDA (made contiguous).
                share_scale_128: one scale per 128 columns instead of per 32.

            Returns:
                ``(q, scales)``: ``float8_e4m3fn [M, K]`` and the swizzled scales of an ``[M, K]`` matrix.
            """
            rows, width = gate_up.shape
            q, scales = quantized(rows, width // 2, gate_up.device)
            xcompute.call("wallx_sm120_silu_mul_quant_mxfp8", q, scales, gate_up.contiguous(), bool(share_scale_128))
            return q, scales

        @sm120("mxfp8_gemm")
        def _gemm(a_q, a_scales, w_q, w_scales, bias=None, *, fast_accum=False):
            """``mxfp8_gemm`` via ``wallx_sm120_mxfp8_gemm_bf16`` (cuBLASLt block-scaled GEMM).

            Not batch invariant: the algorithm is autotuned per shape on the first eager call.

            Args:
                a_q: ``float8_e4m3fn [M, K]`` activations.
                a_scales: swizzled E8M0 scales of ``a_q``.
                w_q: ``float8_e4m3fn [N, K]`` weight.
                w_scales: swizzled E8M0 scales of ``w_q``.
                bias: optional BF16 ``[N]``, applied in the epilogue.
                fast_accum: use the faster, less precise FP8 accumulation.

            Returns:
                BF16 ``[M, N]``.
            """
            out = torch.empty((a_q.shape[0], w_q.shape[0]), dtype=torch.bfloat16, device=a_q.device)
            xcompute.call("wallx_sm120_mxfp8_gemm_bf16", out, a_q, a_scales, w_q, w_scales, bias, bool(fast_accum))
            return out

        constants = {}

        def constant(kind, n, device):
            """Zero bias / empty rope table the mode-1 kernel requires (created on the first, eager, call)."""
            key = (kind, n, device)
            if key not in constants:
                constants[key] = (torch.zeros(n, dtype=torch.bfloat16, device=device) if kind == "bias"
                                  else torch.empty(0, dtype=torch.float32, device=device))
            return constants[key]

        @sm120("mxfp8_gemm_rope", batch_invariant=True)
        def _gemm_rope(a_q, a_scales, w_q, w_scales, bias, table, q_dim, k_dim, head_dim):
            """``mxfp8_gemm_rope`` via ``wallx_sm120_mxfp8_qkv_rope`` mode 0 (CUTLASS, fixed tiles, no split-K).

            Args:
                a_q: ``float8_e4m3fn [M, K]`` activations.
                a_scales: swizzled E8M0 scales of ``a_q``.
                w_q: ``float8_e4m3fn [N, K]`` QKV weight, rows in ``rope_pair_permutation`` order.
                w_scales: swizzled E8M0 scales of ``w_q``.
                bias: BF16 ``[N]`` in the same row order, or None (a cached zero bias is passed).
                table: FP32 ``[head_dim/4, M, 4]`` rope table (``mxfp8.rope_table``).
                q_dim: width of q.
                k_dim: width of k.
                head_dim: head dimension.

            Returns:
                BF16 ``[M, N]`` in the standard ``[q | k | v]`` layout, q and k rotated.
            """
            n = w_q.shape[0]
            out = torch.empty((a_q.shape[0], n), dtype=torch.bfloat16, device=a_q.device)
            bias = constant("bias", n, a_q.device) if bias is None else bias
            xcompute.call("wallx_sm120_mxfp8_qkv_rope", out, a_q, a_scales, w_q, w_scales, bias, table.contiguous(),
                          int(q_dim), int(k_dim), int(head_dim), 0)
            return out

        @sm120("mxfp8_gemm_silu_mul", batch_invariant=True)
        def _gemm_silu_mul(a_q, a_scales, w_q, w_scales, bias=None):
            """``mxfp8_gemm_silu_mul`` via ``wallx_sm120_mxfp8_qkv_rope`` mode 1 (CUTLASS, fixed tiles, no split-K).

            Args:
                a_q: ``float8_e4m3fn [M, K]`` activations.
                a_scales: swizzled E8M0 scales of ``a_q``.
                w_q: ``float8_e4m3fn [N, K]`` gate/up weight, rows in ``gate_up_interleave`` order.
                w_scales: swizzled E8M0 scales of ``w_q``.
                bias: optional BF16 ``[N]`` in the same row order (None passes a cached zero bias).

            Returns:
                BF16 ``[M, N/2]`` ``silu(gate) * up``.
            """
            n = w_q.shape[0]
            out = torch.empty((a_q.shape[0], n // 2), dtype=torch.bfloat16, device=a_q.device)
            bias = constant("bias", n, a_q.device) if bias is None else bias
            xcompute.call("wallx_sm120_mxfp8_qkv_rope", out, a_q, a_scales, w_q, w_scales, bias,
                          constant("table", 0, a_q.device), 0, 0, 4, 1)
            return out

        @sm120("mxfp8_gemm_residual", batch_invariant=True)
        def _gemm_residual(a_q, a_scales, w_q, w_scales, residual, bias=None):
            """``mxfp8_gemm_residual`` via ``wallx_sm120_oproj_resid_add_bf16`` (CUTLASS, fixed tiles, no split-K).

            Args:
                a_q: ``float8_e4m3fn [M, K]`` activations.
                a_scales: swizzled E8M0 scales of ``a_q``.
                w_q: ``float8_e4m3fn [N, K]`` weight.
                w_scales: swizzled E8M0 scales of ``w_q``.
                residual: BF16 ``[M, N]`` (made contiguous), added to the FP32 accumulator.
                bias: must be None.

            Returns:
                BF16 ``[M, N]``.

            Raises:
                NotImplementedError: ``bias`` is given (the kernel has no bias).
            """
            if bias is not None:
                raise NotImplementedError("wallx_sm120 oproj_resid_add_bf16 has no bias")
            residual = residual.contiguous()
            out = torch.empty_like(residual)
            xcompute.call("wallx_sm120_oproj_resid_add_bf16", out, a_q, a_scales, w_q, w_scales, residual)
            return out

        xcompute.registered += ["mxfp8_quantize", "mxfp8_rmsnorm_quantize", "mxfp8_silu_mul_quantize", "mxfp8_gemm",
                           "mxfp8_gemm_rope", "mxfp8_gemm_silu_mul", "mxfp8_gemm_residual"]
    xcompute.registered += xcompute_bf16.register(registry, xcompute, BACKEND,
                                                    f"bf16_sm120 xcompute.so sha256:{xcompute.sha256[:12]}")
    return xcompute
