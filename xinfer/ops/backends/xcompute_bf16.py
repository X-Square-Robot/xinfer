"""xcompute SM120 BF16 bindings for Wall-OSS. Stream-K GEMMs may differ at rounding level across runs."""

from __future__ import annotations

import os

import torch

from xinfer.ops import fused
from xinfer.ops.spec import Contract

UNIFIED = ("gemm", "gemm_glu", "qkv_proj_rope", "prepack_weight", "pack_rope_table", "AdaRms", "rmsnorm", "add_rmsnorm",
           "attention_kvcache", "attention_kvcache_backend", "denoise_output_euler")


def register(registry, library, backend: str, version: str) -> list[str]:
    """Register the operators the library and its Python package provide on ``sm120``; returns their names.

    Args:
        registry: the ``OpRegistry``.
        library: the loaded xcompute library (``xinfer.ops.backends.xcompute._Library``).
        backend: backend name to register under.
        version: version string of the implementations.

    Returns:
        The registered operator names; empty if the matching Python package is unavailable.
    """
    xc = _python_package(library)
    if xc is None:
        return []

    def sm120(op, numerics="tolerance", **kwargs):
        return registry.register(op, backend=backend, platforms=("sm120",), priority=10, version=version,
                                 contract=Contract(numerics=numerics, **kwargs))

    def kn(weight):
        """A ``[N, K]`` view of ``[K, N]`` storage as a weight of xcompute's ``kn`` layout, sharing the storage."""
        packed = xc.prepack_weight(weight, op="gemm", backend="cutlass_bf16_sm120")
        if packed.tensors["w"].data_ptr() != weight.data_ptr():
            raise ValueError("xcompute packed the GEMM weight into a copy")
        return packed

    @sm120("gemm_epilogue", deterministic=False)
    def _gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, residual_gate=None, row_class=None,
                       row_scale=None):
        if not weight.is_contiguous() and weight.t().is_contiguous():
            weight = kn(weight)
        return xc.gemm(x.contiguous(), weight, bias=bias, activation=activation, residual=residual,
                       residual_gate=residual_gate, row_class=row_class, row_scale=row_scale, backend="auto")

    @sm120("gemm_glu_epilogue")
    def _gemm_glu_epilogue(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
        k = x.shape[-1]
        prenorm = None if modulation is None else xc.AdaRms(scale=modulation[:, :k], shift=modulation[:, k:2 * k],
                                                              row_class=row_class, eps=float(eps))
        return xc.gemm_glu(x.contiguous(), weight, bias=bias, activation="silu", prenorm=prenorm, backend="auto")

    @sm120("qkv_rope_epilogue", deterministic=False)
    def _qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim):
        return xc.qkv_proj_rope(x.contiguous(), weight, bias, rope_table=xc.pack_rope_table(cos, sin),
                                num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim, backend="auto")

    @sm120("qkv_rope_kvcache", deterministic=False)
    def _qkv_rope_kvcache(x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads, head_dim):
        return xc.qkv_proj_rope(x.contiguous(), weight, bias, cos=cos, sin=sin, rotary_dim=2 * cos.shape[-1],
                                num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                                kv_cache=(k_cache, v_cache, int(start)), backend="auto")

    @sm120("rms_norm_cast", batch_invariant=True)
    def _rms_norm_cast(x, weight, eps):
        return xc.rmsnorm(x, weight, float(eps), True)

    @sm120("add_rms_norm", batch_invariant=True)
    def _add_rms_norm(x, residual, weight, eps):
        return xc.add_rmsnorm(residual, x, weight, float(eps), inference=True)

    listed = {}  # (entry, shapes, options) -> whether the entry's white list takes the call

    def takes(entry, probe, key, *args, **kwargs):
        if (entry, key) not in listed:
            try:
                probe(*args, **kwargs)
                listed[entry, key] = True
            except ValueError:
                listed[entry, key] = False
        return listed[entry, key]

    @sm120("prefix_kvcache_attention")
    def _prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, *, flash_mask=None, scale=None):
        options = dict(prefix_len=prefix_len, new_pos=int(new_pos), flash_mask=flash_mask, softmax_scale=scale)
        key = (q.shape, k_cache.shape, int(new_pos), flash_mask is not None)
        if not takes("kvcache", xc.attention_kvcache_backend, key, q, k_cache, v_cache, **options):
            return fused.prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, flash_mask=flash_mask,
                                                  scale=scale)
        b, m = q.shape[:2]
        return xc.attention_kvcache(q, k_cache, v_cache, **options).reshape(b, m, -1)

    @sm120("flow_velocity_euler", batch_invariant=True)
    def _flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
        return xc.denoise_output_euler(x, h, weight, dt=float(dt), dof_mask=dof_mask, padding_velocity=padding_velocity,
                                       pinned=None if pinned is None else pinned.to(torch.int32),
                                       pinned_values=pinned_values, out=torch.empty_like(x))

    return ["gemm_epilogue", "gemm_glu_epilogue", "qkv_rope_epilogue", "qkv_rope_kvcache", "rms_norm_cast",
            "add_rms_norm", "prefix_kvcache_attention", "flow_velocity_euler"]


def _python_package(library):
    """The xcompute Python package of ``library``'s build with the entries in ``UNIFIED``, or None."""
    from pathlib import Path

    os.environ.setdefault("XCOMPUTE_LIBRARY_PATH", library.path)
    os.environ.setdefault("XCOMPUTE_SKIP_AUTO_BUILD", "1")
    try:
        import xcompute
    except ImportError:
        return None
    if (Path(xcompute.KERNEL_LIB_PATH).resolve() != Path(library.path)
            or not all(hasattr(xcompute, name) for name in UNIFIED)):
        return None
    return xcompute
