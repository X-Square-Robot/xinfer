"""xcompute BF16 bindings for Wall-OSS. Stream-K GEMMs may differ at rounding level across runs."""

from __future__ import annotations

import os

import torch

from xinfer.ops import fused
from xinfer.ops.backends.xcompute import BACKEND

UNIFIED = ("gemm", "gemm_glu", "qkv_proj_rope", "prepack_weight", "selected_backend", "pack_rope_table", "AdaRms",
           "rmsnorm", "add_rmsnorm", "attention_kvcache", "attention_kvcache_backend", "denoise_output_euler")
LIBRARY_ENTRIES = ("build_target_archs", "gemm_envelope_table")


def register_unified(registry, library) -> None:
    """Register Wall-OSS BF16 operators for the architectures targeted by the xcompute build.

    Args:
        registry: the ``OpRegistry``.
        library: the loaded xcompute library. Its matching Python package must provide the entries in ``UNIFIED``.
    """
    xc = _python_package(library)
    if xc is None:
        return
    platforms = tuple(f"sm{arch}" for arch in str(library.function("build_target_archs")()).split(",")
                      if arch.isdigit())
    if not platforms:
        return

    kernel = registry.implementer(BACKEND, platforms=platforms,
                                  version=f"unified xcompute.so sha256:{library.sha256[:12]}")

    cached = {}

    def listed_weight(op, x, weight, **options):
        """``weight``, or its xcompute packing when the call would run the generic path on the raw ``[N, K]`` weight
        and runs a listed kernel on the packed one.

        A weight is packed once, on its first call that would run the generic path (eager: the engine runs every new
        graph key before capturing it), and the packing is kept only if that call runs a listed kernel with it.
        """
        if xc.selected_backend(op, x, weight, **options) != "generic":
            return weight
        key = ("packed", op, weight.data_ptr(), tuple(weight.shape))
        if key not in cached:
            packed = xc.prepack_weight(weight, op=op)
            cached[key] = packed if xc.selected_backend(op, x, packed, **options) != "generic" else None
        packed = cached[key]
        return weight if packed is None or xc.selected_backend(op, x, packed, **options) == "generic" else packed

    def kn(weight):
        """A ``[N, K]`` view of ``[K, N]`` storage as a weight of xcompute's ``kn`` layout, sharing the storage."""
        packed = xc.prepack_weight(weight, op="gemm", layout="kn")
        if packed.tensors["w"].data_ptr() != weight.data_ptr():
            raise ValueError("xcompute packed the GEMM weight into a copy")
        return packed

    @kernel("gemm_epilogue", deterministic=False)
    def _gemm_epilogue(x, weight, bias=None, *, activation=None, residual=None, residual_gate=None, row_class=None,
                       row_scale=None):
        x = x.contiguous()
        options = dict(bias=bias, activation=activation, residual=residual, residual_gate=residual_gate,
                       row_class=row_class, row_scale=row_scale)
        if not weight.is_contiguous() and weight.t().is_contiguous():
            weight = kn(weight)
        else:
            weight = listed_weight("gemm", x, weight, **options)
        return xc.gemm(x, weight, **options, backend="auto")

    @kernel("gemm_glu_epilogue")
    def _gemm_glu_epilogue(x, weight, *, bias=None, modulation=None, row_class=None, eps=1e-6):
        k = x.shape[-1]
        prenorm = None if modulation is None else xc.AdaRms(scale=modulation[:, :k], shift=modulation[:, k:2 * k],
                                                              row_class=row_class, eps=float(eps))
        x, options = x.contiguous(), dict(bias=bias, activation="silu", prenorm=prenorm)
        return xc.gemm_glu(x, listed_weight("gemm_glu", x, weight, **options), **options, backend="auto")

    @kernel("qkv_rope_epilogue", deterministic=False)
    def _qkv_rope_epilogue(x, weight, bias, cos, sin, num_heads, head_dim):
        return xc.qkv_proj_rope(x.contiguous(), weight, bias, rope_table=xc.pack_rope_table(cos, sin),
                                num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim, backend="auto")

    @kernel("qkv_rope_kvcache", deterministic=False)
    def _qkv_rope_kvcache(x, weight, bias, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads, head_dim):
        return xc.qkv_proj_rope(x.contiguous(), weight, bias, cos=cos, sin=sin, rotary_dim=2 * cos.shape[-1],
                                num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                                kv_cache=(k_cache, v_cache, int(start)), backend="auto")

    @kernel("rms_norm")
    def _rms_norm(x, weight, eps):
        return xc.rmsnorm(x, weight, float(eps), True)

    @kernel("add_rms_norm")
    def _add_rms_norm(x, residual, weight, eps):
        return xc.add_rmsnorm(residual, x, weight, float(eps), inference=True)

    listed = {}  # attention cache shapes and options supported by the kernel

    def takes(entry, probe, key, *args, **kwargs):
        if (entry, key) not in listed:
            try:
                probe(*args, **kwargs)
                listed[entry, key] = True
            except ValueError:
                listed[entry, key] = False
        return listed[entry, key]

    @kernel("prefix_kvcache_attention")
    def _prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, *, flash_mask=None, scale=None):
        options = dict(prefix_len=prefix_len, new_pos=int(new_pos), flash_mask=flash_mask, softmax_scale=scale)
        key = (q.shape, k_cache.shape, int(new_pos), flash_mask is not None)
        if not takes("kvcache", xc.attention_kvcache_backend, key, q, k_cache, v_cache, **options):
            return fused.prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, flash_mask=flash_mask,
                                                  scale=scale)
        b, m = q.shape[:2]
        return xc.attention_kvcache(q, k_cache, v_cache, **options).reshape(b, m, -1)

    @kernel("flow_velocity_euler")
    def _flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
        return xc.denoise_output_euler(x, h, weight, dt=float(dt), dof_mask=dof_mask, padding_velocity=padding_velocity,
                                       pinned=None if pinned is None else pinned.to(torch.int32),
                                       pinned_values=pinned_values, out=torch.empty_like(x))


def _python_package(library):
    """The xcompute Python package of ``library``'s build with the entries in ``UNIFIED``, or None.

    None also when ``library`` lacks a function in ``LIBRARY_ENTRIES``: the package then comes from another build.
    """
    from pathlib import Path

    os.environ.setdefault("XCOMPUTE_LIBRARY_PATH", library.path)
    os.environ.setdefault("XCOMPUTE_SKIP_AUTO_BUILD", "1")
    try:
        import xcompute
    except ImportError:
        return None
    if (Path(xcompute.KERNEL_LIB_PATH).resolve() != Path(library.path)
            or not all(hasattr(xcompute, name) for name in UNIFIED)
            or not all(library.has_function(name) for name in LIBRARY_ENTRIES)):
        return None
    return xcompute
