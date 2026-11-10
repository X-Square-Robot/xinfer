"""flash-attention 2 backend for the attention operators (CUDA).

Registered by ``ops.registry.load_plugins()`` when the ``flash_attn`` package
imports (flash-attn 2.8.3.post1):

| operator | flash-attn call |
|---|---|
| ``varlen_attention`` | ``flash_attn_2_cuda.varlen_fwd`` (the ``flash_attn_varlen_func`` kernel), segments from ``cu_seqlens``: the Wall-X 4B service's ViT call |
| ``causal_attention`` | ``flash_attn_with_kvcache`` with the keys as the cache (``cache_seqlens = L``, ``cache_batch_idx``), ``causal=True``, ``num_splits=1``: the kernel of the service's Stage-1 call |
| ``action_attention`` | ``flash_attn_with_kvcache`` over ``[prompt | own]`` keys, one decode-style row per query (``cache_batch_idx``, ``seqlen_q = 1``, per-row ``cache_seqlens``), ``num_splits=4`` |
| ``prefix_attention`` | ``flash_attn_with_kvcache`` with the keys as the cache, ``cache_seqlens`` = valid length, non-causal, ``num_splits=1`` |
| ``kvcache_attention`` | ``flash_attn_with_kvcache`` appending ``k``/``v`` at ``cache_seqlens`` in the kernel, one sequence per batch row (``cache_leftpad``), non-causal, ``num_splits=4`` |

The first two reproduce the service's attention bit for bit;
``flash_attn_func`` uses another kernel configuration and would not.
For ``action_attention`` the service made two ``num_splits=1`` calls (all
keys, and ``P + pinned`` keys for the pinned rows) with only 16 CTAs on the
Wall-X shapes, which is slower than the torch reference. The per-row form
with a fixed split count is about 4x faster (22 vs 92 µs per call at B1).
Left-padded prompts (``prompt_lengths``) use ``cache_leftpad``.

Contracts against the torch references: tolerance (different accumulation
order), deterministic, graph safe. ``varlen_attention``
takes ``cu_seqlens`` as a host sequence; its device copy is created on the
first (eager) call for each layout and reused, also inside captured graphs.
"""

from __future__ import annotations

import torch

BACKEND = "flash_attn"
ACTION_SPLITS = 4  # fixed split-KV count of action_attention


def available() -> bool:
    """Whether the ``flash_attn`` package and its ``flash_attn_2_cuda`` extension import."""
    try:
        import flash_attn  # noqa: F401
        import flash_attn_2_cuda  # noqa: F401
    except Exception:  # noqa: BLE001 - not installed or no binary for this CUDA
        return False
    return True


def register(registry) -> list[str]:
    """Register the flash-attn implementations (the backend protocol of ``OpRegistry.load_plugins``).

    ``varlen_attention``, ``causal_attention``, ``action_attention``, ``prefix_attention`` and
    ``kvcache_attention`` on CUDA platforms, priority 10, version ``"flash-attn <version>"``, each with a
    tolerance, deterministic, graph-safe contract.

    Args:
        registry: the ``OpRegistry`` to register into.

    Returns:
        The registered operator names; none when flash-attn is not importable.
    """
    if not available():
        return []
    import flash_attn
    import flash_attn_2_cuda

    implements = registry.implementer(BACKEND, platforms=("cuda",), version=f"flash-attn {flash_attn.__version__}")

    segments = {}

    def device_segments(cu_seqlens, device):
        """Device ``int32`` boundaries and the longest segment for a host layout (cached per layout).

        The device tensor is created on the first call for a layout, which must be eager (not under capture).

        Args:
            cu_seqlens: segment boundaries as a host sequence or tensor.
            device: device of the boundary tensor.

        Returns:
            ``(cu_seqlens_device, longest)``: int32 ``[S + 1]`` boundaries on ``device`` and the longest segment length.
        """
        bounds = tuple(int(b) for b in (cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else cu_seqlens))
        key = (bounds, device)
        if key not in segments:
            segments[key] = (torch.tensor(bounds, dtype=torch.int32, device=device),
                             max(e - s for s, e in zip(bounds[:-1], bounds[1:])))
        return segments[key]

    @implements("varlen_attention")
    def _varlen(q, k, v, cu_seqlens, *, scale=None):
        """``varlen_attention`` via ``flash_attn_2_cuda.varlen_fwd`` (the Wall-X 4B service's ViT call).

        Non-causal attention within each segment, no dropout, no window, no softcap.

        Args:
            q: ``[T, H, D]`` packed queries (FP16/BF16).
            k: ``[T, KV, D]`` packed keys.
            v: ``[T, KV, D]`` packed values.
            cu_seqlens: segment boundaries as a host sequence (or tensor); see ``device_segments``.
            scale: softmax scale; None for ``1/sqrt(D)``.

        Returns:
            ``[T, H, D]`` in ``q.dtype``.
        """
        cu_seqlens_device, longest = device_segments(cu_seqlens, q.device)
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        # (q, k, v, out, cu_q, cu_k, seqused_k, leftpad_k, block_table, alibi, max_q, max_k, dropout,
        #  scale, zero_tensors, causal, window_left, window_right, softcap, return_softmax, generator)
        return flash_attn_2_cuda.varlen_fwd(q, k, v, None, cu_seqlens_device, cu_seqlens_device, None, None, None,
                                            None, longest, longest, 0.0, scale, False, False, -1, -1, 0.0, False,
                                            None)[0]

    @implements("causal_attention")
    def _causal(q, k, v, *, scale=None):
        """``causal_attention`` via ``flash_attn_with_kvcache`` (the Wall-X 4B service's Stage-1 kernel).

        The service appended ``k``/``v`` to empty caches with ``num_splits=1``. Here ``k``/``v`` are the
        caches (``cache_seqlens = L``), so no key is written; ``cache_batch_idx`` keeps flash-attn on the
        same split-KV kernel, which gives the service's bits without its per-CTA rewrite of the keys
        (68 -> 28 µs at L = 768, 16/2/128 heads, RTX 5090). Without ``cache_batch_idx`` flash-attn runs
        another kernel configuration and the bits differ.

        Args:
            q: ``[B, L, H, D]`` queries (FP16/BF16).
            k: ``[B, L, KV, D]`` keys.
            v: ``[B, L, KV, D]`` values.
            scale: softmax scale; None for ``1/sqrt(D)``.

        Returns:
            ``[B, L, H, D]`` in ``q.dtype``.
        """
        batch_size, length = q.shape[:2]
        rows = torch.arange(batch_size, dtype=torch.int32, device=q.device)
        return flash_attn.flash_attn_with_kvcache(q, k, v, cache_seqlens=length, cache_batch_idx=rows,
                                                  softmax_scale=scale, causal=True, num_splits=1)

    @implements("prefix_attention")
    def _prefix(q, k, v, valid, *, scale=None):
        """``prefix_attention`` via ``flash_attn_with_kvcache`` with the keys as the cache.

        ``cache_seqlens = valid`` limits every query row to its row's valid keys (non-causal,
        ``num_splits=1``); padded query rows attend to the same keys.

        Args:
            q: ``[B, S, H, D]`` queries (FP16/BF16).
            k: ``[B, S, KV, D]`` keys.
            v: ``[B, S, KV, D]`` values.
            valid: ``[B]`` integer tensor, the number of leading valid positions per row.
            scale: softmax scale; None for ``1/sqrt(D)``.

        Returns:
            ``[B, S, H, D]`` in ``q.dtype``.
        """
        return flash_attn.flash_attn_with_kvcache(q, k, v, cache_seqlens=valid.reshape(-1).to(torch.int32),
                                                  softmax_scale=scale, causal=False, num_splits=1)

    @implements("kvcache_attention")
    def _kvcache(q, k, v, k_cache, v_cache, cache_start, cache_valid, *, scale=None):
        """``kvcache_attention`` as one ``flash_attn_with_kvcache`` call that appends ``k``/``v`` in the kernel.

        Each batch row is one sequence: ``cache_leftpad = cache_start``, ``cache_seqlens = start + valid`` (the
        append position; the kernel writes ``k``/``v`` there and attends to ``[start, start + valid + L)``),
        non-causal, ``num_splits = ACTION_SPLITS``. A row's split-KV boundaries follow its own key range.

        Args:
            q: ``[B, L, H, D]`` queries (FP16/BF16).
            k: ``[B, L, KV, D]`` own keys.
            v: ``[B, L, KV, D]`` own values.
            k_cache: ``[B, S, KV, D]`` key cache, last dim contiguous; modified in place.
            v_cache: ``[B, S, KV, D]`` value cache; modified in place.
            cache_start: ``[B]`` integer tensor, each row's first cached column.
            cache_valid: ``[B]`` integer tensor, each row's number of cached keys.
            scale: softmax scale; None for ``1/sqrt(D)``.

        Returns:
            ``[B, L, H, D]`` in ``q.dtype``.
        """
        start = cache_start.reshape(-1).to(torch.int32)
        return flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v,
                                                  cache_seqlens=start + cache_valid.reshape(-1).to(torch.int32),
                                                  cache_leftpad=start, softmax_scale=scale, causal=False,
                                                  num_splits=ACTION_SPLITS)

    @implements("action_attention")
    def _action(q, k, v, k_prompt, v_prompt, num_pinned, prompt_lengths=None, *, scale=None):
        """``action_attention`` as one decode-style ``flash_attn_with_kvcache`` call per query row.

        Builds per-batch caches ``[prompt | own]`` (``[B, P + L, KV, D]``) and runs the ``B * L`` query rows as
        a batch of single-token queries: ``cache_batch_idx`` maps each to its batch row, ``cache_seqlens`` ends
        at ``P + num_pinned[b]`` for pinned rows and ``P + L`` otherwise, and ``cache_leftpad`` skips the left
        padding of short prompts. ``num_splits`` is the constant ``ACTION_SPLITS``, so a row's split-KV
        boundaries depend on its own key range only.

        Args:
            q: ``[B, L, H, D]`` queries of the own tokens (FP16/BF16).
            k: ``[B, L, KV, D]`` own keys.
            v: ``[B, L, KV, D]`` own values.
            k_prompt: ``[B, KV, P, D]`` prompt keys (head-major).
            v_prompt: ``[B, KV, P, D]`` prompt values.
            num_pinned: ``[B]`` integer tensor, the number of pinned rows per batch row.
            prompt_lengths: optional ``[B]`` integer tensor; row ``b``'s prompt is the last ``prompt_lengths[b]``
                of the ``P`` columns.
            scale: softmax scale; None for ``1/sqrt(D)``.

        Returns:
            ``[B, L, H, D]`` in ``q.dtype``.
        """
        b, n, heads, d = q.shape
        p = k_prompt.shape[2]
        k_cache = torch.cat([k_prompt.transpose(1, 2), k], dim=1)  # [B, P + L, KV, D]
        v_cache = torch.cat([v_prompt.transpose(1, 2), v], dim=1)
        rows = torch.arange(n, device=q.device)[None]
        num_pinned = num_pinned.reshape(b, 1)
        own = torch.where(rows < num_pinned, num_pinned, torch.full_like(rows, n))  # own keys each row sees
        ends = (own + p).to(torch.int32).reshape(-1)  # cache_seqlens: absolute end index
        batch_of_row = torch.arange(b, dtype=torch.int32, device=q.device).repeat_interleave(n)
        leftpad = None
        if prompt_lengths is not None:  # left-padded prompts start at P - length
            leftpad = (p - prompt_lengths.reshape(b).to(torch.int32)).repeat_interleave(n)
        out = flash_attn.flash_attn_with_kvcache(q.reshape(b * n, 1, heads, d), k_cache, v_cache,
                                                 cache_seqlens=ends, cache_batch_idx=batch_of_row,
                                                 cache_leftpad=leftpad, softmax_scale=scale, causal=False,
                                                 num_splits=ACTION_SPLITS)
        return out.reshape(b, n, heads, d)

    return registry.backend_ops(BACKEND)
