"""Internal Ascend ACLNN FIA V5 compositions for Wall-style attention."""

from __future__ import annotations

import math
from itertools import pairwise
from numbers import Integral, Real
from threading import Lock

import torch
from xinfer.ops.backends.ascend.core import CORE
from xinfer.ops.backends.ascend._cann_action_scope import action_scope, memo_action_metadata
try:  # decodeattn-r4: opt-in decode-family kernel pair (XINFER_DECODEATTN_R4)
    from xinfer.ops.backends.ascend import decodeattn_r4 as _decodeattn_r4
except Exception:  # pragma: no cover - the route stays off without the lib
    _decodeattn_r4 = None
try:  # copykill-r2 integration flags (XINFER_COPYKILL_R2, default off)
    from xinfer.ops.backends.ascend.copykill_r2 import enabled as _copykill_enabled
except Exception:  # pragma: no cover - optional mount

    def _copykill_enabled(_family):
        return False

_MAX_I32 = 2**31 - 1


def _check_qkv(q, k, v, rank):
    if (
        q.ndim != rank
        or k.ndim != rank
        or k.shape != v.shape
        or q.device.type != "npu"
        or any(t.device != q.device or t.dtype != torch.bfloat16 for t in (q, k, v))
        or not all(0 < dim <= _MAX_I32 for dim in (*q.shape, *k.shape))
        or q.shape[-1] != k.shape[-1]
        or q.shape[-1] > 512
        or q.shape[-2] % k.shape[-2]
    ):
        raise ValueError("CANN attention requires compatible BF16 Q/K/V and positive dimensions on one NPU")


def _check_tensor(tensor, q, name, shape, dtype):
    if tensor.device != q.device or tensor.dtype != dtype or tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {q.device}")


def _check_gate(gate, q, rank):
    if q.ndim != rank or gate is None:
        raise ValueError(f"gated CANN attention requires rank-{rank} Q and an output gate")
    _check_tensor(gate, q, "gate", (*q.shape[:-2], q.shape[-2] * q.shape[-1]), torch.bfloat16)


def _scale(scale, dim):
    if scale is None:
        return dim**-0.5
    if not isinstance(scale, Real) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("attention scale must be a positive finite host number")
    return float(scale)


class CannAttention:
    """Reuse vendor attention for cache, causal, and packed vision calls.

    Warm up on each device and stream with the shapes and static parameters
    that will be captured. Keep this instance alive while its graphs or
    submissions are live. Returned tensors have their own storage.
    """

    action_scope = staticmethod(action_scope)

    def __init__(self):
        self._bnsd_query = getattr(CORE, "cann_fia_v5_probe_workspace_size", None)
        self._bnsd_launch = getattr(CORE, "cann_fia_v5_probe", None)
        self._tnd_query = getattr(CORE, "cann_fia_vision_probe_workspace_size", None)
        self._tnd_launch = getattr(CORE, "cann_fia_vision_probe", None)
        if not all(callable(entry) for entry in
                   (self._bnsd_query, self._bnsd_launch, self._tnd_query, self._tnd_launch)):
            # Absorbed route (ffi_surface_alignment_v22.md C class): the
            # production library keeps the FIA V5 cores as internal
            # composition primitives without the cann_* FFI exports, so bind
            # their raw C-ABI launchers through ctypes. Builds without either
            # entry leave the pairs unbound and the capability probes below
            # degrade to the canonical attention entries as before.
            from xinfer.ops.backends.ascend import _cann_primitives

            bnsd = _cann_primitives.bnsd_pair()
            tnd = _cann_primitives.tnd_pair()
            self._bnsd_query, self._bnsd_launch = bnsd if bnsd else (None, None)
            self._tnd_query, self._tnd_launch = tnd if tnd else (None, None)
        self._buffers = {}
        self._lock = Lock()

    def _key(self, route, q, k, scale, *metadata):
        stream = torch.npu.current_stream(q.device).npu_stream
        return (route, q.device, stream, tuple(q.shape), tuple(k.shape), scale, *metadata)

    def _warmed(self, key, create):
        if key not in self._buffers:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("warm up CANN attention on this stream and shape before graph capture")
            self._buffers[key] = create()
        return self._buffers[key]

    def _run(self, q, k, v, key, parameters, query, launch):
        if query is None or launch is None:
            raise RuntimeError("the unified core library lacks the CANN FIA launchers")
        out = torch.empty_like(q)

        def allocate_workspace():
            size = query(out, q, k, v, *parameters)
            return torch.empty(size, dtype=torch.uint8, device=q.device)

        workspace = self._warmed(("workspace", *key), allocate_workspace)
        launch(out, workspace, q, k, v, *parameters)
        return out

    def prefix_kvcache_attention(self, q, k_cache, v_cache, prefix_len, new_pos, *, flash_mask=None, scale=None,
                                all_keep=False, return_bnsd=False):
        """Attend to the visible cache prefix and new rows, returning BF16 [B,M,H*D].

        ``all_keep`` (r4): a host guarantee that every cache row is visible
        (prefix_len == L - M, new_pos == prefix_len, no flash mask) -- the
        wall_oss flow case. The drop-mask build and the FIA mask read are
        skipped (``attenMask`` null, the pre/next-token window stays full).
        ``return_bnsd`` (r4): return the attention output in its native BNSD
        [B, H, M, D] buffer instead of a packed [B, M, H*D] copy, for callers
        whose projection reads the BNSD layout directly.
        """
        _check_qkv(q, k_cache, v_cache, 4)
        batch, rows, heads, dim = q.shape
        length = k_cache.shape[1]
        if k_cache.shape[0] != batch or not isinstance(new_pos, Integral) or not 0 <= new_pos <= length - rows:
            raise ValueError("cache batch or host append position is invalid")
        new_pos = int(new_pos)
        _check_tensor(prefix_len, q, "prefix_len", (batch,), torch.int32)
        if flash_mask is not None:
            _check_tensor(flash_mask, q, "flash_mask", (batch, 1, length, 2), torch.int32)
        scale = _scale(scale, dim)
        if all_keep and flash_mask is not None:
            raise ValueError("all_keep cannot be combined with a flash mask")
        if (
            _decodeattn_r4 is not None
            and not all_keep
            and _decodeattn_r4.applies(q, k_cache, v_cache, prefix_len, new_pos, flash_mask, scale)
        ):
            return _decodeattn_r4.prefix_kvcache_attention(
                q, k_cache, v_cache, prefix_len, new_pos, flash_mask, scale, return_bnsd
            )
        key = self._key("cache", q, k_cache, scale, all_keep)

        def positions():
            columns = torch.arange(length, device=q.device)[None, None, None, :]
            query_rows = torch.arange(rows, device=q.device)[None, None, :, None]
            return columns, query_rows

        def make_drop_mask():
            columns, query_rows = self._warmed(("positions", *key), positions)
            keep = (columns < prefix_len.reshape(batch, 1, 1, 1)) | (
                (columns >= new_pos) & (columns < new_pos + rows)
            )
            if flash_mask is not None:
                lts = flash_mask[..., 0].unsqueeze(-2)
                ute = flash_mask[..., 1].unsqueeze(-2)
                keep = keep & (query_rows < lts) & (query_rows >= ute)
            return (~keep.expand(batch, 1, rows, length)).contiguous()

        # Native workspace belongs to this stream. Serialize host submission
        # when two callers share it.
        with self._lock:
            drop_mask = (None if all_keep else
                         memo_action_metadata(
                             ("drop_mask", id(self), key, new_pos), (prefix_len, flash_mask), make_drop_mask
                         ))
            # Feed ACLNN's BNSD layout without per-call copies where the
            # operator takes it: the wall-style caches arrive as permuted
            # (B, L, KV, D) views of contiguous (B, KV, L, D) buffers, so the
            # transpose back to BNSD is a view. The query and the output keep
            # one small transposed copy each: FIA V5 rejects a strided query
            # descriptor (probe 2026-11-11, error 561103) and does not honor
            # strided output writes (probe 2026-11-11 r2, wrong values).
            def cache_bnsd(tensor):
                view = tensor.transpose(1, 2)
                return view if view.is_contiguous() else view.contiguous()

            query = q.transpose(1, 2)
            if not query.is_contiguous():
                if _copykill_enabled("k7"):
                    def qbuf():
                        return torch.empty((batch, heads, rows, dim), dtype=q.dtype, device=q.device)
                    qbuf_v = self._warmed(("qbuf", *key), qbuf)
                    qbuf_v.copy_(query)
                    query = qbuf_v
                else:
                    query = query.contiguous()
            keys, values = cache_bnsd(k_cache), cache_bnsd(v_cache)
            out = torch.empty((batch, heads, rows, dim), dtype=q.dtype, device=q.device)

            def allocate_workspace():
                size = self._bnsd_query(out, query, keys, values, drop_mask, scale, 2, 0)
                return torch.empty(size, dtype=torch.uint8, device=q.device)

            workspace = self._warmed(("workspace", *key), allocate_workspace)
            self._bnsd_launch(out, workspace, query, keys, values, drop_mask, scale, 2, 0)
        if return_bnsd:
            return out
        return out.transpose(1, 2).reshape(batch, rows, heads * dim)

    def gated_kvcache_attention(self, q, k_cache, v_cache, prefix_len, new_pos, gate, *, flash_mask=None, scale=None):
        """Attend to the visible cache prefix and new rows, then apply the BF16 gate."""
        _check_gate(gate, q, 4)
        out = self.prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos,
                                          flash_mask=flash_mask, scale=scale)
        return out * torch.sigmoid(gate)

    def causal_attention(self, q, k, v, *, scale=None, return_bnsd=False):
        """Attend causally over sequence-major BF16 [B,L,H,D] queries and grouped K/V.

        ``return_bnsd`` (r8): return the attention output in its native BNSD
        [B, H, L, D] buffer instead of the transposed [B, L, H, D] view, for
        callers whose projection reads the BNSD layout directly.
        """
        _check_qkv(q, k, v, 4)
        batch, length, _, dim = q.shape
        if k.shape[:2] != (batch, length):
            raise ValueError("causal attention requires matching query and key batch and sequence lengths")
        scale = _scale(scale, dim)
        key = self._key("causal", q, k, scale)
        query, keys, values = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        with self._lock:
            # r7: FIA's native sparse causal modes are rejected by the CANN
            # 9.1.1 runtime for this GQA BNSD config (every sparseMode 1-4 /
            # pre-next window combination sizes 561002; standalone sweep
            # r7-probe/sparse_sweep.cc) -- the dense boolean mask stays. The
            # FFI entry keeps the sparse_mode parameter for the runtime that
            # gains support.
            drop_mask = self._warmed(
                ("mask", *key),
                lambda: torch.ones((batch, 1, length, length), dtype=torch.bool, device=q.device).triu(1),
            )
            out = self._run(query, keys, values, key, (drop_mask, scale, 0, 0),
                            self._bnsd_query, self._bnsd_launch)
        if return_bnsd:
            return out
        return out.transpose(1, 2)

    def gated_causal_attention(self, q, k, v, gate, *, scale=None):
        """Attend causally over the entire padded prompt and apply the BF16 gate."""
        _check_gate(gate, q, 3)
        out = self.causal_attention(q[None], k[None], v[None], scale=scale)
        return out[0].flatten(1) * torch.sigmoid(gate)

    def varlen_attention(self, q, k, v, cu_seqlens, *, scale=None):
        """Attend bidirectionally within static packed TND segments."""
        _check_qkv(q, k, v, 3)
        if q.shape != k.shape:
            raise ValueError("packed attention requires identical Q/K/V shapes")
        if (
            not isinstance(cu_seqlens, (tuple, list))
            or len(cu_seqlens) < 2
            or not all(isinstance(index, Integral) for index in cu_seqlens)
        ):
            raise ValueError("packed attention requires static integer host offsets")
        bounds = tuple(int(index) for index in cu_seqlens)
        if bounds[0] != 0 or bounds[-1] != q.shape[0] or any(end < start for start, end in pairwise(bounds)):
            raise ValueError("packed attention offsets must cover the token sequence in order")
        # ACLNN takes positive cumulative endpoints, omitting empty segments.
        endpoints = tuple(end for start, end in pairwise(bounds) if end > start)
        scale = _scale(scale, q.shape[-1])
        key = self._key("segments", q, k, scale, endpoints)
        q, k, v = (t.contiguous() for t in (q, k, v))
        with self._lock:
            return self._run(q, k, v, key, (endpoints, scale), self._tnd_query, self._tnd_launch)

    def segment_attention(self, q, k, v, cu_seqlens, *, scale=None):
        """Run the packed bidirectional attention used by the vision stage."""
        return self.varlen_attention(q, k, v, cu_seqlens, scale=scale)
