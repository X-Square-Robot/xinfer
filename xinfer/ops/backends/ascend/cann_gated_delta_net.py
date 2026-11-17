"""Opt-in Wall-style GDN prefill using the Ascend CANN chunk recurrence.

gdr-alpha restructure (2026-11-13): the pre-convolution chain now runs
token-major. The causal conv still reads the channel-major view (F.conv1d
layout), but the SiLU and the Q/K RMS normalization consume one contiguous
``[1, T, C]`` transpose copy instead of per-head strided views of the
channel-major storage, and the TND packs for the recurrence are written by
``torch.mul(..., out=)`` directly into contiguous buffers (the mul-by-one
fold replaces the strided ``contiguous()`` copies). Measured on the
production shape (S=448, Hk=Hv=16, D=128, bf16, card 6): 283.2 -> 241.5 us
per call in a hot NPUGraph (-14.7%; -18.7% at S=432), bit-identical outputs
(``torch.equal``) against the previous composition -- the strided per-element
accesses of the old Q/K chain cost more than the one extra 5.5 MB transpose
copy. The beta gate's sigmoid also writes straight into a contiguous buffer.

gdn-beta tail (2026-11-13): the torch output-normalization tail keeps its
exact op sequence but drops one full pass -- the FP32 gate copy is cast
straight off the strided zba rows (``z.view(tokens, hv, dv).float()``) and
SiLU runs in place on it, removing the separate bf16 ``reshape`` copy and
the out-of-place SiLU write. ``F.rms_norm`` was measured bit-identical but
equally fast on this shape, so the fused-tail knife is the in-library
TileLang ``gdr_cann_finish2`` kernel reached through its raw C launcher
(the same second-dlopen refcount binding as ``kernel._cann_primitives``):
one pass over the BF16 core output and the strided zba rows, fp32 chain
with a single final rounding. That route is opt-in via
``XINFER_GDN_FINISH2=1`` and guarded by layout checks with a fail-safe
torch fallback; its numerics differ from the torch chain (fp32 reduction
order, one rounding instead of the intermediate bf16 rounds), so the
default path stays bit-exact while the opt-in path carries its own
action-gate evidence.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
from threading import Lock

import torch
import torch.nn.functional as F
from xinfer.ops.backends.ascend.core import CORE


_FINISH2 = None


def _finish2_entry():
    """Bind the in-library ``gdr_cann_finish2`` launcher, or ``False``.

    the unified core library is loaded once by the provider; this binding
    resolves the internal launcher in-process (the ``kernel._cann_primitives`` precedent). Any binding
    problem leaves the class on the bit-exact torch tail.
    """
    global _FINISH2
    if _FINISH2 is None:
        try:
            from xinfer.ops.backends.ascend import _native

            library = _native.core_library()
            call = library.xinfer_aot_gdr_cann_finish2_call
            call.restype = ctypes.c_int
            call.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 6 + [ctypes.c_void_p]
            error = library.xinfer_aot_gdr_cann_finish2_get_last_error
            error.restype = ctypes.c_char_p
            _FINISH2 = (call, error)
        except Exception:  # noqa: BLE001 - feature probe, fall back to torch
            _FINISH2 = False
    return _FINISH2


def _gdnv_enabled(family: str) -> bool:
    """GDNV route gate (XINFER_GDNV, default off; see glue.gdn_vendor)."""
    try:
        from xinfer.ops.backends.ascend import gdn_vendor

        return gdn_vendor.enabled(family)
    except Exception:  # noqa: BLE001 - any miss keeps the released chain
        return False


def _mover2_enabled(family: str) -> bool:
    """MOVE-R2 route gate (XINFER_MOVER2, default off; see glue.move_r2)."""
    try:
        from xinfer.ops.backends.ascend import move_r2

        return move_r2.enabled(family)
    except Exception:  # noqa: BLE001 - any miss keeps the released chain
        return False


def _soup2_enabled(family):
    """SOUP2 route gate (XINFER_SOUP2, default off; see glue.soup2)."""
    try:
        from xinfer.ops.backends.ascend import soup2

        return soup2.enabled(family)
    except Exception:  # noqa: BLE001 - any miss keeps the released chain
        return False


def _soup2_qksq(x_t, tokens, two_hk, dk):
    try:
        from xinfer.ops.backends.ascend import soup2

        return soup2.qksq(x_t, tokens, two_hk, dk)
    except Exception:  # noqa: BLE001
        return None


def _soup2_rawsq(out, tokens, hv, dv):
    try:
        from xinfer.ops.backends.ascend import soup2

        return soup2.rawsq(out, tokens, hv, dv)
    except Exception:  # noqa: BLE001
        return None


class CannGatedDeltaNet:
    """B1 BF16 prefill with causal convolution and gated RMSNorm around ACLNN.

    Call once eagerly for each device, stream, and shape before capturing it.
    The instance owns the caller buffers referenced by the graph. Retain it
    until all submitted work has finished and its graphs have been destroyed.
    Unsupported inputs raise. This route never clamps the input gate values;
    the caller must qualify their range on the deployed CANN version.
    """

    def __init__(self):
        self._query = CORE.gdr_cann_chunk_workspace_size
        self._launch = CORE.gdr_cann_chunk
        self._buffers = {}
        self._ones = {}
        self._lock = Lock()
        self._finish2_ok = True
        self._finish2_error = ""

    def _unit(self, device, dtype):
        key = (device, dtype)
        unit = self._ones.get(key)
        if unit is None:
            unit = torch.ones((), device=device, dtype=dtype)
            self._ones[key] = unit
        return unit

    def _finish2(self, zba, norm_weight, raw, result, tokens, hv, eps, stream):
        """Launch the finish2 tail; ``False`` falls back to the torch chain."""
        entry = _finish2_entry()
        if entry is False:
            self._finish2_ok = False
            self._finish2_error = "finish2 launcher not exported by the paired library"
            return False
        call, error = entry
        try:
            status = call(
                ctypes.c_void_p(zba.data_ptr()),
                ctypes.c_void_p(norm_weight.data_ptr()),
                ctypes.c_void_p(raw.data_ptr()),
                ctypes.c_void_p(result.data_ptr()),
                int(tokens),
                int(hv),
                int(tokens),
                int(zba.stride(-2)),
                int(result.stride(-2)),
                int(struct.unpack("<I", struct.pack("<f", float(eps)))[0]),
                ctypes.c_void_p(stream),
            )
        except Exception as exc:  # noqa: BLE001 - fail-safe fallback
            self._finish2_ok = False
            self._finish2_error = repr(exc)
            return False
        if status:
            self._finish2_ok = False
            message = error()
            self._finish2_error = (
                message.decode("utf-8", "replace") if message else f"finish2 status {status}"
            )
            return False
        return True

    def __call__(
        self,
        qkv,
        zba,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        eps,
    ):
        """Return BF16 `[1,T,Hv*Dv]` from packed QKV and `[z | b | a]` rows."""
        hk, hv, dk, dv = num_k_heads, num_v_heads, head_k_dim, head_v_dim
        if (
            qkv.ndim != 3
            or qkv.shape[0] != 1
            or qkv.shape[1] < 1
            or qkv.device.type != "npu"
            or qkv.dtype != torch.bfloat16
            or hk != 16
            or hv not in (16, 32)
            or (dk, dv) != (128, 128)
        ):
            raise ValueError("CANN GDN requires B1 BF16 NPU input, Hk16/Hv16 or32 and D128")
        tokens, channels = qkv.shape[1:]
        if (
            channels != 2 * hk * dk + hv * dv
            or zba.shape != (1, tokens, hv * (dv + 2))
            or conv_weight.ndim != 2
            or conv_weight.shape[0] != channels
            or conv_weight.shape[1] < 1
            or A_log.shape != (hv,)
            or dt_bias.shape != (hv,)
            or norm_weight.shape != (dv,)
        ):
            raise ValueError("CANN GDN received an unexpected packed layout or weight shape")
        if any(
            tensor.device != qkv.device
            for tensor in (zba, conv_weight, A_log, dt_bias, norm_weight)
        ):
            raise ValueError("CANN GDN tensors must be on the same NPU device")
        if (
            any(
                tensor.dtype != torch.bfloat16
                for tensor in (zba, conv_weight, norm_weight)
            )
            or A_log.dtype not in (torch.bfloat16, torch.float32)
            or dt_bias.dtype not in (torch.bfloat16, torch.float32)
            or not math.isfinite(eps)
            or eps <= 0
        ):
            raise ValueError("CANN GDN received unsupported weight dtypes or epsilon")

        # Preserve the validated Wall reference boundaries before the ACLNN
        # recurrence. The prefill path does not use a convolution bias.
        # conv-gemm: the env-gated dconv4_c6144 route replaces slice +
        # transpose + aclnnConv2d + the token-major copy with one fused
        # single-pass vector kernel (XINFER_CONVGEMM=1 + XINFER_CONVGEMM_LIB;
        # an envelope miss or launcher failure falls back to the released
        # chain bit for bit -- the SiLU and everything after are untouched).
        x_t = None
        if os.environ.get("XINFER_CONVGEMM", "0") == "1":
            try:
                from xinfer.ops.backends.ascend.conv_gemm import conv_mixed_tokenmajor

                x_t = conv_mixed_tokenmajor(qkv, conv_weight)
            except (ImportError, OSError):
                x_t = None
        # mover2 gdrapply blade (XINFER_MOVER2=gdrapply, default off): with the
        # token-major rows already in hand (dconv4 direct write or the full-width
        # copy below), ONE kernel writes the TND q/k packs (x * rstd, the chain's
        # single RNE) and the v copy -- replacing the width slice + the n Mul_Cast
        # + 2x mul-by-one + AsStrided(v). The SiLU and the aclnn stats
        # (square/sum/rsqrt on identical inputs) stay; any miss or launcher
        # failure falls back to the released segment bit for bit.
        mover2_apply = (
            hk == hv
            and tokens % 8 == 0
            and tokens >= 8
            and _mover2_enabled("gdrapply")
        )
        if mover2_apply:
            try:
                from xinfer.ops.backends.ascend.move_r2 import gdrapply as _gdrapply

                if x_t is None:
                    mixed_full = F.conv1d(
                        qkv.transpose(1, 2),
                        conv_weight[:, None, :],
                        padding=conv_weight.shape[1] - 1,
                        groups=channels,
                    )
                    # the full-width token-major copy: the width slice disappears
                    x_full = mixed_full.transpose(1, 2).contiguous()
                    F.silu(x_full, inplace=True)
                    rows = x_full[0, :tokens]
                else:
                    # the dconv4 direct-write contract: token-major [1, T, C]
                    # pre-SiLU rows; the kernel reads the same [T, C] window
                    F.silu(x_t, inplace=True)
                    rows = x_t[0]
                # SOUP2 qksq pre-pass (opt-in, XINFER_SOUP2=qksq): ONE kernel
                # writes f32(RNE_bf16(qk*qk)) over the strided qk window; the
                # aclnn ReduceSum/Adds/Rsqrt below then read bit-identical
                # bits (the mul + the sum's implicit cast disappear).
                _sq = None
                if x_t is not None and _soup2_enabled("qksq"):
                    _sq = _soup2_qksq(x_t, tokens, 2 * hk, dk)
                if _sq is not None:
                    rs = torch.rsqrt(_sq.sum(dim=-1, keepdim=True) + 1e-6)
                else:
                    qk = rows[:, : 2 * hk * dk].reshape(1, tokens, 2 * hk, dk)
                    with torch.autocast("npu", dtype=qkv.dtype):
                        rs = torch.rsqrt((qk * qk).sum(dim=-1, keepdim=True) + 1e-6)
                q, k, v = _gdrapply(rows, rs.view(tokens, 2 * hk), tokens, hk, dk)
            except (ImportError, OSError, RuntimeError):
                mover2_apply = False
                x_t = None
        if not mover2_apply:
            if x_t is None:
                mixed = F.conv1d(
                    qkv.transpose(1, 2),
                    conv_weight[:, None, :],
                    padding=conv_weight.shape[1] - 1,
                    groups=channels,
                )[..., :tokens]
                # gdr-alpha: one token-major copy, then in-place SiLU on contiguous
                # rows and the Q/K RMS normalization over a stride-one head dim (the
                # old chain reduced strided views of the channel-major storage).
                x_t = mixed.transpose(1, 2).contiguous()
            F.silu(x_t, inplace=True)
            qk = x_t[:, :, : 2 * hk * dk].reshape(1, tokens, 2 * hk, dk)
            with torch.autocast("npu", dtype=qkv.dtype):
                n = qk * torch.rsqrt((qk * qk).sum(dim=-1, keepdim=True) + 1e-6)
            # The recurrence consumes TND q/k/v; mul-by-one writes the strided rms
            # result straight into the contiguous TND buffers (one kernel each).
            q = torch.empty(tokens, hk, dk, dtype=qkv.dtype, device=qkv.device)
            k = torch.empty(tokens, hk, dk, dtype=qkv.dtype, device=qkv.device)
            unit = self._unit(qkv.device, n.dtype)
            torch.mul(n[0, :, :hk], unit, out=q)
            torch.mul(n[0, :, hk:], unit, out=k)
            v = x_t[:, :, 2 * hk * dk:].reshape(tokens, hv, dv).contiguous()
        # GDNV gdrprep blade (XINFER_GDNV=gdrprep, default off): ONE kernel
        # writes beta (the aclnn-sigmoid composition, bitwise over the whole
        # bf16 domain) and the fp32 g-prefix f32(a) + dt; -exp(A_log) and the
        # dt cast are cached at warmup by the same aclnn ops (bitwise).
        # Softplus and the premul multiply stay on aclnn with bit-identical
        # inputs (SoftplusV2's log1p tail is not bitwise reproducible). Any
        # miss or launcher failure falls back to the released chain.
        gdnv = None
        if (
            hv == 16
            and dv == 128
            and tokens >= 2
            and tokens % 2 == 0
            and zba.stride(-1) == 1
            and zba.stride(-2) == hv * (dv + 2)
            and _gdnv_enabled("gdrprep")
        ):
            try:
                from xinfer.ops.backends.ascend import gdn_vendor as _gdnv_mod

                gdnv = _gdnv_mod.gdrprep(zba, A_log, dt_bias, tokens, hv, dv, qkv.dtype)
            except (ImportError, OSError, RuntimeError):
                gdnv = None
        if gdnv is not None:
            # keep z as the lazy split view for the output tail below
            z = zba[..., : hv * dv]
            beta, g = gdnv
        else:
            z, beta_logit, decay_logit = zba.split((hv * dv, hv, hv), dim=-1)
            beta_buf = torch.empty(1, tokens, hv, dtype=qkv.dtype, device=qkv.device)
            torch.sigmoid(beta_logit, out=beta_buf)
            beta = beta_buf[0]
            g = (-A_log.float().exp() * F.softplus(decay_logit.float() + dt_bias))[0].contiguous()
        scale = dk ** -0.5

        capture = torch.npu.is_current_stream_capturing()
        stream = torch.npu.current_stream(qkv.device).npu_stream
        key = (qkv.device, stream, tokens, hk, hv, dk, dv)
        # Keep recurrence and its consumer ordered even if two host threads
        # share a stream. Device submissions remain asynchronous.
        with self._lock:
            if key not in self._buffers:
                if capture:
                    raise RuntimeError("warm up CANN GDN on this stream and shape before graph capture")
                state = qkv.new_zeros((1, hv, dv, dk))
                lengths = torch.full((1,), tokens, dtype=torch.int32, device=qkv.device)
                out = qkv.new_empty((tokens, hv, dv))
                final_state = torch.empty_like(state)
                size = self._query(out, final_state, q, k, v, beta, state, lengths, g, scale)
                workspace = torch.empty(size, dtype=torch.uint8, device=qkv.device)
                self._buffers[key] = state, lengths, out, final_state, workspace
            state, lengths, out, final_state, workspace = self._buffers[key]
            self._launch(out, final_state, workspace, q, k, v, beta, state, lengths, g, scale)
            # gdn-beta fused tail: the in-library finish2 kernel reads the
            # strided zba rows and the [tokens, hv*D] core output in one
            # pass. Opt-in (its rounding chain differs from the torch tail);
            # ROW_TILE=16 needs at least sixteen tokens and the paired-head
            # tiles need an even head count; anything else stays on torch.
            if (
                self._finish2_ok
                and os.environ.get("XINFER_GDN_FINISH2") == "1"
                and tokens >= 16
                and hv % 2 == 0
                and zba.stride(-1) == 1
                and zba.stride(-2) % 2 == 0
                and zba.data_ptr() % 4 == 0
                and norm_weight.is_contiguous()
                and norm_weight.data_ptr() % 16 == 0
            ):
                fused = qkv.new_empty((tokens, hv * dv))
                if self._finish2(
                    zba, norm_weight, out.view(tokens, hv * dv), fused, tokens, hv, eps, stream
                ):
                    return fused.reshape(1, tokens, hv * dv)
            # mover2 gdrtailmix blade (XINFER_MOVER2=gdrtailmix, default off):
            # the aclnn gate chain and the aclnn stats stay; ONE kernel then
            # does the tail's exact three RNE roundings (nbf / wn / gate),
            # replacing the 6-op elementwise chain. Any miss or launcher
            # failure falls through to the released bit-exact tail.
            if (
                tokens % 8 == 0
                and tokens >= 8
                and hv == 16
                and zba.stride(-1) == 1
                and norm_weight.is_contiguous()
                and _mover2_enabled("gdrtailmix")
            ):
                try:
                    from xinfer.ops.backends.ascend.move_r2 import gdrtailmix as _gdrtailmix

                    gate = z.view(tokens, hv, dv).float()
                    F.silu(gate, inplace=True)
                    # SOUP2 rawsq pre-pass (opt-in, XINFER_SOUP2=rawsq): ONE
                    # kernel writes RNE_f32(raw**2); the aclnn Mean/Adds/Rsqrt
                    # read bit-identical bits (cast + square disappear).
                    _sq2 = _soup2_rawsq(out, tokens, hv, dv) if _soup2_enabled("rawsq") else None
                    if _sq2 is not None:
                        rstd = torch.rsqrt(_sq2.mean(dim=-1, keepdim=True) + eps)
                    else:
                        raw = out.reshape(-1, dv).float()
                        rstd = torch.rsqrt(raw.pow(2).mean(dim=-1, keepdim=True) + eps)
                    result = _gdrtailmix(
                        out.view(tokens, hv * dv), rstd.view(tokens, hv),
                        gate.view(tokens, hv * dv), norm_weight, tokens, hv, dv)
                    return result.reshape(1, tokens, hv * dv)
                except (ImportError, OSError, RuntimeError):
                    pass
            # gdn-beta bit-exact micro: the FP32 gate copy is cast straight
            # off the strided zba rows and SiLU runs in place on it, so the
            # old reshape copy and the out-of-place SiLU write disappear
            # without touching the op sequence.
            gate = z.view(tokens, hv, dv).float()
            F.silu(gate, inplace=True)
            raw = out.reshape(-1, dv).float()
            normed = raw * torch.rsqrt(raw.pow(2).mean(dim=-1, keepdim=True) + eps)
            normed = norm_weight * normed.to(qkv.dtype)
            result = (normed * gate.reshape(-1, dv)).to(qkv.dtype)
        return result.reshape(1, tokens, hv * dv)
