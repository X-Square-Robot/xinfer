"""CES action host preparation and independent FP16 flow diagnostics.

Only time/Ada conditioning belongs on the host in the candidate path. The
Euler function is a numerical oracle for device bring-up, never a fallback.
The checkpoint's saved ten-step scheduler does not select serving NFE.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .weights import euler_schedule, pack_adaptive_rows


def flow_schedule() -> tuple[torch.Tensor, np.float16]:
    times, dt = euler_schedule()
    if dt <= 0:
        raise ValueError("wallx2b requires uniform positive FP16 NFE8 dt")
    return times, np.float16(dt)


def _fp32(weights, name, shape):
    value = weights[name]
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise TypeError(f"{name} requires a CPU Torch tensor")
    if value.dtype != torch.float32 or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} requires finite FP32 {shape}")
    return value


@dataclass(frozen=True)
class ActionConditioning:
    """Immutable CPU snapshot, computed once before any request execution.

    Rows 2*i and 2*i+1 condition the token mixer and MLP of layer i;
    row 48 conditions the final norm. Gamma, shift and gate each span 1024.
    MLP down weights are stored /4 and odd-row gates compensate by x4.
    Residual values use x/256; native norms use normalization_scale=64.
    """

    modulation: np.ndarray  # [8,49,3,1024] FP16
    times: np.ndarray  # [9] FP32
    delta_t: np.float16


@torch.inference_mode()
def prepare_conditioning(weights: Mapping[str, torch.Tensor]) -> ActionConditioning:
    stem = "model.action_processor"
    w_in = _fp32(weights, f"{stem}.time_mlp_in.weight", (1024, 1024))
    b_in = _fp32(weights, f"{stem}.time_mlp_in.bias", (1024,))
    w_out = _fp32(weights, f"{stem}.time_mlp_out.weight", (1024, 1024))
    b_out = _fp32(weights, f"{stem}.time_mlp_out.bias", (1024,))
    projections = []
    for i in range(24):
        for norm in ("input_layernorm_action", "post_attention_layernorm_action"):
            key = f"model.language_model.layers.{i}.{norm}.dense"
            projections.append((_fp32(weights, key + ".weight", (3072, 1024)), _fp32(weights, key + ".bias", (3072,))))
    key = "model.language_model.norm_action.dense"
    projections.append((_fp32(weights, key + ".weight", (3072, 1024)), _fp32(weights, key + ".bias", (3072,))))
    times, delta_t = flow_schedule()
    exponent = torch.log(torch.tensor(10000.0, dtype=torch.float32)) / 511
    frequencies = torch.exp(torch.arange(512, dtype=torch.float32) * -exponent)
    steps = []
    # Keep the reference's B1 GEMMs and per-step FP32 order. A batched time
    # projection may select a different reduction and change packed halves.
    with torch.autocast("cpu", enabled=False):
        for time in times[:-1]:
            phase = time.reshape(1, 1) * frequencies.reshape(1, -1)
            value = torch.cat((phase.sin(), phase.cos()), dim=-1)
            value = F.silu(F.linear(value, w_in, b_in))
            condition = F.silu(F.linear(value, w_out, b_out))
            steps.append(pack_adaptive_rows(condition, projections))
        packed = torch.stack(steps).reshape(8, 49, 3, 1024)
    if not bool(torch.isfinite(packed).all()):
        raise ValueError("wallx2b time/Ada conditioning overflows FP16")
    modulation = packed.numpy().copy()
    schedule = times.numpy().copy()
    modulation.flags.writeable = schedule.flags.writeable = False
    return ActionConditioning(modulation, schedule, delta_t)


def action_attention_mask(real_prefix: int, bucket: int) -> np.ndarray:
    """Noncausal action mask, including -Inf over the physical prefix gap."""
    if (
        isinstance(real_prefix, bool)
        or not isinstance(real_prefix, int)
        or not 1 <= real_prefix <= 512
        or bucket != ((real_prefix + 63) // 64) * 64
    ):
        raise ValueError("prefix must be 1..512 with its ceil64 physical bucket")
    mask = np.zeros((16, bucket + 16), dtype=np.float16)
    mask[:, real_prefix:bucket] = -np.inf
    return mask


def euler_fp16_reference(action, velocity, keep_mask, padding_velocity, delta_t):
    """Separate FP16 mask/add/multiply/add boundaries of one device step.

    ``padding_velocity`` is pre-masked during preparation. This oracle
    accepts the physical packed [16,64] row. Mask columns in its tail remain
    unchanged when keep_mask and padding_velocity are zero there.
    """
    for name, value in (("action", action), ("velocity", velocity), ("padding_velocity", padding_velocity)):
        if not isinstance(value, np.ndarray) or value.dtype != np.float16 or value.shape != (16, 64):
            raise ValueError(f"{name} requires FP16 [16,64]")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if (
        not isinstance(keep_mask, np.ndarray)
        or keep_mask.dtype != np.float16
        or keep_mask.shape != (64,)
        or not np.isin(keep_mask, (0, 1)).all()
    ):
        raise ValueError("keep_mask requires binary FP16 [64]")
    if np.any(padding_velocity[:, keep_mask == 1] != 0):
        raise ValueError("padding_velocity must already be zero for kept dimensions")
    dt = np.float16(delta_t)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("delta_t must round to a positive finite FP16 value")
    with np.errstate(over="ignore", invalid="ignore"):
        masked = (velocity * keep_mask).astype(np.float16)
        mixed = (masked + padding_velocity).astype(np.float16)
        increment = (mixed * dt).astype(np.float16)
        result = (action + increment).astype(np.float16)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite FP16 Euler output")
    return result.copy()
