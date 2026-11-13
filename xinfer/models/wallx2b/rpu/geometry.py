"""CPU evaluation geometry for the Wall-X 2B FP16 TP8 relative26 profile.

Adapted from RhinoForge's evaluation helpers; public source links are listed
in docs/models/wallx2b_rpu.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def euler_zyx_to_matrix(value: Any) -> np.ndarray:
    """Float64 Rz(yaw) Ry(pitch) Rx(roll), with input [..., roll,pitch,yaw]."""
    value = np.asarray(value, dtype=np.float64)
    roll, pitch, yaw = np.moveaxis(value, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    result = np.empty(value.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = cy * cp
    result[..., 0, 1] = cy * sp * sr - sy * cr
    result[..., 0, 2] = cy * sp * cr + sy * sr
    result[..., 1, 0] = sy * cp
    result[..., 1, 1] = sy * sp * sr + cy * cr
    result[..., 1, 2] = sy * sp * cr - cy * sr
    result[..., 2, 0] = -sp
    result[..., 2, 1] = cp * sr
    result[..., 2, 2] = cp * cr
    return result


def rows6_to_matrix(value: Any) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    row0, row1 = value[..., :3], value[..., 3:6]
    row0 = row0 / np.maximum(np.linalg.norm(row0, axis=-1, keepdims=True), 1e-9)
    row1 = row1 - np.sum(row0 * row1, axis=-1, keepdims=True) * row0
    row1 = row1 / np.maximum(np.linalg.norm(row1, axis=-1, keepdims=True), 1e-9)
    return np.stack((row0, row1, np.cross(row0, row1)), axis=-2)


def matrix_to_euler_zyx(matrix: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.stack(
        (
            np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]),
            np.arctan2(-matrix[..., 2, 0], np.hypot(matrix[..., 0, 0], matrix[..., 1, 0])),
            np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]),
        ),
        axis=-1,
    ).astype(np.float32)


def relative26_to_absolute14(relative: Any, state26: Any) -> np.ndarray:
    """Physical relative actions [T,26] -> absolute [T,14], no anchor row added."""
    relative = np.asarray(relative, dtype=np.float32)
    state = np.asarray(state26, dtype=np.float32).reshape(26)
    if relative.ndim != 2 or relative.shape[1] != 26:
        raise ValueError(f"relative action must have shape [T,26], got {relative.shape}")
    if not np.isfinite(relative).all() or not np.isfinite(state).all():
        raise ValueError("actions and state must be finite")
    result = np.empty((relative.shape[0], 14), dtype=np.float32)
    for src_pos, src_rot, src_grip, dst in ((0, 3, 9, 0), (10, 13, 19, 7)):
        result[:, dst : dst + 3] = relative[:, src_pos : src_pos + 3] + state[src_pos : src_pos + 3]
        delta_rotation = rows6_to_matrix(relative[:, src_rot : src_rot + 6])
        state_rotation = rows6_to_matrix(state[None, src_rot : src_rot + 6])[0]
        result[:, dst + 3 : dst + 6] = matrix_to_euler_zyx(delta_rotation @ state_rotation)
        result[:, dst + 6] = relative[:, src_grip]
    if not np.isfinite(result).all():
        raise RuntimeError("decoded absolute action contains non-finite values")
    return result


def state26_from_frame(frame: dict) -> np.ndarray:
    parts = []
    for side in ("left", "right"):
        rotation = euler_zyx_to_matrix(frame[f"follow_{side}_rotation"])
        parts.extend(
            (
                np.asarray(frame[f"follow_{side}_position"], dtype=np.float32),
                rotation[:2].reshape(6).astype(np.float32),
                np.asarray([frame[f"follow_{side}_gripper"]], dtype=np.float32),
            )
        )
    result = np.concatenate([*parts, np.zeros(6, dtype=np.float32)])
    if result.shape != (26,) or not np.isfinite(result).all():
        raise ValueError("constructed state must be finite [26]")
    return result


def ground_truth_window(trajectory: list[dict], anchor: int, count: int) -> np.ndarray:
    """The first supervised action is strictly after the observation anchor."""
    if anchor < 0 or count < 1 or anchor + count >= len(trajectory):
        raise ValueError("GT action window must fit after observation anchor+1")
    rows = []
    for frame in trajectory[anchor + 1 : anchor + 1 + count]:
        rows.append(
            np.concatenate(
                [
                    np.concatenate(
                        (
                            np.asarray(frame[f"master_{s}_position"], dtype=np.float32),
                            np.asarray(frame[f"master_{s}_rotation"], dtype=np.float32),
                            np.asarray([frame[f"master_{s}_gripper"]], dtype=np.float32),
                        )
                    )
                    for s in ("left", "right")
                ]
            )
        )
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (count, 14) or not np.isfinite(result).all():
        raise ValueError("ground truth must be finite [T,14]")
    return result


def absolute14_metrics(predicted: Any, target: Any) -> dict[str, float]:
    predicted, target = np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 14:
        raise ValueError("absolute action metrics require matching [T,14] arrays")
    if not len(predicted) or not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("absolute actions must be nonempty and finite")
    angles = []
    for offset in (3, 10):
        pred_rotation = euler_zyx_to_matrix(predicted[:, offset : offset + 3])
        target_rotation = euler_zyx_to_matrix(target[:, offset : offset + 3])
        relative = pred_rotation @ np.swapaxes(target_rotation, -1, -2)
        cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosine)))
    return {
        "abs14_l1": float(np.abs(predicted - target).mean()),
        "position_mae": float(np.abs(predicted[:, (0, 1, 2, 7, 8, 9)] - target[:, (0, 1, 2, 7, 8, 9)]).mean()),
        "rotation_geodesic_deg": float(np.concatenate(angles).mean()),
        "gripper_mae": float(np.abs(predicted[:, (6, 13)] - target[:, (6, 13)]).mean()),
    }
