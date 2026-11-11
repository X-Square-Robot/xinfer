"""Rotation helpers shared by Wall-X models (numba, ported verbatim).

Provenance: embodied-serving gl/main 9633ffd
``src/embodied_serving/adapters/wallx/runtime_helpers/geometry.py``, itself
extracted from the pinned Wall-X inference dependencies (identical, apart from
comments, to ``x2robot_utils.geometry``). The arithmetic order and the numba
specializations (float32 inputs evaluate cos/sin in float32) are part of the
served numerics, so these functions are kept as numba code rather than
rewritten with numpy. ``compose_state_and_delta_to_abs_rpy`` is not used and
was dropped; the code of everything else is unchanged (English docstrings
were added).
"""
import numpy as np
from numba import jit, prange

@jit(nopython=True)
def euler_to_matrix_zyx_6d_nb(eulers):
    """
    Numba 版本：欧拉角 (N, 3) → 前两行展平 (N, 6)

    ZYX Euler angles to the 6D rotation representation: the first two rows of
    ``R = Rz(yaw) · Ry(pitch) · Rx(roll)``, flattened. cos/sin are evaluated in the input dtype.

    Args:
        eulers: ``[N, 3]`` ``[roll, pitch, yaw]`` in radians (float32 or float64).

    Returns:
        ``[N, 6]`` float64 ``[R00, R01, R02, R10, R11, R12]``.
    """
    N = eulers.shape[0]
    R6 = np.empty((N, 6), dtype=np.float64)
    for i in prange(N):
        roll = eulers[i, 0]
        pitch = eulers[i, 1]
        yaw = eulers[i, 2]

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr

        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr

        R6[i, 0] = r00
        R6[i, 1] = r01
        R6[i, 2] = r02
        R6[i, 3] = r10
        R6[i, 4] = r11
        R6[i, 5] = r12
    return R6



@jit(nopython=True)
def euler_to_matrix_zyx_batch_nb(eulers):
    """ZYX Euler angles to rotation matrices ``R = Rz(yaw) · Ry(pitch) · Rx(roll)``.

    Args:
        eulers: ``[N, 3]`` ``[roll, pitch, yaw]`` in radians.

    Returns:
        ``[N, 3, 3]`` float64.
    """
    N = eulers.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float64)
    for i in prange(N):
        roll = eulers[i, 0]
        pitch = eulers[i, 1]
        yaw = eulers[i, 2]

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R[i, 0, 0] = cy * cp
        R[i, 0, 1] = cy * sp * sr - sy * cr
        R[i, 0, 2] = cy * sp * cr + sy * sr

        R[i, 1, 0] = sy * cp
        R[i, 1, 1] = sy * sp * sr + cy * cr
        R[i, 1, 2] = sy * sp * cr - cy * sr

        R[i, 2, 0] = -sp
        R[i, 2, 1] = cp * sr
        R[i, 2, 2] = cp * cr
    return R



def so3_to_euler_zyx_batch_nb(batch_so3):
    """6D rotations to canonical ZYX Euler angles (Gram-Schmidt matrix, extraction, canonicalization).

    Args:
        batch_so3: ``[N, 6]`` first two rotation-matrix rows, flattened.

    Returns:
        ``[N, 3]`` float64 ``[roll, pitch, yaw]`` in radians, canonicalized by
        ``canonicalize_euler_zyx_batch_nb``.
    """
    matrix = so3_to_matrix_batch_nb(batch_so3)
    eulers = matrix_to_euler_zyx_batch_nb(matrix)
    return canonicalize_euler_zyx_batch_nb(eulers)



@jit(nopython=True)
def matrix_to_euler_zyx_batch_nb(Rs):
    """
    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    提取：
      pitch = asin(-R[2,0])
      roll  = atan2(R[2,1], R[2,2])
      yaw   = atan2(R[1,0], R[0,0])

    Rotation matrices to ZYX Euler angles by the formulas above (``-R[2,0]`` clamped to [-1, 1]).

    Args:
        Rs: ``[N, 3, 3]`` rotation matrices.

    Returns:
        ``[N, 3]`` float64 ``[roll, pitch, yaw]`` in radians.
    """
    N = Rs.shape[0]
    eulers = np.empty((N, 3), dtype=np.float64)
    for i in prange(N):
        r00 = Rs[i, 0, 0]
        # r01 = Rs[i, 0, 1]
        # r02 = Rs[i, 0, 2]
        r10 = Rs[i, 1, 0]
        # r11 = Rs[i, 1, 1]
        # r12 = Rs[i, 1, 2]
        r20 = Rs[i, 2, 0]
        r21 = Rs[i, 2, 1]
        r22 = Rs[i, 2, 2]

        # 数值稳定：夹到 [-1, 1]
        x = -r20
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0

        pitch = np.arcsin(x)
        roll = np.arctan2(r21, r22)
        yaw = np.arctan2(r10, r00)

        eulers[i, 0] = roll
        eulers[i, 1] = pitch
        eulers[i, 2] = yaw
    return eulers



@jit(nopython=True)
def so3_to_matrix_batch_nb(batch_so3):
    """6D rotations to rotation matrices by Gram-Schmidt.

    Row 0 is the normalized first 3-vector, row 1 the second 3-vector orthogonalized against row 0 and
    normalized, row 2 their cross product. Norms add ``1e-12`` before dividing.

    Args:
        batch_so3: ``[N, 6]`` first two rotation-matrix rows, flattened (need not be orthonormal).

    Returns:
        ``[N, 3, 3]`` float64.
    """
    N = batch_so3.shape[0]
    R_all = np.empty((N, 3, 3), dtype=np.float64)
    eps = 1e-12
    for i in prange(N):
        r1x, r1y, r1z = batch_so3[i, 0], batch_so3[i, 1], batch_so3[i, 2]
        r2x, r2y, r2z = batch_so3[i, 3], batch_so3[i, 4], batch_so3[i, 5]

        # normalize r1
        n1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z) + eps
        r1x /= n1
        r1y /= n1
        r1z /= n1

        # orthogonalize r2 to r1, then normalize
        dot12 = r1x * r2x + r1y * r2y + r1z * r2z
        r2x -= dot12 * r1x
        r2y -= dot12 * r1y
        r2z -= dot12 * r1z
        n2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z) + eps
        r2x /= n2
        r2y /= n2
        r2z /= n2

        # r3 = r1 x r2
        r3x = r1y * r2z - r1z * r2y
        r3y = r1z * r2x - r1x * r2z
        r3z = r1x * r2y - r1y * r2x

        R_all[i, 0, 0] = r1x
        R_all[i, 0, 1] = r1y
        R_all[i, 0, 2] = r1z
        R_all[i, 1, 0] = r2x
        R_all[i, 1, 1] = r2y
        R_all[i, 1, 2] = r2z
        R_all[i, 2, 0] = r3x
        R_all[i, 2, 1] = r3y
        R_all[i, 2, 2] = r3z
    return R_all



@jit(nopython=True)
def compute_delta_from_state_and_abs_rot(rotations, state):
    """Relative rotation ``R_rel = R_abs · R_stateᵀ`` of absolute rotations with respect to a state rotation.

    Args:
        rotations: ``[N, 3]`` ZYX Euler angles (radians) or ``[N, 6]`` 6D rotations (absolute).
        state: ``[3]`` ZYX Euler angles or ``[6]`` 6D rotation of the reference state.

    Returns:
        In the representation of ``rotations``: ``[N, 3]`` canonical ZYX Euler angles or ``[N, 6]`` 6D
        (first two rows of ``R_rel``), float64.

    Raises:
        ValueError: An input whose last dimension is neither 3 nor 6.
    """
    if rotations.shape[-1] == 3:
        rotations_matrix = euler_to_matrix_zyx_batch_nb(rotations)
        out_is_euler = True
    elif rotations.shape[-1] == 6:
        rotations_matrix = so3_to_matrix_batch_nb(rotations)
        out_is_euler = False
    else:
        raise ValueError(
            f"Only support 3D euler angle or 6D rotation, but got {rotations.shape[-1]}D"
        )

    if state.shape[-1] == 3:
        state_matrix = euler_to_matrix_zyx_batch_nb(state[np.newaxis, :])[0]
    elif state.shape[-1] == 6:
        state_matrix = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]
    else:
        raise ValueError(
            f"Only support 3D euler angle or 6D rotation, but got {state.shape[-1]}D"
        )

    ST = np.empty((3, 3), dtype=np.float64)
    ST[0, 0] = state_matrix[0, 0]
    ST[0, 1] = state_matrix[1, 0]
    ST[0, 2] = state_matrix[2, 0]
    ST[1, 0] = state_matrix[0, 1]
    ST[1, 1] = state_matrix[1, 1]
    ST[1, 2] = state_matrix[2, 1]
    ST[2, 0] = state_matrix[0, 2]
    ST[2, 1] = state_matrix[1, 2]
    ST[2, 2] = state_matrix[2, 2]
    N = rotations_matrix.shape[0]
    R_rel = np.empty((N, 3, 3), dtype=np.float64)
    for i in prange(N):
        A00 = rotations_matrix[i, 0, 0]
        A01 = rotations_matrix[i, 0, 1]
        A02 = rotations_matrix[i, 0, 2]
        A10 = rotations_matrix[i, 1, 0]
        A11 = rotations_matrix[i, 1, 1]
        A12 = rotations_matrix[i, 1, 2]
        A20 = rotations_matrix[i, 2, 0]
        A21 = rotations_matrix[i, 2, 1]
        A22 = rotations_matrix[i, 2, 2]

        R_rel[i, 0, 0] = A00 * ST[0, 0] + A01 * ST[1, 0] + A02 * ST[2, 0]
        R_rel[i, 0, 1] = A00 * ST[0, 1] + A01 * ST[1, 1] + A02 * ST[2, 1]
        R_rel[i, 0, 2] = A00 * ST[0, 2] + A01 * ST[1, 2] + A02 * ST[2, 2]
        R_rel[i, 1, 0] = A10 * ST[0, 0] + A11 * ST[1, 0] + A12 * ST[2, 0]
        R_rel[i, 1, 1] = A10 * ST[0, 1] + A11 * ST[1, 1] + A12 * ST[2, 1]
        R_rel[i, 1, 2] = A10 * ST[0, 2] + A11 * ST[1, 2] + A12 * ST[2, 2]
        R_rel[i, 2, 0] = A20 * ST[0, 0] + A21 * ST[1, 0] + A22 * ST[2, 0]
        R_rel[i, 2, 1] = A20 * ST[0, 1] + A21 * ST[1, 1] + A22 * ST[2, 1]
        R_rel[i, 2, 2] = A20 * ST[0, 2] + A21 * ST[1, 2] + A22 * ST[2, 2]

    if out_is_euler:
        d_euler = matrix_to_euler_zyx_batch_nb(R_rel)
        return canonicalize_euler_zyx_batch_nb(d_euler)
    else:
        out6 = np.empty((N, 6), dtype=np.float64)
        for i in prange(N):
            out6[i, 0] = R_rel[i, 0, 0]
            out6[i, 1] = R_rel[i, 0, 1]
            out6[i, 2] = R_rel[i, 0, 2]
            out6[i, 3] = R_rel[i, 1, 0]
            out6[i, 4] = R_rel[i, 1, 1]
            out6[i, 5] = R_rel[i, 1, 2]
        return out6



@jit(nopython=True)
def canonicalize_euler_zyx_batch_nb(rpy_batch):
    """
    批量 ZYX 欧拉角规范化（并行版）
    输入:  rpy_batch (N, 3)  [roll, pitch, yaw]（弧度）
    输出:  out       (N, 3)  约束到同一分支且每分量在 (-π, π]
    规则:
      1) 先逐分量 wrap 到 (-π, π]
      2) 若 p >  π/2:  p =  π - p; r += π; y += π
         若 p <= -π/2: p = -π - p; r += π; y += π
      3) 最后再逐分量 wrap 到 (-π, π]

    Canonicalize ZYX Euler angles to one branch by the rules above: pitch ends in ``[-π/2, π/2]``. Each wrap
    is ``(x + π) % 2π - π``, which gives ``[-π, π)``.

    Args:
        rpy_batch: ``[N, 3]`` ``[roll, pitch, yaw]`` in radians.

    Returns:
        ``[N, 3]``, same dtype as ``rpy_batch``.
    """
    N = rpy_batch.shape[0]
    out = np.empty_like(rpy_batch)
    two_pi = 2.0 * np.pi

    for i in prange(N):
        r = rpy_batch[i, 0]
        p = rpy_batch[i, 1]
        y = rpy_batch[i, 2]

        # 第一次 wrap
        r = (r + np.pi) % two_pi - np.pi
        p = (p + np.pi) % two_pi - np.pi
        y = (y + np.pi) % two_pi - np.pi

        # 分支规范
        if p > np.pi / 2.0:
            p = np.pi - p
            r = r + np.pi
            y = y + np.pi
        elif p <= -np.pi / 2.0:
            p = -np.pi - p
            r = r + np.pi
            y = y + np.pi

        # 最终 wrap
        r = (r + np.pi) % two_pi - np.pi
        p = (p + np.pi) % two_pi - np.pi
        y = (y + np.pi) % two_pi - np.pi

        out[i, 0] = r
        out[i, 1] = p
        out[i, 2] = y

    return out



@jit(nopython=True)
def compose_state_and_delta_to_abs_6d(delta, state):
    """
    输入:
      delta: (N,6)  -> Δ6D(前两行展平) 6D旋转增量
      state: (6,)   -> 6D(前两行展平) 当前状态6D旋转
    输出:
      abs_6d: (N,6) 绝对姿态的 6D 表示(前两行展平)

    Compose relative 6D rotations with a state rotation: ``R_abs = R_delta · R_state`` (the inverse of
    ``compute_delta_from_state_and_abs_rot``).

    Args:
        delta: ``[N, 6]`` 6D rotation deltas.
        state: ``[6]`` 6D rotation of the current state.

    Returns:
        ``[N, 6]`` float64 6D absolute rotations (first two rows of ``R_abs``).
    """
    R_delta = so3_to_matrix_batch_nb(delta)  # (N,3,3)
    R_state = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]  # (3,3)

    N = R_delta.shape[0]
    R_abs = np.empty((N, 3, 3), dtype=np.float64)

    # 预先复制 R_state 以避免视图问题
    S00 = R_state[0, 0]
    S01 = R_state[0, 1]
    S02 = R_state[0, 2]
    S10 = R_state[1, 0]
    S11 = R_state[1, 1]
    S12 = R_state[1, 2]
    S20 = R_state[2, 0]
    S21 = R_state[2, 1]
    S22 = R_state[2, 2]

    for i in prange(N):
        A00 = R_delta[i, 0, 0]
        A01 = R_delta[i, 0, 1]
        A02 = R_delta[i, 0, 2]
        A10 = R_delta[i, 1, 0]
        A11 = R_delta[i, 1, 1]
        A12 = R_delta[i, 1, 2]
        A20 = R_delta[i, 2, 0]
        A21 = R_delta[i, 2, 1]
        A22 = R_delta[i, 2, 2]

        R_abs[i, 0, 0] = A00 * S00 + A01 * S10 + A02 * S20
        R_abs[i, 0, 1] = A00 * S01 + A01 * S11 + A02 * S21
        R_abs[i, 0, 2] = A00 * S02 + A01 * S12 + A02 * S22

        R_abs[i, 1, 0] = A10 * S00 + A11 * S10 + A12 * S20
        R_abs[i, 1, 1] = A10 * S01 + A11 * S11 + A12 * S21
        R_abs[i, 1, 2] = A10 * S02 + A11 * S12 + A12 * S22

        R_abs[i, 2, 0] = A20 * S00 + A21 * S10 + A22 * S20
        R_abs[i, 2, 1] = A20 * S01 + A21 * S11 + A22 * S21
        R_abs[i, 2, 2] = A20 * S02 + A21 * S12 + A22 * S22

    # 将旋转矩阵转换为6D表示(提取前两行并展平)
    abs_6d = np.empty((N, 6), dtype=np.float64)
    for i in prange(N):
        abs_6d[i, 0] = R_abs[i, 0, 0]
        abs_6d[i, 1] = R_abs[i, 0, 1]
        abs_6d[i, 2] = R_abs[i, 0, 2]
        abs_6d[i, 3] = R_abs[i, 1, 0]
        abs_6d[i, 4] = R_abs[i, 1, 1]
        abs_6d[i, 5] = R_abs[i, 1, 2]

    return abs_6d
