"""Virtual desktop robot with the x2robot client action queue, RTC fields, and synthetic WALL-OSS-0.5 observations.

The client's rules with the settings of the repository's load convention: a 200 Hz consumer, six control points
per model step (linear interpolation), the next request six model steps after the previous chunk's request
(three after the first response), and an ``inference_delay`` of the 75th percentile of the last five delays
(initially 5), clamped to 1..max_delay, plus two.
"""

from __future__ import annotations

import base64
import math
from collections import deque

import numpy as np

CONTROL_DT = 0.005  # s per control point
STEPS_PER_MODEL_STEP = 6  # control points per model step
EXECUTE_HORIZON = 6  # model steps between requests
FIRST_INFERENCE_STEP = 3  # model steps before the request that follows the first response
REQUEST_PERIOD_S = EXECUTE_HORIZON * STEPS_PER_MODEL_STEP * CONTROL_DT  # 180 ms while responses are faster
WIRE_OFFSET = 2  # added to the delay estimate on the wire
PREDICT_ACTION_OFFSET = 1
ACTION_HORIZON = 16  # rows of a chunk, for a model that returns no poses
NUM_FLOW_STEPS = 5  # sent with RTC requests when the service declares no flow steps
DELAY_QUEUE_SIZE = 5  # recent delays behind the estimate
INITIAL_DELAY_ESTIMATE = 5  # model steps, before the first response
# A generic instruction of 41 tokens, the length of the instructions the robots send, so that the served prompts
# have their usual length.
INSTRUCTION = ("Pick up the red cup from the left side of the table with the right arm, place it upright on the tray "
               "next to the blue plate, then move both arms back to their starting positions and stop.")


class Observations:
    """Seeded synthetic observations in the desktop robot wire schema: three JPEG camera views and random arm poses.

    Generate the camera views once per robot so JPEG encoding does not delay each request. Each
    arm's pose is a random xyz position within 0.5 m of the origin, roll/pitch/yaw within pi and a gripper value
    within 0..1, new for every request. WALL-OSS-0.5 writes the state into its prompt as text, so random poses vary the
    prompt length as real robots do.
    """

    def __init__(self, machine_id: str):
        seed = sum((i + 1) * ord(c) for i, c in enumerate(machine_id)) % 997
        self.rng = np.random.default_rng(seed)
        self.pose_rng = np.random.default_rng([seed, 1])
        self.views = {"camera_left": self._jpeg(), "camera_right": self._jpeg(),
                      "camera_front": self._jpeg(720, 1280)}

    def _pose(self) -> list[float]:
        uniform = self.pose_rng.uniform
        return [*uniform(-0.5, 0.5, 3).tolist(), *uniform(-np.pi, np.pi, 3).tolist(), float(uniform(0.0, 1.0))]

    def _jpeg(self, height: int = 480, width: int = 640) -> str:
        import cv2

        ok, encoded = cv2.imencode(".jpg", self.rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def make(self, rtc_fields: dict | None, num_flow_steps: int) -> dict:
        observation = {"state": {"follow1_pos": self._pose(), "follow2_pos": self._pose()},
                       "views": dict(self.views), "instruction": INSTRUCTION}
        if rtc_fields is not None:
            observation.update(rtc_fields)
            observation["num_flow_steps"] = num_flow_steps
        return observation


def actions(response: dict) -> np.ndarray | None:
    """The response's absolute poses ``[T, 14]`` (T >= 2, finite), or None if it has none."""
    try:
        if "action_pred" in response:
            poses = np.asarray(response["action_pred"], dtype=np.float32)
        else:
            poses = np.concatenate((np.asarray(response["follow1_pos"], dtype=np.float32),
                                    np.asarray(response["follow2_pos"], dtype=np.float32)), axis=-1)
    except (KeyError, TypeError, ValueError):
        return None
    if poses.ndim == 3 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 2 or poses.shape[1] != 14 or poses.shape[0] < 2 or not np.isfinite(poses).all():
        return None
    return np.ascontiguousarray(poses)


def shift(chunk: np.ndarray, steps: int) -> np.ndarray:
    """Drop the ``steps`` executed rows and repeat the last row to keep the length (absolute poses hold)."""
    kept = chunk[min(steps, chunk.shape[0] - 1):]
    return np.ascontiguousarray(np.concatenate([kept, np.repeat(chunk[-1:], chunk.shape[0] - kept.shape[0], 0)]))


class VirtualRobot:
    """One robot: an action queue consumed every ``CONTROL_DT`` while requests are in flight.

    Times are seconds on one monotonic clock starting when the robot starts. ``rtc_fields`` is called when an
    observation is captured; ``on_response`` after a successful response. ``next_trigger_s`` is when the next
    request is due (None before the first response). Only the queue's length matters for the requests, so the
    queue is kept as a count of control points. Without poses (a model without RTC) no RTC fields are sent.
    """

    def __init__(self, max_delay: int):
        self.max_delay = max_delay
        self.delays = deque([max(1, min(INITIAL_DELAY_ESTIMATE, max_delay))], maxlen=DELAY_QUEUE_SIZE)
        self.prev_actions: np.ndarray | None = None
        self.last_response_s: float | None = None
        self.next_trigger_s: float | None = None
        self.queue = 0  # control points left in the active chunk
        self.executed = 0  # control points consumed (or missed) since the robot started
        self.clock_s = 0.0
        self.chunk_base = 0  # ``executed`` when the active chunk's request was captured
        self.request_executed: int | None = None
        self.underflow = 0  # control points due while the queue was empty, since the last response

    def advance_to(self, now_s: float) -> None:
        now_s = max(now_s, self.clock_s)
        target = int(math.floor((now_s + 1e-12) / CONTROL_DT))
        ticks = max(0, target - self.executed)
        consumed = min(ticks, self.queue)
        self.queue -= consumed
        if self.last_response_s is not None:
            self.underflow += ticks - consumed
        self.executed, self.clock_s = target, now_s

    def delay_estimate(self) -> int:
        return max(1, min(int(np.percentile(list(self.delays), 75)), self.max_delay))

    def rtc_fields(self, capture_s: float) -> dict | None:
        """RTC request fields at an observation captured at ``capture_s``; None before the first response."""
        self.advance_to(capture_s)
        self.request_executed = self.executed
        if self.prev_actions is None:
            return None
        s_model = min((self.executed - self.chunk_base) // STEPS_PER_MODEL_STEP, ACTION_HORIZON)
        fields = {"rtc_mode": "training_time", "use_rtc": True,
                  "prev_action_chunk": shift(self.prev_actions, s_model) if s_model else self.prev_actions.copy(),
                  "inference_delay": self.delay_estimate() + WIRE_OFFSET, "s_model": s_model,
                  "execute_horizon": EXECUTE_HORIZON}
        index = s_model - PREDICT_ACTION_OFFSET
        if 0 <= index < self.prev_actions.shape[0]:
            fields["current_master_action"] = self.prev_actions[index].copy()
        return fields

    def on_response(self, latency_s: float, recv_s: float, poses: np.ndarray | None = None) -> dict:
        """Replace the queue with the new chunk, minus the control points that passed since its capture.

        Args:
            latency_s: send to response receipt, which feeds the delay estimate.
            recv_s: response receipt.
            poses: the response's absolute poses ``[T, 14]`` (``actions``), which later requests send back as
                RTC fields; None for a model without RTC, whose chunks are taken to have ``ACTION_HORIZON`` rows.

        Returns:
            ``delay_model`` (this response's delay in model steps), ``elapsed_control_steps`` (since capture),
            ``queue_coverage_ms`` after the update and ``underflow_ms``, the time the queue was empty since the
            previous response (from the first response on).
        """
        self.advance_to(recv_s)
        first = self.last_response_s is None
        if poses is not None:
            self.prev_actions = poses
        rows = ACTION_HORIZON if poses is None else poses.shape[0]
        delay_control = math.ceil(latency_s / CONTROL_DT)
        delay_model = max(1, min((delay_control + STEPS_PER_MODEL_STEP - 1) // STEPS_PER_MODEL_STEP, self.max_delay))
        self.delays.append(delay_model)
        dense_len = (rows - 1) * STEPS_PER_MODEL_STEP + 1  # control points of the interpolated chunk
        request = self.executed if self.request_executed is None else self.request_executed
        elapsed = max(0, self.executed - request)
        self.queue = dense_len - min(elapsed, dense_len - 1)
        self.chunk_base = request
        due = (FIRST_INFERENCE_STEP if first else EXECUTE_HORIZON) * STEPS_PER_MODEL_STEP
        self.next_trigger_s = recv_s + max(0, due - elapsed) * CONTROL_DT
        self.last_response_s = recv_s
        underflow, self.underflow = self.underflow, 0
        return {"delay_model": delay_model, "elapsed_control_steps": elapsed,
                "queue_coverage_ms": self.queue * CONTROL_DT * 1000, "underflow_ms": underflow * CONTROL_DT * 1000}
