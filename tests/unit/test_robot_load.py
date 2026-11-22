"""benchmarks/robot_load.py and virtual_robot.py against an in-process fake service (CPU)."""

import contextlib
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack_numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import robot_load  # noqa: E402
import virtual_robot  # noqa: E402


@contextlib.contextmanager
def service(reply, received=None):
    """A WebSocket service that sends a metadata frame, then answers each request with ``reply(request)``."""
    from websockets.sync.server import serve

    def handler(connection):
        connection.send(robot_load.packb({"model": "fake"}))
        for message in connection:
            request = robot_load.unpackb(message)
            if received is not None:
                received.append(request)
            connection.send(robot_load.packb(reply(request)))

    with serve(handler, "127.0.0.1", 0, max_size=None) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        server.shutdown()


def _summary(out):
    return json.loads((out / "summary.json").read_text())


def test_virtual_robot_follows_the_client_queue_and_delay_rules():
    robot = virtual_robot.VirtualRobot(6)
    assert robot.rtc_fields(1.0) is None  # the first request is a flow request (200 control points in)
    poses = np.arange(16 * 14, dtype=np.float32).reshape(16, 14)
    metrics = robot.on_response(latency_s=0.0625, recv_s=1.0625, poses=poses)
    # 12 control points passed: 13 control points of latency are 3 model steps; 91 interpolated points remain 79.
    assert metrics == {"delay_model": 3, "elapsed_control_steps": 12, "queue_coverage_ms": 395.0, "underflow_ms": 0.0}
    assert robot.next_trigger_s == pytest.approx(1.0625 + 6 * 0.005)  # three model steps after the capture
    fields = robot.rtc_fields(1.0925)  # 18 control points since the chunk's capture: 3 model steps executed
    assert fields["s_model"] == 3 and fields["inference_delay"] == 4 + 2  # 75th percentile of [5, 3] is 4
    assert np.array_equal(fields["prev_action_chunk"], np.concatenate([poses[3:], np.repeat(poses[-1:], 3, 0)]))
    assert np.array_equal(fields["current_master_action"], poses[2])
    robot.advance_to(1.0925 + 100 * 0.005)  # 73 points left, 100 due: 27 points of underflow
    assert robot.on_response(0.01, 1.0925 + 100 * 0.005, poses)["underflow_ms"] == pytest.approx(27 * 5)
    robot.rtc_fields(1.5925)  # one point left of the stale chunk
    assert robot.on_response(0.01, 1.6025, poses)["underflow_ms"] == pytest.approx(5)  # 2 due: this response only


def test_the_summary_counts_the_underflow_of_steady_responses():
    def row(robot, warmup, underflow_ms):
        return {"robot": robot, "warmup": warmup, "ok": True, "start_s": 0.0, "end_s": 1.0, "server_infer_ms": 1.0,
                "client_ms": 1.0, "underflow_ms": underflow_ms}

    rows = [row(0, True, 2000.0), row(0, False, 5.0), row(0, False, 7.0), row(1, False, 10.0)]
    assert robot_load.summarize(rows, [{}, {}], 4)["underflow_ms"] == 22.0


def test_a_robot_without_rtc_follows_the_same_triggers_without_rtc_fields():
    robot = virtual_robot.VirtualRobot(6)
    assert robot.rtc_fields(1.0) is None
    metrics = robot.on_response(latency_s=0.0625, recv_s=1.0625)  # no poses: chunks of ACTION_HORIZON (16) rows
    assert metrics["queue_coverage_ms"] == pytest.approx(((16 - 1) * 6 + 1 - 12) * 5.0)
    assert robot.next_trigger_s == pytest.approx(1.0625 + 6 * 0.005) and robot.rtc_fields(1.0925) is None


def test_frozen_requests_are_sent_in_turn_without_rtc_fields(tmp_path):
    requests = tmp_path / "requests.msgpack"
    requests.write_bytes(robot_load.packb([{"x": np.zeros(3, np.float32)}, {"x": np.ones(3, np.float32)}]))
    reply = {"status": "ok", "y": np.zeros(3), "server_timing": {"infer_ms": 1.0}}
    received = []
    with service(lambda request: reply, received) as url:
        code = robot_load.main(["--url", url, "--robots", "2",
                                "--requests-file", str(requests), "--requests", "6", "--warmup-requests", "2",
                                "--rounds", "2", "--out", str(tmp_path / "out")])
    assert code == 0 and len(received) == 2 * 2 * 6
    assert all(set(request) == {"x"} for request in received)  # sent as given, without RTC fields
    summary = _summary(tmp_path / "out")
    assert len(summary["rounds"]) == 2
    for index, item in enumerate(summary["rounds"], 1):
        assert (item["rows"], item["errors"], item["missing"], item["steady"]) == (12, 0, 0, 8)
        assert item["server_infer_ms"]["p50"] == 1.0 and item["sent_inference_delay"] == {}
        assert item["start_delays_ms"] == [round(d * 1000, 1) for d in robot_load.start_delays(2, 0, index)]
    rows = (tmp_path / "out" / "round_1.csv").read_text().splitlines()
    assert len(rows) == 1 + 12 and rows[0].startswith("round,robot,sequence")


def test_start_delays_are_seeded_per_round_within_one_request_period():
    delays = robot_load.start_delays(4, 0, 1)
    assert delays == robot_load.start_delays(4, 0, 1)
    assert delays != robot_load.start_delays(4, 0, 2) and delays != robot_load.start_delays(4, 1, 1)
    assert virtual_robot.REQUEST_PERIOD_S == pytest.approx(0.18)
    assert all(0.0 <= delay < virtual_robot.REQUEST_PERIOD_S for delay in delays)


def test_request_files_may_be_json_or_msgpack_with_one_request_or_a_list(tmp_path):
    (tmp_path / "one.json").write_text(json.dumps({"x": [1.0]}))
    (tmp_path / "two.json").write_text(json.dumps([{"x": [2.0]}, {"x": [3.0]}]))
    (tmp_path / "one.msgpack").write_bytes(robot_load.packb({"x": np.full(1, 4.0, np.float32)}))
    requests = robot_load.load_requests([tmp_path / name for name in ("one.json", "two.json", "one.msgpack")])
    assert [float(np.asarray(request["x"])[0]) for request in requests] == [1.0, 2.0, 3.0, 4.0]


def test_errors_fail_the_run(tmp_path):
    requests = tmp_path / "requests.msgpack"
    requests.write_bytes(robot_load.packb([{"x": 1}, {"x": 2, "fail": True}]))

    def reply(request):
        if request.get("fail"):
            return {"status": "internal_error", "error": "stage failed"}
        return {"status": "ok", "y": 1, "server_timing": {"infer_ms": 5.0}}

    with service(reply) as url:
        assert robot_load.main(["--url", url, "--requests-file", str(requests), "--requests", "4",
                                "--warmup-requests", "1", "--rounds", "1", "--out", str(tmp_path / "out")]) == 1
    item = _summary(tmp_path / "out")["rounds"][0]
    assert item["errors"] == 2 and item["server_infer_ms"]["p50"] == 5.0  # the failing request is sent twice


def test_wallx_robots_send_rtc_fields_from_the_previous_response(tmp_path):
    pytest.importorskip("cv2")
    poses = np.zeros((17, 7), np.float32)
    received = []
    reply = {"status": "ok", "follow1_pos": poses, "follow2_pos": poses, "server_timing": {"infer_ms": 2.0}}
    with service(lambda request: reply, received) as url:
        code = robot_load.main(["--url", url, "--requests", "5", "--warmup-requests", "1", "--rounds", "1",
                                "--out", str(tmp_path / "out")])
    assert code == 0 and len(received) == 5
    assert "prev_action_chunk" not in received[0] and received[0]["views"]["camera_front"].shape == (720, 1280, 3)
    for request in received[1:]:
        assert request["prev_action_chunk"].shape == (17, 14) and 3 <= request["inference_delay"] <= 8
        assert request["num_flow_steps"] == 5 and request["instruction"] == virtual_robot.INSTRUCTION
    item = _summary(tmp_path / "out")["rounds"][0]
    assert (item["rows"], item["errors"], item["steady"]) == (5, 0, 4)
    assert sum(item["sent_inference_delay"].values()) == 4  # every steady request carries RTC fields


@pytest.mark.parametrize(("metadata", "steps"), [({"model": "wall-oss-0.5"}, 5),
                                 ({"model": "wall-oss-1.0-nano", "flow_steps": 10}, 10),
                                                  ({"rtc": {"num_flow_steps": 10}}, 10), (None, 5)])
def test_rtc_requests_ask_for_the_flow_steps_the_service_declares(metadata, steps):
    assert robot_load.flow_steps(metadata) == steps


def test_a_wallx_response_without_poses_is_an_error(tmp_path):
    pytest.importorskip("cv2")
    with service(lambda request: {"status": "ok"}) as url:
        assert robot_load.main(["--url", url, "--requests", "2", "--warmup-requests", "0", "--rounds", "1",
                                "--out", str(tmp_path / "out")]) == 1
    rows = (tmp_path / "out" / "round_1.csv").read_text()
    assert _summary(tmp_path / "out")["rounds"][0]["errors"] == 2 and "absolute poses" in rows


def test_wallx_observations_have_seeded_random_poses_and_views_generated_once():
    pytest.importorskip("cv2")
    first, again = virtual_robot.Observations("robot-0"), virtual_robot.Observations("robot-0")
    assert first.make(None, 5)["state"] == again.make(None, 5)["state"]  # reproducible per robot
    views = [first.make(None, 5)["views"] for _ in range(2)]
    assert all(views[0][name] is views[1][name] for name in views[0])  # no per-request generation
    poses = [pose for _ in range(3) for pose in first.make(None, 5)["state"].values()]
    assert len({tuple(pose) for pose in poses}) == 6
    for pose in poses:
        assert len(pose) == 7 and max(map(abs, pose[:3])) <= 0.5 and max(map(abs, pose[3:6])) <= np.pi
        assert 0.0 <= pose[6] <= 1.0


