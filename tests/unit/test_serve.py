"""End to end: engine process + frontend processes + WebSocket clients (CPU)."""

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack_numpy")
from websockets.sync.client import connect  # noqa: E402

from xinfer.frontend.protocol import packb, unpackb  # noqa: E402
from xinfer.entrypoints.serve import _per_stage, parse_args  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def test_cli_op_backends_pin_operators_over_reference_ops():
    from xinfer import ops
    from xinfer.entrypoints.serve import _op_backends
    from xinfer.platforms import resolve_platform

    assert _op_backends(None) == {} and _op_backends("a=x, b = y") == {"a": "x", "b": "y"}
    table = ops.registry.build_table(resolve_platform("cpu"), backends=_op_backends("linear=reference"),
                                     reference_only=True)
    assert table.selection("linear").backend == "reference"


def test_cli_parses_signed_stream_priorities():
    args = parse_args(["--model", "m", "--checkpoint", "unused", "--stream-priorities", "a=0,b=-1"])
    assert _per_stage(args.stream_priorities) == {"a": 0, "b": -1}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def resolved_path(tmp_path_factory):
    return tmp_path_factory.mktemp("serve") / "resolved.json"


def _start(*options):
    """Start ``xinfer.entrypoints.serve`` with the echo model and wait for READY; returns (port, process, lines)."""
    port = _free_port()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(HERE)]))
    process = subprocess.Popen(
        [sys.executable, "-m", "xinfer.entrypoints.serve", "--model", "echo_model:EchoModel",
         "--checkpoint", "unused", "--device", "cpu", "--host", "127.0.0.1", "--port", str(port),
         "--shm-buffer-mib", "1", "--model-options", '{"scale": 3.0}', *options],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    deadline = time.time() + 60
    lines = []
    for line in process.stdout:
        lines.append(line)
        if line.startswith("XINFER READY"):
            break
        if time.time() > deadline:
            break
    else:
        pytest.fail("server exited: " + "".join(lines))
    threading.Thread(target=lambda: [lines.append(line) for line in process.stdout], daemon=True).start()
    return port, process, lines


@pytest.fixture(scope="module")
def server(resolved_path):
    port, process, lines = _start("--frontends", "2", "--stream-priorities", "act=0",
                                  "--resolved-config", str(resolved_path))
    yield port, process, lines
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)


def test_metadata_frame_and_responses(server):
    port, _, _ = server
    with connect(f"ws://127.0.0.1:{port}", max_size=None) as websocket:
        assert unpackb(websocket.recv()) == {"model": "echo", "scale": 3.0, "env": "x2robot"}
        for i in range(5):
            websocket.send(packb({"x": np.arange(4, dtype=np.float32) + i}))
            reply = unpackb(websocket.recv())
            np.testing.assert_array_equal(reply["y"], (np.arange(4) + i) * 3.0 + 1.0)
            assert reply["status"] == "ok" and reply["server_timing"]["infer_ms"] > 0


def test_concurrent_connections_and_errors(server):
    port, _, _ = server
    failures = []

    def robot(k):
        try:
            with connect(f"ws://127.0.0.1:{port}", max_size=None) as websocket:
                websocket.recv()
                for i in range(20):
                    websocket.send(packb({"x": np.full(3, k * 100 + i, dtype=np.float32)}))
                    y = unpackb(websocket.recv())["y"]
                    if not np.array_equal(y, np.full(3, (k * 100 + i) * 3.0 + 1.0)):
                        failures.append((k, i, y))
                websocket.send(packb({"nope": 1}))
                reply = unpackb(websocket.recv())
                if reply["status"] != "invalid_request" or "missing x" not in reply["error"]:
                    failures.append((k, "error", reply))
        except Exception as exc:  # noqa: BLE001
            failures.append((k, repr(exc)))

    threads = [threading.Thread(target=robot, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not failures


def test_each_connection_gets_its_own_processor_and_protocol_errors_carry_their_info(server):
    port, _, _ = server
    for _ in range(2):
        with connect(f"ws://127.0.0.1:{port}", max_size=None) as websocket:
            websocket.recv()
            calls = []
            for _ in range(3):
                websocket.send(packb({"x": np.zeros(3, dtype=np.float32)}))
                calls.append(unpackb(websocket.recv())["connection_call"])
            assert calls == [1, 2, 3]
            websocket.send(packb({"x": np.zeros(3, dtype=np.float32), "refuse": True}))
            assert unpackb(websocket.recv()) == {"status": "invalid_request", "error": "refused",
                                                  "error_info": {"code": "echo_refused", "retryable": False}}


def test_a_failed_stage_is_an_internal_error_and_the_connection_stays(server):
    port, _, _ = server
    with connect(f"ws://127.0.0.1:{port}", max_size=None) as websocket:
        websocket.recv()
        websocket.send(packb({"x": np.zeros(2, dtype=np.float32), "fail": True}))
        reply = unpackb(websocket.recv())
        assert reply["status"] == "internal_error" and "stage failed" in reply["error"]
        assert "server_timing" not in reply
        websocket.send(packb({"x": np.ones(2, dtype=np.float32)}))
        assert unpackb(websocket.recv())["status"] == "ok"


class _BlockingFinalizer:
    """Holds the frontend's main thread in ``__del__`` until the test has sent SIGTERM."""

    def __init__(self, pipe):
        self.pipe = pipe

    def __del__(self):
        self.pipe.send("finalizing")
        self.pipe.recv()  # SIGTERM arrives here; a Python-level handler would raise inside __del__


def _frontend_inside_a_finalizer(index, pipe, options, notify):
    """``run_frontend`` whose event loop first runs a finalizer that blocks until the test's go-ahead."""
    import asyncio

    from xinfer.frontend import server

    serve = server._serve

    async def blocked(*args):
        asyncio.get_running_loop().call_soon(_BlockingFinalizer, notify)  # the instance is dropped at once
        await serve(*args)

    server._serve = blocked
    server.run_frontend(index, pipe, options)


def test_a_frontend_exits_on_sigterm_received_inside_a_finalizer():
    import multiprocessing
    import uuid

    context = multiprocessing.get_context("spawn")
    engine_end, frontend_end = context.Pipe()
    test_end, finalizer_end = context.Pipe()
    options = dict(model="echo_model:EchoModel", checkpoint="unused", model_options={}, host="127.0.0.1",
                   port=_free_port(), buffers=1, buffer_bytes=1 << 20, workers=1,
                   shm_prefix=f"xinfer-{uuid.uuid4().hex[:8]}")
    frontend = context.Process(target=_frontend_inside_a_finalizer, args=(0, frontend_end, options, finalizer_end))
    frontend.start()
    try:
        assert test_end.poll(60) and test_end.recv() == "finalizing"
        frontend.terminate()  # SIGTERM while the main thread runs the finalizer
        test_end.send("go")
        frontend.join(timeout=30)
        assert frontend.exitcode == 0
    finally:
        if frontend.is_alive():
            frontend.kill()


def test_a_frontend_processes_requests_of_different_connections_at_once():
    """Two connections' requests on one frontend with two processing threads are prepared together: each prepare
    waits at a two-party barrier, which passes only if both run at the same time."""
    port, process, _ = _start("--frontends", "1", "--frontend-workers", "2")
    try:
        with connect(f"ws://127.0.0.1:{port}", max_size=None) as a, \
                connect(f"ws://127.0.0.1:{port}", max_size=None) as b:
            for websocket in (a, b):
                websocket.recv()
                websocket.send(packb({"x": np.ones(2, dtype=np.float32), "rendezvous": True}))
            replies = [unpackb(websocket.recv()) for websocket in (a, b)]
        assert [reply["status"] for reply in replies] == ["ok", "ok"], replies
        assert [reply["y"].tolist() for reply in replies] == [[4.0, 4.0]] * 2
    finally:
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 0


def test_the_server_exits_when_its_engine_stops():
    port, process, lines = _start("--frontends", "1")
    with connect(f"ws://127.0.0.1:{port}", max_size=None) as websocket:
        websocket.recv()
        websocket.send(packb({"x": np.zeros(2, dtype=np.float32), "exit": True}))  # ends the engine thread
    assert process.wait(timeout=30) == 1
    expected = "engine cannot serve (xinfer-engine thread stopped)"
    deadline = time.time() + 10  # the reader thread may still be appending the last lines
    while not any(expected in line for line in lines) and time.time() < deadline:
        time.sleep(0.01)
    assert any(expected in line for line in lines)


def test_frontends_run_the_warmup_requests_through_their_processors_before_ready(tmp_path):
    """Each frontend prepares every ``--warmup`` request and finishes it with the engine's outputs before READY,
    on processors of its own: serving starts with processors that carry no warmup state."""
    import json

    one = tmp_path / "warmup.msgpack"
    one.write_bytes(packb({"x": np.arange(3, dtype=np.float32)}))
    two = tmp_path / "warmup.json"  # a JSON file with a list of requests
    two.write_text(json.dumps([{"x": [1.0, 1.0, 1.0]}, {"x": [2.0, 2.0, 2.0]}]))
    record = tmp_path / "finished.txt"
    port, process, _ = _start("--frontends", "2", "--warmup", str(one), str(two),
                              "--model-options", json.dumps({"scale": 3.0, "record": str(record)}))
    try:
        warmed = [line.split() for line in record.read_text().splitlines()]  # written before READY
        assert len(warmed) == 6 and len({pid for pid, *_ in warmed}) == 2
        assert str(process.pid) not in {pid for pid, *_ in warmed}  # the frontends, not the engine process
        expected = {"1": [1.0, 4.0, 7.0], "2": [4.0] * 3, "3": [7.0] * 3}  # the engine's outputs: 3x + 1
        for pid in {pid for pid, *_ in warmed}:
            mine = {call: [float(v) for v in y] for frontend, call, *y in warmed if frontend == pid}
            assert mine == expected  # every request, in order, on each frontend's own warmup processors
        with connect(f"ws://127.0.0.1:{port}", max_size=None) as websocket:
            websocket.recv()
            websocket.send(packb({"x": np.zeros(2, dtype=np.float32)}))
            reply = unpackb(websocket.recv())
        assert reply["status"] == "ok" and reply["call"] == 1  # the serving processors' first call
    finally:
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 0


def test_health_endpoint(server):
    import urllib.request

    port, _, _ = server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
        assert response.status == 200


def test_shutdown_is_clean(server):
    port, process, lines = server
    shm_before = {p for p in os.listdir("/dev/shm") if p.startswith("xinfer-")}
    assert shm_before  # frontends own their buffers while running
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=30) == 0, "".join(lines[-20:])
    time.sleep(0.5)
    assert not ({p for p in os.listdir("/dev/shm") if p.startswith("xinfer-")} & shm_before)


def test_startup_resolves_and_records_the_configuration(server, resolved_path):
    import json

    settings = {r["name"]: r for r in json.loads(resolved_path.read_text())["settings"]}
    assert settings["max_running_requests"] == {"name": "max_running_requests", "value": 32, "source": "default"}
    assert settings["frontends"] == {"name": "frontends", "value": 2, "source": "explicit"}
    assert settings["cudagraphs"]["value"] is False  # CPU platform
    assert settings["lanes.act"] == {"name": "lanes.act", "value": 2, "source": "default"}
    assert settings["stream_priorities.act"] == {"name": "stream_priorities.act", "value": 0, "source": "explicit"}
