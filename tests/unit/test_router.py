"""xinfer-router against stub servers (CPU): routing, affinity, retries, pausing, admission, discovery."""

import asyncio
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from pathlib import Path

import pytest

pytest.importorskip("websockets")
pytest.importorskip("msgpack_numpy")
import websockets  # noqa: E402
from websockets.exceptions import InvalidStatus  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from xinfer.frontend.protocol import packb, unpackb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Stub:
    """A server stub on 127.0.0.1: the metadata frame, ``GET /info``, and requests answered by ``mode``:

    ``"ok"`` (``{"status": "ok", "server": port}``), ``"close"`` (closes the connection), ``"freeze"`` (blocks its
    event loop for 8 s, as a machine that lost power answers nothing, not even pings), or a status to answer.
    A request's ``sleep`` delays the answer.
    """

    def __init__(self, model="stub", capacity=None, connection_state=False, mode="ok"):
        self.port, self.mode, self.requests = _free_port(), mode, 0
        self.address = f"127.0.0.1:{self.port}"
        hello = packb({"model": model})
        self.info = dict(model=model, capacity=capacity, connection_state=connection_state,
                         metadata_sha256=hashlib.sha256(hello).hexdigest())
        started = threading.Event()

        async def handler(websocket):
            await websocket.send(hello)
            async for message in websocket:
                self.requests += 1
                await asyncio.sleep(unpackb(message).get("sleep", 0))
                if self.mode == "close":
                    await websocket.close()
                    return
                if self.mode == "freeze":
                    time.sleep(8)
                status = "ok" if self.mode == "ok" else self.mode
                await websocket.send(packb({"status": status, "server": self.port}))

        def process_request(connection, request):
            if request.path == "/info":
                return connection.respond(HTTPStatus.OK, json.dumps(self.info))
            return None

        async def run():
            async with websockets.serve(handler, "127.0.0.1", self.port, process_request=process_request):
                started.set()
                await asyncio.Future()

        threading.Thread(target=asyncio.run, args=(run(),), daemon=True).start()
        started.wait(5)


def _router(*workers, joined=None, options=()):
    """Start the router on the stubs (``Stub`` or spec strings) and wait until ``joined`` servers (default: all) joined
    its pool; returns (port, process)."""
    port = _free_port()
    specs = [w.address if isinstance(w, Stub) else w for w in workers]
    process = subprocess.Popen([sys.executable, "-m", "xinfer.entrypoints.router", "--host", "127.0.0.1",
                                "--port", str(port), "--workers", *specs, *options],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               env=dict(os.environ, PYTHONPATH=str(ROOT)))
    lines = []
    threading.Thread(target=lambda: lines.extend(process.stdout), daemon=True).start()
    deadline = time.time() + 20
    while sum("joined the pool" in line for line in lines) < (joined or len(workers)):
        if time.time() > deadline or process.poll() is not None:
            process.kill()
            pytest.fail("".join(lines))
        time.sleep(0.05)
    return port, process


@pytest.fixture
def router():
    started = []

    def start(*workers, joined=None, options=()):
        port, process = _router(*workers, joined=joined, options=options)
        started.append(process)
        return port

    yield start
    for process in started:
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 0


def _ask(websocket, **request):
    websocket.send(packb(request))
    return unpackb(websocket.recv())


def test_requests_go_to_the_server_with_the_fewest_requests_in_flight(router):
    a, b = Stub(), Stub()
    port = router(a, b)
    with connect(f"ws://127.0.0.1:{port}") as first, connect(f"ws://127.0.0.1:{port}") as second:
        assert unpackb(first.recv()) == unpackb(second.recv()) == {"model": "stub"}
        first.send(packb({"sleep": 0.5}))
        time.sleep(0.2)
        quick = _ask(second)
        slow = unpackb(first.recv())
    assert quick["status"] == slow["status"] == "ok" and quick["server"] != slow["server"]


def test_a_failed_request_is_retried_and_a_failing_server_paused(router):
    failing, healthy = sorted([Stub(), Stub()], key=lambda s: s.address)  # the router tries them in this order
    failing.mode = "close"
    port = router(failing, healthy)
    with connect(f"ws://127.0.0.1:{port}") as websocket:
        websocket.recv()
        answers = [_ask(websocket) for _ in range(8)]
    assert all(a["status"] == "ok" and a["server"] == healthy.port for a in answers)
    assert failing.requests == 3  # then paused


def test_a_failed_request_goes_to_every_other_server_within_the_deadline(router):
    """Two of three servers fail (as two draining at once): the router tries them in the order first, third, then
    the second, which answers."""
    first, healthy, third = sorted([Stub(), Stub(), Stub()], key=lambda s: s.address)
    first.mode = third.mode = "close"
    port = router(first, healthy, third, options=("--retry-deadline-ms", "1000"))
    with connect(f"ws://127.0.0.1:{port}") as websocket:
        websocket.recv()
        answer = _ask(websocket)
    assert answer["status"] == "ok" and answer["server"] == healthy.port
    assert first.requests == third.requests == 1


def test_a_request_that_fails_after_the_deadline_is_answered_at_once(router):
    failing, healthy = sorted([Stub(), Stub()], key=lambda s: s.address)
    failing.mode = "queue_full"
    port = router(failing, healthy)
    with connect(f"ws://127.0.0.1:{port}") as websocket:
        websocket.recv()
        assert _ask(websocket, sleep=0.05)["status"] == "queue_full"  # the first server, past the 30 ms deadline
        assert healthy.requests == 0
        assert _ask(websocket)["status"] == "ok"  # the next request: the other server in turn


def test_a_robot_stays_on_one_server_until_it_fails(router):
    stubs = [Stub(connection_state=True), Stub(connection_state=True)]
    port = router(*stubs)
    with connect(f"ws://127.0.0.1:{port}") as websocket:
        websocket.recv()
        served = {_ask(websocket, robot_id="r1")["server"] for _ in range(4)}
    assert len(served) == 1
    with connect(f"ws://127.0.0.1:{port}") as websocket:  # a new connection of the same robot
        websocket.recv()
        assert _ask(websocket, robot_id="r1")["server"] in served
        pinned = next(s for s in stubs if s.port in served)
        pinned.mode = "close"
        moved = _ask(websocket, robot_id="r1")
        assert moved["status"] == "ok" and moved["server"] != pinned.port  # retried on the other server
        pinned.mode = "ok"
        assert _ask(websocket, robot_id="r1")["server"] == moved["server"]  # and stays there


def test_robots_beyond_the_pool_capacity_are_refused(router):
    port = router(Stub(capacity=1), Stub(capacity=1))
    with connect(f"ws://127.0.0.1:{port}"), connect(f"ws://127.0.0.1:{port}"):
        with pytest.raises(InvalidStatus) as refused:
            connect(f"ws://127.0.0.1:{port}")
        assert refused.value.response.status_code == 503


def test_a_server_with_other_metadata_is_left_out(router):
    first, other = sorted([Stub(model="a"), Stub(model="b")], key=lambda s: s.address)  # the first sets the pool
    port = router(first, other, joined=1)
    with connect(f"ws://127.0.0.1:{port}") as websocket:
        assert unpackb(websocket.recv()) == {"model": first.info["model"]}
        assert {_ask(websocket)["server"] for _ in range(4)} == {first.port}
    assert other.requests == 0


def test_servers_found_by_dns(router):
    stub = Stub()
    port = router(f"dns:localhost:{stub.port}")
    with connect(f"ws://127.0.0.1:{port}") as websocket:
        websocket.recv()
        assert _ask(websocket)["server"] == stub.port


def test_a_server_that_stops_answering_is_given_up_within_seconds(router):
    frozen = Stub()
    port = router(frozen)
    with connect(f"ws://127.0.0.1:{port}") as websocket:
        websocket.recv()
        frozen.mode = "freeze"
        started = time.time()
        assert _ask(websocket)["status"] == "unavailable"
        assert time.time() - started < 5  # a ping times out within 3 s, closing takes at most 1 s more

