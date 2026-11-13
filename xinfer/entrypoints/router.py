"""xinfer-router: one port in front of the single-GPU workers (``xinfer-serve``) of one pool (docs/router.md).

Robots connect to the router as they would to a worker. It sends them the pool's metadata frame and forwards each
request message unchanged to a worker, over the robot's own connection to that worker, then sends the response back:

- a model without per-connection state (``connection_state`` false in ``GET /info``): each request goes to the worker
  with the fewest of the router's requests in flight, relative to its ``capacity``;
- a model with it (an InputProcessor with ``connect``): a robot's connection stays with one worker, chosen at its
  first request by rendezvous hashing on the request's ``robot_id`` among the workers below capacity.

A worker that fails ``FAILURES_TO_PAUSE`` requests in a row (connection error, closed connection, or a status in
``RETRYABLE_STATUSES``) gets no requests for ``PAUSE_SECONDS``; then one failure pauses it again. A failed request is
sent to the other workers in turn while it is at most ``--retry-deadline-ms`` old, else its failure is answered at
once. Workers
come from ``host:port`` entries or ``dns:<name>:<port>`` (re-resolved every second); a worker joins once ``GET /info``
answers with the pool's metadata. With every worker's ``capacity`` known, a robot connection beyond their sum is
refused (HTTP 503).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import logging
import signal
import socket
import time
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus

import msgpack
import websockets
from websockets.exceptions import WebSocketException

from xinfer.frontend.protocol import QUEUE_FULL, TIMEOUT, UNAVAILABLE, packb

logger = logging.getLogger("xinfer.router")

FAILURES_TO_PAUSE = 3  # failed requests in a row that pause a worker
PAUSE_SECONDS = 5.0
RETRYABLE_STATUSES = {QUEUE_FULL, TIMEOUT, UNAVAILABLE}  # responses another worker may serve


@dataclass
class Worker:
    """One worker as the router sees it."""

    address: str
    capacity: int | None
    requests_in_flight: int = 0  # the router's requests to it, not answered yet
    assigned_robots: int = 0  # robot connections assigned to it (models with per-connection state)
    consecutive_failures: int = 0  # failed requests in a row
    paused_until: float = 0.0


class Router:
    """The router of one pool (see the module docstring).

    Args:
        worker_addresses: ``host:port`` or ``dns:<name>:<port>`` entries.
        retry_deadline_s: seconds after which a failed request is answered with its failure instead of being sent to
            another worker.
    """

    def __init__(self, worker_addresses: list[str], retry_deadline_s: float):
        self.worker_addresses, self.retry_deadline_s = worker_addresses, retry_deadline_s
        self.workers: dict[str, Worker] = {}
        self.metadata_frame = None  # the pool's metadata frame, from its first worker
        self.metadata_sha256 = None
        self.connection_state = False  # the first worker's connection_state
        self.connected_robots = 0  # open robot connections
        self.round_robin = itertools.count()

    async def discover_workers(self) -> None:
        """Resolve the worker addresses; add each new one whose ``GET /info`` matches the pool, drop vanished ones."""
        loop = asyncio.get_running_loop()
        addresses = set()
        for spec in self.worker_addresses:
            name, port = spec.removeprefix("dns:").rsplit(":", 1)
            if spec.startswith("dns:"):
                try:
                    found = await loop.getaddrinfo(name, int(port), family=socket.AF_INET, type=socket.SOCK_STREAM)
                except OSError:
                    continue
                addresses |= {f"{entry[4][0]}:{port}" for entry in found}
            else:
                addresses.add(spec)
        for address in set(self.workers) - addresses:
            logger.info("%s left the pool", address)
            del self.workers[address]
        for address in sorted(addresses - set(self.workers)):
            try:
                info = await asyncio.to_thread(_fetch_info, address)
                if self.metadata_frame is None:
                    self.metadata_frame = await _fetch_metadata_frame(address)
                    self.metadata_sha256 = info.get("metadata_sha256")
                    self.connection_state = info.get("connection_state")
            except (OSError, ValueError, asyncio.TimeoutError, WebSocketException):
                continue
            if info.get("metadata_sha256") != self.metadata_sha256:
                logger.warning("%s serves other metadata than the pool; left out", address)
                continue
            logger.info("%s joined the pool (capacity %s)", address, info.get("capacity"))
            self.workers[address] = Worker(address, info.get("capacity"))

    def _pick_worker(self, failed_workers, robot_id) -> Worker | None:
        """The worker for a request, among those not in ``failed_workers`` and not paused (all of them when every
        one is paused): the fewest requests in flight relative to capacity, ties going round; for a model with
        per-connection state, rendezvous hashing on ``robot_id`` among the workers below capacity, or the fewest
        robots relative to capacity without an id."""
        workers = [w for w in self.workers.values() if w not in failed_workers]
        now = time.monotonic()
        workers = [w for w in workers if w.paused_until <= now] or workers
        if not workers:
            return None
        if not self.connection_state:
            start = next(self.round_robin) % len(workers)
            return min(workers[start:] + workers[:start], key=lambda w: w.requests_in_flight / (w.capacity or 1))
        if robot_id is None:
            return min(workers, key=lambda w: w.assigned_robots / (w.capacity or 1))
        by_hash = sorted(workers, key=lambda w: hashlib.sha256(f"{robot_id}@{w.address}".encode()).digest())
        return next((w for w in by_hash if w.capacity is None or w.assigned_robots < w.capacity), by_hash[0])

    def _record_result(self, worker: Worker, succeeded: bool) -> None:
        """Count a request's outcome: a success clears the worker's failures, ``FAILURES_TO_PAUSE`` in a row pause
        it."""
        worker.consecutive_failures = 0 if succeeded else worker.consecutive_failures + 1
        if worker.consecutive_failures == FAILURES_TO_PAUSE:
            logger.warning("%s failed %d requests in a row; paused for %.0f s", worker.address, FAILURES_TO_PAUSE,
                           PAUSE_SECONDS)
        if worker.consecutive_failures >= FAILURES_TO_PAUSE:
            worker.paused_until = time.monotonic() + PAUSE_SECONDS

    async def _send_to_worker(self, request, received_at, worker_connections, assigned_worker, robot_id):
        """The response to one request message: from the robot's worker, else from the other workers in turn, each
        once, within the deadline; else the last failure (``unavailable`` when no worker answered).

        Returns:
            ``(response, assigned_worker)``: the response, and the worker the robot's connection is assigned to (models
            with per-connection state), or None.
        """
        failed_workers, response = [], None
        while True:
            worker = assigned_worker or self._pick_worker(failed_workers, robot_id)
            if worker is None:
                return response or packb({"status": UNAVAILABLE, "error": "no worker is available"}), None
            if self.connection_state and assigned_worker is None:
                assigned_worker = worker
                worker.assigned_robots += 1
            worker.requests_in_flight += 1
            try:
                if worker.address not in worker_connections:
                    worker_connections[worker.address] = await _connect_worker(worker.address)
                await worker_connections[worker.address].send(request)
                response = await worker_connections[worker.address].recv()
                succeeded = _top_level_field(response, "status") not in RETRYABLE_STATUSES
            except (OSError, asyncio.TimeoutError, WebSocketException):
                worker_connections.pop(worker.address, None)
                succeeded = False
            finally:
                worker.requests_in_flight -= 1
            self._record_result(worker, succeeded)
            if succeeded:
                return response, assigned_worker
            failed_workers.append(worker)
            if worker is assigned_worker:
                worker.assigned_robots -= 1
                assigned_worker = None
            if time.monotonic() - received_at > self.retry_deadline_s:
                return response or packb({"status": UNAVAILABLE, "error": f"{worker.address} failed"}), assigned_worker

    async def serve_robot(self, websocket) -> None:
        """Serve one robot connection: the metadata frame, then each request through ``_send_to_worker``."""
        self.connected_robots += 1
        worker_connections, assigned_worker, robot_id = {}, None, None
        try:
            await websocket.send(self.metadata_frame)
            async for request in websocket:
                received_at = time.monotonic()
                if self.connection_state and assigned_worker is None:
                    robot_id = _top_level_field(request, "robot_id")
                response, assigned_worker = await self._send_to_worker(request, received_at, worker_connections,
                                                                       assigned_worker, robot_id)
                await websocket.send(response)
        finally:
            self.connected_robots -= 1
            if assigned_worker is not None:
                assigned_worker.assigned_robots -= 1
            for connection in worker_connections.values():
                await connection.close()

    def process_request(self, connection, request):
        """``/healthz``; ``/ready`` (200 once a worker joined); refuse a robot connection without a worker or beyond
        the workers' summed capacity, when every worker reports one."""
        if request.path == "/healthz":
            return connection.respond(HTTPStatus.OK, "ok\n")
        ready = bool(self.workers)
        if request.path == "/ready":
            return connection.respond(HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                                      "ok\n" if ready else "no worker\n")
        capacities = [w.capacity for w in self.workers.values()]
        if not ready or (None not in capacities and self.connected_robots >= sum(capacities)):
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "pool full\n" if ready else "no worker\n")
        return None

    async def run(self, host: str, port: int) -> None:
        """Serve robots on ``host:port``, discovering workers every second, until SIGTERM or SIGINT."""
        async def discover_forever():
            while True:
                await self.discover_workers()
                await asyncio.sleep(1.0)

        stop = asyncio.Event()
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(signal_number, stop.set)
        discovery = asyncio.create_task(discover_forever())
        async with websockets.serve(self.serve_robot, host, port, max_size=None, compression=None, ping_interval=None,
                                    process_request=self.process_request):
            print(f"XINFER ROUTER {host}:{port}", flush=True)
            await stop.wait()
        discovery.cancel()


def _fetch_info(address: str) -> dict:
    with urllib.request.urlopen(f"http://{address}/info", timeout=2) as response:
        return json.loads(response.read())


async def _fetch_metadata_frame(address: str) -> bytes:
    async with websockets.connect(f"ws://{address}", max_size=None, compression=None, open_timeout=2) as connection:
        return await connection.recv()


async def _connect_worker(address: str):
    """A connection to the worker at ``address`` with its metadata frame read. Pings find a worker that vanished
    without closing the connection (a machine that lost power)."""
    connection = await websockets.connect(f"ws://{address}", max_size=None, compression=None, ping_interval=1,
                                          ping_timeout=2, close_timeout=1, open_timeout=2)
    await connection.recv()
    return connection


def _top_level_field(message: bytes, key: str):
    """The top-level ``key`` of a msgpack map, or None, without decoding the other values."""
    unpacker = msgpack.Unpacker(raw=False)
    unpacker.feed(message)
    try:
        for _ in range(unpacker.read_map_header()):
            if unpacker.unpack() == key:
                return unpacker.unpack()
            unpacker.skip()
    except (ValueError, msgpack.UnpackException):
        pass
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Route robot connections to the single-GPU workers of one pool.")
    p.add_argument("--workers", nargs="+", required=True,
                   help="workers: host:port, or dns:<name>:<port> for every address of a name (re-resolved every "
                        "second)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=33985)
    p.add_argument("--retry-deadline-ms", type=float, default=30.0,
                   help="a failed request is sent to the other workers in turn while younger than this, else its "
                        "failure is answered (default 30)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(Router(args.workers, args.retry_deadline_ms / 1000).run(args.host, args.port))


if __name__ == "__main__":
    main()
