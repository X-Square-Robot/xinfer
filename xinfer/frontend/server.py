"""Frontend process: WebSocket + msgpack, model input/output processing, no GPU.

Several frontend processes share one port (SO_REUSEPORT). Each connection
receives the model's metadata frame, then sends requests one at a time. A
request is decoded, turned into model inputs by the model's InputProcessor,
packed into a shared-memory buffer and sent to the engine; the engine's raw
outputs come back over the same pipe and the OutputProcessor builds the
response (docs/design.md §9.2, §15).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any

from xinfer.frontend.protocol import INTERNAL_ERROR, INVALID_REQUEST, OK, TIMEOUT, UNAVAILABLE, packb, unpackb
from xinfer.models.interface import connect


class EngineError(RuntimeError):
    """A request the engine process refused or failed, with the response status it maps to."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def run_frontend(index: int, pipe, options: dict) -> None:
    """Entry point of a spawned frontend process.

    Hides all CUDA devices, uses one CPU thread for torch and the math
    libraries, makes SIGTERM exit cleanly so the shared-memory buffers are
    unlinked, builds the model's CPU processors (no weights) and serves
    WebSocket connections until the process is terminated (SIGTERM: after answering the requests it received).

    Before serving, each ``options["warmup"]`` entry ``(request, engine outputs)`` runs through the
    InputProcessor and OutputProcessor once, so the first robots do not pay first-call costs such as numba
    compilation. Warmup uses processors built only for it: the serving processors are built afterwards and
    carry no state from it (such as an advanced noise generator), so served outputs are unchanged.

    Args:
        index: frontend number; names its shared-memory ring ``<shm_prefix>-f<index>`` and appears in
            ``server_timing``.
        pipe: this end of the pipe to the engine process's bridge thread.
        options: ``model``, ``checkpoint``, ``model_options``, ``host``, ``port``, ``buffers`` (ring
            size), ``buffer_bytes`` (bytes per buffer), ``workers`` (processing threads), ``shm_prefix``,
            ``metrics`` (record ``FrontendMetrics`` and answer ``GET /metrics``) and optionally ``env`` (sent in the
            metadata frame, default ``"x2robot"``), ``info`` (the ``GET /info`` fields the server knows) and
            ``request_timeout`` (seconds the engine may take for a request before it is answered ``timeout``; 0:
            no limit).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # frontends never touch the GPU
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # until the event loop runs; unwind to unlink the buffers
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    import torch

    torch.set_num_threads(1)
    from xinfer.frontend.shm import ShmRing
    from xinfer.models.registry import get_model_class

    model_cls = get_model_class(options["model"])
    if options.get("warmup"):
        warm_prepare, warm_finish, _ = model_cls.build_processors(options["checkpoint"],
                                                                  **options.get("model_options", {}))
        for request, outputs in options["warmup"]:
            warm_finish(outputs, warm_prepare(request))
    prepare, finish, metadata = model_cls.build_processors(options["checkpoint"], **options.get("model_options", {}))
    ring = ShmRing(f"{options['shm_prefix']}-f{index}", options["buffers"], options["buffer_bytes"])
    try:
        asyncio.run(_serve(index, pipe, options, prepare, finish, metadata, ring))
    finally:
        ring.close()


async def _serve(index, pipe, options, prepare, finish, metadata, ring) -> None:
    """Run the frontend's WebSocket server and request loop on the asyncio event loop.

    Pipe protocol with the engine's bridge thread: the frontend sends
    ``("ready", index, None, None)`` once it listens, and
    ``("request", request_id, robot, manifest)`` per request; the engine answers
    ``(request_id, "ok", outputs)`` or ``(request_id, status, error_repr)``. A receiver thread
    resolves the matching futures on the event loop. Input and output
    processing run on ``options["workers"]`` threads: their heavy parts (image
    decode and resize) release the GIL, so requests of different connections
    are processed at the same time. ``GET /healthz`` and ``/ready`` on the same port return 200; with
    ``options["metrics"]``, ``GET /metrics`` returns every process's metrics (``xinfer.observability.metrics``) and
    each answered request is recorded, else it returns 404. On SIGTERM the frontend stops listening, answers
    further messages with ``unavailable``, and once it answered every request it had received (or ``xinfer-serve``
    exited) closes the connections with code 1001 (going away) and returns.

    Args:
        index: frontend number.
        pipe: pipe to the engine's bridge thread.
        options: the frontend options (``run_frontend``).
        prepare: the model's InputProcessor.
        finish: the model's OutputProcessor.
        metadata: wire metadata frame; ``env`` is added and it is sent first on every connection.
        ring: this frontend's shared-memory buffers.
    """
    import websockets

    from xinfer.observability import metrics as server_metrics

    recorded = server_metrics.FrontendMetrics() if options.get("metrics") else None
    loop = asyncio.get_running_loop()
    # SIGTERM drains the frontend (see the end). A handler raising SystemExit is lost when it runs inside a __del__.
    terminated = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, terminated.set)
    pending: dict[int, asyncio.Future] = {}
    answering = 0  # requests received and not answered yet
    request_ids = itertools.count()
    connections = itertools.count()
    send_lock = threading.Lock()
    worker = ThreadPoolExecutor(max_workers=options["workers"], thread_name_prefix=f"xinfer-prepare-{index}")

    def answered(request_id, status, payload):
        future = pending.pop(request_id, None)
        if future is not None:
            _resolve(future, status, payload)

    def engine_gone():
        terminated.set()
        for request_id in list(pending):
            answered(request_id, UNAVAILABLE, "the engine has stopped")

    def receive():
        """Receiver thread: resolve pending futures with the engine's replies until the pipe closes, then drain the
        frontend as on SIGTERM: without the engine process (killed, for example by the OOM killer) it can answer
        nothing, and it must not keep the port."""
        while True:
            try:
                message = pipe.recv()
            except (EOFError, OSError):
                break
            if message is None:
                break
            request_id, status, payload = message
            loop.call_soon_threadsafe(answered, request_id, status, payload)
        loop.call_soon_threadsafe(engine_gone)

    receiver = threading.Thread(target=receive, name=f"xinfer-frontend-rx-{index}", daemon=True)
    receiver.start()
    hello = packb(dict(metadata, env=options.get("env", "x2robot")))
    info = dict(options.get("info", {}), metadata_sha256=hashlib.sha256(hello).hexdigest(),
                connection_state=getattr(prepare, "connect", None) is not None)

    async def handler(websocket):
        """Serve one robot connection: send the metadata frame, then answer requests one at a time.

        Per message: decode, run the InputProcessor, lease a shared-memory
        buffer (waiting while all are leased, which back-pressures this
        frontend), pack the tensors, send the manifest to the engine with
        ``(index, connection number)`` as the robot id, await the raw outputs,
        release the buffer, run the OutputProcessor and send the response with
        ``status`` ``"ok"`` and ``server_timing``: ``infer_ms`` (message received to response built),
        ``prepare_ms`` (decode and InputProcessor), ``engine_ms`` (buffer
        lease, pack and engine round trip), ``output_ms`` (OutputProcessor) and
        ``frontend`` (``index``). The connection's requests go to its own InputProcessor (``connect``), which
        keeps any per-connection state, such as real-time chunking plans. A failure is answered as
        ``{"status": ..., "error": message}`` (``invalid_request`` when decoding or the InputProcessor fails, the
        engine's status for an engine refusal or failure, else ``internal_error``), with the exception's
        ``error_info`` when it has one, without ``server_timing``; the connection stays open.
        """
        connection = (index, next(connections))
        prepare_connection = connect(prepare)
        if recorded is not None:
            recorded.connections.inc()
        try:
            await _answer(websocket, connection, prepare_connection)
        finally:
            if recorded is not None:
                recorded.connections.dec()

    async def _answer(websocket, connection, prepare_connection):
        """Send the metadata frame, then answer the connection's requests until it closes (``handler``)."""
        nonlocal answering
        await websocket.send(hello)
        async for message in websocket:
            if terminated.is_set():
                with contextlib.suppress(websockets.ConnectionClosed):  # it may be closing already
                    await websocket.send(packb({"status": UNAVAILABLE, "error": "the server is shutting down"}))
                continue
            answering += 1
            received_at = time.perf_counter()
            buffer, status = None, INVALID_REQUEST
            try:
                observation = unpackb(message)
                robot_id = observation.get("robot_id")  # else the connection is the robot
                prepared = await loop.run_in_executor(worker, prepare_connection, observation)
                prepared_at = time.perf_counter()
                status = INTERNAL_ERROR
                buffer = await ring.acquire()
                packed = ring.pack(buffer, prepared.tensors)
                if packed is None:
                    raise RuntimeError("prepared request does not fit a shared-memory buffer")
                request_id = next(request_ids)
                future = loop.create_future()
                with send_lock:
                    pipe.send(("request", request_id, connection if robot_id is None else str(robot_id), packed))
                # The receiver schedules replies on this loop. Transfer the lease before yielding to it.
                pending[request_id] = future

                def release(done, index=buffer):
                    ring.release(index)
                    if not done.cancelled():
                        done.exception()  # consume late errors after a timeout or disconnect

                future.add_done_callback(release)
                buffer = None
                try:
                    outputs = await asyncio.wait_for(asyncio.shield(future), options.get("request_timeout") or None)
                except asyncio.TimeoutError:
                    raise EngineError(TIMEOUT, f"not finished within {options['request_timeout']:g} s") from None
                engine_done_at = time.perf_counter()
                response = await loop.run_in_executor(worker, finish, outputs, prepared)
                finished_at = time.perf_counter()
                response["status"] = OK
                response["server_timing"] = dict(
                    infer_ms=(finished_at - received_at) * 1000, prepare_ms=(prepared_at - received_at) * 1000,
                    engine_ms=(engine_done_at - prepared_at) * 1000, output_ms=(finished_at - engine_done_at) * 1000,
                    frontend=index)
                if recorded is not None:
                    recorded.answered(OK, response["server_timing"])
                await websocket.send(packb(response))
            except Exception as exc:  # noqa: BLE001 - the robot gets the error, the connection stays
                if buffer is not None:
                    ring.release(buffer)
                if isinstance(exc, EngineError):
                    status, error = exc.status, str(exc)
                else:
                    error = repr(exc)
                reply = {"status": status, "error": error}
                if getattr(exc, "error_info", None) is not None:  # a protocol error the robot acts on
                    reply.update(error=str(exc), error_info=exc.error_info)
                if recorded is not None:
                    recorded.answered(status)
                await websocket.send(packb(reply))
            finally:
                answering -= 1

    async def process_request(connection, request):
        """Answer ``/healthz``, ``/ready``, ``/info`` (JSON) and ``/metrics``; return None to continue the WebSocket
        handshake."""
        # Plain HTTP health check on the serving port. Frontends start only
        # after the engine finished warmup, so an open port means ready.
        if request.path in ("/healthz", "/ready"):
            return connection.respond(HTTPStatus.OK, "ok\n")
        if request.path == "/info":
            response = connection.respond(HTTPStatus.OK, json.dumps(info))
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json"
            return response
        if request.path.split("?")[0] == "/metrics":
            if recorded is None:
                return connection.respond(HTTPStatus.NOT_FOUND, "metrics are disabled\n")
            body, content_type = await asyncio.to_thread(server_metrics.render)  # reads every process's files
            response = connection.respond(HTTPStatus.OK, body.decode())
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = content_type
            return response
        return None

    server = await websockets.serve(handler, options["host"], options["port"], reuse_port=True,
                                    max_size=None, compression=None, ping_interval=None, close_timeout=1,
                                    process_request=process_request)
    with send_lock:
        pipe.send(("ready", index, None, None))
    await terminated.wait()
    server.close(close_connections=False)  # stop listening; answer the requests already received
    while (answering or pending) and receiver.is_alive():  # the receiver stops when xinfer-serve has exited
        await asyncio.sleep(0.01)
    await asyncio.gather(*(connection.close(1001) for connection in server.connections))  # going away


def _resolve(future: asyncio.Future, status: str, payload: Any) -> None:
    """Complete a pending request's future on the event loop (scheduled by the receiver thread).

    Args:
        future: the request's future; left alone if already done.
        status: ``"ok"`` or the response status of the engine's refusal or failure.
        payload: the raw outputs (name to ``np.ndarray``), or the engine's error text, raised as
            ``EngineError``.
    """
    if future.done():
        return
    if status == OK:
        future.set_result(payload)
    else:
        future.set_exception(EngineError(status, payload))
