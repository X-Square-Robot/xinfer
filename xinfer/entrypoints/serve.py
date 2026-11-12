"""``python -m xinfer.entrypoints.serve``: one engine process per GPU plus CPU frontends.

The engine process loads the model and runs the engine and completion
threads. It spawns frontend processes that share the WebSocket
port; each sends prepared requests through shared memory and a pipe, and a
bridge thread per frontend submits them to the engine and returns the raw
outputs.

Runtime settings are resolved by xinfer.config (docs/configuration.md):
explicit flags win, then the tuned table for (model, architecture,
quantization), then fixed defaults and the model's and platform's
declarations; none depends on how many robots connect. The resolved values and
their sources are logged at startup and written with ``--resolved-config``.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import uuid

logger = logging.getLogger("xinfer.serve")


def _op_backends(text: str | None) -> dict[str, str]:
    """Parse ``--op-backends`` such as ``"varlen_attention=flash_attn"``.

    Args:
        text: comma-separated ``operator=pin`` items (a backend, library or ``backend:library``), or ``None``/empty.

    Returns:
        Operator name to pin; ``build_table`` checks both.
    """
    if not text:
        return {}
    return {name.strip(): backend.strip() for name, _, backend in (item.partition("=") for item in text.split(","))}


def _per_stage(text: str | None) -> dict[str, int]:
    """Parse a per-stage flag such as ``"vision=2,prefill=2,action=2"``.

    Args:
        text: comma-separated ``stage=int`` items, or ``None``/empty.

    Returns:
        Stage name to value; empty when ``text`` is empty. Stage names are checked later by ``resolve``.

    Raises:
        ValueError: a value is not an integer.
    """
    if not text:
        return {}
    out = {}
    for item in text.split(","):
        name, _, value = item.partition("=")
        out[name.strip()] = int(value)
    return out


def parse_args(argv=None):
    """Parse the ``xinfer serve`` command line.

    Options: model and checkpoint (``--model``, ``--checkpoint``,
    ``--model-options`` JSON), network (``--host``, ``--port``), device,
    explicit engine and frontend settings that override the defaults, ``--enforce-eager``,
    tuned-table selection, reporting (``--resolved-config``, ``--dry-run``),
    ``--reference-ops`` and ``--warmup`` request files. Unset optional
    settings stay ``None`` so that ``resolve`` fills them in.

    Args:
        argv: arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        The ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(prog="xinfer serve", description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-options", default="{}",
                   help="JSON options for the model and its processors, e.g. image_preprocess_device (default cpu)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=33985)
    p.add_argument("--device", default=None)
    p.add_argument("--frontends", type=int, default=None, help="frontend processes (default 2)")
    p.add_argument("--disable-metrics", action="store_true",
                   help="do not record Prometheus metrics; GET /metrics answers 404 (docs/metrics.md)")
    p.add_argument("--frontend-workers", type=int, default=None, help="processing threads per frontend (default 4)")
    p.add_argument("--shm-buffers", type=int, default=None, help="shared-memory buffers per frontend")
    p.add_argument("--shm-buffer-mib", type=int, default=32)
    p.add_argument("--lanes", default=None, help="per stage, e.g. vision=2,prefill=2,action=2")
    p.add_argument("--stream-priorities", default=None,
                   help="CUDA stream priority per stage, e.g. action=0; lower is higher; "
                        "default -1 for the final stage on CUDA, else 0")
    p.add_argument("--max-batch", default=None, help="per stage, e.g. action=4")
    p.add_argument("--max-running-requests", type=int, default=None,
                   help="requests admitted and running in the engine (default 32)")
    p.add_argument("--max-queued-requests", type=int, default=None)
    p.add_argument("--per-robot-inflight", type=int, default=None)
    p.add_argument("--enforce-eager", action="store_true",
                   help="run every stage eagerly, without CUDA graphs (default: CUDA graphs where supported)")
    p.add_argument("--tuned", default="auto", help="'auto' (bundled table), 'none', or a tuned-table JSON path")
    p.add_argument("--resolved-config", default=None, help="write the resolved settings and sources to this JSON")
    p.add_argument("--dry-run", action="store_true", help="load the model, print the resolved configuration, exit")
    p.add_argument("--profile-dir", default=None,
                   help="record the engine's spans and write each request's engine segments here at shutdown "
                        "(xinfer-profile load --serve; docs/profiling.md)")
    p.add_argument("--reference-ops", action="store_true")
    p.add_argument("--op-backends", default=None,
                   help="pin implementations per operator by backend, library or both, e.g. "
                        "varlen_attention=flash_attn or rms_norm=cuda:xcompute; pins win over --reference-ops")
    p.add_argument("--warmup", nargs="*", default=(),
                   help="request files run before serving (JSON or msgpack, each one request or a list)")
    return p.parse_args(argv)



def _start_metrics(args, model, platform, quantization):
    """Record Prometheus metrics from this process and the frontends started after it (docs/metrics.md).

    The processes write into ``PROMETHEUS_MULTIPROC_DIR`` when it is set, else into a new temporary directory
    removed at exit.

    Returns:
        ``(EngineMetrics, directory, temporary)``, or ``(None, None, False)`` without ``prometheus_client``.
    """
    import tempfile

    import xinfer
    from xinfer.observability import metrics

    if not metrics.available():
        logger.warning("prometheus_client is not installed, so GET /metrics is disabled (pip install -e '.[serve]')")
        return None, None, False
    given = os.environ.get(metrics.ENV)
    directory = given or tempfile.mkdtemp(prefix="xinfer-metrics-")
    metrics.enable(directory)
    logger.info("recording Prometheus metrics in %s; GET /metrics on the serving port", directory)
    info = dict(model=args.model, quantization=quantization if isinstance(quantization, str) else "custom",
                numerics=str(getattr(model, "numerics", "")), platform=platform.name, version=xinfer.__version__)
    return metrics.EngineMetrics(info), directory, not given


def _stop_metrics(directory, temporary: bool, pids) -> None:
    """Remove the temporary metrics directory, or drop the live gauges of ``pids`` from a given one."""
    import shutil

    from xinfer.observability import metrics

    if temporary:
        shutil.rmtree(directory, ignore_errors=True)
        return
    for pid in pids:
        metrics.mark_dead(pid)

def main(argv=None) -> None:
    """Run the server: the engine process (main thread) and its frontend processes.

    Lifecycle:

    1. Parse the flags and configure logging; set ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and
       ``NUMBA_NUM_THREADS`` to 1 unless already set, and use one torch CPU thread.
    2. Resolve the platform and the ``image_preprocess_device`` model option (default ``"cpu"``,
       ``apply_image_preprocess_device``), load operator plugins, select the operators and load the model with
       ``from_checkpoint``.
    3. Resolve the settings (flags as ``ExplicitSettings``, ``--tuned``), log
       them and write ``--resolved-config`` when given; with ``--dry-run`` print them and return.
    4. Start the ``Engine``. With ``--warmup``, prepare the requests of every request file (``read_requests``) in
       this process and warm up with them (``Engine.warmup``: their CUDA graphs on every lane and at every allowed
       batch size), then finish the warmup. Each frontend later runs these requests through the model's InputProcessor
       and, with the engine's outputs for them, its OutputProcessor before it reports ready (``run_frontend``).
    5. Unless ``--disable-metrics``, start recording Prometheus metrics (``_start_metrics``): the engine's requests
       from now on, after the warmup, through its ``stats_sink``, and the frontends'. Spawn the frontend processes
       (``run_frontend``), each with a pipe and its own shared-memory ring, and start one bridge thread per pipe;
       when every frontend reports that it listens, print ``XINFER READY <host>:<port> frontends=<N>``.
    6. Wait for SIGINT/SIGTERM, the exit of any frontend, or an engine that can no longer serve
       (``Engine.health``, checked every 0.5 s, when the engine's sampled metrics are updated too); then terminate
       and join the frontends, shut the engine down, unregister and unmap the shared memory, remove the temporary
       metrics directory, and leave with ``os._exit`` (status 0 after a signal, 1 when a
       frontend died or the engine stopped), skipping interpreter finalization so CUDA objects are not
       destroyed after the CUDA context.

    Args:
        argv: command-line arguments; ``None`` reads ``sys.argv``.

    Raises:
        ConfigError: an explicit setting conflicts with the platform or the model.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    import numpy as np
    import torch

    from xinfer import ops
    from xinfer.config import ExplicitSettings, apply_image_preprocess_device, load_tuned, resolve
    from xinfer.engine import AdmitError
    from xinfer.frontend.protocol import ADMIT_STATUS, INTERNAL_ERROR, OK, read_requests
    from xinfer.frontend.server import run_frontend
    from xinfer.frontend.shm import ShmReader
    from xinfer.models.interface import PreparedRequest
    from xinfer.models.registry import get_model_class
    from xinfer.platforms import resolve_platform

    torch.set_num_threads(1)
    platform = resolve_platform(args.device)
    if platform.capabilities.execution_model != "cuda-lanes":
        raise SystemExit(f"xinfer serve currently requires execution_model='cuda-lanes'; "
                         f"{platform.name} ({platform.device}) advertises "
                         f"{platform.capabilities.execution_model!r}. Use the in-process Policy or "
                         "that vendor's dedicated entry point.")
    # The views are decoded and resized in the frontends unless the model options say "gpu" (bitwise as the
    # reference services).
    model_options = apply_image_preprocess_device(json.loads(args.model_options), platform, default="cpu")
    ops.registry.load_plugins()
    table = ops.registry.build_table(platform, backends=_op_backends(args.op_backends),
                                     reference_only=args.reference_ops)
    model_cls = get_model_class(args.model)
    started = time.perf_counter()
    with ops.use_ops(table):
        model = model_cls.from_checkpoint(args.checkpoint, device=platform.device, ops=table, **model_options)
    explicit = ExplicitSettings(
        lanes=_per_stage(args.lanes), max_batch=_per_stage(args.max_batch),
        stream_priorities=_per_stage(args.stream_priorities),
        max_running_requests=args.max_running_requests, max_queued_requests=args.max_queued_requests,
        per_robot_inflight=args.per_robot_inflight, cudagraphs=False if args.enforce_eager else None,
        frontends=args.frontends, frontend_workers=args.frontend_workers, shm_buffers=args.shm_buffers)
    quantization = model_options.get("quantization", "bf16")
    tuned = {"auto": True, "none": None}[args.tuned] if args.tuned in ("auto", "none") else load_tuned(args.tuned)
    resolved = resolve(model, platform, model_name=args.model,
                       quantization=quantization if isinstance(quantization, str) else None,
                       explicit=explicit, tuned=tuned)
    logger.info("model loaded in %.1fs on %s; ops %s", time.perf_counter() - started, platform.name,
                {r["op"]: f"{r['backend']}:{r['library']}" if r["library"] else r["backend"] for r in table.report()})
    if platform.capabilities.op_selection == "table":
        ops.warn_slow_fallbacks(args.model, table, model.capabilities().stage_ops, platform.device)
    logger.info("resolved configuration:\n%s", resolved.report())
    if args.resolved_config:
        resolved.write(args.resolved_config, model=args.model, platform=platform.name, ops=table.report(),
                       tuned=None if resolved.tuned is None else dict(resolved.tuned.measured))
    if args.dry_run:
        print(resolved.report(), flush=True)
        return
    from xinfer.entrypoints._factory import create_engine

    engine = create_engine(platform, model, resolved.engine_config())
    warmup = []
    if args.warmup:
        prepare, _, _ = model_cls.build_processors(args.checkpoint, **model_options)
        raw = [request for path in args.warmup for request in read_requests(path)]
        outputs = engine.warmup([prepare(request) for request in raw])
        warmup = [(request, {k: np.asarray(v) for k, v in out.items()}) for request, out in zip(raw, outputs)]
        logger.info("warmup done: %s", engine.cudagraph_stats())
    engine_metrics, metrics_dir, metrics_temporary = None, None, False
    if not args.disable_metrics:
        engine_metrics, metrics_dir, metrics_temporary = _start_metrics(args, model, platform, quantization)
        if engine_metrics is not None:  # after the warmup: its requests, dispatches and captures are not counted
            engine.stats_sink = engine_metrics.record
            engine_metrics.sample(engine)
    recorder = None
    if args.profile_dir:
        from xinfer.observability import annotate, load_profile

        recorder = load_profile.SpanRecorder(platform.capabilities.cuda_interop)
        annotate.configure(recorder)

    mp_context = mp.get_context("spawn")
    options = dict(model=args.model, checkpoint=args.checkpoint, model_options=model_options, host=args.host,
                   port=args.port, buffers=resolved.value("shm_buffers"), buffer_bytes=args.shm_buffer_mib << 20,
                   workers=resolved.value("frontend_workers"),
                   shm_prefix=f"xinfer-{uuid.uuid4().hex[:8]}")
    options["warmup"] = warmup  # the frontends run these requests through their processors before serving
    options["metrics"] = engine_metrics is not None
    reader = ShmReader(register_cuda=platform.capabilities.cuda_interop)
    reader_lock = threading.Lock()
    ready = threading.Semaphore(0)
    processes, pipes = [], []
    for index in range(resolved.value("frontends")):
        parent, child = mp_context.Pipe()
        process = mp_context.Process(target=run_frontend, args=(index, child, options), name=f"xinfer-frontend-{index}",
                           daemon=True)
        process.start()
        processes.append(process)
        pipes.append(parent)

    def bridge(pipe):
        """Serve one frontend's pipe (one ``xinfer-bridge`` thread per frontend).

        Receives ``("ready", index, None, None)``, which counts toward
        startup, or ``("request", request_id, connection, manifest)``: maps the
        shared-memory buffer, submits the tensors with ``connection`` (frontend
        index, connection number) as the robot id, and replies
        ``(request_id, "ok", {name: np.ndarray})`` or ``(request_id, status, repr(exc))``, where ``status`` is
        the admission refusal (``ADMIT_STATUS``) or ``"internal_error"``. Returns when the pipe closes.
        """
        send_lock = threading.Lock()

        def reply(request_id, future):
            """Send a finished request's outputs or error to the frontend.

            Runs on the thread that completed ``future``: the completer, the
            engine thread after a failed dispatch, or the bridge itself when the
            request was refused at submission. A closed pipe is ignored.
            """
            try:
                outputs = {k: np.asarray(v) for k, v in future.result().items()}
                message = (request_id, OK, outputs)
            except AdmitError as exc:
                message = (request_id, ADMIT_STATUS.get(exc.code, INTERNAL_ERROR), repr(exc))
            except Exception as exc:  # noqa: BLE001
                message = (request_id, INTERNAL_ERROR, repr(exc))
            with send_lock:
                try:
                    pipe.send(message)
                except (BrokenPipeError, OSError):
                    pass

        while True:
            try:
                kind, request_id, connection, packed = pipe.recv()
            except (EOFError, OSError):
                return
            if kind == "ready":
                ready.release()
                continue
            try:
                with reader_lock:
                    tensors = reader.unpack(packed)
                future = engine.submit(PreparedRequest(tensors), robot=connection)
            except Exception as exc:  # noqa: BLE001
                with send_lock:
                    pipe.send((request_id, INTERNAL_ERROR, repr(exc)))
                continue
            future.add_done_callback(lambda f, request_id=request_id: reply(request_id, f))

    for pipe in pipes:
        threading.Thread(target=bridge, args=(pipe,), name="xinfer-bridge", daemon=True).start()
    for _ in processes:
        ready.acquire()
    print(f"XINFER READY {args.host}:{args.port} frontends={len(processes)}", flush=True)

    stop = threading.Event()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, lambda *_: stop.set())
    reason = None
    while not stop.is_set() and all(p.is_alive() for p in processes):
        reason = engine.health()
        if engine_metrics is not None:
            engine_metrics.sample(engine)
        if reason is not None:
            logger.error("engine cannot serve (%s); exiting", reason)
            break
        stop.wait(0.5)
    for process in processes:
        process.terminate()
    for process in processes:
        process.join(timeout=10)
    engine.shutdown()
    if recorder is not None:
        logger.info("engine requests written to %s", load_profile.write_engine_requests(recorder, engine.stages,
                                                                                       args.profile_dir))
    with reader_lock:
        reader.close()
    if metrics_dir is not None:
        _stop_metrics(metrics_dir, metrics_temporary, [process.pid for process in processes] + [os.getpid()])
    logger.info("stopped; dispatches %s", dict(engine.stats))
    logging.shutdown()
    sys.stdout.flush()
    # Skip interpreter finalization: CUDA objects (graphs, pools, host
    # registrations) must not be destroyed after the CUDA context.
    os._exit(0 if stop.is_set() and reason is None else 1)


if __name__ == "__main__":
    main()
