"""Engine core: admission, batching, chaining and lifecycle (fake model, CPU and optional CUDA)."""

import contextlib
import math
import queue
import threading
import time
from concurrent.futures import wait
from types import SimpleNamespace

import pytest
import torch

from xinfer.engine import AdmitError, Engine, EngineConfig, core
from xinfer.models.interface import (BucketSpec, Capabilities, DeviceImage, HandoffSpec, OutputSpec, PreparedRequest,
                                     Stage, StageGraph, TensorSpec, VLAModel)
from xinfer.observability.stats import RequestStats
from xinfer.platforms import resolve_platform

CPU = resolve_platform("cpu")


class Pipe(VLAModel):
    """a: h = 2x;  b (batchable): y = h + bias, with h's batch axis 1."""

    stage_graph = StageGraph(Stage("a", "first"), Stage("b", "second", depends_on="a"))
    handoffs = (HandoffSpec("a", "b", {"h": TensorSpec(batch_axis=1)}),)

    def __init__(self, gate=None):
        self.gate, self.batch_sizes = gate, []

    def capabilities(self):
        return Capabilities(batch_sizes={"a": (1,), "b": (1, 2, 4)}, num_steps=0)

    def stage_outputs(self, stage, batch):
        n = batch.inputs["x"].shape[-1]
        return {"h": OutputSpec((1, batch.batch_size, n), torch.float32)} if stage == "a" else \
            {"y": OutputSpec((batch.batch_size, n), torch.float32)}

    def first(self, batch):
        if self.gate is not None:
            self.gate.wait()
        batch.out["h"].copy_((batch.inputs["x"] * 2)[None])

    def second(self, batch):
        self.batch_sizes.append(batch.batch_size)
        batch.out["y"].copy_(batch.inputs["h"][0] + batch.inputs["bias"])


def _req(i, n=3):
    return PreparedRequest({"x": torch.full((1, n), float(i)), "bias": torch.full((1, n), 0.5)})


def _expect(i, n=3):
    return torch.full((1, n), 2.0 * i + 0.5)


@pytest.mark.parametrize("priorities,message", [
    ({"missing": -1}, "unknown stage"),
    ({"a": -1}, "does not support nonzero stream priorities"),
    ({"a": 0.5}, "must be an integer"),
])
def test_direct_engine_rejects_invalid_stream_priorities(priorities, message):
    with pytest.raises(ValueError, match=message):
        Engine(Pipe(), CPU, EngineConfig(stream_priorities=priorities), start=False)


def test_shutdown_after_initialization_still_drains_queued_requests():
    engine = Engine(Pipe(), CPU, start=False)
    initialized, proceed = threading.Event(), threading.Event()
    queued = engine.submit(_req(1))
    engine.start()
    shutdown = threading.Thread(target=engine.shutdown)
    shutdown.start()
    try:
        with engine._wake:
            assert engine._wake.wait_for(lambda: engine._stopping, timeout=10)
    finally:
        proceed.set()
        shutdown.join(timeout=10)
    assert not shutdown.is_alive()
    assert torch.equal(queued.result(timeout=1)["y"], _expect(1))
    assert engine.slots.in_use() == 0


def test_stage_dispatches_run_in_spans_that_are_free_when_annotation_is_off(monkeypatch):
    from xinfer.observability import annotate

    calls, recorded = [], []

    def spy(name, **args):  # the engine's span, recording what it was asked for and what it returned
        context = annotate.span(name, **args)
        calls.append((name, args, context))
        return context

    def sink(name, args):
        recorded.append((name, dict(args)))
        return contextlib.nullcontext()

    monkeypatch.setattr(core, "span", spy)
    engine = Engine(Pipe(), CPU)
    try:
        engine.run(_req(1))
        annotate.configure(sink)
        engine.run(_req(2))
    finally:
        annotate.configure()
        engine.shutdown()
    def spans(request_id):
        return [("admit", {"request": request_id, "robot": None}),
                ("stage::a", {"batch_size": 1, "lane": 0, "requests": str(request_id)}), ("wait", {}),
                ("stage::b", {"batch_size": 1, "lane": 0, "requests": str(request_id)}), ("wait", {})]

    assert [(name, args) for name, args, _ in calls] == spans(0) + spans(1) and recorded == spans(1)
    assert all(context is annotate.span("off") for _, _, context in calls[:5])  # off: the shared no-op context


def test_concurrent_submissions_get_their_own_results():
    engine = Engine(Pipe(), CPU, EngineConfig(max_running_requests=4))
    futures = []

    def client(base):
        for i in range(base, base + 10):
            futures.append((i, engine.submit(_req(i))))

    threads = [threading.Thread(target=client, args=(b,)) for b in (0, 100, 200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, f in futures:
        assert torch.equal(f.result(timeout=10)["y"], _expect(i))
    assert engine.slots.in_use() == 0
    engine.shutdown()


def test_zero_wait_batching_splits_rows_back():
    model = Pipe()
    engine = Engine(model, CPU, EngineConfig(max_running_requests=8, max_batch={"b": 4}), start=False)
    futures = [engine.submit(_req(i)) for i in range(8)]  # queued before the engine thread starts
    engine.start()
    for i, f in enumerate(futures):
        assert torch.equal(f.result(timeout=10)["y"], _expect(i))
    # The first dispatch of a stage runs alone (it learns which inputs the stage
    # reads); then the largest allowed sizes, never padding rows.
    assert model.batch_sizes == [1, 4, 2, 1]
    engine.shutdown()


def test_incompatible_requests_are_not_batched_together():
    model = Pipe()
    engine = Engine(model, CPU, EngineConfig(max_running_requests=8, max_batch={"b": 4}), start=False)
    sizes = [5, 3, 5, 3, 5]
    futures = [engine.submit(_req(i, n=n)) for i, n in enumerate(sizes)]
    engine.start()
    for i, f in enumerate(futures):
        assert torch.equal(f.result(timeout=10)["y"], _expect(i, n=sizes[i]))
    assert model.batch_sizes == [1, 2, 2]  # n=5 alone (learning), then the two n=5 and the two n=3
    engine.shutdown()


def test_admission_limits_and_slot_backpressure():
    gate = threading.Event()
    config = EngineConfig(max_running_requests=1, max_queued_requests=2, per_robot_inflight=1)
    engine = Engine(Pipe(gate=gate), CPU, config)
    first = engine.submit(_req(1), robot="r1")
    while engine.slots.in_use() == 0:  # wait until the first request holds the only slot
        time.sleep(0.001)
    with pytest.raises(AdmitError, match="per_robot_limit"):
        engine.submit(_req(2), robot="r1").result(timeout=5)
    waiting = [engine.submit(_req(i), robot=f"r{i}") for i in (3, 4)]  # wait for the single slot
    with pytest.raises(AdmitError, match="queue_full"):
        engine.submit(_req(5), robot="r5").result(timeout=5)
    gate.set()
    assert torch.equal(first.result(timeout=10)["y"], _expect(1))
    for i, f in zip((3, 4), waiting):
        assert torch.equal(f.result(timeout=10)["y"], _expect(i))
    assert torch.equal(engine.submit(_req(6), robot="r1").result(timeout=10)["y"], _expect(6))
    engine.shutdown()


def test_cancelled_requests_are_dropped_and_release_their_slot():
    engine = Engine(Pipe(), CPU, EngineConfig(max_running_requests=2), start=False)
    doomed = engine.submit(_req(1), robot="r1")
    kept = engine.submit(_req(2), robot="r2")
    assert doomed.cancel()
    engine.start()
    assert torch.equal(kept.result(timeout=10)["y"], _expect(2))
    engine.shutdown()
    assert engine.slots.in_use() == 0
    assert engine.submit(_req(3), robot="r1").exception(timeout=5).code == "stopped"


def test_stage_errors_fail_only_that_request():
    class Flaky(Pipe):
        def second(self, batch):
            if float(batch.inputs["h"][0, 0, 0]) == 2.0:
                raise RuntimeError("boom")
            super().second(batch)

    engine = Engine(Flaky(), CPU, EngineConfig(max_running_requests=4))
    bad, good = engine.submit(_req(1)), engine.submit(_req(2))
    with pytest.raises(RuntimeError, match="boom"):
        bad.result(timeout=10)
    assert torch.equal(good.result(timeout=10)["y"], _expect(2))
    assert engine.slots.in_use() == 0
    engine.shutdown()


def test_a_failed_request_releases_its_robot_before_its_future_fails():
    class Flaky(Pipe):
        def second(self, batch):
            raise RuntimeError("boom")

    gate = threading.Event()
    engine = Engine(Flaky(gate=gate), CPU, EngineConfig(max_running_requests=2))
    retries, retried = [], threading.Event()

    def retry(future):
        retries.append(engine.submit(_req(2), robot="r1"))
        retried.set()

    first = engine.submit(_req(1), robot="r1")
    first.add_done_callback(retry)
    gate.set()  # the stage fails only after the callback is attached
    with pytest.raises(RuntimeError, match="boom"):
        first.result(timeout=10)
    assert retried.wait(timeout=10)
    with pytest.raises(RuntimeError, match="boom"):  # admitted, not refused with per_robot_limit
        retries[0].result(timeout=10)
    engine.shutdown()


def test_a_failed_upload_fails_only_that_request(monkeypatch):
    upload, failed = core._to_device, []

    def fail_once(value, device):
        if not failed:
            failed.append(True)
            raise RuntimeError("upload failed")
        return upload(value, device)

    monkeypatch.setattr(core, "_to_device", fail_once)
    engine = Engine(Pipe(), CPU, EngineConfig(max_running_requests=1))
    with pytest.raises(RuntimeError, match="upload failed"):
        engine.submit(_req(1), robot="r1").result(timeout=10)
    assert engine.slots.in_use() == 0
    assert torch.equal(engine.submit(_req(2), robot="r1").result(timeout=10)["y"], _expect(2))
    engine.shutdown()


def test_a_failed_batch_key_fails_only_the_queue_head(monkeypatch):
    gate = threading.Event()
    engine = Engine(Pipe(gate=gate), CPU, EngineConfig(max_running_requests=4, max_batch={"b": 4}))
    gate.set()
    engine.run(_req(0))  # stage b learns which inputs it reads, so it can batch
    gate.clear()
    batch_key, calls = Engine._batch_key, []

    def fail_first(self, stage, view):
        calls.append(stage)
        if len(calls) == 1:
            raise RuntimeError("bad key")
        return batch_key(self, stage, view)

    monkeypatch.setattr(Engine, "_batch_key", fail_first)
    futures = {i: engine.submit(_req(i)) for i in (1, 2, 3)}
    gate.set()  # stage a held them, so at least two reach stage b's queue together
    assert not wait(futures.values(), timeout=10).not_done
    failed = [i for i, f in futures.items() if f.exception() is not None]
    assert len(failed) == 1 and "bad key" in str(futures[failed[0]].exception())
    for i, f in futures.items():
        if i not in failed:
            assert torch.equal(f.result()["y"], _expect(i))
    assert engine.slots.in_use() == 0
    engine.shutdown()


class Ragged(VLAModel):
    """a: kv = x (length varies per request); b: y = sum of the row's own (left-padded) kv entries."""

    stage_graph = StageGraph(Stage("a", "first"), Stage("b", "second", depends_on="a"))
    handoffs = (HandoffSpec("a", "b", {"kv": TensorSpec(batch_axis=0, seq_axis=1)}),)

    def __init__(self):
        self.calls = []

    def capabilities(self):
        return Capabilities(batch_sizes={"a": (1,), "b": (1, 2, 4)}, num_steps=0)

    def stage_outputs(self, stage, batch):
        if stage == "a":
            return {"kv": OutputSpec((1, batch.inputs["x"].shape[1]), torch.float32)}
        return {"y": OutputSpec((batch.batch_size, 1), torch.float32)}

    def first(self, batch):
        batch.out["kv"].copy_(batch.inputs["x"])

    def second(self, batch):
        kv, lengths = batch.inputs["kv"], batch.inputs["kv_lengths"]
        self.calls.append((tuple(kv.shape), lengths.tolist(), kv.clone()))
        start = kv.shape[1] - lengths.long()
        valid = torch.arange(kv.shape[1])[None] >= start[:, None]
        batch.out["y"].copy_((kv * valid).sum(1, keepdim=True))


def test_requests_of_different_lengths_batch_left_padded():
    model = Ragged()
    engine = Engine(model, CPU, EngineConfig(max_running_requests=8, max_batch={"b": 4}), start=False)
    lengths = [3, 3, 5, 2, 4]
    futures = [engine.submit(PreparedRequest({"x": torch.arange(1.0, n + 1)[None] * (i + 1)}))
               for i, n in enumerate(lengths)]
    engine.start()
    for i, (f, n) in enumerate(zip(futures, lengths)):
        assert float(f.result(timeout=10)["y"]) == (i + 1) * n * (n + 1) / 2
    shapes = [(c[0], c[1]) for c in model.calls]
    assert shapes == [((1, 3), [3]), ((4, 5), [3, 5, 2, 4])]  # first alone, then one padded B4
    padded = model.calls[1][2]
    assert torch.equal(padded[0], torch.tensor([0.0, 0, 2, 4, 6]))  # left-padded with zeros
    engine.shutdown()


def test_a_failed_completion_fails_its_batch_and_the_completer_goes_on(monkeypatch):
    class FailingEvent:  # final-stage work whose synchronization raises
        def query(self):
            return True

        def synchronize(self):
            raise RuntimeError("sync failed")

    monkeypatch.setattr("xinfer.engine.executor.Lane.record", lambda self: FailingEvent())
    engine = Engine(Pipe(), CPU, EngineConfig(max_running_requests=2))
    for i in (1, 2):
        with pytest.raises(RuntimeError, match="sync failed"):
            engine.submit(_req(i)).result(timeout=10)
    assert engine.health() is None  # both threads still run
    assert engine.slots.in_use() == 0
    engine.shutdown()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_health_reports_a_stopped_thread():
    class Exits(Pipe):
        def first(self, batch):
            raise SystemExit  # not caught by the engine loop: the thread ends

    engine = Engine(Exits(), CPU, EngineConfig())
    assert engine.health() is None
    engine.submit(_req(1))
    deadline = time.time() + 10
    while engine.health() is None and time.time() < deadline:
        time.sleep(0.01)
    assert engine.health() == "xinfer-engine thread stopped"


class Variants(Pipe):
    """Pipe whose warmup requests come in three widths."""

    def warmup_requests(self, prepared):
        return [PreparedRequest({"x": torch.zeros(1, n), "bias": torch.zeros(1, n)}) for n in (8, 5, 3)]


def test_warmup_runs_every_variant_the_model_gives():
    model = Variants()
    engine = Engine(model, CPU)
    engine.warmup([_req(1), _req(2)])
    engine.shutdown()
    assert engine.stats["dispatch_a_b1"] == 6 and model.batch_sizes == [1] * 6


class BucketedPipe(Pipe):
    def shape_buckets(self):
        return {"x": BucketSpec(3, max_length=8)}

    def pad(self, prepared, dimension, width):
        assert dimension == "x"
        return _req(1, width)


def test_engine_expands_declared_buckets():
    engine = Engine(BucketedPipe(), CPU)
    try:
        request = _req(1)
        request.meta["x_len_range"] = (3, 8)
        result = engine.warmup([request])
        assert torch.equal(result[0]["y"], _expect(1, 8))  # largest bucket runs first
        assert engine.stats["dispatch_a_b1"] == 3  # widths 8, 6, 3
    finally:
        engine.shutdown()


@pytest.fixture
def decoder(monkeypatch):
    """Replaces the engine's ``DeviceImageDecoder`` with a CPU stand-in: an image's data, as float, becomes its view.

    Records the threads it is created on and, per decoded image, the thread and the image's first data value; a
    value in ``hold`` waits for ``release``, one in ``fail`` raises.
    """
    state = SimpleNamespace(created=[], calls=[], hold=set(), fail=set(), release=threading.Event())

    class Decoder:
        def __init__(self, device):
            state.created.append(threading.current_thread().name)

        def __call__(self, inputs):
            out = dict(inputs)
            for name, value in inputs.items():
                if isinstance(value, DeviceImage):
                    i = int(value.data[0])
                    state.calls.append((threading.current_thread().name, i))
                    if i in state.hold:
                        assert state.release.wait(timeout=10)
                    if i in state.fail:
                        raise ValueError("undecodable view")
                    out[name] = value.data.float().reshape(value.sizes[-1])
            return out

    monkeypatch.setattr(core, "DeviceImageDecoder", Decoder)
    return state


def _image_req(i, n=3):
    """``_req(i)`` with ``x`` as a ``DeviceImage``."""
    image = DeviceImage(torch.full((n,), i, dtype=torch.uint8), ((1, n),), (("bilinear", False),))
    return PreparedRequest({"x": image, "bias": torch.full((1, n), 0.5)})


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("device_images", [False, True])
def test_two_consecutive_cuda_requests_complete_and_report_stage_statistics(monkeypatch, device_images):
    """Real CUDA events cover both upload and image completion; no image codec or model assets are needed."""
    class Decoder:
        def __init__(self, device):
            self.device = device

        def __call__(self, inputs):
            return {name: value.data.to(self.device, dtype=torch.float32).reshape(1, -1)
                    if isinstance(value, DeviceImage) else value for name, value in inputs.items()}

        def close(self):
            pass

    if device_images:
        monkeypatch.setattr(core, "DeviceImageDecoder", Decoder)
    platform = resolve_platform("cuda")
    engine = Engine(Pipe(), platform, EngineConfig(max_running_requests=2))
    reported = queue.Queue()
    engine.stats_sink = reported.put
    try:
        # Resolve each request before submitting the next. A dead completer must fail the second call,
        # and missing timing must fail the statistics assertions even if both outputs were returned.
        for value in (1, 2):
            request = _image_req(value) if device_images else _req(value)
            output = engine.submit(request).result(timeout=10)
            torch.testing.assert_close(output["y"], _expect(value), atol=0, rtol=0)
        for _ in range(2):
            stats = reported.get(timeout=10)
            assert isinstance(stats, RequestStats) and set(stats.stages) == {"a", "b"}
            values = [stats.admission_wait, *(t for stage in stats.stages.values()
                                             for t in (stage.wait, stage.execution))]
            assert all(math.isfinite(value) and value >= 0 for value in values)
            assert all(stage.batch_size == 1 for stage in stats.stages.values())
        assert engine.health() is None
    finally:
        errors = []

        def shutdown():
            try:
                with torch.cuda.device(platform.device):
                    engine.shutdown()
            except Exception as exc:  # cleanup failures still fail the test
                errors.append(exc)

        cleanup = threading.Thread(target=shutdown, daemon=True)
        cleanup.start()
        cleanup.join(timeout=10)
        assert not cleanup.is_alive(), "engine shutdown did not complete"
        if errors:
            raise errors[0]


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_unavailable_event_timing_has_no_statistics(error_type):
    queried = []

    class UnavailableEvent:
        def elapsed_time(self, other):
            queried.append(other)
            raise error_type("event timing unavailable")

    engine = Engine(Pipe(), CPU, start=False)
    request = SimpleNamespace(inputs_ready=UnavailableEvent(),
                              timing={"a": (object(), object(), 1), "b": (object(), object(), 1)})
    assert engine._request_stats(request) is None
    assert len(queried) == 1


def test_device_images_are_decoded_on_their_own_thread_while_other_requests_run(decoder):
    """A request without device images, submitted after one whose decoding is held, completes meanwhile; only the
    request with a device image reaches the decoder, created on the device-image thread at its first use."""
    decoder.hold.add(1)
    engine = Engine(Pipe(), CPU, EngineConfig(max_running_requests=4))
    held = engine.submit(_image_req(1))
    plain = engine.submit(_req(2))
    assert torch.equal(plain.result(timeout=10)["y"], _expect(2))
    assert not held.done()
    decoder.release.set()
    assert torch.equal(held.result(timeout=10)["y"], _expect(1))
    assert decoder.created == ["xinfer-device-images"] and decoder.calls == [("xinfer-device-images", 1)]
    engine.shutdown()
    assert not engine._device_thread.is_alive() and engine.slots.in_use() == 0


def test_device_images_are_decoded_in_admission_order_and_a_failure_fails_only_its_request(decoder):
    decoder.fail.add(2)
    engine = Engine(Pipe(), CPU, EngineConfig(max_running_requests=8), start=False)
    futures = [engine.submit(_image_req(i) if i != 3 else _req(i)) for i in range(6)]
    engine.start()
    with pytest.raises(ValueError, match="undecodable view"):
        futures[2].result(timeout=10)
    for i in (0, 1, 3, 4, 5):
        assert torch.equal(futures[i].result(timeout=10)["y"], _expect(i))
    assert [i for _, i in decoder.calls] == [0, 1, 2, 4, 5] and len(decoder.created) == 1
    assert engine.slots.in_use() == 0
    engine.shutdown()


def test_device_images_fail_their_request_without_a_cuda_device():
    engine = Engine(Pipe(), CPU, EngineConfig())
    assert torch.equal(engine.run(_req(1))["y"], _expect(1))
    for _ in range(2):  # every request with device images fails, the others are served
        with pytest.raises(ValueError, match="need a CUDA or NPU device"):
            engine.run(_image_req(2))
        assert torch.equal(engine.run(_req(3))["y"], _expect(3))
    assert engine.slots.in_use() == 0
    engine.shutdown()
