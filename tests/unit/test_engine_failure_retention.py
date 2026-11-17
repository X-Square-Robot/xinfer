"""Unknown device completion fails requests without recycling asynchronous sources."""

from collections import Counter
from concurrent.futures import Future
import contextlib
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from xinfer.engine import Engine
from xinfer.engine import core
from xinfer.engine.admission_upload import UploadBatcher
from xinfer.engine.core import _FAILED_ENGINES, _Request
from xinfer.models.interface import PreparedRequest
from xinfer.platforms import resolve_platform
from test_engine import Pipe, _req


def _request(engine):
    request = _Request(0, None, PreparedRequest({"x": torch.ones(1, 3)}), Future())
    request.inputs = {"x": torch.ones(1, 3)}
    request.handle = engine.slots.allocate()
    request.upload_slab = SimpleNamespace(host=torch.ones(16, dtype=torch.uint8))
    engine._upload_batcher = UploadBatcher("cpu", slots=1)
    engine._upload_batcher._outstanding = 1
    engine._inflight = Counter({None: 1})
    return request


def test_failed_completion_retains_source_and_request_storage_and_rejects_new_work():
    engine = Engine(Pipe(), resolve_platform("cpu"), start=False)
    request = _request(engine)
    slab, inputs = request.upload_slab, request.inputs

    def fail():
        raise RuntimeError("completion status unknown")

    engine._completions.put((SimpleNamespace(synchronize=fail), [request], [torch.ones(1)]))
    engine._completions.put(None)
    engine._completion_loop()
    assert request.future.done()
    with pytest.raises(RuntimeError, match="completion status unknown"):
        request.future.result()
    assert slab not in engine._upload_batcher._free
    assert slab in engine._upload_batcher._retired
    assert any(slot.tensors.get("failed_dispatch", (None, None))[1] is inputs
               for slot in engine.slots._slots)
    with pytest.raises(RuntimeError, match="completion status unknown"):
        engine.submit(PreparedRequest({})).result(timeout=0.1)


def test_eager_warmup_does_not_schedule_image_graph_capture():
    engine = Engine(Pipe(), resolve_platform("cpu"), start=False)
    engine._device_stream = object()

    class CaptureRequested:
        def put(self, item):
            pytest.fail("eager mode scheduled an image graph capture")

    engine._device_requests = CaptureRequested()
    engine._capture_image_graphs()


def test_image_capture_timeout_fails_readiness_without_waiting(monkeypatch):
    engine = Engine(Pipe(), resolve_platform("cpu"), start=False)
    engine.config = core.EngineConfig(cudagraphs=core.CUDAGraphConfig())
    engine._device_stream = object()
    waits = []

    def timed_out(timeout):
        waits.append(timeout)
        return False

    monkeypatch.setattr(core, "_CaptureImageGraphs",
                        lambda: SimpleNamespace(done=SimpleNamespace(wait=timed_out)))
    with pytest.raises(TimeoutError, match="image graph capture did not finish"):
        engine._capture_image_graphs()
    assert waits == [300]
    assert "image graph capture did not finish" in engine.health()
    with pytest.raises(TimeoutError, match="image graph capture did not finish"):
        engine.submit(PreparedRequest({})).result(timeout=0.1)


def test_shutdown_joins_capture_worker_before_synchronizing_or_closing_decoder():
    engine = Engine(Pipe(), resolve_platform("cpu"), start=False)
    order = []
    engine._device_thread = SimpleNamespace(is_alive=lambda: True, join=lambda: order.append("capture joined"))
    engine._device_module = SimpleNamespace(device=lambda device: contextlib.nullcontext(),
                                            synchronize=lambda: order.append("synchronized"))
    engine._decoder = SimpleNamespace(close=lambda: order.append("decoder closed"))
    engine._fail_device(TimeoutError("capture timed out"))
    engine.shutdown()
    assert order == ["capture joined", "synchronized", "decoder closed"]


def test_worker_teardown_failure_is_retried_after_shutdown_synchronizes():
    engine = Engine(Pipe(), resolve_platform("cpu"), start=False)
    order = []

    def close():
        order.append("close")
        if len(order) == 1:
            raise RuntimeError("deferred copy query failed")

    decoder = engine._decoder = SimpleNamespace(close=close)
    engine._device_requests.put(None)
    engine._device_images_loop()
    assert engine._decoder is decoder and isinstance(engine._device_error, RuntimeError)
    engine._device_module = SimpleNamespace(device=lambda device: contextlib.nullcontext(),
                                            synchronize=lambda: order.append("sync"))
    engine.shutdown()
    assert order == ["close", "sync", "close"] and engine._decoder is None


def test_unrecordable_dispatch_fails_queued_work_and_keeps_stage_temporaries():
    engine = Engine(Pipe(), resolve_platform("cpu"), start=False)
    request = _request(engine)
    pending = engine.submit(_req(2))
    temporary = []

    def stage(batch):
        temporary.append(batch.out["h"])
        raise RuntimeError("stage failed after enqueue")

    def record():
        raise RuntimeError("cannot record fence")

    engine._fns["a"] = stage
    lane = SimpleNamespace(stream=object(), copy_stream=None, cudagraphs=None, index=0,
                           use_stream=contextlib.nullcontext,
                           wait=lambda event: None, record=record)
    with pytest.raises(RuntimeError, match="stage failed after enqueue"):
        engine._execute_stage_batch("a", lane, [request])
    assert request.future.done()
    assert not lane.event.query()
    assert not engine._upload_batcher._free
    assert any(slot.tensors.get("failed_stage", (None, None, None))[2].out["h"] is temporary[0]
               for slot in engine.slots._slots if "failed_stage" in slot.tensors)
    engine._step()
    with pytest.raises(RuntimeError, match="cannot record fence"):
        pending.result(timeout=0.1)
    assert not engine._waiting and not any(engine._queues.values())


def test_query_failure_preserves_storage_and_records_device_failure():
    engine = Engine(Pipe(), resolve_platform("cpu"), start=False)
    request = _request(engine)

    def fail():
        raise RuntimeError("query failed")

    engine._fail_request(request, ValueError("dispatch failed"), after=SimpleNamespace(query=fail))
    assert not engine._upload_batcher._free
    assert len(engine._upload_batcher._retired) == 1
    assert engine.health() == "device completion failed: query failed"


def test_failed_shutdown_keeps_buffers_until_a_successful_synchronize():
    engine = Engine(Pipe(), resolve_platform("cpu"), start=False)
    request = _request(engine)

    def fail():
        raise RuntimeError("device still failed")

    engine._device_module = SimpleNamespace(device=lambda device: contextlib.nullcontext(), synchronize=fail)
    with pytest.raises(RuntimeError, match="device still failed"):
        engine.shutdown()
    assert engine in _FAILED_ENGINES and engine.slots.tensors(request.handle) == {}
    engine._device_module.synchronize = lambda: None
    engine.shutdown()
    assert engine not in _FAILED_ENGINES and engine._upload_batcher is None


@pytest.mark.parametrize("fail_copy", [False, True])
def test_d2h_buffers_belong_to_the_caller_before_copy_or_event_failure(monkeypatch, fail_copy):
    engine = Engine(Pipe(), resolve_platform("cpu"), start=False)
    references = []

    class Host:
        def copy_(self, value, **kwargs):
            if fail_copy:
                raise RuntimeError("D2H copy failed")

    def allocate(*args, **kwargs):
        host = Host()
        references.append(weakref.ref(host))
        return host

    def record(stream):
        raise RuntimeError("D2H completion failed")

    monkeypatch.setattr(torch, "empty", allocate)
    engine._row = lambda *args: SimpleNamespace(shape=(1,), dtype=torch.float32)
    engine._device_module = SimpleNamespace(stream=lambda stream: contextlib.nullcontext(),
                                            Event=lambda: SimpleNamespace(record=record))
    lane = SimpleNamespace(copy_stream=SimpleNamespace(wait_event=lambda event: None), record=lambda: object())
    hosts = []
    with pytest.raises(RuntimeError, match="D2H"):
        engine._copy_to_host("b", lane, {"y": SimpleNamespace(record_stream=lambda stream: None)}, 1, hosts=hosts)
    gc.collect()
    assert len(hosts) == len(references) == 1 and hosts[0]["y"] is references[0]()


def test_failed_dispatch_holds_caller_owned_d2h_buffers_until_its_fence_completes(monkeypatch):
    engine = Engine(Pipe(), resolve_platform("cpu"), start=False)
    engine.final_stages = {"a"}
    request = _request(engine)
    references = []

    class Host:
        pass

    def fail(stage, lane, out, batch_size, *, hosts):
        host = Host()
        references.append(weakref.ref(host))
        hosts.append({"h": host})
        raise RuntimeError("copy was enqueued")

    fence = SimpleNamespace(query=lambda: False)
    monkeypatch.setattr(engine, "_copy_to_host", fail)
    lane = SimpleNamespace(stream=object(), copy_stream=None, cudagraphs=None, index=0,
                           use_stream=contextlib.nullcontext, wait=lambda event: None, record=lambda: fence)
    with pytest.raises(RuntimeError, match="copy was enqueued"):
        engine._execute_stage_batch("a", lane, [request])
    gc.collect()
    assert references[0]() is not None
    # Future exceptions retain tracebacks too; remove those references to isolate the slot's fence.
    request.future.exception().__traceback__ = None
    fence.query = lambda: True
    assert engine.slots.in_use() == 0
    gc.collect()
    assert references[0]() is None
