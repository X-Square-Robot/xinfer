"""A failed dispatch keeps its lane and slot until its enqueued copy to the host completes (CUDA)."""

import threading

import pytest
import torch

from xinfer.engine import Engine, EngineConfig
from xinfer.models.interface import OutputSpec, PreparedRequest, Stage, StageGraph, VLAModel
from xinfer.platforms import resolve_platform

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _wait_for_release(flag):
        value = tl.atomic_add(flag, 0, sem="acquire")
        while value == 0:
            value = tl.atomic_add(flag, 0, sem="acquire")

    @triton.jit
    def _release(flag):
        tl.atomic_xchg(flag, 1, sem="release")


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


class CopyModel(VLAModel):
    stage_graph = StageGraph(Stage("copy", "copy"))

    def stage_outputs(self, stage, batch):
        return {"y": OutputSpec(tuple(batch.inputs["x"].shape), torch.float32)}

    def copy(self, batch):
        batch.out["y"].copy_(batch.inputs["x"])


def request(value, width=16):
    return PreparedRequest({"x": torch.full((1, width), float(value))})


@pytest.mark.skipif(triton is None, reason="Triton required for the explicit GPU release gate")
def test_partial_d2h_failure_keeps_lane_and_slot_until_copy_completes(monkeypatch):
    release, admission_attempted = threading.Event(), threading.Event()
    engine = Engine(CopyModel(), resolve_platform("cuda"), EngineConfig(max_running_requests=1), start=False)
    lane = engine.lanes["copy"][0]
    device = engine.platform.device
    setup = torch.cuda.current_stream(device)
    releaser = torch.cuda.Stream(device=device)
    flag = torch.ones((), device=device, dtype=torch.int32)

    # Compile before blocking the copy stream, and prime the same output shape
    # so a first-use allocation cannot synchronize behind our explicit gate.
    lane.copy_stream.wait_stream(setup)
    with torch.cuda.stream(lane.copy_stream):
        _wait_for_release[(1,)](flag)
    releaser.wait_stream(setup)
    with torch.cuda.stream(releaser):
        _release[(1,)](flag)
    torch.cuda.synchronize(device)
    engine.start()
    try:
        engine.run(request(0))
    except Exception:
        engine.shutdown()
        raise
    primed = [torch.empty((1, 16), pin_memory=True) for _ in range(4)]
    del primed
    flag.zero_()
    lane.copy_stream.wait_stream(setup)

    original_to_host = engine._to_host
    original_record = lane.record
    original_allocate = engine.slots.allocate
    failed = False
    failure_events = []
    premature_allocations = []
    copy_calls = []
    release_was_requested = []

    def to_host(*args, **kwargs):
        nonlocal failed
        assert torch.cuda.current_stream(device) == lane.copy_stream
        copy_calls.append(True)
        if not failed:
            _wait_for_release[(1,)](flag)
        result = original_to_host(*args, **kwargs)
        if not failed:
            failed = True
            raise RuntimeError("failed after D2H was enqueued")
        return result

    def record():
        event = original_record()
        if failed and not failure_events:
            failure_events.append(event)
        return event

    def allocate():
        try:
            handle = original_allocate()
            if failure_events and not release.is_set():
                premature_allocations.append(handle)
            return handle
        finally:
            if failure_events and not release.is_set():
                admission_attempted.set()

    def release_copy():
        # Timeout is a deadlock escape, not the assertion: a healthy issuer
        # reaches the checks below while both D2H and its failure event wait.
        release_was_requested.append(release.wait(timeout=15))
        with torch.cuda.device(device), torch.cuda.stream(releaser):
            _release[(1,)](flag)
        releaser.synchronize()

    monkeypatch.setattr(engine, "_to_host", to_host)
    monkeypatch.setattr(lane, "record", record)
    monkeypatch.setattr(engine.slots, "allocate", allocate)
    watchdog = threading.Thread(target=release_copy, daemon=True)
    watchdog.start()
    try:
        first = engine.submit(request(1))
        with pytest.raises(RuntimeError, match="failed after D2H was enqueued"):
            first.result(timeout=15)
        second = engine.submit(request(2, width=64))
        assert admission_attempted.wait(timeout=15)
        assert len(failure_events) == 1 and not failure_events[0].query(), "failure event did not wait for D2H"
        assert lane.event is failure_events[0] and not lane.event.query(), "lane was reused before D2H completed"
        assert not premature_allocations, "slot was reused before its failed D2H completed"
        assert engine.slots.in_use() == 1
        assert len(copy_calls) == 1
        assert not second.done()
        release.set()
        assert torch.equal(second.result(timeout=15)["y"], torch.full((1, 64), 2.0))
        assert torch.equal(engine.run(request(3))["y"], torch.full((1, 16), 3.0))
        assert engine.slots.in_use() == 0
    finally:
        release.set()
        watchdog.join(timeout=15)
        engine.shutdown()
    assert not watchdog.is_alive()
    assert release_was_requested == [True], "copy stream needed watchdog release instead of host progress"
