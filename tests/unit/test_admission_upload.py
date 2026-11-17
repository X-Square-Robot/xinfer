"""Pinned upload sources remain private until device completion is known."""

from types import SimpleNamespace

import pytest
import torch

from xinfer.engine import admission_upload as upload


@pytest.fixture
def batcher(monkeypatch):
    def slot(size):
        host = torch.empty(size, dtype=torch.uint8)
        return SimpleNamespace(host=host, host_np=host.numpy())

    monkeypatch.setattr(upload, "_RingSlot", slot)
    return upload.UploadBatcher("cpu", slots=1, slab_bytes=128)


def test_packed_values_and_old_device_views_survive_host_slot_reuse(batcher):
    first = {"x": torch.arange(6, dtype=torch.float32).reshape(2, 3), "mask": torch.tensor([True, False])}
    views, slot = batcher.upload(first, list(first))
    assert all(torch.equal(views[name], value) for name, value in first.items())
    pending = SimpleNamespace(query=lambda: False)
    batcher.release(slot, after=pending)
    assert batcher.upload(first, list(first)) is None
    batcher.poll()
    assert batcher.upload(first, list(first)) is None
    pending.query = lambda: True
    batcher.poll()
    second = {"x": first["x"] + 100, "mask": ~first["mask"]}
    newer, reused = batcher.upload(second, list(second))
    assert reused is slot
    assert all(torch.equal(newer[name], value) for name, value in second.items())
    assert all(torch.equal(views[name], value) for name, value in first.items())


@pytest.mark.parametrize("query_fails_at_release", [False, True])
def test_a_failed_event_query_never_recycles_or_frees_source_bytes(batcher, query_fails_at_release):
    values = {"x": torch.arange(4)}
    _, slot = batcher.upload(values, ["x"])

    def failed():
        raise RuntimeError("device error")

    event = SimpleNamespace(query=failed if query_fails_at_release else lambda: False)
    batcher.release(slot, after=event)
    event.query = failed
    batcher.poll()
    assert slot in batcher._retired
    assert not batcher._free and not batcher._releasing
    assert batcher.upload(values, ["x"]) is None


def test_view_failure_occurs_before_upload_and_releases_the_unused_slot(batcher, monkeypatch):
    copies = []

    class InvalidView:
        def narrow(self, *args):
            raise RuntimeError("cannot construct dtype view")

        def copy_(self, *args, **kwargs):
            copies.append(True)

    original_empty = torch.empty

    def empty(*args, **kwargs):
        return InvalidView() if "device" in kwargs else original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    assert batcher.upload({"x": torch.arange(4)}, ["x"]) is None
    assert not copies and len(batcher._free) == 1 and not batcher._retired


def test_enqueue_failure_propagates_and_keeps_its_source_private(batcher, monkeypatch):
    def failed_copy(self, other, **kwargs):
        raise RuntimeError("enqueue failed")

    monkeypatch.setattr(torch.Tensor, "copy_", failed_copy)
    with pytest.raises(RuntimeError, match="enqueue failed"):
        batcher.upload({"x": torch.arange(4)}, ["x"])
    assert not batcher._free and len(batcher._retired) == 1
    assert batcher.upload({"x": torch.arange(4)}, ["x"]) is None


def test_failed_admission_without_a_recordable_event_retires_its_slab(batcher):
    from xinfer.engine.core import Engine

    _, slot = batcher.upload({"x": torch.arange(4)}, ["x"])

    def broken_event():
        raise RuntimeError("cannot record completion")

    engine = Engine.__new__(Engine)
    engine._device_error = None
    engine._device_module = SimpleNamespace(Event=broken_event)
    engine._upload_batcher = batcher
    request = SimpleNamespace(upload_slab=slot)
    assert not engine._upload_fence(object(), request).query()
    assert isinstance(engine._device_error, RuntimeError)
    assert request.upload_slab is None and slot in batcher._retired
    assert not batcher._free
