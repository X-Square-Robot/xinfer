"""DVPP session ownership without an NPU or Python ACL."""

from types import SimpleNamespace

import pytest
import torch

from xinfer.platforms import npu as npu_images


@pytest.fixture
def media(monkeypatch):
    state = SimpleNamespace(device=None, calls=[], init_status=0, exit_status=0)

    class device_context:
        """A re-entrant stand-in for ``torch.npu.device``: the decoder caches its context object
        (WT2b) and re-enters it across calls, which the generator ``@contextmanager`` cannot do."""

        def __init__(self, device):
            self.device = device

        def __enter__(self):
            self._previous = state.device
            state.device = self.device

        def __exit__(self, *exc):
            state.device = self._previous

    def sys_init():
        state.calls.append(("init", state.device))
        return state.init_status

    def sys_exit():
        state.calls.append(("exit", state.device))
        return state.exit_status

    acl = SimpleNamespace(himpi=SimpleNamespace(sys_init=sys_init, sys_exit=sys_exit),
                          rt=SimpleNamespace())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=device_context), raising=False)
    monkeypatch.setattr(npu_images, "_aclfast", None)  # the native fast path needs the real ACL
    monkeypatch.setattr(npu_images, "load_npu_media", lambda: acl)
    monkeypatch.setattr(npu_images, "_MEDIA_USERS", {})
    # The wave decoder has no per-image ``_decode`` step anymore (the transactions start
    # together and the frames drain by size): a deviceless test routes every image to the
    # host fallback -- ``_begin`` answering None -- and stubs that fallback to the passthrough
    # the old ``_decode`` patch was, so the lease and close semantics below run unchanged.
    monkeypatch.setattr(npu_images.NpuJpegDecoder, "_begin", lambda self, jpeg: None)
    monkeypatch.setattr(npu_images.NpuJpegDecoder, "_decode_on_host", lambda self, jpeg: jpeg)
    return state


def test_dvpp_session_is_lazy_shared_per_device_and_closed_once(media):
    device, other_device = torch.device("cpu:0"), torch.device("cpu:1")
    a = npu_images.NpuJpegDecoder(device)
    b = npu_images.NpuJpegDecoder(device)
    c = npu_images.NpuJpegDecoder(other_device)
    unused = npu_images.NpuJpegDecoder(device)
    unused.close()
    assert media.calls == []

    jpeg = torch.ones(1, dtype=torch.uint8)
    for _ in range(3):
        assert torch.equal(a([jpeg])[0], jpeg)
    assert media.calls == [("init", device)]

    b([jpeg])
    c([jpeg])
    a.close()
    a.close()
    with pytest.raises(RuntimeError, match="decoder is closed"):
        a([jpeg])
    b([jpeg])
    assert media.calls == [("init", device), ("init", other_device)]

    b.close()
    c.close()
    assert media.calls == [("init", device), ("init", other_device), ("exit", device), ("exit", other_device)]
    assert not npu_images._MEDIA_USERS


def test_failed_dvpp_initialization_or_close_does_not_leave_a_lease(media):
    device = torch.device("cpu:0")
    jpeg = torch.ones(1, dtype=torch.uint8)
    decode = npu_images.NpuJpegDecoder(device)

    media.init_status = 1
    with pytest.raises(RuntimeError, match="sys_init failed"):
        decode([jpeg])
    assert not npu_images._MEDIA_USERS

    media.init_status = 0
    decode([jpeg])
    media.exit_status = 1
    with pytest.raises(RuntimeError, match="sys_exit failed"):
        decode.close()
    assert not npu_images._MEDIA_USERS

    media.exit_status = 0
    another = npu_images.NpuJpegDecoder(device)
    another([jpeg])
    another.close()
    assert media.calls == [("init", device), ("init", device), ("exit", device), ("init", device), ("exit", device)]


def test_failed_buffer_drain_keeps_the_lease_for_a_close_retry(media, monkeypatch):
    device = torch.device("cpu:0")
    decoder = npu_images.NpuJpegDecoder(device)
    decoder([torch.ones(1, dtype=torch.uint8)])
    drains = []

    def drain():
        drains.append(True)
        if len(drains) == 1:
            raise RuntimeError("copy still pending")

    monkeypatch.setattr(decoder, "_flush_deferred", drain)
    with pytest.raises(RuntimeError, match="copy still pending"):
        decoder.close()
    assert decoder._leased and npu_images._MEDIA_USERS[device] == 1
    decoder.close()
    assert len(drains) == 2 and not decoder._leased and not npu_images._MEDIA_USERS


@pytest.mark.parametrize("batched", [False, True])
def test_partially_freed_buffers_are_never_freed_again_on_close_retry(media, monkeypatch, batched):
    device = torch.device("cpu:0")
    decoder = npu_images.NpuJpegDecoder(device)
    decoder([torch.ones(1, dtype=torch.uint8)])
    decoder._deferred = [(11, 22)]
    attempted = []

    def free(pointer):
        attempted.append(pointer)
        return 0 if pointer == 11 else 1

    def free_batch(pointers):
        attempted.extend(pointers)
        return 1  # one succeeded, without identifying which pointer

    decoder._fast = batched
    monkeypatch.setattr(decoder.acl.himpi, "dvpp_free", free, raising=False)
    monkeypatch.setattr(npu_images, "_aclfast", SimpleNamespace(free_batch=free_batch))
    with pytest.raises(RuntimeError, match="failed") as first:
        decoder.close()
    with pytest.raises(RuntimeError, match="failed") as second:
        decoder.close()
    assert first.value is second.value
    assert attempted == [11, 22]
    assert decoder._leased and npu_images._MEDIA_USERS[device] == 1
    assert decoder._deferred == [(11, 22)]


def test_copy_query_failure_can_retry_before_any_pointer_is_freed(media, monkeypatch):
    device = torch.device("cpu:0")
    decoder = npu_images.NpuJpegDecoder(device)
    decoder([torch.ones(1, dtype=torch.uint8)])
    decoder._deferred = [(11, 22)]
    freed = []

    def query():
        raise RuntimeError("copy query failed")

    decoder._copy_event = SimpleNamespace(query=query)
    monkeypatch.setattr(decoder.acl.himpi, "dvpp_free", lambda pointer: freed.append(pointer) or 0, raising=False)
    with pytest.raises(RuntimeError, match="copy query failed"):
        decoder.close()
    assert decoder._free_error is None and not freed
    decoder._copy_event.query = lambda: True
    decoder.close()
    assert freed == [11, 22] and not decoder._deferred and not npu_images._MEDIA_USERS
