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


def test_abandon_channel_failure_preserves_the_lease_and_failed_channel(media, monkeypatch):
    device = torch.device("cpu:0")
    decoder = npu_images.NpuJpegDecoder(device)
    decoder([torch.ones(1, dtype=torch.uint8)])
    key = decoder._channel_key(16, 16, 1)
    decoder._channels[key], decoder._pending[key] = 42, (1, 1)
    context = dict(channel=42, source=33, output=44, width=16, height=16, size=1)
    freed = []

    def vdec_destroy_chn(channel):
        return 1

    monkeypatch.setattr(decoder.acl.himpi, "vdec_stop_recv_stream", lambda channel: 0, raising=False)
    monkeypatch.setattr(decoder.acl.himpi, "vdec_destroy_chn", vdec_destroy_chn, raising=False)
    monkeypatch.setattr(decoder.acl.himpi, "dvpp_free", lambda pointer: freed.append(pointer) or 0, raising=False)
    with pytest.raises(RuntimeError, match="vdec_destroy_chn failed") as error:
        decoder._abandon(context)
    assert decoder._channels[key] == 42 and decoder._free_error is error.value and not freed
    for _ in range(2):
        with pytest.raises(RuntimeError, match="vdec_destroy_chn failed") as repeated:
            decoder.close()
        assert repeated.value is error.value
    assert decoder._leased and npu_images._MEDIA_USERS[device] == 1 and not freed


@pytest.mark.parametrize("failure", ["create", "record"])
def test_failed_copy_fence_requires_successful_sync_before_free(media, monkeypatch, failure):
    device = torch.device("cpu:0")
    decoder = npu_images.NpuJpegDecoder(device)
    decoder([torch.ones(1, dtype=torch.uint8)])
    decoder._async_copy = True
    # A previous wave's complete event must not stand in for this wave's failed fence.
    decoder._copy_event = SimpleNamespace(query=lambda: True)
    freed, synchronizations = [], []

    def finish(context, planes=False):
        context["channel"] = None
        decoder._deferred.append((context["source"], context["output"]))
        return torch.ones(1, dtype=torch.uint8)

    def fail():
        raise RuntimeError("copy fence failed")

    def synchronize():
        assert media.device == device
        synchronizations.append(True)
        if len(synchronizations) == 1:
            raise RuntimeError("copy sync failed")

    event = SimpleNamespace(record=lambda stream: fail())
    monkeypatch.setattr(torch.npu, "Event", fail if failure == "create" else lambda: event, raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object(), raising=False)
    monkeypatch.setattr(torch.npu, "synchronize", synchronize, raising=False)
    monkeypatch.setattr(decoder.acl.himpi, "dvpp_free", lambda pointer: freed.append(pointer) or 0, raising=False)
    monkeypatch.setattr(decoder, "_begin", lambda jpeg: dict(channel=42, source=11, output=22,
                                                           width=16, height=16, size=1))
    monkeypatch.setattr(decoder, "_finish", finish)
    monkeypatch.setattr(decoder, "_abandon", lambda context: pytest.fail("finished frame abandoned twice"))
    with pytest.raises(RuntimeError, match="copy fence failed"):
        decoder([torch.ones(1, dtype=torch.uint8)])
    assert decoder._deferred == [(11, 22)] and not freed
    with pytest.raises(RuntimeError, match="copy sync failed"):
        decoder.close()
    assert decoder._copy_unfenced and decoder._leased and not freed
    decoder.close()
    assert not decoder._copy_unfenced and not decoder._leased
    assert len(synchronizations) == 2 and freed == [11, 22]


@pytest.mark.parametrize("failure", ["conversion", "release", "abandon"])
def test_failed_frame_conversion_still_fences_queued_copies(media, monkeypatch, failure):
    decoder = npu_images.NpuJpegDecoder(torch.device("cpu:0"))
    decoder._async_copy = True
    calls = []
    transactions = iter([
        dict(channel=42, source=11, output=22, out_size=6, width=16, height=16, size=1),
        dict(channel=43, source=33, output=44, out_size=6, width=32, height=32, size=1),
    ])

    class Event:
        def record(self, stream):
            calls.append("record")

        def query(self):
            return False

        def synchronize(self):
            calls.append("sync")

    def fail_conversion(*args):
        raise RuntimeError("invalid plane metadata")

    def fail_abandon(context):
        raise RuntimeError("abandon failed")

    def release(channel, frame):
        calls.append("release")
        return 1 if failure == "release" else 0
    release.__name__ = "vdec_release_frame"

    monkeypatch.setattr(torch.npu, "Event", Event, raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(npu_stream=0), raising=False)
    monkeypatch.setattr(decoder, "_begin", lambda jpeg: next(transactions))
    monkeypatch.setattr(decoder, "_frame_view", fail_conversion)
    monkeypatch.setattr(decoder.acl.himpi, "vdec_get_frame",
                        lambda channel, timeout: ({"v_frame": {"frame_flag": 0}}, None, None, 0), raising=False)
    monkeypatch.setattr(decoder.acl.himpi, "vdec_release_frame", release, raising=False)
    monkeypatch.setattr(decoder.acl.rt, "memcpy_async", lambda *args: calls.append("copy") or 0, raising=False)
    monkeypatch.setattr(decoder.acl.himpi, "dvpp_free", lambda pointer: calls.append(pointer) or 0, raising=False)
    if failure == "abandon":
        monkeypatch.setattr(decoder, "_abandon", fail_abandon)
    message = {"abandon": "abandon failed", "release": "vdec_release_frame failed",
               "conversion": "invalid plane metadata"}[failure]
    with pytest.raises(RuntimeError, match=message):
        decoder([torch.ones(1, dtype=torch.uint8)] * (2 if failure == "abandon" else 1))
    assert decoder._deferred == [(11, 22)]
    assert calls == ["copy", "release", "record"]
    decoder.close()
    assert calls == ["copy", "release", "record", "sync", 11, 22]


@pytest.mark.parametrize("failure", ["release", "destroy", "frame_flag"])
def test_fast_finish_retains_copied_buffers_until_safe_cleanup(media, monkeypatch, failure):
    decoder = npu_images.NpuJpegDecoder(torch.device("cpu:0"))
    decoder._fast = decoder._async_copy = True
    context = dict(channel=42, source=11, output=22, out_size=6, width=16, height=16, size=1)
    decoder._channels[decoder._channel_key(16, 16, 1)] = 42
    calls = []

    class Event:
        def record(self, stream):
            calls.append("record")

        def query(self):
            return False

        def synchronize(self):
            calls.append("sync")

    def finish(*args):
        calls.append("copy")
        status, phase, flag = (0, 3, 1) if failure == "frame_flag" else (1, 2, 0)
        return status, phase, 0, flag, 0, 0, 0, 0, 0, 0, 0, 0, 0

    def free(pointers):
        calls.extend(pointers)
        return len(pointers)

    def vdec_destroy_chn(channel):
        calls.append("destroy")
        return int(failure == "destroy")

    monkeypatch.setattr(npu_images, "_aclfast", SimpleNamespace(finish=finish, free_batch=free))
    monkeypatch.setattr(torch.npu, "Event", Event, raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(npu_stream=0), raising=False)
    monkeypatch.setattr(decoder, "_begin", lambda jpeg: context)
    monkeypatch.setattr(decoder.acl.himpi, "vdec_stop_recv_stream", lambda channel: calls.append("stop") or 0,
                        raising=False)
    monkeypatch.setattr(decoder.acl.himpi, "vdec_destroy_chn", vdec_destroy_chn, raising=False)
    message = {"release": "vdec_release_frame failed", "destroy": "vdec_destroy_chn failed",
               "frame_flag": "rejected an image"}[failure]
    with pytest.raises(RuntimeError, match=message):
        decoder([torch.ones(1, dtype=torch.uint8)])
    assert decoder._deferred == [(11, 22)] and 11 not in calls and 22 not in calls
    assert calls[-1] == "record"
    if failure == "destroy":
        for _ in range(2):
            with pytest.raises(RuntimeError, match=message):
                decoder.close()
        assert decoder._leased and 11 not in calls and 22 not in calls
    else:
        decoder.close()
        assert calls.index("sync") < calls.index(11) < calls.index(22)
        assert not decoder._deferred and not decoder._leased


def test_decoder_instances_keep_their_selected_quantizers_and_reference_selection(monkeypatch):
    from xinfer.engine import device_images
    from xinfer.ops import registry
    from xinfer.ops.backends.ascend import image as native
    from xinfer.platforms import npu
    from xinfer.platforms.cpu import cpu_platform

    table = registry.build_table(cpu_platform(), reference_only=True)
    assert table.selection("image_quantize").backend == "reference"
    calls = []

    def quantize(view):
        calls.append(True)
        return torch.full(view.shape, 17, dtype=torch.uint8)

    monkeypatch.setattr(native.soup2, "enabled", lambda family: True)
    monkeypatch.setattr(native.soup2, "quant_u8", quantize)
    monkeypatch.setattr(npu, "load_npu_media", lambda: SimpleNamespace(rt=SimpleNamespace()))
    monkeypatch.setattr(npu, "_aclfast", None)
    monkeypatch.setattr(device_images, "jpeg_decoder_for",
                        lambda device, image_quantize: npu.NpuJpegDecoder(torch.device("cpu"), quantize=image_quantize))
    decoders = [device_images.DeviceImageDecoder("cpu", image_quantize=callback)._decode
                for callback in (native.image_quantize, table.image_quantize)]
    fmt = next(fmt for fmt, (h, w, _) in npu._CHROMA.items() if h == w == 1)
    meta = dict(fmt=fmt, h=2, w=2, hs=2, ws=2, luma_size=4, start=4, end=12, ch=2, cs=4)
    data = torch.full((12,), 128, dtype=torch.uint8)
    for decoder in decoders:
        monkeypatch.setattr(decoder, "_plane_meta", lambda *args: meta)
    for index, expected in ((0, 17), (1, 128), (0, 17)):
        out = decoders[index]._frame_view({}, 0, 12, 2, 2, data)
        assert torch.equal(out, torch.full((3, 2, 2), expected, dtype=torch.uint8))
    assert len(calls) == 2


def test_selected_quantizer_propagates_execution_errors(monkeypatch):
    from xinfer.ops.backends.ascend import image

    monkeypatch.setattr(image.soup2, "enabled", lambda family: True)

    def fail(view):
        raise RuntimeError("image quantizer launch failed")

    monkeypatch.setattr(image.soup2, "quant_u8", fail)
    with pytest.raises(RuntimeError, match="image quantizer launch failed"):
        image.image_quantize(torch.ones(3, 2, 2))
