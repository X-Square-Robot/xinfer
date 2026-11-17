"""A cached JPEGD channel cannot be recycled while its output is still in flight."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from xinfer.platforms import npu


@pytest.fixture
def decoder(monkeypatch):
    channels, destroyed = {}, []

    def create(channel, attrs):
        if channel in channels:
            return npu._CHANNEL_EXISTS
        channels[channel] = (attrs["pic_width"], attrs["pic_height"], attrs["stream_buf_size"])
        return 0

    def destroy(channel):
        destroyed.append(channels.pop(channel))
        return 0

    media = SimpleNamespace(sys_init=lambda: 0, vdec_create_chn=create,
                            dvpp_free=lambda pointer: 0,
                            vdec_destroy_chn=destroy, vdec_set_jpegd_precision_mode=lambda *args: 0,
                            vdec_start_recv_stream=lambda *args: 0, vdec_stop_recv_stream=lambda *args: 0)
    monkeypatch.setattr(npu, "_MAX_CHANNELS", 2)
    monkeypatch.setattr(npu, "_MEDIA_USERS", {})
    monkeypatch.setattr(npu, "_aclfast", None)
    monkeypatch.setattr(npu, "load_npu_media", lambda: SimpleNamespace(himpi=media, rt=SimpleNamespace()))
    value = npu.NpuJpegDecoder(torch.device("cpu"))
    monkeypatch.setattr(value, "_on_device", nullcontext)

    def begin(jpeg):
        size = jpeg.numel()
        return dict(channel=value._channel_for(1, 1, size), width=1, height=1, size=size,
                    out_size=1, source=size * 10, output=size * 100)

    def finish(context, planes=False):
        key = value._channel_key(context["width"], context["height"], context["size"])
        assert channels[context["channel"]] == key, "an in-flight channel ID was recycled"
        return context["size"]

    monkeypatch.setattr(value, "_begin", begin)
    monkeypatch.setattr(value, "_finish", finish)
    # The first cached channel will be used by the next wave; the other one is idle.
    value._channel_for(1, 1, 3)
    value._channel_for(1, 1, 4)
    return value, destroyed


@pytest.mark.parametrize("prestarted", [False, True])
def test_wave_retains_its_channels_until_the_frames_are_fetched(decoder, prestarted):
    value, destroyed = decoder
    first, second = torch.zeros(3, dtype=torch.uint8), torch.zeros(5, dtype=torch.uint8)
    started = None
    if prestarted:
        context = value._begin(first)
        value._pending[(1, 1, 3)] = (1, 3)
        started = [context, None]
    assert value([first, second], prestarted=started) == [3, 5]
    assert destroyed == [(1, 1, 4)]
    assert value._pending == {}


def test_cache_limit_never_evicts_active_channels(decoder):
    value, destroyed = decoder
    value._pending = {(1, 1, 3): (1, 3), (1, 1, 4): (1, 4)}
    value._channel_for(1, 1, 5)
    assert destroyed == [] and len(value._channels) == 3
    value._pending.clear()
    value._channel_for(1, 1, 6)
    assert len(value._channels) == 2


@pytest.mark.parametrize("batch", [False, True])
def test_failed_start_does_not_pin_an_idle_channel(decoder, monkeypatch, batch):
    value, _ = decoder
    monkeypatch.setattr(value, "_probe", lambda jpeg: (dict(width=1, height=1), {}))

    def fail(*args):
        raise RuntimeError("send failed")

    monkeypatch.setattr(value, "_launch", fail)
    image = torch.zeros(3, dtype=torch.uint8)
    if batch:
        assert isinstance(value.start_batch([image])[0], RuntimeError)
    else:
        with pytest.raises(RuntimeError, match="send failed"):
            value.start(image)
    assert value._pending == {}


def test_preparation_failure_releases_unsubmitted_reservations(decoder, monkeypatch):
    value, _ = decoder

    def probe(jpeg):
        if jpeg.numel() == 5:
            raise ValueError("invalid image")
        return dict(width=1, height=1), {}

    monkeypatch.setattr(value, "_probe", probe)
    with pytest.raises(ValueError, match="invalid image"):
        value.start_batch([torch.zeros(3, dtype=torch.uint8), torch.zeros(5, dtype=torch.uint8)])
    assert value._pending == {}
