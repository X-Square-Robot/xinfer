"""DVPP session ownership without an NPU or Python ACL."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from xinfer.platforms import npu as npu_images


@pytest.fixture
def media(monkeypatch):
    state = SimpleNamespace(device=None, calls=[], init_status=0, exit_status=0)

    @contextmanager
    def device_context(device):
        previous = state.device
        state.device = device
        try:
            yield
        finally:
            state.device = previous

    def sys_init():
        state.calls.append(("init", state.device))
        return state.init_status

    def sys_exit():
        state.calls.append(("exit", state.device))
        return state.exit_status

    acl = SimpleNamespace(himpi=SimpleNamespace(sys_init=sys_init, sys_exit=sys_exit))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=device_context), raising=False)
    monkeypatch.setattr(npu_images, "load_npu_media", lambda: acl)
    monkeypatch.setattr(npu_images, "_MEDIA_USERS", {})
    monkeypatch.setattr(npu_images.NpuJpegDecoder, "_decode", lambda self, jpeg: jpeg)
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
