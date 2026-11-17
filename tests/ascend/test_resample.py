"""Ascend antialiased resampling workspaces and launch boundaries."""

from types import SimpleNamespace
from contextlib import contextmanager

import pytest
import torch

from xinfer.ops.backends.ascend import _native, aa_resample


@pytest.fixture
def streams(monkeypatch):
    state = SimpleNamespace(device=0, stream=11)
    runtime = SimpleNamespace(current_device=lambda: state.device,
                              current_stream=lambda device: SimpleNamespace(npu_stream=state.stream))
    monkeypatch.setattr(torch, "npu", runtime, raising=False)
    monkeypatch.setattr(aa_resample, "_SHAPES", {})
    return state


def pack(device):
    return aa_resample._shape_pack(device, 480, 640, 160, 224, 13)


def test_streams_do_not_overwrite_each_others_vertical_pass_workspace(streams):
    first = pack("cpu:0")
    first["mid"].fill_(17)
    streams.stream = 12
    second = pack("cpu:0")
    second["mid"].fill_(23)
    assert first["mid"].eq(17).all()
    assert second["mid"].eq(23).all()
    streams.stream = 11
    assert pack("cpu:0") is first
    for name in ("wy", "basey", "wx", "ix"):
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_device_identity_separates_equal_stream_handles_and_resolves_current_device(streams):
    first = pack("cpu:0")
    assert pack("cpu") is first
    streams.device = 1
    second = pack("cpu")
    assert second is pack("cpu:1") and second is not first
    first["mid"].fill_(31)
    second["mid"].fill_(47)
    assert first["mid"].eq(31).all()
    assert second["mid"].eq(47).all()


def test_image_chain_warms_on_the_capture_stream_before_entering_the_graph(monkeypatch):
    from xinfer.platforms import npu_image_graphs as images

    events = []
    upload_stream = object()
    state = SimpleNamespace(stream=upload_stream, capturing=False, warmed=False, synchronized=False)

    class CaptureStream:
        def wait_stream(self, other):
            assert other is upload_stream
            events.append("wait for input")

        def synchronize(self):
            assert state.warmed and not state.capturing
            state.synchronized = True
            events.append("warmup complete")

    capture_stream = CaptureStream()

    @contextmanager
    def use_stream(stream):
        previous = state.stream
        state.stream = stream
        try:
            yield
        finally:
            state.stream = previous

    @contextmanager
    def graph_context(graph, *, pool, stream, capture_error_mode):
        assert state.synchronized
        events.append("capture begins")
        with use_stream(stream):
            state.capturing = True
            try:
                yield
            finally:
                state.capturing = False

    def chain(*args):
        assert state.stream is capture_stream and torch.is_inference_mode_enabled()
        if state.capturing:
            assert state.warmed
            events.append("captured chain")
        else:
            state.warmed = True
            events.append("eager chain")
        return torch.zeros(2, 2, 3, dtype=torch.uint8)

    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda device: upload_stream,
                                                      stream=use_stream, graph=graph_context, NPUGraph=object),
                        raising=False)
    monkeypatch.setattr(images, "chain_view", chain)
    cache = images.ResizeChainGraphs(torch.device("cpu"), SimpleNamespace(rt=SimpleNamespace()), None, None)
    cache._stream, cache._pool = capture_stream, object()
    meta = dict(out_size=4, fmt=0, h=2, w=2, hs=2, ws=2, luma_size=4, ch=0, cs=0, start=4, end=4)
    cache._capture(meta, ((2, 2),), (("bicubic", True),))
    assert events == ["wait for input", "eager chain", "warmup complete", "capture begins", "captured chain"]
    assert cache.stats["captures"] == 1


def test_aa_second_pass_failure_does_not_decline_the_whole_resize(monkeypatch):
    class Image:
        shape = (1, 3, 480, 640)
        dtype = torch.uint8
        device = SimpleNamespace(type="npu")
        def dim(self):
            return 4

        def is_contiguous(self):
            return True

        def view(self, *shape):
            return torch.zeros(shape, dtype=torch.uint8)

    monkeypatch.setattr(aa_resample, "enabled", lambda: True)
    monkeypatch.setattr(aa_resample, "_entry", lambda *args: object())
    monkeypatch.setattr(aa_resample, "_shape_pack", lambda *args: dict(
        vname="vertical", hname="horizontal", mid=None, wy=None, basey=None, wx=None, ix=None))
    empty = torch.empty
    monkeypatch.setattr(aa_resample.torch, "empty", lambda *args, device=None, **kwargs: empty(*args, **kwargs))
    calls = []

    def launch(name, *args):
        calls.append(name)
        if name == "horizontal":
            raise RuntimeError("horizontal launch failed")
        return True

    monkeypatch.setattr(aa_resample, "_launch", launch)
    with pytest.raises(RuntimeError, match="horizontal launch failed"):
        aa_resample.step(Image(), (160, 224))
    assert calls == ["vertical", "horizontal"]


def test_aa_unsupported_and_missing_entries_do_not_launch(monkeypatch):
    launch = aa_resample._launch
    monkeypatch.setattr(aa_resample, "_launch", lambda *args: pytest.fail("unsupported image launched"))
    assert aa_resample.step(torch.zeros(1, 3, 8, 8, dtype=torch.uint8), (4, 4)) is None
    monkeypatch.setattr(_native, "launch_kernel", lambda *args: False)
    monkeypatch.setattr(aa_resample, "_entry", lambda *args: None)
    assert launch("missing", (), ()) is False
