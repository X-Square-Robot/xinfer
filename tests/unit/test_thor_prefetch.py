"""Weight prefetch must fork from the dispatch lane and rejoin it before graph capture ends."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from xinfer.ops.backends.cuda import thor


@pytest.mark.parametrize("fail", [False, True])
def test_prefetch_stream_order(monkeypatch, fail):
    lane, side = Mock(), Mock()
    device = torch.device("cuda", 1)
    lane.device = side.device = device
    weights = (Mock(device=device), Mock(device=device))
    events = []
    current = Mock(return_value=lane)
    new_stream = Mock(return_value=side)
    monkeypatch.setattr(thor, "_integrated_gpu", lambda device: True)
    monkeypatch.setattr(torch.cuda, "current_stream", current)
    monkeypatch.setattr(torch.cuda, "Stream", new_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    side.wait_stream.side_effect = lambda stream: events.append(("fork", stream))
    lane.wait_stream.side_effect = lambda stream: events.append(("join", stream))

    def read(weight, grid):
        assert grid == 2
        events.append(("read", weight))
        if fail:
            raise RuntimeError("kernel failed")

    monkeypatch.setattr(thor, "xc", Mock(l2_prefetch=read))
    if fail:
        with pytest.raises(RuntimeError, match="kernel failed"):
            thor.prefetch_weights(weights)
        assert events == [("fork", lane), ("read", weights[0]), ("join", side)]
    else:
        for _ in range(2):
            ticket = thor.prefetch_weights(weights)
            assert ticket is side
        thor.join_prefetch(ticket)
        assert events == [("fork", lane), ("read", weights[0]), ("read", weights[1])] * 2 + [("join", side)]
    new_stream.assert_called_once_with(device=device)
    assert all(call.args == (device,) for call in current.call_args_list)


def test_prefetch_skips_discrete_gpu(monkeypatch):
    monkeypatch.setattr(thor, "_integrated_gpu", lambda device: False)
    new_stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", new_stream)
    thor.join_prefetch(thor.prefetch_weights((Mock(device=torch.device("cuda", 0)),)))
    new_stream.assert_not_called()


@pytest.mark.gpu
def test_prefetch_graph_replay():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("requires SM110")
    device = torch.device("cuda", torch.cuda.current_device())
    if not thor._integrated_gpu(device) or thor._PREFETCH["requires"] is None:
        pytest.skip("requires integrated SM110 and xcompute L2 prefetch")
    x = torch.randn(128, device=device)
    lane = torch.cuda.Stream(device=device)
    lane.wait_stream(torch.cuda.current_stream(device))

    def step():
        side = thor.prefetch_weights((x,))
        y = x + 1
        thor.join_prefetch(side)
        return y

    with torch.cuda.stream(lane):
        step()
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=lane):
        y = step()
    torch.cuda.current_stream(device).wait_stream(lane)
    x.fill_(3)
    graph.replay()
    torch.testing.assert_close(y, torch.full_like(x, 4))
