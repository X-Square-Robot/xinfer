"""The public operator table uses optional attention libraries on shapes outside the native kernels."""

from types import SimpleNamespace

import pytest
import torch

from xinfer.ops import reference
from xinfer.ops.backends.cuda import _attention_libraries as libraries
from xinfer.ops.registry import observe_reference_calls


def test_declined_signatures_cold_capture_and_confirmed_errors(monkeypatch):
    device = torch.device("cuda", 0)
    args = (SimpleNamespace(device=device),)
    monkeypatch.setattr(libraries, "layout", lambda args: ("shape",))
    monkeypatch.setattr(libraries, "_support", {})
    monkeypatch.setattr(libraries, "capturing", lambda device: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: None))
    lib = SimpleNamespace(__name__="example")
    attempts = []

    def decline():
        attempts.append(1)
        raise ValueError("unsupported")

    def run(fn):
        return libraries.run("op", lib, args, fn, lambda: "reference")

    with observe_reference_calls(lambda _: None):
        assert run(decline) == "reference" and not attempts
        monkeypatch.setattr(libraries, "capturing", lambda device: False)
        assert run(decline) == run(decline) == "reference" and attempts == [1]
        libraries._support.clear()
        assert run(lambda: 42) == 42
        with pytest.raises(ValueError, match="unsupported"):
            run(decline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("op", "library"), [
    ("segment_attention", "flash_attn"), ("gated_causal_attention", "flash_attn"),
    ("prefix_kvcache_attention", "flex_attention"), ("gated_kvcache_attention", "flex_attention"),
    ("gated_delta_net", "fla")])
def test_external_route_matches_reference_and_replays_in_a_graph(op, library):
    from xinfer import ops
    from xinfer.ops.backends.cuda import attn
    from xinfer.platforms import resolve_platform

    available = {"flash_attn": attn.flash_attn, "flex_attention": libraries.FLEX, "fla": libraries.FLA}
    if available[library] is None:
        pytest.skip(f"{library} unavailable")
    ops.registry.load_plugins()
    table = ops.registry.build_table(resolve_platform("cuda"), backends={op: library})
    selected = getattr(table, op)
    generator = torch.Generator(device="cuda").manual_seed(461)

    def tensor(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)

    rows, heads, kv, width = 32, 4, 2, 64
    if op == "segment_attention":
        kv = heads  # the vision segment operator uses the same number of Q, K and V heads
    kwargs = {}
    if op == "gated_delta_net":
        heads = kv = 2
        args = (tensor(1, rows, 3 * heads * width), tensor(1, rows, heads * (width + 2)),
                tensor(3 * heads * width, 4), tensor(heads).float(), tensor(heads).float(), tensor(width),
                heads, heads, width, width, 1e-6)
    elif "kvcache" in op:
        length = 77
        keys, values = (tensor(1, kv, length, width).transpose(1, 2) for _ in range(2))
        args = (tensor(1, rows, heads, width), keys, values, torch.tensor([19], device="cuda", dtype=torch.int32), 40)
        if op.startswith("gated"):
            args += (tensor(1, rows, heads * width),)
        mask = torch.zeros((1, 1, length, 2), dtype=torch.int32, device="cuda")
        mask[..., 0] = rows
        mask[:, :, :4, 1] = 1
        kwargs = {"flash_mask": mask, "scale": 0.13}
    else:
        args = (tensor(rows, heads, width), tensor(rows, kv, width), tensor(rows, kv, width))
        args += ((0, 12, rows),) if op == "segment_attention" else (tensor(rows, heads * width),)

    ran = []
    with observe_reference_calls(lambda source: pytest.fail(f"fell back to {source}"), ran.append):
        actual = selected(*args, **kwargs)
    expected = getattr(reference, op)(*args, **kwargs)
    assert ran == [library] and actual.shape == expected.shape and actual.dtype == expected.dtype
    err = torch.linalg.vector_norm(actual.float() - expected.float())
    assert err < (0.02 if library == "fla" else 0.01) * torch.linalg.vector_norm(expected.float())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = selected(*args, **kwargs)
    graph.replay()
    torch.testing.assert_close(replay, actual, rtol=0, atol=0)
