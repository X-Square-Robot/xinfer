"""Ragged metadata must not wait for an upstream GPU producer on the host.

The GPU producer waits for an explicit host-controlled release, rather than
depending on a sleep duration. A watchdog releases it if an old blocking copy
prevents the host from progressing, so the failure does not deadlock pytest.
"""

import threading

import pytest
import torch

from xinfer.engine.batching import merge

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


def test_ragged_cpu_lengths_and_left_padding():
    views = [{"kv": torch.full((1, n, 2), float(i + 1))} for i, n in enumerate((3, 5, 2))]
    batch = merge(views, {"kv": 0}, {"kv": (0, 1)})
    assert torch.equal(batch["kv_lengths"], torch.tensor([3, 5, 2], dtype=torch.int32))
    expected = torch.zeros(3, 5, 2)
    for i, n in enumerate((3, 5, 2)):
        expected[i, -n:] = i + 1
    assert torch.equal(batch["kv"], expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or triton is None, reason="CUDA + Triton required")
def test_ragged_lengths_leave_issuer_free_while_producer_is_pending():
    device = torch.device("cuda", torch.cuda.current_device())
    setup = torch.cuda.current_stream(device)
    producer, consumer, releaser = (torch.cuda.Stream(device=device) for _ in range(3))
    flag = torch.ones((), device=device, dtype=torch.int32)
    views = [{"kv": torch.full((1, n, 2), float(i + 1), device=device)}
             for i, n in enumerate((3, 5, 2))]
    axes, ragged = {"kv": 0}, {"kv": (0, 1)}

    # Compile the gate kernels before testing a blocked stream. Prime allocations
    # on the same consumer stream to keep first-use cudaMalloc out of the test.
    producer.wait_stream(setup)
    with torch.cuda.stream(producer):
        _wait_for_release[(1,)](flag)
    releaser.wait_stream(setup)
    with torch.cuda.stream(releaser):
        _release[(1,)](flag)
    consumer.wait_stream(setup)
    with torch.cuda.stream(consumer):
        warm = merge(views, axes, ragged)
        lengths, packed = warm["kv_lengths"], warm["kv"]
    torch.cuda.synchronize(device)
    del warm, lengths, packed
    # The test later attempts allocator reuse while the copy is still pending.
    primed = [torch.empty(3, dtype=torch.int32, pin_memory=True) for _ in range(33)]
    del primed

    flag.zero_()
    producer.wait_stream(setup)
    with torch.cuda.stream(producer):
        _wait_for_release[(1,)](flag)
        producer_done = torch.cuda.Event()
        producer_done.record()
    consumer.wait_event(producer_done)

    host_progressed = threading.Event()
    release_was_requested = []

    def release_producer():
        # A blocking metadata upload cannot reach host_progressed.set(). The
        # timeout is only a deadlock escape, not the measured assertion.
        requested = host_progressed.wait(timeout=10)
        release_was_requested.append(requested)
        with torch.cuda.device(device), torch.cuda.stream(releaser):
            _release[(1,)](flag)
        releaser.synchronize()

    watchdog = threading.Thread(target=release_producer, daemon=True)
    watchdog.start()
    try:
        assert not producer_done.query()
        with torch.cuda.stream(consumer):
            batch = merge(views, axes, ragged)
            lengths, packed = batch["kv_lengths"], batch["kv"]
        # The producer cannot finish until explicitly released. If a host API
        # waited for it, only the watchdog could have released this dependency.
        assert not producer_done.query(), "ragged metadata blocked the issuer behind its producer"
        churn = [torch.full((3,), 99, dtype=torch.int32, pin_memory=True) for _ in range(32)]
    finally:
        host_progressed.set()
        watchdog.join(timeout=15)
        consumer.synchronize()
        producer.synchronize()
    assert not watchdog.is_alive()
    assert release_was_requested == [True], "producer required watchdog release instead of host progress"
    assert torch.equal(lengths.cpu(), torch.tensor([3, 5, 2], dtype=torch.int32))
    expected = torch.zeros(3, 5, 2)
    for i, n in enumerate((3, 5, 2)):
        expected[i, -n:] = i + 1
    assert torch.equal(packed.cpu(), expected)
    del churn
