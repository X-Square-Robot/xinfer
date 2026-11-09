import pytest

from xinfer.engine.slots import PoolExhausted, SlotPool, StaleHandle


class FakeEvent:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done


def test_allocate_until_exhausted_then_reuse():
    pool = SlotPool(2)
    a, b = pool.allocate(), pool.allocate()
    assert a.index != b.index
    with pytest.raises(PoolExhausted):
        pool.allocate()
    pool.release(a)
    c = pool.allocate()
    assert c.index == a.index and c.generation == a.generation + 1


def test_stale_handle_is_detected():
    pool = SlotPool(1)
    a = pool.allocate()
    pool.tensors(a)["kv"] = 1
    pool.release(a)
    with pytest.raises(StaleHandle):
        pool.tensors(a)
    with pytest.raises(StaleHandle):
        pool.release(a)
    b = pool.allocate()
    assert pool.tensors(b) == {}


def test_release_waits_for_last_reader():
    pool = SlotPool(1)
    a = pool.allocate()
    event = FakeEvent()
    pool.release(a, after=event)
    assert pool.in_use() == 1
    with pytest.raises(PoolExhausted):
        pool.allocate()
    event.done = True
    assert pool.in_use() == 0
    assert pool.allocate().generation == 1
