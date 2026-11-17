"""Request-derived rotary tables retain the model's frequencies and the destination device."""

from types import SimpleNamespace

import torch

from xinfer.models.wallx.qwen35 import text_rope_tables
from xinfer.models.wallx2b.model import Wallx2b


def _model(frequencies):
    model = Wallx2b.__new__(Wallx2b)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(model=SimpleNamespace(rotary_dim=8, mrope_section=(2, 1, 1)))
    model.register_buffer("inv_freq", torch.tensor(frequencies))
    return model


def _positions():
    return torch.arange(6).reshape(1, 1, -1).expand(3, 1, -1).clone()


def _reference(model, positions):
    cos, sin = text_rope_tables(model.inv_freq, positions, model.config.model.mrope_section)
    return cos[:, :, 0, :4].contiguous(), sin[:, :, 0, :4].contiguous()


def test_tables_belong_to_the_model_that_generated_them():
    positions = _positions()
    first = _model([1.0, 0.1, 0.01, 0.001])
    second = _model([0.5, 0.2, 0.02, 0.002])
    first_pair = first._cached_rope_tables(positions, positions)
    second_pair = second._cached_rope_tables(positions, positions)
    assert not torch.equal(first_pair[0], second_pair[0])
    for actual, expected in zip(second_pair, _reference(second, positions)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert first._cached_rope_tables(positions.clone(), positions)[0] is first_pair[0]


def test_changing_the_frequency_buffer_invalidates_cached_tables():
    positions = _positions()
    model = _model([1.0, 0.1, 0.01, 0.001])
    previous = model._cached_rope_tables(positions, positions)
    model.inv_freq.mul_(0.5)
    changed = model._cached_rope_tables(positions, positions)
    assert changed[0] is not previous[0]
    for actual, expected in zip(changed, _reference(model, positions)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model.inv_freq = model.inv_freq.clone()
    assert model._cached_rope_tables(positions, positions)[0] is not changed[0]


def test_tables_are_not_reused_on_another_device(monkeypatch):
    positions = _positions()
    model = _model([1.0, 0.1, 0.01, 0.001])
    cpu_pair = model._cached_rope_tables(positions, positions)

    def tables_on_destination(inv_freq, target, section):
        return tuple(torch.empty(1, 6, 1, 8, device=target.device) for _ in range(2))

    monkeypatch.setattr("xinfer.models.wallx2b.model.text_rope_tables", tables_on_destination)
    meta_pair = model._cached_rope_tables(positions, positions.to("meta"))
    assert meta_pair[0].device.type == "meta"
    assert model._cached_rope_tables(positions, positions)[0] is cpu_pair[0]


def test_inference_buffers_do_not_require_a_version_counter():
    positions = _positions()
    with torch.inference_mode():
        model = _model([1.0, 0.1, 0.01, 0.001])
        pair = model._cached_rope_tables(positions, positions)
        assert model._cached_rope_tables(positions, positions)[0] is pair[0]
