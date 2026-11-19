"""Policy construction releases its connected input processor on later failure."""

from types import SimpleNamespace

import pytest

from xinfer.entrypoints import policy as policy_module
from xinfer.models.interface import Capabilities


@pytest.mark.parametrize("stage", ["output_processor", "resolve_config", "create_engine"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_constructor_failure_closes_connected_processor_without_masking_error(monkeypatch, stage, cleanup_fails):
    primary = ValueError(f"{stage} failed")
    closed = []

    def close():
        closed.append(True)
        if cleanup_fails:
            raise RuntimeError("processor cleanup failed")

    processor = SimpleNamespace(connect=lambda: SimpleNamespace(close=close))
    model = SimpleNamespace(
        capabilities=lambda: Capabilities(batch_sizes={"step": (1,)}, num_steps=1),
        io_schema=lambda: None,
        input_processor=lambda: processor,
        output_processor=lambda: None,
    )
    monkeypatch.setattr(policy_module, "get_model_class",
                        lambda name: SimpleNamespace(from_checkpoint=lambda *args, **kwargs: model))
    monkeypatch.setattr(policy_module.ops.registry, "load_plugins", lambda: None)
    monkeypatch.setattr(policy_module._engine_factory, "resolve_config",
                        lambda *args, **kwargs: SimpleNamespace(engine_config=lambda graph_policy: None))
    monkeypatch.setattr(policy_module._engine_factory, "create_engine", lambda *args: None)

    def fail(*args, **kwargs):
        raise primary

    target = model if stage == "output_processor" else policy_module._engine_factory
    monkeypatch.setattr(target, stage, fail)
    with pytest.raises(ValueError) as caught:
        policy_module.Policy("test", "unused", device="cpu", reference_ops=True)
    assert caught.value is primary
    assert closed == [True]
