import pytest
import torch

from xinfer.engine import Engine
from xinfer.models.interface import (HandoffSpec, OutputSpec, PreparedRequest, Stage, StageGraph, TensorSpec,
                                     VLAModel)
from xinfer.platforms import resolve_platform


class ThreeStage(VLAModel):
    stage_graph = StageGraph(
        Stage("vision", "encode"),
        Stage("prefill", "prefill", depends_on="vision"),
        Stage("action", "denoise", depends_on="prefill"),
    )
    handoffs = (
        HandoffSpec("vision", "prefill", {"tokens": TensorSpec()}),
        HandoffSpec("prefill", "action", {"kv": TensorSpec()}),
    )

    def __init__(self):
        self.seen = {}

    def stage_outputs(self, stage, batch):
        n = batch.inputs["pixels"].shape[0]
        return {
            "vision": {"tokens": OutputSpec((n,), torch.float32), "scratch": OutputSpec((1,), torch.float32)},
            "prefill": {"kv": OutputSpec((n,), torch.float32)},
            "action": {"actions": OutputSpec((n,), torch.float32)},
        }[stage]

    def encode(self, batch):
        self.seen["vision"] = set(batch.inputs)
        batch.out["tokens"].copy_(batch.inputs["pixels"] * 2)
        batch.out["scratch"].zero_()

    def prefill(self, batch):
        self.seen["prefill"] = set(batch.inputs)
        torch.add(batch.inputs["tokens"], batch.inputs["ids"], out=batch.out["kv"])

    def denoise(self, batch):
        self.seen["action"] = set(batch.inputs)
        torch.sub(batch.inputs["kv"], 1, out=batch.out["actions"])


def _run(model):
    engine = Engine(model, resolve_platform("cpu"))
    request = PreparedRequest({"pixels": torch.ones(3), "ids": torch.arange(3.0)})
    return engine, engine.run(request)


def test_runs_stages_in_order_and_returns_final_outputs():
    model = ThreeStage()
    engine, out = _run(model)
    torch.testing.assert_close(out["actions"], torch.tensor([1.0, 2.0, 3.0]))
    assert engine.slots.in_use() == 0


def test_stage_sees_only_declared_handoffs():
    model = ThreeStage()
    _run(model)
    assert model.seen["vision"] == {"pixels", "ids"}
    assert model.seen["prefill"] == {"pixels", "ids", "tokens"}
    # Undeclared "scratch" and the vision tokens do not reach the action stage.
    assert model.seen["action"] == {"pixels", "ids", "kv"}


def test_undeclared_handoff_output_is_an_error():
    class Broken(ThreeStage):
        def stage_outputs(self, stage, batch):
            return {} if stage == "prefill" else super().stage_outputs(stage, batch)

    engine = Engine(Broken(), resolve_platform("cpu"))
    with pytest.raises(ValueError, match="does not declare"):
        engine.run(PreparedRequest({"pixels": torch.ones(3), "ids": torch.ones(3)}))
    assert engine.slots.in_use() == 0


def test_handoff_must_follow_dependencies():
    class Skips(ThreeStage):
        handoffs = (HandoffSpec("action", "vision", {"x": TensorSpec()}),)

    with pytest.raises(ValueError, match="not along a stage dependency"):
        Engine(Skips(), resolve_platform("cpu"))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_streams_chain_with_events():
    engine = Engine(ThreeStage(), resolve_platform("cuda"))
    out = engine.run(PreparedRequest({"pixels": torch.ones(3), "ids": torch.arange(3.0)}))
    assert out["actions"].device.type == "cpu"
    torch.testing.assert_close(out["actions"], torch.tensor([1.0, 2.0, 3.0]))
    assert len({id(lane.stream) for lanes in engine.lanes.values() for lane in lanes}) == 3
