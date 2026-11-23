"""A two-stage CUDA model whose stages call operators of the active op table (imported by path by xinfer-profile)."""

import torch

from xinfer.models.interface import HandoffSpec, OutputSpec, PreparedRequest, Stage, StageGraph, TensorSpec, VLAModel
from xinfer.ops import active_ops


class Tiny(VLAModel):
    """encode: h = linear(x, W) [1, 256]; decode: y = rms_norm(h)."""

    stage_graph = StageGraph(Stage("encode", "encode"), Stage("decode", "decode", depends_on="encode"))
    handoffs = (HandoffSpec("encode", "decode", {"h": TensorSpec()}),)

    @classmethod
    def from_checkpoint(cls, path, *, device, ops=None, **_):
        model, table = cls(), active_ops()
        model.linear, model.rms_norm = table.linear, table.rms_norm
        model.weight = torch.randn(256, 64, generator=torch.Generator().manual_seed(0)).to(device, torch.bfloat16)
        model.norm_weight = torch.ones(256, device=device)
        return model

    def input_processor(self):
        return lambda observation: PreparedRequest({"x": torch.tensor(observation["x"], dtype=torch.bfloat16)[None]})

    def stage_outputs(self, stage, batch):
        return {"h" if stage == "encode" else "y": OutputSpec((batch.batch_size, 256), torch.bfloat16)}

    def encode(self, batch):
        batch.out["h"].copy_(self.linear(batch.inputs["x"], self.weight))

    def decode(self, batch):
        batch.out["y"].copy_(self.rms_norm(batch.inputs["h"], self.norm_weight, 1e-6))


class ReferenceProjection(Tiny):
    """Exercise a GEMM option the CUDA backend delegates to the torch reference."""

    @classmethod
    def from_checkpoint(cls, path, *, device, ops=None, **options):
        model = super().from_checkpoint(path, device=device, ops=ops, **options)
        model.project = ops.gemm_epilogue
        return model

    def encode(self, batch):
        output = batch.out["h"]
        residual = torch.zeros_like(output)
        output.copy_(self.project(batch.inputs["x"], self.weight, activation="silu", residual=residual))
