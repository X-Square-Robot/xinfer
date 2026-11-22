"""Vision CPU metadata/alias contracts and independent HF position arithmetic."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from xinfer.models.wall_oss_1_0_nano.rpu.vision import (
    VisionPlan,
    interpolate_positions,
    vision_coordinates,
    vision_rope_tables,
)
from wall_oss_1_0_nano_plan_helpers import PlanContext, vision_weights


def test_position_interpolation_preserves_hf_half_boundaries():
    hf = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    weight = torch.linspace(-2, 2, 2304 * 1024).reshape(2304, 1024).half()
    reference = SimpleNamespace(pos_embed=torch.nn.Embedding.from_pretrained(weight), num_grid_per_side=48,
                                config=SimpleNamespace(spatial_merge_size=2))
    grid = torch.tensor([[1, 14, 22], [1, 14, 20], [1, 14, 20]])
    expected = hf.Qwen3_5VisionModel.fast_pos_embed_interpolate(reference, grid)
    np.testing.assert_array_equal(interpolate_positions(weight), expected.numpy())


def test_vision_rope_uses_merge_order_and_fp16_outer_product():
    coordinates = vision_coordinates()
    assert coordinates.shape == (868, 2)
    assert coordinates[:4].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert coordinates[308:312].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    inverse = (1 / (10000 ** (torch.arange(0, 32, 2).float() / 32))).half()
    table = torch.arange(22).half()[:, None] * inverse[None]
    indices = torch.from_numpy(coordinates).long()
    selected = torch.cat((table[indices[:, 0]], table[indices[:, 1]]), dim=1)
    cosine, sine = vision_rope_tables()
    np.testing.assert_array_equal(cosine, selected.cos().numpy())
    np.testing.assert_array_equal(sine, selected.sin().numpy())


@pytest.mark.parametrize("fused", [True, False])
def test_vision_lowering_fits_three_shared_slots_and_native_batch_limits(fused):
    context = PlanContext()
    arena = context.empty((3 * 512 * 2048,), distribution="local")
    plan = VisionPlan(context, SimpleNamespace(action_horizon=16, max_prefix_length=512),
                      vision_weights(context), torch.zeros((2304, 1024), dtype=torch.float16),
                      workspace=arena, fused=fused)
    assert [batch.collectives for batch in context.batches] == [1, 48, 2]
    assert plan.collective_count == 51
    assert plan.operation_count == sum(len(batch.operations) for batch in context.batches)
    assert plan.output_ddr.shape == (217, 2048)
    merger = [op for op in plan.operations if op.name in ("fc1_n512k4096", "fc1_gelu_n512k4096")]
    assert [op.tensors["x"].shape[0] for op in merger] == [128, 89]
    assert context.spm_bytes <= 7 * 1024 * 1024
    assert sum(op.name == "qkv_rope_bias_n384k1024" for op in plan.operations) == (24 if fused else 0)
    assert sum(op.name == "flash_attention_varlen_f16" for op in plan.operations) == (0 if fused else 24)
    # Unfused GELUs keep the checkpoint's numerics: tanh in the tower MLP,
    # erf in the merger (the device kernel's mode 1 / mode 0).
    gelu_modes = [op.integers["mode"] for op in plan.operations if op.name == "gelu_f16"]
    assert gelu_modes == ([] if fused else [1] * 24 + [0, 0])
    prefix = context.empty((448, 2048), storage="ddr", distribution="shared")
    scatter = plan.scatter_to(prefix, (20, 120, 220))
    assert len(scatter) == 3
    assert [op.tensors["input"].shape[0] for op in scatter] == [77, 70, 70]
    with pytest.raises(ValueError, match="ordered"):
        plan.scatter_to(prefix, (20, 30, 220))


def test_vision_reduce_norm_fusion_preserves_owner_shards_and_final_replication():
    context = PlanContext()
    config = SimpleNamespace(action_horizon=16, max_prefix_length=512)
    plan = VisionPlan(context, config, vision_weights(context), torch.zeros((2304, 1024)).half(),
                      fused_reduce_norm=True)
    fused = [op for op in plan.operations if op.name == "all_reduce_residual_layernorm"]
    assert len(fused) == 47
    assert [batch.collectives for batch in context.batches] == [1, 48, 2]
    assert sum(op.name == "layernorm_f16" for op in context.batches[1].operations) == 1
    assert fused[0].tensors["residual"].distribution == "replicated"
    assert all(op.tensors["residual"].distribution == "sequence_shard" for op in fused[1:])
    assert all(op.tensors["x"].distribution == "sequence_shard" and
               op.tensors["x"].shard_axis == 0 and
               op.tensors["norm"].distribution == "replicated" for op in fused)
    final = context.batches[1].operations[-1]
    assert final.name == "all_reduce_residual"
    assert final.tensors["residual"].distribution == "sequence_shard"
    assert final.tensors["out"].distribution == "replicated"
    assert context.spm_bytes <= 7 * 1024 * 1024
