"""Prefill lowering/liveness CPU contracts; no device numerical claim."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from xinfer.models.wallx2b.rpu.prefill import PrefillPlan, PrefillState, prefill_rope_tables
from xinfer.models.wallx2b.rpu.vision import VisionPlan
from wallx2b_plan_helpers import PlanContext, parameter, text_weights, vision_weights


def test_prefill_rope_keeps_logical_positions_and_pads_each_lane():
    length, bucket = 61, 64
    positions = np.stack((np.arange(length), np.arange(length) + 13, np.arange(length) + 27)).astype(np.int64)
    cosine, sine = prefill_rope_tables(positions, prefix_length=length, bucket=bucket)
    for row in (0, 60, 61, 63):
        for column in range(32):
            lane = 1 if column % 3 == 1 and column < 33 else 2 if column % 3 == 2 and column < 30 else 0
            position = row + (0, 13, 27)[lane]
            phase = position / (10000000 ** (2 * column / 64))
            assert cosine[row, column] == np.float16(math.cos(phase))
            assert sine[row, column] == np.float16(math.sin(phase))


@pytest.mark.parametrize("bucket", range(64, 513, 64))
def test_all_prefix_buckets_lower_without_unsafe_live_aliases(bucket):
    context = PlanContext()
    plan = PrefillPlan(context, SimpleNamespace(max_prefix_length=512), text_weights(context),
                       prefix_length=bucket - 3)
    assert plan.execution.collectives == 48
    assert len(plan.layer_operations) == 24
    assert len(plan.gdn_states) == len(plan.conv_states) == 18
    assert len(plan.key_caches) == len(plan.value_caches) == 6
    text_norms = [op for op in plan.operations if op.name == "rmsnorm_coeff_h2048_f16"]
    qk_norms = [op for op in plan.operations if op.name == "rmsnorm_f16"]
    assert len(text_norms) == 49 and len(qk_norms) == 12
    assert all(op.tensors["x"].shape == (bucket, 2048) for op in text_norms)
    assert all(op.tensors["x"].shape == (bucket, 256) for op in qk_norms)
    assert sum(op.name == "gdr_prefill_f16" for op in plan.operations) == 18
    assert sum(op.name == "causal_sdpa_prefix_spm_d256_f16" for op in plan.operations) == 6
    assert not any("lm_head" in op.name for op in plan.operations)
    assert context.spm_bytes <= 7 * 1024 * 1024
    assert plan.operation_count <= 1024
    assert sum(op.name == "gateup_n1536k2048" for op in plan.operations) == 24
    assert sum(op.name == "qkv_zba_n1056k2048" for op in plan.operations) == 18
    assert sum(op.name == "qkv_qgate_n1024k2048" for op in plan.operations) == 6
    assert not any(op.name.endswith("_m512") for op in plan.operations)
    for op in plan.operations:
        if op.name in ("conv_silu_prefill_f16", "gdr_prefill_f16"):
            assert op.integers == {"initial": 1, "valid_rows": bucket - 3}


def test_embedding_then_image_scatter_precedes_text_and_dynamic_ids_refresh():
    context = PlanContext()
    features = parameter(context, (217, 2048))
    plan = PrefillPlan(context, SimpleNamespace(max_prefix_length=512), text_weights(context),
                       prefix_length=405, image_features=features, image_starts=(20, 120, 220))
    inputs = plan.input_executions[0].operations
    assert inputs[0].name == "gather_embedding_tp8_f16"
    assert inputs[1].integers["mode"] == "corewise"
    assert [op.integers["mode"] for op in inputs[2:]] == ["shared"] * 3
    assert plan.operations[0].integers["mode"] == "broadcast"
    positions = np.broadcast_to(np.arange(405), (3, 405)).copy()
    plan.bind(positions, input_ids=np.arange(405, dtype=np.int64)[None])
    first = plan.cosine_ddr.uploaded.copy()
    plan.bind(positions + 10, input_ids=np.arange(405, dtype=np.int64)[None] + 1)
    assert not np.array_equal(first, plan.cosine_ddr.uploaded)
    np.testing.assert_array_equal(plan.ids.uploaded[:405], np.arange(405, dtype=np.int32) + 1)
    np.testing.assert_array_equal(plan.ids.uploaded[405:], 0)
    with pytest.raises(ValueError, match="admitted integer"):
        plan.bind(positions, input_ids=np.full((1, 405), 256277, dtype=np.int64))


def test_different_real_lengths_reuse_one_bucket_plan_and_patch_all_state_extents():
    context = PlanContext()
    features = parameter(context, (217, 2048))
    plan = PrefillPlan(context, SimpleNamespace(max_prefix_length=512), text_weights(context),
                       prefix_length=405, image_features=features, image_starts=(20, 120, 220))
    executions = (*plan.input_executions, *plan.executions)
    allocations, prepared, batches = context.spm_bytes, len(context.prepared), len(context.batches)
    ids_owner = plan.ids
    for length in (405, 432, 448, 386, 405):
        ids = np.arange(length, dtype=np.int32)[None]
        positions = np.broadcast_to(np.arange(length), (3, 1, length)).copy()
        plan.bind(positions, input_ids=ids)
        assert plan.prefix_length == length
        assert plan.ids is ids_owner and plan.ids.shape == (448,)
        np.testing.assert_array_equal(plan.ids.uploaded[:length], ids[0])
        np.testing.assert_array_equal(plan.ids.uploaded[length:], 0)
        valid_ops = [op for op in context.prepared if "valid_rows" in op.integers]
        assert len(valid_ops) == 37  # embedding plus 18 conv and 18 GDR
        assert all(op.integers["valid_rows"] == length for op in valid_ops)
        assert all(op.integers.get("initial", 1) == 1 for op in valid_ops)
    assert context.spm_bytes == allocations and len(context.prepared) == prepared
    assert len(context.batches) == batches
    assert (*plan.input_executions, *plan.executions) == executions
    with pytest.raises(ValueError, match="ceil64 bucket"):
        plan.bind(np.zeros((3, 449), dtype=np.int64), input_ids=np.zeros((1, 449), dtype=np.int32))
    assert plan.prefix_length == 405 and not plan._bound


def test_failed_extent_patch_cannot_replay_and_next_bind_restores_every_operation():
    context = PlanContext()
    plan = PrefillPlan(context, SimpleNamespace(max_prefix_length=512), text_weights(context),
                       prefix_length=405)
    operation = plan._valid_row_operations[5]
    original_rebind = operation.rebind

    def reject(**_):
        raise ValueError("synthetic native validation failure")

    operation.rebind = reject
    with pytest.raises(ValueError, match="native validation"):
        plan.bind(np.zeros((3, 440), dtype=np.int64), input_ids=np.zeros((1, 440), dtype=np.int32))
    assert not plan._bound and plan._valid_row_length is None
    operation.rebind = original_rebind
    plan.bind(np.zeros((3, 405), dtype=np.int64), input_ids=np.zeros((1, 405), dtype=np.int32))
    assert plan._bound and plan._valid_row_length == 405
    assert all(op.integers["valid_rows"] == 405 for op in plan._valid_row_operations)


def test_embedding_free_binding_infers_real_length_inside_existing_bucket():
    context = PlanContext()
    plan = PrefillPlan(context, SimpleNamespace(max_prefix_length=512), text_weights(context),
                       prefix_length=405, token_embedding=False)
    plan.bind(np.broadcast_to(np.arange(440), (3, 440)).copy())
    assert plan.prefix_length == 440
    assert len(plan._valid_row_operations) == 36
    assert all(op.integers["valid_rows"] == 440 for op in plan._valid_row_operations)


def test_vision_and_maximum_prefill_share_one_arena_with_bounded_parameter_staging():
    import torch

    context = PlanContext()
    arena = context.empty((3 * 512 * 2048,), distribution="local")
    config = SimpleNamespace(action_horizon=16, max_prefix_length=512)
    vision = VisionPlan(context, config, vision_weights(context), torch.zeros((2304, 1024), dtype=torch.float16),
                        workspace=arena)
    state = PrefillState(context)
    weights = text_weights(context)
    first = PrefillPlan(context, config, weights, prefix_length=512, workspace=arena, state=state,
                        image_features=vision.output_ddr, image_starts=(20, 120, 220))
    before = context.spm_bytes
    second = PrefillPlan(context, config, weights, prefix_length=511, workspace=arena, state=state,
                         image_features=vision.output_ddr, image_starts=(20, 120, 220))
    assert context.spm_bytes == before <= 7 * 1024 * 1024
    assert first.state is second.state and first.workspace is second.workspace


@pytest.mark.parametrize("window", [0, 16, True, 1.5])
def test_unadmitted_prefill_windows_reject_before_plan_allocation(window):
    context = PlanContext()
    with pytest.raises(ValueError, match=r"collective_window in \[1,15\]"):
        PrefillPlan(context, SimpleNamespace(max_prefix_length=512), {}, prefix_length=405,
                    collective_window=window)
    assert context.spm_bytes == 0 and not context.prepared and not context.batches


@pytest.mark.parametrize("window", [2, 4, 8, 15])
def test_explicit_windows_use_capability_and_preserve_default_input_batch(window):
    class CalibratedContext(PlanContext):
        def __init__(self):
            super().__init__()
            self.admissions = []
        def require_collective_window(self, requested):
            assert requested == window
            self.admissions.append(requested)
            return requested
        def batch(self, operations, *, collective_window=1):
            result = super().batch(operations)
            result.collective_window = collective_window
            return result
    context = CalibratedContext()
    features = parameter(context, (217, 2048))
    plan = PrefillPlan(context, SimpleNamespace(max_prefix_length=512), text_weights(context),
                       prefix_length=447, image_features=features, image_starts=(20,120,220),
                       collective_window=window)
    assert context.admissions == [window]
    assert plan.collective_window == plan.execution.collective_window == window
    assert plan.input_executions[0].collective_window == 1 and plan.execution.collectives == 48


def test_rope_tables_share_two_ddr_owners_across_buckets_and_bind_without_spm_upload():
    context = PlanContext()
    state = PrefillState(context)
    workspace = context.empty((3 * 512 * 2048,), distribution="local")
    weights = text_weights(context)
    config = SimpleNamespace(max_prefix_length=512)
    plans = [PrefillPlan(context, config, weights, prefix_length=length,
                         workspace=workspace, state=state) for length in (447, 449)]
    before = (context.spm_bytes, len(context.prepared), len(context.batches))
    assert state.cosine_ddr.nbytes + state.sine_ddr.nbytes == 65536
    assert state.cosine_ddr is not state.sine_ddr
    assert state.cosine_ddr.storage == state.sine_ddr.storage == "ddr"
    assert state.cosine_ddr.distribution == state.sine_ddr.distribution == "shared"
    for plan in plans:
        assert plan.cosine_ddr.parent is state.cosine_ddr
        assert plan.sine_ddr.parent is state.sine_ddr
        for operation, source, target in zip(plan.operations[:2],
                                             (plan.cosine_ddr, plan.sine_ddr),
                                             (plan.cosine, plan.sine)):
            assert operation.name == "copy_f16" and operation.integers == {"mode": "broadcast"}
            assert operation.tensors == {"input": source, "out": target}
        # Repeated A/B/A binding changes the live bucket view, not its owners or graph.
        for offset in (0, 7, 0):
            length = plan.bucket - 1
            positions = np.stack([np.arange(length) + offset + 13 * lane for lane in range(3)])
            expected = prefill_rope_tables(positions, prefix_length=length, bucket=plan.bucket)
            plan.bind(positions, input_ids=np.arange(length, dtype=np.int32))
            for view, golden in zip((plan.cosine_ddr, plan.sine_ddr), expected):
                np.testing.assert_array_equal(view.uploaded.view(np.uint16), golden.view(np.uint16))
                assert view.uploaded.flags.c_contiguous
            assert plan.cosine.uploaded is plan.sine.uploaded is None
            assert state.cosine.uploaded is state.sine.uploaded is None
    assert (context.spm_bytes, len(context.prepared), len(context.batches)) == before
