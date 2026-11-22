"""Independent CPU checks of TP ownership, lane order and FP16 representation."""

import gc
import hashlib
import weakref

import numpy as np
import pytest
import torch

from xinfer.models.wall_oss_1_0_nano.rpu import weights as weights_module
from xinfer.models.wall_oss_1_0_nano.rpu.config import CheckpointManifest
from xinfer.models.wall_oss_1_0_nano.rpu.weights import (
    WeightPreparer,
    action_down_weight,
    action_input_weight,
    column_shard,
    euler_schedule,
    full_qkv,
    gdn_prefill_scalar,
    gdn_qkv_zba,
    owner0_bias,
    pack_adaptive_rows,
    replicate_kv,
    row_shard,
    split_query_gate,
    zero_centered_gamma,
)


def test_row_shards_sum_to_one_global_linear_and_bias_once():
    weight = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32) / 16
    inputs = torch.arange(3 * 32, dtype=torch.float32).reshape(3, 32) / 8
    shards = row_shard(weight)
    pieces = torch.stack([inputs[:, c * 4 : (c + 1) * 4] @ shards[c].T for c in range(8)])
    bias = torch.arange(16, dtype=torch.float32)
    actual = (pieces + owner0_bias(bias)[:, None]).sum(0)
    torch.testing.assert_close(actual, inputs @ weight.T + bias, rtol=0, atol=0)


def test_column_shards_reconstruct_global_rows():
    weight = torch.arange(32 * 16).reshape(32, 16)
    assert torch.equal(column_shard(weight).reshape_as(weight), weight)
    with pytest.raises(ValueError, match="divisible"):
        row_shard(torch.zeros(16, 17))


def test_query_gate_split_uses_interleaved_heads():
    weight = torch.arange(8 * 2).reshape(8, 2)
    query, gate = split_query_gate(weight, heads=2, head_dim=2)
    assert torch.equal(query, weight[[0, 1, 4, 5]])
    assert torch.equal(gate, weight[[2, 3, 6, 7]])


def test_full_and_compact_kv_have_distinct_physical_ownership():
    qg = torch.arange(4096, dtype=torch.float16)[:, None]
    key = torch.arange(512, dtype=torch.float16)[:, None]
    value = -key
    full = full_qkv(qg, key, value)
    compact = full_qkv(qg, key, value, compact=True)
    assert tuple(full.shape) == (8, 1024, 1)
    assert tuple(compact.shape) == (8, 640, 1)
    for core in range(8):
        head = core // 4
        assert torch.equal(full[core, 512:768, 0], key[head * 256 : (head + 1) * 256, 0])
        assert torch.equal(compact[core, 512:576, 0], key[core * 64 : (core + 1) * 64, 0])
    assert torch.equal(replicate_kv(key), full[:, 512:768])


def test_gdn_qkv_zba_padding_is_inside_each_core_shard():
    qkv = torch.arange(6144, dtype=torch.float32)[:, None]
    z = torch.arange(2048, dtype=torch.float32)[:, None] + 10000
    b = torch.arange(16, dtype=torch.float32)[:, None] + 20000
    a = b + 100
    packed = gdn_qkv_zba(qkv, z, b, a)
    assert tuple(packed.shape) == (8, 1056, 1)
    for core in range(8):
        assert torch.equal(packed[core, :256], qkv[core * 256 : (core + 1) * 256])
        assert torch.equal(packed[core, 768:1024], z[core * 256 : (core + 1) * 256])
        assert torch.equal(packed[core, 1024:1026], b[core * 2 : (core + 1) * 2])
        assert torch.equal(packed[core, 1040:1042], a[core * 2 : (core + 1) * 2])
        assert not bool(packed[core, 1026:1040].any())
        assert not bool(packed[core, 1042:].any())


def test_gdr_scalars_match_public_sixteen_lane_contract():
    source = torch.linspace(-1, 1, 16, dtype=torch.float32)
    dt = gdn_prefill_scalar(source)
    neg_a = gdn_prefill_scalar(source, neg_exp=True)
    expected = -np.exp(source.half().numpy().astype(np.float32)).astype(np.float16)
    assert tuple(dt.shape) == tuple(neg_a.shape) == (8, 16)
    np.testing.assert_array_equal(neg_a[:, :2].numpy().reshape(16), expected)
    assert torch.equal(dt[:, :2].reshape(16), source.half())
    assert torch.equal(neg_a[:, 2:], -torch.ones(8, 14, dtype=torch.float16))
    assert not bool(dt[:, 2:].any())


def test_zero_centered_norm_preserves_the_source_half_boundary():
    source = torch.tensor([0.0004884], dtype=torch.float32)
    gamma = zero_centered_gamma(source)
    assert torch.equal(gamma, source.half() + 1)
    assert not torch.equal(gamma, (source + 1).half())


def test_rpu_profile_io_and_down_scales_are_compensated_before_half_store():
    packed = action_input_weight(torch.full((1024, 52), 4.0))
    assert tuple(packed.shape) == (1024, 64)
    assert torch.equal(packed[:, :52], torch.full((1024, 52), 1 / 16, dtype=torch.float16))
    assert not bool(packed[:, 52:].any())
    # The kernel's separate /4 establishes residual units /256.
    assert float(packed[0, 0] / 4) == 4 / 256
    down = action_down_weight(torch.full((1024, 2048), 4.0))
    assert tuple(down.shape) == (8, 1024, 256)
    assert torch.equal(down, torch.ones_like(down))
    with pytest.raises(ValueError, match="finite"):
        action_input_weight(torch.full((1024, 52), float("inf")))


def test_fp16_dt_is_fixed_for_eight_steps():
    times, delta_t = euler_schedule()
    np.testing.assert_array_equal(times.numpy(), np.linspace(0, 1, 9, dtype=np.float32) * np.float32(0.999))
    assert delta_t == float(np.float16(0.999 / 8))


def test_ada_odd_mlp_rows_restore_down_scale_before_the_single_half_boundary():
    condition = torch.zeros(1, 1024, dtype=torch.float32)
    weight = torch.zeros(3072, 1024, dtype=torch.float32)
    bias = torch.cat((torch.full((1024,), 0.25), torch.full((1024,), 0.5), torch.full((1024,), 2.0)))
    result = pack_adaptive_rows(condition, [(weight, bias)] * 49)
    assert tuple(result.shape) == (49, 3072)
    assert result.dtype == torch.float16
    assert torch.equal(result[:, :1024], torch.full((49, 1024), 1.25, dtype=torch.float16))
    assert torch.equal(result[:, 1024:2048], torch.full((49, 1024), 0.5, dtype=torch.float16))
    assert torch.equal(result[::2, 2048:], torch.full((25, 1024), 2 / 256, dtype=torch.float16))
    assert torch.equal(result[1::2, 2048:], torch.full((24, 1024), 8 / 256, dtype=torch.float16))


@pytest.fixture
def make_source_reader(tmp_path, monkeypatch):
    """Exercise source ownership separately from full checkpoint admission."""
    save_file = pytest.importorskip("safetensors.torch").save_file
    monkeypatch.setattr(CheckpointManifest, "assert_unchanged", lambda self: None)

    def make(tensors):
        save_file(tensors, str(tmp_path / "model.safetensors"))
        return WeightPreparer(CheckpointManifest(tmp_path, None, (), (), ()))

    return make


@pytest.mark.parametrize("half", [True, False])
def test_bounded_reader_hashes_fp32_and_returns_owned_storage(make_source_reader, monkeypatch, half):
    safetensors = pytest.importorskip("safetensors")
    source = torch.tensor([[0.0, -0.0, 0.0004884, 1.0007], [-2.25, 65504.0, 1e-7, -1e-7]])
    preparer = make_source_reader({"source": source})
    real_open = safetensors.safe_open
    mapped_sources, closed = [], []

    class TrackedOpen:
        def __init__(self, *args, **kwargs):
            self.manager = real_open(*args, **kwargs)

        def __enter__(self):
            self.handle = self.manager.__enter__()
            return self

        def get_tensor(self, key):
            tensor = self.handle.get_tensor(key)
            mapped_sources.append(tensor)
            return tensor

        def __exit__(self, *args):
            closed.append(True)
            return self.manager.__exit__(*args)

    monkeypatch.setattr(safetensors, "safe_open", TrackedOpen)
    result = preparer._read("source", half=half)
    assert closed == [True]
    assert result.dtype == (torch.float16 if half else torch.float32)
    assert result.data_ptr() != mapped_sources[0].data_ptr()
    expected = source.numpy().astype(np.float16 if half else np.float32)
    unsigned = np.uint16 if half else np.uint32
    np.testing.assert_array_equal(result.numpy().view(unsigned), expected.view(unsigned))
    info = preparer._source_info["source"]
    assert (info.key, info.shape, info.dtype) == ("source", (2, 4), "torch.float32")
    assert info.sha256 == hashlib.sha256(source.numpy().tobytes()).hexdigest()
    # The captured source deliberately keeps its real mmap alive. Mutating the
    # returned value must still leave that source untouched, including FP32.
    result[0, 0] = 8
    np.testing.assert_array_equal(mapped_sources[0].numpy().view(np.uint32), source.numpy().view(np.uint32))


def test_releasing_host_cache_preserves_callers_and_source_receipts(make_source_reader):
    tensors = {
        "model.action_processor.time_mlp_in.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "model.language_model.layers.0.input_layernorm_action.dense.bias": torch.arange(3, dtype=torch.float32),
        "model.visual.pos_embed.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
    }
    preparer = make_source_reader(tensors)
    first, second = preparer.host_tensors(), preparer.host_tensors()
    assert first.keys() == tensors.keys()
    assert first is not second and all(first[key] is second[key] for key in tensors)
    references = [weakref.ref(value) for value in first.values()]
    receipts = dict(preparer._source_info)
    preparer.release_host_tensors()
    preparer.release_host_tensors()
    assert preparer._host is None
    assert all(reference() is not None for reference in references)
    assert preparer._source_info == receipts
    del first, second
    gc.collect()
    assert all(reference() is None for reference in references)
    reloaded = preparer.host_tensors()
    assert preparer._source_info == receipts
    for key, value in reloaded.items():
        expected = tensors[key].half() if key == "model.visual.pos_embed.weight" else tensors[key]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize("unknown", [False, True])
def test_iterator_coverage_keeps_checkpoint_keys_after_attention_qkv(make_source_reader, monkeypatch, unknown):
    # One full-attention layer exercises the local q/k/v `keys` assignment.
    # Repeated layers are omitted to keep this real safetensors fixture small;
    # the final coverage check and the remaining source transforms still run.
    layer_passes = iter(((3,), (), ()))
    monkeypatch.setattr(weights_module, "range", lambda stop: next(layer_passes), raising=False)
    shapes = {
        "model.language_model.embed_tokens.weight": (8, 16),
        "model.language_model.norm.weight": (16,),
        "model.language_model.layers.3.input_layernorm.weight": (16,),
        "model.language_model.layers.3.post_attention_layernorm.weight": (16,),
        "model.language_model.layers.3.mlp.gate_up_proj.weight": (16, 16),
        "model.language_model.layers.3.mlp.down_proj.weight": (16, 16),
        "model.language_model.layers.3.self_attn.q_proj.weight": (4096, 16),
        "model.language_model.layers.3.self_attn.k_proj.weight": (512, 16),
        "model.language_model.layers.3.self_attn.v_proj.weight": (512, 16),
        "model.language_model.layers.3.self_attn.o_proj.weight": (16, 16),
        "model.language_model.layers.3.self_attn.q_norm.weight": (16,),
        "model.language_model.layers.3.self_attn.k_norm.weight": (16,),
        "model.action_processor.w1.weight": (1024, 52),
        "model.action_processor.action_proj_back.weight": (26, 1024),
        "model.action_processor.unused_host_vector": (2,),
        "model.visual.patch_embed.proj.weight": (1024, 1536),
        "model.visual.patch_embed.proj.bias": (1024,),
        "model.visual.merger.linear_fc1.weight": (16, 16),
        "model.visual.merger.linear_fc1.bias": (16,),
        "model.visual.merger.linear_fc2.weight": (16, 16),
        "model.visual.merger.linear_fc2.bias": (16,),
        "model.visual.merger.norm.weight": (16,),
        "model.visual.merger.norm.bias": (16,),
        "model.visual.pos_embed.weight": (2, 16),
    }
    if unknown:
        shapes["model.extra_unmapped.weight"] = (1,)
    preparer = make_source_reader({key: torch.full(shape, 0.25) for key, shape in shapes.items()})
    if unknown:
        with pytest.raises(ValueError, match="no preparation mapping.*model.extra_unmapped.weight"):
            list(preparer.iter_prepared())
    else:
        records = list(preparer.iter_prepared())
        assert "text.3.qkv" in {record.name for record in records}
        assert "model.visual.pos_embed.weight" not in preparer._source_info
        assert "model.action_processor.unused_host_vector" not in preparer._source_info
