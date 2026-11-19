"""Derived Ascend weight representations follow tensor ownership, not reusable addresses."""

import gc
import weakref

import pytest
import torch

from xinfer.ops.backends.ascend import _native, cann_projection, gemm


@pytest.fixture(autouse=True)
def derived_cache(monkeypatch):
    monkeypatch.setattr(_native, "_derived", {})
    monkeypatch.setattr(_native, "capturing", lambda device: False)


def _projection():
    # These ownership helpers need no CANN library or device allocation.
    return cann_projection.CannProjection.__new__(cann_projection.CannProjection)


def test_transposed_parameter_views_hit_the_same_copy_in_inference_mode():
    projection = _projection()
    weight = torch.nn.Parameter(torch.arange(24.0).reshape(6, 4), requires_grad=False)
    owner = weakref.ref(weight)
    with torch.inference_mode():
        first = projection._stable_contiguous(weight.t())
        for _ in range(3):
            assert projection._stable_contiguous(weight.t()) is first
        torch.testing.assert_close(first, weight.t(), rtol=0, atol=0)
    copied = weakref.ref(first)
    del first, weight
    gc.collect()
    assert owner() is None and copied() is None and not _native._derived


@pytest.mark.parametrize("mode", ["prepack", "splitk"])
def test_packed_projection_weights_do_not_keep_the_model_alive(monkeypatch, mode):
    monkeypatch.setattr(cann_projection, "_MX8_DOWN_MODE", mode)
    monkeypatch.setattr(cann_projection, "_MX8_DOWN_SPLITK", 4)
    projection = _projection()
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn(6, 8, generator=generator)
    x = torch.randn(3, 8, generator=generator)
    owner = weakref.ref(weight)
    if mode == "prepack":
        expected = torch.mm(x, weight.t().contiguous())
    else:
        packed = weight.t().reshape(4, -1, 6).contiguous()
        expected = torch.bmm(x.reshape(3, 4, -1).permute(1, 0, 2), packed).sum(0)
    with torch.no_grad():
        torch.testing.assert_close(projection._mx8_down(x, weight), expected, rtol=0, atol=0)
        entries = tuple(entry[1] for entry in _native._derived.values())
        torch.testing.assert_close(projection._mx8_down(x, weight), expected, rtol=0, atol=0)
        assert len(entries) == 1 and next(iter(_native._derived.values()))[1] is entries[0]
    del weight, entries
    gc.collect()
    assert owner() is None and not _native._derived


def test_existing_kn_storage_is_not_retained_through_a_cached_alias(monkeypatch):
    monkeypatch.setattr(cann_projection, "_MX8_DOWN_MODE", "prepack")
    weight = torch.nn.Parameter(torch.ones(8, 6), requires_grad=False)
    owner = weakref.ref(weight)
    _projection()._mx8_down(torch.ones(3, 8), weight.t())
    assert not _native._derived  # the requested layout is already a view; no derived allocation
    del weight
    gc.collect()
    assert owner() is None


def test_factor_cache_cannot_reuse_old_values_for_new_owners_at_the_same_addresses():
    q, k = torch.ones(8), torch.full((8,), 2.0)
    q_storage, k_storage = q.untyped_storage(), k.untyped_storage()
    old_q, old_k = weakref.ref(q), weakref.ref(k)
    first = gemm._ft_qkv_factor_pair(q, k)
    assert gemm._ft_qkv_factor_pair(q, k)[0] is first[0]
    pointers = q.data_ptr(), k.data_ptr()
    del q, k
    gc.collect()
    assert old_q() is old_k() is None and not _native._derived
    q = torch.empty(0).set_(q_storage, 0, (8,), (1,)).fill_(5)
    k = torch.empty(0).set_(k_storage, 0, (8,), (1,)).fill_(7)
    assert pointers == (q.data_ptr(), k.data_ptr())
    second = gemm._ft_qkv_factor_pair(q, k)
    assert second[0] is not first[0] and second[1] is not first[1]
    assert torch.equal(second[0], torch.full((8,), 6.0))
    assert torch.equal(second[1], torch.full((8,), 8.0))


def test_derived_forms_and_device_layouts_have_independent_entries():
    weight = torch.ones(4, 8)
    first = _native.derived("first", weight, weight.clone)
    other = _native.derived("other", weight, lambda: weight + 1)
    shape = _native.derived("first", weight[:2], lambda: weight[:2].clone())
    meta = torch.ones(4, 8, device="meta")
    on_meta = _native.derived("first", meta, meta.clone)
    assert first is not other and shape.shape == (2, 8)
    assert first.device.type == "cpu" and on_meta.device.type == "meta"
    assert len(_native._derived) == 4
    weight.add_(3)
    updated = _native.derived("first", weight, weight.clone)
    assert updated is not first and torch.equal(updated, torch.full((4, 8), 4.0))


def test_inference_weights_cache_while_their_real_owner_lives():
    with torch.inference_mode():
        q = torch.nn.Parameter(torch.ones(8), requires_grad=False)
        k = torch.nn.Parameter(torch.full((8,), 2.0), requires_grad=False)
        first = gemm._ft_qkv_factor_pair(q, k)
        assert gemm._ft_qkv_factor_pair(q, k)[0] is first[0]
        owner = weakref.ref(q)
        del q
        gc.collect()
        assert owner() is None and len(_native._derived) == 1


def test_inference_views_without_a_base_cannot_outlive_their_own_identity(monkeypatch):
    projection = _projection()
    with torch.inference_mode():
        weight = torch.nn.Parameter(torch.arange(24.0).reshape(6, 4), requires_grad=False)
        view = weight.t()
        assert view._base is None
        first = projection._stable_contiguous(view)
        assert projection._stable_contiguous(view) is first
        del view
        gc.collect()
        assert not _native._derived
        # The storage is still live, but no reliable logical owner links the new view to the old one.
        weight.add_(10)
        second = projection._stable_contiguous(weight.t())
        assert torch.equal(second, first + 10)
        assert not _native._derived
        monkeypatch.setattr(_native, "capturing", lambda device: True)
        with pytest.raises(RuntimeError, match="retain its owner"):
            projection._stable_contiguous(weight.t())


def test_smallm_gateup_pack_follows_weight_lifetime_and_version(monkeypatch):
    from xinfer.ops.backends.ascend import action_gemm_m16

    packs = []
    monkeypatch.setattr(action_gemm_m16, "_launch", lambda name, tensors: packs.append(tensors[1]))
    weight = torch.nn.Parameter(torch.arange(32, dtype=torch.bfloat16).reshape(8, 4), requires_grad=False)
    x = torch.zeros(16, 4, dtype=torch.bfloat16)
    action_gemm_m16.gateup(x, weight)
    action_gemm_m16.gateup(x, weight)
    assert packs[0] is packs[1]
    previous = packs[0].clone()
    weight.add_(2)
    action_gemm_m16.gateup(x, weight)
    assert torch.equal(packs[-1], previous + 2)
    owner, packed = weakref.ref(weight), weakref.ref(packs[-1])
    packs.clear()
    del weight
    gc.collect()
    assert owner() is None and packed() is None and not _native._derived


def test_gdn_gate_preparation_follows_weight_lifetime_and_version(monkeypatch):
    from xinfer.ops.backends.ascend import gdn_vendor

    biases = []

    def launch(name, tensors, scalars):
        biases.append(tensors[1].clone())
        tensors[2].zero_()
        tensors[3].fill_(1)
        return True

    monkeypatch.setattr(gdn_vendor, "_launch", launch)
    log = torch.nn.Parameter(torch.zeros(16), requires_grad=False)
    bias = torch.nn.Parameter(torch.zeros(16, dtype=torch.bfloat16), requires_grad=False)
    values = torch.zeros(1, 2080, dtype=torch.bfloat16)
    first = gdn_vendor.gdrprep(values, log, bias, 1, 16, 128, torch.bfloat16)[1]
    log.add_(1)
    bias.add_(2)
    second = gdn_vendor.gdrprep(values, log, bias, 1, 16, 128, torch.bfloat16)[1]
    torch.testing.assert_close(second, first * torch.exp(torch.tensor(1.0)))
    assert torch.equal(biases[1], biases[0] + 2)
    owners = weakref.ref(log), weakref.ref(bias)
    del log, bias
    gc.collect()
    assert all(owner() is None for owner in owners) and not _native._derived


@pytest.mark.parametrize("helper", ["action", "bias", "fused_rmsnorm", "buffered_add_rmsnorm"])
def test_norm_and_bias_forms_follow_owner_lifetime_and_inplace_updates(helper):
    from xinfer.ops.backends.ascend import cann_action, fused_rmsnorm, buffered_add_rmsnorm

    action = cann_action.CannAction.__new__(cann_action.CannAction)
    convert = {"action": action._one_plus, "bias": cann_projection._bias_fp32,
               "fused_rmsnorm": fused_rmsnorm._weight32, "buffered_add_rmsnorm": buffered_add_rmsnorm._weight32}[helper]
    weight = torch.nn.Parameter(torch.arange(8, dtype=torch.bfloat16), requires_grad=False)
    first = convert(weight)
    assert convert(weight) is first
    pointer = weight.data_ptr()
    weight.add_(2)
    updated = convert(weight)
    assert weight.data_ptr() == pointer and updated is not first
    torch.testing.assert_close(updated, first + 2, rtol=0, atol=0)
    owner, copied = weakref.ref(weight), weakref.ref(updated)
    del weight, first, updated
    gc.collect()
    assert owner() is None and copied() is None and not _native._derived


def test_existing_fp32_norm_and_bias_forms_are_not_cached_as_their_own_owner():
    from xinfer.ops.backends.ascend import fused_rmsnorm, buffered_add_rmsnorm

    weight = torch.ones(8)
    for convert in (cann_projection._bias_fp32, fused_rmsnorm._weight32, buffered_add_rmsnorm._weight32):
        assert convert(weight) is weight
    assert not _native._derived
