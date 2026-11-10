import pytest
import torch

from xinfer import ops
from xinfer.ops import Contract, OpRegistry, OpSpec, use_ops
from xinfer.platforms.interface import Platform, PlatformCapabilities


def _platform(name="cuda-sm120", arch="sm120", device_type="cuda"):
    capabilities = PlatformCapabilities(device_type, arch, True, True, False, None, 170, 32 << 30)
    return Platform(name, f"{device_type}:0" if device_type == "cuda" else "cpu", capabilities)


def _registry():
    reg = OpRegistry()
    reg.define(OpSpec("double", "double(x)", lambda x: x * 2))
    return reg


TOL = Contract(numerics="tolerance")


def test_reference_is_default():
    table = _registry().build_table(_platform())
    assert table.selection("double").backend == "reference"
    assert table.double(3) == 6


def test_highest_priority_supported_backend_wins():
    reg = _registry()
    reg.register("double", backend="fast", contract=TOL, platforms=("sm120",), priority=1)(lambda x: "fast")
    reg.register("double", backend="thor", contract=TOL, platforms=("sm110",), priority=5)(lambda x: "thor")
    assert reg.build_table(_platform()).double(1) == "fast"
    assert reg.build_table(_platform("cuda-sm110", "sm110")).double(1) == "thor"
    assert reg.build_table(_platform("cpu", None, "cpu")).double(1) == 2


def test_reference_only_and_pinned_backends():
    reg = _registry()
    reg.register("double", backend="fast", contract=TOL, platforms=("sm120",), priority=1)(lambda x: "fast")
    assert reg.build_table(_platform(), reference_only=True).double(1) == 2
    assert reg.build_table(_platform(), backends={"double": "fast"}).double(1) == "fast"
    with pytest.raises(RuntimeError, match="does not support"):
        reg.build_table(_platform("cuda-sm110", "sm110"), backends={"double": "fast"})
    with pytest.raises(KeyError):
        reg.build_table(_platform(), backends={"nope": "fast"})


def test_backends_must_declare_their_numerics():
    reg = _registry()
    with pytest.raises(ValueError, match="bitwise or tolerance"):
        reg.register("double", backend="x", contract=Contract())
    with pytest.raises(ValueError, match="numerics"):
        Contract(numerics="close enough")
    reg.register("double", backend="x", contract=TOL)(lambda x: x)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("double", backend="x", contract=TOL)(lambda x: x)


def test_report_lists_selection_and_contract():
    rows = _registry().build_table(_platform()).report()
    assert rows == [
        dict(op="double", backend="reference", version=None, numerics="reference",
             deterministic=True, graph_safe=True)
    ]


def test_active_table_is_scoped():
    table = _registry().build_table(_platform())
    with pytest.raises(RuntimeError, match="no active op table"):
        ops.active_ops()
    with use_ops(table):
        assert ops.active_ops() is table
    with pytest.raises(RuntimeError):
        ops.active_ops()


def test_builtin_reference_ops():
    x = torch.randn(2, 4, 8, 16)
    table = ops.registry.build_table(_platform("cpu", None, "cpu"))
    w = torch.full((16,), 2.0)
    torch.testing.assert_close(table.rms_norm(x, w, 1e-6), 2 * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    # One rounding: BF16 input, FP32 weight -> scale in FP32, round once.
    xb, wf = x.bfloat16(), torch.randn(16)
    n = xb.float() * torch.rsqrt(xb.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.equal(table.rms_norm(xb, wf, 1e-6), (wf * n).bfloat16())
    # Grouped-query attention equals attention with explicitly repeated K/V.
    q, k, v = torch.randn(1, 4, 5, 8), torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
    expect = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), scale=0.5)
    torch.testing.assert_close(ops.reference.attention(q, k, v, scale=0.5), expect)  # used by the attention references


def test_varlen_attention_isolates_segments():
    torch.manual_seed(0)
    q, k, v = (torch.randn(7, 2, 8) for _ in range(3))
    cu_seqlens = torch.tensor([0, 3, 7])
    out = ops.reference.varlen_attention(q, k, v, cu_seqlens, scale=0.3)
    first = ops.reference.varlen_attention(q[:3], k[:3], v[:3], torch.tensor([0, 3]), scale=0.3)
    torch.testing.assert_close(out[:3], first)
    # Changing the second segment does not change the first.
    k2 = k.clone()
    k2[3:] += 1
    torch.testing.assert_close(ops.reference.varlen_attention(q, k2, v, cu_seqlens, scale=0.3)[:3], out[:3])


def test_slow_fallbacks_warn_on_cuda_only(caplog):
    references = OpRegistry()  # the operators without backends, whatever the process has loaded
    for name in ops.registry.names():
        references.define(ops.registry.spec(name))
    table = references.build_table(_platform())  # every operator on its reference
    # linear and silu have plain references; gemm_epilogue and qkv_rope_kvcache stand in for backend kernels
    stage_ops = {"prefill": ("gemm_epilogue", "linear"), "action": ("gemm_epilogue", "qkv_rope_kvcache", "silu")}
    with caplog.at_level("WARNING", logger="xinfer.ops"):
        assert ops.warn_slow_fallbacks("tiny", table, stage_ops, "cpu") == []
        assert not caplog.records
        assert ops.warn_slow_fallbacks("tiny", table, stage_ops, "cuda") == ["gemm_epilogue", "qkv_rope_kvcache"]
    assert "tiny: 2 operators have no fast kernel loaded" in caplog.text and "XINFER_XCOMPUTE_LIBRARY" in caplog.text
