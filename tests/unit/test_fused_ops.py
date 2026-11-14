"""Fused Wall-OSS and Wall-X 2B operators against unfused torch (CPU) and xcompute kernels (GPU)."""

import os

import pytest
import torch
import torch.nn.functional as F

from xinfer import ops
from xinfer.models.wallx.optimized import column_ranges, q_k_v_gate
from xinfer.ops import reference
from xinfer.ops.reference import attn, flow, gemm, norm, rope
from xinfer.platforms import resolve_platform

BF16 = torch.bfloat16


def _randn(*shape, scale=1.0, dtype=BF16, device="cpu", generator=None):
    return (torch.randn(*shape, generator=generator) * scale).to(device=device, dtype=dtype)


def test_gemm_epilogue_rounds_the_gated_branch_before_the_residual():
    g = torch.Generator().manual_seed(0)
    x, w = torch.randn(6, 16, generator=g).bfloat16(), torch.randn(8, 16, generator=g).bfloat16()
    residual, gate = torch.randn(6, 8, generator=g).bfloat16(), torch.randn(3, 8, generator=g).bfloat16()
    rows = torch.tensor([0, 2, 1, 1, 0, 2])
    y = x.float() @ w.float().t()
    want = (residual.float() + (y * gate[rows].float()).bfloat16().float()).bfloat16()
    assert torch.equal(gemm.gemm_epilogue(x, w, residual=residual, residual_gate=gate, row_class=rows), want)
    scale = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    out = gemm.gemm_epilogue(x, w, row_scale=scale, activation="gelu_tanh")
    assert torch.equal(out[1], torch.zeros(8, dtype=torch.bfloat16))
    assert torch.equal(out[0], F.gelu(y[0], approximate="tanh").bfloat16())


def test_gemm_glu_epilogue_is_ada_rms_norm_then_silu_mul():
    g = torch.Generator().manual_seed(1)
    x, w = torch.randn(5, 16, generator=g), torch.randn(2 * 8, 16, generator=g)
    table, rows = torch.randn(3, 48, generator=g) * 0.3, torch.tensor([2, 0, 1, 1, 2])
    normed = reference.ada_rms_norm(x, table[rows, :16], table[rows, 16:32], 1e-6)
    y = normed @ w.t()
    torch.testing.assert_close(gemm.gemm_glu_epilogue(x, w, modulation=table, row_class=rows, eps=1e-6),
                               F.silu(y[:, :8]) * y[:, 8:])


def test_transposed_gemm_weight_on_an_xcompute_build_without_kn(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from xinfer.ops.backends.cuda import gemm

    prepack = Mock(side_effect=ValueError("prepack_weight: layout= only takes 'nk_k64blk' (a BF16 gemm weight)"))
    monkeypatch.setattr(gemm, "xc", SimpleNamespace(prepack_weight=prepack))
    monkeypatch.setattr(gemm, "_supports_kn", True)
    for n in (4, 6):
        weight = torch.arange(3 * n).reshape(3, n).t()
        dense = gemm._kn(weight)
        assert dense.is_contiguous() and torch.equal(dense, weight)
    assert prepack.call_count == 1


def test_a_packed_weight_is_not_reused_for_another_weight_at_the_same_address(monkeypatch):
    """The packing cache is keyed by the weight's address; its entry goes when the weight is freed, so a weight
    allocated later at that address (another model loaded in the same process) gets its own packing."""
    import gc
    import types
    import weakref

    from xinfer.ops.backends.cuda import gemm

    packings = []

    def prepack_weight(weight, op):  # a packing is a copy: it keeps no reference to the weight
        packings.append(weakref.ref(weight))
        return types.SimpleNamespace(source=weakref.ref(weight))

    package = types.SimpleNamespace(
        prepack_weight=prepack_weight,
        selected_backend=lambda op, x, w, **options: "fast" if hasattr(w, "source") else "generic")
    monkeypatch.setattr(gemm, "xc", package)
    monkeypatch.setattr(gemm, "_packed", {})
    x = torch.zeros(2, 8)
    w = torch.ones(4, 8)
    assert gemm._listed_weight("gemm", x, w).source() is w
    assert gemm._listed_weight("gemm", x, w[:2]).source() is not w  # a view of another shape: its own packing
    assert gemm._listed_weight("gemm", x, w).source() is w and len(packings) == 2  # cached
    del w
    gc.collect()
    assert gemm._packed == {}  # the weight is gone: so are its entries


def test_qkv_rope_epilogue_rotates_q_and_k_halves():
    g = torch.Generator().manual_seed(2)
    x, w, b = torch.randn(4, 16, generator=g), torch.randn(3 * 2 * 8, 16, generator=g), torch.randn(48, generator=g)
    angles = torch.rand(4, 4, generator=g)
    out = gemm.qkv_rope_epilogue(x, w, b, angles.cos(), angles.sin(), 2, 8).view(4, 3, 2, 8)
    y = (x @ w.t() + b).view(4, 3, 2, 8)
    c, s = angles.cos()[:, None], angles.sin()[:, None]
    torch.testing.assert_close(out[:, 0, :, :4], y[:, 0, :, :4] * c - y[:, 0, :, 4:] * s)
    torch.testing.assert_close(out[:, 1, :, 4:], y[:, 1, :, 4:] * c + y[:, 1, :, :4] * s)
    torch.testing.assert_close(out[:, 2], y[:, 2])


def test_flow_velocity_euler_is_the_readout_and_masked_euler_step():
    g = torch.Generator().manual_seed(9)
    x, h = torch.randn(1, 6, 4, generator=g), torch.randn(1, 6, 16, generator=g).bfloat16()
    weight = torch.randn(4, 16, generator=g) * 0.2
    dof = torch.tensor([1.0, 1.0, 0.0, 1.0]).expand(1, 6, 4)
    padding, prefix = torch.randn(1, 6, 4, generator=g), torch.randn(1, 6, 4, generator=g)
    p = h.float() @ weight.t()
    torch.testing.assert_close(flow.flow_velocity_euler(x, h, weight, 0.1), x + 0.1 * p)
    want = x + 0.1 * (padding * (1 - dof) + p * dof)
    want[:, :2] = prefix[:, :2]
    torch.testing.assert_close(flow.flow_velocity_euler(x, h, weight, 0.1, dof, padding, torch.tensor([2]), prefix),
                               want)


def test_qkv_rope_kvcache_is_the_biased_gemm_rope_and_kv_write():
    g = torch.Generator().manual_seed(13)
    m, k, heads, kv, d, start = 4, 32, 2, 1, 16, 3
    x, w, b = torch.randn(m, k, generator=g).bfloat16(), (torch.randn(64, k, generator=g) * 0.2).bfloat16(), \
        torch.randn(64, generator=g).bfloat16()
    angles = torch.rand(m, 8, generator=g)
    caches = torch.zeros(2, kv, start + m, d, dtype=torch.bfloat16)
    q = gemm.qkv_rope_kvcache(x, w, b, angles.cos(), angles.sin(), caches[0], caches[1], start, heads, kv, d)
    y = (x.float() @ w.float().t() + b.float()).view(m, heads + 2 * kv, d)
    c, s = angles.cos()[:, None], angles.sin()[:, None]
    rotated = torch.cat((y[..., :8] * c - y[..., 8:] * s, y[..., 8:] * c + y[..., :8] * s), dim=-1)
    assert torch.equal(q, rotated[:, :heads].reshape(m, -1).bfloat16())
    assert torch.equal(caches[0, :, start:], rotated[:, heads:heads + kv].transpose(0, 1).bfloat16())
    assert torch.equal(caches[1, :, start:], y[:, heads + kv:].transpose(0, 1).bfloat16())
    assert not caches[:, :, :start].any()


def test_add_rms_norm_is_the_rounded_sum_and_its_norm():
    g = torch.Generator().manual_seed(14)
    x, residual, w = _randn(3, 16, generator=g), _randn(3, 16, generator=g), torch.rand(16, generator=g) + 0.5
    total, normed = norm.add_rms_norm(x, residual, w, 1e-6)
    assert torch.equal(total, residual + x) and torch.equal(normed, reference.rms_norm(residual + x, w, 1e-6))


def test_an_xcompute_package_of_another_build_is_not_used(monkeypatch, tmp_path):
    """A package whose library lacks the unified entries' functions (another build, such as the MXFP8 one) would fail
    at the first GEMM: the fused operators stay on their references."""
    import sys
    import types

    from xinfer.ops.backends import _xcompute as xcompute

    path = tmp_path / "xcompute.so"
    path.write_bytes(b"")
    package = types.ModuleType("xcompute")
    package.KERNEL_LIB_PATH = str(path)
    for name in xcompute.UNIFIED:
        setattr(package, name, object())
    monkeypatch.setitem(sys.modules, "xcompute", package)
    monkeypatch.setenv("XCOMPUTE_LIBRARY_PATH", str(path))

    class Library:
        def __init__(self, functions):
            self.path, self.functions = str(path), functions

        def has_function(self, name):
            return name in self.functions

        def function(self, name):
            return lambda: "120"  # a CUDA-style build: the arch probe answers with digits

    assert xcompute._python_package(Library(xcompute.LIBRARY_ENTRIES)) is package
    assert xcompute._python_package(Library(())) is None


def _fake_package(monkeypatch, tmp_path):
    """A stand-in ``xcompute`` package whose library is ``tmp_path/xcompute.so``, importable as ``xcompute``."""
    import importlib.machinery
    import sys
    import types

    from xinfer.ops.backends import _xcompute as xcompute

    path = tmp_path / "xcompute.so"
    path.write_bytes(b"")
    package = types.ModuleType("xcompute")
    package.__spec__ = importlib.machinery.ModuleSpec("xcompute", None)
    package.KERNEL_LIB_PATH = str(path)
    for name in xcompute.UNIFIED:
        setattr(package, name, object())
    monkeypatch.setitem(sys.modules, "xcompute", package)
    monkeypatch.setenv("XCOMPUTE_LIBRARY_PATH", str(path))
    return package


@pytest.mark.parametrize("archs,platforms", [("110,120", ("sm110", "sm120")), ("120", ("sm120",)), ("", ())])
def test_xcompute_runs_on_the_architectures_the_build_targets(archs, platforms):
    from xinfer.ops.backends import _xcompute

    class Library:
        def function(self, name):
            assert name == "build_target_archs"
            return lambda: archs

    assert _xcompute.target_archs(Library()) == platforms


def test_the_cuda_modules_register_the_xcompute_operators_when_xcompute_imports(monkeypatch):
    """With xcompute imported, every fused operator is registered on its build's architectures; without it, none."""
    import importlib
    import types

    from xinfer.ops.backends import _xcompute
    from xinfer.ops.registry import BUILTIN_MODULES, OpRegistry

    registry_module = importlib.import_module("xinfer.ops.registry")  # ``xinfer.ops.registry`` is the instance

    def registered(xcompute):
        """A registry of every operator after importing the CUDA modules with ``_xcompute.XCOMPUTE = xcompute``."""
        registry = OpRegistry()
        for name in ops.registry.names():
            registry.define(ops.registry.spec(name))
        monkeypatch.setattr(registry_module, "registry", registry)
        monkeypatch.setattr(_xcompute, "XCOMPUTE", xcompute)
        monkeypatch.setattr(_xcompute, "xc", xcompute["requires"])
        monkeypatch.setattr(_xcompute, "library", None)
        for module in BUILTIN_MODULES:
            importlib.reload(importlib.import_module(module))
        return registry

    package = types.ModuleType("xcompute")
    package.denoise_output_euler_backend = object()
    try:
        registry = registered(dict(requires=package, platforms=("sm120",), version="sha256:0"))
        names = registry.backend_ops("cuda:xcompute")
        assert set(names) == set(FUSED)
        # Filter by implementation: the Ascend backend registers some of the same operators after the CUDA modules.
        assert {i.platforms for name in names for i in registry.implementations(name)
                if i.name == "cuda:xcompute"} == {("sm120",)}
        # A build may have the base GEMM kernels and only some of the 2B kernels.
        package.denoise_input_embed = object()
        enabled = registered(dict(requires=package, platforms=("sm120",), version="sha256:0"))
        assert set(enabled.backend_ops("cuda:xcompute")) == set(FUSED) | {"flow_input_embed"}
        assert registered(dict(requires=None, platforms=(), version=None)).backend_ops("cuda:xcompute") == []
    finally:
        monkeypatch.undo()
        for module in BUILTIN_MODULES:  # register the modules as imported into the process-wide registry again
            importlib.reload(importlib.import_module(module))


@pytest.mark.parametrize("gated,ungated,expected", [(True, True, "gated"), (False, True, "separate"),
                                                    (False, False, "reference")])
def test_gated_kvcache_selects_an_available_kernel(monkeypatch, gated, ungated, expected):
    from types import SimpleNamespace

    from xinfer.ops.backends.cuda import attn as cuda

    def probe(*args, **options):
        if not (gated if "out_gate" in options else ungated):
            raise ValueError("no listed kernel")

    gate = torch.zeros(1, 2, 4)
    xc = SimpleNamespace(attention_kvcache_backend=probe,
                         attention_kvcache=lambda *args, **opts: "gated" if "out_gate" in opts else gate,
                         sigmoid_gate_mul_inplace=lambda *args, **opts: "separate")
    monkeypatch.setattr(cuda, "xc", xc)
    monkeypatch.setattr(cuda, "_listed", {})
    monkeypatch.setattr(attn, "gated_kvcache_attention", lambda *args, **opts: "reference")
    q, cache = gate.view(1, 2, 2, 2), torch.zeros(1, 4, 1, 2)
    assert cuda.gated_kvcache_attention(q, cache, cache, torch.tensor([2]), 2, gate) == expected


def test_cuda_operators_use_reference_when_the_build_rejects_a_shape(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from xinfer.ops.backends.cuda import attn as cuda_attn
    from xinfer.ops.backends.cuda import flow as cuda_flow

    reject = Mock(side_effect=ValueError("no listed kernel"))
    launch = Mock(side_effect=AssertionError("kernel was not admitted"))
    xc = SimpleNamespace(gated_delta_rule_prefill_backend=reject, gated_delta_rule_prefill=launch,
                         denoise_input_embed_backend=reject, denoise_input_embed=launch,
                         denoise_output_euler_backend=reject, denoise_output_euler=launch)
    monkeypatch.setattr(cuda_attn, "xc", xc)
    monkeypatch.setattr(cuda_attn, "_listed", {})
    monkeypatch.setattr(cuda_flow, "xc", xc)
    monkeypatch.setattr(attn, "gated_delta_net", lambda *args: "delta reference")
    qkv, zba = torch.zeros(1, 3, 384, dtype=BF16), torch.zeros(1, 3, 130, dtype=BF16)
    assert cuda_attn.gated_delta_net(qkv, zba, None, None, None, None, 1, 1, 128, 128, 1e-6) == "delta reference"
    x, h = torch.arange(6.).reshape(1, 3, 2), torch.ones(1, 3, 4, dtype=BF16)
    embed, readout = torch.ones(1, 4, 2), torch.ones(2, 4)
    for _ in range(2):
        assert torch.equal(cuda_flow.flow_input_embed(x, embed), flow.flow_input_embed(x, embed))
        assert torch.equal(cuda_flow.flow_velocity_euler(x, h, readout, 0.25),
                           flow.flow_velocity_euler(x, h, readout, 0.25))
    assert reject.call_count == 3
    launch.assert_not_called()


def test_flow_embed_on_a_build_without_a_capability_query(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from xinfer.ops.backends.cuda import flow as cuda

    launch = Mock(return_value="native")
    monkeypatch.setattr(cuda, "xc", SimpleNamespace(denoise_input_embed=launch))
    x, weight = torch.ones(1, 2, 2), torch.ones(1, 4, 2)
    assert cuda.flow_input_embed(x, weight) == "native"
    launch.side_effect = ValueError("denoise_input_embed: no white-list entry accepts this shape")
    assert torch.equal(cuda.flow_input_embed(x, weight), flow.flow_input_embed(x, weight))
    launch.side_effect = ValueError("invalid weights")
    with pytest.raises(ValueError, match="invalid weights"):
        cuda.flow_input_embed(x, weight)


def test_the_xcompute_library_comes_from_the_installed_package(monkeypatch, tmp_path):
    from xinfer.ops.backends import _xcompute

    package = _fake_package(monkeypatch, tmp_path)
    assert _xcompute._package_library() == package.KERNEL_LIB_PATH
    monkeypatch.setattr(_xcompute.importlib.util, "find_spec", lambda name: None)
    assert _xcompute._package_library() is None


def test_xinfer_xcompute_0_skips_xcompute(monkeypatch):
    from xinfer.ops.backends import _xcompute

    calls = []
    monkeypatch.setattr(_xcompute, "_package_library", lambda: calls.append(True))
    monkeypatch.setenv("XINFER_XCOMPUTE", "0")
    assert _xcompute._import() is None and calls == []


# --------------------------------------------------------------------------- xcompute (GPU, RTX 5090)
XCOMPUTE = os.environ.get("XINFER_XCOMPUTE_LIBRARY")
gpu = pytest.mark.skipif(not (XCOMPUTE and torch.cuda.is_available()
                              and torch.cuda.get_device_capability() == (12, 0)),
                         reason="xcompute + an SM120 GPU required")
FUSED = ("gemm_epilogue", "gemm_glu_epilogue", "qkv_rope_epilogue", "qkv_rope_kvcache", "rms_norm",
         "add_rms_norm", "prefix_kvcache_attention", "flow_velocity_euler")


@pytest.fixture(scope="module")
def xc():
    names = (ops.registry.load_plugins(), ops.registry.backend_ops("cuda:xcompute"))[1]
    if "gemm_epilogue" not in names:
        pytest.skip("needs a build of xcompute main and the xcompute Python package of the same build")
    missing = set(FUSED) - set(names)
    assert not missing, missing
    return ops.registry.build_table(resolve_platform("cuda"))


def _check(table, name, args, *, atol=0.0, bitwise=False, equal=False, relative=None, kwargs=None):
    """The selected implementation against the reference on the same inputs (clones for in-place operators).

    ``bitwise``: the contract says so and the outputs are equal; ``equal``: equal on these inputs under a tolerance
    contract; ``relative``: bound on the relative norm error instead of elementwise ``atol``.
    """
    kwargs = kwargs or {}
    clone = lambda values: [a.clone() if torch.is_tensor(a) else a for a in values]  # noqa: E731
    got = getattr(table, name)(*clone(args), **kwargs)
    want = ops.registry.spec(name).reference(*clone(args), **kwargs)
    got, want = (tuple(x) if isinstance(x, tuple) else (x,) for x in (got, want))
    assert table.selection(name).library == "xcompute"
    assert table.selection(name).contract.numerics == ("bitwise" if bitwise else "tolerance")
    for a, b in zip(got, want):
        if bitwise or equal:
            assert torch.equal(a, b), name
        elif relative is not None:
            assert float((a.float() - b.float()).norm() / b.float().norm()) < relative, name
        else:
            torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=0.02, msg=name)


@gpu
@pytest.mark.gpu
def test_xcompute_matches_the_references_at_wall_oss_shapes(xc):
    """Wall-OSS: ViT width 1280 (16 heads of 80, GLU with bias), expert 0 width 2048 and expert 1 width 1024 with
    16/2/128 attention heads, QKV bias and M-RoPE; FP32 norm weights."""
    g = torch.Generator().manual_seed(17)
    r = lambda *s, scale=1.0, dtype=BF16: _randn(*s, scale=scale, dtype=dtype, device="cuda", generator=g)  # noqa: E731
    for m, k in ((2048, 1280), (768, 2048), (16, 1024)):
        x, residual, w = r(m, k), r(m, k), torch.rand(k, device="cuda") + 0.5
        _check(xc, "rms_norm", (x, w, 1e-6), relative=0.01)
        _check(xc, "add_rms_norm", (x, residual, w, 1e-6), relative=0.01)
    _check(xc, "gemm_glu_epilogue", (r(2048, 1280), r(6912, 1280, scale=1 / 36)), kwargs=dict(bias=r(6912)),
           relative=0.01)
    for m, k in ((768, 2048), (16, 1024)):
        angles = torch.rand(m, 64, device="cuda") * 20
        caches = [torch.zeros(2, 2, m + 32, 128, dtype=BF16, device="cuda") for _ in range(2)]
        args = (r(m, k), r(2560, k, scale=k ** -0.5), r(2560), angles.cos(), angles.sin())
        got = xc.qkv_rope_kvcache(*args, caches[0][0], caches[0][1], 32, 16, 2, 128)
        want = ops.registry.spec("qkv_rope_kvcache").reference(*args, caches[1][0], caches[1][1], 32, 16, 2, 128)
        assert xc.selection("qkv_rope_kvcache").library == "xcompute"
        for a, b in ((got, want), (caches[0], caches[1])):
            assert float((a.float() - b.float()).norm() / b.float().norm()) < 0.01
    p, m = 768, 16
    columns = torch.arange(p + m, device="cuda")
    lts = torch.where((columns < 700) | (columns >= p), m, 0)  # prompt padding [700, 768) hidden
    flash_mask = torch.stack((lts, torch.where(columns >= p + 3, 3, 0)), -1).to(torch.int32)[None, None]
    _check(xc, "prefix_kvcache_attention", (r(1, m, 16, 128), r(1, p + m, 2, 128), r(1, p + m, 2, 128),
                                            torch.tensor([p], dtype=torch.int32, device="cuda"), p),
           kwargs=dict(flash_mask=flash_mask.contiguous()), relative=0.01)


@gpu
@pytest.mark.gpu
def test_xcompute_gemm_packs_a_weight_only_a_packed_layout_lists(xc):
    """At 16 rows xcompute lists the gate/up GEMM only for a packed weight: the binding packs the raw weight and runs
    the listed kernel."""
    import xcompute

    g = torch.Generator().manual_seed(17)
    x = _randn(16, 1024, dtype=BF16, device="cuda", generator=g)
    w = _randn(4096, 1024, scale=1 / 32, dtype=BF16, device="cuda", generator=g)
    packed = xcompute.prepack_weight(w, op="gemm_glu")
    assert xcompute.selected_backend("gemm_glu", x, w, activation="silu") == "generic"
    assert xcompute.selected_backend("gemm_glu", x, packed, activation="silu") != "generic"
    assert torch.equal(xc.gemm_glu_epilogue(x, w), xcompute.gemm_glu(x, packed, activation="silu"))


def test_add_zero_centered_norm_returns_the_sum_and_its_norm():
    g = torch.Generator().manual_seed(3)
    x, residual, w = _randn(4, 8, generator=g), _randn(4, 8, generator=g), _randn(8, generator=g)
    total, normed = norm.add_zero_centered_rms_norm(x, residual, w, 1e-6)
    assert torch.equal(total, residual + x)
    assert torch.equal(normed, reference.rms_norm(total, 1 + w.float(), 1e-6))


def test_gated_delta_net_is_the_reference_mixer():
    from xinfer.models.layers.qwen3_5 import GatedDeltaNet

    g = torch.Generator().manual_seed(5)
    torch.manual_seed(5)
    with ops.use_ops(ops.registry.build_table(resolve_platform("cpu"))):
        mixer = GatedDeltaNet(16, 2, 4, 8, 8, 4, 1e-6).to_empty(device="cpu")
    for param in mixer.parameters():
        param.data.copy_(torch.randn(param.shape, generator=g) * 0.3)
    mixer.A_log.data, mixer.norm_weight.data = mixer.A_log.data.abs(), mixer.norm_weight.data + 1
    x = torch.randn(1, 70, 16, generator=g)
    expected = mixer(x, torch.ones(1, 70, 1))
    zba = torch.cat((mixer.in_z(x), mixer.in_b(x), mixer.in_a(x)), -1)
    out = attn.gated_delta_net(mixer.in_qkv(x), zba, mixer.conv_weight[:, 0], mixer.A_log, mixer.dt_bias,
                                mixer.norm_weight, 2, 4, 8, 8, 1e-6)
    torch.testing.assert_close(mixer.out_proj(out), expected)


def test_vision_tables_follow_the_layout_of_the_reference_tower():
    from types import SimpleNamespace

    from xinfer.models.layers.rotary import vision_rotary_angles
    from xinfer.models.layers.vision.qwen3_5_vit import vision_layout

    g = torch.Generator().manual_seed(0)
    grids = [(1, 4, 6), (2, 6, 4)]
    layout = vision_layout(grids, SimpleNamespace(spatial_merge_size=2, grid_side=5))
    inv_freq = 1.0 / 10000 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8)
    assert torch.equal(rope.rot_pos(inv_freq, grids, 2), vision_rotary_angles(inv_freq, layout["positions"]))
    table = torch.randn(25, 8, generator=g)
    corners = table[layout["pos_index"]] * layout["pos_weight"][:, :, None]
    expected = corners[0] + corners[1] + corners[2] + corners[3]
    torch.testing.assert_close(rope.pos_embed_interp(table, grids, 2), expected)


def test_rope_full_rotates_the_half_split_pairs():
    g = torch.Generator().manual_seed(1)
    n, heads, d = 5, 2, 8
    q, k = torch.randn(1, n, heads, d, generator=g), torch.randn(1, n, 1, d, generator=g)
    angles = torch.randn(n, d // 2, generator=g)
    cos, sin = angles.cos(), angles.sin()
    q_out, k_out = rope.rope_full(q, k, cos[None], sin[None], d)
    full = (torch.cat((cos, cos), -1)[None, :, None], torch.cat((sin, sin), -1)[None, :, None])
    torch.testing.assert_close(q_out, reference.apply_rotary(q, *full))
    torch.testing.assert_close(k_out, reference.apply_rotary(k, *full))
    q_part, _ = rope.rope_full(q, k, cos[None, :, :2], sin[None, :, :2], 4)
    assert torch.equal(q_part[..., 4:], q[..., 4:])
    partial = (torch.cat((cos[:, :2],) * 2, -1)[None, :, None], torch.cat((sin[:, :2],) * 2, -1)[None, :, None])
    torch.testing.assert_close(q_part[..., :4], reference.apply_rotary(q[..., :4], *partial))


def test_qkv_norm_split_norms_q_and_k_and_copies_v_and_the_gate():
    g = torch.Generator().manual_seed(2)
    heads, kv, d = 2, 1, 8
    packed = _randn(1, 3, 2 * heads * d + 2 * kv * d, generator=g)
    q_weight, k_weight = _randn(d, scale=0.1, generator=g), _randn(d, scale=0.1, generator=g)
    q, k, v, gate = gemm.qkv_norm_split(packed, q_weight, k_weight, heads, kv, d, 1e-6)
    query = packed[..., :2 * heads * d].reshape(1, 3, heads, 2, d)
    torch.testing.assert_close(q, norm.zero_centered_rms_norm(query[..., 0, :], q_weight, 1e-6), atol=0.02, rtol=0)
    key = packed[..., 2 * heads * d:(2 * heads + kv) * d].reshape(1, 3, kv, d)
    torch.testing.assert_close(k, norm.zero_centered_rms_norm(key, k_weight, 1e-6), atol=0.02, rtol=0)
    assert torch.equal(v, packed[..., (2 * heads + kv) * d:].reshape(1, 3, kv, d))
    assert torch.equal(gate, query[..., 1, :].reshape(1, 3, heads * d))


def test_gated_kvcache_attention_with_column_ranges_is_the_masked_attention():
    """Keys [0, prefix) and the new rows, a mask turned into per-column row ranges, the output gate."""
    g = torch.Generator().manual_seed(3)
    m, prefix, length = 6, 5, 12  # rows 5..10 hold the new keys, row 11 is unused
    q = torch.randn(1, m, 4, 8, generator=g).bfloat16()
    k, v = torch.randn(1, length, 2, 8, generator=g).bfloat16(), torch.randn(1, length, 2, 8, generator=g).bfloat16()
    gate = torch.randn(1, m, 32, generator=g).bfloat16()
    visible = torch.zeros(m, length, dtype=torch.bool)
    visible[:, :prefix] = True
    visible[2:, prefix:prefix + m] = torch.tril(torch.ones(m, m, dtype=torch.bool))[2:]
    visible[:2, prefix:prefix + 2] = True  # the first two rows see each other only
    out = attn.gated_kvcache_attention(q, k, v, torch.tensor([prefix]), prefix, gate,
                                          flash_mask=column_ranges(visible[None, None]))
    o = reference.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask=visible[None, None])
    assert torch.equal(out, o.transpose(1, 2).reshape(1, m, -1) * torch.sigmoid(gate))


def test_flow_input_embed_without_heads_is_one_fp32_projection():
    g = torch.Generator().manual_seed(8)
    z, dof, w = torch.randn(1, 5, 4, generator=g), torch.ones(1, 5, 4), torch.randn(16, 8, generator=g)
    torch.testing.assert_close(flow.flow_input_embed(z, w[None], extra=dof),
                               (torch.cat((z, dof), -1) @ w.t()).bfloat16())


def test_modulated_norms_take_the_pinned_modulation_for_the_first_rows():
    g = torch.Generator().manual_seed(10)
    x, branch = torch.randn(1, 5, 16, generator=g).bfloat16(), torch.randn(1, 5, 16, generator=g).bfloat16()
    mod, pinned = torch.randn(1, 48, generator=g) * 0.3, torch.randn(1, 48, generator=g) * 0.3
    rows = torch.tensor([2], dtype=torch.int32)
    normed, gate = norm.modulated_rms_norm(x, mod, 1e-6, pinned, rows)
    for r, m in ((0, pinned), (1, pinned), (2, mod), (4, mod)):
        assert torch.equal(normed[:, r], reference.ada_rms_norm(x[:, r:r + 1], m[:, None, :16], m[:, None, 16:32],
                                                                1e-6)[:, 0])
        assert torch.equal(gate[:, r], m[:, 32:].bfloat16())
    branch_gate = torch.randn(1, 1, 16, generator=g).bfloat16()
    normed, gate, total = norm.gated_residual_modulated_rms_norm(x, branch, branch_gate, mod, 1e-6)
    assert torch.equal(total, x + branch * branch_gate) and gate.shape == (1, 1, 16)
    assert torch.equal(normed, norm.modulated_rms_norm(total, mod, 1e-6)[0])


def test_identity_layer_norms_are_two_gated_residual_norms_in_place():
    g = torch.Generator().manual_seed(11)
    residual, branch = torch.randn(1, 5, 16, generator=g).bfloat16(), torch.randn(1, 5, 16, generator=g).bfloat16()
    branch_gate = torch.randn(1, 1, 16, generator=g).bfloat16()
    first, second = torch.randn(1, 48, generator=g) * 0.3, torch.randn(1, 48, generator=g) * 0.3
    n1, g1, s1 = norm.gated_residual_modulated_rms_norm(residual, branch, branch_gate, first, 1e-6)
    n2, g2, s2 = norm.gated_residual_modulated_rms_norm(s1, n1, g1, second, 1e-6)
    gate = norm.identity_layer_norms_(residual, branch, branch_gate, first, second, 1e-6)
    assert torch.equal(gate, g2) and torch.equal(residual, s2) and torch.equal(branch, n2)


def test_qkv_norm_rope_kvcache_is_the_gemm_then_the_split_norms_and_rope():
    g = torch.Generator().manual_seed(12)
    m, k, heads, kv, d, start = 4, 32, 2, 1, 16, 3
    x = torch.randn(m, k, generator=g).bfloat16()
    q_gate, wk, wv = (torch.randn(n, k, generator=g).bfloat16() * 0.2 for n in (2 * heads * d, kv * d, kv * d))
    qn, kn = (torch.randn(d, generator=g) * 0.1).bfloat16(), (torch.randn(d, generator=g) * 0.1).bfloat16()
    angles = torch.rand(m, 4, generator=g)
    caches = torch.zeros(2, kv, start + m, d, dtype=torch.bfloat16)
    out = gemm.qkv_norm_rope_kvcache(x, q_k_v_gate(q_gate, wk, wv, heads), qn, kn, angles.cos(), angles.sin(),
                                      caches[0], caches[1], start, heads, kv, d, 1e-6)
    packed = reference.linear(x, torch.cat((q_gate, wk, wv)))[None]
    q, key, value, gate = gemm.qkv_norm_split(packed, qn, kn, heads, kv, d, 1e-6)
    q, key = rope.rope_full(q, key, angles.cos()[None], angles.sin()[None], 8)
    torch.testing.assert_close(out.float(), torch.cat((q.reshape(m, -1), gate.reshape(m, -1)), 1).float(),
                               atol=0.05, rtol=0.05)
    torch.testing.assert_close(caches[0, :, start:].float(), key[0].transpose(0, 1).float(), atol=0.05, rtol=0.05)
    assert torch.equal(caches[1, :, start:], value[0].transpose(0, 1)) and not caches[:, :, :start].any()


@gpu
@pytest.mark.gpu
def test_xcompute_matches_the_references_at_wallx2b_shapes(xc):
    """Wall-X 2B: attention heads 8/2/256 (no gated kernel is listed: ungated attention_kvcache and the gate, and the
    reference for the causal prompt attention), action width 1024, 16 action rows, 26 action values."""
    required = ("gated_causal_attention", "gated_kvcache_attention", "modulated_rms_norm",
                "gated_residual_modulated_rms_norm", "identity_layer_norms_", "qkv_norm_rope_kvcache",
                "flow_input_embed")
    missing = [name for name in required if xc.selection(name).library != "xcompute"]
    if missing:
        pytest.skip(f"installed xcompute build lacks Wall-X 2B kernels: {', '.join(missing)}")
    g = torch.Generator().manual_seed(16)
    r = lambda *s, scale=1.0, dtype=BF16: _randn(*s, scale=scale, dtype=dtype, device="cuda", generator=g)  # noqa: E731
    p, m, a = 430, 16, 1024
    _check(xc, "gated_causal_attention", (r(p, 8, 256), r(p, 2, 256), r(p, 2, 256), r(p, 2048)), equal=True)
    visible = torch.ones(m, p + m, dtype=torch.bool, device="cuda")
    visible[:3, p + 3:] = False  # RTC: the pinned rows do not see the others
    _check(xc, "gated_kvcache_attention", (r(1, m, 8, 256), r(1, p + m, 2, 256), r(1, p + m, 2, 256),
                                           torch.tensor([p], dtype=torch.int32, device="cuda"), p, r(1, m, 2048)),
           kwargs=dict(flash_mask=column_ranges(visible[None, None])), relative=0.01)
    h = r(1, m, a)
    w_t = r(a, 2048, scale=1 / 45).t().contiguous()  # the output and down projections: stored [K, N]
    _check(xc, "gemm_epilogue", (r(m, 2048), w_t.t()), relative=0.01)
    _check(xc, "gemm_glu_epilogue", (h[0], r(4096, a, scale=1 / 32)), relative=0.01)
    mod, pinned = r(1, 3 * a, scale=0.3, dtype=torch.float32), r(1, 3 * a, scale=0.3, dtype=torch.float32)
    rows = torch.tensor([3], dtype=torch.int32, device="cuda")
    pins = dict(pinned_modulation=pinned, pinned_rows=rows)
    _check(xc, "modulated_rms_norm", (h, mod, 1e-6), kwargs=pins, relative=0.01)
    _check(xc, "gated_residual_modulated_rms_norm", (h, r(1, m, a), r(1, m, a), mod, 1e-6), kwargs=pins,
           relative=0.01)
    residual, branch, gate = r(1, m, a), r(1, m, a), r(1, m, a)
    got = (residual.clone(), branch.clone())
    want = (residual.clone(), branch.clone())
    double = dict(pinned_input_modulation=pinned, pinned_post_modulation=mod, pinned_rows=rows)
    got_gate = xc.identity_layer_norms_(*got, gate, mod, pinned, 1e-6, **double)
    want_gate = ops.registry.spec("identity_layer_norms_").reference(*want, gate, mod, pinned, 1e-6, **double)
    assert xc.selection("identity_layer_norms_").library == "xcompute"
    for x, y in zip((*got, got_gate), (*want, want_gate)):
        assert float((x.float() - y.float()).norm() / y.float().norm()) < 0.01
    angles = torch.rand(m, 32, device="cuda") * 20
    caches = [torch.zeros(2, 2, p + m, 256, dtype=BF16, device="cuda") for _ in range(2)]
    args = (h[0], r(5120, a, scale=1 / 32), r(256, scale=0.1), r(256, scale=0.1), angles.cos(), angles.sin())
    got = xc.qkv_norm_rope_kvcache(*args, caches[0][0], caches[0][1], p, 8, 2, 256, 1e-6)
    want = ops.registry.spec("qkv_norm_rope_kvcache").reference(*args, caches[1][0], caches[1][1], p, 8, 2, 256, 1e-6)
    for x, y in ((got, want), (caches[0], caches[1])):
        assert float((x.float() - y.float()).norm() / y.float().norm()) < 0.01
    z, dof = r(1, m, 26, dtype=torch.float32), (torch.rand(1, m, 26, generator=g) > 0.2).float().cuda()
    _check(xc, "flow_input_embed", (z, r(1, a, 52, scale=0.1, dtype=torch.float32)), kwargs=dict(extra=dof),
           relative=1e-4)
    _check(xc, "flow_velocity_euler", (z, h, r(26, a, scale=1 / 32, dtype=torch.float32), 0.1, dof,
                                       r(1, m, 26, dtype=torch.float32), rows, r(1, m, 26, dtype=torch.float32)),
           relative=1e-4)
