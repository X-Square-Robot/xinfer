"""Fused Wall-OSS operators against unfused torch (CPU) and xcompute kernels (GPU)."""

import os

import pytest
import torch
import torch.nn.functional as F

from xinfer import ops
from xinfer.ops import fused, reference
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
    assert torch.equal(fused.gemm_epilogue(x, w, residual=residual, residual_gate=gate, row_class=rows), want)
    scale = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    out = fused.gemm_epilogue(x, w, row_scale=scale, activation="gelu_tanh")
    assert torch.equal(out[1], torch.zeros(8, dtype=torch.bfloat16))
    assert torch.equal(out[0], F.gelu(y[0], approximate="tanh").bfloat16())


def test_gemm_glu_epilogue_is_ada_rms_norm_then_silu_mul():
    g = torch.Generator().manual_seed(1)
    x, w = torch.randn(5, 16, generator=g), torch.randn(2 * 8, 16, generator=g)
    table, rows = torch.randn(3, 48, generator=g) * 0.3, torch.tensor([2, 0, 1, 1, 2])
    normed = reference.ada_rms_norm(x, table[rows, :16], table[rows, 16:32], 1e-6)
    y = normed @ w.t()
    torch.testing.assert_close(fused.gemm_glu_epilogue(x, w, modulation=table, row_class=rows, eps=1e-6),
                               F.silu(y[:, :8]) * y[:, 8:])


def test_qkv_rope_epilogue_rotates_q_and_k_halves():
    g = torch.Generator().manual_seed(2)
    x, w, b = torch.randn(4, 16, generator=g), torch.randn(3 * 2 * 8, 16, generator=g), torch.randn(48, generator=g)
    angles = torch.rand(4, 4, generator=g)
    out = fused.qkv_rope_epilogue(x, w, b, angles.cos(), angles.sin(), 2, 8).view(4, 3, 2, 8)
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
    torch.testing.assert_close(fused.flow_velocity_euler(x, h, weight, 0.1), x + 0.1 * p)
    want = x + 0.1 * (padding * (1 - dof) + p * dof)
    want[:, :2] = prefix[:, :2]
    torch.testing.assert_close(fused.flow_velocity_euler(x, h, weight, 0.1, dof, padding, torch.tensor([2]), prefix),
                               want)


def test_qkv_rope_kvcache_is_the_biased_gemm_rope_and_kv_write():
    g = torch.Generator().manual_seed(13)
    m, k, heads, kv, d, start = 4, 32, 2, 1, 16, 3
    x, w, b = torch.randn(m, k, generator=g).bfloat16(), (torch.randn(64, k, generator=g) * 0.2).bfloat16(), \
        torch.randn(64, generator=g).bfloat16()
    angles = torch.rand(m, 8, generator=g)
    caches = torch.zeros(2, kv, start + m, d, dtype=torch.bfloat16)
    q = fused.qkv_rope_kvcache(x, w, b, angles.cos(), angles.sin(), caches[0], caches[1], start, heads, kv, d)
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
    total, normed = fused.add_rms_norm(x, residual, w, 1e-6)
    assert torch.equal(total, residual + x) and torch.equal(normed, reference.rms_norm(residual + x, w, 1e-6))


def test_an_xcompute_package_of_another_build_is_not_used(monkeypatch, tmp_path):
    """A package whose library lacks the unified entries' functions (another build, such as the MXFP8 one) would fail
    at the first GEMM: the fused operators stay on their references."""
    import sys
    import types

    from xinfer.ops.backends import xcompute_unified

    path = tmp_path / "xcompute.so"
    path.write_bytes(b"")
    package = types.ModuleType("xcompute")
    package.KERNEL_LIB_PATH = str(path)
    for name in xcompute_unified.UNIFIED:
        setattr(package, name, object())
    monkeypatch.setitem(sys.modules, "xcompute", package)
    monkeypatch.setenv("XCOMPUTE_LIBRARY_PATH", str(path))

    class Library:
        def __init__(self, functions):
            self.path, self.functions = str(path), functions

        def has_function(self, name):
            return name in self.functions

    assert xcompute_unified._python_package(Library(xcompute_unified.LIBRARY_ENTRIES)) is package
    assert xcompute_unified._python_package(Library(())) is None


def _fake_package(monkeypatch, tmp_path):
    """A stand-in ``xcompute`` package whose library is ``tmp_path/xcompute.so``, importable as ``xcompute``."""
    import importlib.machinery
    import sys
    import types

    from xinfer.ops.backends import xcompute_unified

    path = tmp_path / "xcompute.so"
    path.write_bytes(b"")
    package = types.ModuleType("xcompute")
    package.__spec__ = importlib.machinery.ModuleSpec("xcompute", None)
    package.KERNEL_LIB_PATH = str(path)
    for name in xcompute_unified.UNIFIED:
        setattr(package, name, object())
    monkeypatch.setitem(sys.modules, "xcompute", package)
    monkeypatch.setenv("XCOMPUTE_LIBRARY_PATH", str(path))
    return package


@pytest.mark.parametrize("archs,platforms", [("110,120", ("sm110", "sm120")), ("120", ("sm120",)), ("", None)])
def test_xcompute_operators_register_on_the_architectures_the_build_targets(monkeypatch, tmp_path, archs, platforms):
    from xinfer.ops.backends import xcompute_unified
    from xinfer.ops.registry import OpRegistry

    package = _fake_package(monkeypatch, tmp_path)

    class Library:
        path = package.KERNEL_LIB_PATH
        sha256 = "0" * 64

        def has_function(self, name):
            return True

        def function(self, name):
            assert name == "build_target_archs"
            return lambda: archs

        def call(self, name, *args):
            raise AssertionError("no kernel runs at registration")

    registry = OpRegistry()
    for name in ops.registry.names():
        registry.define(ops.registry.spec(name))
    xcompute_unified.register_unified(registry, Library())
    names = registry.backend_ops("xcompute")
    if platforms is None:
        assert names == []
    else:
        assert set(names) == set(FUSED)
        assert {registry.implementations(name)[-1].platforms for name in names} == {platforms}


def test_the_xcompute_library_comes_from_the_installed_package(monkeypatch, tmp_path):
    from xinfer.ops.backends import xcompute

    package = _fake_package(monkeypatch, tmp_path)
    assert xcompute._package_library() == package.KERNEL_LIB_PATH
    monkeypatch.setattr(xcompute.importlib.util, "find_spec", lambda name: None)
    assert xcompute._package_library() is None


def test_xinfer_xcompute_0_skips_the_xcompute_backend(monkeypatch):
    from xinfer.ops.backends import xcompute
    from xinfer.ops.registry import OpRegistry

    calls = []
    monkeypatch.setattr(xcompute, "register", lambda registry: calls.append(registry))
    monkeypatch.setenv("XINFER_XCOMPUTE", "0")
    monkeypatch.setenv("XINFER_FLASH_ATTN", "0")
    assert "xcompute" not in OpRegistry().load_plugins() and calls == []


# --------------------------------------------------------------------------- xcompute (GPU, RTX 5090)
XCOMPUTE = os.environ.get("XINFER_XCOMPUTE_LIBRARY")
gpu = pytest.mark.skipif(not (XCOMPUTE and torch.cuda.is_available()
                              and torch.cuda.get_device_capability() == (12, 0)),
                         reason="xcompute + an SM120 GPU required")
FUSED = ("gemm_epilogue", "gemm_glu_epilogue", "qkv_rope_epilogue", "qkv_rope_kvcache", "rms_norm",
         "add_rms_norm", "prefix_kvcache_attention", "flow_velocity_euler")


@pytest.fixture(scope="module")
def xc():
    from xinfer.ops.backends import xcompute
    from xinfer.ops.registry import OpRegistry

    registry = OpRegistry()
    for name in ops.registry.names():
        registry.define(ops.registry.spec(name))
    names = xcompute.register(registry)
    if "gemm_epilogue" not in names:
        pytest.skip("needs a build of xcompute main and the xcompute Python package of the same build")
    missing = set(FUSED) - set(names)
    assert not missing, missing
    return registry.build_table(resolve_platform("cuda"))


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
    assert table.selection(name).backend == "xcompute"
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
        assert xc.selection("qkv_rope_kvcache").backend == "xcompute"
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
