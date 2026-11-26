from types import SimpleNamespace

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
    reg.add("double", lambda x: "fast", backend="fast", contract=TOL, platforms=("sm120",), priority=1)
    reg.add("double", lambda x: "thor", backend="thor", contract=TOL, platforms=("sm110",), priority=5)
    assert reg.build_table(_platform()).double(1) == "fast"
    assert reg.build_table(_platform("cuda-sm110", "sm110")).double(1) == "thor"
    assert reg.build_table(_platform("cpu", None, "cpu")).double(1) == 2


BACKEND = """
from types import SimpleNamespace

from xinfer.ops import register

fast = SimpleNamespace(__name__="fast", __version__="2.0")  # an imported kernel library
missing = None  # a kernel library that did not import


@register("double", backend="cuda", requires=fast, platforms=("sm120",), priority=7, deterministic=False)
def double(x):
    return "fast"


@register("double", backend="cuda", requires=missing)
def double_missing(x):
    return "missing"


@register("double", backend="cuda", priority=1)
def double_plain(x):
    return "plain"


@register("triple", backend="cuda", requires=fast)
def triple(x):
    return "fast triple"
"""


@pytest.fixture
def cuda_ops(tmp_path, monkeypatch):
    """A fresh process-wide registry that defines ``double``, and an importable backend module ``cuda_ops``."""
    import importlib
    import sys

    reg = _registry()
    monkeypatch.setattr(importlib.import_module("xinfer.ops.registry"), "registry", reg)
    monkeypatch.setattr(importlib.import_module("xinfer.ops.registry"), "BUILTIN_MODULES", ())
    (tmp_path / "cuda_ops.py").write_text(BACKEND)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield reg
    sys.modules.pop("cuda_ops", None)


def test_importing_a_backend_module_registers_its_implementations(cuda_ops):
    import importlib

    reg = cuda_ops
    with pytest.raises(KeyError, match="triple"):  # an operator the registry does not define
        importlib.import_module("cuda_ops")
    reg.define(OpSpec("triple", "triple(x)", lambda x: x * 3))
    importlib.import_module("cuda_ops")
    assert [i.name for i in reg.implementations("double")] == ["reference", "cuda:fast", "cuda"]  # not "missing"
    double = reg.implementations("double")[1]
    assert (double.platforms, double.priority, double.version) == (("sm120",), 7, "fast 2.0")
    assert double.contract == Contract(numerics="tolerance", deterministic=False)
    assert reg.implementations("double")[2].platforms == ("cuda",)  # by default the backend's device type
    assert reg.build_table(_platform()).double(1) == "fast"
    assert reg.build_table(_platform("cuda-sm110", "sm110")).double(1) == "plain"
    assert reg.build_table(_platform("cpu", None, "cpu")).double(1) == 2
    assert reg.build_table(_platform(), backends={"double": "cuda"}).double(1) == "fast"  # the highest priority
    assert reg.build_table(_platform(), backends={"double": "fast"}).double(1) == "fast"
    with pytest.raises(RuntimeError, match="does not support"):
        reg.build_table(_platform(), backends={"double": "cuda:missing"})
    assert [row["library"] for row in reg.build_table(_platform()).report()] == ["fast", "fast"]
    importlib.reload(importlib.import_module("cuda_ops"))  # importing again replaces the implementations
    assert [i.name for i in reg.implementations("double")] == ["reference", "cuda:fast", "cuda"]


def test_load_plugins_imports_the_built_in_modules_and_the_entry_points(cuda_ops, monkeypatch):
    """``XINFER_<NAME>=0`` skips an entry point without importing it."""
    import importlib
    import importlib.metadata

    reg = cuda_ops
    reg.define(OpSpec("triple", "triple(x)", lambda x: x * 3))
    monkeypatch.setattr(importlib.import_module("xinfer.ops.registry"), "BUILTIN_MODULES", ())
    loads = []
    entries = [SimpleNamespace(name=name, load=lambda name=name: loads.append(name) or importlib.import_module(name))
               for name in ("cuda_ops", "skipped")]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: entries if group == "xinfer.ops" else [])
    monkeypatch.setenv("XINFER_SKIPPED", "0")
    monkeypatch.delenv("XINFER_CUDA_OPS", raising=False)
    assert reg.load_plugins() == ["cuda:fast", "cuda"]
    assert loads == ["cuda_ops"]  # "skipped" was not imported
    assert reg.load_plugins() == []  # once per registry


def test_the_process_wide_registry_loads_the_plugins_before_it_selects(cuda_ops, monkeypatch):
    loaded = []
    monkeypatch.setattr(cuda_ops, "load_plugins", lambda: loaded.append(True))
    cuda_ops.build_table(_platform())
    assert loaded == [True]
    other = _registry()
    monkeypatch.setattr(other, "load_plugins", lambda: loaded.append(False))
    other.build_table(_platform())
    assert loaded == [True]


def test_an_implementation_takes_the_parameters_of_its_reference():
    with pytest.raises(TypeError, match=r"fast implements 'double' with parameters \['y'\], the reference takes"):
        _registry().add("double", lambda y: y, backend="fast", contract=TOL)


def test_reference_only_and_pinned_backends():
    reg = _registry()
    reg.add("double", lambda x: "fast", backend="fast", contract=TOL, platforms=("sm120",), priority=1)
    assert reg.build_table(_platform(), reference_only=True).double(1) == 2
    assert reg.build_table(_platform(), backends={"double": "fast"}).double(1) == "fast"
    with pytest.raises(RuntimeError, match="does not support"):
        reg.build_table(_platform("cuda-sm110", "sm110"), backends={"double": "fast"})
    with pytest.raises(KeyError):
        reg.build_table(_platform(), backends={"nope": "fast"})


def test_backends_must_declare_their_numerics():
    reg = _registry()
    with pytest.raises(ValueError, match="bitwise, tolerance or unqualified"):
        reg.add("double", lambda x: x, backend="x", contract=Contract())
    with pytest.raises(ValueError, match="numerics"):
        Contract(numerics="close enough")
    reg.add("double", lambda x: x, backend="x", contract=TOL)
    with pytest.raises(ValueError, match="already registered"):
        reg.add("double", lambda x: x, backend="x", contract=TOL)


def test_report_lists_selection_and_contract():
    rows = _registry().build_table(_platform()).report()
    assert rows == [
        dict(op="double", backend="reference", library=None, version=None, numerics="reference",
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
        # "rpu:0" is a platform device string, not a torch device; it must not be parsed.
        assert ops.warn_slow_fallbacks("tiny", table, stage_ops, "rpu:0") == []
        assert not caplog.records
        assert ops.warn_slow_fallbacks("tiny", table, stage_ops, "cuda") == ["gemm_epilogue", "qkv_rope_kvcache"]
    assert "tiny: 2 operators have no fast kernel loaded" in caplog.text and "docs/kernels.md" in caplog.text


def test_slow_fallbacks_fail_in_strict_mode(monkeypatch):
    references = OpRegistry()
    for name in ops.registry.names():
        references.define(ops.registry.spec(name))
    table = references.build_table(_platform())
    monkeypatch.setenv("XINFER_OPS_STRICT", "1")
    with pytest.raises(RuntimeError, match="XINFER_OPS_STRICT"):
        ops.warn_slow_fallbacks("tiny", table, {"action": ("gemm_epilogue",)}, "cuda")
    assert ops.warn_slow_fallbacks("tiny", table, {"action": ("gemm_epilogue",)}, "cpu") == []


def test_cuda_checks_composed_references_when_the_kernel_library_is_absent():
    catalog = OpRegistry()
    names = ("linear", "flow_input_embed", "qkv_norm_split", "nvfp4_gemm_epilogue", "mxfp8_layer_norm_quantize")
    for name in names:
        catalog.define(ops.registry.spec(name))
    table = catalog.build_table(_platform())
    assert ops.warn_slow_fallbacks("tiny", table, {"stage": names}, "cuda") == sorted(names[1:])


@pytest.fixture
def prepared_registry():
    reg = OpRegistry()
    reg.define(OpSpec("prepare", "prepare(context, *, tensors) -> Operation"))
    assert reg.implementations("prepare") == ()

    def prepare(context, *, tensors):
        return context, tensors

    reg.add("prepare", prepare, backend="rpu", platforms=("rpu",),
            contract=Contract(numerics="unqualified", graph_safe=False))
    return reg


def test_prepared_operator_selects_backend_factory_without_a_reference(prepared_registry):
    table = prepared_registry.build_table(_platform("rpu", None, "rpu"))
    context, tensors = object(), {"out": object()}
    selected_context, selected_tensors = table.prepare(context, tensors=tensors)
    assert selected_context is context and selected_tensors is tensors
    assert table.report() == [dict(op="prepare", backend="rpu", library=None, version=None,
                                   numerics="unqualified", deterministic=True, graph_safe=False)]


def test_operators_without_platform_candidates_are_omitted(prepared_registry):
    prepared_registry.define(OpSpec("unavailable", "unavailable()"))
    table = prepared_registry.build_table(_platform("cpu", None, "cpu"), reference_only=True)
    assert table.report() == []
    with pytest.raises(AttributeError, match="not in this op table"):
        table.prepare


def test_reference_only_rejects_supported_prepared_operator_but_pin_takes_precedence(prepared_registry):
    platform = _platform("rpu", None, "rpu")
    with pytest.raises(RuntimeError, match="'prepare' has no reference implementation supporting rpu"):
        prepared_registry.build_table(platform, reference_only=True)
    table = prepared_registry.build_table(platform, reference_only=True, backends={"prepare": "rpu"})
    assert table.selection("prepare").backend == "rpu"


def test_pinned_operators_cannot_silently_disappear(prepared_registry):
    platform = _platform("cpu", None, "cpu")
    with pytest.raises(RuntimeError, match="backend 'rpu' for 'prepare' does not support cpu"):
        prepared_registry.build_table(platform, backends={"prepare": "rpu"})


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "ascend"])
def test_global_torch_references_are_limited_to_torch_platforms(device_type):
    reg = OpRegistry()
    reg.define(ops.registry.spec("linear"))
    assert reg.build_table(_platform(device_type, None, device_type)).selection("linear").backend == "reference"
    assert reg.build_table(_platform("rpu", None, "rpu")).report() == []
