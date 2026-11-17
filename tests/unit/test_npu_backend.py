"""CPU regressions for Ascend loading, default selection, partial support and reference isolation."""

import importlib
from types import ModuleType, SimpleNamespace

import pytest
import torch

from xinfer import ops
from xinfer.ops.backends.ascend import _support
from xinfer.ops.registry import OpRegistry
from xinfer.platforms.interface import Platform, PlatformCapabilities


def _platform(kind="ascend", arch="future-chip"):
    return Platform(f"{kind}-{arch}", "cpu",
                    PlatformCapabilities(kind, arch, True, False, False, None, None, None))


@pytest.fixture
def isolated_backend(monkeypatch):
    """Reload Ascend modules into an isolated registry and restore production bindings afterwards."""
    registry_module = importlib.import_module("xinfer.ops.registry")
    module_names = ("xinfer.ops.backends.ascend._torch_attn", "xinfer.ops.backends.ascend.norm",
                    "xinfer.ops.backends.ascend.gemm", "xinfer.ops.backends.ascend.attn",
                    "xinfer.ops.backends.ascend.rope", "xinfer.ops.backends.ascend.flow",
                    "xinfer.ops.backends.ascend.official")
    registry = OpRegistry()
    for name in ops.registry.names():
        registry.define(ops.registry.spec(name))
    package = ModuleType("xcompute")
    package.xcompute_lib = SimpleNamespace()
    monkeypatch.setattr(registry_module, "registry", registry)
    monkeypatch.setattr(_support, "xc", package)
    monkeypatch.setattr(_support, "bridge", lambda name: None)

    def load():
        return [importlib.reload(importlib.import_module(name)) for name in module_names]

    try:
        yield registry, package, load
    finally:
        monkeypatch.undo()
        for name in module_names:
            importlib.reload(importlib.import_module(name))


def test_available_entry_is_selected_without_pins_and_other_platforms_stay_reference(isolated_backend, monkeypatch):
    registry, package, load = isolated_backend
    projection = SimpleNamespace(gemm_epilogue=ops.reference.gemm_epilogue, accumulator=lambda x, weight: None)
    bridges = {
        "cann_projection": SimpleNamespace(CannProjection=lambda: projection),
        "cann_action": SimpleNamespace(CannAction=lambda accumulator, *, precision: SimpleNamespace()),
    }
    monkeypatch.setattr(_support, "bridge", bridges.get)
    load()
    selected = registry.build_table(_platform()).selection("gemm_epilogue")
    assert selected.backend == "ascend" and selected.library == "xcompute"
    assert registry.build_table(_platform("cuda", "sm120")).selection("gemm_epilogue").backend == "reference"
    assert registry.build_table(_platform("cpu")).selection("gemm_epilogue").backend == "reference"
    assert registry.build_table(_platform()).selection("qkv_rope_epilogue").backend == "reference"
    assert registry.build_table(_platform(), reference_only=True).selection("gemm_epilogue").backend == "reference"
    x, weight = torch.randn(2, 3), torch.randn(4, 3)
    torch.testing.assert_close(selected.function(x, weight), ops.reference.gemm_epilogue(x, weight))


@pytest.mark.parametrize(("module", "factory"), [("cann_projection", "CannProjection"),
                                               ("cann_attention", "CannAttention")])
def test_broken_cann_factories_do_not_silently_fall_back(isolated_backend, monkeypatch, module, factory):
    _, _, load = isolated_backend

    def broken():
        raise AttributeError("broken bridge configuration")

    monkeypatch.setattr(_support, "bridge", lambda name: SimpleNamespace(**{factory: broken})
                        if name == module else None)
    with pytest.raises(AttributeError, match="broken bridge configuration"):
        load()


def test_missing_projection_backend_keeps_reference_fallback(isolated_backend, monkeypatch):
    registry, package, load = isolated_backend

    class MissingBackendError(AttributeError):
        pass

    package.MissingBackendError = MissingBackendError

    def unavailable():
        raise MissingBackendError("this build has no projection export")

    monkeypatch.setattr(_support, "bridge", lambda name: SimpleNamespace(CannProjection=unavailable)
                        if name == "cann_projection" else None)
    load()
    table = registry.build_table(_platform())
    assert table.selection("gemm_epilogue").backend == "reference"
    x, weight = torch.randn(2, 3), torch.randn(4, 3)
    torch.testing.assert_close(table.gemm_epilogue(x, weight), ops.reference.gemm_epilogue(x, weight))


def test_partial_support_caches_options_and_does_not_hide_execution_errors(isolated_backend, monkeypatch):
    registry, _, load = isolated_backend
    attempts = []
    broken = False

    def project(x, weight, bias=None, **options):
        attempts.append(options["activation"])
        if options["activation"] == "silu":
            raise ValueError("unsupported activation")
        if broken:
            raise ValueError("failure after a layout was admitted")
        if options["activation"] == "gelu":
            raise RuntimeError("device failed")
        return ops.reference.gemm_epilogue(x, weight, bias, **options)

    projection = SimpleNamespace(gemm_epilogue=project)
    monkeypatch.setattr(_support, "bridge", lambda name: SimpleNamespace(CannProjection=lambda: projection)
                        if name == "cann_projection" else None)
    load()
    table = registry.build_table(_platform())
    x, weight = torch.randn(2, 3), torch.randn(4, 3)
    for _ in range(2):
        torch.testing.assert_close(table.gemm_epilogue(x, weight, activation="silu"),
                                   ops.reference.gemm_epilogue(x, weight, activation="silu"))
    assert attempts == ["silu"]
    table.gemm_epilogue(x, weight)
    broken = True
    with pytest.raises(ValueError, match="after a layout"):
        table.gemm_epilogue(x, weight)
    broken = False
    with pytest.raises(RuntimeError, match="device failed"):
        table.gemm_epilogue(x, weight, activation="gelu")


def test_graph_attention_is_a_local_backend_and_reference_only_keeps_sdpa(isolated_backend, monkeypatch):
    registry, _, load = isolated_backend
    monkeypatch.setattr(_support, "xc", None)
    load()
    graph = registry.build_table(_platform())
    assert graph.selection("causal_attention").name == "ascend:torch"
    assert graph.selection("varlen_attention").name == "ascend:torch"
    assert registry.build_table(_platform(), reference_only=True).selection("causal_attention").backend == "reference"
    assert registry.build_table(_platform("cuda", "sm120")).selection("causal_attention").backend == "reference"
    calls = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def tracked(*args, **kwargs):
        calls.append(True)
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", tracked)
    q, k, v = [torch.randn(1, 3, 2, 4) for _ in range(3)]
    expected = ops.reference.causal_attention(q, k, v)
    assert calls
    calls.clear()
    actual = graph.causal_attention(q, k, v)
    torch.testing.assert_close(actual, expected)
    assert not calls


def test_native_support_predicate_runs_once_for_each_layout():
    checks, launches = [], []

    def check(x):
        checks.append(tuple(x.shape))
        return x.shape[-1] == 4

    @_support.partial(lambda x: x + 10, check=check)
    def kernel(x):
        launches.append(tuple(x.shape))
        return x + 1

    for _ in range(2):
        assert torch.equal(kernel(torch.zeros(2, 4)), torch.ones(2, 4))
        assert torch.equal(kernel(torch.zeros(2, 3)), torch.full((2, 3), 10.0))
    assert checks == [(2, 4), (2, 3)] and launches == [(2, 4), (2, 4)]


def test_ascend_platform_uses_backend_identity_and_real_cube_core_property(monkeypatch):
    from xinfer.platforms import npu

    props = SimpleNamespace(name="Ascend Future", cube_core_num=32, total_memory=1024)
    runtime = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0, device_count=lambda: 1,
                              get_device_properties=lambda index: props)
    monkeypatch.setattr(npu, "device_module_for", lambda device: runtime)
    monkeypatch.setattr(npu.torch, "device", lambda device: SimpleNamespace(index=0))
    monkeypatch.setattr(npu, "npu_stream_priority_range", lambda module, index: None)
    monkeypatch.setattr(npu, "npu_device_images_available", lambda: True)
    platform = npu.npu_platform("npu:0")
    assert platform.is_npu and platform.device == "npu:0"
    assert platform.capabilities.device_type == "ascend" and platform.capabilities.sm_count == 32
    assert platform.capabilities.device_images


@pytest.mark.parametrize("graph_api", [None, False, lambda: None], ids=["missing", "noncallable", "callable"])
def test_npu_graph_capability_comes_from_the_runtime(monkeypatch, graph_api):
    from xinfer.config import ExplicitSettings, resolve
    from xinfer.platforms import npu
    from test_config import ThreeStage

    props = SimpleNamespace(name="Ascend Future", cube_core_num=32, total_memory=1024)
    runtime = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0, device_count=lambda: 1,
                              get_device_properties=lambda index: props)
    if graph_api is not None:
        runtime.NPUGraph = runtime.graph = runtime.graph_pool_handle = graph_api
    monkeypatch.setattr(npu, "device_module_for", lambda device: runtime)
    monkeypatch.setattr(npu.torch, "device", lambda device: SimpleNamespace(index=0))
    monkeypatch.setattr(npu, "npu_stream_priority_range", lambda module, index: None)
    monkeypatch.setattr(npu, "npu_device_images_available", lambda: False)
    platform = npu.npu_platform("npu:0")
    assert platform.capabilities.cudagraphs is callable(graph_api)
    resolved = resolve(ThreeStage(), platform, model_name="test", tuned=None)
    assert (resolved.engine_config().cudagraphs is not None) is callable(graph_api)
    eager = resolve(ThreeStage(), platform, model_name="test", tuned=None,
                    explicit=ExplicitSettings(cudagraphs=False))
    assert eager.engine_config().cudagraphs is None


def test_legacy_eager_only_entries_do_not_replace_graph_safe_implementations(isolated_backend):
    registry, package, load = isolated_backend

    def eager_only(*args, **kwargs):
        pytest.fail("legacy eager-only entry was called")

    package.gemm = package.gateup_silu = package.gateup_silu_pack_weight = package.attention_varlen = eager_only
    package.xcompute_lib = SimpleNamespace(**{name: eager_only for name in (
        "gemm_envelope_table", "gemm_generic", "gemm_generic_workspace_size", "gemm_gelu_bias",
        "gemm_compat_workspace_size", "gemm_compat_npu", "bf16_gemm_silu_mul_sm110", "attention_varlen_ascend",
    )})
    load()
    table = registry.build_table(_platform())
    for name in ("gemm_epilogue", "gemm_glu_epilogue"):
        assert table.selection(name).backend == "reference"
        assert table.selection(name).contract.graph_safe
    assert table.selection("varlen_attention").name == "ascend:torch"
    assert table.selection("varlen_attention").contract.graph_safe
    x, weight = torch.randn(2, 3), torch.randn(4, 3)
    torch.testing.assert_close(table.gemm_epilogue(x, weight), ops.reference.gemm_epilogue(x, weight))
    torch.testing.assert_close(table.gemm_glu_epilogue(x, weight), ops.reference.gemm_glu_epilogue(x, weight))
    q, k, v = [torch.randn(3, 2, 4) for _ in range(3)]
    torch.testing.assert_close(table.varlen_attention(q, k, v, (0, 1, 3)),
                               ops.reference.varlen_attention(q, k, v, (0, 1, 3)))


@pytest.mark.parametrize("operator", ["varlen_attention", "causal_attention", "action_attention",
                                      "masked_joint_attention", "prefix_attention", "kvcache_attention",
                                      "prefix_kvcache_attention"])
def test_graph_attention_preserves_masks_and_cache_updates(operator):
    from xinfer.ops.backends.ascend import _torch_attn

    rng = torch.Generator().manual_seed(37)
    q = torch.randn(2, 3, 4, 8, generator=rng)
    k, v = [torch.randn(2, 3, 2, 8, generator=rng) for _ in range(2)]
    prompt = [torch.randn(2, 2, 5, 8, generator=rng) for _ in range(2)]
    caches = [torch.randn(2, 8, 2, 8, generator=rng) for _ in range(2)]
    options = {"scale": 0.19}
    if operator == "varlen_attention":
        args = (q[0], q[0] + 1, q[0] - 1, (0, 1, 3))
    elif operator == "causal_attention":
        args = (q, k, v)
    elif operator == "action_attention":
        args = (q, k, v, *prompt, torch.tensor([0, 2]), torch.tensor([5, 3]))
    elif operator == "masked_joint_attention":
        mask = torch.ones(2, 1, 3, 8, dtype=torch.bool)
        mask[:, :, 1, :3] = False
        args = (q, k, v, *prompt, mask)
    elif operator == "prefix_attention":
        args = (q, k, v, torch.tensor([1, 3]))
    elif operator == "kvcache_attention":
        args = (q, k, v, *caches, torch.tensor([1, 0]), torch.tensor([3, 5]))
    else:
        args = (q, *caches, torch.tensor([3, 5], dtype=torch.int32), 5)
        mask = torch.empty(2, 1, 8, 2, dtype=torch.int32)
        mask[..., 0], mask[..., 1] = 3, 0
        mask[:, :, 2, 0] = 1
        options["flash_mask"] = mask
    copied = tuple(arg.clone() if torch.is_tensor(arg) else arg for arg in args)
    expected = getattr(ops.reference, operator)(*copied, **options)
    actual = getattr(_torch_attn, operator)(*args, **options)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for arg, original in zip(args, copied):
        if torch.is_tensor(arg):
            torch.testing.assert_close(arg, original, atol=0, rtol=0)


def test_segment_values_are_part_of_the_partial_support_key():
    from xinfer.ops.backends.ascend.attn import _varlen_key

    q = torch.zeros(4, 2, 8)
    segments = torch.tensor([0, 2, 4])
    first = _varlen_key(q, q, q, segments)
    segments[1] = 1
    assert _varlen_key(q, q, q, segments) != first
    assert _varlen_key(q, q, q, (0, 1, 4)) == _varlen_key(q, q, q, segments)


def test_an_unsupported_calls_reference_error_is_not_retried():
    calls = []

    def reference(x):
        calls.append(x)
        raise ValueError("invalid input")

    @_support.partial(reference, check=lambda x: False)
    def kernel(x):
        pytest.fail("unsupported input launched")

    with pytest.raises(ValueError, match="invalid input"):
        kernel(3)
    assert calls == [3]


def test_the_2b_exports_receive_the_opspec_arguments_and_are_passed_through(isolated_backend):
    """The 2B registrations hand the operator's own arguments to the export and return what it returns.

    The registered object is a dispatch whose ``check`` needs a real NPU tensor, so on this host every call takes the
    reference path. ``functools.wraps`` leaves the registered function on ``__wrapped__``, which is the code that
    delegates to the export, so that is what this test calls: a wrong argument name or order, a reordered result
    tuple or a mutated input fails here without an Ascend device.
    """
    registry, package, load = isolated_backend
    seen = {}

    def export(name, arity=1):
        def run(*args, **kwargs):
            seen[name] = (args, kwargs)
            return tuple("out%d" % index for index in range(arity))
        return run

    class Rmsnorm:
        def __call__(self, input, weight, epsilon, inference=False, zero_centered=False, use_wide=True, **kw):
            seen["zero_centered_rms_norm"] = (input, weight, epsilon, inference, zero_centered)
            return ("normed", None)

    class AddRmsnorm:
        def __call__(self, residual, input, weight, epsilon, inference=False, zero_centered=False):
            seen["add_zero_centered_rms_norm"] = (residual, input, weight, epsilon, zero_centered)
            return ("summed", "normed")

    class Adarmsnorm:
        def __call__(self, input, modulation, epsilon, inference=False, **kw):
            seen["adarmsnorm"] = (input, modulation, epsilon, inference)
            if kw.get("residual") is not None:
                return ("gated", "gate", "residual_out")
            return ("normed", "gate")

    class Rope:
        def full(self, q, k, cos, sin, interleave=False, rotary_dim=None):
            seen["rope_full"] = (q, k, cos, sin, interleave, rotary_dim)
            return ("q_out", "k_out")

    class CannGatedDeltaNet:
        def __call__(self, qkv, zba, conv_weight, a_log, dt_bias, norm_weight, hk, hv, dk, dv, eps):
            seen["gated_delta_net"] = (qkv, zba, conv_weight, a_log, dt_bias, norm_weight, hk, hv, dk, dv)
            return "mixed"

    package.Rmsnorm = Rmsnorm
    package.AddRmsnorm = AddRmsnorm
    package.Adarmsnorm = Adarmsnorm
    package.Rope = Rope
    package.CannGatedDeltaNet = CannGatedDeltaNet
    package.layernorm_inplace = export("layer_norm_cast")
    package.denoise_input_embed = export("flow_input_embed")
    package.denoise_output_euler = export("flow_velocity_euler")
    package.double_adarmsnorm_gated_residual_inplace = export("identity_layer_norms_", 0)

    def attention_varlen(q, k, v, **kw):
        seen["attention_varlen"] = (q, k, v, kw)
        return "attended"

    package.attention_varlen = attention_varlen
    package.attention_varlen_backend = lambda q, k, v, **kw: "ascend_tilelang"

    load()
    table = registry.build_table(_platform())
    rng = torch.Generator().manual_seed(11)

    def bf(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.bfloat16)

    def registered(name):
        """The function the registration wraps, i.e. the delegation itself."""
        return getattr(table, name).__wrapped__

    x, w1, w2 = bf(2, 4), bf(4), bf(4)

    assert registered("layer_norm_cast")(x, w1, w2, 1e-6) == ("out0",)
    assert registered("zero_centered_rms_norm")(x, w1, 1e-6) == ("normed", None)
    assert registered("add_zero_centered_rms_norm")(x, bf(2, 4), w1, 1e-6) == ("summed", "normed")
    assert registered("modulated_rms_norm")(x, bf(2, 12), 1e-6) == ("normed", "gate")
    assert registered("rope_full")(bf(2, 4), bf(2, 4), torch.ones(2, 2), torch.zeros(2, 2), 2) == \
        ("q_out", "k_out")
    assert registered("flow_input_embed")(x, bf(4), None, bf(2, 4)) == ("out0",)

    # the zero-centered operators pass their flag, and RoPE passes interleave=False (half split)
    assert seen["zero_centered_rms_norm"][3] is True and seen["zero_centered_rms_norm"][4] is True
    assert seen["add_zero_centered_rms_norm"][4] is True
    assert seen["rope_full"][4] is False

    # the gated residual form returns (output, gate, residual_output) in that order
    assert registered("gated_residual_modulated_rms_norm")(
        bf(2, 4), bf(2, 4), bf(2, 4), bf(2, 12), 1e-6) == ("gated", "gate", "residual_out")

    # identity_layer_norms_ allocates the gate and returns it
    gate = registered("identity_layer_norms_")(
        bf(2, 4), bf(2, 4), bf(2, 4), bf(2, 12), bf(2, 12), 1e-6)
    assert torch.is_tensor(gate) and gate.shape == (2, 4)

    # gated_delta_net forwards the head configuration positionally, in the operator's order
    registered("gated_delta_net")(bf(1, 2, 64), bf(1, 2, 8), bf(64, 4), bf(2), bf(2), bf(4),
                                  16, 16, 128, 128, 1e-6)
    assert seen["gated_delta_net"][6:10] == (16, 16, 128, 128)


def test_explicit_operator_exports_run_without_optional_cann_dependencies(isolated_backend, monkeypatch):
    from functools import wraps

    registry, package, load = isolated_backend
    names = ("gemm_epilogue", "gemm_glu_epilogue", "qkv_rope_epilogue", "qkv_rope_kvcache",
             "varlen_attention", "causal_attention", "prefix_kvcache_attention", "add_rms_norm")
    called = set()

    def implementation(name):
        reference = ops.registry.spec(name).reference

        @wraps(reference)
        def run(*args, **kwargs):
            called.add(name)
            return reference(*args, **kwargs)

        return run

    for name in names:
        setattr(package, name, implementation(name))

    def unavailable_bridge(name):
        raise ImportError("optional CANN bridge dependencies are not installed")

    monkeypatch.setattr(_support, "bridge", unavailable_bridge)
    load()
    table = registry.build_table(_platform())
    cuda = registry.build_table(_platform("cuda", "sm120"))
    assert all(table.selection(name).name == "ascend:xcompute" for name in names)
    assert all(cuda.selection(name).backend == "reference" for name in names)
    rng = torch.Generator().manual_seed(73)

    def tensor(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.bfloat16)

    x, weight = tensor(2, 3), tensor(4, 3)
    cos, sin = torch.ones(2, 2), torch.zeros(2, 2)
    q, k, v = tensor(1, 2, 2, 4), tensor(1, 2, 1, 4), tensor(1, 2, 1, 4)
    packed = tensor(3, 2, 4)
    calls = {
        "gemm_epilogue": ((x, weight), {"activation": "gelu_tanh", "residual": tensor(2, 4)}),
        "gemm_glu_epilogue": ((x, weight), {"modulation": torch.randn(2, 9, generator=rng),
                                            "row_class": torch.tensor([0, 1])}),
        "qkv_rope_epilogue": ((x, tensor(24, 3), tensor(24), cos, sin, 2, 4), {}),
        "qkv_rope_kvcache": ((x, tensor(16, 3), tensor(16), cos, sin,
                               tensor(1, 4, 4), tensor(1, 4, 4), 1, 2, 1, 4), {}),
        "varlen_attention": ((packed, packed, packed, (0, 1, 3)), {}),
        "causal_attention": ((q, k, v), {}),
        "prefix_kvcache_attention": ((q, tensor(1, 4, 1, 4), tensor(1, 4, 1, 4),
                                        torch.tensor([1], dtype=torch.int32), 2), {}),
        "add_rms_norm": ((x, tensor(2, 3), torch.ones(3), 1e-6), {}),
    }
    for name, (args, options) in calls.items():
        copied = tuple(value.clone() if torch.is_tensor(value) else value for value in args)
        expected = ops.registry.spec(name).reference(*copied, **options)
        result = getattr(table, name)(*args, **options)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        for actual, expected_input in zip(args, copied):
            if torch.is_tensor(actual):
                torch.testing.assert_close(actual, expected_input, atol=0, rtol=0)
    # the projections joined with the residual norm call the exports above in turn
    composed = {"gemm_add_rms_norm": ((x, weight, tensor(1, 2, 4), torch.ones(4), 1e-6), {"bias": tensor(4)}),
                "mlp_add_rms_norm": ((x, tensor(8, 3), tensor(4, 4), tensor(1, 2, 4), torch.ones(4), 1e-6), {})}
    for name, (args, options) in composed.items():
        assert table.selection(name).name == "ascend:xcompute"
        torch.testing.assert_close(getattr(table, name)(*args, **options),
                                   ops.registry.spec(name).reference(*args, **options), atol=0, rtol=0)
    assert called == set(names)


def test_partial_canonical_export_takes_precedence_and_preserves_other_bridges(isolated_backend, monkeypatch):
    from functools import wraps

    registry, package, load = isolated_backend
    called = []

    @wraps(ops.reference.gemm_epilogue)
    def canonical_gemm(*args, **kwargs):
        called.append("canonical")
        return ops.reference.gemm_epilogue(*args, **kwargs)

    @wraps(ops.reference.gemm_glu_epilogue)
    def bridged_glu(*args, **kwargs):
        called.append("bridge")
        return ops.reference.gemm_glu_epilogue(*args, **kwargs)

    def old_gemm(*args, **kwargs):
        pytest.fail("the optional bridge replaced the canonical export")

    package.gemm_epilogue = canonical_gemm
    projection = SimpleNamespace(gemm_epilogue=old_gemm, gemm_glu_epilogue=bridged_glu)
    monkeypatch.setattr(_support, "bridge", lambda name: SimpleNamespace(CannProjection=lambda: projection)
                        if name == "cann_projection" else None)
    load()
    table = registry.build_table(_platform())
    x, weight = torch.ones(2, 3), torch.ones(4, 3)
    torch.testing.assert_close(table.gemm_epilogue(x, weight), ops.reference.gemm_epilogue(x, weight))
    torch.testing.assert_close(table.gemm_glu_epilogue(x, weight), ops.reference.gemm_glu_epilogue(x, weight))
    assert "canonical" in called and "bridge" in called
    assert table.selection("qkv_rope_kvcache").backend == "reference"
    assert table.selection("causal_attention").name == "ascend:torch"


def test_bridge_attention_output_is_flattened_to_the_operator_contract(isolated_backend, monkeypatch):
    """A CANN bridge that answers with ``[B, M, H, D]`` must still satisfy the OpSpec's ``[B, M, H*D]``.

    The reference flattens the head dimension, so nothing else in the pipeline would notice a provider that did
    not: the ascend implementation owns this part of the contract.
    """
    registry, package, load = isolated_backend
    q = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
    k = torch.randn(1, 4, 2, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 2, 4, dtype=torch.bfloat16)
    prefix = torch.tensor([1], dtype=torch.int32)
    expected = ops.reference.prefix_kvcache_attention(q, k, v, prefix, 2)

    def unflattened(*args, **kwargs):
        """The bridge's own layout: head-major output, [B, M, H, D]."""
        return ops.reference.prefix_kvcache_attention(*args, **kwargs).reshape(q.shape[0], q.shape[1], q.shape[2], -1)

    attention = SimpleNamespace(_bnsd_launch=lambda *a, **k: None, _bnsd_query=lambda *a, **k: None,
                                prefix_kvcache_attention=unflattened)
    monkeypatch.setattr(_support, "bridge", lambda name: SimpleNamespace(CannAttention=lambda: attention)
                        if name == "cann_attention" else None)
    load()
    table = registry.build_table(_platform())
    assert table.selection("prefix_kvcache_attention").name == "ascend:xcompute"
    out = table.prefix_kvcache_attention(q, k, v, prefix, 2)
    assert out.shape == expected.shape == (1, 2, 8)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_canonical_export_uses_cached_partial_fallback_and_propagates_native_failures(isolated_backend):
    from functools import wraps

    registry, package, load = isolated_backend
    attempts = []
    broken = False

    @wraps(ops.reference.gemm_epilogue)
    def gemm(*args, **kwargs):
        attempts.append(kwargs["activation"])
        if kwargs["activation"] == "silu":
            raise ValueError("unsupported activation before launch")
        if broken:
            raise RuntimeError("device execution failed")
        return ops.reference.gemm_epilogue(*args, **kwargs)

    package.gemm_epilogue = gemm
    load()
    table = registry.build_table(_platform())
    x, weight = torch.ones(2, 3), torch.ones(4, 3)
    for _ in range(2):
        torch.testing.assert_close(table.gemm_epilogue(x, weight, activation="silu"),
                                   ops.reference.gemm_epilogue(x, weight, activation="silu"))
    assert attempts == ["silu"]
    table.gemm_epilogue(x, weight)
    broken = True
    with pytest.raises(RuntimeError, match="execution failed"):
        table.gemm_epilogue(x, weight)


def test_dynamic_ffi_names_are_not_operator_exports_and_wrong_signatures_fail(isolated_backend):
    registry, package, load = isolated_backend

    def raw(output, x, weight):
        pytest.fail("raw output-first FFI was called")

    def dynamic(name):
        if name == "gemm_epilogue":
            return raw
        raise AttributeError(name)

    package.__getattr__ = dynamic
    load()
    assert registry.build_table(_platform()).selection("gemm_epilogue").backend == "reference"
    package.gemm_epilogue = raw
    with pytest.raises(TypeError, match="gemm_epilogue.*expected"):
        load()

    class PackedFunction:
        @property
        def __signature__(self):
            raise ValueError("native callable has no Python signature")

        def __call__(self, *args, **kwargs):
            pytest.fail("an unadapted native callable was invoked")

    package.gemm_epilogue = PackedFunction()
    with pytest.raises(TypeError, match="gemm_epilogue.*Python operator signature"):
        load()




