"""WALL-OSS-1.0-Nano structure on a tiny random checkpoint.

For both compositions and the optimized one with ``quantization="mxfp8"`` and ``"nvfp4"`` (the optimized ones on the
torch references of their operators): the weight mapping covers every parameter, the three stages run, an RTC request
returns its pinned rows unchanged, uncommanded dimensions move to the normalized zero action as the source's velocity
padding gives, and the stages are deterministic. The optimized compositions stay close to the reference.
"""

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from xinfer import ops
from xinfer.models.interface import PreparedRequest, StageBatch
from xinfer.platforms import resolve_platform

from wall_oss_1_0_nano_checkpoint import D, H, write_checkpoint

ROWS, P = 300, 20


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from xinfer.models.wall_oss_1_0_nano.config import WallOSS10NanoConfig
    from xinfer.models.wall_oss_1_0_nano.model import WallOSS10Nano, weight_mapping

    root = tmp_path_factory.mktemp("wall-oss-1.0-nano")
    write_checkpoint(root)
    torch.manual_seed(0)
    table = ops.registry.build_table(resolve_platform("cpu"))
    with ops.use_ops(table):
        shell = WallOSS10Nano(WallOSS10NanoConfig.load(root), embedding_rows=ROWS, numerics="reference")
    tensors = {}
    for target, source in weight_mapping(shell).items():
        shape = dict(shell.named_parameters())[target].shape
        for name in source.keys:
            tensors.setdefault(name, torch.randn(shape) * 0.02)  # FP32, as the source checkpoint
    save_file(tensors, str(root / "model.safetensors"))
    with ops.use_ops(table):
        model = WallOSS10Nano.from_checkpoint(root, device="cpu", numerics="reference")
    return model


@pytest.fixture(scope="module")
def tiny_optimized(tiny):
    from xinfer.models.wall_oss_1_0_nano.model import WallOSS10Nano

    with ops.use_ops(ops.registry.build_table(resolve_platform("cpu"))):
        return WallOSS10Nano.from_checkpoint(tiny.checkpoint, device="cpu")


@pytest.fixture(scope="module")
def tiny_mxfp8(tiny):
    from xinfer.models.wall_oss_1_0_nano.model import WallOSS10Nano

    with ops.use_ops(ops.registry.build_table(resolve_platform("cpu"))):
        return WallOSS10Nano.from_checkpoint(tiny.checkpoint, device="cpu", quantization="mxfp8")


@pytest.fixture(scope="module")
def tiny_nvfp4(tiny):
    from xinfer.models.wall_oss_1_0_nano.model import WallOSS10Nano

    with ops.use_ops(ops.registry.build_table(resolve_platform("cpu"))):
        return WallOSS10Nano.from_checkpoint(tiny.checkpoint, device="cpu", quantization="nvfp4")


@pytest.fixture(params=["reference", "optimized", "mxfp8", "nvfp4"])
def composition(request, tiny, tiny_optimized, tiny_mxfp8, tiny_nvfp4):
    return {"reference": tiny, "optimized": tiny_optimized, "mxfp8": tiny_mxfp8, "nvfp4": tiny_nvfp4}[request.param]


def _run(model, stage, inputs):
    batch = StageBatch(stage, 1, inputs)
    out = {n: torch.empty(s.shape, dtype=s.dtype) for n, s in model.stage_outputs(stage, batch).items()}
    getattr(model, model.stage_graph[stage].method)(StageBatch(stage, 1, inputs, out))
    return out


def _inputs(model, rtc_delay=0, seed=0):
    from xinfer.models.layers.vision.qwen3_5_vit import vision_layout

    g = torch.Generator().manual_seed(seed)
    c = model.config
    views = [torch.randint(0, 256, (32, 64, 3), dtype=torch.uint8, generator=g) for _ in range(3)]
    grids = [(1, view.shape[0] // 16, view.shape[1] // 16) for view in views]
    layout = vision_layout(grids, c.model.vision)
    tokens = sum(h * w for _, h, w in grids) // 4
    positions = torch.arange(P + H).expand(3, 1, -1).clone()
    action_positions = positions[..., P:].clone()
    action_positions[..., rtc_delay:] += c.rtc_rope_gap
    mask = torch.ones(1, 1, H, P + H, dtype=torch.bool)
    mask[0, 0, :rtc_delay, P + rtc_delay:] = False
    dof = torch.ones(1, H, D)
    dof[..., 20:] = 0  # velocity, height and head: not commanded on the desktop robot
    inputs = dict(
        **{f"image_{i}": view for i, view in enumerate(views)},
        vision_positions=layout["positions"], vision_pos_index=layout["pos_index"],
        vision_pos_weight=layout["pos_weight"], vision_cu_seqlens=layout["cu_seqlens"],
        input_ids=torch.randint(0, ROWS, (1, P), generator=g), position_ids=positions[..., :P].clone(),
        image_token_index=torch.arange(2, 2 + tokens), prompt_len=P, prompt_valid=torch.tensor([P]),
        noise=torch.randn(1, H, D, generator=g), dof_mask=dof, dof_mask_all=False,
        padding_action=torch.full((1, H, D), -0.5), action_positions=action_positions, action_mask=mask)
    if rtc_delay:
        inputs["rtc_actions"] = torch.randn(1, H, D, generator=g)
        inputs["inference_delay"] = torch.tensor([rtc_delay])
    return inputs


def _sample(model, inputs):
    with torch.no_grad():
        inputs["image_embeds"] = _run(model, "vision", inputs)["image_embeds"]
        inputs["prompt_kv"] = _run(model, "prefill", inputs)["prompt_kv"]
        return _run(model, "action", inputs)["predict_action_norm"]


def test_the_weight_mapping_loads_every_parameter(tiny, composition):
    assert tiny.load_report.loaded == len(dict(tiny.named_parameters()))
    assert composition.load_report.unused_keys == ()


def test_the_optimized_composition_is_the_default(tiny_optimized):
    assert tiny_optimized.numerics == "optimized"


def test_quantization_selects_only_the_requested_stage(tiny):
    from xinfer.models.wall_oss_1_0_nano.model import WallOSS10Nano

    with ops.use_ops(ops.registry.build_table(resolve_platform("cpu"))):
        model = WallOSS10Nano.from_checkpoint(tiny.checkpoint, device="cpu",
                                        quantization={"groups": {"prefill": "mxfp8"}})
    declared = model.capabilities().stage_ops
    assert model.prefill_stack.quant == "mxfp8" and model.vision.quant is None
    assert "mxfp8_gemm_epilogue" in declared["prefill"]
    assert not any(op.startswith("mxfp8") for stage in ("vision", "action") for op in declared[stage])
    assert not hasattr(model.prefill_stack.layers[0], "qkv_zba")
    assert model.prefill_stack.layers[0].qkv_zba_q.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("quantization", [{"default": "nvfp4"}, {"groups": {"action": "mxfp8"}},
                                          {"options": {"mxfp8": {"share_scale_128": False}}}])
def test_incompatible_quantization_is_rejected(tiny, quantization):
    with pytest.raises(ValueError, match="quantization"):
        type(tiny).from_checkpoint(tiny.checkpoint, device="cpu", quantization=quantization)


def test_stages_run_and_are_deterministic(composition):
    inputs = _inputs(composition)
    first = _sample(composition, inputs)
    assert first.shape == (1, H, D) and first.dtype == torch.float32 and torch.isfinite(first).all()
    assert torch.equal(first, _sample(composition, _inputs(composition)))
    assert inputs["image_embeds"].shape == (inputs["image_token_index"].numel(), 64)


def test_rtc_returns_the_pinned_rows_and_changes_the_rest(composition):
    flow = _sample(composition, _inputs(composition))
    inputs = _inputs(composition, rtc_delay=3)
    rtc = _sample(composition, inputs)
    assert torch.equal(rtc[:, :3], inputs["rtc_actions"][:, :3])
    assert not torch.equal(rtc[:, 3:], flow[:, 3:])


def test_uncommanded_dimensions_move_to_the_normalized_zero_action(composition):
    inputs = _inputs(composition)
    out = _sample(composition, inputs)
    times = np.linspace(0, 1, 11, dtype=np.float32) * np.float32(0.999)
    noise, target = inputs["noise"][..., 20:].numpy(), np.float32(-0.5)
    expected = noise.copy()
    for k in range(10):  # z += dt · (padding - noise), as the source's v_padding
        expected = expected + np.float32(times[k + 1] - times[k]) * (target - noise)
    np.testing.assert_allclose(out[..., 20:].numpy(), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", ["optimized", "mxfp8", "nvfp4"])
@pytest.mark.parametrize("rtc_delay", [0, 3])
def test_optimized_composition_stays_close_to_the_reference(tiny, tiny_optimized, tiny_mxfp8, tiny_nvfp4, rtc_delay,
                                                            name):
    """The same model with other rounding steps (``mxfp8``: also MXFP8 GEMMs in the ViT blocks and the prompt stream,
    ``nvfp4`` NVFP4 GEMMs in the ViT and the prompt stream, whose E4M3 / E2M1 rounding the random layers amplify in
    the K/V; a wrong row order or scale gives errors near 1): each stage from the same inputs, relative error."""
    def relative(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm())

    model = {"optimized": tiny_optimized, "mxfp8": tiny_mxfp8, "nvfp4": tiny_nvfp4}[name]
    bounds = {"optimized": (0.05, 0.05), "mxfp8": (0.05, 0.3), "nvfp4": (0.1, 0.4)}[name]  # vision, prefill
    inputs = _inputs(tiny, rtc_delay=rtc_delay)
    with torch.no_grad():
        image = _run(tiny, "vision", inputs)["image_embeds"]
        assert relative(_run(model, "vision", inputs)["image_embeds"], image) < bounds[0]
        inputs["image_embeds"] = image
        kv = _run(tiny, "prefill", inputs)["prompt_kv"]
        assert relative(_run(model, "prefill", inputs)["prompt_kv"], kv) < bounds[1]
        inputs["prompt_kv"] = kv
        out = _run(model, "action", inputs)["predict_action_norm"]
        assert relative(out, _run(tiny, "action", inputs)["predict_action_norm"]) < 0.05


@pytest.mark.parametrize("delay", [0, 3])
def test_prompt_buckets_preserve_valid_tokens_and_rtc(composition, delay):
    from xinfer.engine.cudagraph import warmup_requests

    tiny = composition
    prepared = PreparedRequest(_inputs(tiny, rtc_delay=delay), {"prompt_len_range": (17, 40)})
    variants = warmup_requests(tiny, prepared)
    assert [v.tensors["prompt_len"] for v in variants] == [48, 32]
    expected = _sample(tiny, dict(prepared.tensors))
    for variant in variants:
        x, width = variant.tensors, variant.tensors["prompt_len"]
        assert torch.equal(x["input_ids"][:, :P], prepared.tensors["input_ids"])
        assert x["action_positions"] is prepared.tensors["action_positions"]
        assert not x["action_mask"][..., P:width].any()
        assert torch.equal(x["action_mask"][..., width:], prepared.tensors["action_mask"][..., P:])
        result = _sample(tiny, dict(x))
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-4)
        if delay:
            assert torch.equal(result[:, :delay], x["rtc_actions"][:, :delay])
    restored = tiny.pad(variants[0], "prompt", P)
    for name in ("input_ids", "position_ids", "prompt_valid", "action_mask"):
        assert torch.equal(restored.tensors[name], prepared.tensors[name])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_warmed_prompt_buckets_replay_without_capture(tiny):
    from xinfer.engine import Engine, EngineConfig
    from xinfer.engine.cudagraph import CUDAGraphConfig, warmup_requests

    platform = resolve_platform("cuda")
    with ops.use_ops(ops.registry.build_table(platform, reference_only=True)):
        model = type(tiny).from_checkpoint(tiny.checkpoint, device=platform.device, numerics="reference")
    engine = Engine(model, platform, EngineConfig(cudagraphs=CUDAGraphConfig(on_miss_after_warmup="error")))
    eager = Engine(model, platform)
    requests = [PreparedRequest(_inputs(model, rtc_delay=d), {"prompt_len_range": (17, 40)}) for d in (0, 3)]
    try:
        engine.warmup(requests)
        warmed = engine.cudagraph_stats()
        assert warmed["prefill"]["graphs"] == 2
        for prepared in requests:
            for variant in warmup_requests(model, prepared):
                expected = eager.run(variant)["predict_action_norm"]
                for _ in range(2):
                    assert torch.equal(engine.run(variant)["predict_action_norm"], expected)
        served = engine.cudagraph_stats()
        for stage in warmed:
            assert served[stage]["captures"] == warmed[stage]["captures"]
            assert served[stage]["eager"] == warmed[stage]["eager"]
    finally:
        engine.shutdown()
        eager.shutdown()
