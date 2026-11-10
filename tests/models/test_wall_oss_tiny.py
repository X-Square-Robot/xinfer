"""Wall-OSS structure on a tiny random checkpoint (CPU, FP32 numerics not checked).

Checks the weight mapping is complete, the three stages run through the
engine, the stage split equals a joint prefill+action forward, and RTC
pinning returns the pinned actions exactly.
"""

import base64
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file

from xinfer import ops
from xinfer.engine import Engine
from xinfer.engine.cudagraph import warmup_requests
from xinfer.engine.executor import run_stage
from xinfer.models.interface import DeviceImage, PreparedRequest
from xinfer.platforms import resolve_platform

DOF = {
    "follow_left_ee_cartesian_pos_relative": 3, "follow_left_ee_rotation_6D_relative": 6,
    "follow_left_gripper": 1, "follow_right_ee_cartesian_pos_relative": 3,
    "follow_right_ee_rotation_6D_relative": 6, "follow_right_gripper": 1,
    "velocity_decomposed": 3, "height": 1, "head_actions": 2,
}
AGENT = {k.replace("_relative", ""): v for k, v in DOF.items()}
ROWS, IMAGE_TOKEN, H, D = 300, 250, 16, 26


def _config(root: Path):
    cj = {
        "model_type": "qwen2_5_vl", "num_experts": 2, "attention_moe": True, "mlp_moe": True, "norm_moe": True,
        "hidden_act": "silu", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]}, "vocab_size": ROWS,
        "image_token_id": IMAGE_TOKEN, "vision_start_token_id": 251, "vision_end_token_id": 252,
        "experts": [{"hidden_size": 64, "intermediate_size": 96}, {"hidden_size": 32, "intermediate_size": 48}],
        "vision_config": {"depth": 2, "hidden_size": 32, "intermediate_size": 40, "num_heads": 2,
                          "in_chans": 3, "patch_size": 14, "temporal_patch_size": 2, "spatial_merge_size": 2,
                          "window_size": 112, "out_hidden_size": 64, "fullatt_block_indexes": [1],
                          "hidden_act": "silu"},
    }
    cy = {"model_type": "qwen2_5", "dof_config": DOF, "agent_pos_config": AGENT,
          "use_state_string_representation": True, "use_per_token_tau": True, "rtc_rope_gap": 10,
          "rtc_train": {"max_delay": 8}, "data": {"action_horizon_flow": H, "resolution": {}}}
    (root / "config.json").write_text(json.dumps(cj))
    (root / "config.yml").write_text(yaml.safe_dump(cy))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from xinfer.models.wall_oss.config import WallOSSConfig
    from xinfer.models.wall_oss.model import WallOSS, weight_mapping

    root = tmp_path_factory.mktemp("wall_oss")
    _config(root)
    table = ops.registry.build_table(resolve_platform("cpu"))
    torch.manual_seed(0)
    with ops.use_ops(table):
        shell = WallOSS(WallOSSConfig.load(root), root, embedding_rows=ROWS, numerics="reference")
        targets = dict(shell.named_parameters())
        tensors = {}
        for name, source in weight_mapping(shell).items():
            shape = tuple(targets[name].shape)
            if source.keys[0] == "visual.patch_embed.proj.weight":
                shape = (shape[0], 3, 2, 14, 14)
            tensors[source.keys[0]] = torch.randn(shape) * 0.2
        tensors["model.embed_tokens.weight"] = torch.randn(ROWS, 64) * 0.2
        save_file(tensors, str(root / "model.safetensors"))
        stats = {"min.ex_normal": torch.randn(H, D), "delta.ex_normal": torch.rand(H, D) + 0.5}
        torch.save(stats, root / "normalizer_action.pth")
        model = WallOSS.from_checkpoint(root, device="cpu", processing={"prompt_bucket": 0}, numerics="reference")
    model.root = root
    return model


@pytest.fixture(scope="module")
def tiny_optimized(tiny):
    """The optimized composition (the default) on its operators' torch references."""
    from xinfer.models.wall_oss.model import WallOSS

    with ops.use_ops(ops.registry.build_table(resolve_platform("cpu"))):
        model = WallOSS.from_checkpoint(tiny.root, device="cpu", processing={"prompt_bucket": 0})
    model.root = tiny.root
    return model


@pytest.fixture(params=["reference", "optimized"])
def composition(request, tiny, tiny_optimized):
    return {"reference": tiny, "optimized": tiny_optimized}[request.param]


def _request(model, delay=0, rtc_rows=0, seed=1):
    from xinfer.layers.vision.qwen2_5_vit import vision_layout
    from xinfer.models.wall_oss.model import vision_config

    g = torch.Generator().manual_seed(seed)
    grids = [(1, 4, 4)]
    text = torch.randint(0, 200, (12,), generator=g)
    token_ids = torch.cat([text[:5], torch.full((4,), IMAGE_TOKEN), text[5:], torch.full((H,), 210)])[None]
    positions = torch.arange(token_ids.shape[1]).expand(3, 1, -1).clone()
    layout = vision_layout(grids, vision_config(model.config))
    tensors = dict(
        input_ids=token_ids, position_ids=positions,
        image_0=torch.randint(0, 256, (56, 56, 3), dtype=torch.uint8, generator=g),
        dof_mask=torch.ones(1, H, D, dtype=torch.float64), padding_action=torch.zeros(1, H, D),
        noise=torch.randn(1, H, D, generator=g), inference_delay=torch.tensor([delay]), num_steps=5,
        prompt_len=token_ids.shape[1] - H, action_position_base=torch.tensor([int(positions[0, 0, -H])]),
        image_token_index=torch.arange(5, 9), dof_mask_all=False,
        vision_positions=layout["positions"], vision_window_index=layout["window_index"],
        vision_reverse_index=layout["reverse_index"], vision_cu_window=layout["cu_window"],
        vision_cu_full=layout["cu_full"],
    )
    tensors["dof_mask"][..., 20:] = 0
    if rtc_rows:
        tensors["rtc_actions"] = torch.randn(1, rtc_rows, D, generator=g)
    return PreparedRequest(tensors)


def test_all_checkpoint_tensors_are_used(composition):
    unused = set(composition.load_report.unused_keys)
    assert unused <= {"model.norms.0.weight"}  # the prompt's final norm has no consumer


def test_engine_runs_three_stages(composition):
    out = Engine(composition, resolve_platform("cpu")).run(_request(composition))
    assert out["predict_action"].shape == (1, H, D)
    assert torch.isfinite(out["predict_action"]).all()


def test_split_equals_joint_prefill_action_forward(tiny):
    """Step-0 velocity from (prefill stage + action stage) equals one joint pass."""
    request = _request(tiny)
    x = request.tensors
    with torch.inference_mode():
        image = run_stage(tiny, "vision", x, "cpu")["image_embeds"]
        kv = run_stage(tiny, "prefill", {**x, "image_embeds": image}, "cpu")["prompt_kv"]
        p = x["prompt_len"]
        # Joint pass: prompt rows through expert 0, action rows through expert 1,
        # one attention over both with prompt rows blind to action rows.
        from xinfer.layers.rotary import mrope_cos_sin

        h0 = tiny.embed_tokens[x["input_ids"][:, :p]]
        h0 = h0.masked_scatter((x["input_ids"][:, :p] == IMAGE_TOKEN)[..., None].expand_as(h0), image)
        tau = torch.zeros(1, H)
        h1 = tiny.action_head.embed(x["noise"], tau, x["dof_mask"].float()).to(h0.dtype)
        cos, sin = mrope_cos_sin(tiny.inv_freq, x["position_ids"], torch.float32)
        t0 = tiny.rope.tables(cos[:, :, :p], sin[:, :, :p], torch.bfloat16)
        t1 = tiny.rope.tables(cos[:, :, p:], sin[:, :, p:], torch.bfloat16)
        n = p + H
        mask = torch.ones(n, n, dtype=torch.bool).tril()
        mask[p:, p:] = True
        mask = mask[None, None]
        for layer, (l0, l1) in enumerate(zip(tiny.prefill_layers, tiny.action_layers)):
            q0, k0, v0 = (t.transpose(1, 2).contiguous() for t in l0.qkv_states(h0, t0))  # head-major
            q1, k1, v1 = (t.transpose(1, 2).contiguous() for t in l1.qkv_states(h1, t1))
            torch.testing.assert_close(k0, kv[layer, 0], rtol=0, atol=0)
            q, k, v = (torch.cat(a, dim=2) for a in ((q0, q1), (k0, k1), (v0, v1)))
            o = ops.reference.attention(q, k, v, mask=mask).transpose(1, 2).reshape(1, n, -1)
            h0 = h0 + l0.o_proj(o[:, :p])
            h0 = h0 + l0.mlp(l0.post_norm(h0))
            h1 = h1 + l1.o_proj(o[:, p:])
            h1 = h1 + l1.mlp(l1.post_norm(h1))
        joint = tiny.action_head.proj_back(tiny.action_norm(h1).float())
        # The same quantity from the split path (one step from t = 0).
        h = tiny.action_head.embed(x["noise"], tau, x["dof_mask"].float()).to(torch.bfloat16)
        positions = x["position_ids"][:, :, p:]
        cos, sin = mrope_cos_sin(tiny.inv_freq, positions, torch.float32)
        ta = tiny.rope.tables(cos, sin, torch.bfloat16)
        for layer, l1 in enumerate(tiny.action_layers):
            h, _, _ = l1(h, ta, prompt_kv=(kv[layer, 0], kv[layer, 1]))  # no pinned rows: all visible
        split = tiny.action_head.proj_back(tiny.action_norm(h).float())
    torch.testing.assert_close(split, joint, rtol=1e-2, atol=1e-2)


def test_rtc_pins_the_actions_exactly(composition):
    request = _request(composition, delay=3, rtc_rows=15)
    out = Engine(composition, resolve_platform("cpu")).run(request)["predict_action"]
    pinned = request.tensors["rtc_actions"][:, :3]
    expect = ((pinned + 1) / 2) * composition.action_delta[:3] + composition.action_min[:3]
    assert torch.equal(out[:, :3], expect)
    # Rows after the delay are denoised, not copied.
    model = composition
    copied = ((request.tensors["rtc_actions"][:, 3:15] + 1) / 2) * model.action_delta[3:15] + model.action_min[3:15]
    assert not torch.equal(out[:, 3:15], copied)


def test_injected_noise_is_deterministic(composition):
    engine = Engine(composition, resolve_platform("cpu"))
    a = engine.run(_request(composition, seed=7))["predict_action"]
    b = engine.run(_request(composition, seed=7))["predict_action"]
    assert torch.equal(a, b)


def test_the_optimized_composition_is_the_default_and_takes_one_request_per_action_batch(tiny_optimized):
    from xinfer.models.wall_oss.optimized import OPTIMIZED_STAGE_OPS

    assert tiny_optimized.numerics == "optimized"
    assert tiny_optimized.capabilities().batch_sizes["action"] == (1,)
    assert tiny_optimized.capabilities().stage_ops == OPTIMIZED_STAGE_OPS  # what the startup check reads


@pytest.mark.parametrize(("delay", "rtc_rows"), [(0, 0), (3, 15)])
def test_optimized_composition_tracks_the_reference(tiny, tiny_optimized, delay, rtc_rows):
    """The same model with other rounding steps (FP32 epilogues, fused norms): relative error of the actions."""
    platform = resolve_platform("cpu")
    request = _request(tiny, delay=delay, rtc_rows=rtc_rows, seed=3)
    a = Engine(tiny, platform).run(request)["predict_action"]
    b = Engine(tiny_optimized, platform).run(request)["predict_action"]
    assert torch.isfinite(b).all()
    assert float((a - b).norm() / a.norm()) < 0.05


def test_pad_prompt_keeps_the_valid_tokens_and_the_action_tokens(tiny):
    from xinfer.models.wall_oss.config import PAD_TOKEN_ID
    from xinfer.models.wall_oss.processing import pad_prompt

    x = _request(tiny).tensors
    p = x["prompt_len"]
    padded = pad_prompt(x, 32)
    assert padded["prompt_len"] == 32 and padded["prompt_valid"].tolist() == [p]
    assert padded["prompt_valid"].dtype == torch.int32
    ids, positions = padded["input_ids"], padded["position_ids"]
    assert ids.shape == (1, 32 + H) and positions.shape == (3, 1, 32 + H)
    assert torch.equal(ids[:, :p], x["input_ids"][:, :p]) and torch.equal(ids[:, 32:], x["input_ids"][:, p:])
    assert (ids[:, p:32] == PAD_TOKEN_ID).all() and (positions[:, :, p:32] == 0).all()
    assert torch.equal(positions[:, :, :p], x["position_ids"][:, :, :p])
    assert torch.equal(positions[:, :, 32:], x["position_ids"][:, :, p:])
    assert padded["action_position_base"] is x["action_position_base"]
    again = pad_prompt(padded, 48)  # re-padding keeps the same valid tokens
    assert again["prompt_valid"].tolist() == [p] and torch.equal(again["input_ids"][:, :p], ids[:, :p])


def test_left_align_prompts_moves_the_valid_tokens_last():
    from xinfer.models.wall_oss.model import left_align_prompts

    kv = torch.zeros(2, 2, 3, 1, 8, 2)
    lengths = torch.tensor([8, 6, 5], dtype=torch.int32)  # row 1 and 2 left-padded by a batch merge
    valid = torch.tensor([5, 6, 3], dtype=torch.int32)
    for b in range(3):
        start = 8 - int(lengths[b])
        kv[:, :, b, :, start:start + int(valid[b])] = torch.arange(1, int(valid[b]) + 1, dtype=torch.float32)[:, None]
        kv[:, :, b, :, start + int(valid[b]):] = -1.0  # bucket padding
    out, prompt_lengths = left_align_prompts(kv, lengths, valid)
    assert prompt_lengths is valid
    for b in range(3):
        n = int(valid[b])
        assert torch.equal(out[:, :, b, :, 8 - n:], kv[:, :, b, :, 8 - int(lengths[b]):8 - int(lengths[b]) + n])


def test_padded_prompt_gives_the_actions_of_the_exact_prompt(composition):
    """CPU check of the layout: padding changes the reference operators' rounding, so the bits may differ."""
    from xinfer.models.wall_oss.processing import pad_prompt

    request = _request(composition, delay=3, rtc_rows=15)
    padded = PreparedRequest(pad_prompt(request.tensors, 32))
    padded.tensors["input_ids"][:, request.tensors["prompt_len"]:32] = 0  # the tiny vocabulary has no PAD_TOKEN_ID
    engine = Engine(composition, resolve_platform("cpu"))
    exact = engine.run(request)["predict_action"]
    torch.testing.assert_close(engine.run(padded)["predict_action"], exact, rtol=1e-3, atol=1e-3)
    attends_padding = PreparedRequest(dict(padded.tensors, prompt_valid=torch.tensor([32], dtype=torch.int32)))
    assert (engine.run(attends_padding)["predict_action"] - exact).abs().max() > 1e-2


def test_warmup_requests_cover_the_prompt_buckets_of_the_length_range(tiny, monkeypatch):
    from xinfer.models.wall_oss.processing import pad_prompt

    request = _request(tiny)  # a 16-token prompt; its image tokens are at 5..8
    cut = pad_prompt(pad_prompt(request.tensors, 32), 12)  # a smaller bucket, for warmup inputs only
    assert cut["prompt_valid"].tolist() == [12] and torch.equal(cut["input_ids"][:, :12],
                                                                request.tensors["input_ids"][:, :12])
    with pytest.raises(ValueError, match="image tokens"):
        pad_prompt(request.tensors, 8)
    prepared = PreparedRequest(pad_prompt(request.tensors, 16), {"prompt_len_range": (10, 40)})
    assert warmup_requests(tiny, prepared) == [prepared]  # prompt_bucket 0
    monkeypatch.setattr(tiny, "processing", {"prompt_bucket": 16})
    variants = warmup_requests(tiny, prepared)
    assert [v.tensors["prompt_len"] for v in variants] == [48, 32, 16]
    assert [v.tensors["prompt_valid"].item() for v in variants] == [16, 16, 16]
    assert all(v.meta is prepared.meta for v in variants)


def _processors(root, monkeypatch, *options):
    """Input processors of a tiny checkpoint (fake tokenizer), one per ``ProcessingConfig`` field dict."""
    from xinfer.models.wall_oss import processing
    from xinfer.models.wall_oss.config import ACTION_TOKEN, ACTION_TOKEN_ID, WallOSSConfig

    _config(root)
    model_config = replace(WallOSSConfig.load(root), resolution={"face_view": 112, "left_wrist_view": 112})
    for name, shape in (("action", (H, D)), ("propri", (model_config.state_dim,))):
        torch.save({"min.ex_normal": torch.zeros(shape), "delta.ex_normal": torch.ones(shape)},
                   root / f"normalizer_{name}.pth")
    special = {processing.IMAGE_PAD: IMAGE_TOKEN, processing.VISION_START: 251, processing.VISION_END: 252,
               ACTION_TOKEN: ACTION_TOKEN_ID}
    pattern = "(" + "|".join(re.escape(token) for token in special) + ")"
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: SimpleNamespace(
            ids=[special.get(part, 1) for part in re.split(pattern, text) if part]),
        get_added_tokens_decoder=lambda: {index: SimpleNamespace(content=token) for token, index in special.items()})
    monkeypatch.setattr(processing, "load_tokenizer", lambda checkpoint: tokenizer)
    return [processing.WallOSSInputProcessor(root, model_config, processing.ProcessingConfig(**fields))
            for fields in options]


def _observation(views):
    return {"views": views, "state": {"follow1_pos": np.zeros(7), "follow2_pos": np.zeros(7)},
            "instruction": "move the package"}


def test_input_processor_returns_views_in_camera_order(tmp_path, monkeypatch):
    """Raw, base64-JPEG and JPEG-bytes views, decoded and resized concurrently, come back in camera order."""
    cv2 = pytest.importorskip("cv2")
    from xinfer.models.wall_oss import processing

    prepare, = _processors(tmp_path, monkeypatch, dict(noise_seed=1))
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, shape, dtype=np.uint8) for shape in ((137, 221, 3), (221, 137, 3), (84, 112, 3))]
    jpeg = [cv2.imencode(".jpg", image[..., ::-1])[1].tobytes() for image in images[1:]]
    views = dict(zip((camera.wire_key for camera in prepare.config.cameras),
                     [images[0], base64.b64encode(jpeg[0]).decode(), jpeg[1]]))
    prepared = prepare(_observation(views))
    for i, camera in enumerate(prepare.config.cameras):
        expected = processing.resize_view(processing.decode_view(views[camera.wire_key]),
                                          prepare.model_config.resolution.get(camera.view, -1), prepare.config)
        assert torch.equal(prepared.tensors[f"image_{i}"], torch.from_numpy(expected)), camera.wire_key


def test_gpu_preprocessing_sends_the_views_as_they_arrived_with_their_sizes(tmp_path, monkeypatch):
    """image_preprocess_device="gpu": the views are DeviceImage values of the JPEG bytes and frames with the sizes
    the CPU path resizes them to (a PNG is decoded on the CPU and sent as a frame); every other input is
    unchanged."""
    cv2 = pytest.importorskip("cv2")
    from xinfer.models.wall_oss import processing

    cpu, gpu = _processors(tmp_path, monkeypatch, dict(noise_seed=1),
                           dict(noise_seed=1, image_preprocess_device="gpu"))
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, shape, dtype=np.uint8) for shape in ((137, 221, 3), (221, 137, 3), (84, 112, 3))]
    jpeg = cv2.imencode(".jpg", images[1][..., ::-1])[1].tobytes()
    png = cv2.imencode(".png", images[2][..., ::-1])[1].tobytes()
    views = dict(zip((camera.wire_key for camera in cpu.config.cameras),
                     [images[0], base64.b64encode(jpeg).decode(), png]))
    a, b = cpu(_observation(views)), gpu(_observation(views))
    assert list(b.tensors) == list(a.tensors)
    for name, value in a.tensors.items():
        if not name.removeprefix("image_").isdigit():
            assert torch.equal(value, b.tensors[name]) if torch.is_tensor(value) else value == b.tensors[name], name
    staged = [b.tensors[f"image_{i}"] for i in range(3)]
    assert all(isinstance(view, DeviceImage) and view.resize == processing.DEVICE_RESIZE for view in staged)
    assert torch.equal(staged[0].data, torch.from_numpy(images[0]))
    assert staged[1].data.numpy().tobytes() == jpeg
    assert torch.equal(staged[2].data, torch.from_numpy(images[2]))
    assert [view.sizes for view in staged] == [(tuple(a.tensors[f"image_{i}"].shape[:2]),) for i in range(3)]
    assert b.meta["image_size"] == a.meta["image_size"]


def test_image_preprocess_device_is_validated():
    from xinfer.models.wall_oss.processing import ProcessingConfig

    with pytest.raises(ValueError, match="image_preprocess_device"):
        ProcessingConfig(image_preprocess_device="cuda")
