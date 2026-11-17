"""The optimized action loop's o-projection on the CUDA kernel library (no Wall-OSS assets).

The intree replay's BNSD economy (the wall-oss fast qkv/BNSD chains) has the NPU's fused routes hand
the action loop a head-major ``[H, M, D]`` attention output, which the Ascend backend's o-projection
reads directly. The reference composition normalizes that form; the CUDA backend's ``gemm`` takes
packed ``[M, K]`` rows only, so the loop must ask the attention for the BNSD form only when its fused
QKV armed it and feed the projection packed rows otherwise — the final-review round caught the loop
passing ``return_bnsd=True`` unconditionally and the o-projection fallback then failing on CUDA with
``x must be [M, K]``. This file reproduces that path on the tiny random checkpoint (the asset-gated
tests skip without the Wall-OSS checkpoint, which left the composition unexercised on CUDA).
"""

import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from xinfer import ops
from xinfer.engine import Engine
from xinfer.models.interface import PreparedRequest
from xinfer.platforms import resolve_platform

pytestmark = [pytest.mark.gpu]

DOF = {
    "follow_left_ee_cartesian_pos_relative": 3, "follow_left_ee_rotation_6D_relative": 6,
    "follow_left_ee_gripper": 1, "follow_right_ee_cartesian_pos_relative": 3,
    "follow_right_ee_rotation_6D_relative": 6, "follow_right_ee_gripper": 1,
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


def _checkpoint(root: Path, device):
    """The tiny random checkpoint of test_wall_oss_tiny.py, built on ``device`` (optimized numerics)."""
    from xinfer.models.wall_oss.config import WallOSSConfig
    from xinfer.models.wall_oss.model import WallOSS, weight_mapping

    _config(root)
    table = ops.registry.build_table(resolve_platform(device))
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
        model = WallOSS.from_checkpoint(root, device=device, processing={"prompt_bucket": 0})
    model.root = root
    return model, table


def _request(model, seed=1):
    from xinfer.models.layers.vision.qwen2_5_vit import vision_layout
    from xinfer.models.wall_oss.model import vision_config

    g = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    grids = [(1, 4, 4)]
    text = torch.randint(0, 200, (12,), generator=g)
    token_ids = torch.cat([text[:5], torch.full((4,), IMAGE_TOKEN), text[5:], torch.full((H,), 210)])[None]
    positions = torch.arange(token_ids.shape[1]).expand(3, 1, -1).clone()
    layout = vision_layout(grids, vision_config(model.config))
    tensors = dict(
        input_ids=token_ids.to(device), position_ids=positions.to(device),
        image_0=torch.randint(0, 256, (56, 56, 3), dtype=torch.uint8, generator=g).to(device),
        dof_mask=torch.ones(1, H, D, dtype=torch.float64, device=device),
        padding_action=torch.zeros(1, H, D, device=device),
        noise=torch.randn(1, H, D, generator=g).to(device),
        inference_delay=torch.tensor([0], device=device), num_steps=5,
        prompt_len=token_ids.shape[1] - H, action_position_base=torch.tensor([int(positions[0, 0, -H])]),
        image_token_index=torch.arange(5, 9), dof_mask_all=False,
        vision_positions=layout["positions"].to(device), vision_window_index=layout["window_index"].to(device),
        vision_reverse_index=layout["reverse_index"].to(device), vision_cu_window=layout["cu_window"].to(device),
        vision_cu_full=layout["cu_full"].to(device),
    )
    return PreparedRequest(tensors)


@pytest.fixture(scope="module")
def tiny_on_cuda(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    ops.registry.load_plugins()
    root = tmp_path_factory.mktemp("wall_oss_action_gpu")
    model, table = _checkpoint(root, "cuda")
    if table.selection("gemm_epilogue").library != "kernels":
        pytest.skip("needs the CUDA kernel library (docs/kernels.md)")
    return model


@pytest.fixture(scope="module")
def tiny_on_cpu(tmp_path_factory):
    root = tmp_path_factory.mktemp("wall_oss_action_cpu")
    model, _ = _checkpoint(root, "cpu")
    return model


def test_the_action_loop_projects_packed_attention_rows_on_cuda(tiny_on_cuda):
    """The o-projection fallback must take packed ``[M, H·D]`` rows on CUDA, not the NPU's BNSD form.

    The loop's QKV on this backend returns packed Q, so its attention output must stay packed: with the
    BNSD form asked for unconditionally the projection's gemm raises ``x must be [M, K]`` (the P0 of the
    5090 final round; the Ascend backend accepts the BNSD form, which kept the NPU runs green).
    """
    request = _request(tiny_on_cuda)
    out = Engine(tiny_on_cuda, resolve_platform("cuda")).run(request)["predict_action"]
    assert out.shape == (1, H, D)
    assert torch.isfinite(out).all()


def test_the_cuda_action_loop_matches_the_reference_composition(tiny_on_cuda, tiny_on_cpu):
    """The optimized composition on the CUDA kernels stays close to its torch references on CPU."""
    cuda = Engine(tiny_on_cuda, resolve_platform("cuda")).run(_request(tiny_on_cuda))["predict_action"]
    with ops.use_ops(ops.registry.build_table(resolve_platform("cpu"))):
        cpu = Engine(tiny_on_cpu, resolve_platform("cpu")).run(_request(tiny_on_cpu))["predict_action"]
    assert cuda.shape == cpu.shape
    assert (cuda.float().cpu() - cpu.float()).norm() / cpu.float().norm() < 2e-2
