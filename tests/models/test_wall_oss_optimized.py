"""Wall-OSS ``numerics="optimized"`` against the Wall-X eager BF16 reference (GPU, xcompute, external assets).

Uses the assets of ``test_wall_oss_parity.py`` (``XINFER_WALL_OSS_CHECKPOINT``, ``XINFER_WALL_OSS_REFERENCE``,
``XINFER_WALL_OSS_BF16_REFERENCE``) and ``XINFER_XCOMPUTE_LIBRARY``, an xcompute build with the unified entries
(``651195d``) and its Python package importable.

The composition rounds at other points than the eager model, so each stage is compared with the reference's
result from the reference's inputs to it within a tolerance, the full chain must stay closer to the BF16 reference
than the MXFP8 production service, and graph replays must equal eager. The quick100 precision gate is the
end-to-end check.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

CKPT = os.environ.get("XINFER_WALL_OSS_CHECKPOINT")
REF = os.environ.get("XINFER_WALL_OSS_REFERENCE")
BF16 = os.environ.get("XINFER_WALL_OSS_BF16_REFERENCE")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.assets,
    pytest.mark.skipif(not (CKPT and REF and BF16 and os.environ.get("XINFER_XCOMPUTE_LIBRARY")
                            and torch.cuda.is_available()),
                       reason="Wall-OSS GPU assets or xcompute not configured"),
]
INSTRUCTION = (
    "Pick a package, flip it to show the barcode, then pass it. "
    "[action mode: latest; collector: 曾祥煜; quality: excellent; robot: 11118]"
)


@pytest.fixture(scope="module")
def setup():
    from xinfer import ops
    from xinfer.models.wall_oss.model import WallOSS
    from xinfer.models.wall_oss.optimized import OPTIMIZED_STAGE_OPS
    from xinfer.models.wall_oss.processing import ProcessingConfig
    from xinfer.platforms import resolve_platform

    ops.registry.load_plugins()
    platform = resolve_platform("cuda")
    table = ops.registry.build_table(platform)
    if table.selection("gemm_epilogue").backend != "xcompute":
        pytest.skip("needs the xcompute Python package with the unified entries")
    with ops.use_ops(table):  # exact prompts: the fixtures' K/V have the prompt's length
        model = WallOSS.from_checkpoint(CKPT, device=platform.device,
                                        processing=ProcessingConfig(instruction_override=INSTRUCTION, prompt_bucket=0))
    assert model.numerics == "optimized"
    xcompute_ops = {op for stage in OPTIMIZED_STAGE_OPS.values() for op in stage} - {
        "varlen_attention", "causal_attention", "linear", "silu"}
    assert {op: table.selection(op).backend for op in xcompute_ops} == {op: "xcompute" for op in xcompute_ops}
    names = [r["name"] for r in json.loads((Path(BF16) / "bf16_reference.json").read_text())["requests"]]
    return dict(model=model, device=platform.device, prepare=model.input_processor(), names=names,
                requests=pickle.load(open(Path(REF) / "requests.pkl", "rb")),
                old=np.load(Path(REF) / "reference.npz"), bf16=np.load(Path(BF16) / "bf16_reference.npz"))


def _tensor(array, device, bf16=False):
    x = torch.from_numpy(np.ascontiguousarray(array))
    return (x.view(torch.bfloat16) if bf16 else x).to(device)


def _inputs(s, name):
    prepared = s["prepare"](s["requests"][name])
    x = {k: (v.to(s["device"]) if torch.is_tensor(v) else v) for k, v in prepared.tensors.items()}
    x["noise"] = _tensor(s["old"][f"{name}/noise"], s["device"])
    return x


def _relative(a, b):
    return float((a.float().cpu() - b.float().cpu()).norm() / b.float().cpu().norm())


def test_stages_are_close_to_the_reference_from_its_inputs(setup):
    from xinfer.engine.executor import run_stage

    s, model = setup, setup["model"]
    for name in s["names"]:
        x = _inputs(s, name)
        image_ref = _tensor(s["bf16"][f"{name}/image_embeds"], s["device"], bf16=True)
        kv_ref = _tensor(s["bf16"][f"{name}/kv"], s["device"], bf16=True)
        image = run_stage(model, "vision", x, s["device"])["image_embeds"]
        kv = run_stage(model, "prefill", {**x, "image_embeds": image_ref}, s["device"])["prompt_kv"]
        pred = run_stage(model, "action", {**x, "prompt_kv": kv_ref}, s["device"])["predict_action"]
        assert _relative(image, image_ref) < 0.05, name
        assert _relative(kv, kv_ref) < 0.05, name
        assert _relative(pred, _tensor(s["bf16"][f"{name}/pred"], s["device"])) < 0.05, name


def test_full_chain_is_closer_than_the_mxfp8_service_and_graphs_equal_eager(setup):
    from xinfer.engine import Engine, EngineConfig
    from xinfer.engine.cudagraph import CUDAGraphConfig
    from xinfer.models.interface import PreparedRequest
    from xinfer.platforms import resolve_platform

    s = setup
    platform = resolve_platform("cuda")
    eager = Engine(s["model"], platform)
    graphed = Engine(s["model"], platform, EngineConfig(cudagraphs=CUDAGraphConfig()))
    try:
        for name in s["names"]:
            prepared = s["prepare"](s["requests"][name])
            prepared.tensors["noise"] = torch.from_numpy(s["old"][f"{name}/noise"])
            request = PreparedRequest(prepared.tensors, prepared.meta)
            pred = eager.run(request)["predict_action"]
            for _ in range(2):
                assert torch.equal(graphed.run(request)["predict_action"], pred), name
            ref = torch.from_numpy(s["bf16"][f"{name}/pred"])
            ours = (pred - ref).abs()[..., :20].max()
            service = (torch.from_numpy(s["old"][f"{name}/pred"]) - ref).abs()[..., :20].max()
            assert ours < service, (name, float(ours), float(service))
    finally:
        eager.shutdown()
        graphed.shutdown()
