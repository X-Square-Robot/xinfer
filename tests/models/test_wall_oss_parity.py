"""Wall-OSS reference composition (``numerics="reference"``) against the Wall-X eager BF16 reference (GPU, external
assets); the optimized composition is tested in ``test_wall_oss_optimized.py``.

Skipped unless these are set (see docs/models/wall_oss_validation.md):

- ``XINFER_WALL_OSS_CHECKPOINT``: checkpoint directory with ``model.safetensors``.
- ``XINFER_WALL_OSS_REFERENCE``: ``requests.pkl`` and ``reference.npz`` from the
  old service capture (raw requests, injected ``{name}/noise``, and the MXFP8
  service's ``{name}/pred``).
- ``XINFER_WALL_OSS_BF16_REFERENCE``: ``bf16_reference.npz`` from the eager Wall-X
  model (``{name}/image_embeds``, ``{name}/kv``, ``{name}/pred``) and its json.

With reference operators the prefill and action stages reproduce the
reference bit for bit when fed the reference's upstream tensors. The vision
stage differs only through the attention/rotary kernels (flash-attn in the
reference), so the full chain is checked against a tolerance: it must be
closer to the BF16 reference than the MXFP8 production service is.
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
    pytest.mark.skipif(not (CKPT and REF and BF16 and torch.cuda.is_available()),
                       reason="Wall-OSS GPU assets not configured"),
]
INSTRUCTION = (
    "Pick a package, flip it to show the barcode, then pass it. "
    "[action mode: latest; collector: 曾祥煜; quality: excellent; robot: 11118]"
)


@pytest.fixture(scope="module")
def setup():
    from xinfer import ops
    from xinfer.models.wall_oss.model import WallOSS
    from xinfer.models.wall_oss.processing import ProcessingConfig
    from xinfer.platforms import resolve_platform

    platform = resolve_platform("cuda")
    table = ops.registry.build_table(platform, reference_only=True)
    with ops.use_ops(table):  # exact prompts: the reference operators are not invariant to padding
        model = WallOSS.from_checkpoint(CKPT, device=platform.device, numerics="reference",
                                        processing=ProcessingConfig(instruction_override=INSTRUCTION, prompt_bucket=0))
    names = [r["name"] for r in json.loads((Path(BF16) / "bf16_reference.json").read_text())["requests"]]
    return dict(
        model=model, device=platform.device, prepare=model.input_processor(), names=names,
        requests=pickle.load(open(Path(REF) / "requests.pkl", "rb")),
        old=np.load(Path(REF) / "reference.npz"), bf16=np.load(Path(BF16) / "bf16_reference.npz"),
    )


def _tensor(array, device, bf16=False):
    x = torch.from_numpy(np.ascontiguousarray(array))
    return (x.view(torch.bfloat16) if bf16 else x).to(device)


def _inputs(s, name):
    prepared = s["prepare"](s["requests"][name])
    x = {k: (v.to(s["device"]) if torch.is_tensor(v) else v) for k, v in prepared.tensors.items()}
    x["noise"] = _tensor(s["old"][f"{name}/noise"], s["device"])
    return x


def test_prefill_and_action_stages_bitwise(setup):
    from xinfer.engine.executor import run_stage

    s, model = setup, setup["model"]
    for name in s["names"]:
        x = _inputs(s, name)
        image = _tensor(s["bf16"][f"{name}/image_embeds"], s["device"], bf16=True)
        kv_ref = _tensor(s["bf16"][f"{name}/kv"], s["device"], bf16=True)
        kv = run_stage(model, "prefill", {**x, "image_embeds": image}, s["device"])["prompt_kv"]
        pred = run_stage(model, "action", {**x, "prompt_kv": kv_ref}, s["device"])["predict_action"]
        assert torch.equal(kv, kv_ref), name
        assert torch.equal(pred, _tensor(s["bf16"][f"{name}/pred"], s["device"])), name


def test_full_chain_closer_than_production_mxfp8(setup):
    from xinfer.engine import Engine
    from xinfer.models.interface import PreparedRequest
    from xinfer.platforms import resolve_platform

    s = setup
    engine = Engine(s["model"], resolve_platform("cuda"))
    for name in s["names"]:
        prepared = s["prepare"](s["requests"][name])
        prepared.tensors["noise"] = torch.from_numpy(s["old"][f"{name}/noise"])
        pred = engine.run(PreparedRequest(prepared.tensors, prepared.meta))["predict_action"]
        ref = torch.from_numpy(s["bf16"][f"{name}/pred"])
        mxfp8 = torch.from_numpy(s["old"][f"{name}/pred"])
        ours = (pred - ref).abs()[..., :20].max()
        service = (mxfp8 - ref).abs()[..., :20].max()
        assert ours < service, (name, float(ours), float(service))


def test_cuda_graphs_match_eager_bitwise(setup):
    from xinfer.engine import Engine, EngineConfig
    from xinfer.engine.cudagraph import CUDAGraphConfig
    from xinfer.models.interface import PreparedRequest
    from xinfer.platforms import resolve_platform

    s = setup
    platform = resolve_platform("cuda")
    eager = Engine(s["model"], platform)
    graphed = Engine(s["model"], platform, EngineConfig(cudagraphs=CUDAGraphConfig()))
    for name in s["names"]:
        prepared = s["prepare"](s["requests"][name])
        prepared.tensors["noise"] = torch.from_numpy(s["old"][f"{name}/noise"])
        request = PreparedRequest(prepared.tensors, prepared.meta)
        assert torch.equal(graphed.run(request)["predict_action"], eager.run(request)["predict_action"]), name
    # Flow and RTC requests use separate action graphs; RTC delays share one.
    prompt_lengths = {s["prepare"](s["requests"][n]).tensors["prompt_len"] for n in s["names"]}
    assert graphed.cudagraph_stats()["action"]["captures"] <= 2 * len(prompt_lengths)


def test_concurrent_multilane_results_equal_single_request(setup):
    """Same inputs alone and under concurrent load (2 lanes per stage, graphs): bitwise equal."""
    from xinfer.engine import Engine, EngineConfig
    from xinfer.engine.cudagraph import CUDAGraphConfig
    from xinfer.models.interface import PreparedRequest
    from xinfer.platforms import resolve_platform

    s = setup
    platform = resolve_platform("cuda")
    requests = []
    for name in s["names"]:
        prepared = s["prepare"](s["requests"][name])
        prepared.tensors["noise"] = torch.from_numpy(s["old"][f"{name}/noise"])
        requests.append((name, PreparedRequest(prepared.tensors, prepared.meta)))
    alone = Engine(s["model"], platform)
    expected = {name: alone.run(request)["predict_action"] for name, request in requests}
    alone.shutdown()
    engine = Engine(s["model"], platform, EngineConfig(
        lanes={"vision": 2, "prefill": 2, "action": 2}, max_running_requests=16, cudagraphs=CUDAGraphConfig()))
    futures = [(name, engine.submit(request)) for _ in range(3) for name, request in requests]
    for name, future in futures:
        assert torch.equal(future.result(timeout=300)["predict_action"], expected[name]), name
    engine.shutdown()


@pytest.fixture(scope="module")
def served_model(setup):
    """The reference composition as served: flash-attn attention, the batch-invariant action stage and the default
    16-token prompt buckets."""
    from xinfer import ops
    from xinfer.models.wall_oss.model import WallOSS
    from xinfer.models.wall_oss.processing import ProcessingConfig
    from xinfer.ops.backends import batch_invariant, flash_attn
    from xinfer.platforms import resolve_platform

    platform = resolve_platform("cuda")
    registry = ops.OpRegistry()
    for name in ops.registry.names():
        registry.define(ops.registry.spec(name))
    flash_attn.register(registry)
    batch_invariant.register(registry)  # batch-invariant linear / RMSNorm
    table = registry.build_table(platform)
    with ops.use_ops(table):
        return WallOSS.from_checkpoint(CKPT, device=platform.device, numerics="reference",
                                       processing=ProcessingConfig(instruction_override=INSTRUCTION))


def test_served_reference_composition_stays_closer_than_production_mxfp8(setup, served_model):
    """The reference composition as served (flash-attn, batch-invariant action stage, prompt buckets): graphs equal
    eager, and every request is closer to the eager BF16 reference than the MXFP8 service is (flash-attn is a
    tolerance backend)."""
    from xinfer.engine import Engine, EngineConfig
    from xinfer.engine.cudagraph import CUDAGraphConfig
    from xinfer.models.interface import PreparedRequest
    from xinfer.platforms import resolve_platform

    s = setup
    platform = resolve_platform("cuda")
    eager = Engine(served_model, platform)
    graphed = Engine(served_model, platform, EngineConfig(cudagraphs=CUDAGraphConfig()))
    for name in s["names"]:
        prepared = s["prepare"](s["requests"][name])
        prepared.tensors["noise"] = torch.from_numpy(s["old"][f"{name}/noise"])
        request = PreparedRequest(prepared.tensors, prepared.meta)
        pred = eager.run(request)["predict_action"]
        assert torch.equal(graphed.run(request)["predict_action"], pred), name
        ref = torch.from_numpy(s["bf16"][f"{name}/pred"])
        ours = (pred - ref).abs()[..., :20].max()
        service = (torch.from_numpy(s["old"][f"{name}/pred"]) - ref).abs()[..., :20].max()
        assert ours < service, (name, float(ours), float(service))


def test_action_batches_equal_single_requests(setup, served_model):
    """Batch-invariant action stage: requests batched B2..B8, with different prompt lengths
    (left-padded) and RTC delays, get the bits each gets alone."""
    import copy

    from xinfer.engine import Engine, EngineConfig
    from xinfer.engine.cudagraph import CUDAGraphConfig
    from xinfer.models.interface import PreparedRequest
    from xinfer.platforms import resolve_platform

    s, model = setup, served_model
    if not model.capabilities().batch_invariant["action"]:
        pytest.skip("action stage operators are not batch-invariant here (needs flash-attn and Triton)")
    platform = resolve_platform("cuda")
    names = sorted(s["requests"])[:16]
    requests, lengths = [], set()
    for i, name in enumerate(names):
        raw = copy.deepcopy(s["requests"][name])
        width = (1280, 1120, 960, 800)[i % 4]  # different image grids -> different prompt lengths
        front = np.asarray(raw["views"]["camera_front"])  # a raw [1, H, W, 3] array in this capture
        raw["views"]["camera_front"] = np.ascontiguousarray(front[..., :width, :])
        prepared = s["prepare"](raw)
        prepared.tensors["noise"] = torch.from_numpy(s["old"][f"{name}/noise"])
        lengths.add(int(prepared.tensors["prompt_len"]))
        requests.append(PreparedRequest(prepared.tensors, prepared.meta))
    assert len(lengths) >= 3
    alone = Engine(model, platform, EngineConfig(cudagraphs=CUDAGraphConfig()))
    expected = [alone.run(r)["predict_action"] for r in requests]
    alone.shutdown()
    config = EngineConfig(lanes={"vision": 2, "prefill": 2, "action": 1}, max_batch={"action": 8},
                          max_running_requests=64, cudagraphs=CUDAGraphConfig())
    engine = Engine(model, platform, config, start=False)
    futures = [(i, engine.submit(r)) for _ in range(2) for i, r in enumerate(requests)]
    engine.start()
    for i, future in futures:
        assert torch.equal(future.result(timeout=600)["predict_action"], expected[i]), i
    stats = dict(engine.stats)
    engine.shutdown()
    batched = sum(v for k, v in stats.items() if k.startswith("dispatch_action_b") and k != "dispatch_action_b1")
    assert batched > 0, stats
    assert stats.get("padded_dispatch_action", 0) > 0, stats


def test_padded_prompts_give_the_actions_of_exact_prompts(setup, served_model):
    """Prompts padded to 16-token buckets get the bits of their exact prompts: eager, with graphs, and batched
    with requests of other buckets."""
    import copy

    from xinfer.engine import Engine, EngineConfig
    from xinfer.engine.cudagraph import CUDAGraphConfig
    from xinfer.models.interface import PreparedRequest
    from xinfer.platforms import resolve_platform

    s, model = setup, served_model
    if not model.capabilities().batch_invariant["action"]:
        pytest.skip("action stage operators are not batch-invariant here (needs flash-attn and Triton)")
    platform = resolve_platform("cuda")
    padded_prepare = model.input_processor()  # prompt_bucket 16; setup's processor keeps exact prompts
    rng = np.random.default_rng(0)
    exact, padded, buckets = [], [], set()
    for i, name in enumerate(sorted(s["requests"])[:12]):
        raw = copy.deepcopy(s["requests"][name])
        for arm in ("follow1_pos", "follow2_pos"):  # random poses vary the state text, so the prompt length
            pose = np.asarray(raw["state"][arm], dtype=np.float64).copy()
            pose[..., :3] = rng.uniform(-0.5, 0.5, 3)
            pose[..., 3:6] = rng.uniform(-np.pi, np.pi, 3)
            raw["state"][arm] = pose
        noise = torch.from_numpy(s["old"][f"{name}/noise"])
        for prepare, out in ((s["prepare"], exact), (padded_prepare, padded)):
            prepared = prepare(raw)
            prepared.tensors["noise"] = noise
            out.append(PreparedRequest(prepared.tensors, prepared.meta))
        assert "prompt_valid" not in exact[-1].tensors and "prompt_valid" in padded[-1].tensors
        buckets.add(padded[-1].tensors["prompt_len"])
    lengths = {r.tensors["prompt_len"] for r in exact}
    assert len(lengths) > len(buckets) > 1, (lengths, buckets)
    eager = Engine(model, platform)
    expected = [eager.run(r)["predict_action"] for r in exact]
    for i, r in enumerate(padded):
        assert torch.equal(eager.run(r)["predict_action"], expected[i]), i
    eager.shutdown()
    graphed = Engine(model, platform, EngineConfig(cudagraphs=CUDAGraphConfig()))
    for i, r in enumerate(padded):
        assert torch.equal(graphed.run(r)["predict_action"], expected[i]), i
    graphed.shutdown()
    config = EngineConfig(lanes={"vision": 2, "prefill": 2, "action": 1}, max_batch={"action": 8},
                          max_running_requests=64, cudagraphs=CUDAGraphConfig())
    engine = Engine(model, platform, config, start=False)
    futures = [(i, engine.submit(r)) for _ in range(2) for i, r in enumerate(padded)]
    engine.start()
    for i, future in futures:
        assert torch.equal(future.result(timeout=600)["predict_action"], expected[i]), i
    stats = dict(engine.stats)
    engine.shutdown()
    assert stats.get("padded_dispatch_action", 0) > 0, stats  # batches mixed buckets


def test_warmup_from_one_request_of_each_kind_covers_every_pose(setup, served_model):
    """Warmed up with one flow and one RTC request, requests with other arm poses need no new graph."""
    import copy

    from xinfer.engine import Engine, EngineConfig
    from xinfer.engine.cudagraph import CUDAGraphConfig
    from xinfer.platforms import resolve_platform

    s, model = setup, served_model
    if not model.capabilities().batch_invariant["action"]:
        pytest.skip("action stage operators are not batch-invariant here (needs flash-attn and Triton)")
    prepare = model.input_processor()  # prompt_bucket 16
    flow = next(n for n in sorted(s["requests"]) if not s["requests"][n].get("use_rtc"))
    rtc = next(n for n in sorted(s["requests"]) if s["requests"][n].get("use_rtc"))
    engine = Engine(model, resolve_platform("cuda"), EngineConfig(
        max_batch={"action": 2}, max_running_requests=16, cudagraphs=CUDAGraphConfig(on_miss_after_warmup="error")))
    engine.warmup([prepare(s["requests"][flow]), prepare(s["requests"][rtc])])
    rng = np.random.default_rng(0)
    requests, buckets = [], set()
    for i in range(24):
        raw = copy.deepcopy(s["requests"][(flow, rtc)[i % 2]])
        for arm in ("follow1_pos", "follow2_pos"):
            raw["state"][arm] = np.concatenate([rng.uniform(-0.5, 0.5, 3), rng.uniform(-np.pi, np.pi, 3),
                                                rng.uniform(0, 1, 1)])
        requests.append(prepare(raw))
        buckets.add(requests[-1].tensors["prompt_len"])
    futures = [engine.submit(r) for r in requests]
    for future in futures:
        future.result(timeout=300)  # CUDAGraphMiss if a stage had no graph
    engine.shutdown()
    assert len(buckets) > 1, buckets
