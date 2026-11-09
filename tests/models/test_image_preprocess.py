"""``processing.image_preprocess_device="gpu"`` on the served models (GPU, torchvision, external assets).

The Wall-OSS parity assets are configured by ``XINFER_WALL_OSS_CHECKPOINT`` and ``XINFER_WALL_OSS_REFERENCE``.
The recorded dummy-client wrist views are encoded as JPEG and the front view stays a raw frame.

The device path's views stay within a stated pixel tolerance of the CPU path's, and serving with the option
replays the CUDA graphs warmed with CPU-prepared requests (same keys) and captures none after warmup.
"""

import base64
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.assets,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

# Mean and largest per-channel |GPU − CPU| of the uint8 views. nvJPEG and libjpeg-turbo reconstruct sharp chroma
# edges differently: on camera images a few pixels differ by tens of levels while the mean stays far below one
# level, and on uniform noise, where every pixel is an edge, the mean is several levels (Wall-X dummy-client
# views: 5.5 / 55). Resizing alone (a raw frame) differs by at most one level.
NOISE_JPEG = (8.0, 96)
FRAME = (0.01, 1)


def _wall_oss_requests():
    import cv2

    requests = list(pickle.load(open(Path(os.environ["XINFER_WALL_OSS_REFERENCE"]) / "requests.pkl", "rb")).values())
    out = []
    for observation in requests[:6]:
        views = dict(observation["views"])
        for key in ("camera_left", "camera_right"):
            if isinstance(views[key], np.ndarray):
                frame = views[key][0] if views[key].ndim == 4 else views[key]
                views[key] = base64.b64encode(cv2.imencode(".jpg", frame[..., ::-1])[1].tobytes()).decode()
        out.append(dict(observation, views=views))
    return out


@pytest.fixture(scope="module")
def served():
    """A Wall-OSS policy and its CPU-path processor for the same recorded requests."""
    pytest.importorskip("torchvision")
    checkpoint = os.environ.get("XINFER_WALL_OSS_CHECKPOINT")
    if not checkpoint or not os.environ.get("XINFER_WALL_OSS_REFERENCE"):
        pytest.skip("Wall-OSS assets not configured")
    from xinfer import Policy
    from xinfer.engine.cudagraph import CUDAGraphConfig
    from xinfer.models.registry import get_model_class

    policy = Policy("wall_oss", checkpoint, cudagraphs=CUDAGraphConfig(on_miss_after_warmup="error"),
                    image_preprocess_device="gpu")
    cpu_prepare = get_model_class("wall_oss").build_processors(checkpoint)[0]
    yield policy, cpu_prepare, _wall_oss_requests()
    policy.engine.shutdown()


def test_device_views_are_within_tolerance_of_the_cpu_views(served):
    from xinfer.engine.device_images import DeviceImageDecoder
    from xinfer.models.interface import DeviceImage

    policy, cpu_prepare, requests = served
    decode = DeviceImageDecoder(policy.platform.device)
    for index, observation in enumerate(requests):
        cpu, gpu = cpu_prepare(observation), policy.prepare(observation)
        views = decode(gpu.tensors)
        for name, value in cpu.tensors.items():
            if not name.startswith("image_"):
                continue
            ours = views[name].cpu()
            if name.removeprefix("image_").isdigit():
                assert isinstance(gpu.tensors[name], DeviceImage), (index, name)
                assert ours.shape == value.shape and ours.dtype == torch.uint8, (index, name)
                diff = (ours.int() - value.int()).abs().float()
                mean, top = FRAME if gpu.tensors[name].data.dim() == 3 else NOISE_JPEG
                assert diff.mean() <= mean and diff.max() <= top, (index, name, diff.mean().item(), diff.max().item())
            else:  # image_grid_thw, image_token_index
                assert torch.equal(ours, value), (index, name)


def test_option_replays_the_cpu_path_graphs_without_new_captures(served):
    """Warmed with CPU-prepared requests, the engine serves device-prepared ones from the same graphs."""
    policy, cpu_prepare, requests = served
    engine = policy.engine
    engine.warmup([cpu_prepare(observation) for observation in requests])
    keys = {stage: [set(lane.cudagraphs._graphs) for lane in lanes] for stage, lanes in engine.lanes.items()}
    captures = {stage: counters["captures"] for stage, counters in engine.cudagraph_stats().items()}
    futures = [engine.submit(policy.prepare(observation)) for observation in requests]
    for future in futures:
        future.result(timeout=120)
    assert {stage: [set(lane.cudagraphs._graphs) for lane in lanes] for stage, lanes in engine.lanes.items()} == keys
    assert {stage: counters["captures"] for stage, counters in engine.cudagraph_stats().items()} == captures
