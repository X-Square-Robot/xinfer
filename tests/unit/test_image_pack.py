"""ImageNormalize and ImagePack: lookup and patch packing of uint8 views equal the NumPy computation, also on CUDA
and in graphs."""

import numpy as np
import pytest
import torch

from xinfer.models.layers.image_pack import ImageNormalize, ImagePack

SHAPES = ((56, 112, 3), (84, 28, 3), (28, 28, 3))


def _numpy_pack(image, lut, p, t, m):
    h, w = image.shape[:2]
    x = np.stack([lut[c][image[:, :, c]] for c in range(3)])
    x = np.broadcast_to(x[None], (t, 3, h, w)).reshape(t, 3, h // p // m, m, p, w // p // m, m, p)
    return x.transpose(2, 5, 3, 6, 1, 0, 4, 7).reshape(-1, 3 * t * p * p)


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    lut = rng.standard_normal((3, 256)).astype(np.float32)  # channel-specific, so channel mix-ups show
    return lut, [rng.integers(0, 256, shape, dtype=np.uint8) for shape in SHAPES]


def test_normalize_equals_numpy_lookup_per_channel_and_shared():
    lut, images = _inputs()
    per_channel, shared = ImageNormalize(lut), ImageNormalize(lut[0])
    for image in images:
        view = torch.from_numpy(image)
        assert torch.equal(per_channel(view), torch.from_numpy(np.stack([lut[c][image[:, :, c]] for c in range(3)])))
        assert torch.equal(shared(view), torch.from_numpy(lut[0][image].transpose(2, 0, 1)))


@pytest.mark.parametrize("p,t,m", [(14, 2, 2), (7, 3, 2), (14, 1, 1)])
def test_rows_equal_numpy_packing(p, t, m):
    lut, images = _inputs()
    out = ImagePack(lut, patch_size=p, temporal_patch_size=t, merge_size=m)([torch.from_numpy(i) for i in images])
    assert torch.equal(out, torch.from_numpy(np.concatenate([_numpy_pack(i, lut, p, t, m) for i in images])))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_replay_equals_cpu():
    lut, images = _inputs()
    layer = ImagePack(lut, patch_size=14, temporal_patch_size=2, merge_size=2)
    expected = layer([torch.from_numpy(i) for i in images])
    layer = layer.cuda()
    static = [torch.from_numpy(i).cuda() for i in images]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():  # as the engine captures
        layer(static)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = layer(static)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), expected)
    _, changed = _inputs(seed=1)  # new pixels in the captured inputs, as a new request brings
    for buffer, image in zip(static, changed):
        buffer.copy_(torch.from_numpy(image))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), torch.from_numpy(np.concatenate([_numpy_pack(i, lut, 14, 2, 2) for i in changed])))
