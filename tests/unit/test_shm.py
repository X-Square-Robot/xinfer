"""Shared-memory transport of prepared requests between a frontend and the engine (CPU)."""

import pickle
import uuid

import torch

from xinfer.frontend.shm import ShmReader, ShmRing
from xinfer.models.interface import DeviceImage


def test_tensors_and_device_images_cross_through_the_buffer():
    """Tensors and the data of DeviceImage values go through the buffer, not the pickled manifest."""
    g = torch.Generator().manual_seed(0)
    jpeg = DeviceImage(torch.randint(0, 256, (1001,), dtype=torch.uint8, generator=g), ((4, 6),),
                       (("bicubic", False),))
    frame = DeviceImage(torch.randint(0, 256, (72, 128, 3), dtype=torch.uint8, generator=g), ((36, 64), (24, 24)),
                        (("bicubic", True), ("bilinear", True)))
    tensors = {"x": torch.randn(2, 3, generator=g), "image_0": jpeg, "image_1": frame, "steps": 5}
    ring = ShmRing(f"xinfer-test-{uuid.uuid4().hex[:8]}", 1, 1 << 20)
    reader = ShmReader(register_cuda=False)
    try:
        packed = ring.pack(0, tensors)
        assert len(pickle.dumps(packed)) < 1024
        out = reader.unpack(packed)
        assert list(out) == list(tensors) and torch.equal(out["x"], tensors["x"]) and out["steps"] == 5
        for name in ("image_0", "image_1"):
            assert isinstance(out[name], DeviceImage) and torch.equal(out[name].data, tensors[name].data)
            assert (out[name].sizes, out[name].resize) == (tensors[name].sizes, tensors[name].resize)
        del out  # views of the buffer
    finally:
        reader.close()
        ring.close()
