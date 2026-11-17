"""An image prefetch handle belongs to the processor and engine that share it."""

import queue
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from xinfer.engine.image_prefetch import Boundary, ImagePrefetch, Ticket
from xinfer.models.interface import DeviceImage, bind_image_prefetch
from xinfer.models.processing import device_image, device_images_ready


@pytest.mark.parametrize("first_closed", [0, 1])
def test_two_engines_keep_their_own_tickets_after_either_one_closes(first_closed):
    queues = [queue.Queue(), queue.Queue()]
    handles = [ImagePrefetch(target) for target in queues]
    image = DeviceImage(torch.arange(3, dtype=torch.uint8), ((1, 1),), (("bilinear", False),))
    results = [handle.hand_off(image) for handle in handles]
    for target, result, handle in zip(queues, results, handles):
        ticket = target.get_nowait()
        assert isinstance(ticket, Ticket) and ticket.image is result
        handle.flush()
        assert isinstance(target.get_nowait(), Boundary)
    assert results[0].prefetch is not results[1].prefetch
    handles[first_closed].close()
    assert handles[first_closed].hand_off(image) is image
    handles[first_closed].flush()
    assert queues[first_closed].empty()
    other = 1 - first_closed
    result = handles[other].hand_off(image)
    assert queues[other].get_nowait().image is result


def test_processor_binding_is_explicit_and_survives_view_worker_threads():
    class Processor:
        def bind_image_prefetch(self, prefetch):
            self.prefetch = prefetch

        def view(self, value):
            return device_image(value, ((1, 1),), (("bilinear", False),), prefetch=self.prefetch)

    import numpy as np

    targets = [queue.Queue(), queue.Queue()]
    processors = [Processor(), Processor()]
    for processor, target in zip(processors, targets):
        bind_image_prefetch(processor, ImagePrefetch(target))
    pixels = np.arange(3, dtype=np.uint8).reshape(1, 1, 3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(processor.view, pixels) for processor in processors]
        views = [future.result() for future in futures]
    for processor, target, view in zip(processors, targets, views):
        assert target.get_nowait().image is view
        device_images_ready(processor.prefetch)
        assert isinstance(target.get_nowait(), Boundary)
    assert device_image(pixels, ((1, 1),), (("bilinear", False),)).prefetch is None
    bind_image_prefetch(lambda request: request, processors[0].prefetch)  # processors may omit the capability


def test_same_device_request_replay_does_not_reuse_another_engines_decode(monkeypatch):
    from types import SimpleNamespace

    from xinfer.engine import device_images

    target = torch.device("cpu")
    owners = [ImagePrefetch(queue.Queue()), ImagePrefetch(queue.Queue())]
    original = DeviceImage(torch.arange(3, dtype=torch.uint8), ((1, 1),), (("bilinear", False),))
    image = owners[0].hand_off(original)
    image.prefetch.device = target
    image.prefetch.view = torch.tensor([17])
    image.prefetch.txn = object()
    decoded = []

    def decode(images):
        decoded.extend(images)
        return [torch.tensor([23]) for _ in images]

    monkeypatch.setattr(device_images, "jpeg_decoder_for", lambda device: SimpleNamespace(device=target))
    monkeypatch.setattr(device_images, "image_pipeline_for", lambda *args: SimpleNamespace(decode_many=decode))
    first = device_images.DeviceImageDecoder(target, prefetch=owners[0])
    second = device_images.DeviceImageDecoder(target, prefetch=owners[1])
    assert first({"camera": image})["camera"] is image.prefetch.view
    assert not decoded
    assert second({"camera": image})["camera"].item() == 23
    assert len(decoded) == 1 and decoded[0].prefetch is None
    assert decoded[0].data is image.data and image.prefetch.txn is not None
