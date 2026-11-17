"""Torch accelerator APIs used by the shared engine."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import torch


def device_module_for(device: str | torch.device):
    """Return the torch stream and event API for a device, or None on CPU."""
    kind = str(device).partition(":")[0]
    if kind == "cuda":
        return torch.cuda
    if kind == "npu":
        import torch_npu  # noqa: F401 - registers the npu device and torch.npu

        return torch.npu
    return None


def graph_api_for(device: str | torch.device):
    """Return the graph class, pool and capture context for an accelerator."""
    module = device_module_for(device)
    if module is not None and module is not torch.cuda:
        return SimpleNamespace(
            CUDAGraph=module.NPUGraph,
            graph_pool_handle=module.graph_pool_handle,
            graph=partial(module.graph, auto_dispatch_capture=True),
        )
    return module


def jpeg_decoder_for(device: str | torch.device, *, image_quantize=None):
    """Create the selected platform's JPEG decoder, loading no other platform's codec.

    The decoder has a canonical, indexed ``device``, is called with a list of CPU uint8 JPEG tensors,
    and returns CHW uint8 RGB tensors safe to consume on the caller's current device stream. It owns
    any codec resources and stream ordering; ``close()`` is idempotent and runs on the owning thread.
    """
    module = device_module_for(device)
    if module is None:
        raise ValueError('DeviceImage inputs (image_preprocess_device "gpu") '
                         f'need a CUDA or NPU device, not {device}')
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, module.current_device())
    if device.type == "cuda":
        from xinfer.platforms.cuda import CudaJpegDecoder

        return CudaJpegDecoder(device)
    from xinfer.platforms.npu import NpuJpegDecoder

    return NpuJpegDecoder(device) if image_quantize is None else NpuJpegDecoder(device, quantize=image_quantize)


def image_pipeline_for(decoder, resize_image, upload):
    """A platform's optional decode-wave/graph pipeline, using caller-supplied tensor operations."""
    if decoder.device.type != "npu":
        return None
    from xinfer.platforms.npu_images import NpuImagePipeline

    return NpuImagePipeline(decoder, resize_image, upload)
