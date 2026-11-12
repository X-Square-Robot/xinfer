"""Torch accelerator APIs used by the shared engine.

Each torch-vendor package mirror the ``torch.cuda`` API (``torch.npu`` does). ``device_module_for``
is the one lookup: the engine holds ``self._device_module`` and calls ``.Stream`` / ``.Event`` /
``.stream(...)`` on it instead of writing ``torch.cuda.X`` directly, so a new vendor needs only
this function plus its ``platforms/<device>.py``. Platforms whose numbering is not torch's (RPU) return ``None``.
"""

from __future__ import annotations

from types import SimpleNamespace


def device_module_for(device):
    """Return the torch stream and event API for a device, or None on CPU or a non-torch platform."""
    import torch

    kind = str(device).partition(":")[0]
    if kind == "cuda":
        return torch.cuda
    if kind == "npu":
        import torch_npu  # noqa: F401 - registers the npu device and torch.npu

        return torch.npu
    return None


def graph_api_for(device):
    """Return the graph class, pool and capture context for an accelerator.

    CUDA answers with ``torch.cuda``. A vendor whose graph API is named differently
    (``NPUGraph`` on Ascend) is adapted once here so the rest of the engine can keep
    the same call shape as ``CUDAGraph``.
    """
    module = device_module_for(device)
    if module is None:
        return None
    import torch

    if module is torch.cuda:
        return module
    from functools import partial

    return SimpleNamespace(
        CUDAGraph=module.NPUGraph,
        graph_pool_handle=module.graph_pool_handle,
        graph=partial(module.graph, auto_dispatch_capture=True),
    )


def jpeg_decoder_for(device):
    """Create the selected platform's JPEG decoder, loading no other platform's codec.

    The decoder has a canonical, indexed ``device``, is called with a list of CPU uint8 JPEG tensors,
    and returns CHW uint8 RGB tensors safe to consume on the caller's current device stream. It owns
    any codec resources and stream ordering; ``close()`` is idempotent and runs on the owning thread.
    """
    module = device_module_for(device)
    if module is None:
        raise ValueError('DeviceImage inputs (image_preprocess_device "gpu") '
                         f'need a CUDA or NPU device, not {device}')
    import torch

    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, module.current_device())
    if device.type == "cuda":
        from xinfer.platforms.cuda import CudaJpegDecoder

        return CudaJpegDecoder(device)
    from xinfer.platforms.npu import NpuJpegDecoder

    return NpuJpegDecoder(device)
