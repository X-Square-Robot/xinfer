"""The default device-image resize recipe, shared by runtime and reference operators."""

import torch
import torch.nn.functional as F


def quantize_image(view):
    """Round an FP32 image scratch tensor half up, clamp to [0, 255], and cast to uint8.

    The input is temporary storage and may be modified by this operation.
    """
    return view.add_(0.5).floor_().clamp_(0, 255).to(torch.uint8)


def resize_image(view, sizes, resize):
    """CHW uint8 to HWC uint8; interpolate in FP32 and round half up after every resized step."""
    x = view.unsqueeze(0)
    for (mode, antialias), size in zip(resize, sizes):
        if tuple(size) != tuple(x.shape[-2:]):
            x = F.interpolate(x.float(), size=tuple(size), mode=mode, antialias=antialias, align_corners=False)
            x = quantize_image(x)
    return x[0].permute(1, 2, 0).contiguous()
