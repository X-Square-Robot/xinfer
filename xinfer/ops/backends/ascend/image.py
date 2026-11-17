"""Device-image resize using the existing Ascend resample and uint8 conversion entries."""

import torch
import torch.nn.functional as F

from xinfer.ops.backends.ascend import aa_resample, soup2
from xinfer.ops.backends.ascend._native import NATIVE
from xinfer.ops.registry import register
from xinfer.platforms.images import quantize_image


@register("image_quantize", backend="ascend", **NATIVE)
def image_quantize(view):
    out = soup2.quant_u8(view) if soup2.enabled("quant") else None
    return out if out is not None else quantize_image(view)


@register("image_resize", backend="ascend", **NATIVE)
def image_resize(view, sizes, resize):
    """Keep the existing resize/rounding order; use a native entry only where it accepts the call."""
    x = view.unsqueeze(0)
    for (mode, antialias), size in zip(resize, sizes):
        if tuple(size) != tuple(x.shape[-2:]):
            if mode == "bicubic" and antialias and x.dtype == torch.uint8:
                routed = aa_resample.step(x, tuple(size))
                if routed is not None:
                    x = routed
                    continue
            x = F.interpolate(x.float(), size=tuple(size), mode=mode, antialias=antialias, align_corners=False)
            x = image_quantize(x)
    return x[0].permute(1, 2, 0).contiguous()
