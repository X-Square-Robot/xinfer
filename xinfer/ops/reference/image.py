"""Device-image resize semantics: FP32 interpolation, rounded to uint8 after each step."""

from xinfer.platforms.images import quantize_image, resize_image


def image_quantize(view):
    """FP32 image scratch -> uint8 with round-half-up and saturation; input may be consumed."""
    return quantize_image(view)


def image_resize(view, sizes, resize):
    """Resize CHW uint8 RGB through ``sizes``/``(mode, antialias)`` steps and return HWC uint8.

    A step with the input size is skipped. Other steps interpolate in FP32, add 0.5, floor,
    clamp to [0, 255], and cast to uint8 before the next step.
    """
    return resize_image(view, sizes, resize)
