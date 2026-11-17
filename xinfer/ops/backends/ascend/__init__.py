"""Operator backends for Ascend platforms: the kernels in ``src/``, loaded by ``_native.py``."""


# The decoded-frame quantizer the NPU platform's JPEG path calls: dependency inversion, the
# other way round of the replayed mount. The platforms package must not import the operator
# backends (test_layering's direction rule), so this backend registers its SOUP2 quant knife
# (one kernel for the add+floor+clamp+cast of the YUV-to-RGB conversion) at the point the
# registry loads it. The hook keeps the knife module lazy; without its library, without its
# family flag, or on any error the platform's released torch chain serves the same semantics.
def _device_frame_quantizer(x):
    from xinfer.ops.backends.ascend import soup2

    if soup2.enabled("quant"):
        return soup2.quant_u8(x)
    return None


try:
    from xinfer.platforms import npu as _npu_platforms

    _npu_platforms.register_frame_quantizer(_device_frame_quantizer)
except ImportError:  # a build without the NPU platform: the frame chain stays the torch one
    pass
