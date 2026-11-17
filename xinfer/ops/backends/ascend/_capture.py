"""Graph capture state of a device's current stream."""

import torch


def stream_capturing(device):
    """Whether ``device``'s current stream is capturing a graph.

    The runtime query looks at the current device, which need not be the tensor's.
    """
    runtime = torch.npu if device.type == "npu" else torch.cuda
    if device.index is None or device.index == runtime.current_device():
        return runtime.is_current_stream_capturing()
    with runtime.device(device):
        return runtime.is_current_stream_capturing()


def note_use(buf, device):
    """Mark a library-owned buffer used while ``device``'s stream captures a graph (the graph keeps its address)."""
    if not getattr(buf, "_xc_in_graph", False) and stream_capturing(device):
        buf._xc_in_graph = True


def retire(buf, retired):
    """Drop an outgrown buffer, except one a captured graph may still point at (kept in ``retired``)."""
    if buf is not None and getattr(buf, "_xc_in_graph", False):
        retired.append(buf)
