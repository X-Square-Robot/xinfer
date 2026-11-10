"""Normalization of uint8 camera views on the stage's device, and Qwen2.5-VL patch packing."""

from __future__ import annotations

import torch
from torch import nn


class ImageNormalize(nn.Module):
    """``[h, w, 3]`` uint8 view → ``[3, h, w]`` float32 through a per-channel lookup table.

    The table holds the normalized value of each channel level as the model's preprocessing computes it, so the
    lookup is exact: the result equals that computation bit for bit on any device, inside or outside a CUDA graph,
    and the view crosses to the device as uint8.

    Args:
        lookup_table: ``[3, 256]`` float32, or ``[256]`` for a table shared by the channels.
    """

    def __init__(self, lookup_table):
        super().__init__()
        table = torch.as_tensor(lookup_table, dtype=torch.float32)
        # Nonpersistent: it moves with the model and is not a checkpoint tensor.
        self.register_buffer("lookup_table", table.expand(3, 256).contiguous(), persistent=False)

    def forward(self, image):
        h, w = image.shape[:2]
        return self.lookup_table.gather(1, image.reshape(h * w, 3).t().long()).reshape(3, h, w)


class ImagePack(nn.Module):
    """Normalize resized views and pack them into Qwen2.5-VL patch rows.

    Rows are in merge-block order (``[h/(P·m), w/(P·m), m, m]``), each row ``[C, T, P, P]`` with the image repeated
    over the ``T`` temporal slots. Indexing and copies only, so the rows equal the CPU computation bit for bit.

    Args:
        lookup_table: as for ``ImageNormalize``.
        patch_size: ``P``.
        temporal_patch_size: ``T``.
        merge_size: ``m``.
    """

    def __init__(self, lookup_table, *, patch_size: int, temporal_patch_size: int, merge_size: int):
        super().__init__()
        self.normalize = ImageNormalize(lookup_table)
        self.patch_size, self.temporal_patch_size, self.merge_size = patch_size, temporal_patch_size, merge_size

    def forward(self, images):
        """``[h_i, w_i, 3]`` uint8 views (``h_i``, ``w_i`` multiples of ``P·m``) → ``[Σ h_i·w_i / P², 3·T·P²]`` rows."""
        p, t, m = self.patch_size, self.temporal_patch_size, self.merge_size
        rows = []
        for image in images:
            h, w = image.shape[:2]
            x = self.normalize(image).expand(t, 3, h, w).reshape(t, 3, h // p // m, m, p, w // p // m, m, p)
            rows.append(x.permute(2, 5, 3, 6, 1, 0, 4, 7).reshape(-1, 3 * t * p * p))
        return torch.cat(rows)
