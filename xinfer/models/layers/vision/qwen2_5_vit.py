"""Qwen2.5-VL vision tower (window attention ViT with 2-D RoPE and patch merger).

The token layout that depends only on the image grids (patch positions,
window order, attention segments) is computed on the host by
``vision_layout`` and passed in, so the device forward needs no host reads.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from xinfer.models.layers.basic import GatedMLP, Linear, RMSNorm
from xinfer.models.layers.rotary import vision_rotary_angles
from xinfer.ops import active_ops


@dataclass(frozen=True)
class VisionTowerConfig:
    """Structure of a Qwen2.5-VL vision tower.

    Attributes:
        depth: Number of blocks.
        hidden_size: ViT width.
        mlp_size: Width of each half of the fused gate/up weight (possibly padded, as stored).
        num_heads: Attention heads.
        in_features: Flattened patch size ``channels · temporal_patch · patch²``.
        spatial_merge_size: Patches merged per side into one output token.
        window_size: Window side in pixels for window attention.
        patch_size: Patch side in pixels.
        out_hidden_size: Width of the merged output tokens.
        fullatt_block_indexes: Blocks with full attention over each image; the others use windows.
        eps: RMSNorm epsilon.
        rope_theta: Base of the 2-D rotary embedding.
    """

    depth: int
    hidden_size: int
    mlp_size: int  # possibly padded intermediate width of the fused gate/up weight
    num_heads: int
    in_features: int  # channels * temporal_patch * patch * patch
    spatial_merge_size: int
    window_size: int
    patch_size: int
    out_hidden_size: int
    fullatt_block_indexes: tuple[int, ...]
    eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def vision_layout(grids, config: VisionTowerConfig) -> dict:
    """Host-side layout for packed images with grids ``[(t, h, w), ...]``.

    Returns ``positions`` (int64 ``[N, 2]`` patch row/col in merge-block
    order), ``window_index`` (int64 merge-unit permutation into window
    order), ``reverse_index`` (its inverse), and the segment bounds
    ``cu_window`` / ``cu_full`` (lists of ints).

    Windows are ``window_size / merge / patch`` merge units per side (the last row and column of windows may
    be partial); ``cu_window`` holds cumulative patch counts of the windows in window order, ``cu_full``
    those of each image frame.

    The layout is computed once per grid list (a robot's cameras keep their sizes) and copied on each call.

    Args:
        grids: Patch grid ``(t, h, w)`` of each image, in packing order; ``h`` and ``w`` must be multiples of
            ``spatial_merge_size``.
        config: The tower configuration.

    Returns:
        ``{"positions": [N, 2] int64, "window_index": [N / merge²] int64, "reverse_index": [N / merge²]
        int64, "cu_window": list[int], "cu_full": list[int]}``, CPU tensors, ``N = Σ t·h·w``.

    Raises:
        ValueError: A grid side is not divisible by the merge size.
    """
    layout = _layout(tuple((int(t), int(h), int(w)) for t, h, w in grids), config.spatial_merge_size,
                     config.window_size // config.spatial_merge_size // config.patch_size)
    return {name: value.clone() if torch.is_tensor(value) else list(value) for name, value in layout.items()}


@functools.lru_cache(maxsize=64)
def _layout(grids, m, window) -> dict:
    unit = m * m
    positions, window_index, cu_window, cu_full = [], [], [0], [0]
    offset = 0
    for t, h, w in grids:
        if h % m or w % m:
            raise ValueError(f"grid {h}x{w} is not divisible by the merge size {m}")
        row_index = np.broadcast_to(np.arange(h)[:, None], (h, w)).reshape(h // m, m, w // m, m)
        col_index = np.broadcast_to(np.arange(w)[None, :], (h, w)).reshape(h // m, m, w // m, m)
        patch_positions = np.stack([row_index.transpose(0, 2, 1, 3).reshape(-1),
                                    col_index.transpose(0, 2, 1, 3).reshape(-1)], -1)
        positions.append(np.tile(patch_positions, (t, 1)))
        token_rows, token_cols = h // m, w // m
        pad_h, pad_w = (window - token_rows % window) % window, (window - token_cols % window) % window
        index = np.full((t, token_rows + pad_h, token_cols + pad_w), -1, dtype=np.int64)
        index[:, :token_rows, :token_cols] = np.arange(t * token_rows * token_cols).reshape(t, token_rows, token_cols)
        window_rows, window_cols = (token_rows + pad_h) // window, (token_cols + pad_w) // window
        index = index.reshape(t, window_rows, window, window_cols, window).transpose(0, 1, 3, 2, 4)
        index = index.reshape(t, window_rows * window_cols, window * window)
        for row in index.reshape(-1, window * window):
            valid = row[row != -1]
            window_index.append(valid + offset)
            cu_window.append(cu_window[-1] + valid.size * unit)
        offset += t * token_rows * token_cols
        for _ in range(t):
            cu_full.append(cu_full[-1] + h * w)
    window_index = np.concatenate(window_index)
    return dict(
        positions=torch.from_numpy(np.concatenate(positions).astype(np.int64)),
        window_index=torch.from_numpy(window_index),
        reverse_index=torch.from_numpy(np.argsort(window_index, kind="stable")),
        cu_window=cu_window,
        cu_full=cu_full,
    )


class VisionBlock(nn.Module):
    """Pre-norm ViT block: 2-D-rotary attention within segments, then a gated MLP.

    Norms, rotary and residual adds are unfused; the MLP requests the gated activation in the down
    projection's input (``act_fused_into="down"``).

    Args:
        config: The tower configuration.
        group: Quantization group of the block's linear layers.
    """

    def __init__(self, config: VisionTowerConfig, group: str = "vision.blocks"):
        super().__init__()
        self.num_heads, self.head_dim = config.num_heads, config.head_dim
        self.norm1 = RMSNorm(config.hidden_size, config.eps)
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size, bias=True, group=group)
        self.proj = Linear(config.hidden_size, config.hidden_size, bias=True, group=group)
        self.norm2 = RMSNorm(config.hidden_size, config.eps)
        # Quantized, the gated activation is computed while quantizing the down input.
        self.mlp = GatedMLP(config.hidden_size, config.mlp_size, bias=True, group=group, act_fused_into="down")
        ops = active_ops()
        self._rotary, self._attention = ops.rotary_embedding, ops.varlen_attention

    def forward(self, x, cos, sin, cu_seqlens):
        """Run one block over packed patches.

        Args:
            x: ``[N, hidden]`` patches in window order.
            cos: ``[N, 1, head_dim]`` FP32 rotary table (broadcast over heads).
            sin: ``[N, 1, head_dim]`` FP32.
            cu_seqlens: Cumulative segment bounds (host list); attention stays within each segment.

        Returns:
            ``[N, hidden]`` in the dtype of ``x``.
        """
        n = x.shape[0]
        q, k, v = self.qkv(self.norm1(x)).reshape(n, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        q = self._rotary(q, cos, sin)  # FP32 rotation, rounded back to x.dtype
        k = self._rotary(k, cos, sin)
        a = self._attention(q, k, v, cu_seqlens)
        x = x + self.proj(a.reshape(n, -1))
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    """RMSNorm, then ``fc2(gelu(fc1(·)))`` over each group of ``merge²`` consecutive patches.

    Args:
        config: The tower configuration.
        group: Quantization group of ``fc1`` and ``fc2``.
    """

    def __init__(self, config: VisionTowerConfig, group: str = "vision.merger"):
        super().__init__()
        self.width = config.hidden_size * config.spatial_merge_size ** 2
        self.norm = RMSNorm(config.hidden_size, config.eps)
        self.fc1 = Linear(self.width, self.width, bias=True, group=group)
        self.fc2 = Linear(self.width, config.out_hidden_size, bias=True, group=group)
        self._gelu = active_ops().gelu

    def forward(self, x):
        """``[N, hidden]`` → ``[N / merge², out_hidden_size]``."""
        return self.fc2(self._gelu(self.fc1(self.norm(x).reshape(-1, self.width))))


class VisionTower(nn.Module):
    """Qwen2.5-VL ViT: patch embedding, ``depth`` window/full-attention blocks, patch merger.

    Args:
        config: The tower configuration.

    Attributes:
        config: The tower configuration.
        patch_embed: Bias-free ``Linear`` ``in_features → hidden_size`` (group ``"vision.patch_embed"``).
        blocks: The ``VisionBlock`` stack.
        merger: The ``PatchMerger``.
        inv_freq: 2-D rotary inverse frequencies ``[head_dim / 4]`` FP32.
    """

    def __init__(self, config: VisionTowerConfig):
        super().__init__()
        self.config = config
        self.patch_embed = Linear(config.in_features, config.hidden_size, bias=False, group="vision.patch_embed")
        self.blocks = nn.ModuleList(VisionBlock(config) for _ in range(config.depth))
        self.merger = PatchMerger(config)
        half = config.head_dim // 2
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, half, 2, dtype=torch.float) / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values, layout, dtype):
        """``pixel_values`` ``[N, in_features]`` → merged tokens ``[N / merge², out_hidden]``.

        Patches are permuted into window order for the blocks and the merged tokens back to the input
        order. The rotary angles and cos/sin are FP32 and not rounded.

        Args:
            pixel_values: ``[N, in_features]`` packed patch rows (cast to ``dtype``).
            layout: ``vision_layout`` of the images: ``positions``, ``window_index`` and ``reverse_index``
                (on the model's device) and the host lists ``cu_window`` and ``cu_full``.
            dtype: Activation dtype (BF16 for WALL-OSS-0.5).

        Returns:
            ``[N / merge², out_hidden_size]`` in ``dtype``, in the merge-unit order of ``pixel_values``.
        """
        unit = self.config.spatial_merge_size ** 2
        x = self.patch_embed(pixel_values.to(dtype))
        n = x.shape[0]
        window_index = layout["window_index"]
        angles = vision_rotary_angles(self.inv_freq, layout["positions"])
        x = x.reshape(n // unit, unit, -1)[window_index].reshape(n, -1)
        angles = angles.reshape(n // unit, unit, -1)[window_index].reshape(n, -1)
        full_angles = torch.cat((angles, angles), dim=-1)
        cos, sin = full_angles.cos()[:, None, :], full_angles.sin()[:, None, :]  # FP32, broadcast over heads
        for i, block in enumerate(self.blocks):
            cu_seqlens = layout["cu_full"] if i in self.config.fullatt_block_indexes else layout["cu_window"]
            x = block(x, cos, sin, cu_seqlens)
        return self.merger(x)[layout["reverse_index"]]
