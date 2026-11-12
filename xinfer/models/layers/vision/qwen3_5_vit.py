"""Qwen3.5 vision tower: patch embedding, learned position table, full-attention ViT, patch merger.

Numerics follow the Wall-X model under BF16 autocast: LayerNorms run in FP32 and their outputs are rounded to
BF16 by the next projection, every projection runs in BF16, and the 2-D rotary embedding rotates in FP32.
The patch embedding is a GEMM over the flattened patches (the source model's stride-equals-kernel Conv3d as a
linear layer, as the production service runs it): a cuDNN Conv3d does not survive repeated CUDA-graph replays
on RTX 5090 (cuDNN 9.13). The interpolation of the learned position table and the rotary positions depend only
on the image grids; ``vision_layout`` computes them on the host.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from xinfer.models.layers.basic import Linear, _meta_param
from xinfer.models.layers.rotary import vision_rotary_angles
from xinfer.ops import active_ops


@dataclass(frozen=True)
class Qwen35VisionConfig:
    """Structure of a Qwen3.5 vision tower.

    Attributes:
        depth: Number of blocks.
        hidden_size: ViT width.
        intermediate_size: MLP width.
        num_heads: Attention heads.
        in_channels: Image channels.
        patch_size: Patch side in pixels.
        temporal_patch_size: Frames per patch (a single image is repeated).
        spatial_merge_size: Patches merged per side into one output token.
        out_hidden_size: Width of the merged output tokens.
        num_position_embeddings: Rows of the learned position table (a square grid).
        eps: LayerNorm epsilon.
        rope_theta: Base of the 2-D rotary embedding.
    """

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    num_position_embeddings: int
    eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def in_features(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size ** 2

    @property
    def grid_side(self) -> int:
        return int(self.num_position_embeddings ** 0.5)


@functools.lru_cache(maxsize=64)
def _vision_layout_compute(grids: tuple, spatial_merge_size: int, grid_side: int) -> dict:
    """Host-side layout of packed images with grids ``[(t, h, w), ...]``.

    The patch order is the merge-block order of the Qwen2-VL image processor. The learned position table
    (a ``grid_side²`` square) is bilinearly interpolated to each ``h × w`` grid with ``linspace`` sample points,
    as Wall-X's ``fast_pos_embed_interpolate`` does: four corner indices and FP32 weights per patch.

    Args:
        grids: Patch grid ``(t, h, w)`` of each image, in packing order; ``h`` and ``w`` must be multiples of
            ``spatial_merge_size``.
        config: The tower configuration.

    Returns:
        ``{"positions": [N, 2] int64 (patch row, col), "pos_index": [4, N] int64, "pos_weight": [4, N] FP32,
        "cu_seqlens": list[int]}`` (CPU tensors, ``N = Σ t·h·w``; one attention segment per frame).

    Raises:
        ValueError: A grid side is not divisible by the merge size.
    """
    m, side = spatial_merge_size, grid_side
    positions, index, weight, cu_seqlens = [], [[] for _ in range(4)], [[] for _ in range(4)], [0]
    for t, h, w in grids:
        t, h, w = int(t), int(h), int(w)
        if h % m or w % m:
            raise ValueError(f"grid {h}x{w} is not divisible by the merge size {m}")
        rows = np.broadcast_to(np.arange(h)[:, None], (h, w)).reshape(h // m, m, w // m, m)
        cols = np.broadcast_to(np.arange(w)[None, :], (h, w)).reshape(h // m, m, w // m, m)
        patch = np.stack([rows.transpose(0, 2, 1, 3).reshape(-1), cols.transpose(0, 2, 1, 3).reshape(-1)], -1)
        positions.append(np.tile(patch, (t, 1)))
        h_points = torch.linspace(0, side - 1, h)
        w_points = torch.linspace(0, side - 1, w)
        h_floor, w_floor = h_points.int(), w_points.int()
        h_ceil, w_ceil = (h_floor + 1).clip(max=side - 1), (w_floor + 1).clip(max=side - 1)
        dh, dw = h_points - h_floor, w_points - w_floor
        corners = [(h_floor, w_floor, (1 - dh)[:, None] * (1 - dw)[None]),
                   (h_floor, w_ceil, (1 - dh)[:, None] * dw[None]),
                   (h_ceil, w_floor, dh[:, None] * (1 - dw)[None]),
                   (h_ceil, w_ceil, dh[:, None] * dw[None])]
        for i, (hi, wi, wt) in enumerate(corners):
            idx = (hi[:, None] * side + wi[None]).reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).reshape(-1)
            wts = wt.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).reshape(-1)
            index[i].append(idx.long().repeat(t))
            weight[i].append(wts.float().repeat(t))
        for _ in range(t):
            cu_seqlens.append(cu_seqlens[-1] + h * w)
    return dict(
        positions=torch.from_numpy(np.concatenate(positions).astype(np.int64)),
        pos_index=torch.stack([torch.cat(i) for i in index]),
        pos_weight=torch.stack([torch.cat(w) for w in weight]),
        cu_seqlens=cu_seqlens,
    )


def vision_layout(grids, config: Qwen35VisionConfig) -> dict:
    """Host-side layout of packed images with grids ``[(t, h, w), ...]`` (cached compute).

    The layout is a pure function of ``(grids, config)`` and costs ~1.1 ms of host compute per request on
    the wallx2b prepare path (the largest single host item after the wire decodes, probe-prepare-2b); a
    request set that repeats grids hits the cache. Each caller still gets its own tensor objects (clone)
    and its own ``cu_seqlens`` list - the exact fresh-object semantics of the uncached path, so no
    downstream code can observe the sharing.
    """
    cached = _vision_layout_compute(tuple(tuple(grid) for grid in grids), int(config.spatial_merge_size),
                                     int(config.grid_side))
    return {"positions": cached["positions"].clone(), "pos_index": cached["pos_index"].clone(),
            "pos_weight": cached["pos_weight"].clone(), "cu_seqlens": list(cached["cu_seqlens"])}


class LayerNorm(nn.Module):
    """LayerNorm with bias through the ``layer_norm`` operator (FP32 computation and output).

    Args:
        dim: Normalized (last) dimension.
        eps: Variance epsilon.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = _meta_param(dim)
        self.bias = _meta_param(dim)
        self._layer_norm = active_ops().layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., dim]`` → FP32 ``[..., dim]``."""
        return self._layer_norm(x, self.weight, self.bias, self.eps)


class VisionBlock(nn.Module):
    """``x + proj(attn(rope(qkv(LN(x)))))``, then ``x + fc2(gelu_tanh(fc1(LN(x))))``; attention within frames.

    Args:
        config: The tower configuration.
        group: Quantization group of the block's linear layers.
    """

    def __init__(self, config: Qwen35VisionConfig, group: str = "vision.blocks"):
        super().__init__()
        self.num_heads, self.head_dim = config.num_heads, config.head_dim
        self.norm1 = LayerNorm(config.hidden_size, config.eps)
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size, bias=True, group=group)
        self.proj = Linear(config.hidden_size, config.hidden_size, bias=True, group=group)
        self.norm2 = LayerNorm(config.hidden_size, config.eps)
        self.fc1 = Linear(config.hidden_size, config.intermediate_size, bias=True, group=group)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size, bias=True, group=group)
        ops = active_ops()
        self._rotary, self._attention, self._gelu_tanh = ops.rotary_embedding, ops.varlen_attention, ops.gelu_tanh

    def forward(self, x, cos, sin, cu_seqlens):
        """Run one block over packed patches.

        Args:
            x: ``[N, hidden]`` BF16 patches.
            cos: ``[N, 1, head_dim]`` FP32 rotary table (broadcast over heads).
            sin: ``[N, 1, head_dim]`` FP32.
            cu_seqlens: Cumulative frame bounds (host list); attention stays within each frame.

        Returns:
            ``[N, hidden]`` BF16.
        """
        n = x.shape[0]
        qkv = self.qkv(self.norm1(x).to(x.dtype)).reshape(n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        q = self._rotary(q, cos, sin)
        k = self._rotary(k, cos, sin)
        x = x + self.proj(self._attention(q, k, v, cu_seqlens).reshape(n, -1))
        return x + self.fc2(self._gelu_tanh(self.fc1(self.norm2(x).to(x.dtype))))


class PatchMerger(nn.Module):
    """LayerNorm per patch, then ``fc2(gelu(fc1(·)))`` over each group of ``merge²`` consecutive patches.

    Args:
        config: The tower configuration.
        group: Quantization group of ``fc1`` and ``fc2``.
    """

    def __init__(self, config: Qwen35VisionConfig, group: str = "vision.merger"):
        super().__init__()
        self.width = config.hidden_size * config.spatial_merge_size ** 2
        self.norm = LayerNorm(config.hidden_size, config.eps)
        self.fc1 = Linear(self.width, self.width, bias=True, group=group)
        self.fc2 = Linear(self.width, config.out_hidden_size, bias=True, group=group)
        self._gelu = active_ops().gelu

    def forward(self, x):
        """``[N, hidden]`` BF16 → ``[N / merge², out_hidden_size]`` BF16."""
        return self.fc2(self._gelu(self.fc1(self.norm(x).to(x.dtype).reshape(-1, self.width))))


class Qwen35VisionTower(nn.Module):
    """Qwen3.5 ViT: patch embedding plus interpolated position table, ``depth`` blocks, patch merger.

    Args:
        config: The tower configuration.

    Attributes:
        config: The tower configuration.
        patch_embed: ``Linear`` ``C·T·P² → hidden`` with bias (group ``"vision.patch_embed"``).
        pos_embed: Learned position table ``[num_position_embeddings, hidden]``.
        blocks: The ``VisionBlock`` stack.
        merger: The ``PatchMerger``.
        inv_freq: 2-D rotary inverse frequencies ``[head_dim / 4]`` FP32.
    """

    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        self.config = config
        c = config
        self.patch_embed = Linear(c.in_features, c.hidden_size, bias=True, group="vision.patch_embed")
        self.pos_embed = _meta_param(c.num_position_embeddings, c.hidden_size)
        self.blocks = nn.ModuleList(VisionBlock(config) for _ in range(config.depth))
        self.merger = PatchMerger(config)
        half = config.head_dim // 2
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, half, 2, dtype=torch.float) / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values, layout, dtype):
        """``pixel_values`` ``[N, in_features]`` → merged tokens ``[N / merge², out_hidden_size]``.

        Args:
            pixel_values: ``[N, in_features]`` packed patch rows (cast to ``dtype``).
            layout: ``vision_layout`` of the images, with ``positions``, ``pos_index`` and ``pos_weight`` on
                the model's device and the host list ``cu_seqlens``.
            dtype: Activation dtype (BF16).

        Returns:
            ``[N / merge², out_hidden_size]`` in ``dtype``, in the merge-unit order of ``pixel_values``.
        """
        x = self.patch_embed(pixel_values.to(dtype))
        corners = self.pos_embed[layout["pos_index"]] * layout["pos_weight"][:, :, None].to(self.pos_embed.dtype)
        x = x + (corners[0] + corners[1] + corners[2] + corners[3]).to(dtype)
        angles = vision_rotary_angles(self.inv_freq, layout["positions"])
        full_angles = torch.cat((angles, angles), dim=-1)
        cos, sin = full_angles.cos()[:, None, :], full_angles.sin()[:, None, :]
        for block in self.blocks:
            x = block(x, cos, sin, layout["cu_seqlens"])
        return self.merger(x)
