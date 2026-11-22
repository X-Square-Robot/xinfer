"""Rotary position embeddings: 1-D/M-RoPE tables for decoders, 2-D for ViTs."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from xinfer.ops import active_ops


def inverse_frequencies(dim: int, base: float) -> torch.Tensor:
    """``1 / base^(2i/dim)`` in FP32, computed on the CPU as Hugging Face does.

    Args:
        dim: Rotary dimension (the head dimension).
        base: RoPE base ``theta``.

    Returns:
        ``[dim / 2]`` FP32 on the CPU.
    """
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def mrope_cos_sin(inv_freq: torch.Tensor, position_ids: torch.Tensor, dtype: torch.dtype):
    """Cos/sin tables ``[3, B, L, 2·len(inv_freq)]`` for 3-axis position ids ``[3, B, L]``.

    Angles are FP32 products; cos/sin are computed in FP32 and rounded to
    ``dtype``, as the Qwen2.5-VL rotary module does (``torch.float32`` keeps
    them unrounded, for ``MRope.tables``).

    Args:
        inv_freq: ``[D/2]`` inverse frequencies.
        position_ids: ``[3, B, L]`` temporal/height/width positions (integer or float).
        dtype: Output dtype of the tables.

    Returns:
        ``(cos, sin)``, each ``[3, B, L, D]`` in ``dtype`` (``D = 2·len(inv_freq)``), on the device of
        ``position_ids``.
    """
    freqs = position_ids.to(torch.float32)[..., None] * inv_freq.to(torch.float32)
    angles = torch.cat((freqs, freqs), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


class RopeTables:
    """The M-RoPE tables of one forward, shared by every layer of a stack.

    Built from FP32 cos/sin ``[3, B, L, D]``. ``rotate`` is the eager rotation:
    cos/sin rounded to the activation dtype, ``x`` rotated in FP32 and rounded
    back. ``packed`` is the table fused kernels read (``mxfp8_gemm_rope``):
    the unrounded FP32 values of rows ``B·L``. Each is computed on first use.

    Args:
        cos: ``[3, B, L, D]`` FP32.
        sin: ``[3, B, L, D]`` FP32.
        section: M-RoPE band sizes over ``D/2`` (temporal, height, width).
        dtype: Activation dtype; the eager tables are rounded to it.
        rotary: The ``rotary_embedding`` operator.

    Raises:
        ValueError: ``cos`` or ``sin`` is not FP32.
    """

    def __init__(self, cos: torch.Tensor, sin: torch.Tensor, section: Sequence[int], dtype: torch.dtype, rotary):
        if cos.dtype != torch.float32 or sin.dtype != torch.float32:
            raise ValueError("RopeTables are built from FP32 cos/sin")
        self._cos, self._sin, self._section, self._dtype, self._rotary = cos, sin, list(section), dtype, rotary
        self._eager = None
        self._packed = None

    def _select(self, table: torch.Tensor) -> torch.Tensor:
        """``[3, B, L, D]`` → ``[B, L, D/2]``: each frequency band reads its position axis."""
        half = table.shape[-1] // 2
        parts = table[..., :half].split(self._section, dim=-1)
        return torch.cat([p[i % 3] for i, p in enumerate(parts)], dim=-1)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` ``[B, L, H, D]`` in FP32 and round back to its dtype.

        The eager tables are cos/sin rounded to the activation dtype, band-selected and widened back to FP32
        ``[B, L, 1, D]`` (built once per ``RopeTables``).

        Args:
            x: ``[B, L, H, D]`` query or key heads.

        Returns:
            Rotated ``[B, L, H, D]`` in the dtype of ``x``.
        """
        if self._eager is None:
            tables = []
            for t in (self._cos, self._sin):
                band = self._select(t.to(self._dtype).float())
                tables.append(torch.cat((band, band), dim=-1).unsqueeze(2))  # FP32 [B, L, 1, D]
            self._eager = tuple(tables)
        cos, sin = self._eager
        return self._rotary(x, cos, sin)

    def packed(self) -> torch.Tensor:
        """``layout.mxfp8.rope_table`` of the FP32 values: ``[D/4, B·L, 4]``.

        Entry ``[i, m]`` is ``(cos[m, 2i], sin[m, 2i], cos[m, 2i+1], sin[m, 2i+1])`` of the band-selected,
        unrounded half tables; read by ``mxfp8_gemm_rope``. Built once per ``RopeTables``.

        Returns:
            ``[D/4, B·L, 4]`` FP32.
        """
        if self._packed is None:
            from xinfer.ops.layout.mxfp8 import rope_table

            cos, sin = (self._select(t) for t in (self._cos, self._sin))
            self._packed = rope_table(cos.reshape(-1, cos.shape[-1]), sin.reshape(-1, sin.shape[-1]))
        return self._packed


class MRope:
    """Multimodal RoPE: each frequency band reads one position axis.

    ``section`` splits the half head dimension into the temporal, height and
    width bands (Qwen2.5-VL ``mrope_section``).

    Args:
        section: Band sizes, summing to ``D/2`` (WALL-OSS-0.5: 16, 24, 24).

    Attributes:
        section: The band sizes as a list.
    """

    def __init__(self, section: Sequence[int]):
        self.section = list(section)
        self._rotary = active_ops().rotary_embedding

    def tables(self, cos: torch.Tensor, sin: torch.Tensor, dtype: torch.dtype) -> RopeTables:
        """Tables for FP32 cos/sin ``[3, B, L, D]``; ``dtype`` is the activation dtype.

        Args:
            cos: ``[3, B, L, D]`` FP32 (``mrope_cos_sin(..., torch.float32)``).
            sin: ``[3, B, L, D]`` FP32.
            dtype: Activation dtype of the rotated tensors.

        Returns:
            ``RopeTables`` bound to this ``MRope``'s ``rotary_embedding`` operator.
        """
        return RopeTables(cos, sin, self.section, dtype, self._rotary)


def vision_rotary_angles(inv_freq: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Per-patch angles ``[N, 2·len(inv_freq)]`` from (row, col) positions ``[N, 2]``.

    Args:
        inv_freq: ``[F]`` inverse frequencies.
        positions: ``[N, 2]`` patch row and column.

    Returns:
        ``[N, 2·F]`` FP32: row angles in the first ``F`` columns, column angles in the next ``F``.
    """
    return (positions.to(torch.float32)[..., None] * inv_freq.to(torch.float32)).flatten(1)
