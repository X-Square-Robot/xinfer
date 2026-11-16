"""Compact Wall-OSS RoPE preserves the reference tables and their rounding."""

import pytest
import torch

from xinfer.models.layers.rotary import inverse_frequencies, mrope_cos_sin
from xinfer.models.wall_oss.optimized import compact_rope


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=[pytest.mark.gpu,
                               pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]),
])
@pytest.mark.parametrize("section", [(2, 3, 3), (16, 24, 24)])
@pytest.mark.parametrize("length", [0, 1, 16, 384, 1536])
@pytest.mark.parametrize("shared_axes", [False, True])
def test_compact_rope_matches_band_selected_reference(device, section, length, shared_axes):
    inv_freq = inverse_frequencies(2 * sum(section), 1000000).to(device)
    generator = torch.Generator().manual_seed(11)
    positions = torch.randint(-32000, 32000, (3, 1, length), generator=generator).to(device)
    if shared_axes:  # Action tokens share an axis; prompt image tokens do not.
        positions = positions[:1].expand(3, 1, length)
    actual = compact_rope(inv_freq, positions, section)
    expanded = mrope_cos_sin(inv_freq, positions, torch.float32)
    for output, table in zip(actual, expanded):
        half = table.to(torch.bfloat16).float()[:, 0, :, :sum(section)]
        bands = half.split(section, dim=-1)
        expected = torch.cat([band[i % 3] for i, band in enumerate(bands)], dim=-1)
        assert output.shape == (length, sum(section))
        assert output.dtype == torch.float32 and output.is_contiguous()
        assert torch.equal(output.view(torch.int32), expected.view(torch.int32))
