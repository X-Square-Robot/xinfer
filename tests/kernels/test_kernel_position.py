"""rope, rot_pos and pos_embed_interp against torch."""
import math

import pytest
import torch

from xinfer.ops.backends.cuda import position
from conftest import DEVICE, close

GRIDS = [(1, 14, 22), (1, 14, 20), (2, 4, 6)]  # (frames, height, width) in patches


def merge_order(h, w, merge):
    """Patch rows and columns of an h x w grid, in merge x merge blocks."""
    shape = (h // merge, w // merge, merge, merge)
    rows = torch.arange(h, device=DEVICE).view(h // merge, 1, merge, 1).expand(shape).reshape(-1)
    columns = torch.arange(w, device=DEVICE).view(1, w // merge, 1, merge).expand(shape).reshape(-1)
    return rows, columns


def test_rope(rnd):
    b, m, heads, kv_heads, d, rotary = 1, 464, 8, 2, 256, 64
    q, k = rnd(b, m, heads, d), rnd(b, m, kv_heads, d)
    angle = rnd(b, m, rotary // 2, dtype=torch.float32, scale=3.0)
    cos, sin = angle.cos(), angle.sin()
    half = rotary // 2

    def reference(x):
        a, bb = x[..., :half].float(), x[..., half:rotary].float()
        c, s = cos[:, :, None], sin[:, :, None]
        return torch.cat([(a * c - bb * s).to(x.dtype), (bb * c + a * s).to(x.dtype), x[..., rotary:]], dim=-1)

    q_out, k_out = position.rope_full(q, k, cos, sin, rotary)
    assert close(q_out, reference(q), 1e-4) and close(k_out, reference(k), 1e-4)
    assert torch.equal(q_out[..., rotary:], q[..., rotary:])  # the dims past rotary_dim are copied


def test_rot_pos_is_bitwise(rnd):
    inv_freq = 1.0 / (10000 ** (torch.arange(16, dtype=torch.float32, device=DEVICE) / 16))
    expected = []
    for frames, h, w in GRIDS:
        rows, columns = merge_order(h, w, 2)
        expected.append(torch.cat((rows[:, None].float() * inv_freq, columns[:, None].float() * inv_freq), -1)
                        .repeat(frames, 1))
    assert torch.equal(position.rot_pos(inv_freq, GRIDS, 2), torch.cat(expected))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pos_embed_interp(dtype, rnd):
    side, k = 48, 1024
    weight = rnd(side * side, k, dtype=dtype)
    expected = []
    for frames, h, w in GRIDS:
        points = [torch.linspace(0, side - 1, n, device=DEVICE) for n in (h, w)]
        floor = [p.int() for p in points]
        ceil = [(f + 1).clip(max=side - 1) for f in floor]
        dh, dw = (p - f for p, f in zip(points, floor))
        rows, columns = merge_order(h, w, 2)
        dh, dw = dh[rows, None], dw[columns, None]

        def corner(r, c):
            return weight[(r[rows] * side + c[columns]).long()].float()

        value = ((1 - dh) * (1 - dw) * corner(floor[0], floor[1]) + (1 - dh) * dw * corner(floor[0], ceil[1])
                 + dh * (1 - dw) * corner(ceil[0], floor[1]) + dh * dw * corner(ceil[0], ceil[1]))
        expected.append(value.repeat(frames, 1))
    out = position.pos_embed_interp(weight, GRIDS, 2)
    assert out.dtype == dtype and close(out, torch.cat(expected), 5e-3 if dtype == torch.bfloat16 else 1e-5)
    assert math.isqrt(weight.shape[0]) == side
