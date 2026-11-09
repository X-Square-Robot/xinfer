"""Reference operators of the Qwen3.5 layers against direct formulations (CPU)."""

import torch
import torch.nn.functional as F

from xinfer.ops import reference


def _recurrent_gated_delta_rule(q, k, v, g, beta):
    """Token-by-token gated delta rule in FP64 (the definition the chunked form evaluates)."""
    q, k = reference.l2_normalize(q), reference.l2_normalize(k)
    q, k, v, g, beta = (t.double() for t in (q, k, v, g, beta))
    b, n, h, dk = k.shape
    state = torch.zeros(b, h, dk, v.shape[-1], dtype=torch.float64)
    out = torch.empty(b, n, h, v.shape[-1], dtype=torch.float64)
    for t in range(n):
        state = state * g[:, t].exp()[..., None, None]
        k_t, v_t = k[:, t], v[:, t]
        delta = (v_t - torch.einsum("bhkv,bhk->bhv", state, k_t)) * beta[:, t, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        out[:, t] = torch.einsum("bhkv,bhk->bhv", state, q[:, t]) / dk ** 0.5
    return out


def test_chunk_gated_delta_rule_matches_the_recurrence():
    g = torch.Generator().manual_seed(0)
    b, n, h, dk, dv = 2, 150, 3, 8, 5  # several chunks of 64 and a partial one
    q, k = torch.randn(b, n, h, dk, generator=g), torch.randn(b, n, h, dk, generator=g)
    v = torch.randn(b, n, h, dv, generator=g)
    decay = -torch.rand(b, n, h, generator=g) * 0.5
    beta = torch.rand(b, n, h, generator=g)
    out = reference.chunk_gated_delta_rule(q, k, v, decay, beta)
    assert out.dtype == torch.float32 and out.shape == (b, n, h, dv)
    torch.testing.assert_close(out.double(), _recurrent_gated_delta_rule(q, k, v, decay, beta), atol=1e-4, rtol=1e-4)


def test_causal_conv1d_sees_only_the_last_kernel_inputs():
    g = torch.Generator().manual_seed(1)
    x, w = torch.randn(2, 6, 9, generator=g), torch.randn(6, 1, 4, generator=g)
    out = reference.causal_conv1d(x, w)
    padded = F.pad(x, (3, 0))
    expected = torch.stack([(padded[:, :, t:t + 4] * w[:, 0]).sum(-1) for t in range(9)], dim=-1)
    torch.testing.assert_close(out, F.silu(expected))


def test_gated_rms_norm_rounds_the_normalization_before_the_weight():
    g = torch.Generator().manual_seed(2)
    x, gate = torch.randn(5, 16, generator=g).bfloat16(), torch.randn(5, 16, generator=g).bfloat16()
    weight = torch.randn(16, generator=g).bfloat16()
    normed = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16()
    expected = ((weight * normed).float() * F.silu(gate.float())).bfloat16()
    assert torch.equal(reference.gated_rms_norm(x, gate, weight, 1e-6), expected)


def test_masked_joint_attention_is_attention_over_prompt_then_own_keys():
    g = torch.Generator().manual_seed(3)
    b, n, p, h, kv, d = 1, 6, 10, 4, 2, 8
    q = torch.randn(b, n, h, d, generator=g)
    k, v = torch.randn(b, n, kv, d, generator=g), torch.randn(b, n, kv, d, generator=g)
    kp, vp = torch.randn(b, kv, p, d, generator=g), torch.randn(b, kv, p, d, generator=g)
    mask = torch.rand(b, 1, n, p + n, generator=g) > 0.4
    mask[..., 0] = True
    out = reference.masked_joint_attention(q, k, v, kp, vp, mask)
    keys = torch.cat([kp, k.transpose(1, 2)], 2).repeat_interleave(h // kv, 1)
    values = torch.cat([vp, v.transpose(1, 2)], 2).repeat_interleave(h // kv, 1)
    scores = (q.transpose(1, 2) @ keys.transpose(-1, -2)) / d ** 0.5
    expected = (scores.masked_fill(~mask, float("-inf")).softmax(-1) @ values).transpose(1, 2)
    torch.testing.assert_close(out, expected)
