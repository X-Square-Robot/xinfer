"""gated_delta_rule_prefill against the token-by-token recurrence in FP32 torch."""
import pytest
import torch
import torch.nn.functional as F

from xinfer.ops.backends.cuda import attn
from conftest import DEVICE, close, sm

D = 128


def reference(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, heads, eps):
    """S <- S * decay_t + k_t^T (beta_t * (v_t - k_t S * decay_t)), o_t = q_t S, per head; then the gated norm."""
    rows = qkv.shape[0]
    mixed = F.silu(F.conv1d(qkv.float().t()[None], conv_weight.float()[:, None], padding=3, groups=qkv.shape[1])
                   [0, :, :rows].t())
    q, k, v = (t.view(rows, heads, D) for t in mixed.split(heads * D, dim=1))
    q, k = (t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) for t in (q, k))
    z, beta_logit, decay_logit = zba.float().split((heads * D, heads, heads), dim=1)
    beta = torch.sigmoid(beta_logit)
    decay = torch.exp(-A_log.exp() * F.softplus(decay_logit + dt_bias))
    state = torch.zeros(heads, D, D, dtype=torch.float64, device=DEVICE)
    out = torch.empty(rows, heads, D, device=DEVICE)
    for t in range(rows):
        state = state * decay[t, :, None, None].double()
        kt, vt = k[t].double(), v[t].double()
        error = vt - (kt[:, None, :] @ state)[:, 0]
        state = state + kt[:, :, None] * (beta[t, :, None].double() * error)[:, None, :]
        out[t] = ((q[t].double() * D ** -0.5)[:, None, :] @ state)[:, 0].float()
    normed = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps) * norm_weight.float()
    return (normed * F.silu(z.view(rows, heads, D))).view(rows, heads * D)


@sm(110, 120)
@pytest.mark.parametrize("rows", [1, 31, 32, 33, 464])
def test_matches_reference(rows, rnd):
    heads = 16
    qkv, zba = rnd(rows, 3 * heads * D), rnd(rows, heads * (D + 2))
    conv_weight, norm_weight = rnd(3 * heads * D, 4, scale=0.5), rnd(D)
    A_log, dt_bias = (rnd(heads, dtype=torch.float32, scale=0.5) for _ in range(2))
    assert attn.gated_delta_rule_prefill_takes(qkv, zba, num_k_heads=heads)
    out = attn.gated_delta_rule_prefill(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, num_k_heads=heads)
    assert close(out, reference(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, heads, 1e-6), 2e-2)
    assert torch.equal(attn.gated_delta_rule_prefill(qkv, zba, conv_weight, A_log, dt_bias, norm_weight,
                                                   num_k_heads=heads), out)  # the same bits on a second call


def test_takes_only_listed_calls(rnd):
    assert not attn.gated_delta_rule_prefill_takes(rnd(8, 3 * 4 * D), rnd(8, 4 * (D + 2)), num_k_heads=4)
    assert not attn.gated_delta_rule_prefill_takes(rnd(1025, 48 * D), rnd(1025, 16 * (D + 2)), num_k_heads=16)
