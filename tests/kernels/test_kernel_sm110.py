"""The SM110 (Jetson Thor) paths: Q and K in rotary-pair order, the fused MLP, the residual added in the GEMM and
the L2 prefetches, against FP32 torch."""
import pytest
import torch
import torch.nn.functional as F

from xinfer.ops.backends.cuda import gemm
from xinfer.ops.layout import mxfp8 as layout
from conftest import DEVICE, close, sm

pytestmark = sm(110)


def angles(m, rotary_dim):
    position = torch.arange(m, dtype=torch.float32)[:, None]
    frequency = 1.0 / (10000 ** (torch.arange(rotary_dim // 2, dtype=torch.float32) / (rotary_dim // 2)))
    return torch.cos(position * frequency).to(DEVICE), torch.sin(position * frequency).to(DEVICE)


@pytest.mark.parametrize("m", [16, 300, 2112, 4096])
def test_qkv_rope_in_rotary_pair_order(m, rnd):
    heads, d, k = 16, 80, 1280
    n = 3 * heads * d
    x, weight, bias = rnd(m, k), rnd(n, k, scale=0.05), rnd(n)
    cos, sin = angles(m, d)
    table = layout.rope_table(cos, sin)
    args = dict(rope_table=table, num_heads=heads, head_dim=d)
    assert gemm.kernel_name("qkv_proj_rope", x, weight, bias, qk_order="any", **args) == "tcgen05_qkv_rope_bf16_sm110"
    assert gemm.kernel_name("qkv_proj_rope", x, weight, bias, **args) == "generic"
    standard = gemm.qkv_proj_rope(x, weight, bias, **args)
    pairs = gemm.qkv_proj_rope(x, weight, bias, qk_order="any", **args)
    # channel j and j + d / 2 of every Q and K head are adjacent; V is in the standard order
    qk = standard[:, : 2 * heads * d].view(m, 2 * heads, 2, d // 2).transpose(-1, -2).reshape(m, -1)
    assert close(pairs, torch.cat([qk, standard[:, 2 * heads * d :]], 1))
    assert torch.equal(gemm.qkv_proj_rope(x, weight, bias, qk_order="any", **args), pairs)


def test_qkv_rope_pairs_only_on_the_tuned_rows(rnd):
    heads, d, k, m = 16, 80, 1280, 15
    x, weight, bias = rnd(m, k), rnd(3 * heads * d, k, scale=0.05), rnd(3 * heads * d)
    cos, sin = angles(m, d)
    args = dict(rope_table=layout.rope_table(cos, sin), num_heads=heads, head_dim=d)
    assert gemm.kernel_name("qkv_proj_rope", x, weight, bias, qk_order="any", **args) == "generic"
    assert torch.equal(gemm.qkv_proj_rope(x, weight, bias, qk_order="any", **args),
                       gemm.qkv_proj_rope(x, weight, bias, **args))  # the standard order there


def mlp_reference(x, gate_up, down, residual):
    y = x.float() @ gate_up.float().t()
    n = gate_up.shape[0] // 2
    return residual.float() + (F.silu(y[:, :n]) * y[:, n:]).to(torch.bfloat16).float() @ down.float().t()


@pytest.mark.parametrize("m", [520, 768, 400])
def test_mlp_add_rms_norm(m, rnd):
    """Fused for 520 <= M <= 768 (the intermediate padded to 11264), else the gate/up and the down projection
    kernels; the sum is added into the residual in place."""
    hidden, inter = 2048, 11008
    x, gate_up, down = rnd(m, hidden), rnd(2 * inter, hidden, scale=0.03), rnd(hidden, inter, scale=0.02)
    residual, weight = rnd(1, m, hidden), rnd(hidden, dtype=torch.float32)
    expected = mlp_reference(x, gate_up, down, residual.view(m, hidden)).view(1, m, hidden)
    fused = 520 <= m <= 768
    total, normed = gemm.mlp_add_rms_norm(x, gate_up, down, residual, weight, 1e-6)
    assert (total is residual) == fused
    assert close(total, expected, 1e-2)
    rms = expected * torch.rsqrt(expected.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
    assert close(normed, rms, 1e-2)


def test_gemm_add_rms_norm_adds_the_residual_in_the_gemm(rnd):
    m, k, n = 600, 11008, 2048
    x, projection, residual = rnd(m, k), rnd(n, k, scale=0.02), rnd(1, m, n)
    weight = rnd(n, dtype=torch.float32)
    assert gemm.kernel_name("gemm", x, projection, residual=residual.view(m, n)) == "mcast_residual_bf16_sm110"
    total, normed = gemm.gemm_add_rms_norm(x, projection, residual, weight, 1e-6)
    expected = residual.float() + (x.float() @ projection.float().t())[None]
    assert close(total, expected, 1e-2)
    assert torch.equal(normed, gemm.rmsnorm(total, weight, 1e-6))


def test_prefetch_weights_leaves_results_unchanged(rnd):
    x, weights = rnd(16, 1024), [rnd(4096, 1024), rnd(1024, 2048)]
    expected = gemm.gemm(x, weights[0])
    gemm.prefetch_weights(weights)
    gemm.join_prefetch()
    assert torch.equal(gemm.gemm(x, weights[0]), expected)
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    out = torch.empty_like(expected)
    with torch.cuda.stream(stream):
        gemm.prefetch_weights(weights)
        gemm.gemm(x, weights[0], out=out)
        gemm.join_prefetch()
    with torch.cuda.graph(graph, stream=stream):
        gemm.prefetch_weights(weights)
        gemm.gemm(x, weights[0], out=out)
        gemm.join_prefetch()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
