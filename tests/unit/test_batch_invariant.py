"""Batch-invariant Triton linear, RMSNorm and adaRMS (GPU): tolerance vs the references, row invariance, capture."""

import pytest
import torch

from xinfer.ops import reference
from xinfer.ops.backends import batch_invariant as bi

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not bi.available(), reason="Triton + CUDA required")]

# Wall-OSS action-stage shapes (expert 1 BF16, action head FP32) and the ViT patch embed.
SHAPES = [((1024, 2560), torch.bfloat16, True), ((2048, 1024), torch.bfloat16, False),
          ((1024, 4096), torch.bfloat16, False), ((1024, 1024), torch.bfloat16, False),
          ((52, 1024), torch.float32, False), ((2048, 1024), torch.float32, False),
          ((1024, 26), torch.float32, False), ((1176, 1280), torch.bfloat16, False)]


@pytest.mark.parametrize("kn,dtype,bias", SHAPES)
def test_linear_close_to_reference_and_row_invariant(kn, dtype, bias):
    k, n = kn
    g = torch.Generator(device="cuda").manual_seed(k + n)
    w = (torch.randn(n, k, generator=g, device="cuda") * k ** -0.5).to(dtype)
    b = torch.randn(n, generator=g, device="cuda").to(dtype) if bias else None
    x = torch.randn(8, 16, k, generator=g, device="cuda").to(dtype)
    full = bi.linear(x, w, b)
    ref = reference.linear(x.double(), w.double(), None if b is None else b.double())
    tol = 2 ** -7 if dtype == torch.bfloat16 else 1e-5
    assert float((full.double() - ref).abs().max()) <= tol * max(1.0, float(ref.abs().max()))
    for rows in (slice(0, 1), slice(3, 5), slice(4, 8)):
        assert torch.equal(bi.linear(x[rows].contiguous(), w, b), full[rows]), rows


@pytest.mark.parametrize("n,dtype,wdtype", [(1024, torch.bfloat16, torch.float32),
                                            (1280, torch.bfloat16, torch.bfloat16),
                                            (2048, torch.bfloat16, torch.float32)])
def test_rms_norm_close_to_reference_and_row_invariant(n, dtype, wdtype):
    g = torch.Generator(device="cuda").manual_seed(n)
    x = (torch.randn(6, 16, n, generator=g, device="cuda") * 3).to(dtype)
    w = (1 + 0.1 * torch.randn(n, generator=g, device="cuda")).to(wdtype)
    full = bi.rms_norm(x, w, 1e-6)
    ref = reference.rms_norm(x, w, 1e-6)
    assert full.dtype == ref.dtype
    assert float((full.float() - ref.float()).abs().max()) <= 2 ** -7 * float(ref.float().abs().max())
    assert float((full != ref).float().mean()) < 0.05
    assert torch.equal(bi.rms_norm(x[2:3].contiguous(), w, 1e-6), full[2:3])


@pytest.mark.parametrize("broadcast", [False, True])
def test_ada_rms_norm_close_to_reference_and_row_invariant(broadcast):
    """Action-expert norm: [B, 30, 1024] with [B, 1, 3·1024] modulation chunks (or one row broadcast)."""
    g = torch.Generator(device="cuda").manual_seed(7)
    x = (torch.randn(6, 30, 1024, generator=g, device="cuda") * 3).bfloat16()
    mod = (0.3 * torch.randn(1 if broadcast else 6, 1, 3072, generator=g, device="cuda")).bfloat16()
    if broadcast:
        mod = mod.expand(6, -1, -1)
    scale, shift, _ = mod.split(1024, dim=-1)
    full = bi.ada_rms_norm(x, scale, shift, 1e-6)
    ref = reference.ada_rms_norm(x, scale, shift, 1e-6)
    assert full.dtype == ref.dtype == torch.bfloat16
    assert float((full.float() - ref.float()).abs().max()) <= 2 ** -7 * float(ref.float().abs().max())
    assert torch.equal(bi.ada_rms_norm(x[2:3].contiguous(), scale[2:3], shift[2:3], 1e-6), full[2:3])


def test_capture_in_cuda_graphs():
    x = torch.randn(32, 1024, device="cuda").bfloat16()
    w = (torch.randn(2560, 1024, device="cuda") * 0.03).bfloat16()
    b = torch.randn(2560, device="cuda").bfloat16()
    nw = torch.ones(1024, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        eager = (bi.linear(x, w, b), bi.rms_norm(x, nw, 1e-6))  # compiles the kernels
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = (bi.linear(x, w, b), bi.rms_norm(x, nw, 1e-6))
        graph.replay()
    stream.synchronize()
    assert all(torch.equal(c, e) for c, e in zip(captured, eager))


@pytest.mark.parametrize("shape,heads_view",
                         [((2112, 16, 80), "vit"), ((4, 16, 16, 128), "llm"), ((1, 756, 2, 128), "llm")])
def test_rotary_embedding_is_bitwise_equal_to_the_reference(shape, heads_view):
    g = torch.Generator(device="cuda").manual_seed(sum(shape))
    if heads_view == "vit":  # q/k are strided views of a packed [N, 3, H, D] projection
        n, h, d = shape
        qkv = torch.randn(n, 3 * h * d, generator=g, device="cuda").bfloat16()
        x = qkv.reshape(n, 3, h, d).permute(1, 0, 2, 3).unbind(0)[0]
        angles = torch.rand(n, d // 2, generator=g, device="cuda") * 60
        emb = torch.cat((angles, angles), -1)
        cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    else:  # [B, L, H, D] split from a fused QKV row; tables [B, L, 1, D] of BF16-rounded values in FP32
        b, length, h, d = shape
        fused = torch.randn(b, length, h * d + 64, generator=g, device="cuda").bfloat16()
        x = fused[..., :h * d].reshape(b, length, h, d)
        angles = torch.rand(b, length, d // 2, generator=g, device="cuda") * 600
        emb = torch.cat((angles, angles), -1)
        cos = emb.cos().bfloat16().float()[:, :, None, :]
        sin = emb.sin().bfloat16().float()[:, :, None, :]
    got = bi.rotary_embedding(x, cos, sin)
    want = reference.rotary_embedding(x, cos, sin)
    assert got.dtype == x.dtype and got.shape == x.shape
    assert torch.equal(got, want)
    assert torch.equal(bi.rotary_embedding(x.float(), cos, sin), reference.rotary_embedding(x.float(), cos, sin))


def test_rotary_kernel_has_no_fused_multiply_add():
    """The compiled rotary kernel must not contain FFMA (it would change the rounding)."""
    import re
    import shutil
    import subprocess
    import tempfile

    nvdisasm = shutil.which("nvdisasm") or "/usr/local/cuda/bin/nvdisasm"
    if not shutil.which(nvdisasm) and not __import__("os").path.exists(nvdisasm):
        pytest.skip("nvdisasm not available")
    kernels = []
    for dtype in (torch.bfloat16, torch.float32):  # the launch rotary_embedding makes
        x = torch.randn(8, 16, 80, device="cuda").to(dtype)
        cos = torch.rand(8, 80, device="cuda")
        out = torch.empty_like(x)
        kernels.append(bi._rotary[(8,)](x, cos, cos, out, 16, x.stride(0), x.stride(1), HALF=40, BLOCK_H=16,
                                        BLOCK_D=64, num_warps=4))
    for k in kernels:
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(k.asm["cubin"])
            f.flush()
            sass = subprocess.run([nvdisasm, f.name], capture_output=True, text=True, check=True).stdout
        assert not re.search(r"\bFFMA", sass)


def test_gelu_tanh_and_mul_is_bitwise_equal_to_aten_on_every_bf16_input():
    import torch.nn.functional as F

    gate = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda().reshape(64, 1024)
    for up in (torch.ones_like(gate), torch.randn(64, 1024, device="cuda").bfloat16()):
        want = F.gelu(gate, approximate="tanh") * up
        got = bi.gelu_tanh_and_mul(gate, up)
        same = (got.view(torch.int16) == want.view(torch.int16)) | (torch.isnan(got) & torch.isnan(want))
        assert bool(same.all()), int((~same).sum())
    fused = torch.randn(2, 30, 8192, device="cuda").bfloat16()  # halves of a fused gate/up output
    gate, up = fused.split(4096, dim=-1)
    assert torch.equal(bi.gelu_tanh_and_mul(gate, up), reference.gelu_tanh_and_mul(gate, up))


@pytest.mark.parametrize("b", [1, 4])
def test_rotary_rounded_and_gated_residual_are_bitwise_equal_to_the_reference(b):
    """q/k as views of a fused [B, 30, 2560] projection, the prefix, and the expert residual."""
    g = torch.Generator(device="cuda").manual_seed(b)
    qkv = torch.randn(b, 30, 2560, generator=g, device="cuda").bfloat16()
    angles = torch.randn(b, 30, 256, generator=g, device="cuda") * 50
    cos, sin = angles.cos().bfloat16()[:, :, None], angles.sin().bfloat16()[:, :, None]
    for x in (qkv[..., :2048].reshape(b, 30, 8, 256), qkv[..., 2048:2304].reshape(b, 30, 1, 256)):
        assert torch.equal(bi.rotary_embedding_rounded(x, cos, sin), reference.rotary_embedding_rounded(x, cos, sin))
    x = torch.randn(1, 848, 8, 256, generator=g, device="cuda").bfloat16()
    angles = torch.randn(1, 848, 256, generator=g, device="cuda") * 400
    cos, sin = angles.cos().bfloat16()[:, :, None], angles.sin().bfloat16()[:, :, None]
    assert torch.equal(bi.rotary_embedding_rounded(x, cos, sin), reference.rotary_embedding_rounded(x, cos, sin))
    h, y = (torch.randn(b, 30, 1024, generator=g, device="cuda").bfloat16() for _ in range(2))
    gate = torch.randn(1, 1, 3072, generator=g, device="cuda").bfloat16().expand(b, -1, -1)[..., 2048:]
    assert torch.equal(bi.gated_residual(h, y, gate), reference.gated_residual(h, y, gate))
