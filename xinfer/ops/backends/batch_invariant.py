"""Batch-invariant Triton kernels for ``linear``, ``rms_norm`` and ``ada_rms_norm`` (CUDA).

A row's result never depends on how many rows are computed with it, so a
stage built on these operators gives the same bits at B1 and BN (design §6.4):

- ``linear``: the Wall-X 4B service's batch-invariant GEMM (wall-x
  ``02c5351d`` ``wall_x/infer/quant/batch_invariant.py``, via HarriX
  ``engines/mxfp8/sm120/batch_invariant.py``). Tiles of 16 rows; the K split
  and the order of the FP32 partial sums depend on (K, N, dtype) only.
  BF16 and FP32 inputs (FP32 with IEEE products, no TF32), FP32 accumulation,
  one rounding to the input dtype.
- ``rms_norm``: one program per row, the row's sum of squares in one
  reduction, then ``(weight * (x * rsqrt(mean + eps)))`` in FP32 and one
  rounding (the reference semantics).
- ``ada_rms_norm``: the same row reduction, then ``(x * rsqrt(mean + eps))
  * (1 + scale) + shift`` in FP32 with the row's batch-entry modulation, and
  one rounding (adaptive RMSNorm of action experts).

Contracts against the torch references: tolerance (accumulation order),
deterministic, batch invariant (tested), graph safe. Registered by
``ops.registry.load_plugins()`` when Triton imports and a CUDA device is
present (``XINFER_BATCH_INVARIANT=0`` skips it).

``rotary_embedding`` fuses the FP32 rotation and the casts into one kernel whose
products and sum are separate ``.rn`` instructions (no FMA); it is bitwise equal
to the reference (tested, including the SASS having no FFMA) and the default.
Other elementwise chains are fused the same way, each bitwise equal to its
torch composition (tested): ``rotary_embedding_rounded`` (BF16 products rounded
before the add), ``gated_residual`` (``x + bf16(y·gate)``) and
``gelu_tanh_and_mul`` (ATen's tanh GELU, checked on all 65536 BF16 inputs).

``rms_norm`` is faster than the multi-kernel reference everywhere and is the
default. ``linear`` keeps 16-row tiles and split-K for small batches, which is
slow for large M (ViT, prompt), so it ranks below cuBLAS: only the
batch-invariant op table of a batching stage selects it.
"""

import torch

from xinfer.ops.spec import Contract

BACKEND = "batch_invariant"

try:  # Triton is optional (CPU-only installs)
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # noqa: BLE001
    triton = None

if triton is not None:

    @triton.jit
    def _partial_mm(X, W, P, Y, BIAS, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, X0: tl.constexpr,
                    X1: tl.constexpr, W0: tl.constexpr, W1: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SPLIT: tl.constexpr, PREC: tl.constexpr, BIAS_PRESENT: tl.constexpr):
        mm = tl.program_id(0) * 16 + tl.arange(0, 16)
        nn = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        group = tl.program_id(2)
        block_count = tl.cdiv(K, BK * SPLIT)
        acc = tl.full((16, BN), 0, tl.float32)
        for step in range(block_count):
            k = (group * block_count + step) * BK + kk
            x = tl.load(X + mm[:, None] * X0 + k[None, :] * X1, (mm[:, None] < M) & (k[None, :] < K), 0)
            w = tl.load(W + k[:, None] * W0 + nn[None, :] * W1, (k[:, None] < K) & (nn[None, :] < N), 0)
            acc = tl.dot(x, w, acc, input_precision=PREC)
        if SPLIT == 1:
            if BIAS_PRESENT:
                acc += tl.load(BIAS + nn, nn < N, 0)[None, :]
            tl.store(Y + mm[:, None] * N + nn[None, :], acc, (mm[:, None] < M) & (nn[None, :] < N))
        else:
            tl.store(P + group * M * N + mm[:, None] * N + nn[None, :], acc, (mm[:, None] < M) & (nn[None, :] < N))

    @triton.jit
    def _finish_mm(P, Y, BIAS, M: tl.constexpr, N: tl.constexpr, SPLIT: tl.constexpr, BIAS_PRESENT: tl.constexpr,
                   B: tl.constexpr):
        x = tl.program_id(0) * B + tl.arange(0, B)
        value = tl.load(P + x, x < M * N, 0)
        for i in tl.static_range(1, SPLIT):
            value += tl.load(P + i * M * N + x, x < M * N, 0)
        if BIAS_PRESENT:
            value += tl.load(BIAS + x % N, x < M * N, 0)
        tl.store(Y + x, value, x < M * N)

    @triton.jit
    def _rms_norm(X, W, Y, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask, other=0.0).to(tl.float32)
        mean = tl.sum(x * x, axis=0) / N
        inv = 1.0 / tl.sqrt(mean + eps)
        w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
        tl.store(Y + row * N + cols, (w * (x * inv)).to(Y.dtype.element_ty), mask)

    @triton.jit
    def _ada_rms_norm(X, SCALE, SHIFT, Y, N, L, MOD_STRIDE, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        b = row // L  # rows of one batch entry share its modulation
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask, other=0.0).to(tl.float32)
        mean = tl.sum(x * x, axis=0) / N
        inv = 1.0 / tl.sqrt(mean + eps)
        scale = tl.load(SCALE + b * MOD_STRIDE + cols, mask, other=0.0).to(tl.float32)
        shift = tl.load(SHIFT + b * MOD_STRIDE + cols, mask, other=0.0).to(tl.float32)
        tl.store(Y + row * N + cols, ((x * inv) * (1.0 + scale) + shift).to(Y.dtype.element_ty), mask)


if triton is not None:

    # Scalar PTX ``mul.rn``/``add.rn``: rounded separately, never contracted into an FMA.
    # (Triton's enable_fp_fusion=False is not enough: ptxas fused the packed f32x2
    # mul/add it emits for BF16 inputs into FFMA on sm_120.)
    @triton.jit
    def _mul_rn(a, b):
        return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32,
                                         is_pure=True, pack=1)

    @triton.jit
    def _add_rn(a, b):
        return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32,
                                         is_pure=True, pack=1)

    @triton.jit
    def _rotary(X, COS, SIN, Y, H, XS_T, XS_H, HALF: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
        t = tl.program_id(0)
        h = tl.arange(0, BLOCK_H)[:, None]
        d = tl.arange(0, BLOCK_D)[None, :]
        mask = (h < H) & (d < HALF)
        cmask = d < HALF
        x_ptr = X + t * XS_T + h * XS_H + d
        x1 = tl.load(x_ptr, mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + HALF, mask, other=0.0).to(tl.float32)
        c1 = tl.load(COS + t * 2 * HALF + d, cmask, other=0.0)
        c2 = tl.load(COS + t * 2 * HALF + HALF + d, cmask, other=0.0)
        s1 = tl.load(SIN + t * 2 * HALF + d, cmask, other=0.0)
        s2 = tl.load(SIN + t * 2 * HALF + HALF + d, cmask, other=0.0)
        # x * cos + rotate_half(x) * sin with rotate_half(x) = [-x2, x1]; separate roundings
        y1 = _add_rn(_mul_rn(x1, c1), _mul_rn(-x2, s1))
        y2 = _add_rn(_mul_rn(x2, c2), _mul_rn(x1, s2))
        y_ptr = Y + t * H * 2 * HALF + h * 2 * HALF + d
        tl.store(y_ptr, y1.to(Y.dtype.element_ty), mask)
        tl.store(y_ptr + HALF, y2.to(Y.dtype.element_ty), mask)


if triton is not None:

    @triton.jit
    def _rotary_rounded(X, COS, SIN, Y, H, XS_T, XS_H, HALF: tl.constexpr, BLOCK_H: tl.constexpr,
                        BLOCK_D: tl.constexpr):
        t = tl.program_id(0)
        h = tl.arange(0, BLOCK_H)[:, None]
        d = tl.arange(0, BLOCK_D)[None, :]
        mask = (h < H) & (d < HALF)
        cmask = d < HALF
        x_ptr = X + t * XS_T + h * XS_H + d
        x1 = tl.load(x_ptr, mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + HALF, mask, other=0.0).to(tl.float32)
        c1 = tl.load(COS + t * 2 * HALF + d, cmask, other=0.0).to(tl.float32)
        c2 = tl.load(COS + t * 2 * HALF + HALF + d, cmask, other=0.0).to(tl.float32)
        s1 = tl.load(SIN + t * 2 * HALF + d, cmask, other=0.0).to(tl.float32)
        s2 = tl.load(SIN + t * 2 * HALF + HALF + d, cmask, other=0.0).to(tl.float32)
        dt = Y.dtype.element_ty
        # Each product (exact in FP32 for BF16 operands) and the sum are rounded to the output dtype, as the
        # three separate torch kernels round them.
        y1 = _add_rn(_mul_rn(x1, c1).to(dt).to(tl.float32), _mul_rn(-x2, s1).to(dt).to(tl.float32))
        y2 = _add_rn(_mul_rn(x2, c2).to(dt).to(tl.float32), _mul_rn(x1, s2).to(dt).to(tl.float32))
        y_ptr = Y + t * H * 2 * HALF + h * 2 * HALF + d
        tl.store(y_ptr, y1.to(dt), mask)
        tl.store(y_ptr + HALF, y2.to(dt), mask)

    @triton.jit
    def _gated_residual(X, Y, G, OUT, N, L, G_STRIDE, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask, other=0.0).to(tl.float32)
        y = tl.load(Y + row * N + cols, mask, other=0.0).to(tl.float32)
        g = tl.load(G + (row // L) * G_STRIDE + cols, mask, other=0.0).to(tl.float32)
        dt = OUT.dtype.element_ty
        tl.store(OUT + row * N + cols, _add_rn(x, _mul_rn(y, g).to(dt).to(tl.float32)).to(dt), mask)

    @triton.jit
    def _gelu_tanh_and_mul(GATE, UP, OUT, N, G_STRIDE, U_STRIDE, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(GATE + row * G_STRIDE + cols, mask, other=0.0).to(tl.float32)
        up = tl.load(UP + row * U_STRIDE + cols, mask, other=0.0).to(tl.float32)
        dt = OUT.dtype.element_ty
        # ATen's tanh GELU in FP32: 0.5 * x * (1 + tanh(kBeta * (x + kKappa * x^3)))
        cube = _mul_rn(_mul_rn(x, x), x)
        inner = _mul_rn(0.7978845608028654, _add_rn(x, _mul_rn(0.044715, cube)))
        gelu = _mul_rn(_mul_rn(0.5, x), _add_rn(1.0, libdevice.tanh(inner)))
        tl.store(OUT + row * N + cols, _mul_rn(gelu.to(dt).to(tl.float32), up).to(dt), mask)


def rotary_embedding_rounded(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """One kernel for the rounded rotation, bitwise equal to the reference ``rotary_embedding_rounded``.

    Kernel ``_rotary_rounded``: one program per token row covering all heads; ``x·cos`` and ``rotate_half(x)·sin``
    (exact FP32 products of BF16 values) are each rounded to ``x.dtype`` and added with ``add.rn``, then rounded,
    as the three torch kernels do. Registered with a bitwise, batch-invariant, graph-safe contract.

    Args:
        x: ``[..., H, D]`` on CUDA, ``D`` even, last dim contiguous.
        cos: ``[..., 1, D]`` in ``x.dtype``, broadcastable over the heads (copied to a contiguous ``[T, D]``).
        sin: same shape and dtype as ``cos``.

    Returns:
        Contiguous ``[..., H, D]`` in ``x.dtype``.

    Raises:
        ValueError: ``D`` is odd, ``x``'s last dim is not contiguous, or ``cos``/``sin`` are not in ``x.dtype``.
    """
    *lead, h, d = x.shape
    if d % 2 or x.stride(-1) != 1 or cos.dtype != x.dtype or sin.dtype != x.dtype:
        raise ValueError("rotary_embedding_rounded needs an even, contiguous last dim and cos/sin in x.dtype")
    flat = x.reshape(-1, h, d)
    t = flat.shape[0]
    c = cos.expand(*lead, 1, d).reshape(t, d).contiguous()
    s = sin.expand(*lead, 1, d).reshape(t, d).contiguous()
    out = torch.empty((t, h, d), dtype=x.dtype, device=x.device)
    if t:
        half = d // 2
        _rotary_rounded[(t,)](flat, c, s, out, h, flat.stride(0), flat.stride(1), HALF=half,
                              BLOCK_H=triton.next_power_of_2(h), BLOCK_D=triton.next_power_of_2(half), num_warps=4)
    return out.reshape(x.shape)


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x + bf16(y · gate)`` in one kernel, bitwise equal to the reference.

    Kernel ``_gated_residual``: one program per row; the product of the two BF16 values (exact in FP32) is
    rounded to the dtype before ``add.rn`` and the final rounding. Rows of batch entry ``b`` read ``gate[b]``.

    Args:
        x: ``[B, L, K]`` on CUDA.
        y: ``[B, L, K]``, same dtype.
        gate: ``[B, 1, K]`` with unit stride along ``K`` (a chunk of the modulation, possibly broadcast).

    Returns:
        ``[B, L, K]`` in ``x.dtype``.
    """
    b, rows, n = x.shape
    if gate.shape != (b, 1, n) or gate.stride(-1) != 1:
        raise ValueError("gated_residual expects gate [B, 1, K] with a unit stride along K")
    xf, yf = x.reshape(-1, n).contiguous(), y.reshape(-1, n).contiguous()
    out = torch.empty_like(xf)
    if xf.shape[0]:
        _gated_residual[(xf.shape[0],)](xf, yf, gate, out, n, rows, gate.stride(0),
                                        BLOCK=triton.next_power_of_2(n), num_warps=4)
    return out.reshape(x.shape)


def gelu_tanh_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``bf16(gelu_tanh(gate)) * up`` in one kernel, bitwise equal to ATen's two kernels (checked on all inputs).

    Kernel ``_gelu_tanh_and_mul``: ATen's FP32 formula ``0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`` with
    libdevice ``tanh`` (separate ``.rn`` products and sums), rounded to the dtype, then multiplied by ``up`` (exact
    for BF16) and rounded. Equal to ``F.gelu(gate, approximate="tanh") * up`` on all 65536 BF16 inputs (tested;
    contracting ``x + 0.044715·x³`` into an FMA gives the same BF16 results).

    Args:
        gate: ``[..., N]`` on CUDA (may be a view, e.g. half of a fused gate/up output).
        up: ``[..., N]``, same shape and dtype.

    Returns:
        Contiguous ``[..., N]`` in ``gate.dtype``.
    """
    n = gate.shape[-1]
    g, u = gate.reshape(-1, n), up.reshape(-1, n)
    if g.stride(-1) != 1:
        g = g.contiguous()
    if u.stride(-1) != 1:
        u = u.contiguous()
    out = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    if g.shape[0] and n:
        block = 1024
        _gelu_tanh_and_mul[(g.shape[0], triton.cdiv(n, block))](g, u, out, n, g.stride(0), u.stride(0), BLOCK=block,
                                                                num_warps=4)
    return out


def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """One kernel for ``(x32 * cos + rotate_half(x32) * sin).to(x.dtype)``, bitwise equal to the reference.

    Each product and the sum are separate ``.rn`` PTX instructions (no fused
    multiply-add), as in the torch composition; the BF16 store rounds to nearest
    even like ``Tensor.to``.

    Kernel ``_rotary``: one program per token row ``t`` covering all ``H`` heads and half of ``D``; it loads
    ``x1``/``x2`` (the two halves) as FP32 and both halves of the token's ``cos``/``sin`` rows, and stores
    ``x1 * cos1 + (-x2) * sin1`` and ``x2 * cos2 + x1 * sin2`` through inline ``mul.rn``/``add.rn`` PTX (``_mul_rn``,
    ``_add_rn``). Registered as ``rotary_embedding`` with a bitwise, batch-invariant, graph-safe contract.

    Args:
        x: ``[..., H, D]`` on CUDA, ``D`` even, last dim contiguous.
        cos: FP32 ``[..., 1, D]`` broadcastable to ``x`` over the heads (copied to a contiguous ``[T, D]``).
        sin: FP32, same shape as ``cos``.

    Returns:
        Contiguous ``[..., H, D]`` in ``x.dtype``.

    Raises:
        ValueError: ``D`` is odd, ``x``'s last dim is not contiguous, or ``cos``/``sin`` are not FP32.
    """
    *lead, h, d = x.shape
    if d % 2 or x.stride(-1) != 1 or cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("rotary_embedding needs an even, contiguous last dim and FP32 cos/sin")
    flat = x.reshape(-1, h, d)
    t = flat.shape[0]
    c = cos.expand(*lead, 1, d).reshape(t, d).contiguous()
    s = sin.expand(*lead, 1, d).reshape(t, d).contiguous()
    out = torch.empty((t, h, d), dtype=x.dtype, device=x.device)
    if t:
        half = d // 2
        _rotary[(t,)](flat, c, s, out, h, flat.stride(0), flat.stride(1), HALF=half,
                      BLOCK_H=triton.next_power_of_2(h), BLOCK_D=triton.next_power_of_2(half), num_warps=4)
    return out.reshape(x.shape)


def _tiles(k: int, n: int, dtype: torch.dtype):
    """(BN, BK, SPLIT, warps): a function of the weight shape and dtype only, never of M.

    Defaults to 64x64 tiles, no split-K and 4 warps; FP32 with ``K >= 256`` and a few Wall-X BF16 shapes
    use a K split (and narrower tiles).

    Args:
        k: input features ``K``.
        n: output features ``N``.
        dtype: input dtype.

    Returns:
        ``(BN, BK, SPLIT, warps)``: N tile width, K tile depth, number of K groups, warps per program.
    """
    bn, bk, split, warps = 64, 64, 1, 4
    if dtype == torch.float32 and k >= 256:
        bk, split = 32, 8
    elif dtype == torch.bfloat16:
        if (k, n) == (1024, 2560):
            bk, split, warps = 32, 4, 2
        elif (k, n) == (1024, 1024):
            bn, split = 32, 4
        elif (k, n) == (2048, 1024):
            bk, split = 32, 8
    return bn, bk, split, warps


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``x @ weight.T + bias`` (``weight [N, K]``), FP32 accumulation, one rounding to ``x.dtype``.

    Kernels: ``_partial_mm`` runs a ``(ceil(M / 16), ceil(N / BN), SPLIT)`` grid; each program accumulates a
    16-row x ``BN`` tile over its contiguous share of K in ``BK`` steps with ``tl.dot`` (IEEE FP32 precision, no
    TF32). Without a K split it adds the bias and stores directly; with one it writes FP32 partials to a
    ``[SPLIT, M, N]`` buffer, and ``_finish_mm`` adds them in group order, adds the bias and stores. The tiling
    comes from ``_tiles`` (K, N and dtype only), so a row's bits do not depend on ``M``.

    Registered as ``linear`` (tolerance vs the reference, deterministic, batch invariant, graph safe) at a
    priority below the reference, so only batch-invariant op tables select it.

    Args:
        x: ``[..., K]`` BF16 or FP32 on CUDA.
        weight: ``[N, K]`` in ``x.dtype``.
        bias: optional ``[N]`` in ``x.dtype``.

    Returns:
        ``[..., N]`` in ``x.dtype``.

    Raises:
        ValueError: ``x`` is not BF16/FP32, ``weight`` has another dtype, or ``bias`` has another dtype.
    """
    if x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype != x.dtype:
        raise ValueError(f"batch-invariant linear takes BF16 or FP32 inputs and weights, got {x.dtype}/{weight.dtype}")
    if bias is not None and bias.dtype != x.dtype:
        raise ValueError("batch-invariant linear needs the bias in the input dtype")
    flat = x.reshape(-1, x.shape[-1])
    m, k = flat.shape
    n = weight.shape[0]
    weight_t = weight.t()
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    if m == 0 or n == 0:
        return out
    bn, bk, split, warps = _tiles(k, n, x.dtype)
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32) if split > 1 else out
    _partial_mm[(triton.cdiv(m, 16), triton.cdiv(n, bn), split)](
        flat, weight_t, partial, out, bias, m, n, k, *flat.stride(), *weight_t.stride(), bn, bk, split, "ieee",
        bias is not None, num_warps=warps, num_stages=2)
    if split > 1:
        _finish_mm[(triton.cdiv(m * n, 256),)](partial, out, bias, m, n, split, bias is not None, 256)
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with the reference semantics, one Triton program per row.

    Kernel ``_rms_norm``: loads the whole row (block = next power of two of ``K``) as FP32, reduces the sum of
    squares in one ``tl.sum``, and stores ``weight * (x * (1 / sqrt(mean + eps)))`` rounded once to the output
    dtype. Registered as ``rms_norm`` (tolerance vs the reference, deterministic, batch invariant, graph safe)
    and selected by default on CUDA.

    Args:
        x: ``[..., K]`` on CUDA.
        weight: ``[K]``, read as FP32.
        eps: added to the mean square.

    Returns:
        ``[..., K]`` in ``x.dtype``.
    """
    n = x.shape[-1]
    flat = x.reshape(-1, n).contiguous()
    out = torch.empty_like(flat)
    if flat.shape[0]:
        _rms_norm[(flat.shape[0],)](flat, weight.contiguous(), out, n, float(eps),
                                    BLOCK=triton.next_power_of_2(n), num_warps=4 if n <= 2048 else 8)
    return out.reshape(x.shape)


def ada_rms_norm(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    """Adaptive RMSNorm with the reference semantics, one Triton program per row.

    Kernel ``_ada_rms_norm``: loads the row as FP32, reduces the sum of squares in one ``tl.sum`` and stores
    ``(x * (1 / sqrt(mean + eps))) * (1 + scale) + shift`` rounded once to the output dtype; the rows of batch
    entry ``b`` read ``scale[b]`` and ``shift[b]``. Registered as ``ada_rms_norm`` (tolerance vs the reference:
    the compiler may contract the multiply-add; deterministic, batch invariant, graph safe).

    Args:
        x: ``[B, L, K]`` on CUDA.
        scale: ``[B, 1, K]`` with unit stride along ``K`` (for example a chunk of ``[B, 1, 3K]``).
        shift: ``[B, 1, K]``, same layout as ``scale``.
        eps: added to the mean square.

    Returns:
        ``[B, L, K]`` in ``x.dtype``.
    """
    b, rows, n = x.shape
    if scale.shape != (b, 1, n) or shift.shape != (b, 1, n) or scale.stride(-1) != 1 or shift.stride(-1) != 1 \
            or scale.stride(0) != shift.stride(0):
        raise ValueError("ada_rms_norm expects scale and shift [B, 1, K] with equal layouts")
    flat = x.reshape(-1, n).contiguous()
    out = torch.empty_like(flat)
    if flat.shape[0]:
        _ada_rms_norm[(flat.shape[0],)](flat, scale, shift, out, n, rows, scale.stride(0), float(eps),
                                        BLOCK=triton.next_power_of_2(n), num_warps=4 if n <= 2048 else 8)
    return out.reshape(x.shape)


def available() -> bool:
    """Whether Triton imports and a CUDA device is present."""
    return triton is not None and torch.cuda.is_available()


def register(registry) -> str | None:
    """Register the Triton kernels for CUDA platforms.

    ``linear`` (priority -2, below the reference), ``rms_norm`` and ``ada_rms_norm`` (priority 5) with a tolerance
    contract; ``rotary_embedding``, ``rotary_embedding_rounded``, ``gated_residual`` and ``gelu_tanh_and_mul``
    (priority 5) with a bitwise contract; all deterministic, batch invariant and graph safe.

    Args:
        registry: the ``OpRegistry`` to register into.

    Returns:
        The version string (``"triton <version>"``), or None if Triton or CUDA is unavailable.
    """
    if not available():
        return None
    version = f"triton {triton.__version__}"
    contract = Contract(numerics="tolerance", deterministic=True, batch_invariant=True, graph_safe=True)
    for name, function, priority in (("linear", linear, -2), ("rms_norm", rms_norm, 5),
                                     ("ada_rms_norm", ada_rms_norm, 5)):
        registry.register(name, backend=BACKEND, platforms=("cuda",), priority=priority, version=version,
                          contract=contract)(function)
    exact = Contract(numerics="bitwise", deterministic=True, batch_invariant=True, graph_safe=True)
    for name, function in (("rotary_embedding", rotary_embedding),
                           ("rotary_embedding_rounded", rotary_embedding_rounded),
                           ("gated_residual", gated_residual), ("gelu_tanh_and_mul", gelu_tanh_and_mul)):
        registry.register(name, backend=BACKEND, platforms=("cuda",), priority=5, version=version,
                          contract=exact)(function)
    return version
