"""gemm / gemm_glu: every kernel on the shapes it takes, and the generic path, against FP32 torch."""
import pytest
import torch
import torch.nn.functional as F

from xinfer.ops.backends.cuda import _native, gemm
from xinfer.ops.layout import mxfp8 as layout
from conftest import MXFP8_TOLERANCE, close, dequantize, sm

ACT = {None: lambda y: y, "silu": F.silu, "gelu": F.gelu, "gelu_tanh": lambda y: F.gelu(y, approximate="tanh")}


def tuned_calls():
    """(op, kernel class, M, N, K) at both ends and the middle of every tuned range."""
    for op in ("gemm", "gemm_glu"):
        for cls in _native.KERNELS[op]:
            for (n, k), ranges in cls.SHAPES.items():
                for lo, hi in ranges:
                    for m in sorted({lo, (lo + hi) // 2, hi}):
                        yield pytest.param(op, cls, m, n, k, id=f"{cls.__name__}-{m}x{n}x{k}")


def fp32(t):
    return dequantize(t) if isinstance(t, tuple) else t.float()


def reference(op, x, weight, bias, activation, residual=None, row_scale=None, quantize=False):
    y = F.linear(fp32(x), fp32(weight), None if bias is None else bias.float())
    if row_scale is not None:
        y = y * row_scale[:, None]
    if op == "gemm":
        return ACT[activation](y) + (0 if residual is None else residual.float())
    n = weight.shape[0] // 2
    return ACT[activation](y[:, :n]) * y[:, n:]


def row_scale(m, rnd):
    return (rnd(m, dtype=torch.float32) > 0).float()


def arguments(op, cls, m, n, k, rnd):
    """Inputs with the epilogue ``cls`` implements (found by asking ``takes`` on the candidates): x, weight, bias
    and the keyword arguments."""
    rows = 2 * n if op == "gemm_glu" else n
    x, weight = rnd(m, k), rnd(rows, k, scale=0.05)
    if "mxfp8" in getattr(cls, "FEATURES", ()):
        x, weight = gemm.quantize_mxfp8(x), gemm.quantize_mxfp8(weight)
    for bias in (None, rnd(rows)):
        for activation in (("silu",) if op == "gemm_glu" else (None, "gelu", "gelu_tanh")):
            extras = ({},) if op == "gemm_glu" or activation else \
                ({}, {"residual": rnd(m, n)}, {"row_scale": row_scale(m, rnd)})
            for extra in extras:
                for quantize in (False, True):
                    kw = dict(activation=activation, **extra, **({"quantize": True} if quantize else {}))
                    if gemm.kernel_name(op, x, weight, bias, **kw) == cls.__name__:
                        return x, weight, bias, kw
    raise AssertionError(f"{cls.__name__} takes no call at {m}x{n}x{k}")


def tolerance(cls, quantized):
    return MXFP8_TOLERANCE if quantized else getattr(cls, "TOLERANCE", 5e-3)


@pytest.mark.parametrize("op,cls,m,n,k", list(tuned_calls()))
def test_kernel_matches_reference_and_generic(op, cls, m, n, k, rnd):
    x, weight, bias, kw = arguments(op, cls, m, n, k, rnd)
    entry = getattr(gemm, op)
    out = entry(x, weight, bias, **kw)
    quantized = kw.get("quantize", False)
    assert close(fp32(out), reference(op, x, weight, bias, **kw), tolerance(cls, quantized))
    # the generic path on the same call: the same value up to rounding (it rounds the GEMM result to BF16 before
    # the epilogue, a fused kernel rounds once)
    call = gemm._call(op, x, weight, bias, kw["activation"], kw.get("residual"), kw.get("row_scale"), quantized)
    generic = gemm._out(None, (m, n), call)
    (gemm._generic_glu if op == "gemm_glu" else gemm._generic_gemm)(call, generic)
    assert close(fp32(out), fp32(generic), tolerance(cls, quantized))
    if isinstance(x, tuple):
        return
    stored_kn = weight.t().contiguous().t()
    if getattr(cls, "LAYOUT", None) == "nk":  # a kernel that takes weights stored [N, K] only
        assert gemm.kernel_name(op, x, stored_kn, bias, **kw) != cls.__name__
        return
    # the transposed-storage form of the same weight gives the same kernel and bits
    assert gemm.kernel_name(op, x, stored_kn, bias, **kw) == cls.__name__
    if getattr(cls, "DETERMINISTIC", True):
        assert torch.equal(entry(x, stored_kn, bias, **kw), out)


@pytest.mark.parametrize("op,cls,m,n,k", list(tuned_calls()))
def test_outside_the_tuned_range_runs_generic(op, cls, m, n, k, rnd):
    ranges = cls.SHAPES[(n, k)]
    x, weight, bias, kw = arguments(op, cls, m, n, k, rnd)
    for rows in {lo - 1 for lo, _ in ranges} | {hi + 1 for _, hi in ranges}:
        if rows >= 1 and not any(lo <= rows <= hi for lo, hi in ranges):
            other = dict(kw, **{name: rnd(rows, n) if name == "residual" else row_scale(rows, rnd)
                                for name in ("residual", "row_scale") if name in kw})
            shorter = gemm.quantize_mxfp8(rnd(rows, k)) if isinstance(x, tuple) else rnd(rows, k)
            assert gemm.kernel_name(op, shorter, weight, bias, **other) != cls.__name__


@pytest.mark.parametrize("m,n,k", [(1, 8, 16), (7, 24, 40), (300, 1280, 1176), (33, 1024, 2048)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu", "gelu_tanh"])
def test_generic_gemm(m, n, k, bias, activation, rnd):
    x, weight, b = rnd(m, k), rnd(n, k, scale=0.05), (rnd(n) if bias else None)
    assert close(gemm.gemm(x, weight, b, activation=activation), reference("gemm", x, weight, b, activation))
    stored_kn = weight.t().contiguous().t()
    assert close(gemm.gemm(x, stored_kn, b, activation=activation), reference("gemm", x, weight, b, activation))


@pytest.mark.parametrize("m,n,k", [(1, 8, 16), (7, 24, 40), (300, 1280, 1176)])
@pytest.mark.parametrize("bias", [False, True])
def test_generic_residual_and_row_scale(m, n, k, bias, rnd):
    x, weight, b = rnd(m, k), rnd(n, k, scale=0.05), (rnd(n) if bias else None)
    residual, row_scale = rnd(m, n), (rnd(m, dtype=torch.float32) > 0).float()
    assert close(gemm.gemm(x, weight, b, residual=residual), reference("gemm", x, weight, b, None, residual=residual))
    assert close(gemm.gemm(x, weight, b, residual=residual, out=residual.clone()),  # the residual may be the output
                 reference("gemm", x, weight, b, None, residual=residual))
    for activation in (None, "silu"):
        out = gemm.gemm(x, weight, b, activation=activation, row_scale=row_scale)
        assert close(out, reference("gemm", x, weight, b, activation, row_scale=row_scale))
        assert bool((out[row_scale == 0] == (0 if b is None or activation is None else out[row_scale == 0])).all())
    with pytest.raises(ValueError):
        gemm.gemm(x, weight, b, activation="silu", residual=residual)


@pytest.mark.parametrize("m,n,k", [(1, 8, 16), (5, 24, 40), (300, 3456, 1280)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_generic_gemm_glu(m, n, k, bias, activation, rnd):
    x, weight, b = rnd(m, k), rnd(2 * n, k, scale=0.05), (rnd(2 * n) if bias else None)
    assert close(gemm.gemm_glu(x, weight, b, activation=activation), reference("gemm_glu", x, weight, b, activation))


def test_out_argument_and_errors(rnd):
    x, weight = rnd(4, 16), rnd(8, 16)
    out = torch.empty(4, 8, dtype=torch.bfloat16, device=x.device)
    assert gemm.gemm(x, weight, out=out) is out
    with pytest.raises(ValueError):
        gemm.gemm(x, rnd(8, 12))
    with pytest.raises(ValueError):
        gemm.gemm(x.float(), weight)
    with pytest.raises(ValueError):
        gemm.gemm(x, weight, activation="relu")
    with pytest.raises(ValueError):
        gemm.gemm_glu(x, rnd(7, 16))


@sm(110, 120)
def test_repeat_calls_are_bitwise_equal_and_replay_in_a_cuda_graph(rnd):
    """Every kernel of the GPU: the same bits on a second call, and in a captured graph replayed on new inputs (a
    kernel that adds its split-K partial sums in arrival order: the same values up to rounding). Scratch memory
    belongs to a stream, so each call runs once on the capture stream before the capture."""
    stream = torch.cuda.Stream()
    for param in tuned_calls():
        op, cls, m, n, k = param.values
        if m not in [hi for _, hi in cls.SHAPES[(n, k)]]:
            continue
        x, weight, bias, kw = arguments(op, cls, m, n, k, rnd)
        entry = getattr(gemm, op)

        def parts(t):  # a BF16 tensor, or the values and the scales (without their padding) of an MXFP8 pair
            if not isinstance(t, tuple):
                return (t,)
            return t[0], t[1][layout.scale_offsets(*t[0].shape, t[0].device)]

        def same(a, b):
            return torch.equal(a, b) if getattr(cls, "DETERMINISTIC", True) else close(a, b, tolerance(cls, False))

        first = entry(x, weight, bias, **kw)
        assert all(same(a, b) for a, b in zip(parts(entry(x, weight, bias, **kw)), parts(first)))
        out = tuple(torch.empty_like(t) for t in first) if isinstance(first, tuple) else torch.empty_like(first)
        with torch.cuda.stream(stream):
            entry(x, weight, bias, out=out, **kw)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            entry(x, weight, bias, out=out, **kw)
        x2 = gemm.quantize_mxfp8(rnd(m, k)) if isinstance(x, tuple) else rnd(m, k)
        expected = entry(x2, weight, bias, **kw)
        for old, new in zip(x if isinstance(x, tuple) else (x,), x2 if isinstance(x2, tuple) else (x2,)):
            old.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        assert all(same(a, b) for a, b in zip(parts(out), parts(expected))), cls.__name__


def test_growing_scratch_inside_a_capture_is_an_error(rnd):
    x, weight = rnd(3, 16), rnd(8, 16)
    stream = torch.cuda.Stream()
    with pytest.raises(RuntimeError, match="before capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            gemm.gemm(x, weight)
