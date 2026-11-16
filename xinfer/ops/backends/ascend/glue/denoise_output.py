"""Denoise output head + Euler step: the white list, dispatch and backend of :func:`denoise_output_euler`.

Contract: docs/denoise_output/README.md. Rules (AGENTS.md, "去噪输出头族"): there is no generic path. A request is
served by the first white-list entry that accepts it (``backend="auto"``) or by the named backend; anything else
raises with the closest entry and the unmet condition. Shapes outside the white list are for the caller to run as
torch operations.

The white list is the single table ``_TABLE`` below: dispatch reads it, tests/python/test_denoise_output_euler.py is
parametrized by it and benchmarks/denoise_output/README.md is generated from it (tests/perf/denoise_output_readme.py).
Entry ranges come from tests/perf/denoise_output_scan.py: an entry is listed where it is at least 5% faster (P50 GPU
time, S0) than the torch composition its baseline uses, over runs of at least 3 consecutive scan points.
CUDA entries carry their source performance admission. Ascend auto selection is
admitted only over the measured Cube-head envelopes (A32, B<=4, M<=32, K1024,
2026-11-10); elsewhere the Ascend programs stay explicit named candidates and
backend='ascend_tilelang' with its 'fused_tail' alias retains the validated
input domains.

A row's result depends only on that row (fixed reduction tree, independent of M, B and the CTA tiling). No CTA waits
for another one.
"""

from typing import Optional, Union

import torch

from xinfer.ops.backends.ascend import _provider
from xinfer.ops.backends.ascend.glue.core import CORE


_PRENORM = {"none": 0, "rmsnorm": 1, "adarms": 2}

# ---------------------------------------------------------------------------
# White list
# ---------------------------------------------------------------------------
# Fields: backend; arch; k (accepted K values); a, m, batch: inclusive ranges ([lo, hi]); h_dtype / w_dtype (accepted);
# prenorm (accepted); bias, row_mask, dof, pinned (accepted presence: True / False); dt ("scalar" / "tensor");
# priority (auto tries higher first; < 0: by name only); scan_ref: the scan the ranges come from.
# Optional fields (default in brackets): heads, accepted head counts of a [heads, A, K] weight [(1,)]; ada_table,
# accepted presence of per-row modulation tables (ada_index) [(False,)]; round_normed [(False,)]; velocity
# ("flow" / "x_pred") [("flow",)]; padding_optional, dof_mask without padding_velocity (= 0) [False].
_BF16, _FP32 = "bf16", "fp32"
_OPTIONAL = (False, True)
_TABLE = [
    # FP32 hidden (residual stream), adaptive RMSNorm folded into a BF16 head
    dict(
        backend="fused_tail",
        arch=110,
        k=(1024,),
        a=[[26, 32]],
        m=[[1, 32]],
        batch=[[1, 4]],
        h_dtype=(_FP32,),
        w_dtype=(_BF16,),
        prenorm=("adarms",),
        bias=_OPTIONAL,
        row_mask=_OPTIONAL,
        dof=_OPTIONAL,
        pinned=_OPTIONAL,
        dt=("scalar", "tensor"),
        priority=10,
        scan_ref="Thor 2026-11-05 S0 do_adarms (0.07-0.33 of torch)",
    ),
    # hidden after the final norm, FP32 head (dof blend, per-request dt, pinned rows)
    dict(
        backend="fused_tail",
        arch=110,
        k=(1024,),
        a=[[26, 32]],
        m=[[1, 32]],
        batch=[[1, 4]],
        h_dtype=(_BF16, _FP32),
        w_dtype=(_FP32,),
        prenorm=("none",),
        bias=_OPTIONAL,
        row_mask=_OPTIONAL,
        dof=_OPTIONAL,
        pinned=_OPTIONAL,
        dt=("scalar", "tensor"),
        priority=10,
        scan_ref="Thor 2026-11-05 S0 do_fp32head, do_fp32head_scalar (0.05-0.40 of torch)",
    ),
    # BF16 head + bias on a BF16 hidden; at B = 4 the rows exceed one wave and cuBLAS catches up
    dict(
        backend="fused_tail",
        arch=110,
        k=(1024,),
        a=[[26, 32]],
        m=[[1, 32]],
        batch=[[1, 2]],
        h_dtype=(_BF16,),
        w_dtype=(_BF16,),
        prenorm=("none",),
        bias=_OPTIONAL,
        row_mask=_OPTIONAL,
        dof=_OPTIONAL,
        pinned=_OPTIONAL,
        dt=("scalar", "tensor"),
        priority=10,
        scan_ref="Thor 2026-11-05 S0 do_bf16head (B <= 2: 0.10-0.73 of torch; B = 4 fails from M = 21)",
    ),
    # FP32 hidden (residual stream), adaptive RMSNorm folded into a BF16 head
    dict(
        backend="fused_tail",
        arch=120,
        k=(1024,),
        a=[[26, 32]],
        m=[[1, 32]],
        batch=[[1, 4]],
        h_dtype=(_FP32,),
        w_dtype=(_BF16,),
        prenorm=("adarms",),
        bias=_OPTIONAL,
        row_mask=_OPTIONAL,
        dof=_OPTIONAL,
        pinned=_OPTIONAL,
        dt=("scalar", "tensor"),
        priority=10,
        scan_ref="RTX 5090 2026-11-05 S0 do_adarms (0.11-0.17 of torch)",
    ),
    # hidden after the final norm, FP32 head (dof blend, per-request dt, pinned rows)
    dict(
        backend="fused_tail",
        arch=120,
        k=(1024,),
        a=[[26, 32]],
        m=[[1, 32]],
        batch=[[1, 4]],
        h_dtype=(_BF16, _FP32),
        w_dtype=(_FP32,),
        prenorm=("none",),
        bias=_OPTIONAL,
        row_mask=_OPTIONAL,
        dof=_OPTIONAL,
        pinned=_OPTIONAL,
        dt=("scalar", "tensor"),
        priority=10,
        scan_ref="RTX 5090 2026-11-05 S0 do_fp32head, do_fp32head_scalar (0.14-0.32 of torch)",
    ),
    # BF16 head + bias on a BF16 hidden
    dict(
        backend="fused_tail",
        arch=120,
        k=(1024,),
        a=[[26, 32]],
        m=[[1, 32]],
        batch=[[1, 4]],
        h_dtype=(_BF16,),
        w_dtype=(_BF16,),
        prenorm=("none",),
        bias=_OPTIONAL,
        row_mask=_OPTIONAL,
        dof=_OPTIONAL,
        pinned=_OPTIONAL,
        dt=("scalar", "tensor"),
        priority=10,
        scan_ref="RTX 5090 2026-11-05 S0 do_bf16head (0.22-0.47 of torch)",
    ),
]

_TABLE += [
    # K = 1280, up to 80 rows: per-row modulation rows (adaptive RMSNorm, normalized row rounded to BF16), one or
    # two FP32 heads selected per row, x-prediction velocity, dof-masked Euler update (no cross-CTA waits)
    dict(
        backend="cluster_tail",
        arch=120,
        k=(1280,),
        a=[[1, 32]],
        m=[[1, 80]],
        batch=[[1, 1]],
        h_dtype=(_BF16,),
        w_dtype=(_FP32,),
        prenorm=("adarms",),
        bias=(False,),
        row_mask=(False,),
        dof=_OPTIONAL,
        pinned=(False,),
        dt=("scalar", "tensor"),
        heads=(1, 2),
        ada_table=_OPTIONAL,
        round_normed=(True,),
        velocity=("flow", "x_pred"),
        padding_optional=True,
        priority=10,
        scan_ref="RTX 5090 2026-11-06 S0 do_cluster_xpred (A 26, K 1280)",
    ),
]

# Named NPU candidates retain the validated K1024 fused-tail domains. CUDA
# speed-qualified ranges do not qualify NPU auto dispatch. Direct FFI retains
# its wider K1024/A1..32/B*M<=2^20 contract, including RMSNorm.
_TABLE.extend(
    dict(
        backend="ascend_tilelang",
        arch=950,
        k=(1024,),
        a=[[26, 32]],
        m=[[1, 32]],
        batch=[[1, 4]],
        h_dtype=h_dtype,
        w_dtype=w_dtype,
        prenorm=(prenorm,),
        bias=_OPTIONAL,
        row_mask=_OPTIONAL,
        dof=_OPTIONAL,
        pinned=_OPTIONAL,
        dt=("scalar", "tensor"),
        priority=-1,
        scan_ref="Ascend950 named correctness candidate; NPU auto performance admission pending",
    )
    for h_dtype, w_dtype, prenorm in (
        ((_FP32,), (_BF16,), "adarms"),
        ((_BF16, _FP32), (_FP32,), "none"),
        ((_BF16,), (_BF16,), "none"),
    )
)

# This is a target-owned correctness candidate, not a CUDA speed-range clone.
_TABLE.append(dict(
    backend="ascend_cluster_tail", arch=950, k=(1280,), a=[[1, 32]],
    m=[[1, 80]], batch=[[1, 1]], h_dtype=(_BF16,), w_dtype=(_FP32,),
    prenorm=("adarms",), bias=(False,), row_mask=(False,), dof=_OPTIONAL,
    pinned=(False,), dt=("scalar", "tensor"), heads=(1, 2),
    ada_table=_OPTIONAL, round_normed=(True,), velocity=("flow", "x_pred"),
    padding_optional=True, priority=-1,
    scan_ref="Ascend950 K1280 correctness candidate; NPU performance admission pending",
))

# NPU auto admission: the Cube head programs measured faster than every complete
# CANN chain over the paired S0 envelope (A32, B<=4, M<=32, contiguous rows,
# 2026-11-10) and the K1280 cluster tail over its full M in [1,80]
# (2026-11-10). Outside these envelopes auto keeps raising; the wide named
# entries above stay available for explicit backend selection.
_TABLE.extend(
    dict(
        backend="ascend_tilelang",
        arch=950,
        k=(1024,),
        a=[[32, 32]],
        m=[[1, 32]],
        batch=[[1, 4]],
        h_dtype=h_dtype,
        w_dtype=w_dtype,
        prenorm=(prenorm,),
        bias=_OPTIONAL,
        row_mask=_OPTIONAL,
        dof=_OPTIONAL,
        pinned=_OPTIONAL,
        dt=("scalar", "tensor"),
        priority=10,
        scan_ref=(
            "Ascend950 2026-11-10 S0 cube heads vs complete CANN chains "
            "(npu/benchmarks/results/nv_parity_denoise_k1024_v22.json, denoise_auto_admission.json)"
        ),
    )
    for h_dtype, w_dtype, prenorm in (
        ((_FP32,), (_BF16,), "adarms"),
        ((_BF16,), (_FP32,), "none"),
        ((_BF16,), (_BF16,), "none"),
        ((_FP32,), (_FP32,), "none"),
    )
)

# K1280 cluster tail: auto admission over the measured M range
# (denoise_cluster_auto_admission.json, 2026-11-10).
_TABLE.append(dict(
    backend="ascend_cluster_tail", arch=950, k=(1280,), a=[[1, 32]],
    m=[[1, 32]], batch=[[1, 1]], h_dtype=(_BF16,), w_dtype=(_FP32,),
    prenorm=("adarms",), bias=(False,), row_mask=(False,), dof=_OPTIONAL,
    pinned=(False,), dt=("scalar", "tensor"), heads=(1, 2),
    ada_table=_OPTIONAL, round_normed=(True,), velocity=("flow", "x_pred"),
    padding_optional=True, priority=10,
    scan_ref="Ascend950 2026-11-10 S0 cluster tail vs complete chains; every sampled M in "
    "[1,32] wins 1.5-2.9x, M 64/80 lose to cann_rms_wide_mm by 7-29% "
    "(npu/benchmarks/results/denoise_cluster_auto_admission.json)",
))

_BACKENDS = ("fused_tail", "cluster_tail", "ascend_tilelang", "ascend_cluster_tail")
_scan_mode = False


def denoise_output_envelopes():
    """The white list of :func:`denoise_output_euler` (copies of the entries, highest priority first)."""
    return [dict(e) for e in sorted(_TABLE, key=lambda e: -e["priority"])]


def set_denoise_output_scan_mode(enabled: bool):
    """Scan mode (scan scripts only): a named backend ignores the A / M / B ranges of its entry."""
    global _scan_mode
    _scan_mode = bool(enabled)


def _in(ranges, v):
    return any(lo <= v and (hi is None or v <= hi) for lo, hi in ranges)


def _ranges_text(ranges):
    return ",".join(f"[{lo},{'inf' if hi is None else hi}]" for lo, hi in ranges)


def _dtype_name(t):
    return {torch.bfloat16: _BF16, torch.float32: _FP32}.get(t.dtype, str(t.dtype))


class _Request:
    def __init__(
        self,
        arch,
        b,
        m,
        a,
        k,
        h_dtype,
        w_dtype,
        prenorm,
        bias,
        row_mask,
        dof,
        pinned,
        dt,
        heads=1,
        ada_table=False,
        round_normed=False,
        velocity="flow",
        pad=False,
    ):
        self.heads, self.ada_table, self.round_normed, self.velocity, self.pad = (
            heads,
            ada_table,
            round_normed,
            velocity,
            pad,
        )
        self.arch, self.b, self.m, self.a, self.k = arch, b, m, a, k
        self.h_dtype, self.w_dtype, self.prenorm = h_dtype, w_dtype, prenorm
        self.bias, self.row_mask, self.dof, self.pinned, self.dt = (
            bias,
            row_mask,
            dof,
            pinned,
            dt,
        )

    def describe(self):
        return (
            f"denoise_output_euler(SM{self.arch}, B={self.b}, M={self.m}, A={self.a}, K={self.k}, h={self.h_dtype}, "
            f"weight={self.w_dtype}, prenorm={self.prenorm}, bias={self.bias}, row_mask={self.row_mask}, "
            f"dof={self.dof}, pinned={self.pinned}, dt={self.dt}, heads={self.heads}, ada_table={self.ada_table}, "
            f"round_normed={self.round_normed}, velocity={self.velocity})"
        )


def _mismatch(e, r, ignore_ranges=False):
    if e["arch"] != r.arch:
        return f"needs SM{e['arch']}, device is SM{r.arch}"
    if r.k not in e["k"]:
        return f"K={r.k} (entry: {'/'.join(map(str, e['k']))})"
    if r.w_dtype not in e["w_dtype"]:
        return f"weight {r.w_dtype} (entry: {'/'.join(e['w_dtype'])})"
    if r.h_dtype not in e["h_dtype"]:
        return f"h {r.h_dtype} (entry: {'/'.join(e['h_dtype'])})"
    if r.prenorm not in e["prenorm"]:
        return f"prenorm={r.prenorm} (entry: {'/'.join(e['prenorm'])})"
    for f in ("bias", "row_mask", "dof", "pinned"):
        if getattr(r, f) not in e[f]:
            return f"{f} {'given' if getattr(r, f) else 'missing'} (entry: {e[f]})"
    if r.dt not in e["dt"]:
        return f"dt {r.dt} (entry: {'/'.join(e['dt'])})"
    if r.heads not in e.get("heads", (1,)):
        return f"heads={r.heads} (entry: {e.get('heads', (1,))})"
    if r.ada_table not in e.get("ada_table", (False,)):
        return f"ada_index {'given' if r.ada_table else 'missing'} (entry: {e.get('ada_table', (False,))})"
    if r.round_normed not in e.get("round_normed", (False,)):
        return (
            f"round_normed={r.round_normed} (entry: {e.get('round_normed', (False,))})"
        )
    if r.velocity not in e.get("velocity", ("flow",)):
        return (
            f"velocity={r.velocity} (entry: {'/'.join(e.get('velocity', ('flow',)))})"
        )
    if r.dof and not r.pad and not e.get("padding_optional", False):
        return "dof_mask needs padding_velocity (entry)"
    if ignore_ranges:
        return ""
    if not _in(e["a"], r.a):
        return f"A={r.a} outside {_ranges_text(e['a'])}"
    if not _in(e["batch"], r.b):
        return f"B={r.b} outside {_ranges_text(e['batch'])}"
    if not _in(e["m"], r.m):
        return f"M={r.m} outside {_ranges_text(e['m'])}"
    return ""


def _select(r, backend):
    if r.arch == 950 and backend == "cluster_tail":
        backend = "ascend_cluster_tail"
    if r.arch == 950 and backend == "fused_tail":
        backend = "ascend_tilelang"
    if backend != "auto" and backend not in _BACKENDS:
        raise ValueError(
            f"denoise_output_euler: unknown backend '{backend}'; known: auto {' '.join(_BACKENDS)}"
        )
    reasons = []
    for e in sorted(_TABLE, key=lambda e: -e["priority"]):
        if backend == "auto" and e["priority"] < 0:
            continue
        if backend not in ("auto", e["backend"]):
            continue
        why = _mismatch(e, r, ignore_ranges=backend == e["backend"] and _scan_mode)
        if not why:
            return e
        reasons.append((e, why))
    order = [
        "needs SM",
        "K=",
        "weight",
        "h ",
        "prenorm",
        "bias",
        "row_mask",
        "dof",
        "pinned",
        "dt",
        "heads=",
        "ada_index",
        "round_normed",
        "velocity",
        "dof_mask needs",
        "A=",
        "B=",
        "M=",
    ]

    def rank(item):
        return max(
            (i for i, k in enumerate(order) if item[1].startswith(k)), default=-1
        )

    detail = ""
    if reasons:
        e, why = max(reasons, key=rank)
        detail = (
            f"; closest entry {e['backend']} (SM{e['arch']}, weight {'/'.join(e['w_dtype'])}, A {_ranges_text(e['a'])}, "
            f"M {_ranges_text(e['m'])}, B {_ranges_text(e['batch'])}): {why}"
        )
    raise ValueError(
        f"denoise_output_euler: no {'white-list entry' if backend == 'auto' else repr(backend)} accepts "
        f"{r.describe()}{detail}. Shapes outside the white list run as torch operations in the caller"
    )


def _arch(device):
    if device.type == "npu":
        return 950
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def _as_u8(mask, name):
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask.view(torch.uint8)
    if mask.dtype == torch.uint8:
        return mask
    raise ValueError(f"denoise_output_euler: {name} must be bool or uint8")


def _resolve(
    x,
    h,
    weight,
    dt,
    bias,
    prenorm,
    norm_weight,
    ada_scale,
    ada_shift,
    row_mask,
    dof_mask,
    padding_velocity,
    pinned,
    pinned_values,
    head_index=None,
    ada_index=None,
    round_normed=False,
    velocity="flow",
    tau=None,
):
    if h.dim() != 3 or weight.dim() not in (2, 3):
        raise ValueError(
            "denoise_output_euler: h must be [B, M, K] and weight [A, K] (or [heads, A, K])"
        )
    b, m, k = h.shape
    heads = 1 if weight.dim() == 2 else weight.shape[0]
    a = weight.shape[-2]
    if heads > 1 and head_index is None:
        raise ValueError(
            "denoise_output_euler: a [heads, A, K] weight needs head_index"
        )
    if ada_index is not None and (ada_scale is None or ada_scale.dim() != 2):
        raise ValueError(
            "denoise_output_euler: ada_index needs [C, K] ada_scale / ada_shift tables"
        )
    if velocity not in ("flow", "x_pred"):
        raise ValueError(f"denoise_output_euler: unknown velocity '{velocity}'")
    if (velocity == "x_pred") != (tau is not None):
        raise ValueError(
            "denoise_output_euler: velocity='x_pred' goes with tau (and only it)"
        )
    if x.dim() != 3 or tuple(x.shape) != (b, m, a):
        raise ValueError(
            f"denoise_output_euler: x must be [{b}, {m}, {a}], got {tuple(x.shape)}"
        )
    if prenorm not in _PRENORM:
        raise ValueError(f"denoise_output_euler: unknown prenorm '{prenorm}'")
    if prenorm == "adarms" and (ada_scale is None or ada_shift is None):
        raise ValueError("denoise_output_euler: adarms needs ada_scale and ada_shift")
    if prenorm == "rmsnorm" and norm_weight is None:
        raise ValueError("denoise_output_euler: rmsnorm needs norm_weight")
    if prenorm != "adarms" and (ada_scale is not None or ada_shift is not None):
        raise ValueError(
            "denoise_output_euler: ada_scale / ada_shift need prenorm='adarms'"
        )
    if prenorm != "rmsnorm" and norm_weight is not None:
        raise ValueError("denoise_output_euler: norm_weight needs prenorm='rmsnorm'")
    if padding_velocity is not None and dof_mask is None:
        raise ValueError("denoise_output_euler: padding_velocity needs dof_mask")
    if pinned_values is not None and pinned is None:
        raise ValueError("denoise_output_euler: pinned_values needs pinned")
    dt_kind = "tensor" if isinstance(dt, torch.Tensor) else "scalar"
    return _Request(
        _arch(h.device),
        b,
        m,
        a,
        k,
        _dtype_name(h),
        _dtype_name(weight),
        prenorm,
        bias is not None,
        row_mask is not None,
        dof_mask is not None,
        pinned is not None,
        dt_kind,
        heads,
        ada_index is not None,
        bool(round_normed),
        velocity,
        padding_velocity is not None,
    )


def denoise_output_euler_backend(
    x,
    h,
    weight,
    *,
    dt,
    bias=None,
    prenorm="none",
    norm_weight=None,
    ada_scale=None,
    ada_shift=None,
    row_mask=None,
    dof_mask=None,
    padding_velocity=None,
    pinned=None,
    pinned_values=None,
    head_index=None,
    ada_index=None,
    round_normed=False,
    velocity="flow",
    tau=None,
    backend="auto",
    **_ignored,
):
    """Name of the backend :func:`denoise_output_euler` would use (raises like it when none accepts)."""
    r = _resolve(
        x,
        h,
        weight,
        dt,
        bias,
        prenorm,
        norm_weight,
        ada_scale,
        ada_shift,
        row_mask,
        dof_mask,
        padding_velocity,
        pinned,
        pinned_values,
        head_index,
        ada_index,
        round_normed,
        velocity,
        tau,
    )
    return _select(r, backend)["backend"]


def _i32(t, name, device):
    if t is None:
        return None
    if t.dtype != torch.int32 or not t.is_contiguous() or t.device != device:
        raise ValueError(
            f"denoise_output_euler: {name} must be a contiguous int32 tensor on {device}"
        )
    return t


def denoise_output_euler(
    x: torch.Tensor,
    h: torch.Tensor,
    weight: torch.Tensor,
    *,
    dt: Union[float, torch.Tensor],
    bias: Optional[torch.Tensor] = None,
    prenorm: str = "none",
    norm_weight: Optional[torch.Tensor] = None,
    ada_scale: Optional[torch.Tensor] = None,
    ada_shift: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    row_mask: Optional[torch.Tensor] = None,
    dof_mask: Optional[torch.Tensor] = None,
    padding_velocity: Optional[torch.Tensor] = None,
    pinned: Optional[torch.Tensor] = None,
    pinned_values: Optional[torch.Tensor] = None,
    head_index: Optional[torch.Tensor] = None,
    ada_index: Optional[torch.Tensor] = None,
    round_normed: bool = False,
    velocity: str = "flow",
    tau: Optional[Union[float, torch.Tensor]] = None,
    tau_floor: float = 0.05,
    out: Optional[torch.Tensor] = None,
    pdl: bool = False,
    backend: str = "auto",
):
    """One denoising step's tail: ``x_next = x + dt * blend(velocity(weight . prenorm(h) + bias))`` with pinned
    rows kept.

    x [B, M, A] FP32 (updated in place unless ``out`` is given); h [B, M, K] BF16 / FP32; weight [A, K] or
    [heads, A, K] (``head_index`` int32 [B, M] picks a row's head) BF16 / FP32; ``dt`` a float or an FP32 tensor
    [B]. ``ada_scale`` / ``ada_shift`` may be [C, K] tables with ``ada_index`` int32 [B, M]; ``round_normed`` rounds
    the normalized row to BF16 before the projection; ``velocity="x_pred"`` uses
    ``v = (p - x) / max(1 - tau, tau_floor)``. See docs/denoise_output/README.md for every option.
    """
    r = _resolve(
        x,
        h,
        weight,
        dt,
        bias,
        prenorm,
        norm_weight,
        ada_scale,
        ada_shift,
        row_mask,
        dof_mask,
        padding_velocity,
        pinned,
        pinned_values,
        head_index,
        ada_index,
        round_normed,
        velocity,
        tau,
    )
    e = _select(r, backend)
    if out is None:
        out = x
    if dof_mask is not None and dof_mask.dtype != torch.float32:
        dof_mask = _as_u8(dof_mask, "dof_mask")
    if e["backend"] in ("cluster_tail", "ascend_cluster_tail"):
        w3 = weight if weight.dim() == 3 else weight.unsqueeze(0)
        sc = ada_scale if ada_scale.dim() == 2 else ada_scale.unsqueeze(0)
        sh = ada_shift if ada_shift.dim() == 2 else ada_shift.unsqueeze(0)
        CORE.denoise_output_euler_cluster_tail(
            out,
            x,
            h,
            w3,
            _i32(head_index, "head_index", x.device),
            sc,
            sh,
            _i32(ada_index, "ada_index", x.device),
            float(eps),
            dof_mask,
            padding_velocity,
            0.0 if isinstance(dt, torch.Tensor) else float(dt),
            dt if isinstance(dt, torch.Tensor) else None,
            velocity == "x_pred",
            0.0 if tau is None or isinstance(tau, torch.Tensor) else float(tau),
            tau if isinstance(tau, torch.Tensor) else None,
            float(tau_floor),
            bool(pdl),
        )
        return out
    CORE.denoise_output_euler(
        out,
        x,
        h,
        weight,
        bias,
        _PRENORM[prenorm],
        ada_scale if prenorm == "adarms" else norm_weight,
        ada_shift,
        float(eps),
        _as_u8(row_mask, "row_mask"),
        dof_mask,
        padding_velocity,
        pinned,
        pinned_values,
        0.0 if isinstance(dt, torch.Tensor) else float(dt),
        dt if isinstance(dt, torch.Tensor) else None,
        bool(pdl),
    )
    return out


# ---------------------------------------------------------------------------
# Step input embedding (the start of the next denoising step)
# ---------------------------------------------------------------------------
# Same white-list rules. Fields: backend; arch; n (output width), a (action width): inclusive ranges; m, batch:
# inclusive ranges; extra (accepted presence of the second input block); priority; scan_ref.
_EMBED_TABLE = [
    dict(
        backend="feature_tiles",
        arch=120,
        n=[[16, 4096]],
        a=[[1, 32]],
        m=[[1, 80]],
        batch=[[1, 1]],
        extra=_OPTIONAL,
        priority=10,
        scan_ref="RTX 5090 2026-11-06 S0 do_embed (N 1280, A 26)",
    ),
    dict(
        backend="ascend_tilelang",
        arch=950,
        n=[[16, 4096]],
        a=[[1, 32]],
        m=[[1, 80]],
        batch=[[1, 1]],
        extra=_OPTIONAL,
        priority=-1,
        scan_ref="Ascend950 named correctness candidate; NPU auto performance admission pending",
    ),
]


def denoise_input_embed_envelopes():
    """The white list of :func:`denoise_input_embed` (copies of the entries, highest priority first)."""
    return [dict(e) for e in sorted(_EMBED_TABLE, key=lambda e: -e["priority"])]


def _embed_select(arch, b, m, a, n, extra, backend):
    reasons = []
    for e in sorted(_EMBED_TABLE, key=lambda e: -e["priority"]):
        if backend == "auto" and e["priority"] < 0:
            continue
        if backend not in ("auto", e["backend"]):
            continue
        if e["arch"] != arch:
            target = "Ascend950DT" if e["arch"] == 950 else f"SM{e['arch']}"
            why = f"needs {target}"
        elif extra not in e["extra"]:
            why = f"extra {'given' if extra else 'missing'}"
        elif n % 16 or not _in(e["n"], n):
            why = f"N={n} (entry: multiple of 16 in {_ranges_text(e['n'])})"
        elif not _in(e["a"], a):
            why = f"A={a} outside {_ranges_text(e['a'])}"
        elif not _in(e["batch"], b):
            why = f"B={b} outside {_ranges_text(e['batch'])}"
        elif not (_in(e["m"], m) or (_scan_mode and backend == e["backend"])):
            why = f"M={m} outside {_ranges_text(e['m'])}"
        else:
            return e
        reasons.append((e, why))
    detail = (
        f"; closest entry {reasons[-1][0]['backend']}: {reasons[-1][1]}"
        if reasons
        else ""
    )
    device = "Ascend950DT" if arch == 950 else f"SM{arch}"
    raise ValueError(
        f"denoise_input_embed: no {'white-list entry' if backend == 'auto' else repr(backend)} accepts "
        f"{device} B={b} M={m} A={a} N={n} extra={extra}{detail}. Shapes outside the white list run as torch "
        "operations in the caller"
    )


def denoise_input_embed(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    head_index: Optional[torch.Tensor] = None,
    extra: Optional[torch.Tensor] = None,
    input_mask: Optional[torch.Tensor] = None,
    addends=(),
    override: Optional[torch.Tensor] = None,
    override_mask: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    pdl: bool = False,
    backend: str = "auto",
):
    """``out[b, t] = BF16(override if override_mask else ((p + addends[0]) + addends[1]) + addends[2])``.

    ``p = [x | extra][b, t] . weight[head_index[b, t]]^T`` (FP32, inputs in order, zero where ``input_mask`` is 0);
    x / extra FP32 [B, M, A]; weight FP32 [heads, N, A] ([heads, N, 2A] with ``extra``); head_index int32 [B, M]
    (required with two heads); input_mask / override_mask bool or uint8 [B, M]; addends: up to three FP32
    [B, M, N] tensors added in order; override FP32 [B, M, N]. Returns BF16 [B, M, N] (``out`` if given).
    See docs/denoise_output/README.md.
    """
    if x.dim() != 3 or weight.dim() != 3:
        raise ValueError(
            "denoise_input_embed: x must be [B, M, A] and weight [heads, N, A]"
        )
    b, m, a = x.shape
    n = weight.shape[1]
    addends = tuple(addends)
    if len(addends) > 3:
        raise ValueError("denoise_input_embed: at most three addends")
    if weight.shape[0] > 1 and head_index is None:
        raise ValueError("denoise_input_embed: a two-head weight needs head_index")
    _embed_select(_arch(x.device), b, m, a, n, extra is not None, backend)
    if out is None:
        out = torch.empty(b, m, n, dtype=torch.bfloat16, device=x.device)
    if head_index is not None and (
        head_index.dtype != torch.int32 or not head_index.is_contiguous()
    ):
        raise ValueError(
            "denoise_input_embed: head_index must be a contiguous int32 tensor"
        )
    adds = list(addends) + [None] * (3 - len(addends))
    CORE.denoise_input_embed(
        out,
        x,
        extra,
        weight,
        head_index,
        _as_u8(input_mask, "input_mask"),
        adds[0],
        adds[1],
        adds[2],
        override,
        _as_u8(override_mask, "override_mask"),
        bool(pdl),
    )
    return out
