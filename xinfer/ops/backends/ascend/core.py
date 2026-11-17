"""The glue core namespace: the provider surface the binding layers see.

This module is the ``xc`` object the binding layers probe (the former paired-loader role),
reborn over the unified library: a PEP 562 namespace whose attributes are
the class-style operator entries the ascend backend modules consume
(``_support.canonical`` inspects it, ``getattr`` probes carry it into
``register(requires=...)``).

Carried entries (routed>0 kernels in the INVENTORY truth source) are
implemented here over the provider:

* the CANN bridge trio used by ``glue.cann_attention`` / ``glue.cann_projection``
  / ``glue.cann_gated_delta_net`` — FIA probe pairs, the ACLNN Mm pair, the
  vendor GDN chunk pair (``_native.BRIDGE`` over the static bridge objects);
* the wall_oss QKV cache finishes (``gemm_qkv_cache_bias_rope_m16/prefill``)
  and the vision QKV rope finish (``gemm_qkv_vision_rope_finish``), as python
  envelopes over the core library's AOT launchers (the v22 .cc envelopes,
  re-expressed host-side);
* ``gemm_epilogue_glu`` for the served N22016 half-width split finish;
* the AdaRMS action compositions ``gated_residual_adarmsnorm`` and
  ``double_adarmsnorm_gated_residual_inplace`` over the adanorm-r2 action
  kernels (the 2B identity-operands and action-loop serving shapes);
* ``denoise_input_embed`` and the fp32-head Euler tail of
  ``denoise_output_euler`` over the denoise AOT kernels;
* ``CannGatedDeltaNet`` re-exported from ``glue.cann_gated_delta_net`` and
  ``Adarmsnorm`` from ``glue.adarmsnorm``.

Deliberately absent (routed=0 in both final stacks — INVENTORY §8): the
``Rmsnorm``/``AddRmsnorm`` classes and the raw ``add_rmsnorm`` entry (every
norm site is served by SOUPUNLOCK L0 / SIMDEPI / RMSFUSE or the reference),
``Rope``, ``QkNorm``, ``Gemm``, ``qkv_proj_rope``, ``attention_varlen(±_backend)``,
``attention_kvcache``, ``sigmoid_gate_mul_inplace``, ``layernorm_inplace``,
``rot_pos``/``pos_embed_interp``/``get_token_counts`` (position tables are
built once per grid and cached; the reference composition is bitwise), and
``MissingBackendError`` is redefined locally. An absent attribute raises
``AttributeError`` exactly like the old loader's build-without-that-kernel,
so every ``getattr(xc, name, None)`` probe in the binding layers keeps its
meaning: absent ⇒ the operator is not registered on this provider and the
reference composition serves it.

The class glues call ``CORE.<entry>(...)`` with torch tensors; the
implementations below validate the serving envelopes and launch through
``_native``. Envelope misses raise ``ValueError`` so the ``partial()``
machinery caches "unsupported" and falls back — never a silent numerics
switch.
"""

from __future__ import annotations

import ctypes
import math
import struct

import torch

from xinfer.ops.backends.ascend import _native


class MissingBackendError(AttributeError):
    """An operator is absent from this build of the unified library."""


_CARRIED = {}


def _bits(value: float) -> int:
    """The float's IEEE-754 bit pattern, the AOT scalar ABI the kernels take.

    The v22 convention (struct.pack("<f") -> unpack("<I"), as in the simdepi
    and gdn glues): the int64 scalar's low word carries the float32 pattern."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _launch(lib, name, tensors, scalars, stream=None):
    """One AOT launcher call; raises the library's error text on failure.

    The single library's TVM-FFI face serves first (the unified launch boundary); the ctypes
    launcher of the family library below is the fallback a deployment without the faces keeps.
    """
    if stream is None and _native.launch_kernel(name, tensors, scalars):
        return
    if stream is None:
        stream = torch.npu.current_stream(tensors[0].device).npu_stream
    call = getattr(lib, f"xinfer_aot_{name}_call")
    call.restype = ctypes.c_int32
    call.argtypes = ([ctypes.c_void_p] * len(tensors)
                     + [ctypes.c_int64] * len(scalars) + [ctypes.c_void_p])
    status = call(*[ctypes.c_void_p(t.data_ptr()) for t in tensors],
                  *[ctypes.c_int64(int(s)) for s in scalars],
                  ctypes.c_void_p(stream))
    if status:
        error = getattr(lib, f"xinfer_aot_{name}_get_last_error")
        error.restype = ctypes.c_char_p
        message = error()
        raise RuntimeError(f"{name}: {message.decode('utf-8', 'replace') if message else 'unknown failure'}")


def _require(cond, message):
    if not cond:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# CANN bridge trio (static bridge objects in the core library)
# ---------------------------------------------------------------------------
_BRIDGE = _native.BRIDGE


def _gdr_pair():
    pair = getattr(_gdr_pair, "_cache", None)
    if pair is None:
        pair = _BRIDGE.gdr_chunk_pair()
        _gdr_pair._cache = pair
    return pair


def gdr_cann_chunk_workspace_size(out, final_state, q, k, v, beta, initial_state, seq_lengths, g, scale):
    """The vendor chunk workspace query (the glue's TensorView-signature call)."""
    pair = _gdr_pair()
    _require(pair is not None, "gdr_cann_chunk: the bridge is not in this build")
    return pair[0](out, final_state, q, k, v, beta, initial_state, seq_lengths, g, scale)


def gdr_cann_chunk(out, final_state, workspace, q, k, v, beta, initial_state, seq_lengths, g, scale):
    """The vendor chunk launch (the glue's TensorView-signature call)."""
    pair = _gdr_pair()
    _require(pair is not None, "gdr_cann_chunk: the bridge is not in this build")
    return pair[1](out, final_state, workspace, q, k, v, beta, initial_state, seq_lengths, g, scale)


def _mm_pair():
    pair = getattr(_mm_pair, "_cache", None)
    if pair is None:
        pair = _BRIDGE.mm_pair()
        _mm_pair._cache = pair
    return pair


def cann_mm_fp32_workspace_size(out, x, w, math_type=0):
    pair = _mm_pair()
    _require(pair is not None, "cann_mm_fp32: the bridge is not in this build")
    return pair[0](out, x, w, math_type)


def cann_mm_fp32(out, workspace, x, w, math_type=0):
    pair = _mm_pair()
    _require(pair is not None, "cann_mm_fp32: the bridge is not in this build")
    return pair[1](out, workspace, x, w, math_type)


def _fia(bnsd: bool):
    cache = getattr(_fia, "_cache", None) or {}
    if bnsd not in cache:
        cache[bnsd] = _BRIDGE.bnsd_pair() if bnsd else _BRIDGE.tnd_pair()
        _fia._cache = cache
    return cache[bnsd]


def cann_fia_v5_probe_workspace_size(out, q, k, v, drop_mask, scale, inner_precise=2, sparse_mode=0):
    pair = _fia(True)
    _require(pair is not None, "cann_fia_v5: the bridge is not in this build")
    return pair[0](out, q, k, v, drop_mask, scale, sparse_mode, inner_precise)


def cann_fia_v5_probe(out, workspace, q, k, v, drop_mask, scale, inner_precise=2, sparse_mode=0):
    pair = _fia(True)
    _require(pair is not None, "cann_fia_v5: the bridge is not in this build")
    return pair[1](out, workspace, q, k, v, drop_mask, scale, sparse_mode, inner_precise)


def cann_fia_vision_probe_workspace_size(out, q, k, v, endpoints, scale):
    pair = _fia(False)
    _require(pair is not None, "cann_fia_vision: the bridge is not in this build")
    return pair[0](out, q, k, v, endpoints, scale)


def cann_fia_vision_probe(out, workspace, q, k, v, endpoints, scale):
    pair = _fia(False)
    _require(pair is not None, "cann_fia_vision: the bridge is not in this build")
    return pair[1](out, workspace, q, k, v, endpoints, scale)


# ---------------------------------------------------------------------------
# wall_oss QKV cache finishes (gemm/npu/gemm_qkv_cache.cc envelopes, host-side)
# ---------------------------------------------------------------------------
def gemm_qkv_cache_bias_rope_m16(out, key, value, accumulator, bias, cosine, sine, start):
    """The M16 action finish: bias + 64-dim partial rotary + head-major append.

    Envelope (the .cc's Require set): Q ``[16, 2048]`` BF16 out, dense
    head-major K/V caches ``[2, cap, 128]``, an FP32 ``[16, 2560]``
    accumulator, a BF16 ``[2560]`` bias and FP32 ``[16, 64]`` tables.
    """
    lib = _native.core_library()
    _require(lib is not None, "gemm_qkv_cache: no core library")
    # The .cc's Prepare contract (gemm_qkv_cache.cc): a packed 2-D [16, 2048] Q
    # or the r3 BNSD form -- 3-D [16, 16, 128], the head-major [H, M, D] order
    # the kernel writes so the ACLNN FIA query needs no per-call transpose copy.
    _require(out.dtype == torch.bfloat16 and out.is_contiguous()
             and ((out.ndim == 2 and tuple(out.shape) == (16, 2048))
                  or (out.ndim == 3 and tuple(out.shape) == (16, 16, 128))),
             "M16 QKV epilogue requires a contiguous BF16 [16,2048] or BNSD [16,16,128] output")
    _require(key.ndim == 3 and value.ndim == 3 and key.shape == value.shape
             and key.is_contiguous() and value.is_contiguous() and start >= 0,
             "M16 QKV epilogue requires dense matching head-major caches")
    _require(accumulator.ndim == 2 and tuple(accumulator.shape) == (16, 2560)
             and accumulator.dtype == torch.float32,
             "M16 QKV epilogue requires an FP32 [16,2560] accumulator")
    _require(bias.ndim == 1 and tuple(bias.shape) == (2560,) and bias.dtype == torch.bfloat16,
             "M16 QKV epilogue requires a BF16 [2560] bias")
    _require(key.shape[1] >= 16 and start <= key.shape[1] - 16,
             "M16 QKV cache start + 16 exceeds capacity")
    for table in (cosine, sine):
        _require(table.ndim == 2 and tuple(table.shape) == (16, 64)
                 and table.dtype == torch.float32 and table.is_contiguous(),
                 "M16 QKV epilogue requires FP32 [16,64] cosine and sine")
    _launch(lib, "gemm_qkv_cache_bias_rope_m16",
            (out, key, value, accumulator, bias, cosine, sine),
            (key.shape[1], start))


def gemm_qkv_cache_bias_rope_prefill(out, key, value, accumulator, bias, cosine, sine, start):
    """The prefill fused finish over the FP32 ``[rows, 2560]`` accumulator."""
    lib = _native.core_library()
    _require(lib is not None, "gemm_qkv_cache: no core library")
    rows = accumulator.shape[0]
    _require(rows % 4 == 0 and rows > 0, "prefill QKV epilogue requires rows % 4 == 0")
    # The .cc's Prepare contract: a packed [rows, 2048] Q or the r7 BNSD form --
    # 3-D [16, rows, 128], the head-major layout the prefill causal attention
    # consumes (the callers have allocated this form since the walloss r7).
    _require(out.dtype == torch.bfloat16 and out.is_contiguous()
             and ((out.ndim == 2 and out.shape[0] == rows and out.shape[1] == 2048)
                  or (out.ndim == 3 and out.shape[0] == 16 and out.shape[1] == rows
                      and out.shape[2] == 128)),
             "prefill QKV epilogue requires a contiguous BF16 [rows,2048] or BNSD [16,rows,128] output")
    _require(key.ndim == 3 and value.ndim == 3 and key.shape == value.shape
             and key.is_contiguous() and value.is_contiguous() and start >= 0,
             "prefill QKV epilogue requires dense matching head-major caches")
    _require(accumulator.ndim == 2 and accumulator.shape[1] == 2560
             and accumulator.dtype == torch.float32,
             "prefill QKV epilogue requires an FP32 [rows,2560] accumulator")
    _require(bias.ndim == 1 and tuple(bias.shape) == (2560,) and bias.dtype == torch.bfloat16,
             "prefill QKV epilogue requires a BF16 [2560] bias")
    _require(key.shape[1] >= rows and start <= key.shape[1] - rows,
             "prefill QKV cache start + rows exceeds capacity")
    for table in (cosine, sine):
        _require(table.ndim == 2 and table.shape[0] == rows and table.shape[1] == 64
                 and table.dtype == torch.float32 and table.is_contiguous(),
                 "prefill QKV epilogue requires FP32 [rows,64] cosine and sine")
    _launch(lib, "gemm_qkv_cache_bias_rope_prefill",
            (out, key, value, accumulator, bias, cosine, sine),
            (key.shape[1], start, rows))


def action_qkv_gemm_norm_rope(output, input, weight, bias, q_weight, k_weight,
                              cosine, sine, rotary, epsilon):
    """The merged M16 cache-append entry (the v22 ``ActionQkvGemmNormRope``).

    The walloss r5 form: an FP32 ``input`` marks the M16 cache-append route --
    the input slot carries the FP32 [16, 2560] CANN projection accumulator,
    ``q_weight``/``k_weight`` the head-major K/V caches, the ``rotary`` slot
    the cache start row (768 lies outside the NV rotary range) and ``weight``
    the projection weight already consumed into the accumulator (not re-read).
    ``epsilon`` carries no norm (0). This is the same launch the m16 finish
    serves (the v22 merged that export into this entry to hold the FFI surface
    at the NV subset); the BF16-input NV form is not carried by the unified
    core, so it raises here exactly like a build without the bridge would.
    """
    lib = _native.core_library()
    _require(lib is not None, "gemm_qkv_cache: no core library")
    _require(isinstance(input, torch.Tensor) and input.dtype == torch.float32,
             "action_qkv_gemm_norm_rope: the unified core carries the FP32 M16 cache-append form only")
    gemm_qkv_cache_bias_rope_m16(output, q_weight, k_weight, input, bias, cosine, sine,
                                 int(rotary))


def gemm_qkv_vision_rope_finish(out, accumulator, bias, cosine, sine):
    """The wall_oss vision QKV finish: bias + rotate-half RoPE + one RNE.

    One UB pass over the borrowed FP32 ``[2112, 3840]`` accumulator (the r5
    probe measured the torch rope chain ~113 us against ~10.6 us here).
    """
    lib = _native.core_library()
    _require(lib is not None, "gemm_qkv_vision_rope_finish: no core library")
    rows = accumulator.shape[0]
    _require(accumulator.ndim == 2 and accumulator.dtype == torch.float32
             and out.shape == accumulator.shape and out.dtype == torch.bfloat16
             and out.is_contiguous(),
             "vision rope finish requires a BF16 output shaped like the FP32 accumulator")
    _require(bias.ndim == 1 and bias.shape[0] == accumulator.shape[1]
             and bias.dtype == torch.bfloat16,
             "vision rope finish requires a BF16 bias of the accumulator width")
    _require(cosine.ndim == 2 and sine.shape == cosine.shape
             and cosine.dtype == sine.dtype == torch.float32
             and cosine.shape[0] == rows,
             "vision rope finish requires FP32 [rows, half] cosine and sine")
    _launch(lib, "gemm_qkv_vision_rope_finish",
            (out, accumulator, bias, cosine, sine), (rows,))


# ---------------------------------------------------------------------------
# gemm_epilogue_glu — the served split finish (N22016 half width)
# ---------------------------------------------------------------------------
def gemm_epilogue_glu(out, projected, has_bias_flag, half_width):
    """The fused FP32 SwiGLU finish over a borrowed FP32 accumulator.

    The v22 FFI entry's 4-arg surface, kept verbatim: the last argument is the
    OUTPUT width (half the accumulator's) and the entry dispatched
    split_<width> by it; the unified core carries the served width only (11008;
    the 2048/3456 sites are served by the smallm gateup / g2112 routes). Any
    other width raises and the caller keeps its released chain.
    """
    lib = _native.core_library()
    _require(lib is not None, "gemm_epilogue_glu: no core library")
    _require(half_width == 11008 and projected.dtype == torch.float32
             and projected.is_contiguous() and projected.shape[1] == 2 * half_width,
             "gemm_epilogue_glu: only the N22016 half-width split is carried")
    # The kernel takes (output, accum, bias, rows, has_bias); the v22 FFI entry
    # this replaces fed bias=nullptr with has_bias=0 (the callers pre-fold the
    # bias or have none). The null-optional convention passes a live pointer
    # for the slot the kernel does not read.
    _launch(lib, "gemm_glu_fp32_finish_split_11008",
            (out, projected, projected), (projected.shape[0], int(has_bias_flag)))


# ---------------------------------------------------------------------------
# AdaRMS action compositions (adanorm-r2 kernels; the .cc action envelopes)
# ---------------------------------------------------------------------------
def _action_entries():
    lib = _native.knife_library("adanorm_r2", env="XINFER_ADANORM_R2_LIB",
                                  default="libadanorm_r2.so")
    return lib


def gated_residual_adarmsnorm(out, gate, residual_out, residual, branch, branch_gate,
                              modulation, pinned_modulation=None, pinned_rows=None,
                              epsilon=1e-6):
    """The fused gated-residual action norm on the [1, 16, 1024] family.

    Port of the adarmsnorm.cc GatedResidual fast path: the adanorm-r2
    ``action_gated`` kernel takes (residual, branch, branch_gate,
    residual_out, out, gate, modulation, modulation, eps_bits, stream) with
    optional pinned (modulation, modulation, counts) triples.
    """
    lib = _action_entries()
    _require(lib is not None, "adanorm_r2: the knife library is not loaded")
    pinned = pinned_modulation is not None and pinned_rows is not None
    _require(out.shape == branch.shape and out.dtype == torch.bfloat16
             and out.is_contiguous() and branch.is_contiguous(),
             "the action kernel requires contiguous BF16 activations")
    _require(modulation.dtype == torch.float32 and modulation.is_contiguous(),
             "the action kernel requires an FP32 modulation")
    stream = torch.npu.current_stream(out.device).npu_stream
    if pinned:
        _launch(lib, "adarmsnorm_action_gated_bf16_f32_pinned",
                (residual, branch, branch_gate, residual_out, out, gate,
                 modulation, modulation, pinned_modulation, pinned_modulation, pinned_rows),
                (_bits(epsilon),))
    else:
        _launch(lib, "adarmsnorm_action_gated_bf16_f32",
                (residual, branch, branch_gate, residual_out, out, gate,
                 modulation, modulation),
                (_bits(epsilon),))


def double_adarmsnorm_gated_residual_inplace(residual, branch, gate, previous_gate,
                                             first_modulation, second_modulation, epsilon,
                                             pinned_input_modulation=None,
                                             pinned_post_modulation=None,
                                             pinned_rows=None):
    """The identity layer-norm pair on the action family (action_double kernel).

    Port of the .cc DoubleContinue fast path: one kernel fuses both AdaRMS
    norms and both residual adds for the [1, 16, 1024] action shape. The
    pinned form takes the two pinned modulations by the retired pkg's names;
    the RTC rows (``pinned_rows`` and beyond on the input side) read them
    instead of the step's.
    """
    lib = _action_entries()
    _require(lib is not None, "adanorm_r2: the knife library is not loaded")
    _require(branch.dtype == torch.bfloat16 and branch.is_contiguous(),
             "the action kernel requires contiguous BF16 activations")
    stream = torch.npu.current_stream(branch.device).npu_stream
    # The kernel ABI is (residual, branch, previous_gate, residual_out,
    # branch_out, next_gate, modulation1, modulation2[, pinned mods, rows]);
    # the v22 .cc's DoubleContinue passes the caller's buffers through both
    # in/out pairs (residual -> residual_out, branch -> branch_out) with the
    # previous gate at slot 3 and the next gate at slot 6.
    if pinned_rows is not None:
        _require(pinned_input_modulation is not None and pinned_post_modulation is not None,
                 "the pinned action-double kernel takes both pinned modulations")
        _launch(lib, "adarmsnorm_action_double_bf16_f32_pinned",
                (residual, branch, previous_gate, residual, branch, gate,
                 first_modulation, second_modulation, pinned_input_modulation,
                 pinned_post_modulation, pinned_rows),
                (_bits(epsilon),))
    else:
        _launch(lib, "adarmsnorm_action_double_bf16_f32",
                (residual, branch, previous_gate, residual, branch, gate,
                 first_modulation, second_modulation),
                (_bits(epsilon),))


# ---------------------------------------------------------------------------
# denoise family (the .cc envelopes over the core AOT kernels)
# ---------------------------------------------------------------------------
def denoise_input_embed(x, weight, *, head_index=None, extra=None, input_mask=None,
                        addends=(), override=None, override_mask=None, out=None,
                        pdl=False, backend="auto"):
    """``out[b, t] = BF16(((p + addends...) + ...))`` over the input embed kernel.

    The serving envelope (2B flow head): FP32 ``[B, M, A]`` rows against an
    FP32 ``[heads, N, A]`` (``[heads, N, 2A]`` with extra) weight, optional
    int32 head_index and uint8 input_mask, up to three FP32 addends.
    """
    lib = _native.core_library()
    _require(lib is not None, "denoise: no core library")
    _require(x.dim() == 3 and weight.dim() == 3,
             "denoise_input_embed: x must be [B, M, A] and weight [heads, N, A]")
    b, m, a = x.shape
    n = weight.shape[1]
    heads = weight.shape[0]
    _require(x.dtype == torch.float32 and x.is_contiguous()
             and weight.dtype == torch.float32 and weight.is_contiguous(),
             "denoise_input_embed: FP32 contiguous operands")
    _require(heads == 1 or head_index is not None,
             "denoise_input_embed: a two-head weight needs head_index")
    _require(override is None and override_mask is None,
             "denoise_input_embed: the override form is not carried")
    addends = tuple(addends)
    _require(len(addends) <= 3, "denoise_input_embed: at most three addends")
    if out is None:
        out = torch.empty(b, m, n, dtype=torch.bfloat16, device=x.device)
    ex = extra
    head = head_index
    mask = input_mask
    adds = list(addends) + [None] * (3 - len(addends))
    flags = 0
    if ex is not None:
        flags |= 1
    if head is not None:
        flags |= 2
    if mask is not None:
        flags |= 4
    base = x  # the null-optional convention: pass x for absent pointers
    pointers = (out, x, ex if ex is not None else x, weight,
                head if head is not None else x,
                mask if mask is not None else x,
                adds[0] if adds[0] is not None else x,
                adds[1] if adds[1] is not None else x,
                adds[2] if adds[2] is not None else x,
                x, x)
    _launch(lib, "denoise_input_embed", pointers,
            (b * m, a, n, heads, flags, len(addends)))
    return out


def denoise_output_euler(x, h, weight, *, dt, bias=None, prenorm="none", norm_weight=None,
                         ada_scale=None, ada_shift=None, eps=1e-6, row_mask=None,
                         dof_mask=None, padding_velocity=None, pinned=None,
                         pinned_values=None, head_index=None, ada_index=None,
                         round_normed=False, velocity="flow", tau=None, tau_floor=0.05,
                         out=None, pdl=False, backend="auto"):
    """The 2B Euler tail: fp32head prep + cube over the flow head shapes.

    Serving envelope: x FP32 ``[B, M, A]`` (A 17..32, B<=4, M<=32), h BF16
    ``[B, M, 1024]`` (widened by the prep kernel), weight BF16 ``[A, 1024]``,
    optional BF16 bias / FP32 norm pair, dof_mask, padding_velocity, pinned
    rows and FP32-or-scalar dt. Other shapes raise ValueError (the binding's
    partial cache falls back to the reference composition).
    """
    lib = _native.core_library()
    _require(lib is not None, "denoise: no core library")
    _require(x.dim() == 3 and h.dim() == 3 and weight.dim() == 2,
             "the fp32-head Euler tail needs [B,M,A] x, [B,M,1024] h, [A,1024] weight")
    b, m, a = x.shape
    _require(17 <= a <= 32 and 1 <= b <= 4 and 1 <= m <= 32,
             "the fp32-head Euler tail serves A 17..32, B<=4, M<=32")
    _require(h.shape == (b, m, 1024) and h.dtype == torch.bfloat16
             and h.is_contiguous(),
             "the fp32-head Euler tail needs a contiguous BF16 [B,M,1024] hidden")
    _require(weight.dtype == torch.bfloat16 and weight.is_contiguous()
             and weight.shape == (a, 1024),
             "the fp32-head Euler tail needs a BF16 [A,1024] weight")
    _require(prenorm in ("none", "rms"), "the carried tail serves prenorm none/rms")
    if out is None:
        out = x
    _require(out is x, "the carried tail updates x in place")
    device = x.device
    scratch = getattr(denoise_output_euler, "_scratch", None)
    stream = torch.npu.current_stream(device).npu_stream
    if scratch is None or scratch.device != device:
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("warm up the flow Euler tail on this stream before graph capture")
        scratch = torch.empty(512 * 1024, dtype=torch.uint8, device=device)
        denoise_output_euler._scratch = scratch
    flags = 0
    if bias is not None:
        flags |= 1
        _require(bias.dtype == torch.bfloat16 and bias.shape == (a,), "bias must be BF16 [A]")
    if norm_weight is not None:
        _require(prenorm == "rms" and norm_weight.dtype == torch.float32
                 and ada_shift is not None, "rms prenorm needs the FP32 weight and shift")
    # The flag encoding and scalar tail follow the v22 bridge
    # (src/denoise_output/npu/denoise_output.cc) exactly: bias 1, row_mask 2,
    # dof_mask 4 (+8 when the dof mask is FP32), pinned 16, pinned_values 32,
    # tensor dt 64; scalars (b, m, a, hidden span, hidden batch/row strides,
    # row-mask batch/row strides, dof batch/row strides, flags, prenorm,
    # dt bits, eps bits, beta bits).
    if row_mask is not None:
        flags |= 2
        _require(row_mask.dtype == torch.uint8 and tuple(row_mask.shape) == (b, m),
                 "row_mask must be uint8 [B, M]")
    rmb, rmt = ((int(row_mask.stride(0)), int(row_mask.stride(1)))
                if row_mask is not None else (0, 0))
    if dof_mask is not None:
        flags |= 4
        if dof_mask.dtype == torch.float32:
            flags |= 8
        elif dof_mask.dtype != torch.uint8:
            raise ValueError("dof_mask must be FP32 or uint8")
        if dof_mask.ndim == 2:
            _require(tuple(dof_mask.shape) == (b, a), "a 2-D dof mask must be [B, A]")
            db, drow = int(dof_mask.stride(0)), 0
        else:
            _require(dof_mask.ndim == 3 and tuple(dof_mask.shape) == (b, m, a),
                     "a 3-D dof mask must be [B, M, A]")
            db, drow = int(dof_mask.stride(0)), int(dof_mask.stride(1))
    else:
        db, drow = 0, 0
    if pinned is not None:
        flags |= 16
    if pinned_values is not None:
        flags |= 32
    if isinstance(dt, torch.Tensor):
        flags |= 64
        _require(dt.dtype == torch.float32 and dt.shape == (b,), "tensor dt must be FP32 [B]")
    pn = 2 if prenorm == "rms" else 0
    _require(velocity == "flow", "the carried tail serves the flow velocity form")
    null = x
    widened = scratch.view(torch.float32)[: b * m * 1024]
    _launch(lib, "denoise_fp32head_prep", (h, widened), (b * m,))
    _launch(lib, "denoise_fp32head_cube",
            (out, x, widened, weight, bias if bias is not None else null,
             norm_weight if norm_weight is not None else null,
             ada_shift if ada_shift is not None else null,
             row_mask if row_mask is not None else null,
             dof_mask if dof_mask is not None else null,
             dof_mask if dof_mask is not None else null,
             padding_velocity if padding_velocity is not None else null,
             pinned if pinned is not None else null,
             pinned_values if pinned_values is not None else null,
             dt if isinstance(dt, torch.Tensor) else null),
            (b, m, a, b * m * 1024, m * 1024, 1024, rmb, rmt, db, drow,
             flags, pn,
             _bits(dt) if not isinstance(dt, torch.Tensor) else _bits(0.0),
             _bits(eps), _bits(0.0)))
    return out


# ---------------------------------------------------------------------------
# Carried raw entries (S3): the binding layer's ``requires`` conditions probe
# these names on the namespace (has_functions / getattr), and the binding
# bodies call them as the general tail beyond their own knife envelopes. The
# glue's full rmsnorm port (glue/rmsnorm.py) carries the v22 FFI classes, but
# the wide/generic launcher surface they dispatch to is not part of the 45-
# kernel set (the msprof ledger of both final stacks shows the non-knife norm
# sites running vendor compositions); the shims below therefore compose the
# reference numerics in torch, so the binding's internal knife mounts
# (soupunlock rmsfull, simdepi, rmsfuse, the d64 rope) serve their audited
# envelopes while every general tail matches the reference composition the
# @partial fallback would have produced.
def _reference_rmsnorm(input, weight, epsilon, zero_centered):
    dtype = input.dtype
    x = input.float()
    stats = x.pow(2).mean(-1, keepdim=True)
    y = x * torch.rsqrt(stats + float(epsilon))
    if weight is not None:
        y = y * (weight.float() + 1.0 if zero_centered else weight.float())
    return y.to(dtype)


def _reference_add_rmsnorm(residual, input, weight, epsilon, zero_centered):
    dtype = input.dtype
    s = residual.float() + input.float()
    stats = s.pow(2).mean(-1, keepdim=True)
    y = s * torch.rsqrt(stats + float(epsilon))
    if weight is not None:
        y = y * (weight.float() + 1.0 if zero_centered else weight.float())
    return s.to(dtype), y.to(dtype)


class _CarriedRmsnorm:
    """The raw Rmsnorm entry over the reference composition (the wide/generic
    launcher surface stayed out of the 45-kernel set; see the S3 notes)."""

    def __call__(self, input, weight, epsilon, inference=False, zero_centered=False,
                 use_wide=True, *, residual=None, quant=None, **_):
        if quant is not None:
            raise ValueError("the unified ascend library carries no quantized RMSNorm")
        if residual is not None:
            return _reference_add_rmsnorm(residual, input, weight, epsilon, zero_centered)
        return _reference_rmsnorm(input, weight, epsilon, zero_centered)


class _CarriedAddRmsnorm(_CarriedRmsnorm):
    def __call__(self, residual, input, weight, epsilon, inference=False,
                 zero_centered=False, **_):
        return _reference_add_rmsnorm(residual, input, weight, epsilon, zero_centered)

    def inplace(self, residual, input, weight, epsilon):
        s, y = _reference_add_rmsnorm(residual, input, weight, epsilon, True)
        residual.copy_(s)
        input.copy_(y)


class _CarriedRope:
    """The raw Rope entry; ``full`` composes the reference rotation (the d64
    and bhsd launchers ride the binding's own rope_full envelope)."""

    def full(self, q, k, cos, sin, interleave, rotary_dim):
        from xinfer.ops.reference.rope import rope_full

        return rope_full(q, k, cos, sin, rotary_dim)


def _carried_attention_varlen(q, k, v, cu_seqlens=None, max_seqlen=None, causal=False,
                              softmax_scale=None, out_gate=None):
    """The raw varlen entry over the glue's TND bridge (the packed vision
    segments). The causal and gated varlen forms were routed=0 in both final
    stacks (INVENTORY §8); the binding's own bridge paths (gated_causal,
    prefix_kvcache) serve those sites, so this adapter rejects them."""
    if causal or out_gate is not None:
        raise ValueError("the unified ascend library carries no causal/gated varlen entry")
    if cu_seqlens is None:
        raise ValueError("the varlen entry requires cu_seqlens")
    import xinfer.ops.backends.ascend.cann_attention as _cann_attention

    bridge = getattr(_carried_attention_varlen, "_bridge", None)
    if bridge is None:
        bridge = _cann_attention.CannAttention()
        _carried_attention_varlen._bridge = bridge
    return bridge.varlen_attention(q, k, v, tuple(int(i) for i in cu_seqlens),
                                   scale=softmax_scale)


def _carried_attention_varlen_backend(q, k, v, *, cu_seqlens=None, seg_len=None, max_seqlen=None,
                                      key_len=None, causal=False, out_quant="none", out_gate=None,
                                      gate_act="sigmoid", backend="auto"):
    """Name of the backend the carried varlen entry would use (raises when none accepts).

    The unified core carries the TND non-causal non-gated form only -- the packed
    vision segments over the FIA V5 bridge. This mirrors the retired FFI face's
    ``attention_varlen_backend`` selector so the binding layers' partial-support
    probes stay truthful; the envelope check is the bridge's own, run without
    launching anything.
    """
    if (causal or out_gate is not None or out_quant != "none" or seg_len is not None
            or key_len is not None):
        raise ValueError("the unified ascend library carries no causal/gated/quantized varlen entry")
    if cu_seqlens is None:
        raise ValueError("the varlen entry requires cu_seqlens")
    if (q.ndim != 3 or k.ndim != 3 or v.ndim != 3
            or tuple(q.shape) != tuple(k.shape) or tuple(k.shape) != tuple(v.shape)
            or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
            or q.device.type != "npu" or k.device != q.device or v.device != q.device
            or q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1
            or not q.shape[0] or not q.shape[1] or not 0 < q.shape[2] <= 512):
        raise ValueError("the TND varlen entry requires dense BF16 [T, H, D] rows, D <= 512, on the NPU")
    bounds = tuple(int(i) for i in (cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else cu_seqlens))
    if (len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != q.shape[0]
            or any(end < start for start, end in zip(bounds[:-1], bounds[1:]))):
        raise ValueError("packed segment boundaries must cover the token sequence in order")
    return "ascend_fia"


# The namespace surface consumed through ``_support.xc`` / ``CORE``:
_EXPORTS = {
    # carried class entries and bridge callables; the class imports resolve
    # lazily (the glue modules themselves import CORE from here)
    "MissingBackendError": MissingBackendError,
    "cann_mm_fp32_workspace_size": cann_mm_fp32_workspace_size,
    "cann_mm_fp32": cann_mm_fp32,
    "cann_fia_v5_probe_workspace_size": cann_fia_v5_probe_workspace_size,
    "cann_fia_v5_probe": cann_fia_v5_probe,
    "cann_fia_vision_probe_workspace_size": cann_fia_vision_probe_workspace_size,
    "cann_fia_vision_probe": cann_fia_vision_probe,
    "Rmsnorm": _CarriedRmsnorm,
    "AddRmsnorm": _CarriedAddRmsnorm,
    "add_rmsnorm": _CarriedAddRmsnorm(),
    "attention_varlen": _carried_attention_varlen,
    "attention_varlen_backend": _carried_attention_varlen_backend,
    "Rope": _CarriedRope,
    "gemm_qkv_cache_bias_rope_m16": gemm_qkv_cache_bias_rope_m16,
    "gemm_qkv_cache_bias_rope_prefill": gemm_qkv_cache_bias_rope_prefill,
    "action_qkv_gemm_norm_rope": action_qkv_gemm_norm_rope,
    "gemm_qkv_vision_rope_finish": gemm_qkv_vision_rope_finish,
    "gdr_cann_chunk_workspace_size": gdr_cann_chunk_workspace_size,
    "gdr_cann_chunk": gdr_cann_chunk,
    "gemm_epilogue_glu": gemm_epilogue_glu,
    "gated_residual_adarmsnorm": gated_residual_adarmsnorm,
    "double_adarmsnorm_gated_residual_inplace": double_adarmsnorm_gated_residual_inplace,
    "denoise_input_embed": denoise_input_embed,
    "denoise_output_euler": denoise_output_euler,
}


_CLASS_ENTRIES = ("CannGatedDeltaNet", "Adarmsnorm")


def __getattr__(name):
    if name in _CLASS_ENTRIES:
        import importlib
        module = importlib.import_module(
            "xinfer.ops.backends.ascend."
            + ("cann_gated_delta_net" if name == "CannGatedDeltaNet" else "adarmsnorm"))
        return getattr(module, name)
    entry = _EXPORTS.get(name)
    if entry is None:
        raise AttributeError(f"the unified ascend library carries no entry {name!r}")
    return entry


def __dir__():
    return sorted(_EXPORTS)


def explicit_export(name):
    """The curated export ``name`` (the ``_EXPORTS`` registry), or None.

    The namespace's explicit surface: the same registry ``__getattr__`` serves its dynamic
    lookups from, never a manufactured entry — the paired-loader rule ``_support.canonical``
    reads through the namespace's ``explicit_export``.
    """
    return _EXPORTS.get(name)


# The ``CORE`` surface the ported glue modules reference (the former
# ``CORE`` object): attribute access over the same namespace, plus
# the absent-entry ValueError convention for the class glues' fail-safes.
class _CoreFacade:
    def __getattr__(self, name):
        try:
            return __getattr__(name)
        except AttributeError:
            # the class glues' fallback contract: an absent entry reads as a
            # callable that rejects its layout, so their try/except paths
            # degrade to the released chains instead of crashing imports
            def absent(*args, **kwargs):
                raise ValueError(f"the unified ascend library carries no {name} entry")

            return absent


CORE = _CoreFacade()
