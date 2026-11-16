"""Operator call costs (``xinfer.observability.op_cost``) against hand-computed values."""

import inspect

import pytest
import torch

from xinfer import ops
from xinfer.observability.op_cost import FORMULAS, OpCost, op_cost

BF16, F32, I32, I64 = torch.bfloat16, torch.float32, torch.int32, torch.int64
E4M3, U8 = torch.float8_e4m3fn, torch.uint8


def t(*shape, dtype=BF16):
    return torch.empty(shape, dtype=dtype, device="meta")


# Sample shapes, bytes per element in the comments: bf16 2, fp32 4, int64 8, int32 4, e4m3 / uint8 / bool 1.
Q, KV = t(2, 5, 4, 16), t(2, 5, 2, 16)  # q 1280, k / v 640 bytes
PROMPT = t(2, 2, 7, 16)  # 896 bytes
# MXFP8 / NVFP4 scales of 4 or 32 rows: rows padded to 128, scale columns to 4 -> 512 bytes.
A8, W8, S = t(4, 64, dtype=E4M3), t(32, 64, dtype=E4M3), t(512, dtype=U8)  # 256, 2048, 512 bytes
A4, W4, S4 = t(4, 32, dtype=U8), t(32, 32, dtype=U8), t(512, dtype=E4M3)  # 128, 1024, 512 bytes
GEMM = 2 * 4 * 32 * 64  # M = 4, N = 32, K = 64
# Fused operators: BF16 x [M = 4, K = 64] and [N, K] weights; 2 query heads and 1 KV head of D = 8 make [q | k | v]
# 32 rows, 16 gate rows make 48; compact RoPE tables [M, 4], a modulation table [3, 3K], head-major caches.
X, W, W48 = t(4, 64), t(32, 64), t(48, 64)  # 512, 4096, 6144 bytes
ROPE, MOD, CACHE = t(4, 4, dtype=F32), t(3, 192), t(1, 12, 8)  # 64, 1152, 192 bytes
CLASS = t(4, dtype=I32)
# The delta rule of chunk_gated_delta_rule's case below.
DELTA_RULE = 4 * (2 * 64 * 64 * (3 * 16 + 2 * 32) + 3 * 2 * 64 * 16 * 32 + sum(2 * i * i for i in range(64)))
KV_CACHE = t(2, 12, 2, 16)  # sequence-major cache of Q's batch and heads, 12 rows: 1536 bytes
# Flow rows: B = 1, M = 6, A = 4 FP32 values and a K = 32 BF16 stream.
VALUES, STREAM = t(1, 6, 4, dtype=F32), t(1, 6, 32)  # 96 and 384 bytes

# Operator -> (arguments, cost as flops, bytes read, bytes written, pipe), all inputs meta tensors: no data is read.
FORMULA_CASES = {
    "linear": ((t(2, 3, 64), t(32, 64), t(32)),  # M = 6, N = 32, K = 64
               OpCost(2 * 6 * 32 * 64, 768 + 32 * 64 * 2 + 32 * 2, 6 * 32 * 2, "bf16")),
    # 16x16 image, 4x4 patches with stride 4: 4x4 outputs per image and channel.
    "conv2d": ((t(2, 3, 16, 16), t(8, 3, 4, 4), t(8), 4),
               OpCost(2 * (2 * 16) * 8 * (3 * 4 * 4), 2 * 3 * 256 * 2 + 8 * 3 * 16 * 2 + 16, 2 * 16 * 8 * 2, "bf16")),
    # 100 tokens -> 2 chunks of 64, 2 heads, Dk 16, Dv 32. Per chunk: products K·Kᵀ, Q·Kᵀ, A·(K·β·e^g) over Dk and
    # A·(V·β), A·V' over Dv; three products with the [Dk, Dv] state; row i of the triangular solve 2·i² FLOPs.
    "chunk_gated_delta_rule": (
        (t(1, 100, 2, 16), t(1, 100, 2, 16), t(1, 100, 2, 32), t(1, 100, 2, dtype=F32), t(1, 100, 2)),
        OpCost(DELTA_RULE, 6400 + 6400 + 12800 + 800 + 400, 12800, "bf16")),
    # Attention: QKᵀ and PV are 2·D FLOPs each per (query, key) pair and head; D = 16, 4 heads.
    "causal_attention": ((Q, KV, KV), OpCost(4 * 16 * 2 * 4 * 15, 1280 + 640 + 640, 1280, "bf16")),  # 15 = 5·6/2
    "action_attention": ((Q, KV, KV, PROMPT, PROMPT, t(2, dtype=I64), t(2, dtype=I64)),
                         OpCost(4 * 16 * 2 * 4 * 5 * 12, 1280 + 640 + 640 + 896 + 896 + 16 + 16, 1280, "bf16")),
    "masked_joint_attention": ((Q, KV, KV, PROMPT, PROMPT, t(2, 1, 5, 12, dtype=torch.bool)),
                               OpCost(4 * 16 * 2 * 4 * 5 * 12, 1280 + 640 + 640 + 896 + 896 + 120, 1280, "bf16")),
    # S = 12 cache columns: 7 cached keys read per cache, the 5 own keys written into both caches.
    "kvcache_attention": ((Q, KV, KV, t(2, 12, 2, 16), t(2, 12, 2, 16), t(2, dtype=torch.int32),
                           t(2, dtype=torch.int32)),
                          OpCost(4 * 16 * 2 * 4 * 5 * 12, 1280 + 640 + 640 + 8 + 8 + 2 * (2 * 7 * 2 * 16 * 2),
                               1280 + 640 + 640, "bf16")),
    "prefix_attention": ((t(2, 6, 4, 16), t(2, 6, 1, 16), t(2, 6, 1, 16), t(2, dtype=I64)),
                         OpCost(4 * 16 * 2 * 4 * 6 * 6, 1536 + 384 + 384 + 16, 1536, "bf16")),
    "varlen_attention": ((t(7, 4, 16), t(7, 4, 16), t(7, 4, 16), [0, 3, 7]),
                         OpCost(4 * 16 * 4 * (3 * 3 + 4 * 4), 3 * 896, 896, "bf16")),
    "mxfp8_gemm": ((A8, S, W8, S, t(32)), OpCost(GEMM, 256 + 512 + 2048 + 512 + 64, 4 * 32 * 2, "mxfp8")),
    "mxfp8_gemm_rope": ((A8, S, W8, S, t(32), t(4, 4, 4, dtype=F32), 16, 8, 4),
                        OpCost(GEMM, 256 + 512 + 2048 + 512 + 64 + 256, 4 * 32 * 2, "mxfp8")),
    "mxfp8_gemm_silu_mul": ((A8, S, W8, S), OpCost(GEMM, 256 + 512 + 2048 + 512, 4 * 16 * 2, "mxfp8")),
    "mxfp8_gemm_residual": ((A8, S, W8, S, t(4, 32)), OpCost(GEMM, 256 + 512 + 2048 + 512 + 256, 4 * 32 * 2, "mxfp8")),
    "mxfp8_gemm_epilogue": (((A8, S), (W8, S), t(32)), OpCost(GEMM, 256 + 512 + 2048 + 512 + 64, 4 * 32 * 2, "mxfp8")),
    "mxfp8_gemm_glu_quantize": (((A8, S), (t(64, 64, dtype=E4M3), S)),
                                OpCost(2 * GEMM, 256 + 512 + 4096 + 512, 4 * 32 + 512, "mxfp8")),
    "mxfp8_qkv_rope_epilogue": (((A8, S), (W8, S), t(32), t(4, 2, dtype=F32), t(4, 2, dtype=F32), 2, 4),
                                OpCost(GEMM, 256 + 512 + 2048 + 512 + 64 + 64, 4 * 32 * 2, "mxfp8")),
    "nvfp4_gemm": ((A4, S4, W4, S4), OpCost(GEMM, 128 + 512 + 1024 + 512, 4 * 32 * 2, "nvfp4")),
    # weights (q, scales, global scale); an NVFP4 output writes 9/32 of the BF16 bytes
    "nvfp4_gemm_epilogue": (((A4, S4), (W4, S4, 0.5), t(32)),
                            OpCost(GEMM, 128 + 512 + 1024 + 512 + 64, 4 * 32 * 2, "nvfp4")),
    "nvfp4_gemm_glu_quantize": (((A4, S4), (W4, S4, 0.5)),
                                OpCost(GEMM, 128 + 512 + 1024 + 512, 4 * 16 * 2 * 9 // 32, "nvfp4")),
    "nvfp4_qkv_rope_epilogue": (((A4, S4), (W4, S4, 0.5), t(32), t(4, 2, dtype=F32), t(4, 2, dtype=F32), 2, 4),
                                OpCost(GEMM, 128 + 512 + 1024 + 512 + 64 + 64, 4 * 32 * 2, "nvfp4")),
    "gemm_epilogue": ((X, W, t(32)), OpCost(GEMM, 512 + 4096 + 64, 4 * 32 * 2, "bf16")),
    "gemm_glu_epilogue": ((X, W), OpCost(GEMM, 512 + 4096, 4 * 16 * 2, "bf16")),  # 16 gate and 16 up rows
    # bias [32] (64 bytes), residual [1, 4, 32] (256 bytes), FP32 norm weight [32] (128 bytes); writes the sum and
    # its norm
    "gemm_add_rms_norm": ((X, W, t(1, 4, 32), t(32, dtype=F32), 1e-6, t(32)),
                          OpCost(GEMM, 512 + 4096 + 64 + 256 + 128, 2 * 2 * 4 * 32, "bf16")),
    # 16 gate and 16 up rows, down [64, 16], residual [1, 4, 64] (512 bytes), FP32 norm weight [64] (256 bytes);
    # writes the sum and its norm
    "mlp_add_rms_norm": ((X, W, t(64, 16), t(1, 4, 64), t(64, dtype=F32), 1e-6),
                         OpCost(GEMM + 2 * 4 * 16 * 64, 512 + 4096 + 2048 + 512 + 256, 2 * 2 * 4 * 64, "bf16")),
    "qkv_rope_epilogue": ((X, W48, t(48), ROPE, ROPE, 2, 8),  # 2 heads of q, k and v, D = 8
                          OpCost(2 * 4 * 48 * 64, 512 + 6144 + 96 + 128, 4 * 48 * 2, "bf16")),
    # The caches are not read: Q goes to the output, K and V to 4 new cache rows, 4 x 32 values in all.
    "qkv_rope_kvcache": ((X, W, t(32), ROPE, ROPE, CACHE, CACHE, 5, 2, 1, 8),
                         OpCost(GEMM, 512 + 4096 + 64 + 128, 4 * 32 * 2, "bf16")),
    # new_pos 5: every query sees cache rows [0, 10), 1280 bytes of each cache.
    "prefix_kvcache_attention": ((Q, KV_CACHE, KV_CACHE, t(2, dtype=I32), 5),
                                 OpCost(4 * 16 * 2 * 4 * 5 * 10, 1280 + 8 + 2 * 1280, 1280, "bf16")),
    "flow_velocity_euler": ((VALUES, STREAM, t(4, 32), 0.1), OpCost(2 * 6 * 4 * 32, 96 + 384 + 256, 96, "bf16")),
}


def test_formulas_are_registered_operators_with_the_reference_parameters():
    assert set(FORMULA_CASES) == set(FORMULAS) and set(FORMULAS) <= set(ops.registry.names())
    for name, formula in FORMULAS.items():
        reference = ops.registry.spec(name).reference
        assert list(inspect.signature(formula).parameters) == list(inspect.signature(reference).parameters), name


@pytest.mark.parametrize("name", sorted(FORMULA_CASES))
def test_formula(name):
    args, expected = FORMULA_CASES[name]
    assert op_cost(name, args, {}, None) == expected


def test_formula_variants():
    assert op_cost("linear", (t(4, 8, dtype=F32), t(16, 8, dtype=F32)), {}, None) == \
        OpCost(2 * 4 * 16 * 8, 128 + 512, 256, "fp32")
    action = op_cost("action_attention", (Q, KV, KV, PROMPT, PROMPT, t(2, dtype=I64)), {"scale": 0.25}, None)
    assert action.read_bytes == 1280 + 640 + 640 + 896 + 896 + 16  # no prompt lengths
    cu_seqlens = torch.tensor([0, 7])  # a host tensor, read like the reference reads it
    assert op_cost("varlen_attention", (t(7, 4, 16),) * 3, {"cu_seqlens": cu_seqlens}, None).flops == 4 * 16 * 4 * 49
    assert op_cost("mxfp8_gemm_epilogue", ((A8, S), (W8, S)), {"out_quant": True}, None).write_bytes == 4 * 32 + 512
    assert OpCost(1, 2, 3, "bf16").bytes == 5


def test_fused_formulas_read_their_optional_inputs():
    epilogue = {"residual": t(4, 32), "residual_gate": t(3, 32), "row_class": CLASS, "row_scale": t(4, dtype=F32)}
    assert op_cost("gemm_epilogue", (X, W), epilogue, None).read_bytes == 512 + 4096 + 256 + 192 + 16 + 16
    glu = {"bias": t(32), "modulation": MOD, "row_class": CLASS, "eps": 1e-6}
    assert op_cost("gemm_glu_epilogue", (X, W), glu, None).read_bytes == 512 + 4096 + 64 + 1152 + 16
    mask = {"flash_mask": t(2, 1, 12, 2, dtype=I32), "scale": 0.25}  # 192 bytes
    attention = op_cost("prefix_kvcache_attention", (Q, KV_CACHE, KV_CACHE, t(2, dtype=I32), 5), mask, None)
    assert attention.read_bytes == 1280 + 8 + 2 * 1280 + 192
    euler = {"dof_mask": VALUES, "padding_velocity": VALUES, "pinned": t(1, dtype=I32), "pinned_values": VALUES}
    assert op_cost("flow_velocity_euler", (VALUES, STREAM, t(4, 32, dtype=F32), 0.1), euler, None) == \
        OpCost(2 * 6 * 4 * 32, 96 + 384 + 512 + 3 * 96 + 4, 96, "fp32")  # an FP32 readout runs in FP32


def test_other_operators_move_their_inputs_and_outputs_once():
    x, weight = t(6, 64), t(64, dtype=F32)  # 768 and 256 bytes
    assert op_cost("rms_norm", (x, weight, 1e-6), {}, t(6, 64)) == OpCost(0, 768 + 256, 768, "memory")
    gate, up = torch.randn(6, 64, dtype=BF16), torch.randn(6, 64, dtype=BF16)
    out = ops.reference.silu_and_mul(gate, up)
    assert op_cost("silu_and_mul", (gate, up), {}, out) == OpCost(0, 768 + 768, 768, "memory")
    x = torch.randn(4, 64, dtype=BF16)  # 512 bytes
    q, scales = ops.reference.mxfp8.quantize(x, share_scale_128=False)  # E4M3 [4, 64] and 512 swizzled scale bytes
    assert op_cost("mxfp8_quantize", (x,), {"share_scale_128": False}, (q, scales)) == \
        OpCost(0, 512, 4 * 64 + 512, "memory")
    q, scales = ops.reference.nvfp4.quantize(x)  # packed E2M1 [4, 32] and 512 swizzled scale bytes
    assert op_cost("nvfp4_quantize", (x,), {}, (q, scales)) == OpCost(0, 512, 4 * 32 + 512, "memory")
