// Codes and limits shared by the multicast kernels (mcast_bf16_sm110, mcast_ffn_sm110) and their FFI.
#pragma once

// Activation of the GLU kernels (gate/up and the fused MLP): the numbering of gemm.py ACTIVATIONS, so the Python
// side passes its code unchanged.
enum { MCAST_ACT_SILU = 0, MCAST_ACT_GELU_TANH = 1 };
// Activation of the stored output without a residual (mcast_bf16_sm110_gemm_out): not the GLU numbering above.
// None is -1 so that a GLU code passed by mistake (MCAST_ACT_SILU) is rejected rather than read as "none".
enum { MCAST_OUT_ACT_NONE = -1, MCAST_OUT_ACT_GELU_TANH = 1 };
// Largest intermediate size the gate/up kernel takes (its 2 I weight rows stay within int).
enum { MCAST_MAX_INTER = 1 << 20 };
