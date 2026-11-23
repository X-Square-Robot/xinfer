"""Optional CUDA attention implementations tried outside the in-tree kernels' measured shapes."""

from __future__ import annotations

from functools import cache
from importlib.util import find_spec
import logging
import os
from types import SimpleNamespace

import torch

from xinfer.ops.backends.cuda._native import capturing
from xinfer.ops.registry import note_reference_call, note_route

logger = logging.getLogger(__name__)
_support = {}


def _optional(module, name, flag):
    if os.environ.get(flag, "1") == "0" or find_spec(module) is None:
        return None
    return SimpleNamespace(__name__=name)


FLEX = _optional("torch.nn.attention.flex_attention", "flex_attention", "XINFER_FLEX_ATTENTION")
FLA = _optional("fla", "fla", "XINFER_FLA")


def layout(value):
    if torch.is_tensor(value):
        return value.device, value.dtype, tuple(value.shape), value.stride()
    if isinstance(value, (tuple, list)):
        return tuple(layout(item) for item in value)
    return value


def run(op, library, args, external, reference):
    """Only probe a new signature eagerly. Errors on a confirmed path and device faults propagate."""
    key = (op, library and library.__name__, layout(args))
    device = args[0].device
    supported = _support.get(key) if library else False
    if supported:
        result = external()
    elif supported is None and device.type == "cuda" and not capturing(device):
        try:
            result = external()
        except torch.OutOfMemoryError:
            raise
        except (ImportError, RuntimeError, TypeError, ValueError, NotImplementedError) as error:
            torch.cuda.current_stream(device).synchronize()  # propagate asynchronous CUDA errors
            logger.info("%s: %s declined this signature (%s)", op, library.__name__, error)
            _support[key] = supported = False
        else:
            _support[key] = supported = True
    if supported:
        note_route(library.__name__)
        return result
    note_reference_call(op, key[-1])
    return reference()


@cache
def flex():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def attention(q, k, v, prefix_len, new_pos, flash_mask, scale):
        batch, rows = q.shape[:2]
        length = k.shape[1]
        if flash_mask is not None:
            upper, lower = flash_mask[:, 0, :, 0], flash_mask[:, 0, :, 1]

        def visible(b, h, t, c):
            seen = (c < prefix_len[b]) | ((c >= new_pos) & (c < new_pos + rows))
            if flash_mask is not None:
                # Block-mask construction also visits the padded end of the last tile.
                column = torch.clamp(c, max=length - 1)
                seen = seen & (t < upper[b, column]) & (t >= lower[b, column])
            return (c < length) & seen

        mask = create_block_mask(visible, batch, None, rows, length, device=q.device)
        output = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                block_mask=mask, scale=scale, enable_gqa=True,
                                kernel_options={"FORCE_USE_FLEX_ATTENTION": True})
        return output.transpose(1, 2).reshape(batch, rows, -1)

    return torch.compile(attention, dynamic=False, fullgraph=True)


def fla_delta(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, num_k_heads, num_v_heads, head_k_dim,
                         head_v_dim, eps):
    """``gated_delta_net`` on fla: its causal convolution, chunked gated delta rule (q/k L2 norm, decay and write
    strength computed in the kernel) and gated RMSNorm (FP32 inside, one rounding)."""
    from fla.modules.convolution import causal_conv1d
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    b, n, _ = qkv.shape
    key_dim, value_dim = num_k_heads * head_k_dim, num_v_heads * head_v_dim
    q, k, v = causal_conv1d(qkv, conv_weight, activation="silu")[0].split([key_dim, key_dim, value_dim], dim=-1)
    z, beta_logit, a = zba.split([value_dim, num_v_heads, num_v_heads], dim=-1)
    o, _ = chunk_gated_delta_rule(q.reshape(b, n, num_k_heads, head_k_dim), k.reshape(b, n, num_k_heads, head_k_dim),
                                  v.reshape(b, n, num_v_heads, head_v_dim), g=a, beta=beta_logit,
                                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
                                  use_beta_sigmoid_in_kernel=True)
    o = rms_norm_gated(o.reshape(-1, head_v_dim), z.reshape(-1, head_v_dim), norm_weight, None, eps=eps)
    return o.reshape(b, n, value_dim)
