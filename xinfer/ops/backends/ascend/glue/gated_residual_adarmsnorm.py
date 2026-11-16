"""Gated residual add + AdaRMSNorm in one pass.

    residual_output = residual + branch * branch_gate     # BF16 rounding preserved
    output, gate    = adarmsnorm(residual_output, modulation)

Replaces the ``add`` + ``adarmsnorm`` pair.  ``modulation`` is three
concatenated ``hidden``-wide blocks -- (scale, shift, gate) -- and the
effective scale is ``1 + modulation[..., :H]`` (the zero-centred convention).

With ``pinned_modulation`` and ``pinned_rows`` (int32 ``[B]``, read on device)
a shared ``[B, 3H]`` modulation is split by row: the first ``pinned_rows[b]``
tokens of batch row ``b`` use ``pinned_modulation``.  ``branch_gate`` and the
returned ``gate`` are then per token, ``residual.shape``.
"""

import torch

from xinfer.ops.backends.ascend import _provider
from xinfer.ops.backends.ascend.glue.core import CORE



@dump_op_inputs
def gated_residual_adarmsnorm(
    residual,
    branch,
    branch_gate,
    modulation,
    epsilon,
    *,
    pinned_modulation=None,
    pinned_rows=None,
):
    """Returns ``(output, gate, residual_output)``."""
    output = torch.empty_like(residual)
    residual_output = torch.empty_like(residual)
    pinned = pinned_modulation is not None or pinned_rows is not None
    gate = torch.empty(
        residual.shape if pinned else (*residual.shape[:-2], 1, residual.shape[-1]),
        dtype=residual.dtype,
        device=residual.device,
    )
    CORE.gated_residual_adarmsnorm(
        output,
        gate,
        residual_output,
        residual,
        branch,
        branch_gate,
        modulation,
        pinned_modulation,
        pinned_rows,
        float(epsilon),
    )
    return output, gate, residual_output
