"""The ViT and prompt stream of the Wall-X Qwen3.5 models on the CUDA backend's kernels (``numerics="optimized"``).

The Wall-X 2B stages use one operator per fusion from ``xinfer/ops/reference/``.
The backend selects the kernel for each shape. GEMM weights are ``[N, K]``. Differences to the
reference composition (``numerics="reference"``, the wall-x model under autocast):

- ViT: every weight in BF16, including the position table, which one kernel interpolates; the patch embedding is a
  GEMM; the LayerNorms run ``layer_norm_cast``; QKV, bias and RoPE are one GEMM with the rotation on the FP32
  accumulator; the bias and residual add of the projection and fc2, the GELUs of fc1 and the merger are GEMM
  epilogues; attention runs ``segment_attention``;
- prompt stream: the DeltaNet input projections ``[qkv | z | b | a]`` are one GEMM that zeroes the padding rows (the
  padding follows every real token, so real tokens do not see it); the DeltaNet mixer from the convolution to the
  gated norm is one call (``gated_delta_net``); ``[Q·gate | K | V]`` is one GEMM, the q/k norms and split one
  kernel and RoPE another; attention and its output gate are one call; the MLP's SiLU-mul is the gate/up GEMM's
  epilogue; the MLP's down projection stays torch's GEMM (the backend lists no kernel for it, and its cuBLASLt
  fallback with a fixed workspace is slower there).
"""

from __future__ import annotations

import torch
from torch import nn

from xinfer.models.layers.basic import ParamGroup, _meta_param
from xinfer.loader import Source
from xinfer.ops import active_ops

DTYPE = torch.bfloat16
VISION_STAGE_OPS = ("gemm_epilogue", "qkv_rope_epilogue", "layer_norm_cast", "pos_embed_interp", "rot_pos",
                    "segment_attention")
PREFILL_STAGE_OPS = ("gemm_epilogue", "gemm_glu_epilogue", "linear", "zero_centered_rms_norm",
                     "add_zero_centered_rms_norm", "qkv_norm_split", "rope_full", "gated_causal_attention",
                     "gated_delta_net")
# With MXFP8 GEMMs (``quant="mxfp8"``): the patch embedding and the merger stay BF16
VISION_STAGE_OPS_MXFP8 = ("gemm_epilogue", "layer_norm_cast", "pos_embed_interp", "rot_pos", "segment_attention",
                          "mxfp8_quantize", "mxfp8_layer_norm_quantize", "mxfp8_qkv_rope_epilogue",
                          "mxfp8_gemm_epilogue")
PREFILL_STAGE_OPS_MXFP8 = ("qkv_norm_split", "rope_full", "gated_causal_attention", "gated_delta_net", "mxfp8_quantize",
                           "mxfp8_zero_centered_rms_norm_quantize", "mxfp8_add_zero_centered_rms_norm_quantize",
                           "mxfp8_gemm_epilogue", "mxfp8_gemm_glu_quantize")
# With NVFP4 GEMMs (``quant="nvfp4"`` or ``"nvfp4_mse"``): the patch embedding and the DeltaNet z/b/a projection stay
# BF16, the merger's GEMMs are NVFP4
VISION_STAGE_OPS_NVFP4 = ("gemm_epilogue", "layer_norm_cast", "pos_embed_interp", "rot_pos", "segment_attention",
                          "nvfp4_quantize", "nvfp4_layer_norm_quantize", "nvfp4_qkv_rope_epilogue",
                          "nvfp4_gemm_epilogue")
PREFILL_STAGE_OPS_NVFP4 = ("gemm_epilogue", "zero_centered_rms_norm", "add_zero_centered_rms_norm", "qkv_norm_split",
                           "rope_full", "gated_causal_attention", "gated_delta_net", "nvfp4_quantize",
                           "nvfp4_add_zero_centered_rms_norm_quantize", "nvfp4_gemm_epilogue",
                           "nvfp4_gemm_glu_quantize")
QUANT_METHODS = ("mxfp8", "nvfp4", "nvfp4_mse")  # the GEMM formats of a stage; None: BF16



class _FormatStage(nn.Module):
    """A stage whose GEMMs run in one format: BF16 (``quant`` None), MXFP8 or NVFP4 (``"nvfp4"``, ``"nvfp4_mse"``)."""

    def __init__(self, quant: str | None):
        super().__init__()
        if quant not in (None, *QUANT_METHODS):
            raise ValueError(f"{type(self).__name__}: quant must be None or one of {QUANT_METHODS}, got {quant!r}")
        self.quant = quant
        self.fmt = None if quant is None else "mxfp8" if quant == "mxfp8" else "nvfp4"
        self.ops = active_ops()

    def _op(self, name: str):
        """The operator ``name`` of the format: ``name`` in BF16, else ``mxfp8_name`` or ``nvfp4_name``."""
        return getattr(self.ops, name if self.fmt is None else f"{self.fmt}_{name}")

    def _weight(self, group: ParamGroup, name: str):
        """The weight ``name`` of ``group`` in the format (BF16, or ``quantize_weights``' MXFP8 or NVFP4 form)."""
        return getattr(group, name) if self.fmt is None else group.quantized(name)

    def _gemm(self, x, group: ParamGroup, name: str, *args, **kwargs):
        """``gemm_epilogue`` in the format with weight ``name``; a BF16 ``x`` is quantized first in MXFP8 or NVFP4."""
        if self.fmt is not None and isinstance(x, torch.Tensor):
            x = self._op("quantize")(x)
        return self._op("gemm_epilogue")(x, self._weight(group, name), *args, **kwargs)


_SOUPKILL_PROJECTION = []


def _qkv_rope_soupkill(ops, normed, block, cos, sin, cos2, sin2, n, heads, d):
    """ViT QKV projection + bias + RoPE in one pass over the FP32 accumulator.

    Opt-in through ``XINFER_SOUPKILL=vrope`` (xcompute's soupkill mount). Every
    path that is not a clean envelope hit -- flag off, absent knife library, a
    shape or table mismatch, any error from the fused launch -- returns the
    released operator's result, so the model never depends on the knife being
    installed and this call site can be reverted on its own.
    """
    released = lambda: ops.qkv_rope_epilogue(  # noqa: E731 - the single fallback
        normed, block.qkv_weight, block.qkv_bias, cos, sin, heads, d
    )
    try:
        from xcompute.kernel import soupkill
    except Exception:  # noqa: BLE001 - the mount is optional
        return released()
    if not soupkill.enabled("vrope") or normed.dim() != 2 or cos2 is None or cos2.shape[-1] != d:
        return released()
    if not _SOUPKILL_PROJECTION:
        from xcompute.kernel.cann_projection import CannProjection

        _SOUPKILL_PROJECTION.append(CannProjection())
    projection = _SOUPKILL_PROJECTION[0]
    try:
        with projection.accumulator(normed, block.qkv_weight) as acc:
            out = normed.new_empty((normed.shape[0], n), dtype=torch.bfloat16)
            fused = soupkill.vision_qkv_rope(acc, block.qkv_bias, cos2, sin2, out, heads, d)
        if fused is not None:
            return fused
    except Exception:  # noqa: BLE001 - a fused-path failure must not fail the model
        pass
    return released()


def _SOUPFORM_ENABLED(family):
    try:
        from xcompute.kernel import soupform

        return soupform.enabled(family)
    except Exception:  # noqa: BLE001 - the mount is optional
        return False


def _zba_soupform(flat, weight, keep, conv, p):
    """soupform zbasplit: rne(acc*keep) into contiguous conv/zba buffers.

    Opt-in through ``XINFER_SOUPFORM=zbasplit`` (xcompute's soupform mount,
    default off). Any miss -- flag off, absent knife library, wrong rank or a
    fused-path error -- returns None and the caller keeps the released chain.
    """
    if not _SOUPFORM_ENABLED("zbasplit") or flat.dim() != 2:
        return None
    if not _SOUPKILL_PROJECTION:
        from xcompute.kernel.cann_projection import CannProjection

        _SOUPKILL_PROJECTION.append(CannProjection())
    proj = _SOUPKILL_PROJECTION[0]
    try:
        from xcompute.kernel import soupform

        with proj.accumulator(flat, weight) as acc:
            return soupform.zbasplit(acc, keep, conv)
    except (ValueError, RuntimeError):
        return None


def _qkv_rope_packed(ops, normed, block, cos, sin, cos2, sin2, n, heads, d):
    """(qkv, packed): the soupform packed [3, rows, heads, dim] store first.

    Opt-in through ``XINFER_SOUPFORM=vropepack`` (default off): the fused
    finish stores the packed layout directly, so the model-side pack transpose
    disappears (same values, different addresses). Every miss falls through to
    the soupkill vrope knife and then the released operator.
    """
    if _SOUPFORM_ENABLED("vropepack") and normed.dim() == 2 and cos2 is not None and cos2.shape[-1] == d:
        if not _SOUPKILL_PROJECTION:
            from xcompute.kernel.cann_projection import CannProjection

            _SOUPKILL_PROJECTION.append(CannProjection())
        proj = _SOUPKILL_PROJECTION[0]
        try:
            from xcompute.kernel import soupform

            with proj.accumulator(normed, block.qkv_weight) as acc:
                packed = soupform.vision_qkv_rope_packed(acc, block.qkv_bias, cos2, sin2, heads, d)
            if packed is not None:
                return packed, True
        except (ValueError, RuntimeError):
            pass
    return _qkv_rope_soupkill(ops, normed, block, cos, sin, cos2, sin2, n, heads, d), False


class OptimizedVision(_FormatStage):
    """The ViT and patch merger (BF16 weights, ``[out, in]`` as in the checkpoint).

    Args:
        config: ``Qwen35VisionConfig`` of the checkpoint.
        quant: None (BF16), or the GEMM format of the blocks (``"mxfp8"``) or of the blocks and the merger
            (``"nvfp4"``: amax weight scales, ``"nvfp4_mse"``: MSE weight scales); ``quantize_weights`` after loading.
    """

    def __init__(self, config, quant: str | None = None):
        super().__init__(quant)
        self.config = c = config
        self.stage_ops = {None: VISION_STAGE_OPS, "mxfp8": VISION_STAGE_OPS_MXFP8,
                          "nvfp4": VISION_STAGE_OPS_NVFP4}[self.fmt]
        h, i = c.hidden_size, c.intermediate_size
        self.patch = ParamGroup(weight=(h, c.in_features), bias=(h,))
        self.pos_embed = _meta_param(c.num_position_embeddings, h)
        self.blocks = nn.ModuleList(ParamGroup(norm1_weight=(h,), norm1_bias=(h,), qkv_weight=(3 * h, h),
                                               qkv_bias=(3 * h,), proj_weight=(h, h), proj_bias=(h,),
                                               norm2_weight=(h,), norm2_bias=(h,), fc1_weight=(i, h),
                                               fc1_bias=(i,), fc2_weight=(h, i), fc2_bias=(h,))
                                    for _ in range(c.depth))
        self.merge_width = width = h * c.spatial_merge_size ** 2
        self.merger = ParamGroup(norm_weight=(h,), norm_bias=(h,), fc1_weight=(width, width), fc1_bias=(width,),
                                 fc2_weight=(c.out_hidden_size, width), fc2_bias=(c.out_hidden_size,))
        half = c.head_dim // 2
        self.register_buffer("inv_freq", 1.0 / (c.rope_theta ** (torch.arange(0, half, 2, dtype=torch.float) / half)),
                             persistent=False)

    def forward(self, pixel_values, grids, cu_seqlens):
        """Image token embeddings of packed patches.

        Args:
            pixel_values: ``[N, in_features]`` FP32 patch rows in merge-block order.
            grids: Host list of ``(t, h, w)`` per image.
            cu_seqlens: Host list of frame boundaries (attention stays within a frame).

        Returns:
            ``[N / merge², out_hidden_size]`` BF16.
        """
        c, ops = self.config, self.ops
        n, h, heads, d = pixel_values.shape[0], c.hidden_size, c.num_heads, c.head_dim
        x = ops.gemm_epilogue(pixel_values.to(DTYPE), self.patch.weight, self.patch.bias)
        x = x + ops.pos_embed_interp(self.pos_embed, grids, c.spatial_merge_size)
        angles = ops.rot_pos(self.inv_freq, grids, c.spatial_merge_size)
        cos, sin = angles.cos(), angles.sin()
        # soupkill vrope: the rotary table duplicated to the head width, once per
        # vision forward (the fused kernel reads the head's 64 lanes as [c | c]).
        # Two cat kernels per request, not per block; None keeps the released op.
        cos2 = sin2 = None
        try:
            from xcompute.kernel import soupkill

            if soupkill.enabled("vrope") and cos.dim() == 2:
                cos2 = torch.cat((cos, cos), dim=-1).contiguous()
                sin2 = torch.cat((sin, sin), dim=-1).contiguous()
        except Exception:  # noqa: BLE001 - the mount is optional
            cos2 = sin2 = None
        for block in self.blocks:
            x = self._block(block, x, cos, sin, cu_seqlens, cos2, sin2)
        m = self.merger
        merged = ops.layer_norm_cast(x, m.norm_weight, m.norm_bias, c.eps).reshape(-1, self.merge_width)
        if self.fmt == "nvfp4":  # the merger's GEMMs are NVFP4 with NVFP4 blocks, else BF16
            hidden = ops.nvfp4_gemm_epilogue(ops.nvfp4_quantize(merged), m.quantized("fc1_weight"), m.fc1_bias,
                                             activation="gelu", out_quant=True)
            return ops.nvfp4_gemm_epilogue(hidden, m.quantized("fc2_weight"), m.fc2_bias)
        hidden = ops.gemm_epilogue(merged, m.fc1_weight, m.fc1_bias, activation="gelu")
        return ops.gemm_epilogue(hidden, m.fc2_weight, m.fc2_bias)

    def _block(self, block, x, cos, sin, cu_seqlens, cos2=None, sin2=None):
        """One ViT block. With MXFP8 or NVFP4 GEMMs every GEMM input is the BF16 tensor of the BF16 block, quantized:
        the LayerNorms and fc1 write the next GEMM's quantized input, the attention output is quantized for the
        projection. In BF16 the soupkill vrope knife (fused QKV + bias + RoPE) takes the QKV call when its
        mount is enabled; every miss falls back to the released operator."""
        c, ops = self.config, self.ops
        n, h, heads, d = x.shape[0], c.hidden_size, c.num_heads, c.head_dim
        norm = ops.layer_norm_cast if self.fmt is None else self._op("layer_norm_quantize")
        normed = norm(x, block.norm1_weight, block.norm1_bias, c.eps)
        if self.fmt is None:
            qkv, qkv_is_packed = _qkv_rope_packed(ops, normed, block, cos, sin, cos2, sin2, 3 * heads * d, heads, d)
            if qkv_is_packed:
                # SOUP-FORM vropepack: the finish kernel already stored the packed
                # [3, n, heads, dim] layout, so the model-side permute copy of the
                # released chain disappears (same bytes, packed addresses).
                q, k, v = qkv.unbind(0)
            else:
                q, k, v = qkv.view(n, 3, heads, d).unbind(1)
        else:
            qkv = self._op("qkv_rope_epilogue")(normed, self._weight(block, "qkv_weight"), block.qkv_bias, cos, sin,
                                                heads, d)
            q, k, v = qkv.view(n, 3, heads, d).unbind(1)
        o = ops.segment_attention(q, k, v, cu_seqlens, scale=d ** -0.5).reshape(n, h)
        x = self._gemm(o, block, "proj_weight", block.proj_bias, residual=x)
        normed = norm(x, block.norm2_weight, block.norm2_bias, c.eps)
        out_quant = {} if self.fmt is None else {"out_quant": True}
        hidden = self._gemm(normed, block, "fc1_weight", block.fc1_bias, activation="gelu_tanh", **out_quant)
        return self._gemm(hidden, block, "fc2_weight", block.fc2_bias, residual=x)

    def quantize_weights(self) -> None:
        """After loading: the blocks' GEMM weights to the stage's format, and in NVFP4 the merger's too."""
        for block in self.blocks:
            block.quantize(self.quant, self.ops, "qkv_weight", "proj_weight", "fc1_weight", "fc2_weight")
        if self.fmt == "nvfp4":
            self.merger.quantize(self.quant, self.ops, "fc1_weight", "fc2_weight")


class OptimizedPrefill(_FormatStage):
    """The prompt stream up to the last full-attention layer's K/V (``[N, K]`` GEMM weights).

    Args:
        config: The checkpoint's text configuration: ``hidden_size``, ``intermediate_size``, the attention and
            DeltaNet geometry (``num_attention_heads``, ``num_key_value_heads``, ``head_dim``, ``rotary_dim``,
            ``linear_heads``), ``layer_types``, ``full_attention_layers`` and ``rms_norm_eps``.
        quant: None (BF16), or the GEMM format (``"mxfp8"``; ``"nvfp4"``: amax weight scales, ``"nvfp4_mse"``: MSE
            weight scales, with the DeltaNet z/b/a projection BF16); ``quantize_weights`` after loading.
    """

    def __init__(self, config, quant: str | None = None):
        super().__init__(quant)
        self.config = c = config
        self.stage_ops = {None: PREFILL_STAGE_OPS, "mxfp8": PREFILL_STAGE_OPS_MXFP8,
                          "nvfp4": PREFILL_STAGE_OPS_NVFP4}[self.fmt]
        hk, hv, dk, dv, kernel = c.linear_heads
        d, heads, kv = c.hidden_size, c.num_attention_heads, c.num_key_value_heads
        self.conv_dim = conv = 2 * hk * dk + hv * dv  # the DeltaNet convolution's channels: [q | k | v]
        self.layers = nn.ModuleList()
        for kind in c.layer_types[:c.full_attention_layers[-1] + 1]:
            shapes = dict(input_norm=(d,), post_norm=(d,), gate_up=(2 * c.intermediate_size, d),
                          down=(d, c.intermediate_size))
            if kind == "full_attention":
                shapes.update(qkv=((2 * heads + 2 * kv) * c.head_dim, d), q_norm=(c.head_dim,),
                              k_norm=(c.head_dim,), o=(d, heads * c.head_dim))
            else:
                shapes.update(qkv_zba=(conv + hv * dv + 2 * hv, d), conv_weight=(conv, kernel), A_log=(hv,),
                              dt_bias=(hv,), norm_weight=(dv,), out=(d, hv * dv))
            self.layers.append(ParamGroup(**shapes))

    def forward(self, h, cos, sin, prompt_kv, keep):
        """Run the prompt stream and write the post-RoPE K and the V of every full-attention layer.

        With MXFP8 or NVFP4 GEMMs every GEMM input is the BF16 tensor of the BF16 stream, quantized: the norms and the
        gate/up GEMM write the next GEMM's quantized input, the attention and DeltaNet outputs are quantized for their
        projections (``_norm``, ``_qkv``, ``_delta_input``, ``_mlp`` hold the differences between the formats).

        Args:
            h: ``[1, P, hidden]`` BF16 prompt embeddings (padding included).
            cos: FP32 ``[1, P, rotary_dim / 2]`` compact cosines.
            sin: FP32 ``[1, P, rotary_dim / 2]`` compact sines.
            prompt_kv: ``[full-attention layers, 2, 1, kv_heads, P, head_dim]`` output.
            keep: FP32 ``[P]``, 1 for the real tokens and 0 for the padding.
        """
        c, ops = self.config, self.ops
        hk, hv, dk, dv, _ = c.linear_heads
        heads, kv_heads, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        eps, full = c.rms_norm_eps, c.full_attention_layers
        p = h.shape[1]
        rows = keep.to(DTYPE)[:, None] if self.fmt == "nvfp4" else keep  # zeroes the DeltaNet input's padding rows
        h, x = self._norm(h, self.layers[0].input_norm, c.layer_types[0])
        for i, layer in enumerate(self.layers):
            if c.layer_types[i] == "full_attention":
                q, k, v, gate = ops.qkv_norm_split(self._qkv(x, layer).view(1, p, -1), layer.q_norm, layer.k_norm,
                                                   heads, kv_heads, d, eps)
                q, k = ops.rope_full(q, k, cos, sin, c.rotary_dim)
                slot = full.index(i)
                prompt_kv[slot, 0].copy_(k.transpose(1, 2))
                prompt_kv[slot, 1].copy_(v.transpose(1, 2))
                if i == full[-1]:
                    return
                o = ops.gated_causal_attention(q[0], k[0], v[0], gate.reshape(p, -1), scale=d ** -0.5)
                mixed = self._gemm(o.reshape(p, -1), layer, "o")
            else:
                qkv, zba = self._delta_input(x, layer, rows)
                o = ops.gated_delta_net(qkv.view(1, p, -1), zba.view(1, p, -1), layer.conv_weight, layer.A_log,
                                        layer.dt_bias, layer.norm_weight, hk, hv, dk, dv, eps)
                mixed = self._gemm(o.view(p, -1), layer, "out")
            h, x = self._norm(h, layer.post_norm, "mlp", mixed)
            h, x = self._norm(h, self.layers[i + 1].input_norm, c.layer_types[i + 1], self._mlp(x, layer))

    def _norm(self, h, weight, consumer: str, branch=None):
        """Add ``branch`` (``[P, hidden]``, None for the first layer) to the stream ``h`` (``[1, P, hidden]``) and
        give the zero-centered RMSNorm of the sum in the form ``consumer`` (a layer type or ``"mlp"``) takes: BF16
        ``[P, hidden]``, or its MXFP8 or NVFP4. NVFP4 gives a DeltaNet layer the BF16 norm (``_delta_input``).
        Returns ``(h + branch, input)``."""
        ops, eps = self.ops, self.config.rms_norm_eps
        quantized = self.fmt is not None and not (self.fmt == "nvfp4" and consumer == "linear_attention")
        if branch is not None:
            branch = branch.view(h.shape)
            if quantized:
                return self._op("add_zero_centered_rms_norm_quantize")(branch, h, weight, eps)
            h, normed = ops.add_zero_centered_rms_norm(branch, h, weight, eps)
        elif quantized and self.fmt == "mxfp8":
            return h, ops.mxfp8_zero_centered_rms_norm_quantize(h, weight, eps)
        else:
            normed = ops.zero_centered_rms_norm(h, weight, eps)
        normed = normed.view(h.shape[1], -1)
        return h, (ops.nvfp4_quantize(normed) if quantized else normed)

    def _qkv(self, x, layer):
        """``[Q·gate | K | V]``: one GEMM, or in NVFP4 one per checkpoint weight (each with its global scale)."""
        if self.fmt == "nvfp4":
            return torch.cat([self._gemm(x, layer, name) for name in ("q", "k", "v")], -1)
        return self._gemm(x, layer, "qkv")

    def _delta_input(self, x, layer, rows):
        """The DeltaNet input projections ``(qkv, [z | b | a])`` with the padding rows zeroed (``rows``: ``keep``, or
        its BF16 column in NVFP4): one GEMM that zeroes its output rows, or in NVFP4 the NVFP4 ``qkv`` and the BF16
        z/b/a projection of the padding-zeroed BF16 norm."""
        if self.fmt == "nvfp4":
            masked = x * rows
            return self._gemm(masked, layer, "qkv"), self.ops.gemm_epilogue(masked, layer.zba)
        # BF16 scales the output rows by keep, MXFP8 masks the quantized input's rows.
        # In BF16 the soupform zbasplit knife (rne(acc*keep) into contiguous conv/zba buffers, no
        # Mul/Cast/Slice/Slice tail) takes the projection when its mount is enabled; every miss falls
        # back to the released GEMM + split.
        if self.fmt is None:
            split = _zba_soupform(x, layer.qkv_zba, rows, self.conv_dim, x.shape[0])
            if split is not None:
                return split
        padding = dict(row_scale=rows) if self.fmt is None else dict(row_mask=rows)
        qkv_zba = self._gemm(x, layer, "qkv_zba", **padding)
        return qkv_zba[:, :self.conv_dim], qkv_zba[:, self.conv_dim:]

    def _mlp(self, x, layer):
        """gate/up with SiLU·up, then down. BF16: the down projection stays torch's GEMM (no kernel is listed
        for this shape, and its cuBLASLt fallback is slower here); MXFP8 and NVFP4: gate/up writes down's quantized
        input."""
        if self.fmt is None:
            return self.ops.linear(self.ops.gemm_glu_epilogue(x, layer.gate_up), layer.down)
        return self._gemm(self._op("gemm_glu_quantize")(x, self._weight(layer, "gate_up")), layer, "down")

    def quantize_weights(self) -> None:
        """After loading: every GEMM weight to the stage's format; NVFP4 keeps the DeltaNet z/b/a projection BF16 and
        gives each checkpoint weight its own global scale (Q·gate, K and V separately)."""
        c = self.config
        for kind, layer in zip(c.layer_types, self.layers):
            if self.fmt != "nvfp4":
                mixer = ("qkv", "o") if kind == "full_attention" else ("qkv_zba", "out")
            elif kind == "full_attention":
                kv = c.num_key_value_heads * c.head_dim
                layer.split_rows("qkv", q=2 * c.num_attention_heads * c.head_dim, k=kv, v=kv)
                mixer = ("q", "k", "v", "o")
            else:
                layer.split_rows("qkv_zba", qkv=self.conv_dim, zba=layer.qkv_zba.shape[0] - self.conv_dim)
                mixer = ("qkv", "out")
            layer.quantize(self.quant, self.ops, *mixer, "gate_up", "down")


def column_ranges(mask: torch.Tensor) -> torch.Tensor:
    """FlashMask ``(LTS, UTE)`` per key column of a boolean ``[1, 1, M, L]`` mask whose visible rows per column are
    contiguous: column ``c`` is visible to rows ``UTE <= t < LTS`` (``0, 0`` when no row sees it)."""
    vis = mask.reshape(mask.shape[-2], mask.shape[-1])
    m = vis.shape[0]
    seen = vis.any(0)
    first = vis.to(torch.int32).argmax(0)
    last = m - 1 - vis.flip(0).to(torch.int32).argmax(0)
    lts = torch.where(seen, last + 1, torch.zeros_like(last))
    ute = torch.where(seen, first, torch.zeros_like(first))
    return torch.stack((lts, ute), dim=-1).to(torch.int32)[None, None].contiguous()


def q_k_v_gate(q_gate: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    """Rows ``[q_h | gate_h]`` per head, ``k``, ``v`` → ``[q | k | v | gate]``."""
    pairs = q_gate.view(heads, 2, -1, q_gate.shape[-1])
    return torch.cat((pairs[:, 0].flatten(0, 1), k, v, pairs[:, 1].flatten(0, 1))).contiguous()


def vision_prompt_weight_mapping(vision_depth: int, layer_types, prompt_layers: int) -> dict[str, Source]:
    """Checkpoint key → tensor of ``OptimizedVision`` (``vision.*``), ``OptimizedPrefill`` (``prefill_stack.*``)
    and the token embedding, in their dtypes and orientations.

    The checkpoint is FP32. The ViT keeps the checkpoint's ``[out, in]`` weights, all BF16. The prompt stream takes
    ``[N, K]`` BF16 GEMM weights (``[qkv | z | b | a]`` and ``[q·gate | k | v]`` concatenated); its norms are FP32
    and raw, the q/k norms BF16 and raw, ``A_log`` and ``dt_bias`` their BF16 rounding kept in FP32.

    Args:
        vision_depth: ViT blocks.
        layer_types: ``"linear_attention"`` / ``"full_attention"`` per text layer.
        prompt_layers: Prompt-stream layers run (up to the last full-attention layer).

    Returns:
        ``{model parameter name: Source}``.
    """
    bf16, fp32 = torch.bfloat16, torch.float32
    lm, vis = "model.language_model", "model.visual"
    one = lambda key: Source.one(key, bf16)  # noqa: E731
    cat = lambda *keys: Source(tuple(keys), bf16, combine=lambda w: torch.cat(w, 0).contiguous())  # noqa: E731
    mapping = {
        "embed_tokens": one(f"{lm}.embed_tokens.weight"),
        "vision.patch.weight": Source((f"{vis}.patch_embed.proj.weight",), bf16,
                                      combine=lambda w: w[0].reshape(w[0].shape[0], -1)),
        "vision.patch.bias": one(f"{vis}.patch_embed.proj.bias"),
        "vision.pos_embed": one(f"{vis}.pos_embed.weight"),
    }
    for name, key in (("norm_weight", "norm.weight"), ("norm_bias", "norm.bias"), ("fc1_weight", "linear_fc1.weight"),
                      ("fc1_bias", "linear_fc1.bias"), ("fc2_weight", "linear_fc2.weight"),
                      ("fc2_bias", "linear_fc2.bias")):
        mapping[f"vision.merger.{name}"] = one(f"{vis}.merger.{key}")
    for i in range(vision_depth):
        source, target = f"{vis}.blocks.{i}", f"vision.blocks.{i}"
        for name, key in (("norm1", "norm1"), ("qkv", "attn.qkv"), ("proj", "attn.proj"), ("norm2", "norm2"),
                          ("fc1", "mlp.linear_fc1"), ("fc2", "mlp.linear_fc2")):
            for part in ("weight", "bias"):
                mapping[f"{target}.{name}_{part}"] = one(f"{source}.{key}.{part}")
    for i, kind in enumerate(layer_types[:prompt_layers]):
        source, prompt = f"{lm}.layers.{i}", f"prefill_stack.layers.{i}"
        mapping[f"{prompt}.input_norm"] = Source.one(f"{source}.input_layernorm.weight", fp32)
        mapping[f"{prompt}.post_norm"] = Source.one(f"{source}.post_attention_layernorm.weight", fp32)
        mapping[f"{prompt}.gate_up"] = one(f"{source}.mlp.gate_up_proj.weight")
        mapping[f"{prompt}.down"] = one(f"{source}.mlp.down_proj.weight")
        if kind == "full_attention":
            attn = f"{source}.self_attn"
            mapping[f"{prompt}.qkv"] = cat(f"{attn}.q_proj.weight", f"{attn}.k_proj.weight", f"{attn}.v_proj.weight")
            mapping[f"{prompt}.q_norm"] = one(f"{attn}.q_norm.weight")
            mapping[f"{prompt}.k_norm"] = one(f"{attn}.k_norm.weight")
            mapping[f"{prompt}.o"] = one(f"{attn}.o_proj.weight")
        else:
            linear = f"{source}.linear_attn"
            mapping[f"{prompt}.qkv_zba"] = cat(f"{linear}.in_proj_qkv.weight", f"{linear}.in_proj_z.weight",
                                               f"{linear}.in_proj_b.weight", f"{linear}.in_proj_a.weight")
            mapping[f"{prompt}.conv_weight"] = Source((f"{linear}.conv1d.weight",), bf16,
                                                      combine=lambda w: w[0].squeeze(1).contiguous())
            for name in ("A_log", "dt_bias"):
                mapping[f"{prompt}.{name}"] = Source((f"{linear}.{name}",), fp32,
                                                     combine=lambda w: w[0].to(bf16).to(fp32))
            mapping[f"{prompt}.norm_weight"] = one(f"{linear}.norm.weight")
            mapping[f"{prompt}.out"] = one(f"{linear}.out_proj.weight")
    return mapping
