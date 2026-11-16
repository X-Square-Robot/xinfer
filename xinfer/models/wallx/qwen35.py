"""Shared stages and weight loading for the Wall-X 2B Qwen3.5 backbone.

Wall-X 2B runs a vision transformer and a prompt stream with Gated DeltaNet and gated full-attention layers.
Action tokens read the prompt through the full-attention layers. This module loads the backbone weights and
implements vision and prefill. The model subclass provides its configuration, processors, flow head and denoiser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from xinfer.models.layers.basic import _meta_param
from xinfer.models.layers.quantization import QuantConfig, quant_config
from xinfer.models.layers.image_pack import ImagePack
from xinfer.models.layers.qwen3_5 import ActionLayer, AdaptiveRMSNorm, PrefillLayer, prompt_keep_mask
from xinfer.models.layers.vision.qwen3_5_vit import Qwen35VisionTower
from xinfer.loader import SafetensorsCheckpoint, Source, load_mapped
from xinfer.models.interface import HandoffSpec, OutputSpec, Stage, StageGraph, TensorSpec, VLAModel
from xinfer.models.processing import view_inputs
from xinfer.models.wallx.optimized import QUANT_METHODS, OptimizedPrefill, OptimizedVision
from xinfer.models.wallx.processing import pad_prompt, pixel_lookup_table
from xinfer.ops import active_ops, use_ops

DTYPE = torch.bfloat16
NUMERICS = ("optimized", "reference")
QUANT_GROUPS = ("vision", "prefill")  # stages whose GEMMs may run quantized
QUANT_PRESETS = {
    "bf16": QuantConfig(),
    "mxfp8": QuantConfig(groups={group: "mxfp8" for group in QUANT_GROUPS}),
    # the Wall-X NVFP4 recipe: amax weight scales in the ViT, MSE weight scales with a global scale in the prompt stream
    "nvfp4": QuantConfig(groups={"vision": "nvfp4", "prefill": "nvfp4_mse"}),
}


def text_inverse_frequencies(rotary_dim: int, theta: float) -> torch.Tensor:
    """``1 / theta^(2i / rotary_dim)`` in FP32, as the Qwen3.5 rotary embedding computes it."""
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.int64).to(dtype=torch.float) / rotary_dim
    return 1.0 / (theta ** exponents)


def text_rope_tables(inv_freq: torch.Tensor, positions: torch.Tensor, section):
    """FP32 ``(cos, sin)`` ``[B, L, 1, 2F]`` of interleaved M-RoPE, rounded through BF16 as the source model gives them.

    Frequency ``j`` takes the height lane if ``j % 3 == 1`` (``j < 3·section[1]``), the width lane if ``j % 3 == 2``
    (``j < 3·section[2]``), else the temporal lane.

    Args:
        inv_freq: ``[F]`` FP32 inverse frequencies.
        positions: ``[3, B, L]`` integer positions (temporal, height, width).
        section: ``(t, h, w)`` band sizes.
    """
    angles = positions.to(torch.float32)[..., None] * inv_freq.to(torch.float32)  # [3, B, L, F]
    lane = torch.zeros(inv_freq.shape[0], dtype=torch.int64, device=inv_freq.device)
    for dim in (1, 2):
        lane[dim:section[dim] * 3:3] = dim
    angles = angles.gather(0, lane.expand(1, *angles.shape[1:-1], -1)).squeeze(0)
    full = torch.cat((angles, angles), dim=-1)
    return full.cos().to(DTYPE).to(torch.float32)[:, :, None], full.sin().to(DTYPE).to(torch.float32)[:, :, None]


class WallXQwen35(VLAModel, nn.Module):
    """A Wall-X model on the Qwen3.5 backbone, served as ``vision`` → ``prefill`` → ``action`` (module docstring).

    Build it with ``from_checkpoint``; the constructor creates the modules on the "meta" device under the active
    op table, and ``from_checkpoint`` loads the weights.

    Args:
        config: The checkpoint's configuration (``config_class``).
        checkpoint: Checkpoint directory (kept for the processors).
        embedding_rows: Rows of the token embedding table in the checkpoint.
        processing: ``ProcessingConfig``, dict or ``None``, passed to the processors.
        numerics: ``"optimized"`` (the CUDA backend's kernels) or ``"reference"`` (the Wall-X model under BF16
            autocast).
        quant: Stage of the optimized composition (``"vision"``, ``"prefill"``) → the format of its GEMMs
            (``"mxfp8"``, ``"nvfp4"``, ``"nvfp4_mse"``); the other stages run BF16.

    Attributes:
        config_class: The model's configuration class, with ``load(path)``.
        stage_ops: Operators each stage calls in the reference composition; a model's ``capabilities`` reports
            them, or its ``OPTIMIZED_STAGE_OPS``, for the startup check of slow fallbacks.
        geometry: The backbone's sizes (``hidden_size``, ``layer_types``, ``vision``, ...).
        vision: The Qwen3.5 ViT (``OptimizedVision`` for the optimized composition).
        embed_tokens: Token embedding table ``[embedding_rows, hidden]`` BF16.
        flow_head: The model's flow head (``build_flow_head``).
        prefill_layers, action_layers, action_norm: The prompt and action streams (reference composition).
        prefill_stack, action: ``OptimizedPrefill`` and the model's optimized action stage (optimized composition).
        inv_freq: Text rotary inverse frequencies ``[rotary_dim / 2]`` FP32.
    """

    stage_graph = StageGraph(
        Stage("vision", "encode_images"),
        Stage("prefill", "prefill", depends_on="vision"),
        Stage("action", "denoise", depends_on="prefill"),
    )
    handoffs = (
        HandoffSpec("vision", "prefill", {"image_embeds": TensorSpec(batch_axis=0, seq_axis=0)}),
        # [full-attention layers, k/v, batch, kv_heads, prompt_len, head_dim]
        HandoffSpec("prefill", "action", {"prompt_kv": TensorSpec(batch_axis=2)}),
    )
    pad_prompt = staticmethod(pad_prompt)
    stage_ops = {
        "vision": ("layer_norm", "linear", "rotary_embedding", "varlen_attention", "gelu_tanh", "gelu"),
        "prefill": ("linear", "rms_norm", "causal_conv1d", "chunk_gated_delta_rule", "gated_rms_norm",
                    "rotary_embedding", "causal_attention", "silu_and_mul"),
        "action": ("linear", "ada_rms_norm", "rms_norm", "rotary_embedding", "masked_joint_attention",
                   "silu_and_mul", "gated_residual", "silu"),
    }
    config_class: ClassVar[type]

    def __init__(self, config, checkpoint=None, *, embedding_rows: int, processing=None,
                 numerics: str = "optimized", quant=None):
        nn.Module.__init__(self)
        quant = dict(quant or {})
        if numerics not in NUMERICS:
            raise ValueError(f"numerics must be one of {NUMERICS}, got {numerics!r}")
        if quant and numerics != "optimized":
            raise ValueError("quantized GEMMs need numerics='optimized'")
        self.config, self.checkpoint, self.processing, self.numerics = config, checkpoint, processing, numerics
        g = self.geometry
        self.image_pack = ImagePack(pixel_lookup_table(*self.image_normalization()), patch_size=g.vision.patch_size,
                                    temporal_patch_size=g.vision.temporal_patch_size,
                                    merge_size=g.vision.spatial_merge_size)
        self.embed_tokens = _meta_param(embedding_rows, g.hidden_size)
        self.flow_head = self.build_flow_head()
        self.register_buffer("inv_freq", text_inverse_frequencies(g.rotary_dim, g.rope_theta), persistent=False)
        self._op_table = active_ops()
        if numerics == "optimized":
            self.vision = OptimizedVision(g.vision, quant=quant.get("vision"))
            self.prefill_stack = OptimizedPrefill(g, quant=quant.get("prefill"))
            self.action = self.build_optimized_action()
            return
        self.vision = Qwen35VisionTower(g.vision)
        attention = (g.num_attention_heads, g.num_key_value_heads, g.head_dim)
        linear = [t == "linear_attention" for t in g.layer_types]
        self.prefill_layers = nn.ModuleList(
            PrefillLayer(lin, g.hidden_size, g.intermediate_size, attention, g.linear_heads, g.rms_norm_eps,
                         group="prefill") for lin in linear)
        self.action_layers = nn.ModuleList(
            ActionLayer(lin, g.action_hidden_size, g.action_intermediate_size, attention, g.adarms_cond_dim,
                        g.rms_norm_eps, group="action_expert") for lin in linear)
        self.action_norm = AdaptiveRMSNorm(g.action_hidden_size, g.adarms_cond_dim, g.rms_norm_eps,
                                           group="action_expert")

    # ---------------------------------------------------- the model's parts
    @property
    def geometry(self) -> Any:
        """The backbone's sizes: the configuration itself or a part of it."""
        raise NotImplementedError

    def image_normalization(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """``(mean, std)`` of the checkpoint's image processor."""
        raise NotImplementedError

    def build_flow_head(self) -> nn.Module:
        """The flow head, built on the "meta" device under the active op table."""
        raise NotImplementedError

    def build_optimized_action(self) -> nn.Module:
        """The optimized composition's action stage (it uses ``self.flow_head``)."""
        raise NotImplementedError

    def checkpoint_mapping(self) -> dict[str, Source]:
        """Checkpoint key → model tensor of this composition (``numerics``)."""
        raise NotImplementedError

    def action_outputs(self, batch) -> dict[str, OutputSpec]:
        """The action stage's output buffers for ``batch``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ load
    @classmethod
    def from_checkpoint(cls, path, *, device, ops=None, processing=None, numerics="optimized", quantization=None,
                        **_ignored):
        """Load a checkpoint directory (configuration, ``model.safetensors``).

        Args:
            path: Checkpoint directory.
            device: Target device.
            ops: Not read here: the layers bind the op table active (``xinfer.ops.use_ops``) when the model is
                built.
            processing: ``ProcessingConfig``, dict or ``None``, kept for the processors.
            numerics: ``"optimized"`` or ``"reference"``.
            quantization: ``None`` or ``"bf16"``; ``"mxfp8"`` (the GEMMs of the ViT blocks and the prompt stream in
                MXFP8, the action stage BF16; their weights are quantized after loading); ``"nvfp4"`` (the GEMMs of the
                ViT blocks and merger in NVFP4 with amax weight scales, those of the prompt stream except the DeltaNet
                z/b/a projection with MSE weight scales, the action stage BF16); or a ``QuantConfig`` or dict mapping
                the groups ``"vision"`` and ``"prefill"`` to ``"bf16"``, ``"mxfp8"``, ``"nvfp4"`` or ``"nvfp4_mse"``.

        Returns:
            The model on ``device`` in eval mode, with ``load_report`` set.

        Raises:
            ValueError: An unknown ``numerics`` or another quantization.
        """
        quant = quant_config(quantization, QUANT_PRESETS)
        if (quant.default != "bf16" or quant.options or not set(quant.groups).issubset(QUANT_GROUPS)
                or not set(quant.groups.values()).issubset({"bf16", *QUANT_METHODS})):
            raise ValueError(f"{cls.__name__} runs in BF16 with optional {'/'.join(QUANT_METHODS)} GEMMs in "
                             f"{QUANT_GROUPS}; quantization {quantization!r} is not supported")
        stages = {group: quant.method_for(group) for group in QUANT_GROUPS if quant.method_for(group) != "bf16"}
        config = cls.config_class.load(path)
        checkpoint = SafetensorsCheckpoint(path)
        rows = checkpoint.shape("model.language_model.embed_tokens.weight")[0]
        model = cls(config, Path(path), embedding_rows=rows, processing=processing, numerics=numerics, quant=stages)
        model.load_report = load_mapped(model, model.checkpoint_mapping(), checkpoint, device=device)
        model = model.to(device).eval()
        if numerics == "optimized":
            with use_ops(model._op_table):
                model.action.prepare()
                for stage in (model.vision, model.prefill_stack):
                    if stage.quant is not None:
                        stage.quantize_weights()
        return model

    # ------------------------------------------------------- model interface
    def stage_outputs(self, stage, batch):
        """Declare the output buffers of ``stage`` for ``batch``.

        Returns:
            ``vision``: ``image_embeds`` ``[N_tokens, hidden]`` BF16; ``prefill``: ``prompt_kv``
            ``[full-attention layers, 2, B, kv_heads, P, head_dim]`` BF16; ``action``: ``action_outputs``.

        Raises:
            KeyError: Unknown stage.
        """
        x, g = batch.inputs, self.geometry
        if stage == "vision":
            p = g.vision.patch_size * g.vision.spatial_merge_size
            tokens = sum((view.shape[0] // p) * (view.shape[1] // p) for view in view_inputs(x))
            return {"image_embeds": OutputSpec((tokens, g.hidden_size), DTYPE)}
        if stage == "prefill":
            shape = (len(g.full_attention_layers), 2, batch.batch_size, g.num_key_value_heads, int(x["prompt_len"]),
                     g.head_dim)
            return {"prompt_kv": OutputSpec(shape, DTYPE)}
        if stage == "action":
            return self.action_outputs(batch)
        raise KeyError(stage)

    # ----------------------------------------------------------------- stages
    def encode_images(self, batch):
        """Vision stage: pack the views into patch rows, then the ViT and patch merger over all of them.

        Args:
            batch: ``StageBatch`` of one request. Reads the resized views ``image_<i>`` ``[h_i, w_i, 3]`` uint8 in
                prompt order (packed by ``image_pack`` into ``[N_patches, C·T·P²]`` FP32 rows, the pixel values
                of the checkpoint's image processor) and the host-side layout: ``vision_positions``
                ``[N_patches, 2]`` int64, ``vision_pos_index`` ``[4, N_patches]`` int64, ``vision_pos_weight``
                ``[4, N_patches]`` FP32 and the host list ``vision_cu_seqlens``.

        Writes:
            ``batch.out["image_embeds"]`` ``[N_patches / merge², hidden]`` BF16, one row per ``<|image_pad|>``.
        """
        x = batch.inputs
        views = view_inputs(x)
        if self.numerics == "optimized":
            patch = self.geometry.vision.patch_size
            grids = [(1, view.shape[0] // patch, view.shape[1] // patch) for view in views]
            batch.out["image_embeds"].copy_(self.vision(self.image_pack(views), grids, x["vision_cu_seqlens"]))
            return
        layout = dict(positions=x["vision_positions"], pos_index=x["vision_pos_index"],
                      pos_weight=x["vision_pos_weight"], cu_seqlens=x["vision_cu_seqlens"])
        batch.out["image_embeds"].copy_(self.vision(self.image_pack(views), layout, DTYPE))

    def prefill(self, batch):
        """Prefill stage: the prompt stream over the prompt; writes the K/V of every full-attention layer.

        The prompt is padded to ``prompt_len`` tokens after its ``prompt_valid`` real tokens. Image token
        embeddings replace the ``<|image_pad|>`` rows; padding rows are zeroed before every DeltaNet scan, as the
        source model does. Layers after the last full-attention layer are not run.

        Args:
            batch: ``StageBatch`` of one request. Reads ``prompt_len`` (host int ``P``), ``input_ids`` ``[1, P]``
                int64, ``position_ids`` ``[3, 1, P]`` int64 (interleaved M-RoPE t/h/w positions),
                ``prompt_valid`` ``[1]`` int64, ``image_token_index`` ``[N_tokens]`` int64 and ``image_embeds``
                ``[N_tokens, hidden]`` BF16.

        Writes:
            ``batch.out["prompt_kv"]`` ``[full-attention layers, 2, 1, kv_heads, P, head_dim]`` BF16: normed,
            rotated K (index 0) and V (index 1).
        """
        x, g = batch.inputs, self.geometry
        p = int(x["prompt_len"])
        h = self.embed_tokens[x["input_ids"][:, :p]]
        h[0, x["image_token_index"]] = x["image_embeds"].to(h.dtype)
        cos, sin = text_rope_tables(self.inv_freq, x["position_ids"][:, :, :p], g.mrope_section)
        kv = batch.out["prompt_kv"]
        keep = prompt_keep_mask(x["prompt_valid"], p, dtype=h.dtype)
        if self.numerics == "optimized":
            half = g.rotary_dim // 2
            self.prefill_stack(h, cos[:, :, 0, :half].contiguous(), sin[:, :, 0, :half].contiguous(), kv,
                               keep.reshape(-1).to(torch.float32))
            return
        full = g.full_attention_layers
        for i, layer in enumerate(self.prefill_layers[:full[-1] + 1]):
            h, k, v = layer(h, cos, sin, keep, kv_only=i == full[-1])
            if k is not None:
                slot = full.index(i)
                kv[slot, 0].copy_(k.transpose(1, 2))
                kv[slot, 1].copy_(v.transpose(1, 2))


def reference_weight_mapping(model: WallXQwen35) -> dict[str, Source]:
    """Checkpoint key → tensor of the backbone in the reference composition, with its inference dtype.

    The ViT, the token embedding, both streams and the action stream's final norm; the flow head is the model's.
    The ViT patch embedding ``[hidden, C, T, P, P]`` loads as ``[hidden, C·T·P²]``. The prompt-stream layer norms
    and the ViT position table load in FP32, everything else in BF16 (the action stream's adaptive-norm
    projections run in BF16 under the source's autocast). Zero-centered RMSNorm weights load as ``1 + w``
    (prompt norms from FP32, q/k norms from their BF16 rounding).

    Args:
        model: The model shell (built on the "meta" device with ``numerics="reference"``).

    Returns:
        ``{model parameter name: Source}``.
    """
    bf16, fp32 = torch.bfloat16, torch.float32
    g = model.geometry
    plus_one_fp32 = lambda t: t[0].to(fp32) + 1.0  # noqa: E731
    plus_one_bf16 = lambda t: t[0].to(bf16).to(fp32) + 1.0  # noqa: E731
    lm = "model.language_model"
    mapping = {
        "vision.patch_embed.weight": Source(("model.visual.patch_embed.proj.weight",), bf16,
                                            combine=lambda t: t[0].flatten(1)),
        "vision.patch_embed.bias": Source.one("model.visual.patch_embed.proj.bias", bf16),
        "vision.pos_embed": Source.one("model.visual.pos_embed.weight", fp32),
        "embed_tokens": Source.one(f"{lm}.embed_tokens.weight", bf16),
        "action_norm.dense.weight": Source.one(f"{lm}.norm_action.dense.weight", bf16),
        "action_norm.dense.bias": Source.one(f"{lm}.norm_action.dense.bias", bf16),
    }
    for name in ("merger.norm", "merger.linear_fc1", "merger.linear_fc2"):
        target = "vision." + name.replace("linear_", "")
        for part in ("weight", "bias"):
            mapping[f"{target}.{part}"] = Source.one(f"model.visual.{name}.{part}", bf16)
    for i in range(g.vision.depth):
        source, target = f"model.visual.blocks.{i}", f"vision.blocks.{i}"
        for checkpoint_name, module_name in (("norm1", "norm1"), ("norm2", "norm2"), ("attn.qkv", "qkv"),
                                             ("attn.proj", "proj"), ("mlp.linear_fc1", "fc1"),
                                             ("mlp.linear_fc2", "fc2")):
            for part in ("weight", "bias"):
                mapping[f"{target}.{module_name}.{part}"] = Source.one(f"{source}.{checkpoint_name}.{part}", bf16)
    for i, layer in enumerate(model.prefill_layers):
        source, target, action = f"{lm}.layers.{i}", f"prefill_layers.{i}", f"action_layers.{i}"
        mapping[f"{target}.input_norm.weight"] = Source((f"{source}.input_layernorm.weight",), fp32,
                                                        combine=plus_one_fp32)
        mapping[f"{target}.post_norm.weight"] = Source((f"{source}.post_attention_layernorm.weight",), fp32,
                                                       combine=plus_one_fp32)
        mapping[f"{target}.mlp.gate_up.weight"] = Source.one(f"{source}.mlp.gate_up_proj.weight", bf16)
        mapping[f"{target}.mlp.down.weight"] = Source.one(f"{source}.mlp.down_proj.weight", bf16)
        mapping[f"{action}.mlp.gate_up.weight"] = Source.one(f"{source}.mlp_action_expert.gate_up_proj.weight", bf16)
        mapping[f"{action}.mlp.down.weight"] = Source.one(f"{source}.mlp_action_expert.down_proj.weight", bf16)
        for norm, checkpoint_norm in (("input_norm", "input_layernorm_action"),
                                      ("post_norm", "post_attention_layernorm_action")):
            for part in ("weight", "bias"):
                mapping[f"{action}.{norm}.dense.{part}"] = Source.one(f"{source}.{checkpoint_norm}.dense.{part}",
                                                                      bf16)
        if layer.linear:
            attn = f"{source}.linear_attn"
            for checkpoint_name, module_name in (("in_proj_qkv", "in_qkv"), ("in_proj_z", "in_z"),
                                                 ("in_proj_b", "in_b"), ("in_proj_a", "in_a"),
                                                 ("out_proj", "out_proj")):
                mapping[f"{target}.mixer.{module_name}.weight"] = Source.one(f"{attn}.{checkpoint_name}.weight", bf16)
            mapping[f"{target}.mixer.conv_weight"] = Source.one(f"{attn}.conv1d.weight", bf16)
            mapping[f"{target}.mixer.A_log"] = Source.one(f"{attn}.A_log", bf16)
            mapping[f"{target}.mixer.dt_bias"] = Source.one(f"{attn}.dt_bias", bf16)
            mapping[f"{target}.mixer.norm_weight"] = Source.one(f"{attn}.norm.weight", bf16)
        else:
            attn = f"{source}.self_attn"
            for stream, module, suffix in ((target, "mixer", ""), (action, "attn", "_action")):
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    mapping[f"{stream}.{module}.{proj}.weight"] = Source.one(f"{attn}.{proj}{suffix}.weight", bf16)
                for norm in ("q_norm", "k_norm"):
                    mapping[f"{stream}.{module}.{norm}.weight"] = Source((f"{attn}.{norm}{suffix}.weight",), fp32,
                                                                         combine=plus_one_bf16)
    return mapping
