"""Wall-OSS: Qwen2.5-VL backbone with a flow-matching action expert.

The checkpoint is a two-expert mixture of transformers. Expert 0 (the
Qwen2.5 language model, width 2048) processes the prompt; expert 1 (width
1024) processes the action tokens. Both share one attention per layer, in
which prompt tokens never attend to action tokens. That makes the prompt's
per-layer K/V independent of the actions and of the denoising time, so the
model splits exactly into three stages:

- ``vision``: pixels → image token embeddings (Qwen2.5-VL ViT);
- ``prefill``: the prompt → per-layer K/V (expert 0);
- ``action``: N Euler steps of expert 1 attending to the prompt K/V.

The ``numerics`` model option selects the composition, both BF16. ``"optimized"`` (the default) runs the
steps on the CUDA backend's kernels (``optimized.py``). ``"reference"`` follows the Wall-X eager model
(Wall-X 02c5351d): BF16 weights and activations, FP32 RMSNorm weights in the
language stacks, FP32 action head and ODE state. RTC requests use the served attention
semantics (pinned action tokens never attend to denoising tokens, at every
step), which the production services implement; see
docs/models/wall_oss.md.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from xinfer.models.layers.basic import Linear, RMSNorm
from xinfer.models.layers.decoder import DecoderLayer
from xinfer.models.layers.image_pack import ImagePack
from xinfer.models.layers.rotary import MRope, inverse_frequencies, mrope_cos_sin
from xinfer.models.layers.vision.qwen2_5_vit import VisionTower, VisionTowerConfig
from xinfer.loader import SafetensorsCheckpoint, Source, load_mapped
from xinfer.models.interface import (
    Capabilities,
    CameraSpec,
    HandoffSpec,
    IOSchema,
    OutputSpec,
    Stage,
    StageGraph,
    TensorSpec,
    VLAModel,
)
from xinfer.models.processing import view_inputs
from xinfer.models.wall_oss.config import WallOSSConfig
from xinfer.models.wall_oss.optimized import (OPTIMIZED_STAGE_OPS, OptimizedAction, OptimizedPrefill,
                                              OptimizedVision, compact_rope, optimized_weight_mapping)
from xinfer.models.wall_oss.processing import pad_prompt
from xinfer.ops import active_ops

DTYPE = torch.bfloat16

NUMERICS = ("optimized", "reference")
FLOW_TIME_SCALE = 0.999  # Wall-X noise scheduler ``s``: the last Euler step ends at t = 0.999
TIME_EMBED_MAX_PERIOD = 10000.0


def vision_config(config: WallOSSConfig) -> VisionTowerConfig:
    """Build the ViT configuration from the checkpoint's ``vision_config``.

    The MLP width is rounded up to a multiple of 128, as the checkpoint stores it (Wall-OSS: 3420 → 3456),
    and ``in_features`` is the flattened patch size ``in_channels · temporal_patch_size · patch_size²``
    (Wall-OSS: 1176).

    Args:
        config: The loaded Wall-OSS configuration.

    Returns:
        The ``VisionTowerConfig`` of the model's ``VisionTower`` (also used by the input processor for
        ``vision_layout``).
    """
    vision = config.vision
    align = 128  # Wall-X pads the ViT MLP width to a multiple of 128 (stored padded)
    return VisionTowerConfig(
        depth=vision.depth,
        hidden_size=vision.hidden_size,
        mlp_size=(vision.intermediate_size + align - 1) // align * align,
        num_heads=vision.num_heads,
        in_features=vision.in_channels * vision.temporal_patch_size * vision.patch_size ** 2,
        spatial_merge_size=vision.spatial_merge_size,
        window_size=vision.window_size,
        patch_size=vision.patch_size,
        out_hidden_size=vision.out_hidden_size,
        fullatt_block_indexes=vision.fullatt_block_indexes,
    )


class ActionHead(nn.Module):
    """Wall-X action processor (FP32): noisy action + time → expert-1 input; readout.

    All four projections are bias-free ``Linear`` layers of group ``"action_head"`` whose weights are loaded
    in FP32 (``weight_mapping``).

    Args:
        action_dim: Action dimension ``D`` (Wall-OSS: 26).
        hidden: Width of expert 1 (Wall-OSS: 1024).

    Attributes:
        w1: ``[action, dof mask]`` ``2·D`` → ``hidden``.
        w2: ``[w1 output, time embedding]`` ``2·hidden`` → ``hidden``.
        w3: ``hidden`` → ``hidden``, applied after SiLU; its output is the expert-1 input.
        proj_back: Readout ``hidden`` → ``D`` (the velocity), applied by ``WallOSS.denoise`` after the final
            norm.
    """

    def __init__(self, action_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.w1 = Linear(2 * action_dim, hidden, group="action_head")  # [action, dof mask]
        self.w2 = Linear(2 * hidden, hidden, group="action_head")  # [w1 output, time embedding]
        self.w3 = Linear(hidden, hidden, group="action_head")
        self.proj_back = Linear(hidden, action_dim, group="action_head")
        self._silu = active_ops().silu

    def time_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding of the flow time.

        Frequencies are ``exp(-i · ln(10000) / (hidden/2 - 1))`` for ``i < hidden/2``, in FP32.

        Args:
            tau: Flow time per action token, FP32 (``denoise`` passes ``[B, H]``).

        Returns:
            ``[..., hidden]`` FP32: ``sin`` of the angles in the first half, ``cos`` in the second.
        """
        half = self.hidden // 2
        scale = math.log(TIME_EMBED_MAX_PERIOD) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=tau.device, dtype=torch.float32) * -scale)
        angles = tau[..., None] * freqs
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def embed(self, x, tau, dof_mask):
        """Expert-1 input ``w3(silu(w2([w1([x, dof_mask]), time_embedding(tau)])))``.

        Args:
            x: Noisy normalized actions ``[B, H, D]`` FP32.
            tau: Per-token flow time ``[B, H]`` FP32 (1 for pinned RTC tokens).
            dof_mask: DoF mask ``[B, H, D]`` FP32 (1 = commanded dimension).

        Returns:
            ``[B, H, hidden]`` FP32; the caller casts it to BF16.
        """
        action_features = self.w1(torch.cat([x, dof_mask], dim=-1))
        joint_features = self.w2(torch.cat([action_features, self.time_embedding(tau)], dim=-1))
        return self.w3(self._silu(joint_features))


class WallOSS(VLAModel, nn.Module):
    """Wall-OSS served as three stages: ``vision`` → ``prefill`` → ``action`` (see the module docstring).

    Build it with ``from_checkpoint``; the constructor creates the modules on the "meta" device under the
    active op table, and ``from_checkpoint`` loads the weights.

    Args:
        config: The checkpoint's configuration.
        checkpoint: Checkpoint directory (kept for ``input_processor``).
        embedding_rows: Rows of ``model.embed_tokens.weight`` in the checkpoint (Wall-OSS: 155765).
        processing: ``ProcessingConfig``, dict or ``None``, passed to the processors and ``io_schema``.
        numerics: ``"optimized"`` or ``"reference"`` (see the module docstring).

    Attributes:
        stage_graph: ``vision`` (``encode_images``) → ``prefill`` (``prefill``) → ``action`` (``denoise``).
        handoffs: ``image_embeds`` (vision → prefill, packed along rows) and ``prompt_kv`` (prefill → action,
            batch axis 2, ragged sequence axis 4).
        batch_axes: ``position_ids`` batches along axis 1; other prepared inputs along axis 0.
        stage_ops: Operators each stage calls in the reference composition (``OPTIMIZED_STAGE_OPS`` for the
            optimized one); ``capabilities`` reports them for the startup check of slow fallbacks.
        vision: The Qwen2.5-VL ``VisionTower`` (``OptimizedVision`` for the optimized composition).
        embed_tokens: Token embedding table ``[embedding_rows, hidden_size]`` BF16.
        prefill_layers: Expert-0 ``DecoderLayer`` stack (group ``"prefill"``; reference composition).
        action_layers: Expert-1 ``DecoderLayer`` stack (group ``"action_expert"``; reference composition).
        action_norm: Final expert-1 RMSNorm (FP32 weight; reference composition).
        prefill_stack: ``OptimizedPrefill`` (optimized composition).
        action: ``OptimizedAction`` (optimized composition).
        action_head: The FP32 ``ActionHead``.
        inv_freq: RoPE inverse frequencies ``[head_dim / 2]`` FP32 of the prefill and action stages.
        action_min: Per-step action normalizer minimum ``[H, D]`` FP32 (set by ``from_checkpoint``).
        action_delta: Per-step action normalizer range ``[H, D]`` FP32 (set by ``from_checkpoint``).
    """

    stage_graph = StageGraph(
        Stage("vision", "encode_images"),
        Stage("prefill", "prefill", depends_on="vision"),
        Stage("action", "denoise", depends_on="prefill"),
    )
    handoffs = (
        # Image tokens of all views, packed along rows.
        HandoffSpec("vision", "prefill", {"image_embeds": TensorSpec(batch_axis=0, seq_axis=0)}),
        # [layers, k/v, batch, kv_heads, prompt_len, head_dim]; requests with different
        # prompt lengths are batched left-padded, with prompt_kv_lengths per row. A padded
        # prompt (ProcessingConfig.prompt_bucket) is valid in its first prompt_valid columns.
        HandoffSpec("prefill", "action", {"prompt_kv": TensorSpec(batch_axis=2, seq_axis=4)}),
    )
    batch_axes = {"position_ids": 1}
    pad_prompt = staticmethod(pad_prompt)
    # Operators each stage calls in the reference composition.
    stage_ops = {
        "vision": ("linear", "rms_norm", "silu_and_mul", "gelu", "rotary_embedding", "varlen_attention"),
        "prefill": ("linear", "rms_norm", "silu_and_mul", "rotary_embedding", "causal_attention"),
        "action": ("linear", "rms_norm", "silu_and_mul", "rotary_embedding", "action_attention", "silu"),
    }

    def __init__(self, config: WallOSSConfig, checkpoint: Path, *, embedding_rows: int, processing=None,
                 numerics: str = "optimized"):
        nn.Module.__init__(self)
        if numerics not in NUMERICS:
            raise ValueError(f"numerics must be one of {NUMERICS}, got {numerics!r}")
        self.numerics = numerics
        self.config = config
        self.checkpoint = Path(checkpoint)
        from xinfer.models.wall_oss.image_processing import pixel_lookup_table
        from xinfer.models.wall_oss.processing import processing_config

        self.processing = p = processing_config(processing)
        self.image_pack = ImagePack(pixel_lookup_table(p.image_mean, p.image_std), patch_size=p.patch_size,
                                    temporal_patch_size=p.temporal_patch_size, merge_size=p.merge_size)
        self.horizon, self.action_dim = config.action_horizon, config.action_dim
        # Rows = base vocabulary + tokens added at training time (Wall-OSS: 155765,
        # not config.json's vocab_size); taken from the checkpoint.
        self.embed_tokens = nn.Parameter(torch.empty(embedding_rows, config.hidden_size, device="meta"),
                                         requires_grad=False)
        self.register_buffer("inv_freq", inverse_frequencies(config.head_dim, config.rope_theta), persistent=False)
        self.register_buffer("action_min", torch.empty(0), persistent=False)
        self.register_buffer("action_delta", torch.empty(0), persistent=False)
        if numerics == "optimized":
            self.vision = OptimizedVision(vision_config(config))
            self.prefill_stack = OptimizedPrefill(config)
            self.action_head = ActionHead(self.action_dim, config.experts[1].hidden_size)
            self.action = OptimizedAction(config, self.action_head)
            return
        self.vision = VisionTower(vision_config(config))
        self.rope = MRope(config.mrope_section)

        def stack(expert, group):
            return nn.ModuleList(
                DecoderLayer(expert.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                             config.head_dim, expert.intermediate_size, config.rms_norm_eps, group=group)
                for _ in range(config.num_hidden_layers)
            )

        self.prefill_layers = stack(config.experts[0], "prefill")
        self.action_rope = MRope(config.mrope_section)
        self.action_layers = stack(config.experts[1], "action_expert")
        self.action_norm = RMSNorm(config.experts[1].hidden_size, config.rms_norm_eps)
        self.action_head = ActionHead(self.action_dim, config.experts[1].hidden_size)

    # ------------------------------------------------------------------ load
    @classmethod
    def from_checkpoint(cls, path, *, device, ops=None, processing=None, norm_key="ex_normal", quantization=None,
                        numerics="optimized"):
        """Load a checkpoint: the weights through the composition's weight mapping and the action normalizer.

        Args:
            path: Checkpoint directory with ``config.json``, ``config.yml``, the safetensors weights and
                ``normalizer_action.pth``.
            device: Target device, e.g. ``"cuda"`` or ``"cpu"``.
            ops: The caller's op table. Not read here: the layers bind the table active through
                ``xinfer.ops.use_ops`` when the model is built, so callers build inside that context.
            processing: ``ProcessingConfig``, dict or ``None`` for the input processor.
            norm_key: Key suffix of the action statistics (``min.<key>``, ``delta.<key>``).
            quantization: ``None`` or ``"bf16"`` (the only precision).
            numerics: ``"optimized"`` (the CUDA backend's kernels, ``xinfer/models/wall_oss/optimized.py``) or
                ``"reference"`` (the Wall-X eager model).

        Returns:
            The model on ``device`` in eval mode, with ``load_report`` set.

        Raises:
            ValueError: Another quantization or an unknown ``numerics``, or an action normalizer that is not
                ``[action_horizon, action_dim]``.
        """
        if quantization not in (None, "bf16"):
            raise ValueError(f"Wall-OSS runs in BF16; quantization {quantization!r} is not supported")
        config = WallOSSConfig.load(path)
        checkpoint = SafetensorsCheckpoint(path)
        rows = checkpoint.shape("model.embed_tokens.weight")[0]
        model = cls(config, Path(path), embedding_rows=rows, processing=processing, numerics=numerics)
        mapping = (optimized_weight_mapping if numerics == "optimized" else weight_mapping)(model)
        model.load_report = load_mapped(model, mapping, checkpoint, device=device)
        stats = torch.load(Path(path) / "normalizer_action.pth", map_location="cpu", weights_only=True)
        model.action_min = stats[f"min.{norm_key}"].to(torch.float32)
        model.action_delta = stats[f"delta.{norm_key}"].to(torch.float32)
        if model.action_min.shape != (config.action_horizon, config.action_dim):
            raise ValueError("action normalizer does not match the action horizon and dimension")
        return model.to(device).eval()

    # ------------------------------------------------------- model interface
    def capabilities(self) -> Capabilities:
        """What this built model supports.

        Returns:
            ``Capabilities`` with batch sizes ``vision`` (1,), ``prefill`` (1,) and ``action`` (1, 2, 4, 8) in the
            reference composition, (1,) in the optimized one; ``num_steps=5``; RTC with delays
            ``1 .. config.rtc_max_delay``; and ``stage_ops``.
        """
        optimized = self.numerics == "optimized"
        stage_ops = OPTIMIZED_STAGE_OPS if optimized else self.stage_ops
        return Capabilities(
            # Vision and prefill are written for one request (packed image rows,
            # per-request prompt length); the reference action stage takes batches of
            # requests, of equal or (left-padded) different prompt lengths.
            batch_sizes={"vision": (1,), "prefill": (1,), "action": (1,) if optimized else (1, 2, 4, 8)},
            num_steps=5,
            rtc=True,
            rtc_delay_range=(1, self.config.rtc_max_delay),
            stage_ops=stage_ops,
        )

    def io_schema(self) -> IOSchema:
        """Wire-level request layout of the desktop robot protocol.

        Returns:
            ``IOSchema`` with one camera per ``ProcessingConfig.cameras`` wire key, state and action keys
            ``follow1_pos`` / ``follow2_pos``, the action dimension and horizon, and the optional state keys
            ``velocity_decomposed``, ``lift`` and ``head_pos`` in ``extra``.
        """
        from xinfer.models.wall_oss.processing import processing_config

        config = processing_config(self.processing)
        return IOSchema(
            cameras=tuple(CameraSpec(camera.wire_key) for camera in config.cameras),
            state_keys=("follow1_pos", "follow2_pos"),
            action_keys=("follow1_pos", "follow2_pos"),
            action_dim=self.action_dim,
            action_horizon=self.horizon,
            extra={"optional_state_keys": ("velocity_decomposed", "lift", "head_pos")},
        )

    @classmethod
    def build_processors(cls, checkpoint, processing=None, **_ignored):
        """Build the CPU processors and the wire metadata frame without loading weights.

        Args:
            checkpoint: Checkpoint directory (configs, tokenizer, normalizers).
            processing: ``ProcessingConfig``, dict or ``None``.
            **_ignored: Other model options (e.g. ``quantization``), unused here.

        Returns:
            ``(WallOSSInputProcessor, WallOSSOutputProcessor, wire_metadata(config))``.
        """
        from xinfer.models.wall_oss.processing import (WallOSSInputProcessor, WallOSSOutputProcessor,
                                                      wire_metadata)

        config = WallOSSConfig.load(checkpoint)
        return (WallOSSInputProcessor(checkpoint, config, processing), WallOSSOutputProcessor(config),
                wire_metadata(config))

    def shape_buckets(self):
        from xinfer.models.wall_oss.processing import processing_config

        return {"prompt": processing_config(self.processing).bucket_spec()}

    # ----------------------------------------------------------------- stages
    def stage_outputs(self, stage, batch):
        """Declare the output buffers of ``stage`` for ``batch`` (``batch.out`` is still empty).

        Args:
            stage: ``"vision"``, ``"prefill"`` or ``"action"``.
            batch: The ``StageBatch`` about to run; ``vision`` reads the shapes of the ``image_<i>`` views and
                ``prefill`` the host value ``prompt_len``.

        Returns:
            ``{name: OutputSpec}``:

            - ``vision``: ``image_embeds`` ``[N_patches / merge², hidden_size]`` BF16 (Wall-OSS:
              ``[N_tokens, 2048]``);
            - ``prefill``: ``prompt_kv`` ``[layers, 2, B, kv_heads, P, head_dim]`` BF16 (Wall-OSS:
              ``[36, 2, 1, 2, P, 128]``);
            - ``action``: ``predict_action`` ``[B, H, D]`` FP32 (Wall-OSS: ``[B, 16, 26]``).

        Raises:
            KeyError: Unknown stage.
        """
        x, config = batch.inputs, self.config
        if stage == "vision":
            p = self.processing.patch_size
            patches = sum((view.shape[0] // p) * (view.shape[1] // p) for view in view_inputs(x))
            tokens = patches // config.vision.spatial_merge_size ** 2
            return {"image_embeds": OutputSpec((tokens, config.hidden_size), DTYPE)}
        if stage == "prefill":
            shape = (config.num_hidden_layers, 2, batch.batch_size, config.num_key_value_heads, int(x["prompt_len"]),
                     config.head_dim)
            return {"prompt_kv": OutputSpec(shape, DTYPE)}
        if stage == "action":
            return {"predict_action": OutputSpec((batch.batch_size, self.horizon, self.action_dim), torch.float32)}
        raise KeyError(stage)

    def encode_images(self, batch):
        """Vision stage: pack the camera views into patch rows, then run the ViT and patch merger over them.

        Args:
            batch: ``StageBatch`` of one request. Reads the views ``image_<i>`` ``[h_i, w_i, 3]`` uint8 in prompt
                order, packed by ``image_pack`` into ``[N_patches, in_features]`` FP32 rows (Wall-OSS
                ``in_features`` = 1176, cast to BF16 by the tower), and the host-computed ``vision_layout``:
                ``vision_positions``
                ``[N_patches, 2]`` int64, ``vision_window_index`` and ``vision_reverse_index``
                ``[N_patches / merge²]`` int64, and the host lists ``vision_cu_window`` / ``vision_cu_full``
                (cumulative patch bounds of the windowed / full-attention segments).

        Writes:
            ``batch.out["image_embeds"]`` ``[N_patches / merge², hidden_size]`` BF16 (Wall-OSS:
            ``[N_tokens, 2048]``): one row per ``<|image_pad|>`` token, in prompt order.
        """
        x = batch.inputs
        layout = dict(
            positions=x["vision_positions"],
            window_index=x["vision_window_index"],
            reverse_index=x["vision_reverse_index"],
            cu_window=x["vision_cu_window"],
            cu_full=x["vision_cu_full"],
        )
        if self.numerics == "optimized":
            batch.out["image_embeds"].copy_(self.vision(self.image_pack(view_inputs(x)), layout))
            return
        batch.out["image_embeds"].copy_(self.vision(self.image_pack(view_inputs(x)), layout, DTYPE))

    def prefill(self, batch):
        """Prefill stage: run expert 0 over the prompt and write every layer's K/V.

        The prompt is ``input_ids[:, :P]`` (everything before the action tokens). Image token embeddings
        replace the ``<|image_pad|>`` rows of the token embeddings, M-RoPE tables come from
        ``inv_freq`` in FP32, and the layers run causal self-attention. The last layer only computes
        its K/V.

        Args:
            batch: ``StageBatch`` of one request. Reads ``prompt_len`` (host int ``P``), ``input_ids``
                ``[1, P + H]`` int64, ``position_ids`` ``[3, 1, P + H]`` int64 (M-RoPE t/h/w positions),
                ``image_token_index`` ``[N_tokens]`` int64 (prompt positions of the image tokens) and
                ``image_embeds`` ``[N_tokens, hidden_size]`` BF16 from the vision stage.

        Writes:
            ``batch.out["prompt_kv"]`` ``[layers, 2, 1, kv_heads, P, head_dim]`` BF16 (Wall-OSS:
            ``[36, 2, 1, 2, P, 128]``): post-RoPE K (index 0) and V (index 1) of every layer.
        """
        x = batch.inputs
        p = int(x["prompt_len"])
        token_ids = x["input_ids"][:, :p]
        h = self.embed_tokens[token_ids]
        h[0, x["image_token_index"]] = x["image_embeds"].to(h.dtype)
        if self.numerics == "optimized":
            cos, sin = compact_rope(self.inv_freq, x["position_ids"][:, :, :p], self.config.mrope_section)
            self.prefill_stack(h, cos, sin, batch.out["prompt_kv"])
            return
        cos, sin = mrope_cos_sin(self.inv_freq, x["position_ids"][:, :, :p], torch.float32)
        rope = self.rope.tables(cos, sin, DTYPE)
        kv = batch.out["prompt_kv"]  # [layers, k/v, batch, kv_heads, P, head_dim]
        last = len(self.prefill_layers) - 1
        for i, layer in enumerate(self.prefill_layers):
            if i == last:  # only the last layer's K/V is needed downstream
                _, k, v = layer.qkv_states(h, rope)
            else:
                h, k, v = layer(h, rope)  # causal self-attention
            kv[i, 0].copy_(k.transpose(1, 2))
            kv[i, 1].copy_(v.transpose(1, 2))

    def denoise(self, batch):
        """Euler flow over the action tokens; all per-request values are device tensors.

        RTC is selected by the presence of ``rtc_actions`` (a host property of
        the input set, so flow and RTC requests use different graphs); the delay
        itself is a tensor, so all delays share one graph.

        Flow times are ``linspace(0, 1, steps + 1) · 0.999``; each step embeds the current actions with the
        ``ActionHead``, runs expert 1 over the ``H`` action tokens attending to the prompt K/V
        (``action_attention``), reads out the velocity and takes an Euler step. Action positions continue
        the prompt from ``action_position_base``; under RTC, tokens after the pinned prefix are shifted by
        ``config.rtc_rope_gap`` (when ``0 < delay < H``), pinned tokens get ``tau = 1``, attend only to the
        prompt and the pinned group, and are reset to ``rtc_actions``. When ``dof_mask_all`` is false, the
        velocity of uncommanded dimensions is ``padding_action - x_0`` (fixed at step 0).

        Args:
            batch: ``StageBatch`` of ``B`` requests (1, 2, 4 or 8). Reads:

                - ``prompt_kv`` ``[layers, 2, B, kv_heads, P, head_dim]`` BF16 (rows left-padded to the
                  longest prompt when lengths differ);
                - ``prompt_kv_lengths`` ``[B]`` int32, optional: each row's own prompt length (added by the
                  engine);
                - ``prompt_valid`` ``[B]`` int32, present for padded prompts (``ProcessingConfig.prompt_bucket``):
                  each row's valid prompt tokens, the first of its own prompt columns (``left_align_prompts``);
                - ``num_steps`` (host int): Euler steps;
                - ``noise`` ``[B, H, D]`` FP32: initial flow state;
                - ``dof_mask`` ``[B, H, D]`` float64 (cast to FP32) and ``dof_mask_all`` (host bool);
                - ``padding_action`` ``[B, H, D]`` FP32: the normalized zero action;
                - ``action_position_base`` ``[B]`` int64: M-RoPE position of the first action token;
                - RTC only: ``rtc_actions`` ``[B, T, D]`` FP32 (``T <= H`` normalized pinned rows; the last
                  row is repeated up to ``H``) and ``inference_delay`` ``[B]`` int64 (number of pinned rows).

        Writes:
            ``batch.out["predict_action"]`` ``[B, H, D]`` FP32 (Wall-OSS: ``[B, 16, 26]``): the final
            actions unnormalized to checkpoint units, ``(x + 1) / 2 · action_delta + action_min``.
        """
        x_in = batch.inputs
        kv = x_in["prompt_kv"]
        if self.numerics == "optimized":
            normalized = torch.empty_like(batch.out["predict_action"])
            self.action(x_in, kv, normalized)
            batch.out["predict_action"].copy_(((normalized + 1) / 2) * self.action_delta + self.action_min)
            return
        # Rows' own prompt lengths when the batch left-pads different lengths (engine merge).
        prompt_lengths = x_in["prompt_kv_lengths"] if "prompt_kv_lengths" in x_in else None
        if "prompt_valid" in x_in:
            kv, prompt_lengths = left_align_prompts(kv, prompt_lengths, x_in["prompt_valid"])
        device = kv.device
        b, h_len, d_act = batch.batch_size, self.horizon, self.action_dim
        steps = int(x_in["num_steps"])
        rtc = "rtc_actions" in x_in
        delay = x_in["inference_delay"].reshape(b, 1) if rtc else torch.zeros(b, 1, dtype=torch.int64, device=device)
        k = torch.arange(h_len, device=device)[None]  # [1, H]
        pinned = k < delay  # [B, H]

        # Positions: the action block continues the prompt; tokens after the
        # pinned RTC prefix are shifted by rtc_rope_gap (when 0 < delay < H).
        positions = x_in["action_position_base"].reshape(b, 1) + k
        if rtc:
            shifted = (k >= delay) & (delay > 0) & (delay < h_len)
            positions = positions + shifted.to(positions.dtype) * self.config.rtc_rope_gap
        cos, sin = mrope_cos_sin(self.inv_freq, positions[None].expand(3, b, h_len), torch.float32)
        rope = self.action_rope.tables(cos, sin, DTYPE)

        # The prompt is always visible; pinned tokens do not attend to denoising
        # tokens (action_attention with pinned = delay).
        pinned_rows = delay.reshape(b)

        times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32) * FLOW_TIME_SCALE
        dof_mask = x_in["dof_mask"].to(torch.float32)
        dof_flat = dof_mask.reshape(-1, d_act)
        blend = not x_in["dof_mask_all"]
        x = x_in["noise"].to(torch.float32).clone()
        pinned_mask = pinned[:, :, None]
        if rtc:
            pinned_actions = x_in["rtc_actions"].to(torch.float32)[:, :h_len]
            if pinned_actions.shape[1] < h_len:  # repeat the last pinned row
                tail = pinned_actions[:, -1:].expand(-1, h_len - pinned_actions.shape[1], -1)
                pinned_actions = torch.cat([pinned_actions, tail], dim=1)
            x = torch.where(pinned_mask, pinned_actions, x)
        v_pad = (x_in["padding_action"].to(torch.float32) - x).reshape(-1, d_act)

        for i in range(steps):
            if rtc and i > 0:
                x = torch.where(pinned_mask, pinned_actions, x)
            tau = torch.where(pinned, torch.ones_like(times[i]), times[i])
            h = self.action_head.embed(x, tau, dof_mask).to(DTYPE)
            for layer_index, layer in enumerate(self.action_layers):
                h, _, _ = layer(h, rope, prompt_kv=(kv[layer_index, 0], kv[layer_index, 1]), num_pinned=pinned_rows,
                                prompt_lengths=prompt_lengths)
            v = self.action_head.proj_back(self.action_norm(h).to(torch.float32)).reshape(-1, d_act)
            if blend:
                v = v_pad * (1 - dof_flat) + v * dof_flat
            v = v.reshape(b, h_len, d_act)
            if rtc and i > 0:
                v = torch.where(pinned_mask, torch.zeros_like(v), v)
            x = x + (times[i + 1] - times[i]) * v
            if rtc and i == 0:
                x = torch.where(pinned_mask, pinned_actions, x)
        if rtc:
            x = torch.where(pinned_mask, pinned_actions, x)
        batch.out["predict_action"].copy_(((x + 1) / 2) * self.action_delta + self.action_min)


def left_align_prompts(kv: torch.Tensor, lengths: torch.Tensor | None, valid: torch.Tensor):
    """Turn padded prompt K/V into plain left-padded K/V with ``valid`` tokens per row.

    A padded prompt's valid tokens come first in its own columns, which are the last ``lengths[b]`` columns of the
    batch's ``P`` (the engine left-pads shorter rows). Rotating each row right by ``lengths[b] - valid[b]`` puts
    its valid tokens in the last columns, so ``action_attention`` reads them with ``prompt_lengths = valid`` and
    the action tokens follow them directly, as without padding. Only values move, so the attention sees the same
    keys as for the unpadded prompt.

    Args:
        kv: ``[layers, 2, B, kv_heads, P, head_dim]`` prompt K/V.
        lengths: ``[B]`` each row's own columns (``prompt_kv_lengths``); None means all ``P``.
        valid: ``[B]`` each row's valid prompt tokens.

    Returns:
        ``(kv, valid)``: the rotated K/V (a new tensor) and the prompt lengths for ``action_attention``.
    """
    width = kv.shape[4]
    padding = (width if lengths is None else lengths.reshape(-1).long()) - valid.reshape(-1).long()
    source = (torch.arange(width, device=kv.device)[None] - padding[:, None]) % width  # [B, P]
    return kv.gather(4, source[None, None, :, None, :, None].expand(kv.shape)), valid


def weight_mapping(model: WallOSS) -> dict[str, Source]:
    """Checkpoint key → model tensor, with the inference dtype of each tensor.

    Wall-X keeps the language-model RMSNorm weights and the action head in
    FP32 and casts everything else to BF16.

    The ViT patch embedding ``[hidden, C, T, P, P]`` is reshaped to
    ``[hidden, in_features]``. Checkpoint expert ``expert`` of each layer maps to ``prefill_layers`` (``expert = 0``)
    or ``action_layers`` (``expert = 1``).

    Args:
        model: The model shell (``WallOSS`` built on the "meta" device).

    Returns:
        ``{model parameter name: Source}``; each ``Source`` names the checkpoint key(s), the inference dtype
        and, where needed, a ``combine`` function.
    """
    bf16, fp32 = torch.bfloat16, torch.float32
    tower_config = model.vision.config
    mapping: dict[str, Source] = {
        "vision.patch_embed.weight": Source(
            ("visual.patch_embed.proj.weight",), bf16,
            combine=lambda t: t[0].reshape(tower_config.hidden_size, -1)),
        "vision.merger.norm.weight": Source.one("visual.merger.ln_q.weight", bf16),
        "vision.merger.fc1.weight": Source.one("visual.merger.mlp.0.weight", bf16),
        "vision.merger.fc1.bias": Source.one("visual.merger.mlp.0.bias", bf16),
        "vision.merger.fc2.weight": Source.one("visual.merger.mlp.2.weight", bf16),
        "vision.merger.fc2.bias": Source.one("visual.merger.mlp.2.bias", bf16),
        "embed_tokens": Source.one("model.embed_tokens.weight", bf16),
        "action_norm.weight": Source.one("model.norms.1.weight", fp32),
        "action_head.w1.weight": Source.one("action_preprocessor.w1.weight", fp32),
        "action_head.w2.weight": Source.one("action_preprocessor.w2.weight", fp32),
        "action_head.w3.weight": Source.one("action_preprocessor.w3.weight", fp32),
        "action_head.proj_back.weight": Source.one("action_preprocessor.action_proj_back.weight", fp32),
    }
    for i in range(tower_config.depth):
        source, target = f"visual.blocks.{i}", f"vision.blocks.{i}"
        for checkpoint_name, module_name in (("norm1", "norm1"), ("norm2", "norm2")):
            mapping[f"{target}.{module_name}.weight"] = Source.one(f"{source}.{checkpoint_name}.weight", bf16)
        for checkpoint_name, module_name in (("attn.qkv", "qkv"), ("attn.proj", "proj"),
                                             ("mlp.gate_up_proj", "mlp.gate_up"), ("mlp.down_proj", "mlp.down")):
            mapping[f"{target}.{module_name}.weight"] = Source.one(f"{source}.{checkpoint_name}.weight", bf16)
            mapping[f"{target}.{module_name}.bias"] = Source.one(f"{source}.{checkpoint_name}.bias", bf16)
    norm_dtype = {0: fp32, 1: fp32}
    for i in range(model.config.num_hidden_layers):
        source = f"model.layers.{i}"
        for expert, stack in ((0, "prefill_layers"), (1, "action_layers")):
            target = f"{stack}.{i}"
            mapping[f"{target}.input_norm.weight"] = Source.one(f"{source}.input_layernorms.{expert}.weight",
                                                                norm_dtype[expert])
            mapping[f"{target}.post_norm.weight"] = Source.one(f"{source}.post_attention_layernorms.{expert}.weight",
                                                       norm_dtype[expert])
            mapping[f"{target}.qkv.weight"] = Source.one(f"{source}.self_attn.qkv_proj_experts.{expert}.weight", bf16)
            mapping[f"{target}.qkv.bias"] = Source.one(f"{source}.self_attn.qkv_proj_experts.{expert}.bias", bf16)
            mapping[f"{target}.o_proj.weight"] = Source.one(f"{source}.self_attn.o_proj_experts.{expert}.weight", bf16)
            mapping[f"{target}.mlp.gate_up.weight"] = Source.one(
                f"{source}.moe.experts.{expert}.gate_up_proj.weight", bf16)
            mapping[f"{target}.mlp.down.weight"] = Source.one(f"{source}.moe.experts.{expert}.down_proj.weight", bf16)
    return mapping
