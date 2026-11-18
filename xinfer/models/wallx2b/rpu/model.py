"""Wall-X 2B FP16 TP8 model with resident Vision, cache-only Text and NFE8 Action.

Model placement and scratch liveness live here and in the stage lowerings.
All device allocation, DMA, packing, submission and completion use the native provider's
public TP8 interface. There is no CPU or RhinoForge execution fallback.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from xinfer.models.interface import CameraSpec, Capabilities, IOSchema, Stage, StageGraph, VLAModel
from xinfer.platforms.rpu.plan import RpuRequestPlan

from .config import WallX2BConfig, preflight_checkpoint


class WallX2B(VLAModel):
    """Experimental FP16 TP8 B1 H16 non-RTC policy with explicit admission."""

    stage_graph = StageGraph(
        Stage("vision", "encode_images"),
        Stage("prefill", "prefill", "vision"),
        Stage("action", "denoise", "prefill"),
    )
    rpu_retained_segments = 1536
    rpu_file_descriptor_budget = 4096
    workspace_bytes = 6 * 1024 * 1024

    @classmethod
    def from_checkpoint(cls, path, *, device, ops=None, quantization="fp16", **options):
        if device not in ("rpu", "rpu:0"):
            raise ValueError("wallx2b first profile requires device rpu:0")
        if quantization != "fp16":
            raise ValueError("wallx2b accepts only FP16")
        allowed = {"allow_candidate", "image_workers", "audit_inputs", "max_cached_signatures",
                   "prefill_collective_window", "action_collective_window", "action_whole_stage",
                   "action_fused_steps", "vision_whole_stage", "vision_fused_stage",
                   "prefill_whole_stage"}
        if set(options) - allowed:
            raise ValueError(f"unsupported wallx2b execution options: {sorted(set(options) - allowed)}")
        if options.get("allow_candidate", False) is not True:
            raise ValueError("wallx2b experimental RPU execution requires explicit allow_candidate=True")
        capacity = options.get("max_cached_signatures", 2)
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 2:
            raise ValueError("wallx2b max_cached_signatures must be in [1,2]")
        window = options.get("prefill_collective_window", 1)
        if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 15:
            raise ValueError("prefill_collective_window must be in [1,15]")
        action_window = options.get("action_collective_window", 1)
        if isinstance(action_window, bool) or not isinstance(action_window, int) or action_window not in (1, 4):
            raise ValueError("action_collective_window must be1 or4")
        whole_stage = options.get("action_whole_stage", False)
        if not isinstance(whole_stage, bool):
            raise ValueError("action_whole_stage must be an explicit boolean")
        if whole_stage and action_window != 1:
            raise ValueError("action_whole_stage replaces windowing; use action_collective_window1")
        action_fused_steps = options.get("action_fused_steps", False)
        if not isinstance(action_fused_steps, bool):
            raise ValueError("action_fused_steps must be an explicit boolean")
        if action_fused_steps and whole_stage:
            raise ValueError("action_whole_stage already folds the action entry")
        if action_fused_steps and action_window != 1:
            raise ValueError("action_fused_steps replaces windowing; use action_collective_window1")
        vision_whole_stage = options.get("vision_whole_stage", False)
        if not isinstance(vision_whole_stage, bool):
            raise ValueError("vision_whole_stage must be an explicit boolean")
        vision_fused_stage = options.get("vision_fused_stage", False)
        if not isinstance(vision_fused_stage, bool):
            raise ValueError("vision_fused_stage must be an explicit boolean")
        if vision_fused_stage and vision_whole_stage:
            raise ValueError("vision_whole_stage already fuses the whole vision stage")
        prefill_whole_stage = options.get("prefill_whole_stage", False)
        if not isinstance(prefill_whole_stage, bool):
            raise ValueError("prefill_whole_stage must be an explicit boolean")
        if prefill_whole_stage and window != 1:
            raise ValueError("prefill_whole_stage replaces windowing; use prefill_collective_window1")
        manifest = preflight_checkpoint(path)
        model = cls()
        model.checkpoint, model.config = Path(path), manifest.config
        model.manifest = manifest
        model.quantization = quantization
        model.rpu_allow_candidate = True
        model.max_cached_signatures = capacity
        model.prefill_collective_window = window
        model.action_collective_window = action_window
        model.action_whole_stage = whole_stage
        model.action_fused_steps = action_fused_steps
        model.vision_whole_stage = vision_whole_stage
        model.vision_fused_stage = vision_fused_stage
        model.prefill_whole_stage = prefill_whole_stage
        model._processor, model._output_processor, _ = cls.build_processors(
            path, image_workers=options.get("image_workers", 6), audit_inputs=options.get("audit_inputs", False)
        )
        model.context = None
        model._initial_resources = model._base_resources = frozenset()
        model.weights = {}
        model.weight_manifest = []
        model.stats = Counter()
        model._prepared = model._invalid = False
        model._attempted = False
        model._signatures = OrderedDict()
        model._request_release = None
        return model

    def capabilities(self):
        return Capabilities(batch_sizes={s.name: (1,) for s in self.stage_graph}, num_steps=8)

    def io_schema(self):
        from .processing import DEFAULT_CAMERAS

        return IOSchema(cameras=tuple(CameraSpec(name) for name in DEFAULT_CAMERAS),
                        state_keys=("state",), action_keys=("predict_action",), action_dim=26,
                        action_horizon=16, extra={"initial_noise": "FP32 [1,16,26]", "rtc": False})

    @classmethod
    def build_processors(cls, checkpoint, *, image_workers=6, audit_inputs=False, **options):
        from .processing import WallX2BInputProcessor, WallX2BOutputProcessor

        unknown = set(options) - {"quantization", "allow_candidate", "max_cached_signatures",
                                  "prefill_collective_window", "action_collective_window"}
        if unknown or options.get("quantization", "fp16") != "fp16":
            raise ValueError("wallx2b processors require the fixed FP16 profile")
        config = WallX2BConfig.load(checkpoint)
        return (
            WallX2BInputProcessor(checkpoint, config, image_workers=image_workers, audit_inputs=audit_inputs),
            WallX2BOutputProcessor(config),
            dict(model="wallx2b", profile=config.name, action_dim=26, action_horizon=16,
                 num_steps=8, rtc=False, output_space="physical_relative26", execution="experimental_rpu"),
        )

    def input_processor(self):
        return self._processor

    def output_processor(self):
        return self._output_processor

    def prepare_rpu(self, context):
        """Load and pack weights once, before the first timed policy call."""
        if self._attempted:
            raise RuntimeError("model preparation was already attempted; construct a fresh model")
        self._attempted = True
        self.context = context
        self._initial_resources = self._base_resources = frozenset(context._resources)
        context.require_file_descriptor_budget(self.rpu_file_descriptor_budget)
        if self.action_collective_window != 1:
            context.require_collective_window(self.action_collective_window, family="M16")
        if (self.action_whole_stage or self.action_fused_steps or self.vision_whole_stage
                or self.vision_fused_stage or self.prefill_whole_stage):
            context.require_batch_fused()
        from .action import prepare_conditioning
        from .weights import WeightPreparer

        preparer = WeightPreparer(self.manifest)
        host = preparer.host_tensors()
        self.conditioning = prepare_conditioning(host)
        self.positions = host["model.visual.pos_embed.weight"].numpy().copy()
        self.normalizers = preparer.load_normalizers()
        del host
        preparer.release_host_tensors()
        for record in preparer.iter_prepared():
            self.weights[record.name] = context.prepare_weight(record.array, layout=record.layout)
            self.weight_manifest.append(record.manifest_row())
        if len(self.weights) != 654:
            raise ValueError("the admitted checkpoint must produce exactly 654 device weight records")
        self.workspace = context.empty((self.workspace_bytes // 2,), distribution="local")
        self._base_resources = frozenset(context._resources)
        self._build_plans()
        self._prepared = True
        self.stats["weight_preparations"] += 1

    def _build_plans(self):
        from .denoise import DenoiseWorkspace, PreparedDenoise
        from .prefill import PrefillState
        from .vision import VisionPlan

        self._signatures.clear()
        self.vision = VisionPlan(
            self.context, self.config, self.weights, self.positions,
            workspace=self.workspace, fused_reduce_norm=True,
            whole_stage=self.vision_whole_stage,
            fused_stage=self.vision_fused_stage,
        )
        self.prefix_state = PrefillState(self.context)
        self.action_workspace = DenoiseWorkspace(self.context, arena=self.workspace)
        caches = tuple(zip((self.prefix_state.key_caches[i] for i in (3, 7, 11, 15, 19, 23)),
                           (self.prefix_state.value_caches[i] for i in (3, 7, 11, 15, 19, 23))))
        self.action = PreparedDenoise(self.context, self.weights, self.conditioning, caches,
                                     workspace=self.action_workspace, max_cached_buckets=2,
                                     collective_window=self.action_collective_window,
                                     whole_stage=self.action_whole_stage,
                                     fused_steps=self.action_fused_steps)
        self._invalid = False

    @staticmethod
    def _image_starts(prepared):
        types = prepared.tensors["mm_token_type_ids"].numpy().reshape(-1)
        starts = np.flatnonzero((types != 0) & np.r_[True, types[:-1] == 0]).tolist()
        ends = (np.flatnonzero((types != 0) & np.r_[types[1:] == 0, True]) + 1).tolist()
        if len(starts) != 3 or [end - begin for begin, end in zip(starts, ends)] != [77, 70, 70]:
            raise ValueError("the admitted prefix requires three canonical image runs of 77/70/70 tokens")
        return tuple(starts)

    @staticmethod
    def _close_resources(resources):
        failure = None
        for resource in reversed(tuple(resources)):
            try:
                resource.close()
            except BaseException as error:
                # Native execution close releases its handle even when a
                # previously reported wait failure is rethrown. Finish closing
                # the adapter owner, but never suppress a live release failure.
                if not getattr(resource.native, "_id", 1):
                    resource.close()
                else:
                    failure = failure or error
        if failure is not None:
            raise failure

    def _prefix_plan(self, prepared):
        from .prefill import PrefillPlan

        length = prepared.tensors["prefix_length"]
        starts = self._image_starts(prepared)
        key = (prepared.tensors["prefix_bucket"], starts)
        if key in self._signatures:
            self._signatures.move_to_end(key)
            self.stats["prefill_replays"] += 1
            return self._signatures[key][0]
        if len(self._signatures) == self.max_cached_signatures:
            _, (_, resources) = self._signatures.popitem(last=False)
            self._close_resources(resources)
            self.stats["prefill_evictions"] += 1
        before = frozenset(self.context._resources)
        try:
            plan = PrefillPlan(self.context, self.config, self.weights, prefix_length=length,
                               bucket=prepared.tensors["prefix_bucket"], workspace=self.workspace,
                               state=self.prefix_state, image_features=self.vision.output_ddr,
                               image_starts=starts, collective_window=self.prefill_collective_window,
                               whole_stage=self.prefill_whole_stage)
        except BaseException:
            self._close_resources(r for r in self.context._resources if r not in before)
            raise
        resources = tuple(r for r in self.context._resources if r not in before)
        self._signatures[key] = plan, resources
        self.stats["prefill_builds"] += 1
        return plan

    def _request_rollback(self, release, error):
        """Release a partial binding without recovering the invalidated plans."""
        self._invalid = True
        self._request_release = release
        try:
            if release is not None:
                release()
        except BaseException as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(f"RPU request rollback failed: {cleanup_error}")
        else:
            self._request_release = None

    def _incomplete_request(self, release):
        self._invalid = True
        self._request_release = release

    def rollback_rpu_request(self, error):
        """Release a request which failed before its plan was returned.

        Captured plans remain invalid until explicit reset. A failed binding
        release is retained for retry without replacing the original error.
        """
        self._invalid = True
        release, self._request_release = self._request_release, None
        if release is not None:
            self._request_rollback(release, error)

    def begin_rpu_request(self, prepared, context):
        """Admit the text side of a request while pixels are still packing.

        Context/idle checks, the (cached) prefix plan, stale-state
        invalidation, and the prefix + action host writes happen here; every
        buffer touched belongs to the text/admission side and none depends
        on pixel_values. Returns the (prefix, action) binding for the
        matching finish_rpu_request call.
        """
        if context is not self.context or not self._prepared or self._invalid:
            raise RuntimeError("wallx2b requires its initialized, recovered execution context")
        prefix = self._prefix_plan(prepared)
        state = self.prefix_state
        owners = (state.recurrent, state.history, state.history_out,
                  *state.key_caches.values(), *state.value_caches.values(),
                  *(cache for pair in self.action.destination_caches for cache in pair))
        try:
            for owner in owners:
                owner.invalidate()
            prefix.bind(prepared.tensors["position_ids"], input_ids=prepared.tensors["input_ids"])
            return prefix, self.action.bind(prepared)
        except BaseException:
            # A failed upload/rebind can occur before DenoiseBinding exists.
            # create_rpu_request records rollback ownership for the engine.
            self._invalid = True
            raise

    def finish_rpu_request(self, binding, prepared):
        """Bind pixels and assemble the plan for a begun request.

        The three binds write disjoint request buffers, so this vision bind
        after prefix/action admission is equivalent to the original single
        create_rpu_request ordering while the device stays idle.
        """
        from .processing import denormalize_actions

        prefix, action = binding
        try:
            self.vision.bind(prepared.tensors["pixel_values"])
        except BaseException:
            self._incomplete_request(action.close)
            raise

        def collect():
            normalized = action.collect_normalized()
            physical = denormalize_actions(normalized, self.normalizers["action_min"],
                                           self.normalizers["action_delta"])
            self.stats["requests_completed"] += 1
            return {"predict_action": physical}

        def abort(error):
            self._invalid = True
            self.stats["requests_failed"] += 1

        return RpuRequestPlan(
            stages={"vision": self.vision.executions,
                    "prefill": (*prefix.input_executions, *prefix.executions),
                    "action": action.executions},
            collect=collect, abort=abort, release=action.close,
        )

    def create_rpu_request(self, prepared, context):
        if context is not self.context or not self._prepared or self._invalid:
            raise RuntimeError("wallx2b requires its initialized, recovered execution context")
        binding = None
        try:
            binding = self.begin_rpu_request(prepared, context)
            return self.finish_rpu_request(binding, prepared)
        except BaseException:
            if binding is None:
                self._incomplete_request(None)
            raise

    def reset_rpu(self):
        """Rebuild captured generations while retaining immutable packed weights."""
        self._invalid = True
        release = self._request_release
        if release is not None:
            release()
            self._request_release = None
        self._close_resources(r for r in self.context._resources if r not in self._base_resources)
        self._build_plans()
        self.stats["resets"] += 1

    def close_rpu(self):
        if self.context is not None:
            self._close_resources(r for r in self.context._resources if r not in self._initial_resources)
            self._signatures.clear()
            self.weights.clear()
            self._prepared = False

    def describe_rpu(self):
        return dict(profile=self.config.name, stats=dict(self.stats),
                    prefill_collective_window=self.prefill_collective_window,
                    action_collective_window=self.action_collective_window,
                    action_whole_stage=self.action_whole_stage,
                    action_fused_steps=self.action_fused_steps,
                    vision_whole_stage=self.vision_whole_stage,
                    vision_fused_stage=self.vision_fused_stage,
                    prefill_whole_stage=self.prefill_whole_stage,
                    cached_prefix_signatures=[list(key) for key in self._signatures],
                    workspace_bytes_per_core=self.workspace_bytes,
                    weight_records=len(self.weights), action=self.action.describe(),
                    runtime=self.context.describe())
