"""Prepared TP8 CES vision tower with three resident, phase-reused slots.

The model owns tensor placement and explicit reductions. xcompute owns all
packing, device kernels, prepared DMA and completion. No Torch device tensor
or RhinoForge executor appears in this path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch

from .processing import IMAGE_GRIDS


PATCH_ROWS = 868
MERGED_ROWS = 217
# Per-camera packed-row regions of the pixel buffer, in the canonical
# camera order (face_view 14x22=308, then two 14x20=280). Must match the
# processor's per-camera patch-row layout exactly.
CAMERA_PATCH_ROWS = (308, 280, 280)
SLOT_BYTES = PATCH_ROWS * 1024 * 2
WORKSPACE_BYTES = 3 * SLOT_BYTES


def vision_coordinates() -> np.ndarray:
    """Patch coordinates in the checkpoint's merge-block-major order."""
    coordinates = []
    for _, height, width in IMAGE_GRIDS:
        for row in range(0, height, 2):
            for column in range(0, width, 2):
                coordinates.extend((row + a, column + b) for a in range(2) for b in range(2))
    return np.asarray(coordinates, dtype=np.int32)


@torch.inference_mode()
def vision_rope_tables() -> tuple[np.ndarray, np.ndarray]:
    """Retain HF's FP16 inv_freq and FP16 position outer product."""
    inverse = (1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))).half()
    phase = torch.arange(22, dtype=torch.float16)[:, None] * inverse[None]
    coordinates = torch.from_numpy(vision_coordinates()).long()
    cosine = torch.cat((phase.cos()[coordinates[:, 0]], phase.cos()[coordinates[:, 1]]), dim=1)
    sine = torch.cat((phase.sin()[coordinates[:, 0]], phase.sin()[coordinates[:, 1]]), dim=1)
    return cosine.contiguous().numpy(), sine.contiguous().numpy()


@torch.inference_mode()
def interpolate_positions(weight) -> np.ndarray:
    """HF Qwen3.5's four-corner interpolation with each FP16 boundary intact."""
    weight = torch.as_tensor(weight, device="cpu")
    if weight.dtype != torch.float16 or tuple(weight.shape) != (2304, 1024) or not bool(torch.isfinite(weight).all()):
        raise ValueError("vision position weight requires finite CPU FP16 [2304,1024]")
    indices, coefficients = [[] for _ in range(4)], [[] for _ in range(4)]
    for _, height, width in IMAGE_GRIDS:
        rows, cols = torch.linspace(0, 47, height), torch.linspace(0, 47, width)
        low_r, low_c = rows.int(), cols.int()
        high_r, high_c = (low_r + 1).clamp(max=47), (low_c + 1).clamp(max=47)
        dr, dc = rows - low_r, cols - low_c
        corners = ((low_r, low_c, 1 - dr, 1 - dc), (low_r, high_c, 1 - dr, dc),
                   (high_r, low_c, dr, 1 - dc), (high_r, high_c, dr, dc))
        for i, (r, c, a, b) in enumerate(corners):
            indices[i].extend((r[:, None] * 48 + c[None]).flatten().tolist())
            coefficients[i].extend((a[:, None] * b[None]).flatten().tolist())
    values = weight[torch.tensor(indices)] * torch.tensor(coefficients, dtype=torch.float16)[:, :, None]
    values = values[0] + values[1] + values[2] + values[3]
    output, offset = [], 0
    for _, height, width in IMAGE_GRIDS:
        part = values[offset:offset + height * width]
        output.append(part.reshape(height // 2, 2, width // 2, 2, 1024).permute(0, 2, 1, 3, 4).reshape(-1, 1024))
        offset += height * width
    return torch.cat(output).contiguous().numpy()


class VisionPlan:
    """Build once, bind fresh CPU pixels, execute retained native batches.

    One 868-row stream contains three attention islands. Full-width residuals
    use slots A/B/C; dead inputs are reused only after their last consumer.
    Merger chunks contain 512 and 356 patches (128/89 output rows), respecting
    the public projection's 147-row limit. The final output owns stable DDR.

    Whole-stage mode keeps the plain embedding batch — the
    counterless all_gather keeps its host-verified isolation segment, the
    full-core barrier the vendor doc requires — and fuses the tower and
    merger into one retained queue with per-occurrence vendor pool
    counters: the per-batch epochs concatenated (tower 50..3
    descending, merger re-arming 50,49 at graph distance 48, the same-id
    reuse class the action whole-stage queue proved on the board).
    """

    workspace_bytes = WORKSPACE_BYTES
    output_rows = MERGED_ROWS

    def __init__(self, context, config, weights: Mapping, position_embedding, *, workspace=None,
                 fused=True, fused_reduce_norm=False, whole_stage=False, fused_stage=False):
        if config.action_horizon != 16 or config.max_prefix_length != 512:
            raise ValueError("VisionPlan requires the admitted CES profile")
        if not isinstance(fused, bool):
            raise TypeError("fused must be an explicit bool")
        if not isinstance(fused_reduce_norm, bool):
            raise TypeError("fused_reduce_norm must be an explicit bool")
        if not isinstance(whole_stage, bool):
            raise TypeError("whole_stage must be an explicit bool")
        if not isinstance(fused_stage, bool):
            raise TypeError("fused_stage must be an explicit bool")
        if whole_stage and fused_stage:
            raise ValueError("whole_stage already fuses the entire vision stage; "
                             "fused_stage is the embedding-plain/tower-merger-fused mode")
        self.context, self.config, self.weights, self.fused = context, config, weights, fused
        self.fused_reduce_norm = fused_reduce_norm
        self.whole_stage = whole_stage
        self.fused_stage = fused_stage
        self._collective_occurrence = 0
        self.workspace = (workspace if workspace is not None
                          else context.empty((WORKSPACE_BYTES // 2,), distribution="local"))
        if math.prod(self.workspace.shape) * 2 < WORKSPACE_BYTES or self.workspace.context is not context:
            raise ValueError("VisionPlan workspace is too small or belongs to another context")
        self.pixels = context.empty((PATCH_ROWS, 1536), storage="ddr", distribution="shared")
        if sum(CAMERA_PATCH_ROWS) != PATCH_ROWS:
            raise ValueError("camera patch-row layout must tile the pixel buffer")
        offset = 0
        self._pixel_views = []
        for rows in CAMERA_PATCH_ROWS:
            self._pixel_views.append(context.view(
                self.pixels, (rows, 1536), offset_bytes=offset * 1536 * 2))
            offset += rows
        self._camera_uploads = frozenset()
        self.output_ddr = context.empty((MERGED_ROWS, 2048), storage="ddr", distribution="shared")
        positions = interpolate_positions(position_embedding)
        self.position = context.prepare_weight(positions, layout="row_major")
        self.cosine, self.sine = (context.empty((PATCH_ROWS, 32), distribution="replicated") for _ in range(2))
        cosine, sine = vision_rope_tables()
        self.cosine.copy_from(cosine)
        self.sine.copy_from(sine)
        # All model parameters remain in DDR. Only the current norm/bias is
        # staged into these small shared owners by prepared device copies.
        self._norm_parameters = context.empty((2048,), distribution="replicated")
        self._column_bias = context.empty((512,), distribution="local")
        self._row_bias = context.empty((2048,), distribution="owner0")
        self.operations = []
        self._batch_operations = []
        self.executions = []
        if whole_stage or fused_stage:
            self.context.require_batch_fused()
        if whole_stage:
            # The original's vision stage is ONE SDK segment end to end —
            # embedding, all_gather, tower, and merger in a single retained
            # queue (original vision seg0: 182C/167D/273S with the
            # counterless owner-push all_gather inside). The native fused
            # admission carries the all_gather in-segment with the
            # position-broadcast DMA burst as the cross-core slack fence —
            # the same structure the original's per-layer weight streaming
            # provides.
            self._build_embedding()
            self._build_tower()
            self._build_merger()
            self._finish_batch_fused()
        elif fused_stage:
            # Keep the embedding as its own plain batch and pack
            # tower+merger into ONE fused retained queue — two executions
            # per vision stage.
            self._build_embedding()
            self._finish_batch()
            self._build_tower()
            self._build_merger()
            self._finish_batch_fused()
        else:
            self._build_embedding()
            self._finish_batch()
            self._build_tower()
            # The plain path keeps the original three retained batches: the
            # tower alone is 48 AR collectives — exactly the tp_batch
            # 48-epoch pool limit — so the merger (2 more ARRs) must stay a
            # separate batch, matching the original three-execution stage.
            self._finish_batch()
            self._build_merger()
            self._finish_batch()
        self.executions = tuple(self.executions)
        self.operations = tuple(self.operations)
        self.operation_count = len(self.operations)
        self.collective_count = 50  # patch gather + 24*(O+down) + merger reduce
        # There are two merger chunks, each with its own explicit reduction.
        self.collective_count += 1

    def _v(self, slot, shape, *, byte_offset=0, distribution="replicated", shard_axis=-1):
        return self.context.view(self.workspace, shape, offset_bytes=slot * SLOT_BYTES + byte_offset,
                                 distribution=distribution, shard_axis=shard_axis)

    def _w(self, name):
        key = "vision." + name
        if key not in self.weights:
            raise KeyError(f"missing prepared device weight {key}")
        value = self.weights[key]
        if value.storage != "ddr" or value.layout != "row_major":
            return value
        if ".norm" in name:
            offset = 0 if name.endswith(".weight") else 2048
            target = self.context.view(self._norm_parameters, (1024,), offset_bytes=offset)
            self._copy(value, target, mode="broadcast")
            return target
        if name.endswith("_bias"):
            columns = value.shape[-1]
            if name.endswith((".o_bias", ".down_bias", "linear_fc2_bias")):
                target = self.context.view(self._row_bias, (columns,))
                source = self.context.view(value, (columns,))
                self._copy(source, target, mode="core0")
                return target
            else:
                distribution = "local" if name.endswith(".qkv_bias") else "sharded"
                target = self.context.view(self._column_bias, (columns,), distribution=distribution,
                                           shard_axis=0 if distribution == "sharded" else -1)
            self._copy(value, target, mode="corewise")
            return target
        return value

    def _op(self, name, tensors, *, integers=None, scalars=None, **selectors):
        operation = self.context.prepare(name, tensors=tensors, integers=integers, scalars=scalars,
                                         **selectors)
        self.operations.append(operation)
        self._batch_operations.append(operation)
        return operation

    def _copy(self, source, target, *, mode):
        operation = self.context.prepare_copy(source, target, mode=mode)
        self.operations.append(operation)
        self._batch_operations.append(operation)

    def _finish_batch(self):
        self.executions.append(self.context.batch(self._batch_operations))
        self._batch_operations = []

    def _finish_batch_fused(self):
        self.executions.append(self.context.batch_fused(self._batch_operations))
        self._batch_operations = []

    def _next_counter(self):
        """Vendor pool id for the next counted collective (whole-stage mode).

        The per-batch epochs concatenated: tower occurrences 0..47 walk the
        descending 50..3; the two merger reductions re-arm 50,49 — same-id
        reuse at graph distance 48, as the original's vision segment does.
        """
        occurrence = self._collective_occurrence
        self._collective_occurrence += 1
        return 50 - occurrence if occurrence < 48 else 50 - (occurrence - 48)

    def _norm(self, src, out, stem):
        self._op("layernorm", {"x": src, "out": out,
                 "weight": self._w(stem + ".weight"), "bias": self._w(stem + ".bias")},
                 scalars={"epsilon": 1.0e-6})

    def _reduce_norm(self, partial, residual, x, norm, stem):
        # The fused queue (whole-stage or tower/merger fusion)
        # needs the per-occurrence counters (tower 50..3; the merger's ARR
        # re-arms 50,49) — only the three-plain-batch path stays counterless.
        integers = ({"counter": self._next_counter()}
                    if self.whole_stage or self.fused_stage else None)
        self._op("all_reduce", {
            "x": partial, "residual": residual, "out": x, "norm": norm,
            "gamma": self._w(stem + ".weight"), "beta": self._w(stem + ".bias"),
        }, integers=integers, scalars={"epsilon": 1.0e-6})

    def _build_embedding(self):
        pixel = self._v(1, (PATCH_ROWS, 1536))
        shard = self._v(0, (PATCH_ROWS, 128), distribution="sharded", shard_axis=1)
        gathered = self._v(1, (PATCH_ROWS, 1024))
        position = self._v(2, (PATCH_ROWS, 1024))
        residual = self._v(0, (PATCH_ROWS, 1024))
        self._copy(self.pixels, pixel, mode="broadcast")
        self._op("gemm", {"x": pixel, "out": shard, "w": self._w("patch"),
                                    "bias": self._w("patch_bias")})
        self._op("all_gather", {"x": shard, "out": gathered})
        self._copy(self.position, position, mode="broadcast")
        self._op("binary", {"lhs": gathered, "rhs": position, "out": residual},
                 integers={"vectors_per_block": 32}, operation="add")

    def _build_tower(self):
        residual = self._v(0, (PATCH_ROWS, 1024))
        residual_shard = (self._v(0, (PATCH_ROWS, 1024), distribution="sequence_shard", shard_axis=0)
                          if self.fused_reduce_norm else residual)
        post = self._v(1, (PATCH_ROWS, 1024))
        post_shard = (self._v(1, (PATCH_ROWS, 1024), distribution="sequence_shard", shard_axis=0)
                      if self.fused_reduce_norm else post)
        tmp = self._v(2, (PATCH_ROWS, 1024))
        partial = self._v(2, (PATCH_ROWS, 1024), distribution="partial")
        q, k, v = (self._v(2, (PATCH_ROWS, 128), byte_offset=i * PATCH_ROWS * 128 * 2,
                           distribution="sharded", shard_axis=1) for i in range(3))
        attended = self._v(1, (PATCH_ROWS, 128), distribution="sharded", shard_axis=1)
        up = self._v(0, (PATCH_ROWS, 512), distribution="sharded", shard_axis=1)
        if not self.fused:
            self.vision_caches = []
            for _, h, w in IMAGE_GRIDS:
                capacity = math.ceil(h * w / 16) * 16
                self.vision_caches.append(tuple(self.context.empty((8, capacity, 128), storage="ddr",
                    distribution="shared", state_kind="cache", layout=layout)
                    for layout in ("vision_k_cache", "vision_v_cache")))
        for layer in range(24):
            stem = str(layer)
            if not self.fused_reduce_norm or layer == 0:
                self._norm(residual, post, stem + ".norm1")
            if self.fused:
                self._op("qkv_proj", {"x": post, "out_q": q, "out_k": k, "out_v": v,
                    "w": self._w(stem + ".qkv"), "bias": self._w(stem + ".qkv_bias"),
                    "cos": self.cosine, "sin": self.sine})
                self._op("attention", {"q": q, "k": k, "v": v, "out": attended},
                         integers={"length0": 308, "length1": 280, "length2": 280})
            else:
                self._op("qkv_proj", {"x": post, "out_q": q, "out_k": k, "out_v": v,
                    "w": self._w(stem + ".qkv"), "bias": self._w(stem + ".qkv_bias")})
                for tensor, i in ((q, 0), (k, 1)):
                    reshaped = self._v(2, (PATCH_ROWS, 2, 64), byte_offset=i * PATCH_ROWS * 128 * 2,
                                       distribution="sharded", shard_axis=1)
                    self._op("rope", {"x": reshaped, "out": reshaped,
                        "cos": self.cosine, "sin": self.sine}, integers={"tp8": 1})
                offset = 0
                attention = {"q": q, "out": attended}
                for camera, ((_, h, w), (kc, vc)) in enumerate(zip(IMAGE_GRIDS, self.vision_caches)):
                    rows, capacity = h * w, math.ceil(h * w / 16) * 16
                    for plane, cache in ((1, kc), (2, vc)):
                        source = self._v(2, (rows, 128), byte_offset=(plane * PATCH_ROWS + offset) * 128 * 2,
                                         distribution="sharded", shard_axis=1)
                        self._op("kv_cache_insert", {"input": source, "cache": cache},
                                 integers={"position": 0, "capacity": capacity})
                    attention["key_cache" + str(camera)] = kc
                    attention["value_cache" + str(camera)] = vc
                    offset += rows
                self._op("attention", attention,
                         integers={"length0": 308, "length1": 280, "length2": 280})
            self._op("gemm", {"x": attended, "out": partial, "w": self._w(stem + ".o"),
                              "bias": self._w(stem + ".o_bias")}, parallel="row")
            if self.fused_reduce_norm:
                self._reduce_norm(partial, residual if layer == 0 else residual_shard,
                                  post_shard, tmp, stem + ".norm2")
            else:
                self._op("all_reduce", {"x": partial, "residual": residual, "out": post},
                         integers={"with_residual": 1})
                self._norm(post, tmp, stem + ".norm2")
            # The fused projection applies GELU before its FP16 store; the
            # unfused path rounds the projection first (different numerics).
            self._op("gemm", {"x": tmp, "out": up,
                "w": self._w(stem + ".up"), "bias": self._w(stem + ".up_bias")},
                activation="gelu" if self.fused else "none")
            if not self.fused:
                self._op("activation", {"x": up, "out": up}, function="gelu_tanh")
            self._op("gemm", {"x": up, "out": partial, "w": self._w(stem + ".down"),
                              "bias": self._w(stem + ".down_bias")}, parallel="row")
            if self.fused_reduce_norm and layer < 23:
                self._reduce_norm(partial, post_shard, residual_shard, post,
                                  str(layer + 1) + ".norm1")
            else:
                arr_integers = {"with_residual": 1}
                if self.whole_stage or self.fused_stage:
                    arr_integers["counter"] = self._next_counter()
                self._op("all_reduce", {
                    "x": partial, "residual": post_shard if self.fused_reduce_norm else post,
                    "out": residual,
                }, integers=arr_integers)

    def _build_merger(self):
        for patch_offset, patch_rows in ((0, 512), (512, 356)):
            rows = patch_rows // 4
            source = self._v(0, (patch_rows, 1024), byte_offset=patch_offset * 1024 * 2)
            normalized = self._v(1, (patch_rows, 1024))
            self._norm(source, normalized, "merger.norm")
            grouped = self._v(1, (rows, 4096))
            hidden = self._v(2, (rows, 512), distribution="sharded", shard_axis=1)
            self._op("gemm", {"x": grouped, "out": hidden,
                "w": self._w("merger.linear_fc1"), "bias": self._w("merger.linear_fc1_bias")},
                activation="gelu" if self.fused else "none")
            if not self.fused:
                self._op("activation", {"x": hidden, "out": hidden}, function="gelu_erf")
            partial = self._v(1, (rows, 2048), distribution="partial")
            self._op("gemm", {"x": hidden, "out": partial,
                "w": self._w("merger.linear_fc2"), "bias": self._w("merger.linear_fc2_bias")},
                parallel="row")
            output = self._v(0, (rows, 2048), byte_offset=(patch_offset // 4) * 2048 * 2)
            arr_integers = {"with_residual": 0}
            if self.whole_stage or self.fused_stage:
                arr_integers["counter"] = self._next_counter()
            self._op("all_reduce", {"x": partial, "out": output}, integers=arr_integers)
        self.output = self._v(0, (MERGED_ROWS, 2048))
        self._copy(self.output, self.output_ddr, mode="core0")

    def begin_camera_uploads(self):
        """Reset the per-camera upload set at request admission.

        Must run before the input processor submits camera jobs (a camera
        can complete before the request's text stage does).
        """
        self._camera_uploads = frozenset()

    def bind_camera(self, index, array):
        """Upload one camera's packed rows into its pixel-buffer region.

        Called from the input processor's image-pool callbacks while the
        device is idle, so the 2.7 MiB pixel upload overlaps the remaining
        camera work instead of serializing after the image join. The
        exponent scan is deliberately absent: every packed pixel on every
        processor path is an entry of the finite _UINT8_TO_FP16 table
        (validated once at processing-module import), so a scan here is
        vacuous by construction.
        """
        if (not isinstance(index, int) or isinstance(index, bool)
                or not 0 <= index < len(CAMERA_PATCH_ROWS)):
            raise ValueError("camera index out of range")
        rows = CAMERA_PATCH_ROWS[index]
        if (not isinstance(array, np.ndarray) or array.dtype != np.float16
                or array.shape != (rows, 1536) or not array.flags.c_contiguous):
            raise ValueError(f"camera {index} requires contiguous FP16 rows [{rows},1536]")
        self._pixel_views[index].copy_from(array)
        self._camera_uploads = self._camera_uploads | {index}

    def bind(self, pixels):
        if isinstance(pixels, torch.Tensor):
            if pixels.device.type != "cpu":
                raise ValueError("VisionPlan pixels must be CPU")
            pixels = pixels.detach().contiguous().numpy()
        if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.float16
                or pixels.shape != (PATCH_ROWS, 1536) or not pixels.flags.c_contiguous):
            raise ValueError("VisionPlan requires finite contiguous FP16 patches [868,1536]")
        if self._camera_uploads:
            # Split admission already uploaded each camera's rows under the
            # image-pool tail; a full copy would duplicate that work. A
            # partial set is an admission bug, never a fallback.
            if len(self._camera_uploads) != len(CAMERA_PATCH_ROWS):
                raise RuntimeError("pixel upload state is inconsistent: "
                                   f"{sorted(self._camera_uploads)} of "
                                   f"{list(range(len(CAMERA_PATCH_ROWS)))} cameras")
            return
        # Whole-buffer upload (non-split paths). No exponent scan: every
        # packed pixel is a finite _UINT8_TO_FP16 entry by construction
        # (see processing module import check).
        self.pixels.copy_from(pixels)

    def scatter_to(self, prefix_ddr, starts):
        """Prepare three device-only image-run writes into an embedding prefix."""
        if len(starts) != 3 or len(prefix_ddr.shape) != 2 or prefix_ddr.shape[1] != 2048:
            raise ValueError("three image runs and a DDR [prefix,2048] target are required")
        operations, offset, previous_end = [], 0, 0
        for start, rows in zip(starts, (77, 70, 70)):
            if (isinstance(start, bool) or not isinstance(start, int)
                    or start < previous_end or start + rows > prefix_ddr.shape[0]):
                raise ValueError("image runs must be ordered, disjoint and inside the prefix")
            source = self.context.view(self.output_ddr, (rows, 2048), offset_bytes=offset * 2048 * 2)
            target = self.context.view(prefix_ddr, (rows, 2048), offset_bytes=start * 2048 * 2)
            operations.append(self.context.prepare_copy(source, target, mode="shared"))
            offset += rows
            previous_end = start + rows
        return tuple(operations)

    def execute(self):
        for execution in self.executions:
            execution.execute().wait()
