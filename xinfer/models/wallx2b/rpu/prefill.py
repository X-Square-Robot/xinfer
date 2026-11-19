"""Cache-only Qwen3.5 TP8 prefix lowering with fused, bounded GDR state.

Every request starts at logical position zero. Three full-width SPM slots
cover all 24 layers; GDR scratch reuses dead normalized inputs, and its output
reuses consumed convolution pads. Only persistent outputs are copied to DDR.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np
import torch

from .config import FULL_ATTENTION_LAYERS


MAX_WORKSPACE_BYTES = 3 * 512 * 2048 * 2

# prefill_rope_tables is a pure function of the (extended) position bytes:
# identical input content always yields identical tables, so a content-keyed
# memo hit is byte-identical to a fresh build by construction. Entries are
# read-only (bind only copies FROM them); treat returned arrays as immutable.
_ROPE_TABLE_MEMO = {}
_ROPE_TABLE_MEMO_LIMIT = 64


def _rope_tables(positions):
    """FP64 frequency construction + lane interleave + FP16 tables."""
    inverse = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    frequencies = positions.to(torch.float64)[:, :, None] * inverse[None, None]
    merged = frequencies[0].clone()
    for lane, offset, length in ((1, 1, 33), (2, 2, 30)):
        merged[:, offset:length:3] = frequencies[lane, :, offset:length:3]
    return merged.cos().half().contiguous().numpy(), merged.sin().half().contiguous().numpy()


@torch.inference_mode()
def prefill_rope_tables(position_ids, *, prefix_length: int, bucket: int):
    """Token-indexed interleaved M-RoPE, matching the frozen FP64 CPU builder."""
    if (isinstance(prefix_length, bool) or not isinstance(prefix_length, int)
            or not 1 < prefix_length <= 512 or bucket != math.ceil(prefix_length / 64) * 64):
        raise ValueError("prefill requires complete prefix 2..512 and its ceil64 bucket")
    positions = torch.as_tensor(position_ids, device="cpu")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("position IDs must be int32/int64")
    if positions.ndim == 3 and positions.shape[1] == 1:
        positions = positions[:, 0]
    if positions.ndim != 2 or positions.shape[0] != 3 or positions.shape[1] not in (prefix_length, bucket):
        raise ValueError("prefill position IDs require [3,1,L] or [3,L] covering real/padded prefix")
    if bool((positions < 0).any()):
        raise ValueError("prefill positions cannot be negative")
    if positions.shape[1] < bucket:
        steps = torch.arange(1, bucket - positions.shape[1] + 1, dtype=positions.dtype)[None]
        positions = torch.cat((positions, positions[:, -1:] + steps), dim=1)
    positions = positions.contiguous()
    array = positions.numpy()
    key = (str(array.dtype), array.shape,
           hashlib.sha256(memoryview(array.reshape(-1).view(np.uint8))).hexdigest())
    cached = _ROPE_TABLE_MEMO.get(key)
    if cached is not None:
        return cached
    cosine, sine = _rope_tables(positions)
    if len(_ROPE_TABLE_MEMO) >= _ROPE_TABLE_MEMO_LIMIT:
        _ROPE_TABLE_MEMO.pop(next(iter(_ROPE_TABLE_MEMO)))
    _ROPE_TABLE_MEMO[key] = (cosine, sine)
    return cosine, sine


class PrefillState:
    """Small shared SPM state owners and persistent per-layer DDR results.

    Initial prefill kernels overwrite recurrent/conv state each layer/request.
    Persistent DDR snapshots retain every layer's final state without keeping
    eighteen mutually exclusive SPM recurrent workspaces live.
    """

    def __init__(self, context):
        self.context = context
        self.recurrent = context.empty((2, 128, 128), distribution="sharded", shard_axis=0, state_kind="state")
        self.history = context.empty((3, 4, 256), distribution="sharded", shard_axis=2, state_kind="history")
        self.history_out = context.empty((3, 4, 256), distribution="sharded", shard_axis=2, state_kind="history")
        self.norm_weight = context.empty((2048,), distribution="replicated")
        self.gdn_norm = context.empty((128,), distribution="replicated")
        self.dt = context.empty((16,), distribution="sharded", shard_axis=0)
        self.neg_a = context.empty((16,), distribution="sharded", shard_axis=0)
        self.cosine, self.sine = (context.empty((512, 32), distribution="replicated") for _ in range(2))
        self.gdn_states, self.conv_states = {}, {}
        self.key_caches, self.value_caches = {}, {}
        for layer in range(24):
            if layer in FULL_ATTENTION_LAYERS:
                self.key_caches[layer] = context.empty((8, 784, 256), storage="ddr", layout="causal_k_cache",
                                                      distribution="shared", state_kind="cache")
                self.value_caches[layer] = context.empty((8, 784, 256), storage="ddr", layout="causal_v_cache",
                                                        distribution="shared", state_kind="cache")
            else:
                self.gdn_states[layer] = context.empty((8, 2, 128, 128), storage="ddr", distribution="shared")
                self.conv_states[layer] = context.empty((8, 3, 4, 256), storage="ddr", distribution="shared")
        # Keep the existing SPM table owners and refresh them inside the main
        # prepared batch. Host binding writes each table once through shared DDR.
        self.cosine_ddr, self.sine_ddr = (
            context.empty((512, 32), storage="ddr", distribution="shared") for _ in range(2)
        )


class PrefillPlan:
    """A complete 24-layer prepared initial-prefix batch; 48 explicit reductions.

    ``prefix_length`` is a mutable execution binding within this plan's bucket.
    Reuse a ``PrefillState`` and workspace across signatures; bind every live
    request's M-RoPE tables before replay. Native state failures require the
    context's explicit recovery/rebuild boundary. No eager warmup runs here.
    """

    collective_count = 48

    def __init__(self, context, config, weights: Mapping, *, prefix_length: int, bucket=None,
                 workspace=None, input_ddr=None, state: PrefillState | None = None,
                 image_features=None, image_starts=None, token_embedding=True, collective_window=1,
                 whole_stage=False):
        if (isinstance(prefix_length, bool) or not isinstance(prefix_length, int)
                or not 1 < prefix_length <= 512):
            raise ValueError("prefill requires a complete prefix of 2..512 real tokens")
        expected_bucket = math.ceil(prefix_length / 64) * 64
        if bucket is not None and bucket != expected_bucket:
            raise ValueError("prefill bucket must be ceil64 of real length")
        if config.max_prefix_length != 512:
            raise ValueError("PrefillPlan requires the admitted Wall-X 2B FP16 TP8 configuration")
        if (isinstance(collective_window, bool) or not isinstance(collective_window, int)
                or not 1 <= collective_window <= 15):
            raise ValueError("Prefill supports collective_window in [1,15] "
                             "(the calibrated AR window boundary)")
        if not isinstance(whole_stage, bool):
            raise ValueError("Prefill whole_stage must be an explicit boolean")
        if whole_stage and collective_window != 1:
            raise ValueError("Prefill whole_stage replaces windowing; use collective_window1")
        if collective_window != 1:
            context.require_collective_window(collective_window)
        if whole_stage:
            context.require_batch_fused()
        self.collective_window = collective_window
        self.whole_stage = whole_stage
        self.context, self.config, self.weights = context, config, weights
        self.prefix_length, self.bucket = prefix_length, expected_bucket
        self.slot_bytes = self.bucket * 2048 * 2
        self.workspace_bytes = 3 * self.slot_bytes
        self.workspace = (workspace if workspace is not None
                          else context.empty((self.workspace_bytes // 2,), distribution="local"))
        if self.workspace.context is not context or math.prod(self.workspace.shape) * 2 < self.workspace_bytes:
            raise ValueError("prefill workspace is too small or belongs to another context")
        self.state = state if state is not None else PrefillState(context)
        if self.state.context is not context:
            raise ValueError("prefill state must belong to this context")
        self.input_ddr = (input_ddr if input_ddr is not None
                          else context.empty((self.bucket, 2048), storage="ddr", distribution="shared"))
        if self.input_ddr.shape != (self.bucket, 2048) or self.input_ddr.context is not context:
            raise ValueError("prefill input requires a stable DDR [bucket,2048] owner")
        self.cosine = context.view(self.state.cosine, (self.bucket, 32))
        self.sine = context.view(self.state.sine, (self.bucket, 32))
        self.cosine_ddr = context.view(self.state.cosine_ddr, (self.bucket, 32))
        self.sine_ddr = context.view(self.state.sine_ddr, (self.bucket, 32))
        self.operations = []
        self.layer_operations = []
        self.token_embedding = bool(token_embedding)
        self._bound = False
        self._valid_row_operations = []
        self._valid_row_length = prefix_length
        self.image_starts = None if image_starts is None else tuple(image_starts)
        if self.token_embedding:
            # Stable bucket-sized storage avoids new native queues/ID owners
            # when a request's real length changes inside the same bucket.
            self.ids = context.empty((self.bucket,), dtype="int32", storage="ddr", distribution="shared")
            self.ids.copy_from(np.zeros((self.bucket,), dtype=np.int32))
            rows = self.bucket // 8
            gathered = self._v(0, (rows, 2048), distribution="sharded", shard_axis=0)
            self._op("embedding", {"ids": self.ids, "vocab": self._weight("text.embedding"),
                                   "out": gathered}, integers={"valid_rows": prefix_length})
            packed = context.view(self.input_ddr, (8, rows, 2048))
            self._copy(gathered, packed, mode="corewise")
            if (image_features is None) != (image_starts is None):
                raise ValueError("provide both image features and three image run starts")
            if image_features is not None:
                if image_features.shape != (217, 2048) or len(image_starts) != 3:
                    raise ValueError("Wall-X 2B FP16 TP8 image features require [217,2048] and three run starts")
                offset, previous_end = 0, 0
                for start, count in zip(image_starts, (77, 70, 70)):
                    if (isinstance(start, bool) or not isinstance(start, int)
                            or start < previous_end or start + count > prefix_length):
                        raise ValueError("image runs must be ordered, disjoint and inside the real prefix")
                    source = context.view(image_features, (count, 2048), offset_bytes=offset * 2048 * 2)
                    target = context.view(self.input_ddr, (count, 2048), offset_bytes=start * 2048 * 2)
                    self._copy(source, target, mode="shared")
                    offset, previous_end = offset + count, start + count
            self.input_operations = tuple(self.operations)
            if whole_stage:
                # The input ops flow into the stage's single fused queue;
                # no separate input batch or init sweeps.
                self.input_executions = ()
            else:
                self.input_executions = (context.batch(self.input_operations),)
                self.operations = []
        else:
            if image_features is not None or image_starts is not None:
                raise ValueError("image scatter belongs to the explicit token-embedding input batch")
            self.input_operations, self.input_executions = (), ()
        self._copy(self.cosine_ddr, self.cosine, mode="broadcast")
        self._copy(self.sine_ddr, self.sine, mode="broadcast")
        self._copy(self.input_ddr, self._v(0, (self.bucket, 2048)), mode="broadcast")
        self._collective_occurrence = 0
        for layer in range(24):
            begin = len(self.operations)
            self._layer(layer)
            self.layer_operations.append(tuple(self.operations[begin:]))
        self.output = self._v(0, (self.bucket, 2048))  # 24 swaps return to slot A
        self._rms(self.output, self.output, "text.final_norm")
        self.operations = tuple(self.operations)
        if self.whole_stage:
            # One retained queue for the whole stage — input ops, rope
            # table and input broadcasts, the 24-layer body (48 ARs at
            # vendor pool ids 50..3 descending — the exact pool fit), and
            # the final norm — the original's prefill seg0 shape.
            self.execution = context.batch_fused(self.operations)
        else:
            self.execution = (context.batch(self.operations) if self.collective_window == 1 else
                              context.batch(self.operations, collective_window=self.collective_window))
        self.executions = (self.execution,)
        self.operation_count = len(self.operations)
        self.key_caches = tuple(self.state.key_caches[i] for i in FULL_ATTENTION_LAYERS)
        self.value_caches = tuple(self.state.value_caches[i] for i in FULL_ATTENTION_LAYERS)
        self.gdn_states, self.conv_states = self.state.gdn_states, self.state.conv_states

    def _v(self, slot, shape, *, byte_offset=0, distribution="replicated", shard_axis=-1):
        return self.context.view(self.workspace, shape, offset_bytes=slot * self.slot_bytes + byte_offset,
                                 distribution=distribution, shard_axis=shard_axis)

    def _op(self, name, tensors, *, integers=None, scalars=None, **selectors):
        operation = self.context.prepare(name, tensors=tensors, integers=integers, scalars=scalars,
                                         **selectors)
        self.operations.append(operation)
        if integers is not None and "valid_rows" in integers:
            self._valid_row_operations.append(operation)
        return operation

    def _copy(self, source, target, *, mode):
        self.operations.append(self.context.prepare_copy(source, target, mode=mode))

    def _next_counter(self):
        """Vendor pool id for the next counted AR (whole-stage mode).

        The 24-layer body holds exactly 48 all_reduce_residual collectives —
        the full vendor pool: occurrences 0..47 walk the descending 50..3
        in the original's per-occurrence static discipline.
        """
        occurrence = self._collective_occurrence
        self._collective_occurrence += 1
        return 50 - occurrence

    def _weight(self, name):
        if name not in self.weights:
            raise KeyError(f"missing prepared device weight {name}")
        return self.weights[name]

    def _param(self, key, target, *, mode="broadcast"):
        source = self._weight(key)
        if source.storage == "ddr":
            self._copy(source, target, mode=mode)
            return target
        return source

    def _rms(self, source, target, key):
        columns = source.shape[1]
        gamma = self.context.view(self.state.norm_weight, (columns,))
        gamma = self._param(key, gamma)
        # H=2048 text norms keep the coefficient-first product (different FP16 rounding).
        mode = "coefficient" if columns == 2048 else "default"
        self._op("rmsnorm", {"x": source, "out": target, "weight": gamma}, scalars={"epsilon": 1.0e-6},
                 mode=mode)

    def _layer(self, layer):
        m = self.bucket
        a, b, c = layer % 2, 1 - layer % 2, 2
        original, normalized = self._v(a, (m, 2048)), self._v(b, (m, 2048))
        mixed = self._v(c, (m, 2048))
        stem = f"text.{layer}."
        self._rms(original, normalized, stem + "input_layernorm")
        if layer in FULL_ATTENTION_LAYERS:
            branch = self._full_attention(layer, b, c)
            weight = self._weight(stem + "o")
        else:
            branch = self._gdn(layer, b, c)
            weight = self._weight(stem + "gdn_out")
        partial = self._v(b, (m, 2048), distribution="partial")
        self._op("gemm", {"x": branch, "out": partial, "w": weight}, parallel="row")
        self._op("all_reduce", {"x": partial, "residual": original, "out": mixed},
                 integers=({"with_residual": 1, "counter": self._next_counter()}
                           if self.whole_stage else {"with_residual": 1}))
        mlp_input = self._v(a, (m, 2048))
        self._rms(mixed, mlp_input, stem + "post_attention_layernorm")
        gate = self._v(b, (m, 768), distribution="sharded", shard_axis=1)
        up = self._v(b, (m, 768), byte_offset=m * 768 * 2, distribution="sharded", shard_axis=1)
        # The interleaved projection writes gate and up; SiLU-mul follows separately.
        self._op("gemm_glu", {"x": mlp_input,
            "out_gate": gate, "out_up": up, "w": self._weight(stem + "gateup")}, activation="none")
        self._op("silu_and_mul", {"x": gate, "y": up, "out": gate})
        down = self._v(a, (m, 2048), distribution="partial")
        self._op("gemm", {"x": gate, "out": down, "w": self._weight(stem + "down")}, parallel="row")
        self._op("all_reduce", {"x": down, "residual": mixed, "out": self._v(b, (m, 2048))},
                 integers=({"with_residual": 1, "counter": self._next_counter()}
                           if self.whole_stage else {"with_residual": 1}))

    def _gdn(self, layer, normalized_slot, data_slot):
        m, stem = self.bucket, f"text.{layer}."
        cursor, pads, convolved = 0, [], []
        for _ in range(3):
            pads.append(self._v(data_slot, (m + 3, 256), byte_offset=cursor, distribution="sharded", shard_axis=1))
            cursor += (m + 3) * 256 * 2
        for _ in range(3):
            convolved.append(self._v(data_slot, (m, 256), byte_offset=cursor, distribution="sharded", shard_axis=1))
            cursor += m * 256 * 2
        z = self._v(data_slot, (m, 256), byte_offset=cursor, distribution="sharded", shard_axis=1)
        cursor += m * 256 * 2
        b = self._v(data_slot, (m, 16), byte_offset=cursor, distribution="sharded", shard_axis=1)
        cursor += m * 16 * 2
        alpha = self._v(data_slot, (m, 16), byte_offset=cursor, distribution="sharded", shard_axis=1)
        cursor += m * 16 * 2
        if cursor > self.slot_bytes:
            raise MemoryError("GDN live producer buffers exceed the declared phase slot")
        projected = [self.context.view(pad, (m, 256), offset_bytes=3 * 256 * 2,
                                       distribution="sharded", shard_axis=1) for pad in pads]
        self._op("qkv_proj", {
            "x": self._v(normalized_slot, (m, 2048)), "out_q": projected[0], "out_k": projected[1],
            "out_v": projected[2], "out_z": z, "out_b": b, "out_a": alpha,
            "w": self._weight(stem + "qkv_zba")})
        conv = {"history": self.state.history, "history_out": self.state.history_out}
        for path, pad, output in zip(("q", "k", "v"), pads, convolved):
            conv[path + "pad"], conv[path + "out"] = pad, output
            conv[path + "weight"] = self._weight(stem + "conv_" + path)
        self._op("causal_conv1d", conv,
                 integers={"initial": 1, "valid_rows": self.prefix_length})
        self._copy(self.state.history_out, self.state.conv_states[layer], mode="corewise")
        dt = self._param(stem + "dt", self.state.dt, mode="corewise")
        neg_a = self._param(stem + "neg_a", self.state.neg_a, mode="corewise")
        norm = self._param(stem + "gdn_norm", self.state.gdn_norm)
        # Producer pads have reached their last consumer. GDR output may use
        # their storage; all Q/K/V/Z/A/B inputs and GDR scratch remain separate.
        output = self._v(data_slot, (m, 256), distribution="sharded", shard_axis=1)
        scratch = self._v(normalized_slot, (197632 * (m // 64) // 2,), distribution="local")
        self._op("gated_delta_rule_prefill", {"q": convolved[0], "k": convolved[1], "v": convolved[2],
            "z": z, "a": alpha, "b": b, "dt": dt, "neg_a": neg_a, "norm_weight": norm,
            "state": self.state.recurrent, "out": output, "scratch": scratch},
            integers={"initial": 1, "valid_rows": self.prefix_length})
        self._copy(self.state.recurrent, self.state.gdn_states[layer], mode="corewise")
        return output

    def _full_attention(self, layer, normalized_slot, data_slot):
        m, stem = self.bucket, f"text.{layer}."
        # Each local D256 plane is a complete effective attention head. K/V
        # replicas follow the checkpoint's head0 cores0..3/head1 cores4..7 map.
        q, gate, key, value, _staged, attended = [self._v(data_slot, (m, 256),
            byte_offset=i * m * 256 * 2, distribution="sharded", shard_axis=1) for i in range(6)]
        local_key = self._v(data_slot, (m, 256), byte_offset=2 * m * 256 * 2, distribution="local")
        local_value = self._v(data_slot, (m, 256), byte_offset=3 * m * 256 * 2, distribution="local")
        self._op("qkv_proj", {
            "x": self._v(normalized_slot, (m, 2048)), "out_q": q, "out_qgate": gate,
            "out_k": local_key, "out_v": local_value, "w": self._weight(stem + "qkv")})
        for plane, name in ((0, "q_norm"), (2, "k_norm")):
            local = self._v(data_slot, (m, 256), byte_offset=plane * m * 256 * 2, distribution="local")
            self._rms(local, local, stem + name)
            head = self._v(data_slot, (m, 1, 256), byte_offset=plane * m * 256 * 2,
                           distribution="sharded", shard_axis=1)
            self._op("rope", {"x": head, "out": head, "cos": self.cosine, "sin": self.sine},
                     integers={"tp8": 1})
        kc, vc = self.state.key_caches[layer], self.state.value_caches[layer]
        self._op("kv_cache_insert", {"input": key, "cache": kc}, integers={"position": 0})
        transposed = self._v(data_slot, (256, m), byte_offset=4 * m * 256 * 2,
                              distribution="sharded", shard_axis=0)
        self._op("kv_cache_insert", {"input": value, "value_spm": transposed, "cache": vc},
                 integers={"position": 0})
        self._op("attention", {"query": q, "key": key, "value": transposed, "out": attended},
                 integers={"kv_length": m})
        self._op("activation", {"x": gate, "out": gate}, function="sigmoid")
        self._op("binary", {"lhs": attended, "rhs": gate, "out": attended}, operation="mul")
        return attended

    def bind(self, position_ids, *, input_ids=None):
        self._bound = False
        if self.token_embedding:
            if isinstance(input_ids, torch.Tensor):
                if input_ids.device.type != "cpu":
                    raise ValueError("prefill input IDs must be on CPU")
                input_ids = input_ids.detach().contiguous().numpy()
            ids = np.asarray(input_ids)
            if (ids.ndim not in (1, 2) or (ids.ndim == 2 and ids.shape[0] != 1)
                    or ids.dtype not in (np.dtype("int32"), np.dtype("int64"))
                    or (ids < 0).any() or (ids >= 256277).any()):
                raise ValueError("prefill requires admitted integer IDs covering the full real prefix")
            length = int(ids.size)
        else:
            positions = torch.as_tensor(position_ids, device="cpu")
            length = int(positions.shape[-1]) if positions.ndim in (2, 3) else 0
        if not 1 < length <= 512 or math.ceil(length / 64) * 64 != self.bucket:
            raise ValueError("prefill binding must remain in this plan's ceil64 bucket")
        if self.image_starts is not None and self.image_starts[-1] + 70 > length:
            raise ValueError("image runs must remain inside the real prefix")
        cosine, sine = prefill_rope_tables(position_ids, prefix_length=length, bucket=self.bucket)
        # Bind only while idle. These patches preserve all addresses, entry
        # points, launch geometry and cache ownership; only valid token counts
        # and the kernels' corresponding state extents change.
        if self.token_embedding:
            padded = np.zeros((self.bucket,), dtype=np.int32)
            padded[:length] = ids.reshape(-1)
            self.ids.copy_from(padded)
        self.cosine_ddr.copy_from(cosine)
        self.sine_ddr.copy_from(sine)
        if length != self._valid_row_length:
            # A failed native validation can leave earlier operations patched.
            # Keep this dirty until the complete list succeeds so the next
            # binding restores every extent, including the previous length.
            self._valid_row_length = None
            for operation in self._valid_row_operations:
                operation.rebind(integers={"valid_rows": length})
            self._valid_row_length = length
        self.prefix_length = length
        self._bound = True

    def execute(self):
        if not self._bound:
            raise RuntimeError("bind current prefix token IDs and positions before execution")
        for execution in self.input_executions:
            execution.execute().wait()
        return self.execution.execute()
