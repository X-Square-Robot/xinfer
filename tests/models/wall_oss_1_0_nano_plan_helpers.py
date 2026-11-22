"""Shape/alias planning oracle; it deliberately cannot execute device graphs."""

import math
from types import SimpleNamespace

import numpy as np


class Buffer:
    def __init__(self, context, shape, *, storage="spm", layout="row_major", distribution="local",
                 shard_axis=-1, state_kind="none", dtype="float16", immutable=False, parent=None, offset=0):
        self.context, self.shape = context, tuple(shape)
        self.storage, self.layout, self.distribution = storage, layout, distribution
        self.shard_axis, self.state_kind = shard_axis, state_kind
        self.dtype, self.immutable = np.dtype(dtype), immutable
        self.parent = parent
        self.offset = (parent.offset if parent else 0) + offset
        self.allocation = parent.allocation if parent else object()
        self.uploaded = None

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.dtype.itemsize

    def copy_from(self, value):
        assert value.dtype == self.dtype
        assert value.shape in (self.shape, (8, *self.shape))
        self.uploaded = value.copy()


def disjoint(a, b):
    return (a.allocation is not b.allocation
            or a.offset + a.nbytes <= b.offset or b.offset + b.nbytes <= a.offset)


def catalog_call(op, tensors, integers=None, selectors=None, *, with_names=False):
    """Normalize a family call (gemm, rmsnorm, ...) to its catalog entry.

    Uses the operator backend's host-side resolver, so these planning fakes keep asserting
    the catalog names and argument names the device receives. A catalog call
    is returned unchanged. Returns (entry, tensors by catalog name, integers),
    plus the caller-name -> catalog-name map when with_names is set.
    """
    integers = dict(integers or {})
    selectors = dict(selectors or {})
    resolver = native_resolver()
    if not selectors and op not in resolver.FAMILIES:
        result = (op, dict(tensors), integers)
        names = {name: name for name in tensors}
    else:
        entry, names, resolved = resolver.resolve(op, tensors, integers, **selectors)
        result = (entry, {names[name]: tensor for name, tensor in tensors.items()}, resolved)
    return (*result, names) if with_names else result


def native_resolver():
    """Load the in-tree host resolver; missing native dependencies are explicit skips."""
    import pytest

    try:
        from xinfer.ops.backends import rpu
        from xinfer.ops.backends.rpu import rpu_tp
        rpu.load_backend()
    except (ImportError, FileNotFoundError) as error:
        pytest.skip("RPU planning tests need the in-tree host family resolver; build xinfer_rpu.so and set "
                    f"XINFER_RPU_LIBRARY ({type(error).__name__}: {error})")
    return rpu_tp


class Operation(SimpleNamespace):
    """Metadata patching for planner tests; no native plan or execution."""

    def rebind(self, *, tensors=None, integers=None, scalars=None):
        if tensors:
            self.tensors.update(tensors)
        if integers:
            self.integers.update(integers)
        if scalars:
            self.scalars.update(scalars)
        return self


class PlanContext:
    """Independent native-contract subset for real lowered tensor bindings."""

    def __init__(self):
        self.spm_bytes = 0
        self.prepared = []
        self.batches = []

    def empty(self, shape, **description):
        result = Buffer(self, shape, **description)
        if result.storage == "spm":
            self.spm_bytes += math.ceil(result.nbytes / 256) * 256
            if self.spm_bytes > 7 * 1024 * 1024:
                raise MemoryError("7 MiB per-core allocation budget exceeded")
        return result

    def prepare_weight(self, value, *, layout):
        return Buffer(self, value.shape, storage="ddr", distribution="shared", layout=layout, immutable=True)

    def view(self, parent, shape, *, offset_bytes=0, distribution=None, shard_axis=-1):
        assert offset_bytes % 8 == 0
        assert offset_bytes + math.prod(shape) * parent.dtype.itemsize <= parent.nbytes
        assert parent.state_kind == "none" and parent.layout == "row_major"
        return Buffer(self, shape, storage=parent.storage, layout=parent.layout,
                      distribution=parent.distribution if distribution is None else distribution,
                      shard_axis=shard_axis, dtype=parent.dtype, immutable=parent.immutable,
                      parent=parent, offset=offset_bytes)

    def prepare_copy(self, source, target, *, mode):
        assert source.dtype == target.dtype
        assert disjoint(source, target)
        if mode == "broadcast":
            assert source.storage == "ddr" and target.storage == "spm"
            assert source.shape == target.shape and target.distribution == "replicated"
        elif mode == "corewise":
            if source.storage == "spm" and target.storage == "ddr":
                assert target.shape == (8, *source.shape)
            elif source.storage == "ddr" and target.storage == "spm":
                assert source.shape == (8, *target.shape)
                assert target.distribution != "replicated"
            else:
                assert source.shape == target.shape
        elif mode == "core0":
            assert source.shape == target.shape
            if target.storage == "spm":
                assert target.distribution in ("single", "owner0")
            else:
                assert source.distribution in ("single", "replicated")
        elif mode == "shared":
            assert source.storage == target.storage == "ddr" and source.shape == target.shape
        else:
            raise AssertionError(mode)
        result = Operation(name="copy_f16", tensors={"input": source, "out": target},
                           integers={"mode": mode}, scalars={})
        self.prepared.append(result)
        return result

    def prepare(self, name, *, tensors, integers=None, scalars=None, **selectors):
        name, tensors, integers = catalog_call(name, tensors, integers, selectors)
        i, s, t = integers or {}, scalars or {}, tensors
        if name == "layernorm_f16":
            assert t["input"].shape == t["out"].shape
            assert t["weight"].shape == t["bias"].shape == (t["input"].shape[-1],)
            assert disjoint(t["input"], t["out"])
        elif name == "all_reduce_residual":
            assert t["input"].distribution == "partial" and t["out"].distribution == "replicated"
            assert t["input"].shape == t["out"].shape and disjoint(t["input"], t["out"])
            if i.get("with_residual"):
                assert t["residual"].shape == t["out"].shape and disjoint(t["residual"], t["out"])
        elif name == "all_reduce_residual_layernorm":
            assert t["x"].distribution == "sequence_shard" and t["x"].shard_axis == 0
            assert t["norm"].distribution == "replicated"
            assert all(disjoint(t["x"], t[key]) for key in ("partial", "residual", "norm"))
            assert disjoint(t["partial"], t["residual"])
            assert t["residual"].distribution in ("replicated", "sequence_shard")
            assert s["epsilon"] == 1e-6
        elif name == "conv_silu_prefill_f16":
            sources = [t[p + "pad"] for p in ("q", "k", "v")] + [t["history"]]
            outputs = [t[p + "out"] for p in ("q", "k", "v")] + [t["history_out"]]
            assert all(disjoint(a, b) for a in sources for b in outputs)
            assert i["initial"] == 1 and 1 <= i["valid_rows"] <= t["qout"].shape[0]
        elif name == "gdr_prefill_f16":
            sources = [t[k] for k in ("q", "k", "v", "z", "a", "b", "dt", "neg_a", "norm_weight", "state")]
            assert all(disjoint(t["out"], item) and disjoint(t["scratch"], item) for item in sources)
            assert disjoint(t["out"], t["scratch"])
            assert t["scratch"].nbytes == 197632 * (t["q"].shape[0] // 64)
        elif name == "causal_sdpa_prefix_spm_d256_f16":
            values = [t[k] for k in ("query", "key", "value", "out")]
            assert all(disjoint(a, b) for index, a in enumerate(values) for b in values[index + 1:])
            assert i["kv_length"] == t["query"].shape[0]
        elif name.startswith("fc1_"):
            source = t["x"] if "x" in t else t["src"]
            assert source.shape[0] <= 147 and disjoint(source, t["out"])
        elif name in ("qkv_rope_bias_n384k1024", "qkv_bias_n384k1024"):
            src = t.get("x", t.get("src"))
            # qkv_rope_bias_n384k1024 takes q/k/v; qkv_bias_n384k1024 takes out_q/out_k/out_v.
            outputs = [t[k] for k in (("q", "k", "v") if "_rope_" in name else ("out_q", "out_k", "out_v"))]
            assert all(disjoint(src, out) for out in outputs)
            assert all(disjoint(a, b) for index, a in enumerate(outputs) for b in outputs[index + 1:])
        assert name != "rmsnorm_gdn_f16", "device-produced GDR must use the validated fused boundary"
        result = Operation(name=name, tensors=t, integers=i, scalars=s)
        self.prepared.append(result)
        return result

    def batch(self, operations):
        operations = tuple(operations)
        count = sum(op.name in ("all_reduce_residual", "all_gather_f16", "all_reduce_residual_layernorm")
                    for op in operations)
        assert 1 <= len(operations) <= 1024 and count <= 48
        result = SimpleNamespace(operations=operations, collectives=count)
        self.batches.append(result)
        return result


def parameter(ctx, shape):
    return Buffer(ctx, shape, storage="ddr", distribution="shared", immutable=True)


def vision_weights(ctx):
    out = {"vision.patch": Buffer(ctx, (8, 128, 1536), storage="ddr", layout="tp8_nk16", distribution="shared"),
           "vision.patch_bias": parameter(ctx, (8, 128))}
    for layer in range(24):
        stem = f"vision.{layer}."
        for name, shape in (("qkv", (8, 384, 1024)), ("o", (8, 1024, 128)),
                            ("up", (8, 512, 1024)), ("down", (8, 1024, 512))):
            out[stem + name] = Buffer(ctx, shape, storage="ddr", layout="tp8_nk16", distribution="shared")
        for name, shape in (("qkv_bias", (8, 384)), ("o_bias", (8, 1024)),
                            ("up_bias", (8, 512)), ("down_bias", (8, 1024))):
            out[stem + name] = parameter(ctx, shape)
        for norm in ("norm1", "norm2"):
            for name in ("weight", "bias"):
                out[stem + norm + "." + name] = parameter(ctx, (1024,))
    for name, shape in (("linear_fc1", (8, 512, 4096)), ("linear_fc2", (8, 2048, 512))):
        out["vision.merger." + name] = Buffer(ctx, shape, storage="ddr", layout="tp8_nk16", distribution="shared")
        out["vision.merger." + name + "_bias"] = parameter(ctx, (8, shape[1]))
    for name in ("weight", "bias"):
        out["vision.merger.norm." + name] = parameter(ctx, (1024,))
    return out


def text_weights(ctx):
    from xinfer.models.wall_oss_1_0_nano.rpu.config import FULL_ATTENTION_LAYERS
    out = {"text.embedding": parameter(ctx, (256277, 2048)), "text.final_norm": parameter(ctx, (2048,))}
    for layer in range(24):
        stem = f"text.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            out[stem + name] = parameter(ctx, (2048,))
        for name, shape in (("gateup", (8, 1536, 2048)), ("down", (8, 2048, 768))):
            out[stem + name] = Buffer(ctx, shape, storage="ddr", layout="tp8_nk16", distribution="shared")
        if layer in FULL_ATTENTION_LAYERS:
            for name, shape in (("qkv", (8, 1024, 2048)), ("o", (8, 2048, 256))):
                out[stem + name] = Buffer(ctx, shape, storage="ddr", layout="tp8_nk16", distribution="shared")
            for name in ("q_norm", "k_norm"):
                out[stem + name] = parameter(ctx, (256,))
        else:
            out[stem + "qkv_zba"] = Buffer(ctx, (8, 1056, 2048), storage="ddr",
                                            layout="tp8_nk16", distribution="shared")
            out[stem + "gdn_out"] = Buffer(ctx, (8, 2048, 256), storage="ddr", layout="tp8_nk16", distribution="shared")
            for name in ("q", "k", "v"):
                out[stem + "conv_" + name] = parameter(ctx, (8, 4, 256))
            for name in ("dt", "neg_a"):
                out[stem + name] = parameter(ctx, (8, 16))
            out[stem + "gdn_norm"] = parameter(ctx, (128,))
    return out
