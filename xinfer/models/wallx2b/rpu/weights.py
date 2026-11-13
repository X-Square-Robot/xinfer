"""One-time model transforms into xcompute's *logical* TP8 weight inputs.

This module chooses column/row partition, replicas, bias ownership and profile
scales. xcompute owns the subsequent physical ``tp8_nk16``/``nk16`` packing
and device allocation. No RhinoForge, vendor binary or alternate FFI is used.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch

from .config import CheckpointManifest, FULL_ATTENTION_LAYERS


def _half(value: torch.Tensor, name: str) -> torch.Tensor:
    value = value.detach().to(device="cpu", dtype=torch.float16).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"wallx2b {name} is not finite after FP16 conversion")
    return value


def column_shard(weight: torch.Tensor, *, tp: int = 8) -> torch.Tensor:
    """Global [N,K] to natural [core,N/tp,K], without physical swizzling."""
    if weight.ndim != 2 or weight.shape[0] % tp:
        raise ValueError("column shard requires a matrix with N divisible by TP")
    return weight.reshape(tp, weight.shape[0] // tp, weight.shape[1]).contiguous()


def row_shard(weight: torch.Tensor, *, tp: int = 8) -> torch.Tensor:
    """Global [N,K] to [core,N,K/tp]; each result remains a partial sum."""
    if weight.ndim != 2 or weight.shape[1] % tp:
        raise ValueError("row shard requires a matrix with K divisible by TP")
    return weight.reshape(weight.shape[0], tp, weight.shape[1] // tp).permute(1, 0, 2).contiguous()


def split_query_gate(weight: torch.Tensor, *, heads: int = 8, head_dim: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint rows are [head0 Q,head0 gate,head1 Q,head1 gate,...]."""
    if weight.ndim != 2 or weight.shape[0] != heads * 2 * head_dim:
        raise ValueError("invalid per-head interleaved query/gate shape")
    value = weight.reshape(heads, 2, head_dim, weight.shape[1])
    return value[:, 0].reshape(heads * head_dim, -1).contiguous(), value[:, 1].reshape(
        heads * head_dim, -1
    ).contiguous()


def replicate_kv(weight: torch.Tensor, *, heads: int = 2, head_dim: int = 256, tp: int = 8) -> torch.Tensor:
    """Head0 is owned by cores0..3 and head1 by cores4..7, one full head/core."""
    if weight.ndim != 2 or weight.shape[0] != heads * head_dim or tp % heads:
        raise ValueError("KV replication requires complete heads and TP divisible by KV heads")
    return weight.reshape(heads, head_dim, -1).repeat_interleave(tp // heads, dim=0).contiguous()


def corewise_concat(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    if not parts or any(p.ndim != 3 or p.shape[0] != 8 or p.shape[2] != parts[0].shape[2] for p in parts):
        raise ValueError("corewise concatenation requires parallel [8,N,K] parts")
    return torch.cat(tuple(parts), dim=1).contiguous()


def full_qkv(
    query_gate: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, compact: bool = False
) -> torch.Tensor:
    query, gate = split_query_gate(query_gate)
    if compact:
        # Natural [2,256,K] -> [8,64,K] preserves (head=c//4,slice=c%4).
        key, value = column_shard(key), column_shard(value)
    else:
        key, value = replicate_kv(key), replicate_kv(value)
    return corewise_concat((column_shard(query), column_shard(gate), key, value))


def gdn_qkv_zba(qkv: torch.Tensor, z: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    if qkv.ndim != 2 or qkv.shape[0] != 6144 or z.shape != (2048, qkv.shape[1]):
        raise ValueError("GDN requires qkv[6144,H] and z[2048,H]")
    if b.shape != (16, qkv.shape[1]) or a.shape != b.shape:
        raise ValueError("GDN requires b/a[16,H]")
    padded = []
    for scalar_projection in (b, a):
        out = scalar_projection.new_zeros((8, 16, qkv.shape[1]))
        out[:, :2] = scalar_projection.reshape(8, 2, -1)
        padded.append(out)
    return corewise_concat(tuple(column_shard(p) for p in (*qkv.chunk(3, dim=0), z)) + tuple(padded))


def owner0_bias(bias: torch.Tensor) -> torch.Tensor:
    if bias.ndim != 1:
        raise ValueError("row-parallel bias must be a vector")
    out = bias.new_zeros((8, bias.shape[0]))
    out[0] = bias
    return out


def gdn_prefill_scalar(source: torch.Tensor, *, neg_exp: bool = False) -> torch.Tensor:
    """Public GDR consumes16 lanes/core; exactly the first2 are valid heads.

    The reference converts A_log to FP16 before CPU FP32 exp and a final FP16
    store. Padding of A_log is zero, so its neg-exp padding is -1.
    """
    if tuple(source.shape) != (16,):
        raise ValueError("Wall GDN scalar must have16 checkpoint heads")
    source = _half(source, "GDN scalar")
    out = source.new_zeros((8, 16))
    out[:, :2] = source.reshape(8, 2)
    return _half(-torch.exp(out.float()), "GDN neg-exp-A") if neg_exp else out


def zero_centered_gamma(source: torch.Tensor) -> torch.Tensor:
    # RhinoForge loads the source to half first, then computes 1+w in half.
    return (_half(source, "zero-centered norm") + 1.0).contiguous()


def action_input_weight(source: torch.Tensor) -> torch.Tensor:
    if tuple(source.shape) != (1024, 52):
        raise ValueError("Wall action input weight must be [1024,52]")
    return torch.nn.functional.pad(_half(source.float() * (4.0 / 256.0), "W1"), (0, 12))


def action_down_weight(source: torch.Tensor) -> torch.Tensor:
    if tuple(source.shape) != (1024, 2048):
        raise ValueError("Wall action down weight must be [1024,2048]")
    return row_shard(_half(source.float() / 4.0, "Action down"))


def euler_schedule() -> tuple[torch.Tensor, float]:
    """Saved scheduler steps10 remain distinct from the fixed serving NFE8."""
    times = torch.linspace(0.0, 1.0, 9, dtype=torch.float32) * 0.999
    steps = (times[1:] - times[:-1]).half()
    if not bool(steps.eq(steps[0]).all()):
        raise RuntimeError("Wall FP16 Euler schedule must have a uniform rounded dt")
    return times, float(steps[0])


def pack_adaptive_rows(condition: torch.Tensor, rows: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """CPU FP32 projections with a single FP16 boundary for all49 Ada rows."""
    if condition.device.type != "cpu" or tuple(condition.shape) != (1, 1024) or len(rows) != 49:
        raise ValueError("Wall Ada requires CPU condition[1,1024] and49 projection rows")
    packed = []
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        for index, (weight, bias) in enumerate(rows):
            if weight.device.type != "cpu" or bias.device.type != "cpu":
                raise ValueError("Wall Ada projections must remain on CPU")
            if tuple(weight.shape) != (3072, 1024) or tuple(bias.shape) != (3072,):
                raise ValueError("Wall Ada requires weight[3072,1024],bias[3072]")
            if weight.dtype != torch.float32 or bias.dtype != torch.float32 or condition.dtype != torch.float32:
                raise ValueError("Wall Ada condition and projections must be FP32")
            scale, shift, gate = torch.nn.functional.linear(condition, weight, bias).chunk(3, dim=-1)
            if index % 2:
                gate = gate * 4.0
            packed.append(torch.cat((scale + 1.0, shift, gate / 256.0), dim=-1))
    return _half(torch.stack(packed)[:, 0], "Wall Ada schedule")


@dataclass(frozen=True)
class SourceTensor:
    key: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str


@dataclass(frozen=True)
class PreparedWeight:
    name: str
    array: np.ndarray
    sources: tuple[SourceTensor, ...]
    layout: str
    distribution: str
    shard_axis: int | None = None
    transform: str = "FP32 source -> FP16"
    kind: str = "weight"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(memoryview(self.array).cast("B")).hexdigest()

    def manifest_row(self) -> dict:
        return dict(
            name=self.name,
            shape=list(self.array.shape),
            dtype=str(self.array.dtype),
            sha256=self.sha256,
            layout=self.layout,
            distribution=self.distribution,
            shard_axis=self.shard_axis,
            transform=self.transform,
            kind=self.kind,
            sources=[dict(key=s.key, shape=list(s.shape), dtype=s.dtype, sha256=s.sha256) for s in self.sources],
        )


class WeightPreparer:
    """Stream one admitted checkpoint once; prepared values own immutable arrays.

    Upload each yielded record during model preparation and retain the backend
    owner. The iterator does not retain all weights or invoke any request path.
    A failed preparation must be discarded and constructed again from the
    unchanged admitted checkpoint. Host conditioning tensors are loaded lazily
    and cached by ``host_tensors``; they are never packed as device GEMM weights.
    """

    def __init__(self, manifest: CheckpointManifest):
        if not isinstance(manifest, CheckpointManifest):
            raise TypeError("WeightPreparer requires an admitted CheckpointManifest")
        manifest.assert_unchanged()
        self.manifest = manifest
        self._started = False
        self._host: dict[str, torch.Tensor] | None = None
        self._source_info: dict[str, SourceTensor] = {}

    def _read(self, key: str, *, half: bool = True) -> torch.Tensor:
        # Keep each file mapping local to this source. Hash the original FP32
        # bytes before conversion; FP32 consumers require independent storage.
        from safetensors import safe_open

        with safe_open(str(self.manifest.root / "model.safetensors"), framework="pt", device="cpu") as handle:
            source = handle.get_tensor(key).detach().contiguous()
            if source.device.type != "cpu" or source.dtype != torch.float32:
                raise ValueError(f"wallx2b source {key} must be CPU FP32")
            self._source_info[key] = SourceTensor(
                key,
                tuple(source.shape),
                str(source.dtype),
                hashlib.sha256(memoryview(source.numpy()).cast("B")).hexdigest(),
            )
            return _half(source, key) if half else source.clone()

    def _record(
        self,
        name: str,
        value: torch.Tensor,
        sources: Sequence[str],
        *,
        layout: str = "tp8_nk16",
        distribution: str = "shared",
        shard_axis: int | None = None,
        transform: str = "FP32 source -> FP16",
        kind: str = "weight",
    ) -> PreparedWeight:
        if value.dtype != torch.float16 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"prepared {name} must be finite FP16")
        array = value.contiguous().numpy().copy()
        array.flags.writeable = False
        return PreparedWeight(
            name, array, tuple(self._source_info[k] for k in sources), layout, distribution, shard_axis, transform, kind
        )

    def host_tensors(self) -> dict[str, torch.Tensor]:
        """CPU FP32 time/Ada/normalizers; FP16 vision positional table."""
        if self._host is None:
            self.manifest.assert_unchanged()
            from safetensors import safe_open

            result = {}
            with safe_open(str(self.manifest.root / "model.safetensors"), framework="pt", device="cpu") as handle:
                keys = tuple(handle.keys())
            for key in keys:
                if (
                    key.startswith("model.action_processor.")
                    and not key.endswith(("w1.weight", "action_proj_back.weight"))
                ) or ".dense." in key:
                    value = self._read(key, half=False)
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"host conditioning tensor {key} must be finite")
                    result[key] = value
            key = "model.visual.pos_embed.weight"
            result[key] = self._read(key)
            self.manifest.assert_unchanged()
            self._host = result
        return dict(self._host)

    def release_host_tensors(self):
        """Release the FP32 conditioning cache after owned derived data exists."""
        self._host = None

    def load_normalizers(self) -> dict[str, np.ndarray]:
        host = self.host_tensors()
        result = {}
        for name, stem in (("state", "propri"), ("action", "action")):
            for statistic in ("min", "delta"):
                value = host[f"model.action_processor.normalizer_{stem}.{statistic}.x2_normal"]
                expected = (16, 26) if name == "action" else (26,)
                if tuple(value.shape) != expected or (statistic == "delta" and bool(value.eq(0).any())):
                    raise ValueError(f"invalid {name} {statistic} normalizer")
                array = value.numpy().copy()
                array.flags.writeable = False
                result[f"{name}_{statistic}"] = array
        return result

    def iter_prepared(self, *, compact_language_qkv: bool = False) -> Iterator[PreparedWeight]:
        if self._started:
            raise RuntimeError("wallx2b preparation was already attempted; construct a fresh WeightPreparer")
        self._started = True
        self.manifest.assert_unchanged()
        from safetensors import safe_open

        with safe_open(str(self.manifest.root / "model.safetensors"), framework="pt", device="cpu") as handle:
            checkpoint_keys = tuple(handle.keys())

        def read(key):
            return self._read(key)

        record = self._record
        # The embedding rows are checkpoint-owned (256277), distinct from
        # the saved text_config vocabulary (248320). No LM head is emitted.
        key = "model.language_model.embed_tokens.weight"
        yield record("text.embedding", read(key), (key,), layout="row_major")
        key = "model.language_model.norm.weight"
        yield record(
            "text.final_norm",
            (read(key) + 1.0),
            (key,),
            layout="row_major",
            distribution="replicated",
            kind="parameter",
            transform="FP32 -> FP16 -> FP16(1+w)",
        )
        for i in range(24):
            stem = f"model.language_model.layers.{i}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                key = f"{stem}.{norm}.weight"
                yield record(
                    f"text.{i}.{norm}",
                    read(key) + 1.0,
                    (key,),
                    layout="row_major",
                    distribution="replicated",
                    kind="parameter",
                    transform="FP32 -> FP16 -> FP16(1+w)",
                )
            key = f"{stem}.mlp.gate_up_proj.weight"
            gate, up = read(key).chunk(2, dim=0)
            yield record(
                f"text.{i}.gateup",
                corewise_concat((column_shard(gate), column_shard(up))),
                (key,),
                transform="FP16; corewise [gate768,up768]",
            )
            key = f"{stem}.mlp.down_proj.weight"
            yield record(
                f"text.{i}.down",
                row_shard(read(key)),
                (key,),
                transform="FP16 row shard; output partial, explicit reduce required",
            )
            if i in FULL_ATTENTION_LAYERS:
                attention = f"{stem}.self_attn"
                keys = tuple(f"{attention}.{p}_proj.weight" for p in ("q", "k", "v"))
                yield record(
                    f"text.{i}.qkv",
                    full_qkv(*(read(k) for k in keys), compact=compact_language_qkv),
                    keys,
                    transform="FP16; per-head Q/gate split; corewise Q,gate,K,V; "
                    + ("KV64 slices" if compact_language_qkv else "KV complete-head replicas"),
                )
                key = f"{attention}.o_proj.weight"
                yield record(
                    f"text.{i}.o",
                    row_shard(read(key)),
                    (key,),
                    transform="FP16 row shard; output partial, explicit reduce required",
                )
                for norm in ("q_norm", "k_norm"):
                    key = f"{attention}.{norm}.weight"
                    yield record(
                        f"text.{i}.{norm}",
                        read(key) + 1.0,
                        (key,),
                        layout="row_major",
                        distribution="replicated",
                        kind="parameter",
                        transform="FP32 -> FP16 -> FP16(1+w)",
                    )
            else:
                attention = f"{stem}.linear_attn"
                keys = tuple(f"{attention}.in_proj_{p}.weight" for p in ("qkv", "z", "b", "a"))
                yield record(
                    f"text.{i}.qkv_zba",
                    gdn_qkv_zba(*(read(k) for k in keys)),
                    keys,
                    transform="FP16; corewise Q256,K256,V256,Z256,B16,A16; B/A only first2 valid",
                )
                key = f"{attention}.out_proj.weight"
                yield record(
                    f"text.{i}.gdn_out",
                    row_shard(read(key)),
                    (key,),
                    transform="FP16 row shard; output partial, explicit reduce required",
                )
                key = f"{attention}.conv1d.weight"
                conv = read(key).squeeze(1)
                for p, value in zip(("q", "k", "v"), conv.chunk(3, dim=0)):
                    value = value.reshape(8, 256, 4).transpose(1, 2).contiguous()
                    yield record(
                        f"text.{i}.conv_{p}",
                        value,
                        (key,),
                        layout="row_major",
                        transform="FP16 [heads,channels,K4] -> [core,K4,256]",
                    )
                for name in ("A_log", "dt_bias"):
                    key = f"{attention}.{name}"
                    value = gdn_prefill_scalar(read(key), neg_exp=name == "A_log")
                    target = "neg_a" if name == "A_log" else "dt"
                    yield record(
                        f"text.{i}.{target}",
                        value,
                        (key,),
                        layout="row_major",
                        distribution="sharded",
                        shard_axis=0,
                        kind="parameter",
                        transform="FP32->FP16"
                        + (" -> CPU FP32 -exp -> FP16" if name == "A_log" else "")
                        + "; corewise first2 of16 lanes valid",
                    )
                key = f"{attention}.norm.weight"
                yield record(
                    f"text.{i}.gdn_norm",
                    read(key),
                    (key,),
                    layout="row_major",
                    distribution="replicated",
                    kind="parameter",
                    transform="FP16 plain gated-norm gamma (no +1)",
                )

        for i in range(24):
            stem = f"model.language_model.layers.{i}"
            key = f"{stem}.mlp_action_expert.gate_up_proj.weight"
            gate, up = read(key).chunk(2, dim=0)
            yield record(f"action.{i}.gate", column_shard(gate), (key,), transform="FP16 column shard256")
            yield record(f"action.{i}.up", column_shard(up), (key,), transform="FP16 column shard256")
            key = f"{stem}.mlp_action_expert.down_proj.weight"
            down = action_down_weight(self._read(key, half=False))
            yield record(
                f"action.{i}.down",
                down,
                (key,),
                transform="FP32 source /4 -> FP16 row shard; odd Ada gate restores x4",
            )
            if i in FULL_ATTENTION_LAYERS:
                attention = f"{stem}.self_attn"
                keys = tuple(f"{attention}.{p}_proj_action.weight" for p in ("q", "k", "v"))
                query, gate = split_query_gate(read(keys[0]))
                planes = torch.stack(
                    (
                        column_shard(query),
                        replicate_kv(read(keys[1])),
                        replicate_kv(read(keys[2])),
                        column_shard(gate),
                    )
                )
                yield record(
                    f"action.{i}.qkv",
                    planes,
                    keys,
                    transform="FP16 [Q,K,V,Qgate] planes; KV heads replicated groupwise",
                )
                key = f"{attention}.o_proj_action.weight"
                yield record(
                    f"action.{i}.o",
                    row_shard(read(key)),
                    (key,),
                    transform="FP16 row shard; output partial, explicit reduce required",
                )
                keys = tuple(f"{attention}.{p}_norm_action.weight" for p in ("q", "k"))
                yield record(
                    f"action.{i}.qk_norm",
                    torch.stack(tuple(read(k) + 1.0 for k in keys)),
                    keys,
                    layout="row_major",
                    distribution="replicated",
                    kind="parameter",
                    transform="FP32 -> FP16 -> FP16(1+w), [Q,K]",
                )
            # The other eighteen layers have identity token mixing. They
            # intentionally retain both Ada rows, the MLP and its residual.
        key = "model.action_processor.w1.weight"
        weight = action_input_weight(self._read(key, half=False))
        yield record(
            "action.input",
            weight,
            (key,),
            layout="nk16",
            transform="FP32 W1*(4/256) -> FP16; K52->64 zero pad; kernel retains separate /4 rounding",
        )
        key = "model.action_processor.action_proj_back.weight"
        yield record(
            "action.output",
            torch.nn.functional.pad(read(key), (0, 0, 0, 38)),
            (key,),
            layout="nk16",
            transform="FP16; N26->64 zero pad; final Ada restores physical scale",
        )

        key = "model.visual.patch_embed.proj.weight"
        patch = read(key).reshape(1024, 1536)
        yield record(
            "vision.patch",
            column_shard(patch),
            (key,),
            transform="FP16 Conv3d[Cout,Cin,T,H,W] flatten to [1024,1536]; column shard",
        )
        key = "model.visual.patch_embed.proj.bias"
        yield record(
            "vision.patch_bias",
            read(key).reshape(8, 128),
            (key,),
            layout="row_major",
            distribution="sharded",
            shard_axis=0,
            kind="parameter",
        )
        for i in range(24):
            stem = f"model.visual.blocks.{i}"
            key = f"{stem}.attn.qkv.weight"
            parts = read(key).chunk(3, dim=0)
            yield record(
                f"vision.{i}.qkv",
                corewise_concat(tuple(column_shard(v) for v in parts)),
                (key,),
                transform="FP16; corewise [Q128,K128,V128]",
            )
            key = f"{stem}.attn.qkv.bias"
            bias = read(key).reshape(3, 8, 128).permute(1, 0, 2).reshape(8, 384)
            yield record(
                f"vision.{i}.qkv_bias", bias, (key,), layout="row_major", distribution="local", kind="parameter"
            )
            for name, source, partition in (
                ("o", "attn.proj", "row"),
                ("up", "mlp.linear_fc1", "column"),
                ("down", "mlp.linear_fc2", "row"),
            ):
                key = f"{stem}.{source}.weight"
                value = read(key)
                value = row_shard(value) if partition == "row" else column_shard(value)
                yield record(
                    f"vision.{i}.{name}",
                    value,
                    (key,),
                    transform=f"FP16 {partition} shard"
                    + ("; explicit reduce required" if partition == "row" else ""),
                )
                key = f"{stem}.{source}.bias"
                value = read(key)
                value = owner0_bias(value) if partition == "row" else value.reshape(8, -1)
                yield record(
                    f"vision.{i}.{name}_bias",
                    value,
                    (key,),
                    layout="row_major",
                    distribution="owner0" if partition == "row" else "sharded",
                    shard_axis=0 if partition == "column" else None,
                    kind="parameter",
                    transform="FP16; bias once on core0" if partition == "row" else "FP16 column shard",
                )
            for norm in ("norm1", "norm2"):
                for parameter in ("weight", "bias"):
                    key = f"{stem}.{norm}.{parameter}"
                    yield record(
                        f"vision.{i}.{norm}.{parameter}",
                        read(key),
                        (key,),
                        layout="row_major",
                        distribution="replicated",
                        kind="parameter",
                    )
        for name, partition in (("linear_fc1", "column"), ("linear_fc2", "row")):
            key = f"model.visual.merger.{name}.weight"
            value = read(key)
            yield record(
                f"vision.merger.{name}",
                column_shard(value) if partition == "column" else row_shard(value),
                (key,),
                transform=f"FP16 merger {partition} shard",
            )
            key = f"model.visual.merger.{name}.bias"
            value = read(key)
            yield record(
                f"vision.merger.{name}_bias",
                value.reshape(8, -1) if partition == "column" else owner0_bias(value),
                (key,),
                layout="row_major",
                distribution="sharded" if partition == "column" else "owner0",
                shard_axis=0 if partition == "column" else None,
                kind="parameter",
            )
        for name in ("weight", "bias"):
            key = f"model.visual.merger.norm.{name}"
            yield record(
                f"vision.merger.norm.{name}",
                read(key),
                (key,),
                layout="row_major",
                distribution="replicated",
                kind="parameter",
            )
        host_keys = {
            key
            for key in checkpoint_keys
            if (
                (
                    key.startswith("model.action_processor.")
                    and not key.endswith(("w1.weight", "action_proj_back.weight"))
                )
                or ".dense." in key
                or key == "model.visual.pos_embed.weight"
            )
        }
        unmapped = set(checkpoint_keys) - set(self._source_info) - host_keys
        if unmapped:
            raise ValueError(f"wallx2b checkpoint tensors have no preparation mapping: {sorted(unmapped)}")
        self.manifest.assert_unchanged()
