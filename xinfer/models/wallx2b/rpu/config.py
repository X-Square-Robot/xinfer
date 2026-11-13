"""Admission for the single wallx2b Wall Qwen3.5 FP16 profile.

The constants are frozen from RhinoForge 2535a277's accepted checkpoint.
Configuration inspection is board free. It does not imply that the operator
backend can execute the model, or that the profile has passed task gates.
"""

from __future__ import annotations

import hashlib
import json
import stat
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23)
TEXT_LAYER_TYPES = tuple("full_attention" if i in FULL_ATTENTION_LAYERS else "linear_attention" for i in range(24))
ACTION_LAYER_TYPES = tuple("full_attention" if i in FULL_ATTENTION_LAYERS else "identity" for i in range(24))
CHECKPOINT_VOCAB_SIZE = 256_277
FILE_SHA256 = {
    "model.safetensors": "db2997acfee5ac867c6b398c52b72dd5d68bf56fb8be3db022458d932f7f60b3",
    "config.json": "76aa51dd1cc450bf753c9790ffae668e21467964b4cb3dc8b0035853b6227c62",
    "config.yml": "e9afe295772145f0132caeee80a73b24c8fb6977cdcc56a3eb13547e2c3f795b",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "normalizer_action.pth": "2bc21dec0617215b9afeefe3f7713f5e9045f33756d7b9e651a572947ebbe967",
    "normalizer_propri.pth": "1930950a3c9e2ca4356f6879d06adaeff732ff1d16d8da1c43d9233491f8254e",
}
TENSOR_MANIFEST = (
    785,
    "06106f99f876fefb38b6a3eef7161c4ed9bf8c75b9c788b9266d190833a1688e",
    192,
    "0d211e9fd1c5306f2f13ae4bd110fdb20b2d92ea59d9fa453a0f8d43c6819db8",
    593,
    "1f1232048e4772a46023e7a0302489e581ebf3913d2e5c26b9b28433bbc2bba9",
)


def _expect(name: str, value: Any, expected: Any) -> None:
    if value != expected:
        raise ValueError(f"wallx2b {name}={value!r}; required {expected!r}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino


def _local_file(root: Path, name: str) -> Path:
    path = (root / name).resolve(strict=True)
    if path.parent != root or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"wallx2b requires a local regular checkpoint file: {name}")
    return path


@dataclass(frozen=True)
class WallX2BConfig:
    """Checkpoint semantics and the deliberately narrow first serving profile.

    ``from_metadata`` checks semantic metadata only and is useful for tools and
    fixture tests. ``load`` additionally checks the two accepted config hashes.
    Full weight/asset admission is issued only by ``preflight_checkpoint``.
    """

    raw_json: Mapping[str, Any]
    raw_yaml: Mapping[str, Any]
    name: str = field(default="wallx2b-fp16-r1", init=False)
    action_horizon: int = field(default=16, init=False)
    action_dim: int = field(default=26, init=False)
    state_bins: int = field(default=512, init=False)
    inference_steps: int = field(default=10, init=False)
    num_flow_steps: int = field(default=8, init=False)
    image_targets: tuple[int, int, int] = field(default=(224, 224, 224), init=False)
    rtc_rope_gap: int = field(default=10, init=False)
    rtc_max_delay: int = field(default=8, init=False)
    action_normalizer_per_step: bool = field(default=True, init=False)
    use_state_string_representation: bool = field(default=True, init=False)
    has_propri_projection: bool = field(default=False, init=False)
    max_seq_length: int = field(default=780, init=False)
    embodiment_name: str = field(default="X2 EX001 Desktop", init=False)
    frequency_hz: int = field(default=32, init=False)
    image_processor_resize: bool = field(default=True, init=False)
    max_prefix_length: int = field(default=512, init=False)
    fp16_residual_scale: int = field(default=256, init=False)
    fp16_action_down_scale: int = field(default=4, init=False)
    patch_grids: tuple[tuple[int, int], ...] = field(default=((14, 22), (14, 20), (14, 20)), init=False)

    @property
    def profile(self) -> WallX2BConfig:
        return self

    @property
    def default_num_flow_steps(self) -> int:
        return self.num_flow_steps

    @property
    def prefix_buckets(self) -> tuple[int, ...]:
        return tuple(range(64, self.max_prefix_length + 1, 64))

    @property
    def text(self) -> Mapping[str, Any]:
        return self.raw_json["text_config"]

    @property
    def vision(self) -> Mapping[str, Any]:
        return self.raw_json["vision_config"]

    @property
    def vla(self) -> Mapping[str, Any]:
        return self.raw_json["vla_config"]

    @property
    def text_layer_types(self) -> tuple[str, ...]:
        return TEXT_LAYER_TYPES

    @property
    def action_layer_types(self) -> tuple[str, ...]:
        return ACTION_LAYER_TYPES

    @classmethod
    def load(cls, checkpoint: str | Path) -> WallX2BConfig:
        root = Path(checkpoint).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("wallx2b checkpoint must be a local directory")
        for name in ("config.json", "config.yml"):
            path = _local_file(root, name)
            before = _identity(path)
            _expect(f"{name} SHA256", file_sha256(path), FILE_SHA256[name])
            if _identity(path) != before:
                raise RuntimeError(f"checkpoint changed while reading {name}")
        import yaml

        return cls.from_metadata(
            json.loads((root / "config.json").read_text()),
            yaml.safe_load((root / "config.yml").read_text()),
        )

    @classmethod
    def from_metadata(cls, model: Mapping[str, Any], training: Mapping[str, Any]) -> WallX2BConfig:
        try:
            text = model["text_config"]
            vision = model["vision_config"]
            vla = model["vla_config"]
            _expect("architectures", tuple(model["architectures"]), ("Qwen3_5ForConditionalGeneration",))
            _expect("model_type", model["model_type"], "qwen3_5")
            for name, required in {
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 24,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "attention_bias": False,
                "attn_output_gate": True,
                "hidden_act": "silu",
                "rms_norm_eps": 1.0e-6,
                "linear_num_value_heads": 16,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
            }.items():
                _expect(f"text_config.{name}", text[name], required)
            _expect("text layer_types", tuple(text["layer_types"]), TEXT_LAYER_TYPES)
            rope = text["rope_parameters"]
            for name, required in {
                "rope_theta": 10_000_000.0,
                "rope_type": "default",
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.25,
            }.items():
                _expect(f"rope_parameters.{name}", rope[name], required)
            _expect("mrope_section", tuple(rope["mrope_section"]), (11, 11, 10))
            for name, required in {
                "depth": 24,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "out_hidden_size": 2048,
                "num_heads": 16,
            }.items():
                _expect(f"vision_config.{name}", vision[name], required)
            _expect("vla_config.dim_inputs", tuple(vla["dim_inputs"]), (2048, 1024))
            for name, required in {
                "action_hidden_size": 1024,
                "state_hidden_size": 2560,
                "norm_moe": True,
                "mlp_moe": True,
                "use_mot": True,
                "use_adarms": True,
                "adarms_cond_dim": 1024,
                "action_linear_attention": False,
                "action_mlp_intermediate_size": 2048,
                "use_state_string_representation": True,
                "causal_action_attention_mask": False,
            }.items():
                _expect(f"vla_config.{name}", vla[name], required)
            scheduler = vla["noise_scheduler"]
            for name, required in {
                "beta_alpha": 1.5,
                "beta_beta": 1.0,
                "s": 0.999,
                "num_inference_timesteps": 10,
            }.items():
                _expect(f"noise_scheduler.{name}", scheduler[name], required)
            task = training["task"]
            dataset = training["data"]["dataset_config_effective"]["dataset"]
            processors = dataset["processors"]
            epilogue = processors["epilogue"]["params"]
            _expect("task.action_horizon", task["action_horizon"], 16)
            _expect("task.action_horizon_flow", task["action_horizon_flow"], 16)
            _expect("window.action_horizon", dataset["window"]["action_horizon"], 16)
            _expect("task.action_dim", sum(task["dof_config"].values()), 26)
            _expect("state string", task["use_state_string_representation"], True)
            _expect("epilogue state string", epilogue["use_state_string_representation"], True)
            _expect("state_bins", epilogue["state_bins"], 512)
            _expect("max_seq_length", epilogue["max_seq_length"], 780)
            _expect("rope_gap", epilogue["rtc_train"]["rope_gap"], 10)
            _expect("max_delay", epilogue["rtc_train"]["max_delay"], 8)
            resolution = processors["vision"]["params"]["resolution"]
            _expect(
                "effective image targets",
                tuple(resolution[k] for k in ("face_view", "left_wrist_view", "right_wrist_view")),
                (224, 224, 224),
            )
            text_processor = processors["text"]["params"]
            embodiment_type = (dataset.get("globals") or {}).get("embodiment_type")
            embodiment_name = (text_processor.get("embodiment_type_to_name") or {}).get(
                embodiment_type, (text_processor.get("robot_name_mapping") or {}).get("x2_normal", "X2 ex001_desktop")
            )
            _expect("embodiment_name", embodiment_name, "X2 EX001 Desktop")
            _expect("frequency_hz", (text_processor.get("freq_mapping") or {}).get("x2_normal", 32), 32)
        except (KeyError, TypeError) as error:
            raise ValueError(f"incomplete wallx2b checkpoint configuration: {error}") from error
        return cls(raw_json=model, raw_yaml=training)


def is_action_tensor(name: str) -> bool:
    return name.startswith("model.action_processor.") or (
        name.startswith("model.language_model.")
        and (
            "_action" in name
            or ".action_" in name
            or ".mlp_action_expert." in name
            or name.startswith("model.language_model.norm_action.")
        )
    )


def read_safetensors_header(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read only metadata; no tensor, Torch, SDK or device is materialized."""
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors length")
        length = struct.unpack("<Q", prefix)[0]
        if length < 2 or length > min(64 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError("invalid safetensors header size")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return {k: v for k, v in header.items() if k != "__metadata__"}


_ADMISSION_TOKEN = object()


@dataclass(frozen=True)
class CheckpointManifest:
    root: Path
    config: WallX2BConfig
    file_sha256: tuple[tuple[str, str], ...]
    file_stats: tuple[tuple[str, tuple[int, int, int, int]], ...]
    tensor_manifest: tuple[Any, ...]
    _token: object = field(default=None, init=False, repr=False, compare=False)

    def assert_unchanged(self) -> None:
        if self._token is not _ADMISSION_TOKEN:
            raise ValueError("manifest must be issued by preflight_checkpoint in this process")
        for name, identity in self.file_stats:
            if _identity(_local_file(self.root, name)) != identity:
                raise RuntimeError(f"checkpoint changed after admission: {name}")

    def as_dict(self) -> dict[str, Any]:
        return dict(
            root=str(self.root),
            profile=self.config.name,
            file_sha256=dict(self.file_sha256),
            tensor_manifest=list(self.tensor_manifest),
        )


def preflight_checkpoint(checkpoint: str | Path) -> CheckpointManifest:
    """Hash all accepted assets and close the key/dtype/shape manifest first."""
    root = Path(checkpoint).expanduser().resolve(strict=True)
    config = WallX2BConfig.load(root)
    identities = []
    for name, expected in FILE_SHA256.items():
        path = _local_file(root, name)
        before = _identity(path)
        _expect(f"{name} SHA256", file_sha256(path), expected)
        if _identity(path) != before:
            raise RuntimeError(f"checkpoint changed while hashing {name}")
        identities.append((name, before))
    header = read_safetensors_header(root / "model.safetensors")
    digests = [hashlib.sha256() for _ in range(3)]
    counts = [0, 0, 0]
    for name in sorted(header):
        value = header[name]
        row = f"{name}|{value['dtype']}|{','.join(str(d) for d in value['shape'])}\n".encode()
        branch = 1 if is_action_tensor(name) else 2
        for index in (0, branch):
            digests[index].update(row)
            counts[index] += 1
    manifest = tuple(v for n, d in zip(counts, digests) for v in (n, d.hexdigest()))
    _expect("safetensors key/dtype/shape manifest", manifest, TENSOR_MANIFEST)
    result = CheckpointManifest(root, config, tuple(FILE_SHA256.items()), tuple(identities), manifest)
    object.__setattr__(result, "_token", _ADMISSION_TOKEN)
    result.assert_unchanged()
    return result
