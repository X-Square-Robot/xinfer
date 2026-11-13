"""Board-free semantic admission; fixture metadata is not a weight capability."""

import copy
import json
import struct

import pytest

from xinfer.models.wallx2b.rpu.config import (
    ACTION_LAYER_TYPES,
    FULL_ATTENTION_LAYERS,
    TEXT_LAYER_TYPES,
    CheckpointManifest,
    WallX2BConfig,
    read_safetensors_header,
)


def _metadata():
    model = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "attention_bias": False,
            "attn_output_gate": True,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "linear_num_value_heads": 16,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "layer_types": list(TEXT_LAYER_TYPES),
            "rope_parameters": {
                "rope_theta": 10_000_000,
                "rope_type": "default",
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.25,
            },
        },
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "out_hidden_size": 2048,
            "num_heads": 16,
        },
        "vla_config": {
            "dim_inputs": [2048, 1024],
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
            "noise_scheduler": {"beta_alpha": 1.5, "beta_beta": 1.0, "s": 0.999, "num_inference_timesteps": 10},
        },
    }
    training = {
        "task": {
            "action_horizon": 16,
            "action_horizon_flow": 16,
            "dof_config": {"action": 26},
            "use_state_string_representation": True,
        },
        "data": {
            "dataset_config_effective": {
                "dataset": {
                    "window": {"action_horizon": 16},
                    "globals": {"embodiment_type": "EX001_DESKTOP"},
                    "processors": {
                        "epilogue": {
                            "params": {
                                "use_state_string_representation": True,
                                "state_bins": 512,
                                "max_seq_length": 780,
                                "rtc_train": {"rope_gap": 10, "max_delay": 8},
                            }
                        },
                        "vision": {
                            "params": {
                                "resolution": {"face_view": 224, "left_wrist_view": 224, "right_wrist_view": 224}
                            }
                        },
                        "text": {"params": {"embodiment_type_to_name": {"EX001_DESKTOP": "X2 EX001 Desktop"}}},
                    },
                }
            }
        },
    }
    return model, training


def test_saved_scheduler_does_not_override_serving_nfe():
    config = WallX2BConfig.from_metadata(*_metadata())
    assert config.inference_steps == 10
    assert config.num_flow_steps == config.default_num_flow_steps == 8
    assert config.fp16_residual_scale == 256
    assert config.fp16_action_down_scale == 4
    assert config.rtc_rope_gap == 10
    assert config.prefix_buckets == tuple(range(64, 513, 64))
    assert sum(h * w for h, w in config.patch_grids) == 868


def test_action_has_identity_mixer_and_mlp_at_eighteen_layers():
    config = WallX2BConfig.from_metadata(*_metadata())
    assert config.action_layer_types == ACTION_LAYER_TYPES
    assert (
        tuple(i for i, kind in enumerate(config.action_layer_types) if kind == "full_attention")
        == FULL_ATTENTION_LAYERS
    )
    assert config.action_layer_types.count("identity") == 18
    assert config.text_layer_types.count("linear_attention") == 18
    assert config.vla["action_mlp_intermediate_size"] == 2048


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("text_config", "num_hidden_layers", 28),
        ("text_config", "linear_num_value_heads", 32),
        ("text_config", "attn_output_gate", False),
        ("vision_config", "depth", 32),
        ("vla_config", "action_linear_attention", True),
        ("vla_config", "causal_action_attention_mask", True),
    ],
)
def test_nearby_model_profiles_are_rejected(section, key, value):
    model, training = _metadata()
    model[section][key] = value
    with pytest.raises(ValueError, match=key):
        WallX2BConfig.from_metadata(model, training)


def test_effective_dataset_settings_take_precedence():
    model, training = _metadata()
    training["data"]["resolution"] = {"face_view": 1000}
    WallX2BConfig.from_metadata(model, training)
    bad = copy.deepcopy(training)
    bad["data"]["dataset_config_effective"]["dataset"]["processors"]["epilogue"]["params"]["rtc_train"]["rope_gap"] = 0
    with pytest.raises(ValueError, match="rope_gap"):
        WallX2BConfig.from_metadata(model, bad)


def test_fixture_metadata_cannot_forge_weight_admission(tmp_path):
    config = WallX2BConfig.from_metadata(*_metadata())
    manifest = CheckpointManifest(tmp_path, config, (), (), ())
    with pytest.raises(ValueError, match="issued by preflight"):
        manifest.assert_unchanged()


def test_local_header_reader_ignores_non_tensor_metadata(tmp_path):
    header = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "__metadata__": {"format": "pt"}}
    payload = json.dumps(header).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"\0" * 4)
    assert read_safetensors_header(path) == {"weight": header["weight"]}
    path.write_bytes(struct.pack("<Q", 1000000))
    with pytest.raises(ValueError, match="header size"):
        read_safetensors_header(path)
