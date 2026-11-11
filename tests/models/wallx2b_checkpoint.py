"""A synthetic Wall-X 2B checkpoint directory for the CPU tests: the processing configuration of the desktop
deployment and a tiny model structure (no weights; tests that need them write their own)."""

import json

import torch
import yaml

H, D = 16, 26
DOF = {"master_left_ee_cartesian_pos_relative": 3, "master_left_ee_rotation_6D_relative": 6, "master_left_gripper": 1,
       "master_right_ee_cartesian_pos_relative": 3, "master_right_ee_rotation_6D_relative": 6,
       "master_right_gripper": 1, "velocity_decomposed": 3, "height": 1, "head_actions": 2}
AGENT = {"follow_left_ee_cartesian_pos": 3, "follow_left_ee_rotation_6D": 6, "follow_left_gripper": 1,
         "follow_right_ee_cartesian_pos": 3, "follow_right_ee_rotation_6D": 6, "follow_right_gripper": 1,
         "velocity_decomposed": 3, "height": 1, "head_actions": 2}
TEXT = {
    "hidden_size": 64, "intermediate_size": 96, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"], "rms_norm_eps": 1e-6, "linear_num_key_heads": 2,
    "linear_num_value_heads": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
    "attn_output_gate": True, "hidden_act": "silu", "attention_bias": False,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                        "mrope_section": [1, 1, 0], "mrope_interleaved": True},
}
VISION = {"model_type": "qwen3_5", "depth": 2, "hidden_size": 32, "intermediate_size": 48, "num_heads": 2,
          "in_channels": 3, "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 64,
          "num_position_embeddings": 16, "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": []}
VLA = {"dim_inputs": [64, 32], "action_hidden_size": 32, "adarms_cond_dim": 32, "action_mlp_intermediate_size": 48,
       "use_mot": True, "norm_moe": True, "mlp_moe": True, "use_adarms": True, "action_linear_attention": False,
       "causal_action_attention_mask": False, "use_state_string_representation": True,
       "noise_scheduler": {"s": 0.999, "num_inference_timesteps": 10}}


def write_checkpoint(root):
    """Write config.json, config.yml, preprocessor_config.json and the two normalizers into ``root``."""
    (root / "config.json").write_text(json.dumps({
        "image_token_id": 248056, "vision_start_token_id": 248053, "vision_end_token_id": 248054,
        "text_config": TEXT, "vision_config": VISION, "vla_config": VLA}))
    (root / "preprocessor_config.json").write_text(json.dumps({
        "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16, "temporal_patch_size": 2,
        "merge_size": 2, "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
        "image_processor_type": "Qwen2VLImageProcessorFast"}))
    processors = {
        "vision": {"type": "qwen3_5", "params": {"resolution": {"face_view": 224}}},
        "text": {"type": "wallx_task_instruction", "params": {
            "use_embodied_prompt_ratio": 0.8, "embodiment_type_to_name": {"TEST_DESKTOP": "Test desktop robot"},
            "action_chunk_size": H, "use_relative_action": True}},
        "epilogue": {"params": {"state_bins": 512, "max_seq_length": 780, "use_state_string_representation": True,
                                "rtc_train": {"max_delay": 8, "rope_gap": 10}}},
    }
    (root / "config.yml").write_text(yaml.safe_dump({
        "model_type": "qwen3_5", "task": {"dof_config": DOF, "agent_pos_config": AGENT, "action_horizon_flow": H},
        "model": {"new_special_tokens": ["<point>"]},
        "data": {"resolution": {"face_view": 256}, "normalizer_config": {"min_key": "q001"},
                 "dataset_config_effective": {"dataset": {"globals": {"embodiment_type": "TEST_DESKTOP"},
                                                          "processors": processors}}}}, sort_keys=False))
    generator = torch.Generator().manual_seed(0)
    torch.save({"min.x2_normal": torch.randn(H, D, generator=generator) * 0.1,
                "delta.x2_normal": torch.rand(H, D, generator=generator) + 0.5}, root / "normalizer_action.pth")
    torch.save({"min.x2_normal": torch.zeros(D), "delta.x2_normal": torch.ones(D)}, root / "normalizer_propri.pth")
