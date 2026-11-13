"""Wall-OSS processing parity with the reference service capture.

Needs external assets, so it is skipped unless both variables are set:

- ``XINFER_WALL_OSS_CHECKPOINT``: checkpoint directory (config.json, config.yml,
  tokenizer.json, normalizer_action.pth, normalizer_propri.pth).
- ``XINFER_WALL_OSS_REFERENCE``: directory with ``requests.pkl`` (raw requests by
  name) and ``reference.npz`` (the old service's ``s1`` model inputs, ``pred``
  and ``result`` per request), produced from a capture of the production
  service; see docs/models/wall_oss.md.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

CKPT = os.environ.get("XINFER_WALL_OSS_CHECKPOINT")
REF = os.environ.get("XINFER_WALL_OSS_REFERENCE")
pytestmark = [
    pytest.mark.assets,
    pytest.mark.skipif(not (CKPT and REF), reason="Wall-OSS assets not configured"),
]

# The reference deployment rewrote every instruction to this task string.
REFERENCE_INSTRUCTION = (
    "Pick a package, flip it to show the barcode, then pass it. "
    "[action mode: latest; collector: 曾祥煜; quality: excellent; robot: 11118]"
)
INPUTS = ("input_ids", "position_ids", "pixel_values", "image_grid_thw", "proprioception",
          "agent_pos_mask", "dof_mask", "padding_action", "rtc_actions")
# XInfer name -> the service capture's name, where they differ
SERVICE_NAMES = {"rtc_actions": "prefix_actions"}
OUTPUTS = ("follow1_pos", "follow2_pos", "predict_relative_chunk", "action_pred")


@pytest.fixture(scope="module")
def corpus():
    from xinfer.models.wall_oss.config import WallOSSConfig
    from xinfer.models.wall_oss.processing import (ProcessingConfig, WallOSSInputProcessor,
                                                  WallOSSOutputProcessor)

    model_config = WallOSSConfig.load(CKPT)
    config = ProcessingConfig(instruction_override=REFERENCE_INSTRUCTION, prompt_bucket=0)  # the service's prompts
    requests = pickle.load(open(Path(REF) / "requests.pkl", "rb"))
    reference = np.load(Path(REF) / "reference.npz")
    return WallOSSInputProcessor(CKPT, model_config, config), WallOSSOutputProcessor(model_config), requests, reference


def _bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def test_inputs_and_response_match_reference_bitwise(corpus):
    """Model inputs, with the views packed as the vision stage packs them, and responses equal the service's."""
    from xinfer.models.layers.image_pack import ImagePack
    from xinfer.models.wall_oss.image_processing import pixel_lookup_table

    prepare, finish, requests, ref = corpus
    c = prepare.config
    pack = ImagePack(pixel_lookup_table(c.image_mean, c.image_std), patch_size=c.patch_size,
                     temporal_patch_size=c.temporal_patch_size, merge_size=c.merge_size)
    mismatches = []
    for name, observation in requests.items():
        prepared = prepare(observation)
        prepared.tensors["pixel_values"] = pack([prepared.tensors[f"image_{i}"] for i in range(len(c.cameras))])
        for key in INPUTS:
            ref_key = f"{name}/s1/{SERVICE_NAMES.get(key, key)}"
            if ref_key not in ref.files:
                assert key not in prepared.tensors, f"{name}: unexpected {key}"
                continue
            if not _bitwise(prepared.tensors[key].numpy(), ref[ref_key]):
                mismatches.append(f"{name}: {key}")
        response = finish({"predict_action": ref[f"{name}/pred"]}, prepared)
        for key in OUTPUTS:
            expected = ref[f"{name}/result/0:{key}"]
            if not _bitwise(np.asarray(response[key], dtype=expected.dtype), expected):
                mismatches.append(f"{name}: response {key}")
    assert not mismatches, mismatches


def test_bucketed_prompts_are_the_exact_prompts_padded(corpus):
    from xinfer.models.wall_oss.config import WallOSSConfig
    from xinfer.models.wall_oss.processing import ProcessingConfig, WallOSSInputProcessor, pad_prompt

    requests, model_config = corpus[2], WallOSSConfig.load(CKPT)
    exact, bucketed = (WallOSSInputProcessor(CKPT, model_config, ProcessingConfig(
        instruction_override=REFERENCE_INSTRUCTION, prompt_bucket=bucket, noise_seed=0)) for bucket in (0, 16))
    for name, observation in requests.items():
        unpadded, padded = exact(observation).tensors, bucketed(observation).tensors
        assert padded["prompt_len"] % 16 == 0 and padded["prompt_len"] - unpadded["prompt_len"] < 16, name
        expected = pad_prompt(unpadded, padded["prompt_len"])
        assert set(padded) == set(expected), name
        for key, value in expected.items():
            same = torch.equal(padded[key], value) if torch.is_tensor(value) else padded[key] == value
            assert same, (name, key)


def test_prompt_length_range_spans_every_state(corpus):
    from xinfer.models.wall_oss.processing import load_tokenizer

    tokenizer = load_tokenizer(Path(CKPT))
    for values in (1, 7, 20):  # one token per digit: 1..3 digit values change the prompt by 2 tokens each
        short, long = (len(tokenizer.encode(f"Proprioception: {' '.join([v] * values)}\n",
                                            add_special_tokens=False).ids) for v in ("0", "511"))
        assert long - short == 2 * values
    prepare, requests = corpus[0], corpus[2]
    for name, observation in requests.items():
        prepared = prepare(observation)
        shortest, longest = prepared.meta["prompt_len_range"]
        values = int((prepared.tensors["agent_pos_mask"] != 0).sum())
        assert shortest <= prepared.tensors["prompt_len"] <= longest and longest - shortest == 2 * values, name
