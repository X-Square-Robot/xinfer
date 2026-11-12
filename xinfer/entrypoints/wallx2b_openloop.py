"""Inspect CES assets and replay explicit, frozen wallx2b request inputs.

The NPZ request format holds RGB arrays under face_view/left_wrist_view/
right_wrist_view, state[26], initial_noise[1,16,26] FP32, and instruction.
It contains no pickled objects. Raw video decoding and GT evaluation remain
outside the public policy-call timing boundary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("wallx2b",), default="wallx2b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--request", type=Path, help="frozen request NPZ including explicit initial_noise")
    parser.add_argument(
        "--check",
        "--dry-run",
        action="store_true",
        dest="check",
        help="inspect config and optionally preprocess a request without importing xcompute",
    )
    parser.add_argument("--validate-assets", action="store_true", help="also hash all checkpoint assets")
    parser.add_argument("--output-dir", type=Path, help="external directory for manifests and prepared input arrays")
    parser.add_argument("--device", choices=("rpu", "rpu:0"), default="rpu:0")
    parser.add_argument("--prec", choices=("fp16",), default="fp16")
    parser.add_argument("--nfe", type=int, choices=(8,), default=8)
    parser.add_argument("--allow-candidate", action="store_true", help="admit the experimental REF-only RPU build")
    return parser.parse_args(argv)


def _request(path):
    import numpy as np

    from xinfer.models.wallx2b.rpu.processing import DEFAULT_CAMERAS

    with np.load(path, allow_pickle=False) as source:
        expected = {*DEFAULT_CAMERAS, "state", "instruction", "initial_noise"}
        if set(source.files) != expected:
            raise ValueError(f"request NPZ must contain exactly {sorted(expected)}")
        noise = source["initial_noise"].copy()
        if noise.dtype != np.float32 or noise.shape != (1, 16, 26) or not np.isfinite(noise).all():
            raise ValueError("initial_noise must be finite FP32 [1,16,26]; a seed cannot substitute for bytes")
        instruction = source["instruction"]
        if instruction.shape != () or instruction.dtype.kind != "U":
            raise ValueError("instruction must be a Unicode scalar")
        return dict(
            images={name: source[name].copy() for name in DEFAULT_CAMERAS},
            state=source["state"].copy(),
            instruction=instruction.item(),
            initial_noise=noise,
        )


def main(argv=None):
    args = parse_args(argv)
    from xinfer.models.registry import available_models
    from xinfer.models.wallx2b.rpu.config import WallX2BConfig, preflight_checkpoint

    config = WallX2BConfig.load(args.checkpoint)
    report = dict(
        model="wallx2b",
        profile=config.name,
        registered_models=available_models(),
        precision="fp16",
        tp=8,
        batch=1,
        horizon=16,
        action_dim=26,
        nfe=8,
        saved_scheduler_steps=10,
        rtc=False,
        whole_model_execution="not_executed",
        acceptance_5_percent="not_measured",
    )
    if args.validate_assets:
        report["checkpoint"] = preflight_checkpoint(args.checkpoint).as_dict()
    if args.request is None and not args.check:
        raise ValueError("execution requires --request with frozen initial_noise bytes")
    prepared = None
    if args.request:
        import hashlib

        request_path = args.request.resolve(strict=True)
        digest = hashlib.sha256()
        with request_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        report["request"] = dict(path=str(request_path), sha256=digest.hexdigest())
        observation = _request(args.request)
        if args.check:
            from xinfer.models.wallx2b.rpu.model import WallX2B

            prepare, _, _ = WallX2B.build_processors(args.checkpoint, audit_inputs=True)
            try:
                prepared = prepare(observation)
            finally:
                prepare.close()
            report["inputs"] = prepared.meta["inputs"]
            report["noise"] = prepared.meta["noise"]
            report["resized_images"] = prepared.meta["resized_images"]
            report["camera_order"] = prepared.meta["camera_order"]
            report["gt_prediction_frame_offset"] = prepared.meta["gt_prediction_frame_offset"]
            report["prefix_length"] = prepared.tensors["prefix_length"]
            report["prefix_bucket"] = prepared.tensors["prefix_bucket"]
        else:
            from xinfer.entrypoints.policy import Policy

            startup = time.perf_counter_ns()
            with Policy("wallx2b", args.checkpoint, device=args.device, quantization="fp16",
                        allow_candidate=args.allow_candidate) as policy:
                report["startup_ms"] = (time.perf_counter_ns() - startup) / 1e6
                start = time.perf_counter_ns()
                result = policy.infer(observation)
                report["policy_call_ms"] = (time.perf_counter_ns() - start) / 1e6
                # Materialization above is included. Evaluation and file IO below are not.
                report["whole_model_execution"] = "executed_without_acceptance_claim"
                report["execution"] = policy.model.describe_rpu()
                if args.output_dir:
                    import numpy as np

                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    np.save(args.output_dir / "predict_action.npy", result["predict_action"])
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if prepared is not None:
            import numpy as np
            import torch

            arrays = {key: value.numpy() for key, value in prepared.tensors.items() if isinstance(value, torch.Tensor)}
            np.savez(args.output_dir / "prepared_inputs.npz", **arrays)
        (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
