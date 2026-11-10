"""XInfer Wall-OSS adapter for embodied-serving ``scripts/check_precision.py`` (quick100, scope model_batch).

See docs/precision.md. The harness, the frozen quick100 assets and the
original BF16 reference belong to embodied-serving; this module only runs
XInfer on its requests.

Runs XInfer Wall-OSS on the frozen quick100 requests: XInfer's own input
processing on the wire observation, the request's saved FP32 noise, vision
and prefill per request, then ONE action-stage call over the whole request
group (B1, or the scenario's B2/B4/B8; different prompt lengths are merged
left-padded, as the engine does), and XInfer's output processing. The
recorded call trace is that action-stage call with its actual batch and the
delays the model used. The optimized composition takes one request per action call, so it covers the B1
scenario only; a batched request group raises.

Config JSON keys: ``xinfer_src``, ``checkpoint``, ``numerics`` ("optimized" | "reference"), optional
``xcompute_library``, ``op_backends`` ({op: backend}), ``processing`` (ProcessingConfig fields),
``image_preprocess_device`` ("cpu", the default, or "gpu"; the top-level model option of docs/configuration.md),
``norm_key``.

Use: PRECISION_ADAPTER=wall_oss_quick100_adapter:Adapter with this directory on PYTHONPATH.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Inputs the Wall-OSS action stage reads (model.denoise).
ACTION_INPUTS = ("prompt_kv", "prompt_valid", "noise", "dof_mask", "dof_mask_all", "padding_action",
                 "inference_delay", "action_position_base", "rtc_actions", "num_steps")


class Adapter:
    scope = "model_batch"

    def __init__(self, *, config, checkpoint_hashes, role, device):
        if role != "candidate":
            raise ValueError("the XInfer adapter evaluates candidates; the BF16 reference stays the original model")
        self.config = config
        ckpt = Path(config["checkpoint"])
        for name, expected in checkpoint_hashes.items():
            if _sha256(ckpt / name) != expected:
                raise ValueError(f"checkpoint hash mismatch: {name}")
        os.environ.update(CUDA_VISIBLE_DEVICES=str(device), NUMBA_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        if config.get("xcompute_library"):
            os.environ["XINFER_XCOMPUTE_LIBRARY"] = config["xcompute_library"]
        self.xinfer_src = Path(config["xinfer_src"]).resolve()
        sys.path.insert(0, str(self.xinfer_src))
        import torch

        from xinfer import ops
        from xinfer.config import apply_image_preprocess_device
        from xinfer.engine.device_images import DeviceImageDecoder
        from xinfer.models.wall_oss.model import WallOSS
        from xinfer.models.wall_oss.processing import ProcessingConfig
        from xinfer.platforms import resolve_platform

        import xinfer
        if not Path(xinfer.__file__).resolve().is_relative_to(self.xinfer_src):
            raise ValueError("imported xinfer does not come from the configured source")
        self.torch = torch
        torch.set_grad_enabled(False)
        self.platform = resolve_platform("cuda")
        ops.registry.load_plugins()
        self.table = ops.registry.build_table(self.platform, backends=config.get("op_backends") or None)
        options = {"processing": config.get("processing", {}),
                   "image_preprocess_device": config.get("image_preprocess_device")}
        options = apply_image_preprocess_device(options, self.platform, default="cpu")
        processing = ProcessingConfig(**options["processing"])
        with ops.use_ops(self.table):
            self.model = WallOSS.from_checkpoint(ckpt, device=self.platform.device,
                                                 numerics=config.get("numerics", "optimized"), processing=processing)
        # image_preprocess_device="gpu": the views arrive as DeviceImage values, decoded here as the engine does
        self.decode = DeviceImageDecoder(self.platform.device) if processing.image_preprocess_device == "gpu" else None
        self.prepare = self.model.input_processor()
        self.finish = self.model.output_processor()
        stats = torch.load(ckpt / "normalizer_action.pth", map_location="cpu", weights_only=True)
        key = config.get("norm_key", "ex_normal")
        self.minimum = stats[f"min.{key}"].float().numpy().astype(np.float64)[..., :20]
        self.delta = stats[f"delta.{key}"].float().numpy().astype(np.float64)[..., :20]
        self.source_hashes = self._sources()

    def _sources(self):
        return {str(p.relative_to(self.xinfer_src)): _sha256(p)
                for p in sorted((self.xinfer_src / "xinfer").rglob("*.py"))}

    def identity(self):
        if self._sources() != self.source_hashes:
            raise ValueError("XInfer source changed during the precision run")
        library = os.environ.get("XINFER_XCOMPUTE_LIBRARY")
        return {"engine": "xinfer", "xinfer_src": str(self.xinfer_src), "source_sha256": self.source_hashes,
                "config": self.config, "ops": self.table.report(), "numerics": self.model.numerics,
                "xcompute_sha256": _sha256(library) if library else None, "adapter_sha256": _sha256(__file__),
                "torch": self.torch.__version__, "cuda": self.torch.version.cuda,
                "gpu": self.torch.cuda.get_device_name()}

    def begin_scenario(self, name, batch):
        self.torch.cuda.synchronize()

    def infer(self, requests):
        from precision_wallx import normalize_wire  # the harness's own wire normalization

        from xinfer.engine import batching
        from xinfer.engine.executor import run_stage

        torch, dev, model = self.torch, self.platform.device, self.model
        if len(requests) not in model.capabilities().batch_sizes["action"]:
            raise ValueError(f"the {model.numerics} composition takes no action batch of {len(requests)}")
        prepared, views = [], []
        for r in requests:
            p = self.prepare(r["observation"])
            noise = np.asarray(r["noise"], dtype=np.float32)
            p.tensors["noise"] = torch.from_numpy(noise.reshape(1, *noise.shape).copy())
            x = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in p.tensors.items()}
            if self.decode is not None:
                x = self.decode(x)
            x["image_embeds"] = run_stage(model, "vision", x, dev)["image_embeds"]
            x["prompt_kv"] = run_stage(model, "prefill", x, dev)["prompt_kv"]
            prepared.append(p)
            views.append(x)
        views = [{k: v for k, v in x.items() if k in ACTION_INPUTS} for x in views]
        axes = {n: model.batch_axes.get(n, 0) for n, v in views[0].items() if torch.is_tensor(v)}
        merged = batching.merge(views, axes, batching.ragged_inputs(model, "action"))  # left-pads prompt lengths
        actions = run_stage(model, "action", merged, dev, batch_size=len(requests))["predict_action"]
        delays = [int(v["inference_delay"].reshape(-1)[0]) if "rtc_actions" in v else 0 for v in views]
        outputs = []
        for i, (r, p) in enumerate(zip(requests, prepared)):
            value = self.finish({"predict_action": actions[i:i + 1].cpu()}, p)
            outputs.append(normalize_wire(value, r["state"], self.minimum, self.delta))
        torch.cuda.synchronize()
        calls = [{"ids": [r["id"] for r in requests], "batch": len(requests), "delays": delays,
                  "context_delays": delays}]
        return np.stack(outputs), calls
