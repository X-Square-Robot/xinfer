"""A one-stage CPU model for end-to-end serving tests (imported by path in subprocesses)."""

import itertools
import os
import threading
import time

import numpy as np
import torch

from xinfer.models.interface import IOSchema, OutputSpec, PreparedRequest, Stage, StageGraph, VLAModel


# Prepares of requests marked "rendezvous" wait here for each other: two pass only when they run at the same time.
RENDEZVOUS = threading.Barrier(2, timeout=10)


class Refused(ValueError):
    """A protocol error with the information the robot acts on."""

    error_info = {"code": "echo_refused", "retryable": False}


class Prepare:
    """The input processor; ``connect`` gives a connection its own request count, like per-connection plans."""

    def __init__(self):
        self.calls = itertools.count(1)  # per-processor state, like a seeded noise generator

    def __call__(self, observation, connection_calls=None):
        if "x" not in observation:
            raise ValueError("missing x")
        if observation.get("refuse"):
            raise Refused("refused")
        if observation.get("rendezvous"):
            RENDEZVOUS.wait()
        tensors = {"x": torch.tensor(np.asarray(observation["x"], dtype=np.float32))[None]}
        for key in ("fail", "exit", "hang", "spin", "slow"):  # stage behaviors, see EchoModel.act
            if observation.get(key):
                tensors[key] = True
        meta = {"call": next(self.calls)}
        if connection_calls is not None:
            meta["connection_call"] = next(connection_calls)
        return PreparedRequest(tensors, meta=meta)

    def connect(self):
        calls = itertools.count(1)
        return lambda observation: self(observation, calls)


class EchoModel(VLAModel):
    stage_graph = StageGraph(Stage("act", "act"))

    @classmethod
    def from_checkpoint(cls, path, *, device, ops=None, scale=2.0, **_):
        model = cls()
        model.scale = scale
        return model

    @classmethod
    def build_processors(cls, checkpoint, scale=2.0, record=None, **_):
        def finish(outputs, prepared):
            y = np.asarray(outputs["y"])[0]
            if record:  # file that every finished request is appended to, with the process and its prepare call
                with open(record, "a") as f:
                    f.write(f"{os.getpid()} {prepared.meta['call']} {' '.join(map(str, y.tolist()))}\n")
            return {"y": y, **{k: prepared.meta[k] for k in ("call", "connection_call") if k in prepared.meta}}

        return Prepare(), finish, {"model": "echo", "scale": scale}

    def io_schema(self):
        return IOSchema(cameras=(), state_keys=(), action_keys=("y",), action_dim=3, action_horizon=1,
                        instruction=False)

    def input_processor(self):
        return self.build_processors("unused", scale=self.scale)[0]

    def output_processor(self):
        return self.build_processors("unused", scale=self.scale)[1]

    def stage_outputs(self, stage, batch):
        return {"y": OutputSpec(tuple(batch.inputs["x"].shape), torch.float32)}

    def act(self, batch):
        if "fail" in batch.inputs:
            raise RuntimeError("stage failed")
        if "exit" in batch.inputs:
            raise SystemExit  # not caught by the engine loop
        if "slow" in batch.inputs:
            time.sleep(1)  # a request in flight for a second
        if "hang" in batch.inputs:
            threading.Event().wait()  # a dispatch that never returns
        if "spin" in batch.inputs:  # a hung device; kept to seconds (2**34 cycles), as the kernel outlives the server
            torch.cuda._sleep(1 << 34)
        torch.add(batch.inputs["x"] * self.scale, 1.0, out=batch.out["y"])
