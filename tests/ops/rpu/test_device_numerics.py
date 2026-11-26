# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Opt-in board tests of the built REF assets against independent Torch oracles.

The source RPU integration tests established exact GELU saturation/zero anchors
only. The dense and random probes below are additional gates over the selected
device assets.
"""

import os
from contextlib import ExitStack

import pytest

pytestmark = [
    pytest.mark.rpu,
    pytest.mark.skipif(os.environ.get("XINFER_RPU_RUN_BOARD") != "1",
                       reason="set XINFER_RPU_RUN_BOARD=1 only with an exclusively allocated Rhino RPU board"),
]


@pytest.mark.parametrize("function,approximate", [("gelu_tanh", "tanh"), ("gelu_erf", "none")])
@pytest.mark.parametrize("input_kind", ["exact_anchors", "dense", "random"])
def test_gelu_prepared_replay_matches_torch(function, approximate, input_kind, record_property):
    import numpy as np
    import torch
    import torch.nn.functional as functional

    from xinfer.ops.backends import rpu

    provider = rpu.load_backend()
    # Opting in is an instruction to execute: a missing runtime/SDK/asset is a
    # failure, never a successful skip or a substituted host implementation.
    assert provider.provenance["runtime_available"], "board tests require a runtime xinfer_rpu.so"
    record_property("rpu_library_sha256", provider.provenance["library_sha256"])
    record_property("rpu_catalog_sha256", provider.provenance["catalog_sha256"])
    shape = (17, 256)
    elements = 8 * shape[0] * shape[1]
    if input_kind == "exact_anchors":
        # Retain the original RPU test's zero-tolerance saturation domain.
        host = np.resize(np.array([-32, -16, 0, 16, 32], dtype=np.float16), (8, *shape))
        tolerance = {"atol": 0, "rtol": 0}
    else:
        if input_kind == "dense":
            values = torch.linspace(-8, 8, elements, dtype=torch.float32)
        else:
            generator = torch.Generator().manual_seed(1729)
            values = torch.randn(elements, generator=generator, dtype=torch.float32) * 2
        host = values.half().reshape(8, *shape).numpy()
        # Pin saturation, signed-zero and subnormal inputs alongside ordinary
        # nonlinear values, on every physical core.
        edges = np.array([-65504, -32, -16, -10, -4, -1, -0.0, 0, 2 ** -24,
                          2 ** -14, 1, 4, 10, 16, 32, 65504], dtype=np.float16)
        host[:, 0, :len(edges)] = edges
        # Native FP16 arithmetic (without FP32 intermediate promotion) rounds
        # each polynomial/sigmoid/exp stage in binary16. Observed max absolute
        # deviation vs FP32-cast Torch on R1 hardware is 1.95e-3 (gelu_tanh)
        # and 7.32e-4 (gelu_erf), so use a tight 2e-3 absolute bound with
        # standard 1e-3 relative tolerance while exact_anchors stays at 0.
        tolerance = {"atol": 2e-3, "rtol": 1e-3}
    with ExitStack() as stack:
        context = stack.enter_context(provider.backend_factory(allow_candidate=True))
        source = stack.enter_context(context.empty(shape))
        output = stack.enter_context(context.empty(shape))
        operation = stack.enter_context(context.prepare(
            "activation", tensors={"x": source, "out": output}, function=function))
        execution = stack.enter_context(context.batch([operation]))
        assert operation.kernel == "gelu_f16"
        for replay in range(3):
            changed = np.ascontiguousarray(np.roll(host, replay * 13, axis=-1))
            source.copy_from(changed)
            output.copy_from(np.full((8, *shape), np.nan, dtype=np.float16))
            execution.execute(async_=True).wait()
            actual = torch.from_numpy(output.numpy())
            expected = functional.gelu(torch.from_numpy(changed).float(), approximate=approximate).half()
            assert torch.isfinite(actual).all(), f"{function} {input_kind}: output was not completely written"
            torch.testing.assert_close(actual, expected, **tolerance,
                                       msg=lambda message: f"{function} {input_kind} replay {replay}: {message}")
