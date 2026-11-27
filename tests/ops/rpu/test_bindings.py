# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Python ownership and argument contracts with an explicit fake native module.

Host binding contract tests; this does not execute RPU kernels.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from xinfer.ops.backends.rpu import _native, rpu, rpu_tp


@pytest.fixture
def binding(monkeypatch):
    calls = []
    functions = {
        "rpu_tp_context_create": lambda *args: calls.append(("context", args)) or 1,
        "rpu_tp_context_create_with_pool": lambda *args: calls.append(("pool", args)) or 2,
        "rpu_context_create": lambda *args: calls.append(("single", args)) or 3,
        "rpu_release": lambda handle: calls.append(("release", handle)),
        "rpu_tp_selection": lambda _: ["fixture_kernel", "", "", ""],
        "rpu_tp_wait": lambda handle: calls.append(("wait", handle)),
        "rpu_tp_execute": lambda *args: calls.append(("execute", args)),
    }
    monkeypatch.setattr(_native.lib, "_functions", functions)
    monkeypatch.setattr(rpu, "require_runtime_assets", lambda: {"manifest_sha256": "a" * 64})
    monkeypatch.setattr(rpu_tp, "require_runtime_assets", lambda: {"manifest_sha256": "a" * 64})
    return SimpleNamespace(calls=calls, functions=functions)


def tensor(handle, context, *, shape=(16,), layout="row_major", storage="spm", distribution="local"):
    return rpu_tp.RpuTpTensor(handle, context, shape, "float16", storage, layout, distribution)


def test_retained_pool_and_candidate_options_preserve_default_ffi(binding):
    rpu_tp.RpuTpContext(allow_candidate=True).close()
    rpu_tp.RpuTpContext(allow_candidate=True, retained_segments=6144).close()
    assert binding.calls == [("context", (True,)), ("release", 1), ("pool", (True, 6144)), ("release", 2)]


@pytest.mark.parametrize("value,exception", [(0, ValueError), (129, ValueError), (8193, ValueError), (True, TypeError)])
def test_invalid_pool_never_allocates(binding, value, exception):
    with pytest.raises(exception, match="retained_segments"):
        rpu_tp.RpuTpContext(retained_segments=value)
    assert binding.calls == []


def test_rebind_forwards_explicit_patch_and_rejects_foreign_context(binding):
    binding.functions["rpu_tp_rebind"] = lambda *args: binding.calls.append(("rebind", args))
    with rpu_tp.RpuTpContext() as context, rpu_tp.RpuTpContext() as other:
        with tensor(7, context) as source, tensor(8, other) as foreign:
            with rpu_tp.RpuTpOperation(9, context, "copy_f16") as operation:
                assert operation.rebind(tensors={"input": source}, integers={"valid_rows": 15}) is operation
                assert binding.calls[-1] == ("rebind", (9, ["input"], [7], ["valid_rows"], [15], [], []))
                before = list(binding.calls)
                with pytest.raises(ValueError, match="TP context"):
                    operation.rebind(tensors={"input": foreign})
                assert binding.calls == before


def test_view_retains_owner_and_explicit_copy_mode(binding):
    binding.functions["rpu_tp_view"] = lambda *args: binding.calls.append(("view", args)) or 8
    binding.functions["rpu_tp_prepare"] = lambda *args: binding.calls.append(("prepare", args)) or 9
    with rpu_tp.RpuTpContext() as context, tensor(7, None, shape=(1024,)) as owner:
        owner.context = context
        with context.view(owner, (16, 16), offset_bytes=256, distribution="sharded", shard_axis=1) as view:
            assert view._owner is owner
            assert binding.calls[-1] == ("view", (7, (16, 16), 256, "sharded", 1))
            with context.prepare_copy(view, view, mode="corewise"):
                assert binding.calls[-1][0] == "prepare"
                args = binding.calls[-1][1]
                assert args[1] == "copy_f16"
                assert args[4:6] == (["mode"], [0])
            with pytest.raises(ValueError, match="copy mode"):
                context.prepare_copy(view, view, mode="implicit")


def test_readonly_weights_and_uploads_keep_dtype_bytes_and_caller_flags(binding):
    original = np.arange(256, dtype=np.uint16).view(np.float16).reshape(16, 16)
    original.setflags(write=False)
    expected = original.tobytes()
    seen = []

    def consume(array):
        actual = np.from_dlpack(array)
        assert actual.dtype == original.dtype
        assert actual.shape == original.shape
        assert actual.tobytes() == expected
        assert not np.shares_memory(array, original)
        seen.append(array)
        return 11

    binding.functions.update({
        "rpu_prepare_weight": lambda _, host: consume(host),
        "rpu_tp_weight": lambda _, host, layout: consume(host),
        "rpu_buffer_upload": lambda _, host: consume(host),
        "rpu_tp_upload": lambda _, host: consume(host),
    })
    with rpu.RpuContext() as single, rpu_tp.RpuTpContext() as tp:
        single.prepare_weight(original).close()
        tp.prepare_weight(original, layout="nk16").close()
        rpu.RpuBuffer(12, single, original.shape).copy_from(original).close()
        tensor(13, tp, shape=original.shape, distribution="replicated").copy_from(original).close()
    assert len(seen) == 4
    assert original.tobytes() == expected
    assert not original.flags.writeable


def test_writable_upload_is_forwarded_and_noncontiguous_readonly_rejected(binding):
    original = np.zeros((16, 16), dtype=np.float16)
    seen = []
    binding.functions["rpu_tp_upload"] = lambda _, host: seen.append(host)
    with tensor(7, None, shape=original.shape, distribution="replicated") as resident:
        assert resident.copy_from(original) is resident
        assert seen[0] is original
        noncontiguous = np.zeros((16, 32), dtype=np.float16)[:, ::2]
        noncontiguous.setflags(write=False)
        with pytest.raises(ValueError, match="C-contiguous"):
            resident.copy_from(noncontiguous)
        assert len(seen) == 1
        assert not noncontiguous.flags.writeable


def test_readonly_download_target_is_not_repaired(binding):
    target = np.zeros((16, 16), dtype=np.float16)
    target.setflags(write=False)
    binding.functions["rpu_tp_download"] = lambda host, _: host.__dlpack__()
    with tensor(7, None, shape=target.shape, distribution="replicated") as resident:
        with pytest.raises(BufferError):
            resident.copy_to(target)
    assert not target.flags.writeable


@pytest.mark.parametrize("layout,shape", [("nk16", (16, 32)), ("tp8_nk16", (8, 16, 32)),
                                         ("tp8_nk16", (4, 8, 16, 32))])
def test_packed_weights_cannot_download(binding, layout, shape):
    with tensor(7, None, shape=shape, layout=layout, storage="ddr", distribution="shared") as resident:
        with pytest.raises(ValueError, match="packed"):
            resident.numpy()
        with pytest.raises(ValueError, match="packed"):
            resident.copy_to(object())


def test_failed_wait_still_releases_execution_once(binding):
    def fail_wait(handle):
        binding.calls.append(("wait", handle))
        raise RuntimeError("completion marker missing")

    binding.functions["rpu_tp_wait"] = fail_wait
    execution = rpu_tp.RpuTpExecution(7)
    with pytest.raises(RuntimeError, match="completion marker"):
        execution.close()
    assert binding.calls == [("wait", 7), ("release", 7)]
    execution.close()
    assert binding.calls == [("wait", 7), ("release", 7)]
    with pytest.raises(RuntimeError, match="closed"):
        execution.execute()


def test_default_batch_has_no_implicit_window_or_execution(binding):
    binding.functions["rpu_tp_batch"] = lambda *args: binding.calls.append(("batch", args)) or 10
    with rpu_tp.RpuTpContext() as context:
        with rpu_tp.RpuTpOperation(7, context, "all_reduce_residual") as ar:
            with rpu_tp.RpuTpOperation(8, context, "down_allreduce_rmsnorm_m16n1024k256_acc32") as down:
                batch = context.batch([ar, down])
                assert binding.calls == [("context", (False,)), ("batch", (1, [7, 8]))]
                batch.close()
                assert binding.calls[-2:] == [("wait", 10), ("release", 10)]
                before = list(binding.calls)
                with pytest.raises(RuntimeError, match="cannot mix"):
                    context.batch([ar, down], collective_window=4)
                assert binding.calls == before
