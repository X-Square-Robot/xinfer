// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common/rpu/execution.hpp"
#include <string>
#include <vector>

namespace xinfer::rpu::execution {
XINFER_RPU_API Handle tp_context_create(bool allow_candidate);
XINFER_RPU_API Handle tp_context_create_with_pool(bool allow_candidate, std::int64_t retained_segments);
XINFER_RPU_API std::vector<std::int64_t> tp_batch_stats(Handle batch);
XINFER_RPU_API std::vector<std::int64_t> tp_context_stats(Handle context);
XINFER_RPU_API std::vector<std::int64_t> tp_pool_stats(Handle context);
XINFER_RPU_API Handle tp_empty(Handle context, const std::vector<std::int64_t>& shape,
    const std::string& dtype, const std::string& storage, const std::string& layout,
    const std::string& distribution, std::int64_t shard_axis, const std::string& state_kind);
XINFER_RPU_API void tp_upload(Handle tensor, const void* data,
    const std::vector<std::int64_t>& shape, const std::string& dtype);
XINFER_RPU_API void tp_download(void* data, Handle tensor,
    const std::vector<std::int64_t>& shape, const std::string& dtype);
XINFER_RPU_API Handle tp_weight(Handle context, const void* data,
    const std::vector<std::int64_t>& shape, const std::string& layout);
XINFER_RPU_API void tp_reset(Handle tensor);
XINFER_RPU_API void tp_invalidate(Handle tensor);
XINFER_RPU_API Handle tp_view(Handle tensor, const std::vector<std::int64_t>& shape,
    std::int64_t offset_bytes, const std::string& distribution, std::int64_t shard_axis);
XINFER_RPU_API Handle tp_prepare(Handle context, const std::string& op,
    const std::vector<std::string>& names, const std::vector<Handle>& tensors,
    const std::vector<std::string>& integer_names, const std::vector<std::int64_t>& integers,
    const std::vector<std::string>& scalar_names, const std::vector<double>& scalars);
// The kernel a prepared operation selected and the catalog entry it resolved
// to: {kernel, entry, family, admission}. entry is the catalog name the
// request lowered through; kernel may differ when an automatic variant is selected.
XINFER_RPU_API std::vector<std::string> tp_selection(Handle operation);
XINFER_RPU_API Handle tp_batch(Handle context, const std::vector<Handle>& nodes);
// Host-only AR-window protocol capability;0 means unavailable/quarantined.
XINFER_RPU_API std::int64_t tp_ar_window_max();
// Calibrated M16 DownNorm-only windows2..4; does not create a new device.
XINFER_RPU_API std::int64_t tp_down_norm_window_max();
XINFER_RPU_API Handle tp_batch_with_window(Handle context, const std::vector<Handle>& nodes,
    std::int64_t collective_window);
// Whole-stage counter discipline: one retained queue for the whole batch,
// request-pinned counter ids 3..49 with per-launch zeroing {0,1,own}, no
// isolation segments and no initialization sweeps. Requires candidate admission.
XINFER_RPU_API Handle tp_batch_fused(Handle context, const std::vector<Handle>& nodes);
XINFER_RPU_API void tp_rebind(Handle operation, const std::vector<std::string>& names,
    const std::vector<Handle>& tensors, const std::vector<std::string>& integer_names,
    const std::vector<std::int64_t>& integers, const std::vector<std::string>& scalar_names,
    const std::vector<double>& scalars);
XINFER_RPU_API void tp_execute(Handle batch, bool asynchronous);
XINFER_RPU_API void tp_wait(Handle batch);
}  // namespace xinfer::rpu::execution
