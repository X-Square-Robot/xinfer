// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/tp_contract.hpp"
#include "common/rpu/attention_abi.hpp"
#include "common/rpu/causal_attention_abi.hpp"
#include "common/rpu/conv_abi.hpp"
#include "common/rpu/conv_silu_prefill_abi.hpp"
#include "common/rpu/elementwise_abi.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace xinfer::rpu::tp::state_validation {
inline void need(bool value, const std::string& message) {
  if (!value) throw std::invalid_argument("RPU TP: " + message);
}
inline std::uint64_t count_elements(const Tensor& t) {
  need(!t.shape.empty(), "tensor must have positive rank");
  std::uint64_t size = 1;
  for (auto d : t.shape) {
    need(d > 0 && std::uint64_t(d) <= std::numeric_limits<std::uint64_t>::max() / size,
            "tensor shape is empty or overflows");
    size *= d;
  }
  need(size <= std::numeric_limits<std::uint64_t>::max() / 2 &&
              t.bytes >= 2 * size,
          "tensor allocation is shorter than its FP16 shape");
  return size;
}
inline const Tensor& data(const Request& r, const std::string& name,
                   const std::string& storage = "spm",
                   const std::string& layout = "row_major") {
  const auto& t = r.tensor(name);
  need(t.dtype == "float16", name + " requires float16");
  need(t.storage == storage, name + " requires " + storage + " storage");
  need(t.layout == layout, name + " requires layout " + layout);
  count_elements(t);
  if (storage == "spm") {
    need(t.address <= std::numeric_limits<std::uint32_t>::max(), name + " SPM address overflows");
    abi::detail::span(static_cast<std::uint32_t>(t.address), 2 * count_elements(t));
  } else {
    need(t.distribution == "shared" && t.shard_axis == -1,
            name + " DDR tensor must be shared without a shard axis");
    abi::detail::ddr_span(t.address, t.bytes);
  }
  return t;
}
inline void expect_shape(const Tensor& t, std::initializer_list<std::int64_t> wanted,
           const std::string& name) {
  need(t.shape == std::vector<std::int64_t>(wanted), name + " has incompatible shape");
}
inline void same_expect_shape(const Tensor& a, const Tensor& b, const std::string& name) {
  need(a.shape == b.shape, name + " shapes differ");
}
inline void expect_distribution(const Tensor& t, const std::string& wanted, const std::string& name) {
  need(t.distribution == wanted, name + " requires distribution " + wanted);
}
inline unsigned integer(const Request& r, const std::string& name, std::int64_t fallback,
                 unsigned maximum = std::numeric_limits<unsigned>::max()) {
  const auto n = r.integer(name, fallback);
  need(n >= 0 && std::uint64_t(n) <= maximum, name + " is negative or out of range");
  return static_cast<unsigned>(n);
}
inline std::uint32_t local(const Tensor& t) { return static_cast<std::uint32_t>(t.address); }
inline unsigned rows(const Tensor& t, unsigned columns, const std::string& name) {
  need(t.shape.size() == 2 && t.shape[1] == columns &&
              t.shape[0] <= std::numeric_limits<unsigned>::max(),
          name + " has incompatible row-major matrix shape");
  return static_cast<unsigned>(t.shape[0]);
}
inline std::uint16_t finite_half(double value, const std::string& name) {
  need(std::isfinite(value) && std::abs(value) <= 65504.0,
          name + " must be finite and representable as finite FP16");
  return fp32_to_fp16(static_cast<float>(value));
}
inline const Tensor& cache(const Request& r, const std::string& name, bool value,
                    bool causal, unsigned capacity) {
  const auto& t = data(r, name, "ddr", causal ? (value ? "causal_v_cache" : "causal_k_cache")
                                            : (value ? "vision_v_cache" : "vision_k_cache"));
  expect_shape(t, {8, capacity, causal ? 256 : 128}, name);
  need(t.state_kind == "cache", name + " requires a cache state owner");
  return t;
}
inline void local_pair(const Tensor& out, const Tensor& input) {
  same_expect_shape(out, input, "out/input");
  need(out.distribution == input.distribution && out.shard_axis == input.shard_axis,
          "local operation cannot change tensor distribution");
}
inline void conv_output(const Tensor& t, unsigned channel_axis, const std::string& name) {
  // Each core reads its own [4,256] weights. Shared DDR storage does not
  // establish equal weights, so convolution cannot promise replicated output.
  need(t.distribution == "local" || t.distribution == "sharded",
       name + " convolution output must be local or channel-sharded");
  need(t.shard_axis == (t.distribution == "sharded" ? std::int64_t(channel_axis) : -1),
       name + " convolution shard_axis must identify channels");
}
inline void conv_dependency(const Tensor& t, const Tensor& output, unsigned channel_axis,
                     const std::string& name, bool allow_replicated = true) {
  need(t.distribution == output.distribution ||
           (allow_replicated && t.distribution == "replicated"),
       name + " must match the convolution output distribution or be replicated");
  need(t.shard_axis == (t.distribution == "sharded" ? std::int64_t(channel_axis) : -1),
       name + " convolution shard_axis must identify channels");
}
inline void finish_state(const Request& r, Lowered& result) {
  const auto& op = r.op;
  // Do not consume an unsolicited output merely because the caller supplied
  // that name: e.g. prefix-copy writes its named caches, never a generic out.
  if (r.used_tensors.count("out")) result.writes = {"out"};
  if (op == "kcache_insert_f16" || op == "vcache_insert_f16" ||
      op == "kcache_insert_d256_f16" || op == "vcache_insert_d256_f16")
    result.writes = {"cache"};
  if (op == "vcache_insert_staged_d256_f16") result.writes = {"cache", "value_spm"};
  if (op == "causal_depthwise_conv1d_prefill_f16") result.writes = {"pad", "carry"};
  if (op == "causal_depthwise_conv1d_step_f16") result.writes = {"state"};
  if (op == "conv_silu_prefill_f16") result.writes = {"qout", "kout", "vout", "history_out"};
  if (op == "copy_physical_prefix_f16")
    for (unsigned layer = 0; layer < 6; ++layer) {
      result.writes.push_back("dst_k" + std::to_string(layer));
      result.writes.push_back("dst_v" + std::to_string(layer));
    }
  for (const auto& name : result.writes) writable(r.tensor(name));
  r.finish();
}
}  // namespace xinfer::rpu::tp::state_validation
