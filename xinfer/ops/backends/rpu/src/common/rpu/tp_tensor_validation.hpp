// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/tp_contract.hpp"
#include "common/rpu/denoise_abi.hpp"
#include "common/rpu/causal_attention_abi.hpp"
#include "common/rpu/gdr_prefill_abi.hpp"
#include "common/rpu/elementwise_abi.hpp"
#include "common/rpu/mlp_attention_abi.hpp"
#include "common/rpu/mlp_projection.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace xinfer::rpu::tp::tensor_validation {
inline void check_tensor(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("RPU TP: " + message);
}
inline std::uint32_t u32(std::int64_t value, const std::string& name) {
  check_tensor(value >= 0 && std::uint64_t(value) <= UINT32_MAX,
          name + " must fit uint32");
  return static_cast<std::uint32_t>(value);
}
inline std::uint64_t elements(const Tensor& t, const std::string& name) {
  check_tensor(!t.shape.empty() && t.shape.size() <= 6, name + " rank must be 1..6");
  std::uint64_t size = 1;
  for (const auto dimension : t.shape) {
    check_tensor(dimension > 0 && std::uint64_t(dimension) <= UINT64_MAX / size,
            name + " has invalid/overflowing shape");
    size *= dimension;
  }
  return size;
}
inline const Tensor& tensor(const Request& r, const char* name,
                     const std::string& storage = "spm",
                     const std::string& layout = "row_major",
                     const std::string& dtype = "float16") {
  const auto& t = r.tensor(name);
  check_tensor(t.storage == storage, std::string(name) + " requires " + storage);
  check_tensor(t.layout == layout, std::string(name) + " requires layout " + layout);
  check_tensor(t.dtype == dtype, std::string(name) + " requires " + dtype);
  const auto count = elements(t, name);
  const auto itemsize = dtype == "int32" ? 4U : 2U;
  check_tensor(count <= UINT64_MAX / itemsize && t.bytes >= count * itemsize,
          std::string(name) + " allocation is smaller than its shape");
  if (storage == "spm") {
    check_tensor(t.address <= UINT32_MAX, std::string(name) + " is not a LocalSPM offset");
    abi::detail::span(static_cast<std::uint32_t>(t.address), count * itemsize);
  } else {
    abi::detail::ddr_span(t.address, count * itemsize);
  }
  return t;
}
inline void shape(const Tensor& t, const std::vector<std::int64_t>& expected,
           const std::string& name) {
  check_tensor(t.shape == expected, name + " has incompatible logical shape");
}
inline void distribution(const Tensor& t, const char* expected, const char* name) {
  check_tensor(t.distribution == expected,
          std::string(name) + " requires distribution " + expected);
  if (std::string(expected) == "sharded")
    check_tensor(t.shard_axis >= 0 && std::size_t(t.shard_axis) < t.shape.size(),
               std::string(name) + " requires an explicit shard axis");
}
inline void same_distribution(const Tensor& output, const Tensor& input,
                       bool nonlinear = false) {
  check_tensor(output.distribution == input.distribution,
          "output distribution must match input distribution");
  if (input.distribution == "sharded" || input.distribution == "sequence_shard") {
    check_tensor(input.shard_axis >= 0 &&
                   std::size_t(input.shard_axis) < input.shape.size() &&
                   output.shard_axis == input.shard_axis,
               "output shard axis must match the valid input shard axis");
  }
  check_tensor(!nonlinear || input.distribution != "partial",
          "nonlinear operation cannot consume an unreduced TP partial");
}
inline std::uint32_t address(const Tensor& t) {
  return static_cast<std::uint32_t>(t.address);
}
inline const Tensor& matrix(const Request& r, const char* name,
                     std::int64_t rows, std::int64_t cols,
                     const char* dist = nullptr) {
  const auto& t = tensor(r, name);
  shape(t, {rows, cols}, name);
  if (dist) {
    distribution(t, dist, name);
    if (std::string(dist) == "sharded")
      check_tensor(t.shard_axis == 1, std::string(name) + " must shard columns (axis 1)");
  }
  return t;
}
inline std::uint32_t rows(const Tensor& t, const char* name) {
  check_tensor(t.shape.size() == 2, std::string(name) + " must have rank 2");
  return u32(t.shape[0], name);
}
inline const Tensor& vector(const Request& r, const char* name, std::int64_t count,
                     const char* dist = nullptr) {
  const auto& t = tensor(r, name);
  shape(t, {count}, name);
  if (dist) distribution(t, dist, name);
  return t;
}
inline std::uint32_t integer(const Request& r, const char* name, std::int64_t fallback) {
  return u32(r.integer(name, fallback), name);
}
inline bool flag(const Request& r, const char* name, bool fallback) {
  const auto value = r.integer(name, fallback ? 1 : 0);
  check_tensor(value == 0 || value == 1, std::string(name) + " must be 0 or 1");
  return value != 0;
}
inline float scalar(const Request& r, const char* name, double fallback) {
  const auto value = r.scalar(name, fallback);
  check_tensor(std::isfinite(value) && std::isfinite(static_cast<float>(value)),
          std::string(name) + " must be finite FP32");
  if (std::string(name) == "epsilon" || std::string(name) == "delta_t") {
    const auto rounded = fp16_to_fp32(fp32_to_fp16(static_cast<float>(value)));
    check_tensor(value > 0 && rounded > 0 && std::isfinite(rounded),
               std::string(name) + " must round to positive finite FP16");
  }
  return static_cast<float>(value);
}
inline bool done(const Request& r, Lowered& out, const abi::LaunchPlan& p) {
  out.plan = p;
  if (r.op == "all_reduce_residual_layernorm") out.writes = {"x", "norm"};
  else if (r.op.find("down_allreduce_rmsnorm_") == 0) {
    out.writes = {"partial", "stream", "norm"};
    if (r.used_tensors.count("w_out")) out.writes.push_back("w_out");
  } else if (r.op == "gdr_prefill_f16") {
    out.writes = {"state", "scratch", "out"};
    // Optional outputs become writes only after their enabled branch binds
    // and validates them. finish() rejects extra taps when debug is disabled.
    if (r.used_tensors.count("raw_out")) out.writes.push_back("raw_out");
    if (r.used_tensors.count("norm_out")) out.writes.push_back("norm_out");
  } else if (r.op.find("euler_update_") == 0) out.writes = {"action", "velocity"};
  else {
    for (const auto& name : r.used_tensors)
      if (name == "out" || name.find("out_") == 0) out.writes.push_back(name);
  }
  for (const auto& name : out.writes) writable(r.tensor(name));
  r.finish();
  return true;
}
inline const Tensor& weight(const Request& r, std::int64_t n, std::int64_t k) {
  const auto& w = tensor(r, "weight", "ddr", "tp8_nk16");
  shape(w, {8, n, k}, "weight");
  check_tensor(w.immutable, "weight must be prepared immutable storage");
  distribution(w, "shared", "weight");
  return w;
}
inline abi::Entry entry(const std::string& name) {
  for (std::size_t i = 0; i < abi::names.size(); ++i)
    if (abi::names[i] && name == abi::names[i]) return static_cast<abi::Entry>(i);
  throw std::invalid_argument("RPU TP: unknown entry " + name);
}
template<class Plan>
inline abi::LaunchPlan convert(const Plan& source, abi::Entry which) {
  abi::LaunchPlan p{which};
  p.regs = source.regs;
  p.register_count = source.register_count;
  p.grid = source.grid;
  p.cores = source.cores;
  return p;
}
inline const Tensor& cache(const Request& r, const char* name, bool value) {
  const auto& t = tensor(r, name, "ddr", value ? "causal_v_cache" : "causal_k_cache");
  shape(t, {8, 784, 256}, name);
  distribution(t, "shared", name);
  check_tensor(t.state_kind == "cache", std::string(name) + " must be an owned cache state");
  return t;
}
}  // namespace xinfer::rpu::tp::tensor_validation
