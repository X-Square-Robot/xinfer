// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/tp_contract.hpp"
#include "common/rpu/denoise_abi.hpp"
#include "common/rpu/attention_abi.hpp"
#include "common/rpu/mlp_attention_abi.hpp"
#include "common/rpu/mlp_projection.hpp"

#include <limits>

namespace xinfer::rpu::tp::matrix_validation {
inline std::uint32_t dim(std::int64_t value, const char* name) {
  require(value > 0 && std::uint64_t(value) <= std::numeric_limits<std::uint32_t>::max(),
          std::string(name) + " must be positive uint32");
  return static_cast<std::uint32_t>(value);
}
inline std::uint32_t nonnegative(std::int64_t value, const char* name) {
  require(value >= 0 && std::uint64_t(value) <= std::numeric_limits<std::uint32_t>::max(),
          std::string(name) + " must be nonnegative uint32");
  return static_cast<std::uint32_t>(value);
}
inline float scalar(const Request& r, const char* name, double fallback) {
  const auto value = r.scalar(name, fallback);
  const auto result = static_cast<float>(value);
  require(std::isfinite(result), std::string(name) + " exceeds float32 range");
  if (std::string(name) == "epsilon") {
    const auto rounded = fp16_to_fp32(fp32_to_fp16(result));
    require(rounded > 0 && std::isfinite(rounded),
            "epsilon must round to positive finite FP16");
  }
  return result;
}
inline const Tensor& local(const Request& r, const char* name,
                    std::initializer_list<std::int64_t> dims) {
  const auto& t = r.tensor(name);
  shape(t, dims); fp16(t); layout(t, "row_major"); spm(t);
  return t;
}
inline void role(const Tensor& t, const char* value, std::int64_t axis = -1) {
  distribution(t, value);
  require(t.shard_axis == axis, "incompatible TP shard axis");
}
inline void per_core(const Tensor& t) {
  require(t.distribution == "local" || t.distribution == "replicated" ||
              t.distribution == "sharded", "local operation cannot consume partial or single data");
  if (t.distribution == "sharded")
    require(t.shard_axis >= 0 && std::size_t(t.shard_axis) < t.shape.size(),
            "sharded tensor axis is outside its rank");
  else require(t.shard_axis == -1, "unsharded tensor must have shard_axis=-1");
}
inline void same_role(const Tensor& out, const Tensor& input) {
  per_core(input); per_core(out);
  require(out.distribution == input.distribution && out.shard_axis == input.shard_axis,
          "output TP identity differs from input");
}
inline void replicated_parameter(const Tensor& input, const Tensor& parameter, const char* name) {
  require(input.distribution != "replicated" || parameter.distribution == "replicated",
          std::string(name) + " must be replicated when input/output are replicated");
}
inline const Tensor& weight(const Request& r, const char* name,
                     std::initializer_list<std::int64_t> dims) {
  const auto& t = r.tensor(name);
  shape(t, dims); fp16(t); layout(t, "tp8_nk16"); role(t, "shared"); ddr(t);
  require(t.immutable, "TP weight must be prepared and immutable");
  return t;
}
inline const Tensor& table(const Request& r, const char* name,
                    std::initializer_list<std::int64_t> dims) {
  const auto& t = r.tensor(name);
  shape(t, dims); fp16(t); layout(t, "row_major"); role(t, "shared"); ddr(t);
  return t;
}
inline void finish(const Request& r, Lowered& out, std::initializer_list<const char*> writes) {
  for (const auto* name : writes) {
    writable(r.tensor(name)); out.writes.emplace_back(name);
  }
  r.finish();
}
inline abi::Entry entry(const std::string& name) {
  for (std::size_t i = 0; i < abi::names.size(); ++i)
    if (abi::names[i] && name == abi::names[i]) return static_cast<abi::Entry>(i);
  throw std::invalid_argument("unknown RPU entry: " + name);
}
inline void fused(const denoise::Plan& plan, Lowered& out) {
  out.plan.entry = entry(plan.name); out.plan.regs = plan.regs;
  out.plan.register_count = plan.register_count; out.plan.grid = plan.grid;
  out.plan.cores = plan.cores;
  for (auto id : plan.zero_counters) out.zero_counters.push_back(128 + 2 * id);
}
inline std::vector<unsigned> lengths(const Request& r) {
  std::vector<unsigned> result;
  bool ended = false;
  for (unsigned i = 0; i < 3; ++i) {
    const auto n = r.integer("length" + std::to_string(i), 0);
    require(n >= 0 && n <= 308 && (!ended || n == 0),
            "attention lengths require contiguous positive lengths <=308");
    if (n == 0) ended = true;
    else result.push_back(static_cast<unsigned>(n));
  }
  require(!result.empty(), "attention requires length0");
  require(r.scalar("scale", 0.125) == 0.125, "attention supports scale=1/8 only");
  require(r.integer("causal", 0) == 0, "Vision attention is noncausal");
  require(r.integer("return_lse", 0) == 0, "RPU attention does not produce LSE");
  return result;
}
}  // namespace xinfer::rpu::tp::matrix_validation
