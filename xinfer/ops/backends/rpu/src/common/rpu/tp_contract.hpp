// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/kernel_abi.hpp"
#include "common/rpu/library.hpp"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace xinfer::rpu::tp {
inline void require(bool value, const std::string& message) {
  if (!value) throw std::invalid_argument("RPU TP: " + message);
}
// SPM shape is per physical core; DDR shape describes the whole allocation.
// The runtime supplies addresses, identity, generation and upload statistics.
// Callers cannot manufacture these fields through the FFI.
struct Tensor {
  std::vector<std::int64_t> shape;
  std::string dtype = "float16", storage = "spm", layout = "row_major";
  std::string distribution = "local";
  std::uint64_t address = 0, bytes = 0, identity = 0;
  std::int64_t shard_axis = -1;
  std::string state_kind = "none";
  std::uint64_t generation = 0;
  bool immutable = false, values_known = false, finite = false;
  double minimum = 0, maximum = 0;
};
inline std::uint64_t elements(const Tensor& t) {
  require(!t.shape.empty() && t.shape.size() <= 6, "tensor rank must be 1..6");
  std::uint64_t count = 1;
  for (auto d : t.shape) {
    require(d > 0 && std::uint64_t(d) <= (1ULL << 40) / count,
            "tensor shape is empty or overflows");
    count *= static_cast<std::uint64_t>(d);
  }
  return count;
}
inline std::uint32_t spm(const Tensor& t) {
  require(t.storage == "spm", "expected LocalSPM tensor");
  require(t.address < abi::spm_bytes, "SPM address exceeds core range");
  abi::detail::span(static_cast<std::uint32_t>(t.address), t.bytes);
  return static_cast<std::uint32_t>(t.address);
}
inline std::uint64_t ddr(const Tensor& t) {
  require(t.storage == "ddr", "expected DDR tensor");
  abi::detail::ddr_span(t.address, t.bytes);
  return t.address;
}
inline void shape(const Tensor& t, std::initializer_list<std::int64_t> dims) {
  require(t.shape == std::vector<std::int64_t>(dims), "incompatible tensor shape");
}
inline void fp16(const Tensor& t) {
  require(t.dtype == "float16", "tensor dtype must be float16");
  require(t.bytes >= elements(t) * 2, "FP16 tensor capacity is insufficient");
}
inline void layout(const Tensor& t, const char* value) {
  require(t.layout == value, std::string("expected layout ") + value);
}
inline void distribution(const Tensor& t, const char* value) {
  require(t.distribution == value, std::string("expected distribution ") + value);
}
inline void writable(const Tensor& t) { require(!t.immutable, "output is immutable"); }
struct ValueRequirement { std::string tensor; double minimum, maximum; };
// Cache intervals use logical token positions. Other state owners use [0,1).
// Ordered batches validate reads and publish writes in caller-supplied order.
struct StateAccess {
  std::string tensor;
  std::uint64_t begin = 0, end = 1;
  bool write = false;
};
struct Request {
  std::string op;
  std::map<std::string, Tensor> tensors;
  std::map<std::string, std::int64_t> integers;
  std::map<std::string, double> scalars;
  mutable std::set<std::string> used_tensors, used_integers, used_scalars;
  mutable std::vector<ValueRequirement> value_requirements;
  const Tensor& tensor(const std::string& name) const {
    const auto found = tensors.find(name);
    require(found != tensors.end(), op + " requires tensor " + name);
    used_tensors.insert(name);
    return found->second;
  }
  bool has_tensor(const std::string& name) const { return tensors.count(name) != 0; }
  std::int64_t integer(const std::string& name, std::int64_t fallback) const {
    const auto found = integers.find(name);
    if (found == integers.end()) return fallback;
    used_integers.insert(name); return found->second;
  }
  std::int64_t integer(const std::string& name) const {
    require(integers.count(name), op + " requires integer " + name);
    return integer(name, 0);
  }
  double scalar(const std::string& name, double fallback) const {
    const auto found = scalars.find(name);
    if (found == scalars.end()) return fallback;
    used_scalars.insert(name);
    require(std::isfinite(found->second), op + " scalar " + name + " must be finite");
    return found->second;
  }
  void require_values(const std::string& name, double lo, double hi) const {
    const auto& t = tensor(name);
    require(t.values_known && t.finite && t.minimum >= lo && t.maximum <= hi,
            op + " requires uploaded finite " + name + " in [" +
            std::to_string(lo) + "," + std::to_string(hi) + "]");
    value_requirements.push_back({name, lo, hi});
  }
  void finish() const {
    for (const auto& item : tensors)
      require(used_tensors.count(item.first), op + " unsupported tensor " + item.first);
    for (const auto& item : integers)
      require(used_integers.count(item.first), op + " unsupported integer " + item.first);
    for (const auto& item : scalars)
      require(used_scalars.count(item.first), op + " unsupported scalar " + item.first);
  }
};
struct Lowered {
  abi::LaunchPlan plan{};
  int transfer_mode = -1;
  // 32-bit atomic register IDs, e.g. 128/130, reinitialized in every replay.
  std::vector<unsigned> zero_counters;
  bool collective = false;
  std::vector<std::string> writes;
  std::vector<StateAccess> state_accesses;
};
XINFER_RPU_API bool lower_operator(const Request&, Lowered&);
}  // namespace xinfer::rpu::tp
