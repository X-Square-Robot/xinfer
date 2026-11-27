// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/tp_contract.hpp"

namespace xinfer::rpu::tp {
// A view has the same storage identity as its owner. Runtime aliases retain
// that owner and share invalidation; this descriptor alone grants no lifetime.
inline Tensor contiguous_view(const Tensor& owner,
    const std::vector<std::int64_t>& shape, std::int64_t offset,
    const std::string& distribution_name, std::int64_t axis) {
  require(owner.layout == "row_major" && owner.state_kind == "none",
          "views require row_major storage without state/cache semantics");
  require(offset >= 0 && offset % 8 == 0, "view offset must be nonnegative and 8-byte aligned");
  Tensor result = owner;
  result.shape = shape;
  require(owner.dtype == "float16" || owner.dtype == "int32", "unsupported view dtype");
  result.bytes = elements(result) * (owner.dtype == "float16" ? 2 : 4);
  require(result.bytes % 8 == 0, "view region must contain complete 8-byte words");
  require(std::uint64_t(offset) <= owner.bytes && result.bytes <= owner.bytes - offset,
          "view exceeds its parent region");
  if (owner.storage == "ddr") {
    require(distribution_name == "shared" && axis == -1, "DDR view must remain shared");
  } else {
    const std::set<std::string> roles{"local", "replicated", "sharded", "partial",
                                    "single", "owner0", "sequence_shard"};
    require(roles.count(distribution_name), "invalid SPM view distribution");
    require(owner.distribution == "local" || distribution_name == owner.distribution,
            "view cannot change an established TP distribution");
    require(owner.distribution == distribution_name ||
                (distribution_name != "single" && distribution_name != "owner0"),
            "single/owner0 views require an owner with zero nonowner cores");
    const bool sharded = distribution_name == "sharded" || distribution_name == "sequence_shard";
    require(sharded ? axis >= 0 && std::size_t(axis) < shape.size() : axis == -1,
            "view shard axis does not match its distribution");
  }
  result.address += static_cast<std::uint64_t>(offset);
  result.distribution = distribution_name;
  result.shard_axis = axis;
  return result;
}

// mode: 0=per-core, 1=shared DDR broadcast, 2=core0, 3=shared DDR.
// Transfers are SDK DMA commands in the same retained execution, not kernels
// or host downloads. Cache codecs and packed matrices are deliberately absent.
inline bool lower_copy(const Request& r, Lowered& lowered) {
  if (r.op != "copy_f16") return false;
  const auto& src = r.tensor("input");
  const auto& dst = r.tensor("out");
  fp16(src); fp16(dst); layout(src, "row_major"); layout(dst, "row_major"); writable(dst);
  const auto mode = r.integer("mode", 0);
  require(mode >= 0 && mode <= 3, "invalid copy mode");
  const bool src_spm = src.storage == "spm", dst_spm = dst.storage == "spm";
  require((src_spm || src.storage == "ddr") && (dst_spm || dst.storage == "ddr"),
          "copy storage must be SPM or DDR");
  auto expanded = [](const Tensor& t) {
    auto dims = t.shape; dims.insert(dims.begin(), 8); return dims;
  };
  if (mode == 0) {
    require(src_spm || dst_spm, "corewise copy requires an SPM endpoint");
    if (src_spm && dst_spm) {
      require(src.shape == dst.shape && src.distribution == dst.distribution &&
              src.shard_axis == dst.shard_axis, "corewise copy TP identity/shape mismatch");
    } else {
      const auto& local = src_spm ? src : dst;
      const auto& shared = src_spm ? dst : src;
      require(shared.shape == expanded(local), "corewise DDR shape must be [8,*SPM shape]");
      require(src_spm || dst.distribution != "replicated",
              "per-core DDR contents cannot establish replicated output; use broadcast mode");
    }
    require((!src_spm || (src.distribution != "single" && src.distribution != "owner0")) &&
            (!dst_spm || (dst.distribution != "single" && dst.distribution != "owner0")),
            "single/owner0 transfers require core0 mode");
  } else if (mode == 1) {
    require(!src_spm && dst_spm && dst.distribution == "replicated" && src.shape == dst.shape,
            "broadcast copy requires shared DDR to replicated SPM of the same shape");
  } else if (mode == 2) {
    require(src_spm != dst_spm && src.shape == dst.shape,
            "core0 copy requires one SPM and one DDR endpoint of the same shape");
    const auto& local = src_spm ? src : dst;
    require(local.distribution == "single" || local.distribution == "owner0" ||
                (src_spm && local.distribution == "replicated"),
            "core0 copy needs single/owner0 SPM, or a replicated source");
  } else {
    require(!src_spm && !dst_spm && src.shape == dst.shape,
            "shared copy requires two DDR tensors of the same shape");
  }
  for (const auto* t : {&src, &dst}) {
    if (t->storage == "ddr") distribution(*t, "shared");
    require(t->address % 8 == 0 && t->bytes % 8 == 0,
            "DMA regions must be 8-byte aligned and sized");
  }
  if (src.storage == dst.storage && src.identity == dst.identity)
    require(src.address + src.bytes <= dst.address || dst.address + dst.bytes <= src.address,
            "copy regions must not overlap");
  lowered.transfer_mode = static_cast<int>(mode);
  lowered.writes = {"out"};
  if (src.state_kind != "none") lowered.state_accesses.push_back({"input", 0, 1, false});
  if (dst.state_kind != "none") lowered.state_accesses.push_back({"out", 0, 1, true});
  r.finish();
  return true;
}
}  // namespace xinfer::rpu::tp
