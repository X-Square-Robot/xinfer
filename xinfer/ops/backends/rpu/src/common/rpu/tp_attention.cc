// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Validate and lower attention requests to native launch plans.
#include "common/rpu/tp_lowering.hpp"
#include "common/rpu/tp_matrix_validation.hpp"
#include "common/rpu/tp_tensor_validation.hpp"
#include "common/rpu/tp_state_validation.hpp"

namespace xinfer::rpu::tp::detail {
namespace {
bool lower_matrix_validation(const Request& r, Lowered& out) {
  using namespace matrix_validation;
  const auto& op = r.op;
  if (op == "rope_f16") {
    const auto& x = r.tensor("x");
    require(x.shape.size() == 3, "RoPE requires [tokens,heads,head_dim]");
    const auto tokens = dim(x.shape[0], "tokens"), heads = dim(x.shape[1], "heads"), d = dim(x.shape[2], "head_dim");
    const auto& c = r.tensor("cosine");
    require(c.shape.size() == 2, "RoPE cosine requires [table_rows,rotary_dim/2]");
    const auto table_rows = dim(c.shape[0], "table_rows");
    const auto rotary = dim(c.shape[1], "half_rotary_dim") * 2ULL;
    require(rotary <= std::numeric_limits<std::uint32_t>::max(), "rotary_dim overflow");
    local(r, "x", {tokens, heads, d});
    const auto& y = local(r, "out", {tokens, heads, d});
    local(r, "cosine", {table_rows, std::int64_t(rotary / 2)});
    const auto& s = local(r, "sine", {table_rows, std::int64_t(rotary / 2)});
    const auto tp8 = r.integer("tp8", 0);
    require(tp8 == 0 || tp8 == 1, "RoPE tp8 must be 0 or 1");
    if (tp8) {
      role(x, "sharded", 1); role(y, "sharded", 1);
      role(c, "replicated"); role(s, "replicated");
    } else {
      role(x, "single"); role(y, "single"); role(c, "single"); role(s, "single");
    }
    const abi::RopeArgs args{spm(y), spm(x), spm(c), spm(s), tokens, heads, d,
                            std::uint32_t(rotary), table_rows,
                            nonnegative(r.integer("position_offset", 0), "position_offset")};
    const bool vision = tp8 && tokens == 868 && heads == 2 && d == 64 && rotary == 64;
    out.plan = tp8 && !vision ? abi::partial_rope_tp8(args) : abi::rope(args);
    if (vision) out.plan.cores = 8;
    finish(r, out, {"out"}); return true;
  }

  if (op == "qk_rmsnorm_mrope_m16d256_f16" || op == "qk_rmsnorm_mrope_m32d256_f16") {
    const unsigned rows = op == "qk_rmsnorm_mrope_m16d256_f16" ? 16 : 32;
    const auto& y = local(r, "out", {rows, 256});
    const auto& x = local(r, "x", {rows, 256}); same_role(y, x);
    // A local D256 vector is one complete head even when heads are distributed.
    if (x.distribution == "sharded") require(x.shard_axis == 1, "Action QK shards complete heads along axis1");
    const auto& w = local(r, "weight", {256});
    require(w.distribution == "local" || w.distribution == "replicated", "QK weight must be local or replicated");
    require(w.shard_axis == -1, "QK weight must be unsharded");
    replicated_parameter(x, w, "weight");
    const auto& c = r.tensor("cosine");
    require(c.shape.size() == 2, "cosine requires [table_rows,32]");
    const auto n = dim(c.shape[0], "table_rows"); table(r, "cosine", {n, 32});
    const auto& s = table(r, "sine", {n, 32});
    out.plan = abi::qk_rmsnorm_mrope({spm(y), spm(x), spm(w), ddr(c), ddr(s), n,
        nonnegative(r.integer("position_offset", 0), "position_offset"), scalar(r, "epsilon", 1.0e-6), rows});
    finish(r, out, {"out"}); return true;
  }

  if (op == "flash_attention_spm_d384" || op == "flash_attention_varlen_f16") {
    const auto image_lengths = lengths(r);
    const auto rows = attention::packed_rows(image_lengths);
    const auto& q = local(r, "q", {rows, 128}); role(q, "sharded", 1);
    const auto& y = local(r, "out", {rows, 128}); role(y, "sharded", 1);
    if (op == "flash_attention_spm_d384") {
      const auto& k = local(r, "k", {rows, 128}); role(k, "sharded", 1);
      const auto& v = local(r, "v", {rows, 128}); role(v, "sharded", 1);
      out.plan = abi::flash_attention_spm_d384({spm(y), spm(q), spm(k), spm(v), image_lengths});
    } else {
      std::vector<abi::AttentionCache> caches;
      for (unsigned i = 0; i < image_lengths.size(); ++i) {
        const auto& k = r.tensor("key_cache" + std::to_string(i));
        const auto& v = r.tensor("value_cache" + std::to_string(i));
        require(k.shape.size() == 3, "Vision cache requires [8,capacity,128]");
        const auto capacity = dim(k.shape[1], "capacity");
        shape(k, {8, capacity, 128}); shape(v, {8, capacity, 128}); fp16(k); fp16(v);
        layout(k, "vision_k_cache"); layout(v, "vision_v_cache"); role(k, "shared"); role(v, "shared");
        require(k.bytes >= attention::cache_elements(capacity) * 2 && v.bytes >= attention::cache_elements(capacity) * 2,
                "Vision cache physical padding capacity is insufficient");
        require(k.state_kind == "cache" && v.state_kind == "cache", "attention requires persistent cache owners");
        require(!abi::detail::ddr_overlap(ddr(k), k.bytes, ddr(v), v.bytes), "key and value cache allocations overlap");
        caches.push_back({ddr(k), ddr(v), capacity});
        // Each image owns a separate cache beginning at token zero. The device
        // loads whole 16-token tiles but masks invalid K and zeroes padded V
        // before softmax/PV; only the logical prefix contributes to outputs.
        out.state_accesses.push_back({"key_cache" + std::to_string(i), 0, image_lengths[i], false});
        out.state_accesses.push_back({"value_cache" + std::to_string(i), 0, image_lengths[i], false});
      }
      out.plan = abi::flash_attention({spm(y), spm(q), image_lengths, caches});
    }
    finish(r, out, {"out"}); return true;
  }

  return false;
}
bool lower_tensor_validation(const Request& r, Lowered& out) {
  using namespace tensor_validation;
  const auto& op = r.op;
  if (op == "causal_sdpa_d256_f16" || op == "causal_sdpa_prefix_spm_d256_f16" ||
      op == "sdpa_masked_m16d256_f16" || op == "sdpa_masked_m32d256_f16") {
    const auto& query = tensor(r, "query");
    const auto m = rows(query, "query");
    shape(query, {m, 256}, "query");
    distribution(query, "sharded", "query");
    check_tensor(query.shard_axis == 1, "query must shard along head dimension axis 1");
    const auto& dst = matrix(r, "out", m, 256, "sharded");
    const auto length = integer(r, "kv_length", m);
    if (op == "causal_sdpa_prefix_spm_d256_f16") {
      const auto& key = matrix(r, "key", m, 256, "sharded");
      const auto& value = tensor(r, "value");
      shape(value, {256, m}, "value");
      distribution(value, "sharded", "value");
      check_tensor(value.shard_axis == 0, "transposed value must shard dimension axis 0");
      return done(r, out, abi::causal_sdpa_prefix_spm({address(dst), address(query),
          address(key), address(value), m, length}));
    }
    const auto& key = cache(r, "key_cache", false);
    const auto& value = cache(r, "value_cache", true);
    out.state_accesses.push_back({"key_cache", 0, length, false});
    out.state_accesses.push_back({"value_cache", 0, length, false});
    if (op == "causal_sdpa_d256_f16")
      return done(r, out, abi::causal_sdpa({address(dst), address(query),
          key.address, value.address, m, length}));
    check_tensor(m == (op == "sdpa_masked_m16d256_f16" ? 16U : 32U),
            "masked SDPA rows do not match the selected entry");
    const auto& mask = matrix(r, "mask", m, length, "replicated");
    return done(r, out, abi::masked_sdpa({address(dst), address(query),
        address(mask), key.address, value.address, length, m}));
  }
  return false;
}
bool lower_state_validation(const Request& r, Lowered& result) {
  using namespace state_validation;
  const auto& op = r.op;
  if (op == "kcache_insert_f16" || op == "vcache_insert_f16") {
    const auto& input = data(r, "input");
    const auto capacity = integer(r, "capacity", -1, 4096);
    need(input.distribution == "sharded" && input.shard_axis == 1,
            "cache input must declare per-core channel shards");
    need(capacity > 0 && capacity % 16 == 0, "vision cache capacity must be positive and 16-aligned");
    const bool value = op == "vcache_insert_f16";
    const auto& target = cache(r, "cache", value, false, capacity);
    const auto m = rows(input, 128, "input");
    const auto position = integer(r, "position", -1, 4096);
    result.plan = abi::cache_insert({local(input), m, position, capacity, target.address, value});
    result.state_accesses = {{"cache", position, position + m, true}};
    finish_state(r, result); return true;
  }
  if (op == "kcache_insert_d256_f16" || op == "vcache_insert_d256_f16" ||
             op == "vcache_insert_staged_d256_f16") {
    const auto& input = data(r, "input");
    const auto m = rows(input, 256, "input");
    need(input.distribution == "sharded" && input.shard_axis == 1,
            "cache input must declare per-core channel shards");
    const bool value = op != "kcache_insert_d256_f16";
    const auto& target = cache(r, "cache", value, true, 784);
    const auto position = integer(r, "position", -1, 780);
    if (op == "vcache_insert_staged_d256_f16") {
      const auto& staged = data(r, "value_spm");
      expect_shape(staged, {256, m}, "value_spm");
      need(staged.distribution == input.distribution && staged.shard_axis == 0,
              "staged transposed value must declare channel shard axis 0");
      result.plan = abi::causal_value_cache_insert_staged({local(input), local(staged),
                                                         target.address, m, position});
    } else {
      result.plan = abi::causal_cache_insert({local(input), m, position, target.address, value});
    }
    result.state_accesses = {{"cache", position, position + m, true}};
    finish_state(r, result); return true;
  }
  if (op == "copy_physical_prefix_f16") {
    abi::CopyPhysicalPrefixF16Args args{};
    args.prefix_len = integer(r, "prefix_len", -1, 512);
    args.bucket_len = integer(r, "bucket_len", -1, 512);
    args.capacity_tokens = integer(r, "capacity", -1, 65520);
    need(args.capacity_tokens > 0 && args.capacity_tokens % 16 == 0,
            "prefix cache capacity must be positive and 16-aligned");
    for (unsigned layer = 0; layer < 6; ++layer) {
      const auto suffix = std::to_string(layer);
      auto address = [&](const std::string& name, bool value) {
        return cache(r, name + suffix, value, true, args.capacity_tokens).address;
      };
      args.layers[layer] = {address("src_k", false), address("src_v", true),
                            address("dst_k", false), address("dst_v", true)};
      // The last exact-prefix tile may contain logically invisible padding;
      // only prefix_len source tokens and bucket_len output tokens are valid.
      for (const auto* kind : {"k", "v"}) {
        result.state_accesses.push_back({std::string("src_") + kind + suffix, 0, args.prefix_len, false});
        result.state_accesses.push_back({std::string("dst_") + kind + suffix, 0, args.bucket_len, true});
      }
    }
    result.plan = abi::copy_physical_prefix_f16(args);
    finish_state(r, result); return true;
  }
  return false;
}
}  // namespace

bool lower_attention(const Request& r, Lowered& out) {
  return lower_matrix_validation(r, out) ||
         lower_tensor_validation(r, out) ||
         lower_state_validation(r, out);
}
}  // namespace xinfer::rpu::tp::detail
