// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/tp_contract.hpp"
#include "common/rpu/gdr_prefill_abi.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

namespace tp = xinfer::rpu::tp;
namespace abi = xinfer::rpu::abi;
namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
tp::Request fresh(tp::Request r) {
  r.used_tensors.clear(); r.used_integers.clear(); r.used_scalars.clear();
  r.value_requirements.clear();
  return r;
}
struct Case {
  explicit Case(std::string op) { request.op = std::move(op); }
  tp::Request request;
  std::uint64_t spm_at = 0x10000, ddr_at = 0x100000000ULL, identity = 1;
  void add(const std::string& name, std::vector<std::int64_t> shape,
           const std::string& distribution = "local",
           const std::string& storage = "spm",
           const std::string& layout = "row_major",
           const std::string& dtype = "float16") {
    tp::Tensor t;
    t.shape = std::move(shape); t.distribution = distribution;
    t.storage = storage; t.layout = layout; t.dtype = dtype;
    t.bytes = tp::elements(t) * (dtype == "int32" ? 4 : 2);
    t.identity = identity++;
    t.shard_axis = distribution == "sharded" ? (t.shape.size() == 1 ? 0 : 1) : -1;
    t.values_known = true; t.finite = true; t.minimum = 0; t.maximum = 0;
    auto& at = storage == "spm" ? spm_at : ddr_at;
    t.address = at; at += (t.bytes + 255) & ~std::uint64_t{255};
    request.tensors.emplace(name, t);
  }
  void weight(std::int64_t n, std::int64_t k) {
    add("weight", {8, n, k}, "shared", "ddr", "tp8_nk16");
    request.tensors.at("weight").immutable = true;
  }
  void cache(const char* name, bool value) {
    add(name, {8, 784, 256}, "shared", "ddr",
        value ? "causal_v_cache" : "causal_k_cache");
    request.tensors.at(name).state_kind = "cache";
    request.tensors.at(name).generation = 3;
  }
};
std::vector<Case> make_cases() {
  std::vector<Case> c;
  for (const auto* op : {"gelu_f16", "unary_f16"}) {
    Case x(op); x.add("src", {32, 64}); x.add("out", {32, 64}); c.push_back(x);
  }
  for (const auto* op : {"binary_add_f16", "add_vpb_f16"}) {
    Case x(op); x.add("lhs", {32, 64}); x.add("rhs", {32, 64});
    x.add("out", {32, 64}); c.push_back(x);
  }
  for (const auto* op : {"bcast_rows_f16", "bcast_tiles_f16"}) {
    Case x(op); const int cols = x.request.op == "bcast_rows_f16" ? 64 : 4096;
    x.add("src", {16, cols}); x.add("out", {16, cols}); x.add("a", {16});
    c.push_back(x);
  }
  { Case x("cumsum_fp16"); x.add("src", {2, 64}); x.add("out", {2, 64}); c.push_back(x); }
  { Case x("unit_tril_inv_fp16"); x.add("src", {2, 64, 64});
    x.add("out", {2, 64, 64}); c.push_back(x); }
  for (const auto m : {16, 32}) {
    Case x("gemm_ddr_row_m" + std::to_string(m) + "n1024k256_acc32");
    x.add("src", {m, 256}, "sharded"); x.add("out", {m, 1024}, "partial");
    x.weight(1024, 256); c.push_back(x);
    Case residual("residual_gate_m" + std::to_string(m) + "h1024_f16");
    for (const auto* name : {"out", "branch", "residual"})
      residual.add(name, {m, 1024}, "replicated");
    residual.add("gate", {1024}, "replicated"); c.push_back(residual);
    Case euler("euler_update_m" + std::to_string(m) + "d64_f16");
    for (const auto* name : {"action", "velocity", "padding_velocity"})
      euler.add(name, {m, 64}, "replicated");
    euler.add("keep_mask", {64}, "replicated"); euler.request.scalars["delta_t"] = 0.125;
    c.push_back(euler);
  }
  for (const auto* op : {"qkv_qgate_n1024k2048", "qkv_qgate_m512", "qkv_zba_n1056k2048",
       "qkv_zba_m512", "gateup_n1536k2048", "gateup_m512", "o_n2048k256",
       "down_n2048k768", "tp_qkv_compact_f16"}) {
    Case x(op); const auto& name = x.request.op;
    const bool partial = name == "o_n2048k256" || name == "down_n2048k768";
    const bool zba = name.find("zba") != std::string::npos;
    const bool gateup = name.find("gateup") != std::string::npos;
    const bool compact = name == "tp_qkv_compact_f16";
    const unsigned m = name.find("m512") != std::string::npos ? 128 : 64;
    const unsigned k = name == "o_n2048k256" ? 256 : name == "down_n2048k768" ? 768 : 2048;
    const unsigned n = partial ? 2048 : zba ? 1056 : gateup ? 1536 : compact ? 640 : 1024;
    x.add("src", {m, k}, partial ? "sharded" : "replicated");
    x.weight(n, k);
    if (partial || compact) x.add("out", {m, n}, partial ? "partial" : "local");
    else if (gateup) {
      x.add("out_gate", {m, 768}, "sharded"); x.add("out_up", {m, 768}, "sharded");
    } else if (zba) {
      for (const auto* port : {"out_q", "out_k", "out_v", "out_z"})
        x.add(port, {m, 256}, "sharded");
      x.add("out_b", {m, 16}, "sharded"); x.add("out_a", {m, 16}, "sharded");
    } else {
      x.add("out_q", {m, 256}, "sharded"); x.add("out_qgate", {m, 256}, "sharded");
      x.add("out_k", {m, 256}); x.add("out_v", {m, 256});
    }
    c.push_back(x);
  }
  for (const auto* op : {"qkv_bias_n384k1024", "o_n1024k128", "down_n1024k512", "down_n2048k512"}) {
    Case x(op); const auto& name = x.request.op;
    const bool qkv = name == "qkv_bias_n384k1024";
    const unsigned n = qkv ? 384 : name == "down_n2048k512" ? 2048 : 1024;
    const unsigned k = qkv ? 1024 : name == "o_n1024k128" ? 128 : 512;
    x.add("src", {16, k}, qkv ? "replicated" : "sharded");
    x.weight(n, k); x.add("bias", {n}, qkv ? "sharded" : "owner0");
    if (qkv) for (const auto* port : {"out_q", "out_k", "out_v"})
      x.add(port, {16, 128}, "sharded");
    else x.add("out", {16, n}, "partial");
    c.push_back(x);
  }
  for (const auto* op : {"causal_sdpa_d256_f16", "causal_sdpa_prefix_spm_d256_f16",
       "sdpa_masked_m16d256_f16", "sdpa_masked_m32d256_f16"}) {
    Case x(op); const auto& name = x.request.op;
    const bool masked = name.find("masked") != std::string::npos;
    const unsigned m = masked ? (name.find("m16") != std::string::npos ? 16 : 32) : 64;
    x.add("query", {m, 256}, "sharded"); x.add("out", {m, 256}, "sharded");
    if (name == "causal_sdpa_prefix_spm_d256_f16") {
      x.add("key", {m, 256}, "sharded"); x.add("value", {256, m}, "sharded");
      x.request.tensors.at("value").shard_axis = 0;
    } else { x.cache("key_cache", false); x.cache("value_cache", true); }
    if (masked) {
      x.add("mask", {m, m + 64}, "replicated"); x.request.integers["kv_length"] = m + 64;
    }
    c.push_back(x);
  }
  { Case x("all_reduce_residual_layernorm");
    x.add("partial", {128, 1024}, "partial"); x.add("residual", {128, 1024}, "replicated");
    x.add("x", {128, 1024}, "sequence_shard"); x.request.tensors.at("x").shard_axis = 0;
    x.add("norm", {128, 1024}, "replicated");
    x.add("gamma", {1024}, "replicated"); x.add("beta", {1024}, "replicated");
    c.push_back(x);
  }
  for (const auto m : {16, 32}) {
    Case x("down_allreduce_rmsnorm_m" + std::to_string(m) + "n1024k256_acc32");
    x.add("src", {m, 256}, "sharded"); x.add("partial", {m, 1024}, "partial");
    for (const auto* port : {"residual", "stream", "norm"}) x.add(port, {m, 1024}, "replicated");
    x.weight(1024, 256);
    x.add("gate_row", {3, 1024}, "shared", "ddr");
    x.add("next_row", {3, 1024}, "shared", "ddr"); c.push_back(x);
  }
  { Case x("gdr_prefill_f16");
    for (const auto* port : {"q", "k", "v", "z", "out"}) x.add(port, {64, 256}, "sharded");
    x.add("a", {64, 16}, "sharded"); x.add("b", {64, 16}, "sharded");
    x.add("dt", {16}, "sharded"); x.add("neg_a", {16}, "sharded");
    x.add("norm_weight", {128}, "replicated"); x.add("state", {2, 128, 128}, "sharded");
    auto& state = x.request.tensors.at("state"); state.shard_axis = 0; state.state_kind = "state";
    x.add("scratch", {static_cast<std::int64_t>(xinfer::rpu::fusion::gdr_scratch_bytes_for_rows(64) / 2)});
    x.request.integers["initial"] = 1; x.request.integers["valid_rows"] = 64; c.push_back(x);
  }
  { Case x("gather_embedding_tp8_f16");
    x.add("ids", {64}, "shared", "ddr", "row_major", "int32");
    x.add("vocab", {256277, 2048}, "shared", "ddr");
    x.request.tensors.at("vocab").immutable = true;
    x.add("out", {8, 2048}, "sharded"); x.request.tensors.at("out").shard_axis = 0;
    c.push_back(x);
  }
  return c;
}
void rejects(tp::Request r) {
  r = fresh(std::move(r));
  try { tp::Lowered lowered; tp::lower_operator(r, lowered); }
  catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("expected a native contract rejection");
}
}  // namespace

int main() try {
  const auto cases = make_cases();
  check(cases.size() == 36, "TP integration fixture coverage count");
  std::set<std::string> names;
  unsigned rejected = 0;
  for (const auto& c : cases) {
    auto r = fresh(c.request); tp::Lowered lowered;
    check(tp::lower_operator(r, lowered), "known TP integration operation not lowered");
    // Plain denoise catalog name resolves to the hardware kernel entry.
    std::string expected = c.request.op;
    if (expected == "down_allreduce_rmsnorm_m16n1024k256_acc32")
      expected = "down_allreduce_rmsnorm_hw_m16n1024k256_acc32";
    else if (expected == "down_allreduce_rmsnorm_m32n1024k256_acc32")
      expected = "down_allreduce_rmsnorm_hw_m32n1024k256_acc32";
    check(expected == abi::name(lowered.plan.entry), "wrong selected entry");
    check(lowered.plan.cores == 8 && lowered.plan.register_count > 0, "bad TP8 plan");
    check(!lowered.writes.empty(), "write set missing");
    check(names.insert(c.request.op).second, "duplicate TP integration fixture");
    auto invalid = c.request; invalid.integers["unknown_parameter"] = 0;
    rejects(invalid); ++rejected;
    invalid = c.request; invalid.tensors.begin()->second.dtype = "float32";
    rejects(invalid); ++rejected;
    invalid = c.request; invalid.tensors.at(lowered.writes.front()).immutable = true;
    rejects(invalid); ++rejected;
    invalid = c.request; invalid.tensors.begin()->second.bytes = 1;
    rejects(invalid); ++rejected;
    if (c.request.op == "gemm_ddr_row_m16n1024k256_acc32") {
      check(lowered.plan.regs[4] == (c.request.tensors.at("out").address & 0xffff),
            "Action row output register offset mismatch");
      invalid = c.request; invalid.tensors.at("out").distribution = "replicated";
      rejects(invalid); ++rejected;
      invalid = c.request; invalid.tensors.at("src").shard_axis = 0;
      rejects(invalid); ++rejected;
      invalid = c.request; invalid.tensors.at("weight").layout = "row_major";
      rejects(invalid); ++rejected;
    }
    if (c.request.op == "unary_f16" ||
        c.request.op == "bcast_rows_f16" || c.request.op == "bcast_tiles_f16") {
      const std::int64_t first_invalid = c.request.op == "unary_f16" ? 4 : 3;
      for (const auto value : {first_invalid, std::int64_t{65536}, std::int64_t{65537}}) {
        invalid = c.request; invalid.integers["operation"] = value;
        rejects(invalid); ++rejected;
      }
    }
    if (c.request.op == "bcast_rows_f16" || c.request.op == "bcast_tiles_f16") {
      auto broadcast = [&](const char* src_role, const char* a_role,
                           unsigned operation, unsigned x_first) {
        auto request = fresh(c.request);
        request.tensors.at("src").distribution = src_role;
        request.tensors.at("out").distribution = src_role;
        request.tensors.at("a").distribution = a_role;
        request.integers["operation"] = operation;
        request.integers["x_first"] = x_first;
        return request;
      };
      for (unsigned order : {0U, 1U}) {
        for (unsigned operation : {0U, 1U}) {
          for (const auto* a_role : {"partial", "owner0", "single"}) {
            auto valid = broadcast("partial", a_role, operation, order);
            tp::Lowered plan;
            check(tp::lower_operator(valid, plan), "linear broadcast partial combination rejected");
          }
          for (const auto* a_role : {"replicated", "local"}) {
            rejects(broadcast("partial", a_role, operation, order)); ++rejected;
          }
        }
        auto scaled = broadcast("partial", "replicated", 2, order);
        tp::Lowered plan;
        check(tp::lower_operator(scaled, plan), "common multiplier of partials was rejected");
        for (const auto* a_role : {"partial", "local", "owner0", "single"}) {
          rejects(broadcast("partial", a_role, 2, order)); ++rejected;
        }
      }
      for (unsigned operation : {0U, 1U, 2U}) {
        rejects(broadcast("replicated", "local", operation, 0)); ++rejected;
        rejects(broadcast("local", "partial", operation, 0)); ++rejected;
        for (const auto* role : {"local", "replicated"}) {
          auto valid = broadcast(role, role, operation, 0); tp::Lowered plan;
          check(tp::lower_operator(valid, plan), "resolved broadcast combination rejected");
        }
        auto sharded = broadcast("sharded", "replicated", operation, 0);
        sharded.tensors.at("src").shard_axis = 1;
        sharded.tensors.at("out").shard_axis = 1;
        tp::Lowered plan;
        check(tp::lower_operator(sharded, plan), "broadcast common row parameter over column shards rejected");
      }
      for (const auto* role : {"owner0", "single"}) {
        for (unsigned operation : {0U, 1U, 2U}) for (unsigned order : {0U, 1U}) {
          for (const auto* a_role : {"owner0", "single"}) {
            auto owned = broadcast(role, a_role, operation, order); tp::Lowered plan;
            check(tp::lower_operator(owned, plan), "core0-owned broadcast operands were rejected");
          }
          for (const auto* a_role : {"local", "replicated"}) {
            auto common = broadcast(role, a_role, operation, order);
            common.tensors.at("a").minimum = 1;
            common.tensors.at("a").maximum = 1;
            if (operation != 2) { rejects(common); ++rejected; }
            else {
              tp::Lowered plan;
              check(tp::lower_operator(common, plan), "finite common multiplier of owner0/single rejected");
              check(common.value_requirements.size() == 1,
                    "owned broadcast multiplier lost its replay finite-value guard");
            }
            common = fresh(common);
            common.tensors.at("a").minimum = 0;
            common.tensors.at("a").maximum = 0;
            tp::Lowered zero;
            check(tp::lower_operator(common, zero), "proven zero broadcast parameter rejected");
            common.tensors.at("a").values_known = false;
            rejects(common); ++rejected;
          }
        }
        auto nonfinite = broadcast(role, "replicated", 2, 0);
        nonfinite.tensors.at("a").finite = false;
        rejects(nonfinite); ++rejected;
      }
    }
    if (c.request.op == "gelu_f16") {
      for (const std::int64_t value : {2, 65536}) {
        invalid = c.request; invalid.integers["mode"] = value;
        rejects(invalid); ++rejected;
      }
    }
    if (c.request.op == "gelu_f16" || c.request.op == "unary_f16") {
      for (const auto* role : {"owner0", "single"}) {
        auto owned = fresh(c.request);
        owned.tensors.at("src").distribution = role;
        owned.tensors.at("out").distribution = role;
        if (c.request.op == "unary_f16") {
          for (unsigned operation : {0U, 1U, 3U}) {
            invalid = owned; invalid.integers["operation"] = operation;
            rejects(invalid); ++rejected;
          }
          owned.integers["operation"] = 2; tp::Lowered plan;
          check(tp::lower_operator(owned, plan), "zero-preserving owned SiLU was rejected");
        } else {
          for (unsigned mode : {0U, 1U}) {
            auto valid = owned; valid.integers["mode"] = mode; tp::Lowered plan;
            check(tp::lower_operator(valid, plan), "zero-preserving owned GELU was rejected");
          }
        }
      }
    }
    if (c.request.op == "cumsum_fp16" || c.request.op == "unit_tril_inv_fp16") {
      for (const auto* role : {"sharded", "sequence_shard"}) {
        auto sharded = fresh(c.request);
        for (const auto* port : {"src", "out"}) {
          sharded.tensors.at(port).distribution = role;
          sharded.tensors.at(port).shard_axis = 0;
        }
        tp::Lowered plan;
        check(tp::lower_operator(sharded, plan), "independent row/batch sharding was rejected");
        for (unsigned axis = 1; axis < sharded.tensors.at("src").shape.size(); ++axis) {
          invalid = fresh(sharded);
          invalid.tensors.at("src").shard_axis = axis;
          invalid.tensors.at("out").shard_axis = axis;
          rejects(invalid); ++rejected;
        }
        invalid = fresh(sharded);
        invalid.tensors.at("out").shard_axis = 1;
        rejects(invalid); ++rejected;
      }
      for (const auto* role : {"owner0", "single"}) {
        auto owned = fresh(c.request);
        owned.tensors.at("src").distribution = role;
        owned.tensors.at("out").distribution = role;
        if (c.request.op == "unit_tril_inv_fp16") { rejects(owned); ++rejected; }
        else {
          tp::Lowered owned_plan;
          check(tp::lower_operator(owned, owned_plan), "zero-preserving owned cumsum was rejected");
        }
      }
    }
    if (c.request.op.find("euler_update_") == 0) {
      invalid = c.request; invalid.tensors.at("keep_mask").distribution = "local";
      rejects(invalid); ++rejected;
      invalid.tensors.at("keep_mask").distribution = "partial";
      rejects(invalid); ++rejected;
      auto independent = fresh(c.request);
      for (auto& port : independent.tensors) port.second.distribution = "local";
      tp::Lowered plan;
      check(tp::lower_operator(independent, plan), "local Euler with local keep_mask was rejected");
      auto sharded = fresh(c.request);
      for (const auto* port : {"action", "velocity", "padding_velocity"}) {
        sharded.tensors.at(port).distribution = "sharded";
        sharded.tensors.at(port).shard_axis = 0;
      }
      tp::Lowered sharded_plan;
      check(tp::lower_operator(sharded, sharded_plan), "row-sharded Euler with replicated mask rejected");
      for (const auto* role : {"owner0", "single"}) {
        for (const auto* mask_role : {"local", "replicated", "sharded"}) {
          auto owned = fresh(c.request);
          for (const auto* port : {"action", "velocity", "padding_velocity"})
            owned.tensors.at(port).distribution = role;
          auto& mask = owned.tensors.at("keep_mask");
          mask.distribution = mask_role;
          mask.shard_axis = std::string(mask_role) == "sharded" ? 0 : -1;
          mask.minimum = -65504; mask.maximum = 65504;
          tp::Lowered finite_plan;
          check(tp::lower_operator(owned, finite_plan), "owned Euler with finite keep_mask was rejected");
          check(owned.value_requirements.size() == 1 &&
                    owned.value_requirements[0].tensor == "keep_mask" &&
                    owned.value_requirements[0].minimum == -65504 &&
                    owned.value_requirements[0].maximum == 65504,
                "owned Euler lost its replay finite-mask guard");
          invalid = fresh(owned); invalid.tensors.at("keep_mask").values_known = false;
          rejects(invalid); ++rejected;
          for (const double nonfinite : {std::numeric_limits<double>::quiet_NaN(),
                                         std::numeric_limits<double>::infinity()}) {
            invalid = fresh(owned);
            auto& bad_mask = invalid.tensors.at("keep_mask");
            bad_mask.finite = false; bad_mask.minimum = bad_mask.maximum = nonfinite;
            rejects(invalid); ++rejected;
          }
        }
        for (const auto* mask_role : {"owner0", "single"}) {
          auto owned = fresh(c.request);
          for (const auto* port : {"action", "velocity", "padding_velocity"})
            owned.tensors.at(port).distribution = role;
          auto& mask = owned.tensors.at("keep_mask");
          mask.distribution = mask_role; mask.values_known = mask.finite = false;
          tp::Lowered owned_plan;
          check(tp::lower_operator(owned, owned_plan), "owned Euler mask requires no whole-tensor proof");
          check(owned.value_requirements.empty(), "owned mask unnecessarily acquired a value guard");
        }
      }
      for (const auto* role : {"local", "replicated"}) {
        auto unchanged = fresh(c.request);
        for (auto& port : unchanged.tensors) port.second.distribution = role;
        auto& mask = unchanged.tensors.at("keep_mask");
        mask.values_known = mask.finite = false;
        tp::Lowered unchanged_plan;
        check(tp::lower_operator(unchanged, unchanged_plan), "ordinary Euler gained an unsupported range restriction");
        check(unchanged.value_requirements.empty(), "ordinary Euler mask unnecessarily acquired a value guard");
      }
    }
    if (c.request.op == "all_reduce_residual_layernorm") {
      check(lowered.collective && lowered.zero_counters == std::vector<unsigned>({128, 130, 228}),
            "collective atomic reset registers missing");
      invalid = c.request; invalid.integers["sync_epoch"] = 48;
      rejects(invalid); ++rejected;
      invalid = c.request; invalid.tensors.at("residual").distribution = "sequence_shard";
      invalid.tensors.at("residual").shard_axis = 1;
      rejects(invalid); ++rejected;
      for (const auto m : {128,868}) {
        Case shard("all_reduce_residual_layernorm");
        shard.add("partial", {m,1024}, "partial");
        shard.add("residual", {m,1024}, "sequence_shard");
        shard.request.tensors.at("residual").shard_axis = 0;
        shard.add("x", {m,1024}, "sequence_shard"); shard.request.tensors.at("x").shard_axis = 0;
        shard.add("norm", {m,1024}, "replicated");
        shard.add("gamma", {1024}, "replicated"); shard.add("beta", {1024}, "replicated");
        auto input = fresh(shard.request); tp::Lowered planned;
        check(tp::lower_operator(input, planned), "Vision sequence-shard residual was rejected");
        check(planned.plan.entry == lowered.plan.entry, "Vision residual layout changed the kernel entry");
      }
    }
    if (c.request.op == "down_allreduce_rmsnorm_m16n1024k256_acc32") {
      invalid = c.request; invalid.integers["global_counter"] = 50;
      rejects(invalid); ++rejected;
    }
    if (c.request.op.find("down_allreduce_rmsnorm_") == 0) {
      // A w_out without next2_row is not passed to the kernel and must not
      // silently become a declared output or consume an unchecked tensor.
      invalid = c.request;
      invalid.tensors["w_out"] = invalid.tensors.at("norm");
      rejects(invalid); ++rejected;
      auto extended = c;
      extended.add("next2_row", {3, 1024}, "shared", "ddr");
      extended.add("w_out", {c.request.tensors.at("src").shape[0], 1024}, "replicated");
      auto enabled = fresh(extended.request); tp::Lowered plan;
      check(tp::lower_operator(enabled, plan), "next2_row/w_out pair was rejected");
      check(std::find(plan.writes.begin(), plan.writes.end(), "w_out") != plan.writes.end(),
            "enabled next2 output missing from write set");
    }
    if (c.request.op == "gdr_prefill_f16") {
      for (const auto* name : {"raw_out", "norm_out"}) {
        invalid = c.request;
        invalid.tensors[name] = invalid.tensors.at("out");
        rejects(invalid); ++rejected;
      }
      auto debug = c;
      debug.add("raw_out", {64, 256}, "sharded");
      debug.add("norm_out", {64, 256}, "sharded");
      debug.request.integers["debug"] = 1;
      auto enabled = fresh(debug.request); tp::Lowered plan;
      check(tp::lower_operator(enabled, plan), "enabled GDR debug taps were rejected");
      for (const auto* name : {"raw_out", "norm_out"})
        check(std::find(plan.writes.begin(), plan.writes.end(), name) != plan.writes.end(),
              "enabled GDR debug tap missing from write set");
      // Scratch is private workspace populated from different head shards;
      // its contents cannot carry replicated or zero-nonowner identities.
      for (const auto* role : {"replicated", "owner0", "single", "partial", "sharded", "sequence_shard"}) {
        invalid = c.request;
        auto& scratch = invalid.tensors.at("scratch");
        scratch.distribution = role;
        scratch.shard_axis = (std::string(role) == "sharded" || std::string(role) == "sequence_shard") ? 0 : -1;
        rejects(invalid); ++rejected;
      }
    }
    if (c.request.op == "gelu_f16") {
      invalid = c.request;
      invalid.tensors.at("src").distribution = "sharded";
      invalid.tensors.at("out").distribution = "sharded";
      invalid.tensors.at("src").shard_axis = 0;
      invalid.tensors.at("out").shard_axis = 1;
      rejects(invalid); ++rejected;
      invalid.tensors.at("src").shard_axis = -1;
      invalid.tensors.at("out").shard_axis = -1;
      rejects(invalid); ++rejected;
    }
    if (c.request.op == "gdr_prefill_f16") {
      check(lowered.zero_counters == std::vector<unsigned>({128, 130}), "GDR replay counter resets");
      check(lowered.state_accesses.size() == 1 && lowered.state_accesses[0].write,
            "initial GDR must initialize state without reading stale contents");
      invalid = c.request; invalid.integers.erase("initial"); rejects(invalid); ++rejected;
      invalid = c.request; invalid.tensors.at("state").state_kind = "none";
      rejects(invalid); ++rejected;
      invalid = c.request; invalid.integers["valid_rows"] = 65; rejects(invalid); ++rejected;
    }
    if (c.request.op == "causal_sdpa_d256_f16")
      check(lowered.state_accesses.size() == 2 && !lowered.state_accesses[0].write &&
            lowered.state_accesses[0].end == 64, "attention cache read interval");
    if (c.request.op == "gather_embedding_tp8_f16") {
      invalid = c.request; invalid.tensors.at("vocab").immutable = false;
      rejects(invalid); ++rejected;
      invalid = c.request; invalid.tensors.at("vocab").distribution = "local";
      rejects(invalid); ++rejected;
      invalid = c.request; invalid.tensors.at("ids").maximum = 256277;
      rejects(invalid); ++rejected;
      invalid = c.request; invalid.tensors.at("ids").values_known = false;
      rejects(invalid); ++rejected;
    }
  }
  tp::Request unknown; unknown.op = "unknown"; tp::Lowered lowered;
  check(!tp::lower_operator(unknown, lowered), "unknown operation must be delegated");
  std::cout << "TP integration contracts: " << names.size() << " entries, " << rejected
            << " negative cases passed; no board or numerical claim\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "tp_integration_contract_test: " << e.what() << '\n';
  return 1;
}
