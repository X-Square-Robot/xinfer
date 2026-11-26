// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/tp_contract.hpp"
#include "common/rpu/tp_cache_layout.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tp = xinfer::rpu::tp;
namespace abi = xinfer::rpu::abi;
namespace {
void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct Inputs {
  std::uint64_t spm = 256, ddr = 0x100000000ULL, identity = 1;
  tp::Tensor tensor(std::initializer_list<std::int64_t> shape,
                    std::string distribution = "local", std::string storage = "spm",
                    std::string layout = "row_major", std::string state = "none") {
    tp::Tensor t;
    t.shape = shape; t.distribution = std::move(distribution);
    if (t.distribution == "sharded") t.shard_axis = t.shape.size() - 1;
    t.storage = storage; t.layout = std::move(layout); t.state_kind = std::move(state);
    t.bytes = 2 * tp::elements(t); t.identity = identity++;
    auto& next = storage == "spm" ? spm : ddr;
    t.address = next; next += (t.bytes + 255) & ~std::uint64_t(255);
    return t;
  }
};
tp::Request make(const std::string& op) {
  Inputs a;
  tp::Request r; r.op = op;
  if (op == "all_reduce_residual") {
    r.tensors["input"] = a.tensor({16, 1024}, "partial");
    r.tensors["out"] = a.tensor({16, 1024}, "replicated");
  } else if (op == "all_gather_f16") {
    r.tensors["input"] = a.tensor({16, 128}, "sharded");
    r.tensors["out"] = a.tensor({16, 1024}, "replicated");
  } else if (op == "qkv_expand_kv_f16") {
    r.tensors["input"] = a.tensor({64, 640});
    r.tensors["out"] = a.tensor({64, 1024});
  } else if (op == "fill_f16") {
    r.tensors["out"] = a.tensor({17}); r.scalars["value"] = -0.5;
  } else if (op == "binary_sub_f16" || op == "binary_mul_f16" ||
             op == "binary_max_f16") {
    for (const auto* key : {"out", "lhs", "rhs"}) r.tensors[key] = a.tensor({17});
  } else if (op == "scalar_mul_f16") {
    r.tensors["out"] = a.tensor({17}); r.tensors["input"] = a.tensor({17});
    r.scalars["value"] = -1;
  } else if (op == "slice_last_f16") {
    r.tensors["out"] = a.tensor({64, 2}); r.tensors["input"] = a.tensor({64, 16});
  } else if (op == "transpose_nbc_c128_f16" || op == "transpose_nbc_c1_f16") {
    const auto c = op == "transpose_nbc_c128_f16" ? 128 : 1;
    r.tensors["out"] = a.tensor({2, 64, c}); r.tensors["input"] = a.tensor({64, 2, c});
  } else if (op == "l2norm_f16") {
    r.tensors["out"] = a.tensor({16, 128}); r.tensors["input"] = a.tensor({16, 128});
  } else if (op.find("gemm_ddr_bias_scale_") == 0) {
    const auto m = op.find("_m16") != std::string::npos ? 16 : 32;
    r.tensors["out"] = a.tensor({m, 1024}, "single");
    r.tensors["input"] = a.tensor({m, 64}, "single");
    r.tensors["bias"] = a.tensor({1024}, "single");
    r.tensors["weight"] = a.tensor({1024, 64}, "shared", "ddr", "nk16");
  } else if (op == "gemm_ddr_bias_m16n64k1024_acc32" ||
             op == "gemm_ddr_bias_m32n64k1024_acc32") {
    const auto m = op.find("_m16") != std::string::npos ? 16 : 32;
    r.tensors["out"] = a.tensor({m, 64}, "single");
    r.tensors["input"] = a.tensor({m, 1024}, "replicated");
    r.tensors["bias"] = a.tensor({64}, "single");
    r.tensors["weight"] = a.tensor({64, 1024}, "shared", "ddr", "nk16");
    r.tensors["weight"].immutable = true;
  } else if (op.find("cache_insert") != std::string::npos) {
    const bool causal = op.find("d256") != std::string::npos;
    const bool value = op[0] == 'v';
    const auto n = causal ? 256 : 128;
    r.tensors["input"] = a.tensor({64, n}, "sharded");
    r.tensors["cache"] = a.tensor({8, causal ? 784 : 128, n}, "shared", "ddr",
        causal ? (value ? "causal_v_cache" : "causal_k_cache")
               : (value ? "vision_v_cache" : "vision_k_cache"), "cache");
    if (!causal) r.integers["capacity"] = 128;
    r.integers["position"] = 0;
    if (op == "vcache_insert_staged_d256_f16") {
      r.tensors["value_spm"] = a.tensor({256, 64}, "sharded");
      r.tensors["value_spm"].shard_axis = 0;
    }
  } else if (op == "copy_physical_prefix_f16") {
    for (unsigned layer = 0; layer < 6; ++layer)
      for (const std::string key : {"src_k", "src_v", "dst_k", "dst_v"})
        r.tensors[key + std::to_string(layer)] = a.tensor({8, 128, 256}, "shared", "ddr",
            key.back() == 'k' ? "causal_k_cache" : "causal_v_cache", "cache");
    r.integers = {{"prefix_len", 65}, {"bucket_len", 128}, {"capacity", 128}};
  } else if (op == "causal_depthwise_conv1d_prefill_f16") {
    r.tensors["pad"] = a.tensor({67, 256});
    r.tensors["carry"] = a.tensor({3, 256}, "local", "spm", "row_major", "history");
    r.tensors["halo"] = a.tensor({8, 3, 256});
    r.tensors["weight"] = a.tensor({8, 4, 256}, "shared", "ddr");
  } else if (op == "causal_depthwise_conv1d_step_f16") {
    r.tensors["state"] = a.tensor({4, 256}, "local", "spm", "row_major", "history");
    r.tensors["input"] = a.tensor({256});
    r.tensors["weight"] = a.tensor({8, 4, 256}, "shared", "ddr");
  } else if (op == "conv_silu_prefill_f16") {
    for (const std::string path : {"q", "k", "v"}) {
      r.tensors[path + "pad"] = a.tensor({67, 256});
      r.tensors[path + "out"] = a.tensor({64, 256});
      r.tensors[path + "weight"] = a.tensor({8, 4, 256}, "shared", "ddr");
    }
    for (const auto* name : {"history", "history_out"})
      r.tensors[name] = a.tensor({3, 4, 256}, "local", "spm", "row_major", "history");
    r.integers = {{"valid_rows", 1}, {"initial", 0}};
  } else {
    throw std::runtime_error("fixture has no operation");
  }
  return r;
}
void rejects(tp::Request r, const std::function<void(tp::Request&)>& change) {
  r.used_tensors.clear(); r.used_integers.clear(); r.used_scalars.clear();
  change(r);
  try {
    tp::Lowered plan;
    tp::lower_operator(r, plan);
  } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error(r.op + " accepted an invalid request");
}
}  // namespace

int main() try {
  const std::vector<std::string> names{
      "all_reduce_residual", "all_gather_f16", "qkv_expand_kv_f16", "fill_f16",
      "binary_sub_f16", "binary_mul_f16", "binary_max_f16",
      "scalar_mul_f16", "slice_last_f16", "transpose_nbc_c128_f16",
      "transpose_nbc_c1_f16", "l2norm_f16",
      "gemm_ddr_bias_scale_m16n1024k64_acc32", "gemm_ddr_bias_scale_m32n1024k64_acc32",
      "gemm_ddr_bias_m16n64k1024_acc32", "gemm_ddr_bias_m32n64k1024_acc32",
      "kcache_insert_f16", "vcache_insert_f16", "kcache_insert_d256_f16",
      "vcache_insert_d256_f16", "vcache_insert_staged_d256_f16", "copy_physical_prefix_f16",
      "causal_depthwise_conv1d_prefill_f16", "causal_depthwise_conv1d_step_f16",
      "conv_silu_prefill_f16"};
  unsigned guards = 0;
  for (const auto& name : names) {
    const auto request = make(name);
    tp::Lowered lowered;
    check(tp::lower_operator(request, lowered), "known operation was not recognized");
    check(abi::name(lowered.plan.entry) == name, "lowered kernel differs from request");
    check(!lowered.writes.empty(), "mutable targets were not declared");
    rejects(request, [](auto& r) { r.integers["unsupported"] = 1; }); ++guards;
    rejects(request, [](auto& r) { r.scalars["unsupported"] = 1; }); ++guards;
    rejects(request, [](auto& r) { r.tensors["unused"] = r.tensors.begin()->second; }); ++guards;
    rejects(request, [](auto& r) { r.tensors.begin()->second.dtype = "float32"; }); ++guards;
    rejects(request, [](auto& r) { r.tensors.begin()->second.bytes = 1; }); ++guards;
    rejects(request, [&lowered](auto& r) { r.tensors.at(lowered.writes.front()).immutable = true; }); ++guards;
  }
  auto reduce = make("all_reduce_residual");
  for (const auto* name : {"gemm_ddr_bias_m16n64k1024_acc32", "gemm_ddr_bias_m32n64k1024_acc32"}) {
    auto projection = make(name);
    tp::Lowered output;
    check(tp::lower_operator(projection, output), "output projection was not recognized");
    check(output.plan.cores == 1 && !output.collective &&
              output.writes == std::vector<std::string>{"out"},
          "output projection must be core0-only with no implicit reduction");
    auto single_input = projection;
    single_input.tensors.at("input").distribution = "single";
    check(tp::lower_operator(single_input, output), "complete core0 input was rejected");
    for (const auto* role : {"partial", "sharded", "local", "owner0", "sequence_shard"}) {
      rejects(projection, [role](auto& r) { r.tensors.at("input").distribution = role; }); ++guards;
    }
    rejects(projection, [](auto& r) { r.tensors.at("input").shard_axis = 1; }); ++guards;
    rejects(projection, [](auto& r) { r.tensors.at("input").shape[1] = 256; }); ++guards;
    rejects(projection, [](auto& r) { r.tensors.at("out").distribution = "replicated"; }); ++guards;
    rejects(projection, [](auto& r) { r.tensors.at("weight").immutable = false; }); ++guards;
    rejects(projection, [](auto& r) { r.tensors.at("weight").layout = "tp8_nk16"; }); ++guards;
    rejects(projection, [](auto& r) { r.tensors.at("bias").distribution = "replicated"; }); ++guards;
    rejects(projection, [](auto& r) {
      r.tensors.at("out").address = r.tensors.at("input").address;
    }); ++guards;
  }
  tp::Lowered p; tp::lower_operator(reduce, p);
  check(p.collective, "all-reduce must be collective");
  check(p.zero_counters == std::vector<unsigned>({128, 130, 228}),
        "wrong initial all-reduce counter registers");
  for (unsigned epoch = 0; epoch < 48; ++epoch) {
    const auto abi_plan = abi::all_reduce_residual({
        static_cast<std::uint32_t>(reduce.tensors.at("out").address),
        static_cast<std::uint32_t>(reduce.tensors.at("input").address),
        0, 16, 0, 1024, false, epoch});
    check(abi_plan.regs.at(12) == 50 - epoch,
          "wrong internal all-reduce counter epoch");
  }
  rejects(reduce, [](auto& r) { r.integers["sync_epoch"] = 0; }); ++guards;
  rejects(reduce, [](auto& r) { r.integers["global_counter"] = 50; }); ++guards;
  rejects(reduce, [](auto& r) { r.tensors["input"].distribution = "replicated"; }); ++guards;
  rejects(reduce, [](auto& r) { r.tensors["out"].distribution = "partial"; }); ++guards;
  rejects(reduce, [](auto& r) { r.integers["with_residual"] = 1; }); ++guards;
  auto residual = make("all_reduce_residual");
  residual.integers["with_residual"] = 1;
  residual.tensors["residual"] = residual.tensors["out"];
  residual.tensors["residual"].address += 2 * residual.tensors["out"].bytes;
  residual.tensors["residual"].identity = 999;
  tp::Lowered rp; check(tp::lower_operator(residual, rp), "residual form not recognized");
  rejects(residual, [](auto& r) { r.tensors["residual"].distribution = "sharded"; }); ++guards;
  for (const auto m : {128,868}) {
    Inputs source;
    tp::Request shard_reduce; shard_reduce.op = "all_reduce_residual";
    shard_reduce.tensors["input"] = source.tensor({m,1024}, "partial");
    shard_reduce.tensors["out"] = source.tensor({m,1024}, "replicated");
    shard_reduce.tensors["residual"] = source.tensor({m,1024}, "sequence_shard");
    shard_reduce.tensors.at("residual").shard_axis = 0;
    shard_reduce.integers["with_residual"] = 1;
    tp::Lowered selected;
    check(tp::lower_operator(shard_reduce, selected), "Vision sequence-shard residual was rejected");
    rejects(shard_reduce, [](auto& r) { r.tensors.at("residual").shard_axis = 1; }); ++guards;
  }
  auto wrong_domain = residual;
  wrong_domain.tensors.at("residual").distribution = "sequence_shard";
  wrong_domain.tensors.at("residual").shard_axis = 0;
  rejects(wrong_domain, [](auto&) {}); ++guards;
  rejects(make("scalar_mul_f16"), [](auto& r) { r.scalars["value"] = 0.5; }); ++guards;
  rejects(make("l2norm_f16"), [](auto& r) { r.scalars["epsilon"] = 1.0e-5; }); ++guards;
  for (const auto* nonlinear : {"binary_mul_f16", "binary_max_f16", "l2norm_f16"}) {
    rejects(make(nonlinear), [](auto& r) {
      for (auto& item : r.tensors) item.second.distribution = "partial";
    }); ++guards;
  }
  for (const auto* linear : {"binary_sub_f16", "scalar_mul_f16"}) {
    auto partial = make(linear);
    for (auto& item : partial.tensors) item.second.distribution = "partial";
    tp::Lowered linear_plan;
    check(tp::lower_operator(partial, linear_plan), "linear partial operation was rejected");
  }
  for (const auto* role : {"owner0", "single"}) {
    auto fill = make("fill_f16"); fill.tensors.at("out").distribution = role;
    for (const auto value : {1.0, -1.0, 0x1p-24}) {
      rejects(fill, [value](auto& r) { r.scalars["value"] = value; }); ++guards;
    }
    for (const auto value : {0.0, -0.0, 1.0e-30}) {
      auto zero = fill; zero.scalars["value"] = value; tp::Lowered plan;
      check(tp::lower_operator(zero, plan), "owner0/single zero fill was rejected");
      check((plan.plan.regs[4] & 0x7fffU) == 0, "nonowner fill must encode FP16 zero");
    }
  }
  for (const auto* role : {"local", "replicated", "sharded", "partial"}) {
    auto fill = make("fill_f16"); fill.tensors.at("out").distribution = role;
    fill.tensors.at("out").shard_axis = std::string(role) == "sharded" ? 0 : -1;
    fill.scalars["value"] = 1.0; tp::Lowered plan;
    check(tp::lower_operator(fill, plan), "ordinary TP8 nonzero fill was rejected");
  }
  rejects(make("kcache_insert_d256_f16"), [](auto& r) { r.integers["position"] = 721; }); ++guards;
  rejects(make("kcache_insert_d256_f16"), [](auto& r) { r.tensors["cache"].state_kind = "none"; }); ++guards;
  rejects(make("kcache_insert_d256_f16"), [](auto& r) { r.tensors["cache"].layout = "causal_v_cache"; }); ++guards;
  rejects(make("vcache_insert_staged_d256_f16"), [](auto& r) { r.integers["position"] = 16; }); ++guards;
  rejects(make("conv_silu_prefill_f16"), [](auto& r) { r.integers.erase("initial"); }); ++guards;
  rejects(make("causal_depthwise_conv1d_step_f16"), [](auto& r) { r.tensors["state"].state_kind = "none"; }); ++guards;
  rejects(make("gemm_ddr_bias_scale_m16n1024k64_acc32"), [](auto& r) { r.tensors["out"].distribution = "replicated"; }); ++guards;
  rejects(make("copy_physical_prefix_f16"), [](auto& r) {
    Inputs a; r.tensors["out"] = a.tensor({17});
  }); ++guards;
  rejects(make("all_gather_f16"), [](auto& r) { r.tensors["input"].shard_axis = 0; }); ++guards;
  rejects(make("qkv_expand_kv_f16"), [](auto& r) { r.tensors["out"].shard_axis = 0; }); ++guards;
  rejects(make("qkv_expand_kv_f16"), [](auto& r) { r.tensors["input"].distribution = "sharded"; }); ++guards;
  rejects(make("vcache_insert_staged_d256_f16"), [](auto& r) { r.tensors["value_spm"].shard_axis = 1; }); ++guards;
  auto insert = make("kcache_insert_d256_f16");
  insert.integers["position"] = 17;
  tp::Lowered ip; tp::lower_operator(insert, ip);
  check(ip.state_accesses.size() == 1 && ip.state_accesses[0].tensor == "cache" &&
        ip.state_accesses[0].begin == 17 && ip.state_accesses[0].end == 81 &&
        ip.state_accesses[0].write, "cache write valid interval was lost");
  for (const auto initial : {0, 1}) {
    auto conv = make("conv_silu_prefill_f16"); conv.integers["initial"] = initial;
    tp::Lowered cp; tp::lower_operator(conv, cp);
    check(cp.state_accesses.size() == (initial ? 1U : 2U), "conv initial history access mismatch");
    check(cp.state_accesses.back().tensor == "history_out" && cp.state_accesses.back().write,
          "conv output history access missing");
  }
  for (const auto* name : {"causal_depthwise_conv1d_prefill_f16",
                          "causal_depthwise_conv1d_step_f16", "conv_silu_prefill_f16"}) {
    const auto conv = make(name);
    for (const auto* role : {"partial", "replicated", "single", "sequence_shard"}) {
      rejects(conv, [role](auto& r) {
        for (auto& port : r.tensors) if (port.second.storage == "spm") {
          port.second.distribution = role;
          port.second.shard_axis = std::string(role) == "sequence_shard" ? 0 : -1;
        }
      }); ++guards;
    }
    auto sharded = make(name);
    for (auto& port : sharded.tensors) if (port.second.storage == "spm") {
      port.second.distribution = "sharded";
      port.second.shard_axis = port.second.shape.size() - 1;
    }
    tp::Lowered plan;
    check(tp::lower_operator(sharded, plan), "channel-sharded convolution was rejected");
    const auto output = std::string(name) == "causal_depthwise_conv1d_prefill_f16" ? "pad" :
                        std::string(name) == "causal_depthwise_conv1d_step_f16" ? "state" : "qout";
    rejects(sharded, [output](auto& r) { r.tensors.at(output).shard_axis = 0; }); ++guards;
    if (std::string(name) != "conv_silu_prefill_f16") {
      auto common = make(name);
      const auto input = std::string(name) == "causal_depthwise_conv1d_prefill_f16" ? "halo" : "input";
      common.tensors.at(input).distribution = "replicated";
      tp::Lowered common_plan;
      check(tp::lower_operator(common, common_plan), "replicated convolution read dependency was rejected");
    }
  }
  for (const auto initial : {0, 1}) {
    auto conv = make("conv_silu_prefill_f16"); conv.integers["initial"] = initial;
    for (const auto* name : {"qpad", "kpad", "vpad", "history", "history_out"})
      conv.tensors.at(name).distribution = "replicated";
    tp::Lowered plan;
    check(tp::lower_operator(conv, plan), "replicated raw history handoff was rejected");
    rejects(conv, [](auto& r) { r.tensors.at("qpad").distribution = "local"; }); ++guards;
    if (!initial) {
      rejects(conv, [](auto& r) { r.tensors.at("history").distribution = "local"; }); ++guards;
      conv.integers["valid_rows"] = 3;
    }
    conv.tensors.at("history").distribution = "local";
    tp::Lowered independent_history;
    check(tp::lower_operator(conv, independent_history), "unused old history incorrectly constrained replicated handoff");
  }
  tp::Lowered pp; tp::lower_operator(make("copy_physical_prefix_f16"), pp);
  check(pp.state_accesses.size() == 24, "prefix copy must track all source/destination ranges");
  for (const auto& access : pp.state_accesses)
    check(access.begin == 0 && access.end == (access.write ? 128U : 65U),
          "prefix copy mixed physical padding with valid logical range");

  tp::Request unknown; unknown.op = "unknown"; tp::Lowered result;
  check(!tp::lower_operator(unknown, result), "unknown operator claimed by TP state dispatch");
  std::uint64_t codec_elements = 0;
  // Enumerate the physical tile axes independently and check every logical
  // coordinate, including all cores, partial tile boundaries and capacity tail.
  for (const std::string layout : {"vision_k_cache", "vision_v_cache",
                                    "causal_k_cache", "causal_v_cache"}) {
    Inputs a;
    const bool vision = layout.find("vision") == 0;
    const bool value = layout.find("_v_") != std::string::npos;
    const unsigned capacity = vision ? 48 : 784;
    const unsigned width = vision ? 128 : 256;
    const auto t = a.tensor({8, capacity, width}, "shared", "ddr", layout, "cache");
    std::uint64_t physical = 0;
    for (unsigned block = 0; block < capacity / 16; ++block)
      for (unsigned head = 0; head < (vision ? 2U : 1U); ++head)
        for (unsigned dimension_tile = 0; dimension_tile < (vision ? 4U : 16U); ++dimension_tile)
          for (unsigned core = 0; core < 8; ++core)
            for (unsigned major = 0; major < 16; ++major)
              for (unsigned minor = 0; minor < 16; ++minor) {
                const auto token = block * 16 + (value ? minor : major);
                const auto channel = head * 64 + dimension_tile * 16 + (value ? major : minor);
                check(tp::cache_offset(t, core, token, channel) == physical++, "cache codec axis order mismatch");
              }
    check(physical == tp::elements(t), "cache codec does not cover full allocation");
    codec_elements += physical;
    bool rejected = false;
    try { tp::cache_offset(t, 8, 0, 0); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "cache codec accepted invalid core");
  }
  std::cout << "TP state contracts: " << names.size() << " admitted entries, "
            << guards << " rejection checks, 48 counter epochs, " << codec_elements
            << " cache codec elements passed\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
