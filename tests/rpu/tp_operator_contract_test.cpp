// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/tp_contract.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace xinfer::rpu::tp;
unsigned accepted = 0, rejected = 0;
void check(bool value, const char* what) {
  if (!value) throw std::runtime_error(what);
}
void rejects(Request r, const std::function<void(Request&)>& mutate) {
  mutate(r);
  r.used_tensors.clear(); r.used_integers.clear(); r.used_scalars.clear();
  try {
    Lowered result;
    lower_operator(r, result);
  } catch (const std::invalid_argument&) { ++rejected; return; }
  throw std::runtime_error("invalid request accepted: " + r.op);
}
struct Fixture {
  Request request;
  std::uint64_t spm_address = 0x1000, ddr_address = 0x100000000ULL;
  explicit Fixture(std::string op) { request.op = std::move(op); }
  void add(const std::string& name, std::initializer_list<std::int64_t> dims,
           const char* distribution = "local", std::int64_t axis = -1,
           const char* storage = "spm", const char* layout_name = "row_major",
           bool immutable = false, const char* dtype = "float16") {
    Tensor t;
    t.shape = dims; t.distribution = distribution; t.shard_axis = axis;
    t.storage = storage; t.layout = layout_name; t.immutable = immutable;
    t.dtype = dtype; t.bytes = elements(t) * (t.dtype == "int32" ? 4 : 2);
    if (t.layout == "vision_k_cache" || t.layout == "vision_v_cache") {
      t.bytes = std::uint64_t((t.shape[1] + 15) / 16) * 16 * 8 * 128 * 2;
      t.state_kind = "cache";
    }
    auto& next = t.storage == "spm" ? spm_address : ddr_address;
    t.address = next; t.identity = next;
    next += ((t.bytes + 255) & ~std::uint64_t(255)) + 256;
    t.values_known = true; t.finite = true; t.minimum = 0; t.maximum = 1;
    request.tensors.emplace(name, std::move(t));
  }
  void weight(const std::string& name, std::initializer_list<std::int64_t> dims) {
    add(name, dims, "shared", -1, "ddr", "tp8_nk16", true);
  }
  void table(const std::string& name, std::initializer_list<std::int64_t> dims) {
    add(name, dims, "shared", -1, "ddr");
  }
};
const std::vector<std::string> names{
    "rmsnorm_f16", "rmsnorm_gdn_f16", "rmsnorm_coeff_h2048_f16", "rope_f16",
    "sigmoid_mul_m16d256_f16", "sigmoid_mul_m32d256_f16",
    "bmm_fp16_spm", "bmm_fp16_spm_128x128x64_colcol",
    "bmm_fp16_spm_64x128x128_rowcol", "bmm_fp16_spm_64x128x64_rowcol",
    "bmm_fp16_spm_64x64x128_rowrow", "gemm_ddr_col_m16n256k1024_acc32",
    "gemm_ddr_col_m32n256k1024_acc32", "patch_embed_n128k1536", "fc1_n512k4096",
    "up_n512k1024", "gather_embedding_f16", "gate_up_silu_m16n256k1024_acc32",
    "gate_up_silu_m32n256k1024_acc32", "up_gelu_n512k1024", "fc1_gelu_n512k4096",
    "qkv_rope_bias_n384k1024", "qknorm_qkv_rope_m16k1024_acc32",
    "qknorm_qkv_rope_m32k1024_acc32", "qk_rmsnorm_mrope_m16d256_f16",
    "qk_rmsnorm_mrope_m32d256_f16", "adaptive_rmsnorm_m16h1024_f16",
    "adaptive_rmsnorm_m32h1024_f16", "flash_attention_spm_d384",
    "flash_attention_varlen_f16"};
Request fixture(const std::string& op) {
  Fixture f(op);
  const auto rows = op.find("m16") != std::string::npos ? 16 : 32;
  if (op.find("gemm_ddr_col_") == 0 || op.find("gate_up_silu_") == 0) {
    f.add("x", {rows, 1024}, "replicated"); f.add("out", {rows, 256}, "sharded", 1);
    if (op.find("gemm_ddr_col_") == 0) f.weight("weight", {8, 256, 1024});
    else { f.weight("gate_weight", {8, 256, 1024}); f.weight("up_weight", {8, 256, 1024}); }
  } else if (op.find("bmm_") == 0) {
    const bool colcol = op.find("colcol") != std::string::npos;
    const bool rowrow = op == "bmm_fp16_spm" || op.find("rowrow") != std::string::npos;
    const auto m = colcol ? 128 : 64, n = rowrow ? 64 : 128;
    const auto k = rowrow || op == "bmm_fp16_spm_64x128x128_rowcol" ? 128 : 64;
    f.add("a", {2, colcol ? k : m, colcol ? m : k});
    f.add("b", {2, rowrow ? n : k, rowrow ? k : n}); f.add("out", {2, m, n});
    f.request.integers["transpose_a"] = colcol;
    f.request.integers["transpose_b"] = rowrow;
  } else if (op == "rmsnorm_coeff_h2048_f16") {
    f.add("x", {128, 2048}); f.add("out", {128, 2048}); f.add("weight", {2048});
  } else if (op == "rmsnorm_f16" || op == "rmsnorm_gdn_f16") {
    f.add("x", {128, 128}); f.add("out", {128, 128}); f.add("weight", {128});
  } else if (op == "rope_f16") {
    f.add("x", {16, 2, 64}, "single"); f.add("out", {16, 2, 64}, "single");
    f.add("cosine", {32, 32}, "single"); f.add("sine", {32, 32}, "single");
    f.request.integers["position_offset"] = 5;
  } else if (op.find("sigmoid_mul_") == 0) {
    f.add("out", {rows, 256}); f.add("gate", {rows, 256}); f.add("value", {rows, 256});
  } else if (op.find("adaptive_rmsnorm_") == 0) {
    f.add("out", {rows, 1024}); f.add("x", {rows, 1024}); f.add("modulation", {3, 1024});
  } else if (op.find("qk_rmsnorm_mrope_") == 0) {
    f.add("out", {rows, 256}, "sharded", 1); f.add("x", {rows, 256}, "sharded", 1);
    f.add("weight", {256}, "replicated"); f.table("cosine", {64, 32}); f.table("sine", {64, 32});
    f.request.integers["position_offset"] = 5;
  } else if (op.find("qknorm_qkv_rope_") == 0) {
    f.add("x", {rows, 1024}, "replicated"); f.add("planes", {4, rows, 256});
    f.add("qk_weight", {2, 256}, "replicated"); f.weight("weight", {4, 8, 256, 1024});
    f.table("cosine", {64, 32}); f.table("sine", {64, 32});
    f.request.integers["position_offset"] = 5;
  } else if (op == "qkv_rope_bias_n384k1024") {
    f.add("x", {16, 1024}, "replicated"); f.add("q", {16, 128}, "sharded", 1);
    f.add("k", {16, 128}, "sharded", 1); f.add("v", {16, 128}, "sharded", 1);
    f.add("bias", {384}); f.weight("weight", {8, 384, 1024});
    f.add("cosine", {16, 32}, "replicated"); f.add("sine", {16, 32}, "replicated");
  } else if (op == "flash_attention_spm_d384" || op == "flash_attention_varlen_f16") {
    f.add("q", {32, 128}, "sharded", 1); f.add("out", {32, 128}, "sharded", 1);
    f.request.integers["length0"] = 17; f.request.integers["length1"] = 15;
    if (op == "flash_attention_spm_d384") {
      f.add("k", {32, 128}, "sharded", 1); f.add("v", {32, 128}, "sharded", 1);
    } else {
      for (unsigned i = 0; i < 2; ++i) {
        f.add("key_cache" + std::to_string(i), {8, 17, 128}, "shared", -1, "ddr", "vision_k_cache");
        f.add("value_cache" + std::to_string(i), {8, 17, 128}, "shared", -1, "ddr", "vision_v_cache");
      }
    }
  } else if (op == "gather_embedding_f16") {
    f.add("out", {16, 2048}, "single");
    f.add("ids", {13}, "shared", -1, "ddr", "row_major", false, "int32");
    f.add("vocab", {256277, 2048}, "shared", -1, "ddr", "row_major", true);
  } else {
    const bool patch = op == "patch_embed_n128k1536";
    const bool fc1 = op.find("fc1_") == 0;  // fc1_n512k4096 / fc1_gelu_n512k4096
    const auto n = patch ? 128 : 512, k = patch ? 1536 : fc1 ? 4096 : 1024;
    f.add("x", {16, k}, "replicated"); f.add("out", {16, n}, "sharded", 1);
    f.add("bias", {n}, "sharded", 0); f.weight("weight", {8, n, k});
  }
  return f.request;
}
std::uint32_t reg32(const Lowered& p, unsigned slot) {
  return p.plan.regs[slot] | std::uint32_t(p.plan.regs[slot + 1]) << 16;
}
}  // namespace

int main() try {
  for (const auto& name : names) {
    const auto r = fixture(name);
    Lowered result;
    check(lower_operator(r, result), "TP operator catalog entry not handled");
    check(name == xinfer::rpu::abi::name(result.plan.entry), "wrong selected entry");
    check(!result.writes.empty(), "missing mutable-resource declaration");
    ++accepted;
    const auto output = result.writes.front();
    rejects(r, [](Request& x) { x.integers["unsupported_parameter"] = 1; });
    rejects(r, [output](Request& x) { x.tensors[output].immutable = true; });
    rejects(r, [output](Request& x) { x.tensors[output].dtype = "float32"; });
    rejects(r, [output](Request& x) { x.tensors[output].bytes = 2; });
    rejects(r, [output](Request& x) { x.tensors[output].layout = "unknown_stride"; });
    rejects(r, [output](Request& x) { x.tensors[output].storage = "ddr"; });
    rejects(r, [output](Request& x) { x.tensors[output].shape[0] = -1; });
  }

  auto column = fixture("gemm_ddr_col_m16n256k1024_acc32");
  Lowered col; check(lower_operator(column, col), "column lowering failed");
  check(col.plan.cores == 8 && col.plan.grid[0] == 2 && col.plan.register_count == 6,
        "Action column launch geometry changed");
  check(reg32(col, 0) == column.tensors.at("x").address &&
        reg32(col, 2) == column.tensors.at("weight").address / 256 &&
        reg32(col, 4) == column.tensors.at("out").address, "Action DDR units/ports wrong");
  rejects(column, [](Request& r) { r.tensors["weight"].layout = "row_major"; });
  rejects(column, [](Request& r) { r.tensors["weight"].immutable = false; });
  rejects(column, [](Request& r) { r.tensors["out"].distribution = "replicated"; });
  rejects(column, [](Request& r) { r.tensors["out"].shard_axis = 0; });
  rejects(column, [](Request& r) { r.tensors["out"].address = r.tensors["x"].address; });
  rejects(column, [](Request& r) { r.tensors["bias"] = r.tensors["out"]; });

  auto bmm = fixture("bmm_fp16_spm");
  rejects(bmm, [](Request& r) { r.integers["transpose_a"] = 1; });
  rejects(bmm, [](Request& r) {
    for (auto& item : r.tensors) {
      item.second.distribution = "sharded"; item.second.shard_axis = 1;
    }
  });
  rejects(bmm, [](Request& r) {
    for (auto& item : r.tensors) { item.second.shape[0] = 14; item.second.bytes *= 7; }
  });
  auto fixed = fixture("bmm_fp16_spm_64x64x128_rowrow");
  for (auto& item : fixed.tensors) { item.second.shape[0] = 16; item.second.bytes *= 8; }
  fixed.tensors["a"].address = 0x1000; fixed.tensors["b"].address = 0x100000;
  fixed.tensors["out"].address = 0x200000;
  Lowered fixed_plan; check(lower_operator(fixed, fixed_plan) && fixed_plan.plan.grid[2] == 16,
                            "fixed BMM batch16 support lost");

  auto coeff = fixture("rmsnorm_coeff_h2048_f16");
  auto inplace_coeff = coeff;
  inplace_coeff.tensors["out"] = inplace_coeff.tensors["x"];
  Lowered coeff_plan;
  check(lower_operator(inplace_coeff, coeff_plan) &&
            coeff_plan.plan.entry == xinfer::rpu::abi::Entry::RmsNormCoeffH2048F16,
        "Text coefficient RMS exact alias lost");
  rejects(coeff, [](Request& r) { r.tensors["out"].address = r.tensors["x"].address + 32; });
  rejects(coeff, [](Request& r) { r.tensors["weight"].address = r.tensors["x"].address; });
  rejects(coeff, [](Request& r) { r.tensors["x"].shape[0] = r.tensors["out"].shape[0] = 65; });
  rejects(coeff, [](Request& r) { r.scalars["epsilon"] = 1e-5; });
  const auto gdn = fixture("rmsnorm_gdn_f16");
  rejects(gdn, [](Request& r) { r.tensors["x"].maximum = 1.001; });
  rejects(gdn, [](Request& r) { r.tensors["x"].values_known = false; });
  rejects(gdn, [](Request& r) { r.tensors["x"].finite = false; });
  rejects(gdn, [](Request& r) { r.scalars["epsilon"] = 1.0e-5; });
  auto gdn_req = gdn; Lowered gdn_plan; lower_operator(gdn_req, gdn_plan);
  check(gdn_req.value_requirements.size() == 1, "GDN replay value guard not retained");
  const auto embedding = fixture("gather_embedding_f16");
  rejects(embedding, [](Request& r) { r.tensors["ids"].minimum = -1; });
  rejects(embedding, [](Request& r) { r.tensors["ids"].maximum = 256277; });
  rejects(embedding, [](Request& r) { r.tensors["ids"].dtype = "int64"; });

  const auto qkv = fixture("qknorm_qkv_rope_m16k1024_acc32");
  Lowered qkv_plan; lower_operator(qkv, qkv_plan);
  check(qkv_plan.zero_counters == std::vector<unsigned>({128, 130}), "QKV counter replay initialization lost");
  rejects(qkv, [](Request& r) { r.integers["position_offset"] = 49; });
  rejects(qkv, [](Request& r) { r.scalars["epsilon"] = 1.0e-5; });
  rejects(qkv, [](Request& r) { r.tensors["planes"].distribution = "sharded"; });
  const auto adaptive = fixture("adaptive_rmsnorm_m32h1024_f16");
  rejects(adaptive, [](Request& r) { r.scalars["normalization_scale"] = 64; });
  for (const auto* name : {"rmsnorm_f16", "rmsnorm_gdn_f16", "rmsnorm_coeff_h2048_f16",
                          "adaptive_rmsnorm_m16h1024_f16", "adaptive_rmsnorm_m32h1024_f16",
                          "qk_rmsnorm_mrope_m16d256_f16", "qk_rmsnorm_mrope_m32d256_f16"}) {
    auto norm = fixture(name);
    const bool adaptive_norm = norm.op.find("adaptive_") == 0;
    const char* parameter = adaptive_norm ? "modulation" : "weight";
    for (const auto* port : {"x", "out"}) {
      norm.tensors.at(port).distribution = "replicated";
      norm.tensors.at(port).shard_axis = -1;
    }
    norm.tensors.at(parameter).distribution = "local";
    // Core-dependent gamma/shift cannot produce a replicated result.
    rejects(norm, [](Request&) {});
    norm.tensors.at(parameter).distribution = "replicated";
    Lowered replicated;
    check(lower_operator(norm, replicated), "replicated norm parameters were rejected");
    norm = fixture(name);
    norm.tensors.at(parameter).distribution = "local";
    for (const auto* port : {"x", "out"}) {
      norm.tensors.at(port).distribution = "local";
      norm.tensors.at(port).shard_axis = -1;
    }
    Lowered independent;
    check(lower_operator(norm, independent), "independent local norm parameters were rejected");
    norm = fixture(name);
    norm.tensors.at(parameter).distribution = "local";
    for (const auto* port : {"x", "out"}) {
      norm.tensors.at(port).distribution = "sharded";
      norm.tensors.at(port).shard_axis = norm.op.find("qk_") == 0 ? 1 : 0;
    }
    Lowered sharded;
    check(lower_operator(norm, sharded), "valid sharded norm with local parameters was rejected");
  }
  for (const auto* name : {"adaptive_rmsnorm_m16h1024_f16", "adaptive_rmsnorm_m32h1024_f16"}) {
    const auto norm = fixture(name);
    for (const auto epsilon : {65520.0, 1.0e30})
      rejects(norm, [epsilon](Request& r) { r.scalars["epsilon"] = epsilon; });
    for (const auto epsilon : {1.0e-6, 65504.0}) {
      auto valid = norm; valid.scalars["epsilon"] = epsilon; Lowered plan;
      check(lower_operator(valid, plan), "positive finite FP16 epsilon was rejected");
      check(plan.plan.regs[8] == (epsilon == 65504.0 ? 0x7bff : 0x0011),
            "adaptive epsilon was not encoded as the expected finite FP16");
    }
  }

  // RNE/LUT private variant branches are gone.

  const auto attention = fixture("flash_attention_varlen_f16");
  Lowered attention_plan; lower_operator(attention, attention_plan);
  check(attention_plan.plan.regs[16] == 17 && attention_plan.plan.regs[17] == 15,
        "attention length order changed");
  check(reg32(attention_plan, 4) == attention.tensors.at("key_cache0").address / 256,
        "attention cache DDR addressing wrong");
  check(attention_plan.state_accesses.size() == 4 &&
        attention_plan.state_accesses[0].begin == 0 &&
        attention_plan.state_accesses[0].end == 17 &&
        attention_plan.state_accesses[2].end == 15 &&
        !attention_plan.state_accesses[0].write,
        "attention cache valid-prefix requirements missing");
  rejects(attention, [](Request& r) { r.scalars["scale"] = 0.0625; });
  rejects(attention, [](Request& r) { r.integers["return_lse"] = 1; });
  rejects(attention, [](Request& r) { r.integers["causal"] = 1; });
  rejects(attention, [](Request& r) { r.tensors["key_cache0"].layout = "vision_v_cache"; });
  rejects(attention, [](Request& r) { r.tensors["key_cache0"].state_kind = "none"; });
  rejects(attention, [](Request& r) { r.tensors["key_cache0"].bytes = 8 * 17 * 128 * 2; });
  Request unknown; unknown.op = "not_a_kernel"; Lowered unknown_plan;
  check(!lower_operator(unknown, unknown_plan), "unknown operation claimed by TP operator dispatch");
  std::cout << "{\"operator_entries\":" << accepted << ",\"rejections\":" << rejected
            << ",\"passed\":true}" << std::endl;
  return 0;
} catch (const std::exception& error) {
  std::cerr << "tp_operator_contract_test: " << error.what() << '\n'; return 1;
}
