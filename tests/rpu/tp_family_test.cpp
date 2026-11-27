// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Board-free contract: each family entry point lowers to exactly the launch
// packet of its catalog entry. A family request differs from the catalog
// request only in names and selectors, so equal packets mean equal device
// work, and board validation of the catalog entries carries over unchanged.
#include "common/rpu/tp_contract.hpp"
#include "common/rpu/tp_family.hpp"
#include "common/rpu/gdr_prefill_abi.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace tp = xinfer::rpu::tp;
namespace abi = xinfer::rpu::abi;

namespace {
unsigned checks = 0;
void check(bool ok, const std::string& message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
template <class F> void rejects(F f, const std::string& needle) {
  ++checks;
  try {
    f();
  } catch (const std::invalid_argument& error) {
    if (std::string(error.what()).find(needle) == std::string::npos)
      throw std::runtime_error("unexpected rejection: " + std::string(error.what()) +
                               " (expected " + needle + ")");
    return;
  }
  throw std::runtime_error("expected rejection containing: " + needle);
}

// One tensor, described the way the runtime records it.
struct Spec {
  std::vector<std::int64_t> shape;
  std::string distribution = "local", storage = "spm", layout = "row_major";
  std::string dtype = "float16", state = "none";
  std::int64_t axis = -2;  // -2: derive from distribution
  bool immutable = false;
};

// Tensors keep their identity, address and description across names, so the
// same object can be bound under its catalog name and its family name.
struct Pool {
  std::uint64_t spm = 0x10000, ddr = 0x100000000ULL, next = 1;
  std::map<std::string, tp::Tensor> made;
  const tp::Tensor& get(const std::string& key, const Spec& s) {
    auto found = made.find(key);
    if (found != made.end()) return found->second;
    tp::Tensor t;
    t.shape = s.shape; t.distribution = s.distribution; t.storage = s.storage;
    t.layout = s.layout; t.dtype = s.dtype; t.state_kind = s.state;
    t.bytes = tp::elements(t) * (s.dtype == "int32" ? 4 : 2);
    t.identity = next++;
    t.shard_axis = s.axis != -2 ? s.axis
                   : s.distribution == "sharded" ? (t.shape.size() == 1 ? 0 : 1) : -1;
    t.values_known = true; t.finite = true; t.minimum = 0; t.maximum = 0;
    t.immutable = s.immutable;
    if (s.state != "none") t.generation = 3;
    auto& at = s.storage == "spm" ? spm : ddr;
    t.address = at; at += (t.bytes + 255) & ~std::uint64_t{255};
    return made.emplace(key, t).first->second;
  }
};

struct Bind { std::string name, key; };

tp::Request request(Pool& pool, const std::string& op, const std::map<std::string, Spec>& specs,
                    const std::vector<Bind>& binds, std::map<std::string, std::int64_t> ints = {},
                    std::map<std::string, double> scalars = {}) {
  tp::Request r;
  r.op = op;
  for (const auto& b : binds) r.tensors.emplace(b.name, pool.get(b.key, specs.at(b.key)));
  r.integers = std::move(ints);
  r.scalars = std::move(scalars);
  return r;
}

tp::Lowered lower(tp::Request r) {
  tp::resolve_family(r);
  tp::Lowered out;
  check(tp::lower_operator(r, out), "no lowerer accepted " + r.op);
  r.finish();
  return out;
}

bool same(const tp::Lowered& a, const tp::Lowered& b) {
  auto key = [](const tp::Lowered& l) {
    std::vector<std::string> accesses;
    for (const auto& s : l.state_accesses)
      accesses.push_back(s.tensor + ":" + std::to_string(s.begin) + "-" +
                         std::to_string(s.end) + (s.write ? "w" : "r"));
    return accesses;
  };
  return a.plan.entry == b.plan.entry && a.plan.regs == b.plan.regs &&
         a.plan.register_count == b.plan.register_count && a.plan.grid == b.plan.grid &&
         a.plan.cores == b.plan.cores && a.transfer_mode == b.transfer_mode &&
         a.zero_counters == b.zero_counters && a.collective == b.collective &&
         key(a) == key(b);
}

// Writes are recorded by name; compare them through the caller's renaming.
bool same_writes(const tp::Lowered& catalog, const tp::Lowered& family,
                 const std::map<std::string, std::string>& family_to_catalog) {
  std::vector<std::string> mapped;
  for (const auto& w : family.writes) {
    const auto found = family_to_catalog.find(w);
    mapped.push_back(found == family_to_catalog.end() ? w : found->second);
  }
  return mapped == catalog.writes;
}

struct Case {
  std::string label;
  std::string catalog, family;
  std::map<std::string, Spec> specs{};
  std::vector<Bind> catalog_binds{}, family_binds{};
  std::map<std::string, std::int64_t> catalog_ints{}, family_ints{};
  std::map<std::string, double> scalars{};
  std::string expect_entry{};  // kernel name; empty = same as catalog
};

std::vector<std::string> resolved_pairs;  // "family catalog" for --pairs

void equivalent(const Case& c) {
  Pool pool;
  auto legacy = request(pool, c.catalog, c.specs, c.catalog_binds, c.catalog_ints, c.scalars);
  auto modern = request(pool, c.family, c.specs, c.family_binds, c.family_ints, c.scalars);
  const auto a = lower(legacy), b = lower(modern);
  check(same(a, b), c.label + ": family packet differs from " + c.catalog);
  // Resolved names line up with the catalog names, so writes match by name.
  auto resolved = modern;
  std::map<std::string, std::string> renamed;
  tp::resolve_family(resolved, &renamed);
  check(resolved.op == c.catalog, c.label + ": resolved to " + resolved.op);
  resolved_pairs.push_back(c.family + " " + c.catalog);
  check(same_writes(a, b, {}), c.label + ": write sets differ");
  if (!c.expect_entry.empty())
    check(std::string(abi::name(b.plan.entry)) == c.expect_entry,
          c.label + ": selected " + abi::name(b.plan.entry));
}

Spec spm(std::vector<std::int64_t> shape, std::string dist = "local", std::int64_t axis = -2) {
  Spec s; s.shape = std::move(shape); s.distribution = std::move(dist); s.axis = axis; return s;
}
Spec tp8_weight(std::int64_t n, std::int64_t k) {
  Spec s; s.shape = {8, n, k}; s.distribution = "shared"; s.storage = "ddr";
  s.layout = "tp8_nk16"; s.immutable = true; return s;
}
Spec nk16_weight(std::int64_t n, std::int64_t k) {
  Spec s; s.shape = {n, k}; s.distribution = "shared"; s.storage = "ddr";
  s.layout = "nk16"; s.immutable = true; return s;
}
Spec table(std::int64_t rows, std::int64_t cols, std::string storage) {
  Spec s; s.shape = {rows, cols}; s.storage = storage;
  if (storage == "ddr") { s.distribution = "shared"; s.immutable = true; }
  else s.distribution = "replicated";
  return s;
}
Spec cache(std::vector<std::int64_t> shape, std::string layout) {
  Spec s; s.shape = std::move(shape); s.distribution = "shared"; s.storage = "ddr";
  s.layout = std::move(layout); s.state = "cache"; return s;
}

std::vector<Case> gemm_cases() {
  std::vector<Case> c;
  // Language row projections (partial outputs; no implicit reduction).
  for (auto [k, name] : {std::pair{256, "o_n2048k256"}, std::pair{768, "down_n2048k768"}})
    c.push_back({std::string("gemm row ") + name, name, "gemm",
                 {{"x", spm({128, k}, "sharded")}, {"w", tp8_weight(2048, k)},
                  {"out", spm({128, 2048}, "partial")}},
                 {{"src", "x"}, {"weight", "w"}, {"out", "out"}},
                 {{"x", "x"}, {"w", "w"}, {"out", "out"}},
                 {}, {{"parallel", 1}}});
  for (int m : {16, 32}) {
    const auto rows = std::to_string(m);
    c.push_back({"gemm row m" + rows, "gemm_ddr_row_m" + rows + "n1024k256_acc32", "gemm",
                 {{"x", spm({m, 256}, "sharded")}, {"w", tp8_weight(1024, 256)},
                  {"out", spm({m, 1024}, "partial")}},
                 {{"src", "x"}, {"weight", "w"}, {"out", "out"}},
                 {{"x", "x"}, {"w", "w"}, {"out", "out"}},
                 {}, {{"parallel", 1}}});
    c.push_back({"gemm column m" + rows, "gemm_ddr_col_m" + rows + "n256k1024_acc32", "gemm",
                 {{"x", spm({m, 1024}, "replicated")}, {"w", tp8_weight(256, 1024)},
                  {"out", spm({m, 256}, "sharded")}},
                 {{"x", "x"}, {"weight", "w"}, {"out", "out"}},
                 {{"x", "x"}, {"w", "w"}, {"out", "out"}}});
    // Core0 projections over nk16 DDR weights.
    Spec single_in = spm({m, 64}, "single"), single_bias = spm({1024}, "single"),
         single_out = spm({m, 1024}, "single");
    c.push_back({"gemm input projection /4 m" + rows,
                 "gemm_ddr_bias_scale_m" + rows + "n1024k64_acc32", "gemm",
                 {{"x", single_in}, {"w", nk16_weight(1024, 64)}, {"bias", single_bias},
                  {"out", single_out}},
                 {{"input", "x"}, {"weight", "w"}, {"bias", "bias"}, {"out", "out"}},
                 {{"x", "x"}, {"w", "w"}, {"bias", "bias"}, {"out", "out"}},
                 {}, {{"output_scale", 1}}});
    c.push_back({"gemm output projection m" + rows,
                 "gemm_ddr_bias_m" + rows + "n64k1024_acc32", "gemm",
                 {{"x", spm({m, 1024}, "single")}, {"w", nk16_weight(64, 1024)},
                  {"bias", spm({64}, "single")}, {"out", spm({m, 64}, "single")}},
                 {{"input", "x"}, {"weight", "w"}, {"bias", "bias"}, {"out", "out"}},
                 {{"x", "x"}, {"w", "w"}, {"bias", "bias"}, {"out", "out"}}});
  }
  // Vision / merger / patch projections with bias.
  struct Proj { const char* name; std::int64_t n, k; bool row, gelu; };
  for (const Proj& p : {Proj{"up_n512k1024", 512, 1024, false, false},
                        Proj{"up_gelu_n512k1024", 512, 1024, false, true},
                        Proj{"fc1_n512k4096", 512, 4096, false, false},
                        Proj{"fc1_gelu_n512k4096", 512, 4096, false, true},
                        Proj{"patch_embed_n128k1536", 128, 1536, false, false},
                        Proj{"o_n1024k128", 1024, 128, true, false},
                        Proj{"down_n1024k512", 1024, 512, true, false},
                        Proj{"down_n2048k512", 2048, 512, true, false}}) {
    Case x{std::string("gemm ") + p.name, p.name, "gemm", {}, {}, {}};
    if (p.row) {
      x.specs = {{"x", spm({16, p.k}, "sharded")}, {"w", tp8_weight(p.n, p.k)},
                 {"bias", spm({p.n}, "owner0")}, {"out", spm({16, p.n}, "partial")}};
      x.catalog_binds = {{"src", "x"}, {"weight", "w"}, {"bias", "bias"}, {"out", "out"}};
      x.family_ints = {{"parallel", 1}};
    } else {
      x.specs = {{"x", spm({16, p.k}, "replicated")}, {"w", tp8_weight(p.n, p.k)},
                 {"bias", spm({p.n}, "sharded")}, {"out", spm({16, p.n}, "sharded")}};
      x.catalog_binds = {{"x", "x"}, {"weight", "w"}, {"bias", "bias"}, {"out", "out"}};
      if (p.gelu) x.family_ints = {{"activation", 1}};
    }
    x.family_binds = {{"x", "x"}, {"w", "w"}, {"bias", "bias"}, {"out", "out"}};
    c.push_back(x);
  }
  return c;
}

std::vector<Case> projection_cases() {
  std::vector<Case> c;
  // Language QKV: base bucket, and 448/512 rows select the K128 programs.
  for (unsigned m : {64u, 384u, 448u, 512u}) {
    const auto rows = std::to_string(m);
    const std::string k128 = m == 448 ? "qkv_qgate_m448_k128" : m == 512 ? "qkv_qgate_m512_k128" : "";
    c.push_back({"qkv_proj language rows " + rows, "qkv_qgate_n1024k2048", "qkv_proj",
                 {{"x", spm({m, 2048}, "replicated")}, {"w", tp8_weight(1024, 2048)},
                  {"q", spm({m, 256}, "sharded")}, {"qg", spm({m, 256}, "sharded")},
                  {"k", spm({m, 256})}, {"v", spm({m, 256})}},
                 {{"src", "x"}, {"weight", "w"}, {"out_q", "q"}, {"out_qgate", "qg"},
                  {"out_k", "k"}, {"out_v", "v"}},
                 {{"x", "x"}, {"w", "w"}, {"out_q", "q"}, {"out_qgate", "qg"},
                  {"out_k", "k"}, {"out_v", "v"}},
                 {}, {}, {}, k128});
    std::map<std::string, Spec> zba{{"x", spm({m, 2048}, "replicated")},
                                    {"w", tp8_weight(1056, 2048)}};
    std::vector<Bind> legacy{{"src", "x"}, {"weight", "w"}}, modern{{"x", "x"}, {"w", "w"}};
    for (const char* port : {"out_q", "out_k", "out_v", "out_z"}) {
      zba.emplace(port, spm({m, 256}, "sharded"));
      legacy.push_back({port, port}); modern.push_back({port, port});
    }
    for (const char* port : {"out_b", "out_a"}) {
      zba.emplace(port, spm({m, 16}, "sharded"));
      legacy.push_back({port, port}); modern.push_back({port, port});
    }
    c.push_back({"qkv_proj linear-attention rows " + rows, "qkv_zba_n1056k2048", "qkv_proj",
                 zba, legacy, modern});
    c.push_back({"gemm_glu language rows " + rows, "gateup_n1536k2048", "gemm_glu",
                 {{"x", spm({m, 2048}, "replicated")}, {"w", tp8_weight(1536, 2048)},
                  {"gate", spm({m, 768}, "sharded")}, {"up", spm({m, 768}, "sharded")}},
                 {{"src", "x"}, {"weight", "w"}, {"out_gate", "gate"}, {"out_up", "up"}},
                 {{"x", "x"}, {"w", "w"}, {"out_gate", "gate"}, {"out_up", "up"}},
                 {}, {{"activation", 0}}});
  }
  // Vision QKV, plain and with fused RoPE.
  c.push_back({"qkv_proj vision", "qkv_bias_n384k1024", "qkv_proj",
               {{"x", spm({16, 1024}, "replicated")}, {"w", tp8_weight(384, 1024)},
                {"bias", spm({384}, "sharded")}, {"q", spm({16, 128}, "sharded")},
                {"k", spm({16, 128}, "sharded")}, {"v", spm({16, 128}, "sharded")}},
               {{"src", "x"}, {"weight", "w"}, {"bias", "bias"}, {"out_q", "q"},
                {"out_k", "k"}, {"out_v", "v"}},
               {{"x", "x"}, {"w", "w"}, {"bias", "bias"}, {"out_q", "q"},
                {"out_k", "k"}, {"out_v", "v"}}});
  c.push_back({"qkv_proj vision rope", "qkv_rope_bias_n384k1024", "qkv_proj",
               {{"x", spm({16, 1024}, "replicated")}, {"w", tp8_weight(384, 1024)},
                {"bias", spm({384})}, {"q", spm({16, 128}, "sharded")},
                {"k", spm({16, 128}, "sharded")}, {"v", spm({16, 128}, "sharded")},
                {"cos", table(16, 32, "spm")}, {"sin", table(16, 32, "spm")}},
               {{"x", "x"}, {"weight", "w"}, {"bias", "bias"}, {"q", "q"}, {"k", "k"},
                {"v", "v"}, {"cosine", "cos"}, {"sine", "sin"}},
               {{"x", "x"}, {"w", "w"}, {"bias", "bias"}, {"out_q", "q"}, {"out_k", "k"},
                {"out_v", "v"}, {"cos", "cos"}, {"sin", "sin"}}});
  // Action QKV + QK RMSNorm + RoPE and the fused gate/up SiLU.
  for (int m : {16, 32}) {
    const auto rows = std::to_string(m);
    Spec planes_weight; planes_weight.shape = {4, 8, 256, 1024}; planes_weight.distribution = "shared";
    planes_weight.storage = "ddr"; planes_weight.layout = "tp8_nk16"; planes_weight.immutable = true;
    c.push_back({"qkv_proj action m" + rows, "qknorm_qkv_rope_m" + rows + "k1024_acc32", "qkv_proj",
                 {{"x", spm({m, 1024}, "replicated")}, {"planes", spm({4, m, 256})},
                  {"qkw", spm({2, 256}, "replicated")}, {"w", planes_weight},
                  {"cos", table(64, 32, "ddr")}, {"sin", table(64, 32, "ddr")}},
                 {{"x", "x"}, {"planes", "planes"}, {"qk_weight", "qkw"}, {"weight", "w"},
                  {"cosine", "cos"}, {"sine", "sin"}},
                 {{"x", "x"}, {"planes", "planes"}, {"qk_weight", "qkw"}, {"w", "w"},
                  {"cos", "cos"}, {"sin", "sin"}},
                 {}, {}, {{"epsilon", 1e-6}}});
    c.push_back({"gemm_glu action m" + rows, "gate_up_silu_m" + rows + "n256k1024_acc32",
                 "gemm_glu",
                 {{"x", spm({m, 1024}, "replicated")}, {"out", spm({m, 256}, "sharded")},
                  {"wg", tp8_weight(256, 1024)}, {"wu", tp8_weight(256, 1024)}},
                 {{"x", "x"}, {"out", "out"}, {"gate_weight", "wg"}, {"up_weight", "wu"}},
                 {{"x", "x"}, {"out", "out"}, {"w_gate", "wg"}, {"w_up", "wu"}}});
  }
  return c;
}

std::vector<Case> norm_and_pointwise_cases() {
  std::vector<Case> c;
  c.push_back({"rmsnorm default", "rmsnorm_f16", "rmsnorm",
               {{"x", spm({512, 256})}, {"out", spm({512, 256})}, {"w", spm({256})}},
               {{"x", "x"}, {"out", "out"}, {"weight", "w"}},
               {{"x", "x"}, {"out", "out"}, {"weight", "w"}}, {}, {}, {{"epsilon", 1e-6}}});
  c.push_back({"rmsnorm coefficient-first", "rmsnorm_coeff_h2048_f16", "rmsnorm",
               {{"x", spm({512, 2048})}, {"out", spm({512, 2048})}, {"w", spm({2048})}},
               {{"x", "x"}, {"out", "out"}, {"weight", "w"}},
               {{"x", "x"}, {"out", "out"}, {"weight", "w"}}, {}, {{"mode", 1}},
               {{"epsilon", 1e-6}}});
  for (int m : {16, 32}) {
    const auto rows = std::to_string(m);
    c.push_back({"rmsnorm adaptive m" + rows, "adaptive_rmsnorm_m" + rows + "h1024_f16", "rmsnorm",
                 {{"x", spm({m, 1024})}, {"out", spm({m, 1024})}, {"mod", spm({3, 1024})}},
                 {{"x", "x"}, {"out", "out"}, {"modulation", "mod"}},
                 {{"x", "x"}, {"out", "out"}, {"modulation", "mod"}}, {}, {},
                 {{"epsilon", 1e-6}}});
    c.push_back({"sigmoid_mul m" + rows, "sigmoid_mul_m" + rows + "d256_f16", "sigmoid_mul",
                 {{"g", spm({m, 256})}, {"v", spm({m, 256})}, {"o", spm({m, 256})}},
                 {{"gate", "g"}, {"value", "v"}, {"out", "o"}},
                 {{"gate", "g"}, {"value", "v"}, {"out", "o"}}});
    c.push_back({"residual_gate m" + rows, "residual_gate_m" + rows + "h1024_f16", "residual_gate",
                 {{"b", spm({m, 1024}, "replicated")}, {"r", spm({m, 1024}, "replicated")},
                  {"o", spm({m, 1024}, "replicated")}, {"g", spm({1024}, "replicated")}},
                 {{"branch", "b"}, {"residual", "r"}, {"out", "o"}, {"gate", "g"}},
                 {{"branch", "b"}, {"residual", "r"}, {"out", "o"}, {"gate", "g"}}});
    c.push_back({"euler_update m" + rows, "euler_update_m" + rows + "d64_f16", "euler_update",
                 {{"a", spm({m, 64}, "replicated")}, {"v", spm({m, 64}, "replicated")},
                  {"p", spm({m, 64}, "replicated")}, {"mask", spm({64}, "replicated")}},
                 {{"action", "a"}, {"velocity", "v"}, {"padding_velocity", "p"}, {"keep_mask", "mask"}},
                 {{"action", "a"}, {"velocity", "v"}, {"padding_velocity", "p"}, {"keep_mask", "mask"}},
                 {}, {}, {{"delta_t", 0.125}}});
  }
  c.push_back({"layernorm", "layernorm_f16", "layernorm",
               {{"x", spm({129, 1024})}, {"out", spm({129, 1024})}, {"w", spm({1024})},
                {"b", spm({1024})}},
               {{"input", "x"}, {"out", "out"}, {"weight", "w"}, {"bias", "b"}},
               {{"x", "x"}, {"out", "out"}, {"weight", "w"}, {"bias", "b"}}, {}, {},
               {{"epsilon", 1e-5}}});
  // silu_and_mul lowers in the execution adapter (LUT injection), not in the
  // shared lowerer; its family name is checked by the resolver test below.
  c.push_back({"activation gelu tanh", "gelu_f16", "activation",
               {{"x", spm({16, 512})}, {"o", spm({16, 512})}},
               {{"src", "x"}, {"out", "o"}}, {{"x", "x"}, {"out", "o"}},
               {{"mode", 1}}, {{"function", 0}}});
  c.push_back({"activation silu", "unary_f16", "activation",
               {{"x", spm({512, 256})}, {"o", spm({512, 256})}},
               {{"src", "x"}, {"out", "o"}}, {{"x", "x"}, {"out", "o"}},
               {{"operation", 2}}, {{"function", 4}}});
  c.push_back({"binary mul", "binary_mul_f16", "binary",
               {{"l", spm({512, 256})}, {"r", spm({512, 256})}, {"o", spm({512, 256})}},
               {{"lhs", "l"}, {"rhs", "r"}, {"out", "o"}}, {{"lhs", "l"}, {"rhs", "r"}, {"out", "o"}},
               {}, {{"operation", 2}}});
  c.push_back({"binary add tuned", "add_vpb_f16", "binary",
               {{"l", spm({868, 1024})}, {"r", spm({868, 1024})}, {"o", spm({868, 1024})}},
               {{"lhs", "l"}, {"rhs", "r"}, {"out", "o"}}, {{"lhs", "l"}, {"rhs", "r"}, {"out", "o"}},
               {{"vectors_per_block", 32}}, {{"operation", 0}, {"vectors_per_block", 32}}});
  return c;
}

std::vector<Case> attention_and_comm_cases() {
  std::vector<Case> c;
  Spec cos512 = table(512, 32, "spm"), sin512 = table(512, 32, "spm");
  c.push_back({"rope language", "rope_f16", "rope",
               {{"x", spm({512, 1, 256}, "sharded")}, {"cos", cos512}, {"sin", sin512}},
               {{"x", "x"}, {"out", "x"}, {"cosine", "cos"}, {"sine", "sin"}},
               {{"x", "x"}, {"out", "x"}, {"cos", "cos"}, {"sin", "sin"}},
               {{"tp8", 1}}, {{"tp8", 1}}});
  c.push_back({"attention prefix", "causal_sdpa_prefix_spm_d256_f16", "attention",
               {{"q", spm({512, 256}, "sharded")}, {"k", spm({512, 256}, "sharded")},
                {"v", spm({256, 512}, "sharded", 0)}, {"o", spm({512, 256}, "sharded")}},
               {{"query", "q"}, {"key", "k"}, {"value", "v"}, {"out", "o"}},
               {{"query", "q"}, {"key", "k"}, {"value", "v"}, {"out", "o"}},
               {{"kv_length", 512}}, {{"kv_length", 512}}});
  c.push_back({"attention masked m16", "sdpa_masked_m16d256_f16", "attention",
               {{"q", spm({16, 256}, "sharded")}, {"o", spm({16, 256}, "sharded")},
                {"mask", spm({16, 528}, "replicated")},
                {"kc", cache({8, 784, 256}, "causal_k_cache")},
                {"vc", cache({8, 784, 256}, "causal_v_cache")}},
               {{"query", "q"}, {"out", "o"}, {"mask", "mask"}, {"key_cache", "kc"},
                {"value_cache", "vc"}},
               {{"query", "q"}, {"out", "o"}, {"mask", "mask"}, {"key_cache", "kc"},
                {"value_cache", "vc"}},
               {{"kv_length", 528}}, {{"kv_length", 528}}});
  c.push_back({"kv_cache_insert causal key", "kcache_insert_d256_f16", "kv_cache_insert",
               {{"in", spm({512, 256}, "sharded")}, {"kc", cache({8, 784, 256}, "causal_k_cache")}},
               {{"input", "in"}, {"cache", "kc"}}, {{"input", "in"}, {"cache", "kc"}},
               {{"position", 0}}, {{"position", 0}}});
  c.push_back({"kv_cache_insert staged value", "vcache_insert_staged_d256_f16", "kv_cache_insert",
               {{"in", spm({512, 256}, "sharded")}, {"st", spm({256, 512}, "sharded", 0)},
                {"vc", cache({8, 784, 256}, "causal_v_cache")}},
               {{"input", "in"}, {"value_spm", "st"}, {"cache", "vc"}},
               {{"input", "in"}, {"value_spm", "st"}, {"cache", "vc"}},
               {{"position", 0}}, {{"position", 0}}});
  c.push_back({"all_reduce residual", "all_reduce_residual", "all_reduce",
               {{"p", spm({512, 2048}, "partial")}, {"r", spm({512, 2048}, "replicated")},
                {"o", spm({512, 2048}, "replicated")}},
               {{"input", "p"}, {"residual", "r"}, {"out", "o"}},
               {{"x", "p"}, {"residual", "r"}, {"out", "o"}},
               {{"with_residual", 1}}, {}});
  c.push_back({"all_reduce plain", "all_reduce_residual", "all_reduce",
               {{"p", spm({16, 1024}, "partial")}, {"o", spm({16, 1024}, "replicated")}},
               {{"input", "p"}, {"out", "o"}}, {{"x", "p"}, {"out", "o"}},
               {{"with_residual", 0}}, {}});
  // all_reduce + layernorm: the semantic out is the catalog x, and the
  // semantic x is the catalog partial (a simultaneous, chained rename).
  for (const char* residual_distribution : {"replicated", "sequence_shard"}) {
    Spec residual = spm({128, 1024}, residual_distribution, residual_distribution ==
                        std::string("sequence_shard") ? 0 : -1);
    c.push_back({std::string("all_reduce layernorm residual ") + residual_distribution,
                 "all_reduce_residual_layernorm", "all_reduce",
                 {{"p", spm({128, 1024}, "partial")}, {"r", residual},
                  {"x", spm({128, 1024}, "sequence_shard", 0)},
                  {"n", spm({128, 1024}, "replicated")}, {"g", spm({1024}, "replicated")},
                  {"b", spm({1024}, "replicated")}},
                 {{"partial", "p"}, {"residual", "r"}, {"x", "x"}, {"norm", "n"},
                  {"gamma", "g"}, {"beta", "b"}},
                 {{"x", "p"}, {"residual", "r"}, {"out", "x"}, {"norm", "n"},
                  {"gamma", "g"}, {"beta", "b"}},
                 {{"counter", 50}}, {{"counter", 50}}, {{"epsilon", 1e-6}}});
  }
  c.push_back({"all_gather", "all_gather_f16", "all_gather",
               {{"in", spm({868, 128}, "sharded")}, {"o", spm({868, 1024}, "replicated")}},
               {{"input", "in"}, {"out", "o"}}, {{"x", "in"}, {"out", "o"}}});
  return c;
}

Spec history(std::vector<std::int64_t> shape) {
  Spec s = spm(std::move(shape)); s.state = "history"; return s;
}
Spec ddr_row(std::vector<std::int64_t> shape) {
  Spec s; s.shape = std::move(shape); s.distribution = "shared"; s.storage = "ddr"; return s;
}

// Families whose catalog entries are single kernels or selected by operands;
// fixtures follow tests/rpu/tp_{operator,integration,state}_contract_test.cpp.
std::vector<Case> remaining_family_cases() {
  std::vector<Case> c;
  // bmm: the transposes and geometry choose a specialized tile.
  struct Bmm { const char* entry; std::int64_t m, n, k; int ta, tb; };
  for (const Bmm& b : {Bmm{"bmm_fp16_spm_64x64x128_rowrow", 64, 64, 128, 0, 1},
                       Bmm{"bmm_fp16_spm_128x128x64_colcol", 128, 128, 64, 1, 0},
                       Bmm{"bmm_fp16_spm_64x128x64_rowcol", 64, 128, 64, 0, 0},
                       Bmm{"bmm_fp16_spm_64x128x128_rowcol", 64, 128, 128, 0, 0}}) {
    std::vector<std::int64_t> a = b.ta ? std::vector<std::int64_t>{2, b.k, b.m}
                                       : std::vector<std::int64_t>{2, b.m, b.k};
    std::vector<std::int64_t> bb = b.tb ? std::vector<std::int64_t>{2, b.n, b.k}
                                        : std::vector<std::int64_t>{2, b.k, b.n};
    c.push_back({std::string("bmm ") + b.entry, b.entry, "bmm",
                 {{"a", spm(a)}, {"b", spm(bb)}, {"out", spm({2, b.m, b.n})}},
                 {{"a", "a"}, {"b", "b"}, {"out", "out"}}, {{"a", "a"}, {"b", "b"}, {"out", "out"}},
                 {{"transpose_a", b.ta}, {"transpose_b", b.tb}},
                 {{"transpose_a", b.ta}, {"transpose_b", b.tb}}});
  }
  for (auto [cols, entry] : {std::pair{64, "bcast_rows_f16"}, std::pair{4096, "bcast_tiles_f16"}})
    c.push_back({std::string("broadcast ") + entry, entry, "broadcast",
                 {{"src", spm({16, cols})}, {"out", spm({16, cols})}, {"a", spm({16})}},
                 {{"src", "src"}, {"out", "out"}, {"a", "a"}},
                 {{"src", "src"}, {"out", "out"}, {"a", "a"}}});
  c.push_back({"causal_conv1d prefill", "causal_depthwise_conv1d_prefill_f16", "causal_conv1d",
               {{"pad", spm({67, 256})}, {"carry", history({3, 256})}, {"halo", spm({8, 3, 256})},
                {"w", ddr_row({8, 4, 256})}},
               {{"pad", "pad"}, {"carry", "carry"}, {"halo", "halo"}, {"weight", "w"}},
               {{"pad", "pad"}, {"carry", "carry"}, {"halo", "halo"}, {"weight", "w"}}});
  c.push_back({"causal_conv1d step", "causal_depthwise_conv1d_step_f16", "causal_conv1d",
               {{"state", history({4, 256})}, {"in", spm({256})}, {"w", ddr_row({8, 4, 256})}},
               {{"state", "state"}, {"input", "in"}, {"weight", "w"}},
               {{"state", "state"}, {"input", "in"}, {"weight", "w"}}});
  {
    Case x{"causal_conv1d fused prefill", "conv_silu_prefill_f16", "causal_conv1d"};
    for (const std::string path : {"q", "k", "v"}) {
      x.specs.emplace(path + "pad", spm({67, 256}));
      x.specs.emplace(path + "out", spm({64, 256}));
      x.specs.emplace(path + "weight", ddr_row({8, 4, 256}));
      for (const auto* port : {"pad", "out", "weight"}) {
        x.catalog_binds.push_back({path + port, path + port});
        x.family_binds.push_back({path + port, path + port});
      }
    }
    for (const auto* name : {"history", "history_out"}) {
      x.specs.emplace(name, history({3, 4, 256}));
      x.catalog_binds.push_back({name, name});
      x.family_binds.push_back({name, name});
    }
    x.catalog_ints = x.family_ints = {{"valid_rows", 1}, {"initial", 0}};
    c.push_back(x);
  }
  {
    Case x{"copy_prefix", "copy_physical_prefix_f16", "copy_prefix"};
    for (unsigned layer = 0; layer < 6; ++layer)
      for (const std::string key : {"src_k", "src_v", "dst_k", "dst_v"}) {
        const auto name = key + std::to_string(layer);
        x.specs.emplace(name, cache({8, 128, 256}, key.back() == 'k' ? "causal_k_cache"
                                                                     : "causal_v_cache"));
        x.catalog_binds.push_back({name, name});
        x.family_binds.push_back({name, name});
      }
    x.catalog_ints = x.family_ints = {{"prefix_len", 65}, {"bucket_len", 128}, {"capacity", 128}};
    c.push_back(x);
  }
  c.push_back({"cumsum", "cumsum_fp16", "cumsum", {{"src", spm({2, 64})}, {"out", spm({2, 64})}},
               {{"src", "src"}, {"out", "out"}}, {{"src", "src"}, {"out", "out"}}});
  c.push_back({"unit_tril_inverse", "unit_tril_inv_fp16", "unit_tril_inverse",
               {{"src", spm({2, 64, 64})}, {"out", spm({2, 64, 64})}},
               {{"src", "src"}, {"out", "out"}}, {{"src", "src"}, {"out", "out"}}});
  for (int m : {16, 32}) {
    const auto rows = std::to_string(m);
    c.push_back({"down_allreduce_norm m" + rows,
                 "down_allreduce_rmsnorm_m" + rows + "n1024k256_acc32", "down_allreduce_norm",
                 {{"src", spm({m, 256}, "sharded")}, {"partial", spm({m, 1024}, "partial")},
                  {"residual", spm({m, 1024}, "replicated")}, {"stream", spm({m, 1024}, "replicated")},
                  {"norm", spm({m, 1024}, "replicated")}, {"w", tp8_weight(1024, 256)},
                  {"gate_row", ddr_row({3, 1024})}, {"next_row", ddr_row({3, 1024})}},
                 {{"src", "src"}, {"partial", "partial"}, {"residual", "residual"},
                  {"stream", "stream"}, {"norm", "norm"}, {"weight", "w"},
                  {"gate_row", "gate_row"}, {"next_row", "next_row"}},
                 {{"x", "src"}, {"partial", "partial"}, {"residual", "residual"},
                  {"stream", "stream"}, {"norm", "norm"}, {"w", "w"},
                  {"gate_row", "gate_row"}, {"next_row", "next_row"}},
                 {}, {}, {{"epsilon", 1e-6}}});
  }
  {
    Spec ids; ids.shape = {64}; ids.distribution = "shared"; ids.storage = "ddr"; ids.dtype = "int32";
    Spec vocab = ddr_row({256277, 2048}); vocab.immutable = true;
    c.push_back({"embedding tp8", "gather_embedding_tp8_f16", "embedding",
                 {{"ids", ids}, {"vocab", vocab}, {"out", spm({8, 2048}, "sharded", 0)}},
                 {{"ids", "ids"}, {"vocab", "vocab"}, {"out", "out"}},
                 {{"ids", "ids"}, {"vocab", "vocab"}, {"out", "out"}}});
  }
  c.push_back({"expand_kv", "qkv_expand_kv_f16", "expand_kv",
               {{"in", spm({64, 640})}, {"out", spm({64, 1024})}},
               {{"input", "in"}, {"out", "out"}}, {{"x", "in"}, {"out", "out"}}});
  c.push_back({"fill", "fill_f16", "fill", {{"out", spm({17})}}, {{"out", "out"}}, {{"out", "out"}},
               {}, {}, {{"value", -0.5}}});
  {
    Case x{"gated_delta_rule_prefill", "gdr_prefill_f16", "gated_delta_rule_prefill"};
    for (const auto* port : {"q", "k", "v", "z", "out"}) x.specs.emplace(port, spm({64, 256}, "sharded"));
    for (const auto* port : {"a", "b"}) x.specs.emplace(port, spm({64, 16}, "sharded"));
    for (const auto* port : {"dt", "neg_a"}) x.specs.emplace(port, spm({16}, "sharded"));
    x.specs.emplace("norm_weight", spm({128}, "replicated"));
    Spec state = spm({2, 128, 128}, "sharded", 0); state.state = "state";
    x.specs.emplace("state", state);
    x.specs.emplace("scratch", spm({static_cast<std::int64_t>(
        xinfer::rpu::fusion::gdr_scratch_bytes_for_rows(64) / 2)}));
    for (const auto& item : x.specs) {
      x.catalog_binds.push_back({item.first, item.first});
      x.family_binds.push_back({item.first, item.first});
    }
    x.catalog_ints = x.family_ints = {{"initial", 1}, {"valid_rows", 64}};
    c.push_back(x);
  }
  c.push_back({"l2norm", "l2norm_f16", "l2norm",
               {{"in", spm({16, 128})}, {"out", spm({16, 128})}},
               {{"input", "in"}, {"out", "out"}}, {{"x", "in"}, {"out", "out"}}});
  for (int m : {16, 32}) {
    const auto rows = std::to_string(m);
    c.push_back({"qk_norm_rope m" + rows, "qk_rmsnorm_mrope_m" + rows + "d256_f16", "qk_norm_rope",
                 {{"x", spm({m, 256}, "sharded", 1)}, {"out", spm({m, 256}, "sharded", 1)},
                  {"w", spm({256}, "replicated")}, {"cos", table(64, 32, "ddr")},
                  {"sin", table(64, 32, "ddr")}},
                 {{"x", "x"}, {"out", "out"}, {"weight", "w"}, {"cosine", "cos"}, {"sine", "sin"}},
                 {{"x", "x"}, {"out", "out"}, {"weight", "w"}, {"cos", "cos"}, {"sin", "sin"}},
                 {{"position_offset", 5}}, {{"position_offset", 5}}});
  }
  c.push_back({"scale", "scalar_mul_f16", "scale",
               {{"in", spm({17})}, {"out", spm({17})}},
               {{"input", "in"}, {"out", "out"}}, {{"x", "in"}, {"out", "out"}},
               {}, {}, {{"value", -1.0}}});
  c.push_back({"slice_last", "slice_last_f16", "slice_last",
               {{"in", spm({64, 16})}, {"out", spm({64, 2})}},
               {{"input", "in"}, {"out", "out"}}, {{"x", "in"}, {"out", "out"}}});
  for (auto [ch, entry] : {std::pair{1, "transpose_nbc_c1_f16"},
                           std::pair{128, "transpose_nbc_c128_f16"}})
    c.push_back({std::string("transpose ") + entry, entry, "transpose",
                 {{"in", spm({64, 2, ch})}, {"out", spm({2, 64, ch})}},
                 {{"input", "in"}, {"out", "out"}}, {{"x", "in"}, {"out", "out"}}});
  return c;
}

void rejections() {
  Pool pool;
  std::map<std::string, Spec> s{{"x", spm({16, 4096}, "replicated")}, {"w", tp8_weight(320, 4096)},
                                {"bias", spm({320}, "sharded")}, {"out", spm({16, 320}, "sharded")}};
  rejects([&] {
    lower(request(pool, "gemm", s, {{"x", "x"}, {"w", "w"}, {"bias", "bias"}, {"out", "out"}}));
  }, "no column-parallel kernel for (N,K)=(320,4096)");
  std::map<std::string, Spec> t{{"x", spm({16, 1024}, "replicated")}, {"w", tp8_weight(256, 1024)},
                                {"out", spm({16, 256}, "sharded")}};
  rejects([&] {
    lower(request(pool, "gemm", t, {{"x", "x"}, {"w", "w"}, {"out", "out"}}, {{"activation", 2}}));
  }, "activation none or gelu");
  rejects([&] {
    lower(request(pool, "gemm", t, {{"x", "x"}, {"w", "w"}, {"out", "out"}}, {{"parallel", 7}}));
  }, "parallel must be column or row");
  // Each rejection below uses a fresh pool: pooled tensors are keyed by name.
  Pool norm_pool;
  std::map<std::string, Spec> u{{"nx", spm({512, 256})}, {"nout", spm({512, 256})},
                                {"nw", spm({256})}, {"nz", spm({512, 256})}};
  rejects([&] {
    lower(request(norm_pool, "rmsnorm", u, {{"x", "nx"}, {"out", "nout"}, {"weight", "nw"}},
                  {{"mode", 9}}));
  }, "mode must be 0");
  // Unknown semantic arguments still fail in the catalog lowerer.
  rejects([&] {
    lower(request(norm_pool, "rmsnorm", u,
                  {{"x", "nx"}, {"out", "nout"}, {"weight", "nw"}, {"zz", "nz"}},
                  {}, {{"epsilon", 1e-6}}));
  }, "unsupported tensor zz");
  // A rejected family request is left unchanged for the caller.
  auto r = request(pool, "gemm", s, {{"x", "x"}, {"w", "w"}, {"bias", "bias"}, {"out", "out"}});
  const auto before = r.tensors.size();
  try { tp::resolve_family(r); } catch (const std::invalid_argument&) {}
  check(r.op == "gemm" && r.tensors.size() == before && r.tensors.count("w"),
        "rejected family request was modified");
  // Catalog names pass through untouched.
  auto legacy = request(norm_pool, "rmsnorm_f16", u, {{"x", "nx"}, {"out", "nout"}, {"weight", "nw"}});
  check(!tp::resolve_family(legacy) && legacy.op == "rmsnorm_f16", "catalog request rewritten");
  check(tp::is_family("gemm") && !tp::is_family("qkv_qgate_n1024k2048"), "family table");
  auto silu = request(norm_pool, "silu_and_mul", u, {{"x", "nx"}, {"out", "nout"}});
  check(tp::resolve_family(silu) && silu.op == "silu_mul_fp16", "silu_and_mul family");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    unsigned cases = 0;
    for (auto list : {gemm_cases(), projection_cases(), norm_and_pointwise_cases(),
                      attention_and_comm_cases(), remaining_family_cases()})
      for (const auto& c : list) { equivalent(c); ++cases; }
    rejections();
    // --pairs prints each proven family -> catalog resolution for the host
    // test that checks the documented family table (tests/perf/rpu_perf.py).
    if (argc == 2 && std::string(argv[1]) == "--pairs") {
      for (const auto& pair : resolved_pairs) std::cout << pair << "\n";
      return 0;
    }
    std::cout << "tp_family_test: " << cases << " family cases, " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "tp_family_test failed: " << e.what() << "\n";
    return 1;
  }
}
