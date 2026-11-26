// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Family entry points: rewrite a family request into its catalog entry.
#include "common/rpu/tp_family.hpp"

#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace xinfer::rpu::tp {
namespace {

[[noreturn]] void fail(const std::string& family, const std::string& message) {
  throw std::invalid_argument("RPU TP " + family + ": " + message);
}

const Tensor* find(const Request& r, const char* name) {
  const auto found = r.tensors.find(name);
  return found == r.tensors.end() ? nullptr : &found->second;
}

const Tensor& need(const Request& r, const std::string& family, const char* name) {
  const auto* t = find(r, name);
  if (!t) fail(family, std::string("requires tensor ") + name);
  return *t;
}

std::int64_t dim(const Tensor& t, std::size_t axis, const std::string& family,
                 const char* name) {
  if (t.shape.size() <= axis)
    fail(family, std::string(name) + " requires rank >= " + std::to_string(axis + 1));
  return t.shape[axis];
}

bool has_integer(const Request& r, const char* name) { return r.integers.count(name) != 0; }

// Selectors consumed by the active resolve_family call; they choose the
// kernel at prepare time and cannot be rebound.
thread_local std::set<std::string>* active_selectors = nullptr;

// Remove a family-only selector so the legacy lowerer does not see it.
std::int64_t take_integer(Request& r, const char* name, std::int64_t fallback) {
  if (active_selectors) active_selectors->insert(name);
  const auto found = r.integers.find(name);
  if (found == r.integers.end()) return fallback;
  const auto value = found->second;
  r.integers.erase(found);
  return value;
}

// Renames performed by the active resolve_family call (caller -> catalog).
thread_local std::map<std::string, std::string>* active_renames = nullptr;

// Rename semantic tensor arguments to the names the legacy lowerer reads.
// The map is applied simultaneously, so chains such as {x -> partial,
// out -> x} are valid. A target may only collide with an argument that is
// itself renamed away; a rejected request is left untouched.
void rename(Request& r, std::initializer_list<std::pair<const char*, const char*>> map) {
  std::set<std::string> moving;
  for (const auto& [from, to] : map)
    if (std::string(from) != to && r.tensors.count(from)) moving.insert(from);
  std::set<std::string> targets;
  for (const auto& [from, to] : map) {
    if (!moving.count(from)) continue;
    if (!targets.insert(to).second || (r.tensors.count(to) && !moving.count(to)))
      throw std::invalid_argument("RPU TP " + r.op + ": arguments " + from + " and " + to +
                                  " name the same parameter");
  }
  std::vector<std::pair<std::string, Tensor>> moved;
  for (const auto& [from, to] : map) {
    if (!moving.count(from)) continue;
    auto found = r.tensors.find(from);
    moved.emplace_back(to, std::move(found->second));
    r.tensors.erase(found);
    if (active_renames) (*active_renames)[from] = to;
  }
  for (auto& [to, tensor] : moved) r.tensors.emplace(to, std::move(tensor));
}

std::string selector_name(const std::map<std::int64_t, const char*>& names,
                          std::int64_t value) {
  const auto found = names.find(value);
  return found == names.end() ? std::to_string(value) : found->second;
}

// Selector encodings shared by the Python API.
constexpr std::int64_t kParallelColumn = 0, kParallelRow = 1;
constexpr std::int64_t kActivationNone = 0, kActivationGelu = 1, kActivationSilu = 2;
const std::map<std::int64_t, const char*> kActivationNames{
    {kActivationNone, "none"}, {kActivationGelu, "gelu"}, {kActivationSilu, "silu"}};

// gemm: out = x @ w.T + bias, TP8 column- or row-parallel.
//   column: x replicated, out sharded along axis 1, w [8,N,K] tp8_nk16.
//   row:    x sharded along axis 1, out partial (no implicit reduction).
// The kernel follows from (rows, N, K, bias, activation); numerics-changing
// fusions (activation, output scale) are explicit selectors.
void resolve_gemm(Request& r) {
  const std::string f = "gemm";
  const auto& x = need(r, f, "x");
  const auto& w = need(r, f, "w");
  const auto rows = dim(x, 0, f, "x");
  const auto parallel = take_integer(r, "parallel", kParallelColumn);
  const auto activation = take_integer(r, "activation", kActivationNone);
  const auto scale = take_integer(r, "output_scale", 0);
  const bool bias = find(r, "bias") != nullptr;
  if (parallel != kParallelColumn && parallel != kParallelRow)
    fail(f, "parallel must be column or row");
  if (scale != 0 && scale != 1) fail(f, "output_scale must be none or quarter");

  // Core0 input projections over an nk16 DDR weight (single-core layout).
  if (w.layout == "nk16") {
    if (activation != kActivationNone) fail(f, "nk16 projections have no fused activation");
    const auto n = dim(w, 0, f, "w"), k = dim(w, 1, f, "w");
    if (rows != 16 && rows != 32) fail(f, "nk16 projections require x rows 16 or 32");
    const std::string m = std::to_string(rows);
    if (n == 1024 && k == 64 && scale)
      r.op = "gemm_ddr_bias_scale_m" + m + "n1024k64_acc32";
    else if (n == 64 && k == 1024 && !scale)
      r.op = "gemm_ddr_bias_m" + m + "n64k1024_acc32";
    else
      fail(f, "nk16 projections support (N,K)=(1024,64) with output_scale=quarter "
              "or (64,1024); got (" + std::to_string(n) + "," + std::to_string(k) + ")");
    rename(r, {{"x", "input"}, {"w", "weight"}});
    return;
  }
  if (scale) fail(f, "output_scale applies to nk16 input projections only");
  if (w.shape.size() != 3) fail(f, "w must be a tp8_nk16 weight [8,N,K]");
  const auto n = w.shape[1], k = w.shape[2];
  const auto shape = "(N,K)=(" + std::to_string(n) + "," + std::to_string(k) + ")";

  if (parallel == kParallelRow) {
    if (activation != kActivationNone) fail(f, "row-parallel gemm has no fused activation");
    if (!bias && n == 2048 && (k == 256 || k == 768)) {
      r.op = k == 256 ? "o_n2048k256" : "down_n2048k768";
      rename(r, {{"x", "src"}, {"w", "weight"}});
    } else if (!bias && n == 1024 && k == 256 && (rows == 16 || rows == 32)) {
      r.op = "gemm_ddr_row_m" + std::to_string(rows) + "n1024k256_acc32";
      rename(r, {{"x", "src"}, {"w", "weight"}});
    } else if (bias && n == 1024 && (k == 128 || k == 512)) {
      r.op = k == 128 ? "o_n1024k128" : "down_n1024k512";
      rename(r, {{"x", "src"}, {"w", "weight"}});
    } else if (bias && n == 2048 && k == 512) {
      r.op = "down_n2048k512";
      rename(r, {{"x", "src"}, {"w", "weight"}});
    } else {
      fail(f, "no row-parallel kernel for " + shape + (bias ? " with bias" : " without bias") +
              "; supported: (2048,256|768) without bias, (1024,256) rows 16/32 without bias, "
              "(1024,128|512) or (2048,512) with bias");
    }
    return;
  }

  // Column-parallel projections.
  const bool gelu = activation == kActivationGelu;
  if (activation != kActivationNone && !gelu)
    fail(f, "column-parallel gemm supports activation none or gelu, got " +
            selector_name(kActivationNames, activation));
  if (bias && n == 512 && (k == 1024 || k == 4096)) {
    r.op = k == 1024 ? (gelu ? "up_gelu_n512k1024" : "up_n512k1024")
                     : (gelu ? "fc1_gelu_n512k4096" : "fc1_n512k4096");
  } else if (!gelu && bias && n == 128 && k == 1536) {
    r.op = "patch_embed_n128k1536";
  } else if (!gelu && !bias && n == 256 && k == 1024 && (rows == 16 || rows == 32)) {
    r.op = "gemm_ddr_col_m" + std::to_string(rows) + "n256k1024_acc32";
  } else {
    fail(f, "no column-parallel kernel for " + shape + (bias ? " with bias" : " without bias") +
            (gelu ? " and gelu" : "") +
            "; supported: (512,1024|4096) with bias [and gelu], (128,1536) with bias, "
            "(256,1024) rows 16/32 without bias");
  }
  rename(r, {{"w", "weight"}});
}

// qkv_proj: fused TP8 Q/K/V projection, output heads split per core.
//   Language full attention:  out_q, out_qgate, out_k, out_v      (w [8,1024,2048])
//   Language linear attention: out_q, out_k, out_v, out_z, out_b, out_a (w [8,1056,2048])
//   Vision: out_q, out_k, out_v with bias, optional rope=(cos, sin)  (w [8,384,1024])
//   Action: one planes output [4,M,256] with qk_weight and rope       (w [4,8,256,1024])
// Language rows 448/512 select the K128 programs inside the gemm lowerer.
void resolve_qkv_proj(Request& r) {
  const std::string f = "qkv_proj";
  const auto& x = need(r, f, "x");
  const auto& w = need(r, f, "w");
  const bool rope = find(r, "cos") || find(r, "sin");
  if (rope && !(find(r, "cos") && find(r, "sin"))) fail(f, "rope requires both cos and sin");
  if (find(r, "planes")) {  // Action QKV + QK RMSNorm + RoPE.
    const auto rows = dim(x, 0, f, "x");
    if (rows != 16 && rows != 32) fail(f, "action qkv requires x rows 16 or 32");
    if (!rope || !find(r, "qk_weight")) fail(f, "action qkv requires cos, sin and qk_weight");
    r.op = "qknorm_qkv_rope_m" + std::to_string(rows) + "k1024_acc32";
    rename(r, {{"w", "weight"}, {"cos", "cosine"}, {"sin", "sine"}});
    return;
  }
  if (w.shape.size() != 3) fail(f, "w must be a tp8_nk16 weight [8,N,K]");
  const auto n = w.shape[1];
  if (n == 384) {  // Vision.
    if (!find(r, "bias")) fail(f, "vision qkv requires bias");
    if (rope) {
      r.op = "qkv_rope_bias_n384k1024";
      rename(r, {{"w", "weight"}, {"out_q", "q"}, {"out_k", "k"}, {"out_v", "v"},
                 {"cos", "cosine"}, {"sin", "sine"}});
    } else {
      r.op = "qkv_bias_n384k1024";
      rename(r, {{"x", "src"}, {"w", "weight"}});
    }
    return;
  }
  if (rope) fail(f, "language qkv applies rope separately (rope op)");
  if (n == 1024 && find(r, "out_qgate")) r.op = "qkv_qgate_n1024k2048";
  else if (n == 1056 && find(r, "out_z")) r.op = "qkv_zba_n1056k2048";
  else if (n == 640) r.op = "tp_qkv_compact_f16";
  else
    fail(f, "no qkv kernel for N=" + std::to_string(n) +
            "; supported: 1024 (out_q,out_qgate,out_k,out_v), 1056 (out_q,out_k,out_v,out_z,"
            "out_b,out_a), 640 (compact out), 384 (vision, with bias)");
  rename(r, {{"x", "src"}, {"w", "weight"}});
}

// gemm_glu: out = act(x @ Wg.T) * (x @ Wu.T), column-parallel.
//   Language: one interleaved w [8,1536,2048] -> out_gate, out_up (activation
//   applied by the caller's separate silu_and_mul; the projection writes both).
//   Action:   w_gate and w_up [8,256,1024] -> fused SiLU product out [M,256].
void resolve_gemm_glu(Request& r) {
  const std::string f = "gemm_glu";
  const auto& x = need(r, f, "x");
  const auto rows = dim(x, 0, f, "x");
  const auto activation = take_integer(r, "activation", kActivationSilu);
  if (find(r, "w_gate") || find(r, "w_up")) {
    if (activation != kActivationSilu) fail(f, "fused gate/up supports activation silu only");
    if (rows != 16 && rows != 32) fail(f, "fused gate/up requires x rows 16 or 32");
    r.op = "gate_up_silu_m" + std::to_string(rows) + "n256k1024_acc32";
    rename(r, {{"w_gate", "gate_weight"}, {"w_up", "up_weight"}});
    return;
  }
  if (activation != kActivationNone)
    fail(f, "the interleaved language projection returns gate and up separately; "
            "pass activation=none and apply silu_and_mul");
  r.op = "gateup_n1536k2048";
  rename(r, {{"x", "src"}, {"w", "weight"}});
}

// rmsnorm: out = rmsnorm(x) * weight, or the adaptive form with modulation.
//   mode 0: x * inverse then * weight (default); mode 1: coefficient-first
//   (weight * inverse first, H=2048 only); mode 2: gated (gated delta rule
//   output norm, x proven in [-1,1]).
void resolve_rmsnorm(Request& r) {
  const std::string f = "rmsnorm";
  const auto& x = need(r, f, "x");
  const auto mode = take_integer(r, "mode", 0);
  if (find(r, "modulation")) {
    if (mode != 0) fail(f, "adaptive rmsnorm has no mode selector");
    const auto rows = dim(x, 0, f, "x");
    if (rows != 16 && rows != 32) fail(f, "adaptive rmsnorm requires x rows 16 or 32");
    r.op = "adaptive_rmsnorm_m" + std::to_string(rows) + "h1024_f16";
    return;
  }
  if (mode == 0) r.op = "rmsnorm_f16";
  else if (mode == 1) r.op = "rmsnorm_coeff_h2048_f16";
  else if (mode == 2) r.op = "rmsnorm_gdn_f16";
  else fail(f, "mode must be 0 (default), 1 (coefficient-first) or 2 (gated)");
}

// qk_norm_rope: per-head RMSNorm then multi-axis RoPE on a [M,256] head.
void resolve_qk_norm_rope(Request& r) {
  const std::string f = "qk_norm_rope";
  const auto rows = dim(need(r, f, "x"), 0, f, "x");
  if (rows != 16 && rows != 32) fail(f, "requires x rows 16 or 32");
  r.op = "qk_rmsnorm_mrope_m" + std::to_string(rows) + "d256_f16";
  rename(r, {{"cos", "cosine"}, {"sin", "sine"}});
}

// layernorm and l2norm keep a single kernel each; the family name aligns the
// argument names with the CUDA family (x, weight, bias, out, epsilon).
void resolve_layernorm(Request& r) {
  r.op = "layernorm_f16";
  rename(r, {{"x", "input"}});
}
void resolve_l2norm(Request& r) {
  r.op = "l2norm_f16";
  rename(r, {{"x", "input"}});
}

// silu_and_mul: out = silu(x) * y; sigmoid_mul: out = sigmoid(gate) * value.
void resolve_silu_and_mul(Request& r) { r.op = "silu_mul_fp16"; }
void resolve_sigmoid_mul(Request& r) {
  const std::string f = "sigmoid_mul";
  const auto rows = dim(need(r, f, "gate"), 0, f, "gate");
  if (rows != 16 && rows != 32) fail(f, "requires gate rows 16 or 32");
  r.op = "sigmoid_mul_m" + std::to_string(rows) + "d256_f16";
}

// activation: elementwise gelu / sigmoid / exp / silu / softplus.
//   function: 0 gelu_tanh, 1 gelu_erf, 2 sigmoid, 3 exp, 4 silu, 5 softplus.
void resolve_activation(Request& r) {
  const std::string f = "activation";
  const auto function = take_integer(r, "function", -1);
  rename(r, {{"x", "src"}});
  if (function == 0 || function == 1) {
    r.op = "gelu_f16";
    r.integers["mode"] = function == 0 ? 1 : 0;
  } else if (function >= 2 && function <= 5) {
    r.op = "unary_f16";
    r.integers["operation"] = function - 2;
  } else {
    fail(f, "function must be gelu_tanh, gelu_erf, sigmoid, exp, silu or softplus");
  }
}

// rope: rotary embedding on [tokens, heads, head_dim].
void resolve_rope(Request& r) {
  r.op = "rope_f16";
  rename(r, {{"cos", "cosine"}, {"sin", "sine"}});
}

// attention: query against key/value. The kernel follows from the operands:
//   key/value SPM (value transposed [256,M]) -> causal prefix attention;
//   key_cache/value_cache + mask  -> masked attention, rows 16 or 32;
//   key_cache/value_cache         -> causal cache attention;
//   q/k/v [T,128] with length0..2 -> noncausal varlen (SPM or cache).
void resolve_attention(Request& r) {
  const std::string f = "attention";
  if (has_integer(r, "length0")) {
    if (find(r, "k") && find(r, "v")) {
      r.op = "flash_attention_spm_d384";
    } else if (find(r, "key_cache0") && find(r, "value_cache0")) {
      r.op = "flash_attention_varlen_f16";
    } else {
      fail(f, "varlen attention requires k and v, or key_cache0..2 and value_cache0..2");
    }
    return;
  }
  const auto& q = need(r, f, "query");
  const auto rows = dim(q, 0, f, "query");
  if (find(r, "key") && find(r, "value")) {
    r.op = "causal_sdpa_prefix_spm_d256_f16";
  } else if (find(r, "key_cache") && find(r, "value_cache")) {
    if (find(r, "mask")) {
      if (rows != 16 && rows != 32) fail(f, "masked attention requires query rows 16 or 32");
      r.op = "sdpa_masked_m" + std::to_string(rows) + "d256_f16";
    } else {
      r.op = "causal_sdpa_d256_f16";
    }
  } else {
    fail(f, "requires key and value, or key_cache and value_cache");
  }
}

// kv_cache_insert: write input rows into a K or V cache at position.
//   The cache layout selects the kernel: vision_k/v_cache (D128) or
//   causal_k/v_cache (D256); value_spm adds the staged transposed copy.
void resolve_kv_cache_insert(Request& r) {
  const std::string f = "kv_cache_insert";
  const auto& cache = need(r, f, "cache");
  const bool staged = find(r, "value_spm") != nullptr;
  if (cache.layout == "vision_k_cache" || cache.layout == "vision_v_cache") {
    if (staged) fail(f, "value_spm applies to causal value caches only");
    r.op = cache.layout == "vision_k_cache" ? "kcache_insert_f16" : "vcache_insert_f16";
  } else if (cache.layout == "causal_k_cache") {
    if (staged) fail(f, "value_spm applies to causal value caches only");
    r.op = "kcache_insert_d256_f16";
  } else if (cache.layout == "causal_v_cache") {
    r.op = staged ? "vcache_insert_staged_d256_f16" : "vcache_insert_d256_f16";
  } else {
    fail(f, "cache layout must be vision_k_cache, vision_v_cache, causal_k_cache or "
            "causal_v_cache, got " + cache.layout);
  }
}

// all_reduce: out = sum over the TP group of the partial x (+ residual).
//   Fused forms: layernorm(gamma, beta) -> x_out and norm outputs.
void resolve_all_reduce(Request& r) {
  const std::string f = "all_reduce";
  if (find(r, "gamma") || find(r, "beta") || find(r, "norm")) {
    r.op = "all_reduce_residual_layernorm";
    rename(r, {{"x", "partial"}, {"out", "x"}});
    return;
  }
  r.op = "all_reduce_residual";
  rename(r, {{"x", "input"}});
  if (!has_integer(r, "with_residual"))
    r.integers["with_residual"] = find(r, "residual") ? 1 : 0;
}
void resolve_all_gather(Request& r) {
  r.op = "all_gather_f16";
  rename(r, {{"x", "input"}});
}

// all_reduce_residual_norm: Action down projection + all-reduce + residual
// gate + adaptive RMSNorm, rows 16 or 32.
void resolve_down_allreduce_norm(Request& r) {
  const std::string f = "down_allreduce_norm";
  const auto rows = dim(need(r, f, "x"), 0, f, "x");
  if (rows != 16 && rows != 32) fail(f, "requires x rows 16 or 32");
  r.op = "down_allreduce_rmsnorm_m" + std::to_string(rows) + "n1024k256_acc32";
  rename(r, {{"x", "src"}, {"w", "weight"}});
}

// residual_gate: out = residual + branch * gate, rows 16 or 32.
void resolve_residual_gate(Request& r) {
  const std::string f = "residual_gate";
  const auto rows = dim(need(r, f, "branch"), 0, f, "branch");
  if (rows != 16 && rows != 32) fail(f, "requires branch rows 16 or 32");
  r.op = "residual_gate_m" + std::to_string(rows) + "h1024_f16";
}

// euler_update: action += delta_t * velocity with a keep mask, rows 16 or 32.
void resolve_euler_update(Request& r) {
  const std::string f = "euler_update";
  const auto rows = dim(need(r, f, "action"), 0, f, "action");
  if (rows != 16 && rows != 32) fail(f, "requires action rows 16 or 32");
  r.op = "euler_update_m" + std::to_string(rows) + "d64_f16";
}

// binary: out = lhs (op) rhs. operation: 0 add, 1 sub, 2 mul, 3 max.
//   vectors_per_block selects the block-tuned add.
void resolve_binary(Request& r) {
  const std::string f = "binary";
  const auto operation = take_integer(r, "operation", -1);
  const bool tuned = has_integer(r, "vectors_per_block");
  static const char* names[] = {"binary_add_f16", "binary_sub_f16",
                                "binary_mul_f16", "binary_max_f16"};
  if (operation < 0 || operation > 3) fail(f, "operation must be add, sub, mul or max");
  if (tuned && operation != 0) fail(f, "vectors_per_block applies to add only");
  r.op = tuned ? "add_vpb_f16" : names[operation];
}

// broadcast: out = src (op) a along rows/columns/groups.
void resolve_broadcast(Request& r) {
  const std::string f = "broadcast";
  const auto& src = need(r, f, "src");
  const auto columns = dim(src, 1, f, "src");
  r.op = columns % 4096 == 0 ? "bcast_tiles_f16" : "bcast_rows_f16";
}

// embedding: gather vocab rows by ids; out sharded rows (TP8) or core0.
void resolve_embedding(Request& r) {
  const std::string f = "embedding";
  const auto& out = need(r, f, "out");
  r.op = out.distribution == "sharded" ? "gather_embedding_tp8_f16" : "gather_embedding_f16";
}

// causal_conv1d: depthwise causal convolution (K=4). Prefill takes pad +
// carry, one step takes input + state; the fused q/k/v + SiLU prefill takes
// qpad/kpad/vpad and history.
void resolve_causal_conv1d(Request& r) {
  const std::string f = "causal_conv1d";
  if (find(r, "qpad")) r.op = "conv_silu_prefill_f16";
  else if (find(r, "pad")) r.op = "causal_depthwise_conv1d_prefill_f16";
  else if (find(r, "state")) r.op = "causal_depthwise_conv1d_step_f16";
  else fail(f, "requires pad (prefill), state (step) or qpad/kpad/vpad (fused prefill)");
}

// bmm: out[B] = a[B] @ b[B] with transpose_a/transpose_b; the geometry and
// transposes select a specialized tile when one exists.
void resolve_bmm(Request& r) {
  const std::string f = "bmm";
  const auto& out = need(r, f, "out");
  const auto& a = need(r, f, "a");
  if (out.shape.size() != 3 || a.shape.size() != 3) fail(f, "a and out require rank 3");
  const auto batch = out.shape[0], m = out.shape[1], n = out.shape[2];
  const auto ta = r.integers.count("transpose_a") ? r.integers.at("transpose_a") : 0;
  const auto tb = r.integers.count("transpose_b") ? r.integers.at("transpose_b") : 0;
  const auto k = a.shape[ta ? 1 : 2];
  const bool even = batch >= 2 && batch <= 16 && batch % 2 == 0;
  r.op = "bmm_fp16_spm";
  if (m == 128 && n == 128 && k == 64 && ta == 1 && tb == 0 && batch == 2)
    r.op = "bmm_fp16_spm_128x128x64_colcol";
  else if (m == 64 && n == 64 && k == 128 && ta == 0 && tb == 1 && even)
    r.op = "bmm_fp16_spm_64x64x128_rowrow";
  else if (m == 64 && n == 128 && k == 64 && ta == 0 && tb == 0 && even)
    r.op = "bmm_fp16_spm_64x128x64_rowcol";
  else if (m == 64 && n == 128 && k == 128 && ta == 0 && tb == 0 && batch == 2)
    r.op = "bmm_fp16_spm_64x128x128_rowcol";
}

// gated_delta_rule_prefill: conv-free recurrence + gated RMSNorm per chunk.
void resolve_gated_delta_rule_prefill(Request& r) { r.op = "gdr_prefill_f16"; }
void resolve_cumsum(Request& r) { r.op = "cumsum_fp16"; }
void resolve_unit_tril_inverse(Request& r) { r.op = "unit_tril_inv_fp16"; }
void resolve_fill(Request& r) { r.op = "fill_f16"; }
void resolve_scale(Request& r) {
  r.op = "scalar_mul_f16";
  rename(r, {{"x", "input"}});
}

// Layout helpers keep one kernel each; family names describe the operation.
void resolve_slice_last(Request& r) {
  r.op = "slice_last_f16";
  rename(r, {{"x", "input"}});
}
void resolve_transpose(Request& r) {
  const std::string f = "transpose";
  const auto& x = need(r, f, "x");
  const auto c = dim(x, 2, f, "x");
  if (c != 1 && c != 128) fail(f, "x channel dimension must be 1 or 128");
  r.op = c == 1 ? "transpose_nbc_c1_f16" : "transpose_nbc_c128_f16";
  rename(r, {{"x", "input"}});
}
void resolve_expand_kv(Request& r) {
  r.op = "qkv_expand_kv_f16";
  rename(r, {{"x", "input"}});
}
void resolve_copy_prefix(Request& r) { r.op = "copy_physical_prefix_f16"; }

using Resolver = void (*)(Request&);
const std::map<std::string, Resolver>& families() {
  static const std::map<std::string, Resolver> table{
      {"gemm", resolve_gemm},
      {"qkv_proj", resolve_qkv_proj},
      {"gemm_glu", resolve_gemm_glu},
      {"bmm", resolve_bmm},
      {"rmsnorm", resolve_rmsnorm},
      {"layernorm", resolve_layernorm},
      {"l2norm", resolve_l2norm},
      {"qk_norm_rope", resolve_qk_norm_rope},
      {"silu_and_mul", resolve_silu_and_mul},
      {"sigmoid_mul", resolve_sigmoid_mul},
      {"activation", resolve_activation},
      {"rope", resolve_rope},
      {"attention", resolve_attention},
      {"kv_cache_insert", resolve_kv_cache_insert},
      {"copy_prefix", resolve_copy_prefix},
      {"expand_kv", resolve_expand_kv},
      {"all_reduce", resolve_all_reduce},
      {"all_gather", resolve_all_gather},
      {"down_allreduce_norm", resolve_down_allreduce_norm},
      {"residual_gate", resolve_residual_gate},
      {"euler_update", resolve_euler_update},
      {"binary", resolve_binary},
      {"broadcast", resolve_broadcast},
      {"scale", resolve_scale},
      {"fill", resolve_fill},
      {"embedding", resolve_embedding},
      {"causal_conv1d", resolve_causal_conv1d},
      {"gated_delta_rule_prefill", resolve_gated_delta_rule_prefill},
      {"cumsum", resolve_cumsum},
      {"unit_tril_inverse", resolve_unit_tril_inverse},
      {"slice_last", resolve_slice_last},
      {"transpose", resolve_transpose},
  };
  return table;
}

}  // namespace

bool is_family(const std::string& op) { return families().count(op) != 0; }

bool resolve_family(Request& request, std::map<std::string, std::string>* renamed,
                    std::set<std::string>* selectors) {
  const auto found = families().find(request.op);
  if (found == families().end()) return false;
  // Resolve a copy so a rejected family request leaves the caller's intact.
  Request candidate = request;
  std::map<std::string, std::string> renames;
  std::set<std::string> consumed;
  active_renames = &renames;
  active_selectors = &consumed;
  try {
    found->second(candidate);
  } catch (...) {
    active_renames = nullptr;
    active_selectors = nullptr;
    throw;
  }
  active_renames = nullptr;
  active_selectors = nullptr;
  request = std::move(candidate);
  if (renamed) *renamed = std::move(renames);
  if (selectors) *selectors = std::move(consumed);
  return true;
}

}  // namespace xinfer::rpu::tp
