// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Validate and lower gemm requests to native launch plans.
#include "common/rpu/tp_lowering.hpp"
#include "common/rpu/tp_matrix_validation.hpp"
#include "common/rpu/tp_tensor_validation.hpp"
#include "common/rpu/tp_state_validation.hpp"

namespace xinfer::rpu::tp::detail {
namespace {
bool lower_matrix_validation(const Request& r, Lowered& out) {
  using namespace matrix_validation;
  const auto& op = r.op;
  if (op == "gemm_ddr_col_m16n256k1024_acc32" ||
      op == "gemm_ddr_col_m32n256k1024_acc32") {
    const auto rows = op == "gemm_ddr_col_m16n256k1024_acc32" ? 16 : 32;
    const auto& x = local(r, "x", {rows, 1024}); role(x, "replicated");
    const auto& y = local(r, "out", {rows, 256}); role(y, "sharded", 1);
    const auto& w = weight(r, "weight", {8, 256, 1024});
    out.plan = abi::io_gemm({abi::IoGemmKind::ColumnParallel,
                                spm(y), spm(x), ddr(w), 0, unsigned(rows)});
    finish(r, out, {"out"}); return true;
  }

  if (op == "bmm_fp16_spm" || op == "bmm_fp16_spm_128x128x64_colcol" ||
      op == "bmm_fp16_spm_64x64x128_rowrow" ||
      op == "bmm_fp16_spm_64x128x64_rowcol" ||
      op == "bmm_fp16_spm_64x128x128_rowcol") {
    const auto& y = r.tensor("out");
    require(y.shape.size() == 3, "BMM output requires [batch,M,N]");
    const auto batch = dim(y.shape[0], "batch"), m = dim(y.shape[1], "M"), n = dim(y.shape[2], "N");
    const auto ta = r.integer("transpose_a", 0), tb = r.integer("transpose_b", 0);
    require((ta == 0 || ta == 1) && (tb == 0 || tb == 1) && !(ta && tb),
            "BMM supports (transpose_a,transpose_b)=(0,0),(0,1),(1,0)");
    const auto& a = r.tensor("a");
    require(a.shape.size() == 3, "BMM a requires rank3");
    const auto k = dim(a.shape[ta ? 1 : 2], "K");
    local(r, "out", {batch, m, n});
    local(r, "a", {batch, ta ? k : m, ta ? m : k});
    const auto& b = local(r, "b", {batch, tb ? n : k, tb ? k : n});
    same_role(y, a); same_role(y, b);
    require(y.distribution != "sharded" || y.shard_axis == 0,
            "BMM TP sharding is supported on the batch axis only");
    const auto mode = ta ? abi::BmmOrientation::ColByCol : tb ? abi::BmmOrientation::RowByRow : abi::BmmOrientation::RowByCol;
    const abi::BmmArgs args{spm(y), spm(a), spm(b), m, n, k, batch, mode};
    if (op == "bmm_fp16_spm") out.plan = abi::bmm_fp16_spm(args);
    else if (op == "bmm_fp16_spm_128x128x64_colcol") out.plan = abi::bmm_fp16_spm_colcol128(args);
    else if (op == "bmm_fp16_spm_64x64x128_rowrow") out.plan = abi::bmm_fp16_spm_rowrow64(args);
    else if (op == "bmm_fp16_spm_64x128x64_rowcol") out.plan = abi::bmm_fp16_spm_rowcol64k64(args);
    else out.plan = abi::bmm_fp16_spm_rowcol64k128(args);
    finish(r, out, {"out"}); return true;
  }

  if (op == "patch_embed_n128k1536" || op == "fc1_n512k4096" || op == "up_n512k1024" ||
      op == "fc1_gelu_n512k4096" || op == "up_gelu_n512k1024") {
    const bool patch = op == "patch_embed_n128k1536";
    const bool merger = op == "fc1_n512k4096" || op == "fc1_gelu_n512k4096";
    const auto n = patch ? 128 : 512, k = patch ? 1536 : merger ? 4096 : 1024;
    const auto& x = r.tensor("x");
    require(x.shape.size() == 2, "projection input requires rank2");
    const auto rows = dim(x.shape[0], "rows"); local(r, "x", {rows, k}); role(x, "replicated");
    const auto& y = local(r, "out", {rows, n}); role(y, "sharded", 1);
    const auto& b = local(r, "bias", {n}); role(b, "sharded", 0);
    const auto& w = weight(r, "weight", {8, n, k});
    if (op == "up_gelu_n512k1024") out.plan = abi::up_gelu_n512k1024(spm(y), spm(x), spm(b), rows, ddr(w));
    else if (op == "fc1_gelu_n512k4096") out.plan = abi::fc1_gelu_n512k4096(spm(y), spm(x), spm(b), rows, ddr(w));
    else if (op == "up_n512k1024") out.plan = abi::qkv_up_gemm({abi::Entry::UpN512K1024, spm(y), spm(x), spm(b), rows, ddr(w)});
    else out.plan = abi::mlp_projection({patch ? abi::Entry::PatchEmbed : abi::Entry::Fc1N512K4096,
                                           spm(y), spm(x), spm(b), rows, ddr(w)});
    finish(r, out, {"out"}); return true;
  }

  if (op == "gate_up_silu_m16n256k1024_acc32" || op == "gate_up_silu_m32n256k1024_acc32") {
    const unsigned rows = op == "gate_up_silu_m16n256k1024_acc32" ? 16 : 32;
    const auto& x = local(r, "x", {rows, 1024}); role(x, "replicated");
    const auto& y = local(r, "out", {rows, 256}); role(y, "sharded", 1);
    const auto& gate = weight(r, "gate_weight", {8, 256, 1024});
    const auto& up = weight(r, "up_weight", {8, 256, 1024});
    const denoise::GateUpSiluArgs args{spm(x), spm(y), ddr(gate), ddr(up), rows};
    fused(denoise::gate_up_silu(args), out);
    finish(r, out, {"out"}); return true;
  }

  if (op == "qknorm_qkv_rope_m16k1024_acc32" || op == "qknorm_qkv_rope_m32k1024_acc32") {
    const unsigned rows = op == "qknorm_qkv_rope_m16k1024_acc32" ? 16 : 32;
    const auto& x = local(r, "x", {rows, 1024}); role(x, "replicated");
    const auto& y = local(r, "planes", {4, rows, 256}); role(y, "local");
    const auto& w = weight(r, "weight", {4, 8, 256, 1024});
    const auto& qk = local(r, "qk_weight", {2, 256}); role(qk, "replicated");
    const auto& c = r.tensor("cosine");
    require(c.shape.size() == 2, "cosine requires [table_rows,32]");
    const auto n = dim(c.shape[0], "table_rows"); table(r, "cosine", {n, 32});
    const auto& s = table(r, "sine", {n, 32});
    require(!abi::detail::ddr_overlap(ddr(c), c.bytes, ddr(s), s.bytes), "cosine and sine overlap");
    fused(denoise::qkv_rope({spm(x), spm(y), spm(qk), ddr(w), ddr(c), ddr(s), n,
        nonnegative(r.integer("position_offset", 0), "position_offset"), scalar(r, "epsilon", 1.0e-6), rows}), out);
    finish(r, out, {"planes"}); return true;
  }

  if (op == "qkv_rope_bias_n384k1024") {
    const auto& x = r.tensor("x"); require(x.shape.size() == 2, "Vision QKV input requires rank2");
    const auto rows = dim(x.shape[0], "rows"); local(r, "x", {rows, 1024}); role(x, "replicated");
    const auto& q = local(r, "q", {rows, 128}); role(q, "sharded", 1);
    const auto& k = local(r, "k", {rows, 128}); role(k, "sharded", 1);
    const auto& v = local(r, "v", {rows, 128}); role(v, "sharded", 1);
    const auto& b = local(r, "bias", {384}); role(b, "local");
    const auto& w = weight(r, "weight", {8, 384, 1024});
    const auto& c = r.tensor("cosine"); require(c.shape.size() == 2, "cosine requires rank2");
    const auto n = dim(c.shape[0], "table_rows"); local(r, "cosine", {n, 32}); role(c, "replicated");
    const auto& s = local(r, "sine", {n, 32}); role(s, "replicated");
    out.plan = abi::qkv_rope_bias_n384k1024({spm(q), spm(x), spm(b), rows, ddr(w), spm(k), spm(v), spm(c), spm(s), n});
    finish(r, out, {"q", "k", "v"}); return true;
  }

  return false;
}
bool lower_tensor_validation(const Request& r, Lowered& out) {
  using namespace tensor_validation;
  const auto& op = r.op;
  if (op == "gemm_ddr_row_m16n1024k256_acc32" ||
      op == "gemm_ddr_row_m32n1024k256_acc32") {
    const unsigned m = op == "gemm_ddr_row_m16n1024k256_acc32" ? 16 : 32;
    const auto& src = matrix(r, "src", m, 256, "sharded");
    const auto& dst = matrix(r, "out", m, 1024, "partial");
    const auto& w = weight(r, 1024, 256);
    return done(r, out, abi::io_gemm({abi::IoGemmKind::RowParallel,
        address(dst), address(src), w.address, 0, m}));
  }
  const bool qkv_qgate_n1024k2048 = op == "qkv_qgate_n1024k2048" ||
      op == "qkv_qgate_m512" ||
      op == "qkv_qgate_m448_k128" ||
      op == "qkv_qgate_m512_k128";
  const bool qkv_zba = op == "qkv_zba_n1056k2048" ||
      op == "qkv_zba_m512" ||
      op == "qkv_zba_m448_k128" ||
      op == "qkv_zba_m512_k128";
  const bool gateup_n1536k2048 = op == "gateup_n1536k2048" ||
      op == "gateup_m512" ||
      op == "gateup_m448_k128" ||
      op == "gateup_m512_k128";
  if (qkv_qgate_n1024k2048 || qkv_zba || gateup_n1536k2048 || op == "o_n2048k256" ||
      op == "down_n2048k768" || op == "tp_qkv_compact_f16") {
    const auto& src = tensor(r, "src");
    const auto m = rows(src, "src");
    auto which = entry(op);
    // Default large-bucket projections select the qualified K128 program.
    // Explicit legacy *_m512 names retain K64, including their low-M ABI.
    if (m == 448 || m == 512) {
      switch (which) {
        case abi::Entry::QkvZba:
          which = m == 448 ? abi::Entry::QkvZbaM448K128 : abi::Entry::QkvZbaM512K128;
          break;
        case abi::Entry::Qkv:
          which = m == 448 ? abi::Entry::QkvQgateM448K128 : abi::Entry::QkvQgateM512K128;
          break;
        case abi::Entry::GateUp:
          which = m == 448 ? abi::Entry::GateUpM448K128 : abi::Entry::GateUpM512K128;
          break;
        default: break;
      }
    }
    const auto s = abi::streamed_gemm_shape(which);
    shape(src, {m, s.k}, "src");
    const bool partial = op == "o_n2048k256" || op == "down_n2048k768";
    distribution(src, partial ? "sharded" : "replicated", "src");
    check_tensor(!partial || src.shard_axis == 1, "row projection src must shard columns");
    const auto& w = weight(r, s.columns, s.k);
    std::vector<const char*> names;
    if (qkv_qgate_n1024k2048)
      names = {"out_q", "out_qgate", "out_k", "out_v"};
    else if (qkv_zba)
      names = {"out_q", "out_k", "out_v", "out_z", "out_b", "out_a"};
    else if (gateup_n1536k2048)
      names = {"out_gate", "out_up"};
    else names = {"out"};
    std::array<std::uint32_t, 6> outputs{};
    for (std::size_t i = 0; i < names.size(); ++i) {
      const bool replicated_kv = (qkv_qgate_n1024k2048) && i >= 2;
      const auto* dist = partial ? "partial" :
          (replicated_kv || op == "tp_qkv_compact_f16") ? "local" : "sharded";
      const auto& dst = matrix(r, names[i], m, s.output_columns[i], dist);
      outputs[i] = address(dst);
    }
    return done(r, out, abi::streamed_gemm({which, outputs, address(src), m, w.address}));
  }
  if (op == "qkv_bias_n384k1024" || op == "o_n1024k128" || op == "down_n1024k512" ||
      op == "down_n2048k512") {
    const auto which = entry(op);
    const auto& src = tensor(r, "src");
    const auto m = rows(src, "src");
    const bool qkv = op == "qkv_bias_n384k1024";
    const auto n = op == "down_n2048k512" ? 2048U : abi::qkv_up_shape(which).n;
    const auto k = op == "down_n2048k512" ? 512U : abi::qkv_up_shape(which).k;
    shape(src, {m, k}, "src");
    distribution(src, qkv ? "replicated" : "sharded", "src");
    check_tensor(qkv || src.shard_axis == 1, "Vision row projection src must shard columns");
    const auto& w = weight(r, n, k);
    const auto& bias = vector(r, "bias", n);
    if (qkv) {
      const auto& q = matrix(r, "out_q", m, 128, "sharded");
      const auto& key = matrix(r, "out_k", m, 128, "sharded");
      const auto& v = matrix(r, "out_v", m, 128, "sharded");
      return done(r, out, abi::qkv_up_gemm({which, address(q), address(src),
          address(bias), m, w.address, address(key), address(v)}));
    }
    // A core0-only bias is a prepared semantic object, not eight replicated
    // copies of one bias. The runtime validates this distribution on upload.
    distribution(bias, "owner0", "bias");
    const auto& dst = matrix(r, "out", m, n, "partial");
    return done(r, out, op == "down_n2048k512"
        ? abi::mlp_projection({which, address(dst), address(src), address(bias),
                                  m, w.address})
        : abi::qkv_up_gemm({which, address(dst), address(src), address(bias),
                            m, w.address}));
  }
  return false;
}
bool lower_state_validation(const Request& r, Lowered& result) {
  using namespace state_validation;
  const auto& op = r.op;
  if (op == "gemm_ddr_bias_scale_m16n1024k64_acc32" ||
             op == "gemm_ddr_bias_scale_m32n1024k64_acc32") {
    const auto m = op == "gemm_ddr_bias_scale_m16n1024k64_acc32" ? 16 : 32;
    const auto& out = data(r, "out");
    const auto& input = data(r, "input");
    const auto& bias = data(r, "bias");
    const auto& weight = data(r, "weight", "ddr", "nk16");
    expect_shape(out, {m, 1024}, "out"); expect_shape(input, {m, 64}, "input");
    expect_shape(bias, {1024}, "bias"); expect_shape(weight, {1024, 64}, "weight");
    expect_distribution(out, "single", "out"); expect_distribution(input, "single", "input");
    expect_distribution(bias, "single", "bias");
    result.plan = abi::io_gemm({abi::IoGemmKind::InputProjectionScaled,
        local(out), local(input), weight.address, local(bias), static_cast<unsigned>(m)});
    finish_state(r, result); return true;
  }
  if (op == "gemm_ddr_bias_m16n64k1024_acc32" ||
             op == "gemm_ddr_bias_m32n64k1024_acc32") {
    const auto m = op == "gemm_ddr_bias_m16n64k1024_acc32" ? 16 : 32;
    const auto& out = data(r, "out");
    const auto& input = data(r, "input");
    const auto& bias = data(r, "bias");
    const auto& weight = data(r, "weight", "ddr", "nk16");
    expect_shape(out, {m, 64}, "out");
    expect_shape(input, {m, 1024}, "input");
    expect_shape(bias, {64}, "bias");
    expect_shape(weight, {64, 1024}, "weight");
    expect_distribution(out, "single", "out");
    expect_distribution(bias, "single", "bias");
    need(input.distribution == "single" || input.distribution == "replicated",
         "output projection input must be single or replicated, not a TP partial/shard");
    need(input.shard_axis == -1 && out.shard_axis == -1 && bias.shard_axis == -1,
         "output projection requires unsharded complete tensors");
    need(weight.immutable, "output projection weight must be prepared immutable storage");
    // The existing I/O kernel reads only core0. Reading a replicated full
    // hidden vector does not imply a gather or an all-reduce here.
    result.plan = abi::io_gemm({abi::IoGemmKind::OutputProjection,
        local(out), local(input), weight.address, local(bias), static_cast<unsigned>(m)});
    finish_state(r, result); return true;
  }
  return false;
}
}  // namespace

bool lower_gemm(const Request& r, Lowered& out) {
  return lower_matrix_validation(r, out) ||
         lower_tensor_validation(r, out) ||
         lower_state_validation(r, out);
}
}  // namespace xinfer::rpu::tp::detail
