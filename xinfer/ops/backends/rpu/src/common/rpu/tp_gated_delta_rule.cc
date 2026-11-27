// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Validate and lower gated delta rule requests to native launch plans.
#include "common/rpu/tp_lowering.hpp"
#include "common/rpu/tp_tensor_validation.hpp"

namespace xinfer::rpu::tp::detail {
namespace {
bool lower_tensor_validation(const Request& r, Lowered& out) {
  using namespace tensor_validation;
  const auto& op = r.op;
  if (op == "cumsum_fp16" || op == "unit_tril_inv_fp16") {
    const auto& src = tensor(r, "src");
    const auto& dst = tensor(r, "out");
    shape(dst, src.shape, "out");
    same_distribution(dst, src, op == "unit_tril_inv_fp16");
    if (op == "cumsum_fp16") {
      const auto m = rows(src, "src");
      shape(src, {m, 64}, "src");
      check_tensor((src.distribution != "sharded" && src.distribution != "sequence_shard") ||
                     src.shard_axis == 0,
                 "cumsum src/output may shard rows only; axis1 requires cross-core prefix sums");
      return done(r, out, abi::cumsum_fp16({address(dst), address(src), m}));
    }
    check_tensor(src.shape.size() == 3, "unit_tril_inv_fp16 src must have rank 3");
    const auto batch = u32(src.shape[0], "batch");
    shape(src, {batch, 64, 64}, "src");
    check_tensor((src.distribution != "sharded" && src.distribution != "sequence_shard") ||
                   src.shard_axis == 0,
               "unit_tril_inv src/output may shard the batch axis only");
    check_tensor(src.distribution != "owner0" && src.distribution != "single",
               "unit_tril_inv cannot preserve zero on nonowner cores; inverse of zero input is identity");
    return done(r, out, abi::unit_tril_inv_fp16({address(dst), address(src), batch}));
  }
  if (op == "gdr_prefill_f16") {
    const auto& q = tensor(r, "q");
    const auto m = rows(q, "q");
    shape(q, {m, 256}, "q");
    distribution(q, "sharded", "q");
    check_tensor(q.shard_axis == 1, "GDR q must shard head columns");
    const auto& k = matrix(r, "k", m, 256, "sharded");
    const auto& v = matrix(r, "v", m, 256, "sharded");
    const auto& z = matrix(r, "z", m, 256, "sharded");
    const auto& a = matrix(r, "a", m, 16, "sharded");
    const auto& b = matrix(r, "b", m, 16, "sharded");
    const auto& dt = vector(r, "dt", 16, "sharded");
    const auto& neg_a = vector(r, "neg_a", 16, "sharded");
    const auto& norm = vector(r, "norm_weight", 128, "replicated");
    const auto& state = tensor(r, "state");
    shape(state, {2, 128, 128}, "state");
    distribution(state, "sharded", "state");
    check_tensor(state.shard_axis == 0 && state.state_kind == "state",
               "GDR state must be owned state sharded along head axis 0");
    const auto& scratch = tensor(r, "scratch");
    shape(scratch, {static_cast<std::int64_t>(fusion::gdr_scratch_bytes_for_rows(m) / 2)}, "scratch");
    // The workspace contains per-head temporaries from different core shards.
    // Neither identical replicas nor zero nonowner cores survive this write.
    distribution(scratch, "local", "scratch");
    const auto& dst = matrix(r, "out", m, 256, "sharded");
    check_tensor(r.integers.count("initial") && r.integers.count("valid_rows"),
            "GDR requires explicit initial and valid_rows; no implicit state reset");
    const auto debug = flag(r, "debug", false);
    std::uint32_t raw = 0, norm_out = 0;
    if (debug) {
      raw = address(matrix(r, "raw_out", m, 256, "sharded"));
      norm_out = address(matrix(r, "norm_out", m, 256, "sharded"));
    }
    const bool initial = flag(r, "initial", false);
    if (!initial) out.state_accesses.push_back({"state", 0, 1, false});
    out.state_accesses.push_back({"state", 0, 1, true});
    const auto p = fusion::gdr_prefill({address(q), address(k), address(v),
        address(z), address(a), address(b), address(dt), address(neg_a), address(norm),
        address(state), address(dst), address(scratch), m, integer(r, "valid_rows", m),
        initial, raw, norm_out, debug});
    for (const auto id : p.zero_counters) out.zero_counters.push_back(128 + 2 * id);
    return done(r, out, convert(p, abi::Entry::GdrPrefillFusion));
  }
  return false;
}
}  // namespace

bool lower_gated_delta_rule(const Request& r, Lowered& out) {
  return lower_tensor_validation(r, out);
}
}  // namespace xinfer::rpu::tp::detail
