// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Validate and lower elementwise requests to native launch plans.
#include "common/rpu/tp_lowering.hpp"
#include "common/rpu/tp_matrix_validation.hpp"
#include "common/rpu/tp_tensor_validation.hpp"
#include "common/rpu/tp_state_validation.hpp"

namespace xinfer::rpu::tp::detail {
namespace {
bool lower_matrix_validation(const Request& r, Lowered& out) {
  using namespace matrix_validation;
  const auto& op = r.op;
  if (op == "gather_embedding_f16") {
    const auto& y = r.tensor("out"); require(y.shape.size() == 2, "embedding output requires rank2");
    const auto rows = dim(y.shape[0], "output_rows"); local(r, "out", {rows, 2048}); role(y, "single");
    const auto& ids = r.tensor("ids");
    require(ids.shape.size() == 1 && ids.dtype == "int32", "embedding ids require int32 rank1");
    const auto valid = dim(ids.shape[0], "valid_rows");
    layout(ids, "row_major"); role(ids, "shared");
    require(ids.bytes >= elements(ids) * 4, "embedding ids capacity is insufficient");
    r.require_values("ids", 0.0, 256276.0);
    const auto& vocab = table(r, "vocab", {256277, 2048});
    require(vocab.immutable, "embedding vocabulary must be an immutable prepared DDR tensor");
    out.plan = abi::gather_embedding_f16({ddr(ids), ddr(vocab), spm(y), 256277, valid, rows});
    finish(r, out, {"out"}); return true;
  }
  return false;
}
bool lower_tensor_validation(const Request& r, Lowered& out) {
  using namespace tensor_validation;
  const auto& op = r.op;
  if (op == "binary_add_f16" || op == "add_vpb_f16") {
    const auto& lhs = tensor(r, "lhs");
    const auto& rhs = tensor(r, "rhs");
    const auto& dst = tensor(r, "out");
    shape(rhs, lhs.shape, "rhs");
    shape(dst, lhs.shape, "out");
    same_distribution(dst, lhs);
    same_distribution(lhs, rhs);
    const auto count = u32(elements(lhs, "lhs"), "elements");
    return done(r, out, op == "binary_add_f16"
        ? abi::binary_f16({address(dst), address(lhs), address(rhs), count,
                               abi::BinaryOp::Add})
        : abi::add_vpb({address(dst), address(lhs), address(rhs), count,
                          integer(r, "vectors_per_block", 32)}));
  }
  if (op == "bcast_rows_f16" || op == "bcast_tiles_f16") {
    const auto& src = tensor(r, "src");
    const auto m = rows(src, "src"), n = u32(src.shape[1], "columns");
    const auto& dst = matrix(r, "out", m, n);
    const auto& a = tensor(r, "a");
    same_distribution(dst, src);
    const auto mode = integer(r, "mode", 0);
    const auto operation = integer(r, "operation", 0);
    check_tensor(operation <= 2, "broadcast operation must be Add/Sub/Mul before enum conversion");
    if (src.distribution == "partial") {
      if (operation == 2) {
        // A common multiplier commutes with the later sum. Multiplying two
        // partial operands would omit the cross-core products.
        check_tensor(a.distribution == "replicated",
                   "broadcast a must be replicated when multiplying partial src");
      } else {
        // Both subtraction orders are linear. A replicated addend would be
        // counted eight times; a partial or core0-owned addend is counted once.
        check_tensor(a.distribution == "partial" || a.distribution == "owner0" ||
                       a.distribution == "single",
                   "broadcast a must be partial or core0-owned when adding/subtracting partial src");
      }
    } else {
      check_tensor(a.distribution != "partial",
                 "broadcast a cannot be partial when src/output are resolved");
      check_tensor(src.distribution != "replicated" || a.distribution == "replicated",
                 "broadcast a must be replicated when src/output are replicated");
      const bool owned_src = src.distribution == "owner0" || src.distribution == "single";
      const bool owned_a = a.distribution == "owner0" || a.distribution == "single";
      if (owned_src && !owned_a) {
        // Add/Sub need zero on nonowner cores; multiplication needs a finite
        // multiplier because 0*Inf/NaN would destroy the zero-owner invariant.
        // Uploaded value provenance is rechecked by the runtime on every replay.
        if (operation == 2) r.require_values("a", -65504.0, 65504.0);
        else r.require_values("a", 0.0, 0.0);
      }
    }
    const auto group = integer(r, "group_rows", 1);
    if (mode == 0) shape(a, {m}, "a");
    else if (mode == 1) shape(a, {n}, "a");
    else if (mode == 2) {
      check_tensor(group > 0 && m % group == 0, "group_rows must divide src rows");
      shape(a, {m / group, n}, "a");
    } else throw std::invalid_argument("RPU TP: unsupported broadcast mode");
    check_tensor((op == "bcast_tiles_f16") == (n % 4096 == 0),
            "broadcast entry does not match the supplied columns");
    auto p = abi::bcast_f16({address(dst), address(src), address(a), m, n,
        group, static_cast<abi::BcastMode>(mode),
        static_cast<abi::BcastOp>(operation), flag(r, "x_first", false)});
    return done(r, out, p);
  }
  if (op == "gather_embedding_tp8_f16") {
    const auto& ids = tensor(r, "ids", "ddr", "row_major", "int32");
    check_tensor(ids.shape.size() == 1, "embedding ids must have rank 1");
    const auto& vocab = tensor(r, "vocab", "ddr");
    shape(vocab, {abi::vla_embedding_vocab_rows, 2048}, "vocab");
    check_tensor(vocab.immutable, "vocab must be prepared immutable DDR storage");
    distribution(ids, "shared", "ids");
    distribution(vocab, "shared", "vocab");
    const auto& dst = tensor(r, "out");
    const auto local_rows = rows(dst, "out");
    shape(dst, {local_rows, 2048}, "out");
    distribution(dst, "sharded", "out");
    check_tensor(dst.shard_axis == 0, "embedding output must shard rows (axis 0)");
    const auto output_rows = u32(std::uint64_t(local_rows) * 8, "output_rows");
    const auto valid = integer(r, "valid_rows", ids.shape[0]);
    check_tensor(valid <= elements(ids, "ids"), "embedding ids do not cover valid_rows");
    r.require_values("ids", 0, abi::vla_embedding_vocab_rows - 1);
    // Runtime rechecks uploaded ID statistics on every execution.
    return done(r, out, abi::gather_embedding_tp8_f16({ids.address, vocab.address,
        address(dst), abi::vla_embedding_vocab_rows, valid, output_rows}));
  }
  return false;
}
bool lower_state_validation(const Request& r, Lowered& result) {
  using namespace state_validation;
  const auto& op = r.op;
  if (op == "fill_f16") {
    const auto& out = data(r, "out");
    const auto value = finite_half(r.scalar("value", 0.0), "value");
    need((out.distribution != "owner0" && out.distribution != "single") ||
             (value & 0x7fffU) == 0,
         "fill value must round to zero for owner0/single out because launch writes all eight cores");
    result.plan = abi::fill({local(out), static_cast<unsigned>(count_elements(out)),
        value,
        integer(r, "vectors_per_block", 32, 64)});
    finish_state(r, result); return true;
  }
  if (op == "binary_sub_f16" || op == "binary_mul_f16" ||
             op == "binary_max_f16") {
    const auto& out = data(r, "out");
    const auto& lhs = data(r, "lhs");
    const auto& rhs = data(r, "rhs");
    local_pair(out, lhs); local_pair(out, rhs);
    const auto operation = op == "binary_sub_f16" ? abi::BinaryOp::Sub
                         : op == "binary_mul_f16" ? abi::BinaryOp::Mul
                                                        : abi::BinaryOp::Max;
    need(operation == abi::BinaryOp::Sub || out.distribution != "partial",
            "multiply/maximum require resolved tensors, not partial sums");
    result.plan = abi::binary_f16({local(out), local(lhs), local(rhs),
        static_cast<unsigned>(count_elements(out)), operation});
    finish_state(r, result); return true;
  }
  if (op == "scalar_mul_f16") {
    const auto& out = data(r, "out");
    const auto& input = data(r, "input");
    local_pair(out, input);
    result.plan = abi::scalar_mul_f16({local(out), local(input),
        static_cast<unsigned>(count_elements(out)), finite_half(r.scalar("value", -1.0), "value")});
    finish_state(r, result); return true;
  }
  return false;
}
}  // namespace

bool lower_elementwise(const Request& r, Lowered& out) {
  return lower_matrix_validation(r, out) ||
         lower_tensor_validation(r, out) ||
         lower_state_validation(r, out);
}
}  // namespace xinfer::rpu::tp::detail
