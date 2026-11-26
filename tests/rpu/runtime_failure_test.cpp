// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Compile the real resource registry, upload statistics and execution lifecycle
// against host-only fault-injection doubles. No device arithmetic is simulated.
#include "common/rpu/execution.cc"
#include <iostream>

namespace xinfer::rpu {
std::int32_t load_program(rhino_lkn::Program_t&, const char*, bool) noexcept { return 0; }
const char* last_error() noexcept { return "host runtime test double"; }
// Authored admission metadata, not embedded device code or SDK capability.
unsigned ar_cap_fixture_mode = 1, m16_cap_fixture_mode = 0;
// Family admission fixture: list rmsnorm_gdn_f16 as a candidate kernel.
bool family_admission_fixture = false;
std::size_t kernel_count() noexcept { return family_admission_fixture ? 3 : 2; }
const char* kernel_name(std::size_t i) noexcept {
  if (i == 2) return "rmsnorm_gdn_f16";
  return i == 0 ? (ar_cap_fixture_mode ? "all_reduce_residual_window" : "missing_ar")
                : (m16_cap_fixture_mode ? "down_allreduce_rmsnorm_hw_window_m16n1024k256_acc32" : "missing_m16");
}
const char* kernel_admission(std::size_t i) noexcept {
  if (i == 2) return "candidate";
  return (i ? m16_cap_fixture_mode : ar_cap_fixture_mode) == 3 ? "unknown" : "candidate";
}
const char* kernel_execution_status(std::size_t i) noexcept {
  if (i == 2) return "";
  return (i ? m16_cap_fixture_mode : ar_cap_fixture_mode) == 2 ? "quarantined" : "";
}

}  // namespace xinfer::rpu

namespace {
using namespace xinfer::rpu::execution;
unsigned checks = 0;
void expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
  ++checks;
}
template<class F> void rejects(F&& call, const char* message) {
  try { call(); }
  catch (const std::exception& error) {
    if (std::string(error.what()).find(message) == std::string::npos)
      throw std::runtime_error(std::string("wrong failure: ") + error.what());
    ++checks; return;
  }
  throw std::runtime_error(std::string("expected rejection: ") + message);
}
struct Handles {
  std::vector<Handle> values;
  Handle keep(Handle id) { values.push_back(id); return id; }
  ~Handles() { for (auto it = values.rbegin(); it != values.rend(); ++it) release(*it); }
};
// Exercise the real upload-proof scanner, including its ARM vector/tail
// branch, without loading the SDK or simulating device numerical execution.
void fp16_statistics_guards() {
  Handles h;
  auto context = h.keep(tp_context_create(true));
  auto owner = h.keep(tp_empty(context, {128}, "float16", "ddr", "row_major", "shared", -1, "none"));
  auto tensor = get<TpTensor>(owner, Kind::TpTensor, "statistics owner");
  auto double_bits = [](double value) {
    std::uint64_t result; std::memcpy(&result, &value, sizeof(result)); return result;
  };
  auto compare = [&](const std::vector<std::uint16_t>& words, std::size_t offset, std::size_t count) {
    const auto saved = words;
    double minimum = std::numeric_limits<double>::infinity(), maximum = -minimum;
    bool finite = true;
    for (std::size_t i = 0; i < count; ++i) {
      const double value = xinfer::rpu::fp16_to_fp32(words[offset + i]);
      finite = finite && std::isfinite(value);
      minimum = std::min(minimum, value); maximum = std::max(maximum, value);
    }
    tp_stats(*tensor, count ? words.data() + offset : nullptr, count);
    expect(tensor->desc.values_known && tensor->desc.finite == finite, "FP16 proof classification changed");
    expect(double_bits(tensor->desc.minimum) == double_bits(minimum), "FP16 proof minimum bits/zero sign changed");
    expect(double_bits(tensor->desc.maximum) == double_bits(maximum), "FP16 proof maximum bits/zero sign changed");
    expect(words == saved, "FP16 proof scanner changed input or guards");
  };
  // Full encoding domain spans all vector/tail positions without using a
  // scalar mirror of the integer ordering algorithm as the oracle.
  for (unsigned bits = 0; bits < 65536; ++bits) {
    std::vector<std::uint16_t> words(19, 0x3c00);
    words[0] = 0x1357; words[18] = 0x2468;
    words[1 + bits % 17] = static_cast<std::uint16_t>(bits);
    compare(words, 1, 17);
  }
  for (unsigned count : {0U, 1U, 7U, 8U, 9U, 15U, 16U, 17U, 33U})
    for (std::uint16_t first : {0x0000, 0x8000})
      for (std::uint16_t fill : {0x0000, 0x8000, 0x3c00, 0xbc00}) {
        std::vector<std::uint16_t> words(count + 2, fill);
        words.front() = 0x1357; words.back() = 0x2468;
        if (count) words[1 + count / 2] = first;
        compare(words, 1, count);
      }
}
void submission_guards() {
  Handles h;
  const auto context = h.keep(context_create(0, true));
  auto allocate = [&] { return h.keep(buffer_empty(context, 16, 32)); };
  const auto x = allocate(), y = allocate(), out = allocate();
  const auto execution = h.keep(prepare_silu_mul(out, x, y));
  const std::vector<std::uint16_t> words(16 * 32, 0x3c00);
  for (bool throws : {false, true}) {
    if (throws) rhino_lkn::throw_on_enqueue = true;
    else rhino_lkn::next_enqueue_status = 7;
    rejects([&] { execute(execution, false); },
            throws ? "test enqueue_batch exception" : "enqueue_batch failed: SDK status 7");
    const auto prepared = get<Prepared>(execution, Kind::Execution, "e");
    expect(!prepared->in_flight && !device_state().in_flight,
           "enqueue failure left a permanently outstanding execution");
    // Recovery must permit ordinary host transfer without a mandatory wait.
    buffer_upload(x, words.data(), 16, 32);
  }
  execute(execution, true);
  rejects([&] { buffer_upload(x, words.data(), 16, 32); }, "outstanding execution");
  wait(execution);
  buffer_upload(x, words.data(), 16, 32);
}

void finite_guards() {
  fp16_statistics_guards();
  Handles h;
  auto ctx = h.keep(tp_context_create(true));
  auto tensor = [&](std::vector<std::int64_t> dims) {
    return h.keep(tp_empty(ctx, dims, "float16", "spm", "row_major", "local", -1, "none"));
  };
  auto x = tensor({128,128}), out = tensor({128,128}), weight = tensor({128});
  std::vector<std::uint16_t> input(128*128, 0), ones(128, 0x3c00);
  tp_upload(weight, ones.data(), {128}, "float16");
  auto prepare_norm = [&] {
    return tp_prepare(ctx, "rmsnorm_gdn_f16", {"out","x","weight"}, {out,x,weight}, {}, {}, {}, {});
  };
  for (const std::uint16_t bits : {0x7e00, 0xfe00, 0x7c01, 0x7c00, 0xfc00}) {
    input[17] = bits; tp_upload(x, input.data(), {128,128}, "float16");
    rejects([&] { h.keep(prepare_norm()); }, "requires uploaded finite");
  }
  input[17] = 0; tp_upload(x, input.data(), {128,128}, "float16");
  auto node = h.keep(prepare_norm()); auto batch = h.keep(tp_batch(ctx, {node}));
  input[17] = 0x7e00; tp_upload(x, input.data(), {128,128}, "float16");
  const auto submitted = rhino_lkn::enqueue_count;
  rejects([&] { tp_execute(batch, false); }, "replay input violates value constraint");
  expect(rhino_lkn::enqueue_count == submitted, "invalid replay reached the SDK");
  // Use the production scalar gate, including NaN, both infinities, and a
  // normal scalar control; do not rely on constant-folded test-only isfinite.
  for (double value : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
    rejects([&] { h.keep(tp_prepare(ctx, "fill_f16", {"out"}, {out}, {}, {}, {"value"}, {value})); },
            "must be finite");
  }
  h.keep(tp_prepare(ctx, "fill_f16", {"out"}, {out}, {}, {}, {"value"}, {0.25}));
  ++checks;
}
void memory_guards() {
  Handles h;
  const auto ctx = h.keep(tp_context_create(true));
  const auto parent = h.keep(tp_empty(ctx, {128,128}, "float16", "spm", "row_major", "local", -1, "none"));
  const auto view = h.keep(tp_view(parent, {128,128}, 0, "local", -1));
  const auto sibling = h.keep(tp_view(parent, {128,128}, 0, "local", -1));
  std::vector<std::uint16_t> zeros(128*128, 0), result(8*128*128, 0xffff);
  tp_upload(parent, zeros.data(), {128,128}, "float16");
  expect(get<TpTensor>(parent, Kind::TpTensor, "parent")->desc.values_known,
         "owner upload did not establish its value proof");
  tp_upload(view, zeros.data(), {128,128}, "float16");
  expect(!get<TpTensor>(parent, Kind::TpTensor, "parent")->desc.values_known,
         "view upload preserved an owner proof");
  expect(!get<TpTensor>(sibling, Kind::TpTensor, "sibling")->desc.values_known,
         "view upload preserved a sibling proof");
  tp_download(result.data(), sibling, {8,128,128}, "float16");
  expect(std::all_of(result.begin(), result.end(), [](auto v) { return v == 0; }),
         "view did not alias physical SPM bytes");
  rejects([&] { h.keep(tp_view(parent, {128}, 1, "local", -1)); }, "8-byte aligned");
  rejects([&] { h.keep(tp_view(parent, {128,128}, 8, "local", -1)); }, "exceeds its parent");
  rejects([&] { h.keep(tp_view(parent, {128}, 0, "single", -1)); }, "zero nonowner");
  const auto table = h.keep(tp_weight(ctx, zeros.data(), {128,128}, "row_major"));
  const auto readonly = h.keep(tp_view(table, {128}, 8, "shared", -1));
  rejects([&] { tp_upload(readonly, zeros.data(), {128}, "float16"); }, "immutable tensor");
  const auto packed = h.keep(tp_weight(ctx, zeros.data(), {128,128}, "nk16"));
  rejects([&] { h.keep(tp_view(packed, {128}, 0, "shared", -1)); }, "row_major");
  const auto state = h.keep(tp_empty(ctx, {128}, "float16", "spm", "row_major", "local", -1, "state"));
  rejects([&] { h.keep(tp_view(state, {128}, 0, "local", -1)); }, "without state");
  const auto target = h.keep(tp_empty(ctx, {128,128}, "float16", "spm", "row_major", "local", -1, "none"));
  const auto copy = h.keep(tp_prepare(ctx, "copy_f16", {"out","input"}, {target,view}, {}, {}, {}, {}));
  rhino_lkn::next_dma_status = 9;
  rejects([&] { h.keep(tp_batch(ctx, {copy})); }, "SDK status 9");
  const auto enqueues = rhino_lkn::enqueue_count;
  expect(enqueues == 0, "rejected DMA reached an enqueue");
  const auto before = rhino_lkn::dma_count;
  h.keep(tp_batch(ctx, {copy}));
  expect(rhino_lkn::dma_count == before + 8, "corewise copy did not retain eight DMA endpoints");
  const auto affine = h.keep(tp_empty(ctx, {128}, "float16", "spm", "row_major", "replicated", -1, "none"));
  const auto norm = h.keep(tp_prepare(ctx, "layernorm_f16", {"out","input","weight"},
                                     {target,view,affine}, {}, {}, {"epsilon"}, {1e-5}));
  expect(get<TpNode>(norm, Kind::TpNode, "norm")->lowered.plan.cores == 8,
         "TP LayerNorm selected a single-core launcher");
  rejects([&] { h.keep(tp_prepare(ctx, "layernorm_f16", {"out","input","weight"},
                                  {target,view,affine}, {}, {}, {"epsilon"}, {0})); },
          "epsilon must round to positive finite FP16");
  // Closing handles does not release allocations retained by a view/operation.
  release(parent);
  h.values.erase(std::find(h.values.begin(), h.values.end(), parent));
  tp_download(result.data(), view, {8,128,128}, "float16");
  expect(std::all_of(result.begin(), result.end(), [](auto v) { return v == 0; }),
         "view did not retain its owner after the owner handle was closed");
}
void state_failure(bool asynchronous, bool missing_completion) {
  Handles h;
  auto ctx = h.keep(tp_context_create(true));
  auto state = h.keep(tp_empty(ctx, {4,256}, "float16", "spm", "row_major", "local", -1, "history"));
  tp_reset(state);
  auto writer = h.keep(tp_prepare(ctx, "fill_f16", {"out"}, {state}, {}, {}, {"value"}, {0.25}));
  auto batch = h.keep(tp_batch(ctx, {writer}));
  rhino_lkn::next_enqueue_status = missing_completion ? 0 : 7;
  const char* error = missing_completion ? "completion marker" : "SDK status 7";
  if (asynchronous) { tp_execute(batch, true); rejects([&] { tp_wait(batch); }, error); }
  else rejects([&] { tp_execute(batch, false); }, error);
  rejects([&] { tp_wait(batch); }, error);

  auto input = h.keep(tp_empty(ctx, {256}, "float16", "spm", "row_major", "local", -1, "none"));
  std::vector<std::uint16_t> weight_data(8*4*256, 0);
  auto weight = h.keep(tp_weight(ctx, weight_data.data(), {8,4,256}, "row_major"));
  auto reader = h.keep(tp_prepare(ctx, "causal_depthwise_conv1d_step_f16",
      {"state","input","weight"}, {state,input,weight}, {}, {}, {}, {}));
  auto consumer = h.keep(tp_batch(ctx, {reader}));
  const auto submitted = rhino_lkn::enqueue_count;
  rhino_lkn::next_enqueue_status = 8;
  rejects([&] { tp_execute(consumer, false); }, "read outside initialized state/cache range");
  expect(rhino_lkn::enqueue_count == submitted, "invalid history reached an SDK enqueue");

  // Re-upload is an explicit recovery; the same prepared consumer now reaches
  // submission. Its injected error proves preflight accepted restored state.
  std::vector<std::uint16_t> restored(4*256, 0);
  tp_upload(state, restored.data(), {4,256}, "float16");
  const auto generation = get<TpTensor>(state, Kind::TpTensor, "state")->desc.generation;
  tp_invalidate(state);
  expect(get<TpTensor>(state, Kind::TpTensor, "state")->desc.generation == generation,
         "new-request invalidation changed captured state generation");
  rejects([&] { tp_execute(consumer, false); }, "read outside initialized state/cache range");
  tp_upload(state, restored.data(), {4,256}, "float16");
  rejects([&] { tp_execute(consumer, false); }, "SDK status 8");
  expect(rhino_lkn::enqueue_count == submitted + 1, "restored state did not reach SDK");
  if (asynchronous) {
    // Destroying an in-flight failed async batch without tp_wait() must still
    // invalidate written state owners under device_state().mutex in ~TpBatch().
    tp_upload(state, restored.data(), {4,256}, "float16");
    auto unawaited = tp_batch(ctx, {writer});
    rhino_lkn::next_enqueue_status = missing_completion ? 0 : 7;
    tp_execute(unawaited, true);
    release(unawaited);
    rejects([&] { tp_execute(consumer, false); }, "read outside initialized state/cache range");
  }
}
// Memo hits must have exactly the same caller-visible binding as a fresh
// validation, including aliases, scalar bits and mutable upload proofs.
void rebind_memo_guards() {
  Handles h;
  const auto ctx = h.keep(tp_context_create(true));
  const auto parent = h.keep(tp_empty(ctx, {32}, "float16", "ddr", "row_major", "shared", -1, "none"));
  const auto first = h.keep(tp_view(parent, {16}, 0, "shared", -1));
  const auto second = h.keep(tp_view(parent, {16}, 32, "shared", -1));
  const auto wrong_shape = h.keep(tp_view(parent, {32}, 0, "shared", -1));
  const auto output_owner = h.keep(tp_empty(ctx, {16}, "float16", "spm", "row_major", "local", -1, "none"));
  const auto output = h.keep(tp_view(output_owner, {16}, 0, "replicated", -1));
  const auto wrong_role = h.keep(tp_view(output_owner, {16}, 0, "local", -1));
  const auto copy = h.keep(tp_prepare(ctx, "copy_f16", {"input","out"}, {first,output}, {"mode"}, {1}, {}, {}));
  auto copy_node = get<TpNode>(copy, Kind::TpNode, "copy");
  for (const auto source : {first,second,first,second}) {
    tp_rebind(copy, {"input"}, {source}, {}, {}, {}, {});
    expect(copy_node->tensors.at("input") == get<TpTensor>(source, Kind::TpTensor, "source"),
           "memo confused distinct views of one owner");
  }
  expect(copy_node->rebind_memos.size() == 2, "identical view rebinds lost their memo hits");
  const auto revision = copy_node->revision;
  rejects([&] { tp_rebind(copy, {"input"}, {wrong_shape}, {}, {}, {}, {}); }, "same shape");
  rejects([&] { tp_rebind(copy, {"out"}, {wrong_role}, {}, {}, {}, {}); }, "broadcast copy");
  expect(copy_node->revision == revision && copy_node->tensors.at("input") ==
             get<TpTensor>(second, Kind::TpTensor, "source"),
         "rejected view rebind changed the published binding");

  const auto fill = h.keep(tp_prepare(ctx, "fill_f16", {"out"}, {output}, {}, {}, {"value"}, {0.25}));
  auto fill_node = get<TpNode>(fill, Kind::TpNode, "fill");
  for (const double value : {1e-7,2e-7,1e-7,0.0,-0.0,0.0,-0.0}) {
    tp_rebind(fill, {}, {}, {}, {}, {"value"}, {value});
    const auto actual = fill_node->request.scalars.at("value");
    expect(std::memcmp(&actual, &value, sizeof(value)) == 0,
           "memo changed the requested scalar bits");
    expect(fill_node->lowered.plan.regs[4] == xinfer::rpu::fp32_to_fp16(static_cast<float>(value)),
           "memo restored stale FP16 scalar registers");
  }
  expect(fill_node->rebind_memos.size() == 4, "exact scalar repeats lost their memo hits");
  for (unsigned value = 1; value <= 10; ++value)
    tp_rebind(fill, {}, {}, {}, {}, {"value"}, {static_cast<double>(value)});
  expect(fill_node->rebind_memos.size() == 8, "rebind memo no longer obeys its capacity bound");

  auto tensor = [&](std::vector<std::int64_t> shape) {
    return h.keep(tp_empty(ctx, shape, "float16", "spm", "row_major", "local", -1, "none"));
  };
  const auto x = tensor({128,128}), out = tensor({128,128}), weight = tensor({128});
  std::vector<std::uint16_t> input(128*128, 0), ones(128, 0x3c00);
  tp_upload(x, input.data(), {128,128}, "float16");
  tp_upload(weight, ones.data(), {128}, "float16");
  const auto norm = h.keep(tp_prepare(ctx, "rmsnorm_gdn_f16", {"out","x","weight"},
                                     {out,x,weight}, {}, {}, {}, {}));
  tp_rebind(norm, {}, {}, {}, {}, {}, {});
  auto norm_node = get<TpNode>(norm, Kind::TpNode, "norm");
  const auto norm_revision = norm_node->revision;
  for (const std::uint16_t bits : {0x4000,0x7e00}) {
    input[17] = bits; tp_upload(x, input.data(), {128,128}, "float16");
    rejects([&] { tp_rebind(norm, {}, {}, {}, {}, {}, {}); }, "value constraint");
    expect(norm_node->revision == norm_revision, "memo published an invalidated input proof");
  }
  input[17] = 0; tp_upload(x, input.data(), {128,128}, "float16");
  tp_rebind(norm, {}, {}, {}, {}, {}, {});
  expect(norm_node->revision == norm_revision + 1 && norm_node->rebind_memos.size() == 1,
         "restored valid input did not reuse its checked memo");
}
void mutable_guards() {
  Handles h;
  const auto ctx = h.keep(tp_context_create(true));
  const auto out = h.keep(tp_empty(ctx, {16}, "float16", "spm", "row_major", "replicated", -1, "none"));
  const auto fill = h.keep(tp_prepare(ctx, "fill_f16", {"out"}, {out}, {}, {}, {"value"}, {0.25}));
  tp_rebind(fill, {}, {}, {}, {}, {"value"}, {0.5});
  auto node = get<TpNode>(fill, Kind::TpNode, "fill");
  expect(node->revision == 1 && node->request.scalars.at("value") == 0.5,
         "valid scalar rebind was not published");
  rejects([&] { tp_rebind(fill, {}, {}, {}, {}, {"value"},
                         {std::numeric_limits<double>::quiet_NaN()}); }, "must be finite");
  expect(node->revision == 1 && node->request.scalars.at("value") == 0.5,
         "invalid scalar rebind modified the previous binding");
  const auto src = h.keep(tp_empty(ctx, {16}, "float16", "ddr", "row_major", "shared", -1, "none"));
  const auto next = h.keep(tp_empty(ctx, {16}, "float16", "ddr", "row_major", "shared", -1, "none"));
  const auto copy = h.keep(tp_prepare(ctx, "copy_f16", {"input","out"}, {src,out}, {"mode"}, {1}, {}, {}));
  const auto batch = h.keep(tp_batch(ctx, {copy,fill}));
  const auto allocations = tp_context_stats(ctx);
  const auto initial_nodes = tp_batch_stats(batch);
  tp_rebind(copy, {"input"}, {next}, {}, {}, {}, {});
  tp_rebind(fill, {}, {}, {}, {}, {"value"}, {0.75});
  const auto syncs = rhino_lkn::mutable_sync_count, dmas = rhino_lkn::dma_update_count;
  rhino_lkn::next_enqueue_status = 7;
  rejects([&] { tp_execute(batch, false); }, "SDK status 7");
  expect(rhino_lkn::mutable_sync_count == syncs + 1 && rhino_lkn::dma_update_count == dmas + 8,
         "rebound packets were not synchronized once before submission");
  expect(tp_batch_stats(batch) == initial_nodes && tp_context_stats(ctx) == allocations,
         "rebind allocated another queue or SPM owner");
  const auto wrong = h.keep(tp_empty(ctx, {32}, "float16", "ddr", "row_major", "shared", -1, "none"));
  rejects([&] { tp_rebind(copy, {"input"}, {wrong}, {}, {}, {}, {}); }, "same shape");
  expect(get<TpNode>(copy, Kind::TpNode, "copy")->tensors.at("input") ==
         get<TpTensor>(next, Kind::TpTensor, "next"), "rejected DMA rebind changed its endpoint");
  tp_rebind(fill, {}, {}, {}, {}, {"value"}, {1.0});
  const auto before_enqueue = rhino_lkn::enqueue_count;
  rhino_lkn::next_sync_status = 12;
  rejects([&] { tp_execute(batch, false); }, "SDK status 12");
  rejects([&] { tp_wait(batch); }, "SDK status 12");
  expect(rhino_lkn::enqueue_count == before_enqueue, "failed mutable sync enqueued device work");
  const auto state = h.keep(tp_empty(ctx, {16}, "float16", "spm", "row_major", "local", -1, "state"));
  const auto writer = h.keep(tp_prepare(ctx, "fill_f16", {"out"}, {state}, {}, {}, {"value"}, {0}));
  tp_reset(state);
  rejects([&] { tp_rebind(writer, {}, {}, {}, {}, {"value"}, {0.5}); }, "stale prepared state");
  const auto storage = h.keep(tp_empty(ctx, {128}, "float16", "spm", "row_major", "local", -1, "none"));
  const auto alias = h.keep(tp_view(storage, {128}, 0, "local", -1));
  auto drop = [&](Handle id) { release(id); h.values.erase(std::find(h.values.begin(), h.values.end(), id)); };
  const auto held = tp_context_stats(ctx)[1]; drop(storage);
  expect(tp_context_stats(ctx)[1] == held, "closing owner freed SPM still retained by an alias");
  drop(alias);
  expect(tp_context_stats(ctx)[1] == held - 256, "last alias release did not return SPM to the native arena");
}
// Sigmoid/rsqrt private table accounting is gone: table owners, their
// weak_ptr slots and the corresponding stats indices no longer exist.
void completion_tag_guards() {
  Handles h;
  const auto ctx = h.keep(tp_context_create(true));
  const auto input = h.keep(tp_empty(ctx, {1,1024}, "float16", "spm", "row_major", "partial", -1, "none"));
  const auto output = h.keep(tp_empty(ctx, {1,1024}, "float16", "spm", "row_major", "replicated", -1, "none"));
  const auto fill = h.keep(tp_prepare(ctx, "fill_f16", {"out"}, {output}, {}, {}, {"value"}, {0}));
  const auto reduce = h.keep(tp_prepare(ctx, "all_reduce_residual", {"input","out"},
      {input,output}, {"with_residual"}, {0}, {}, {}));
  const auto id = h.keep(tp_batch(ctx, {fill,reduce,fill,reduce,fill}));
  auto batch = get<TpBatch>(id, Kind::TpBatch, "batch");
  expect(batch->segments.size() == 5 && batch->collective_count == 2,
         "marker optimization changed producer/collective segment boundaries");
  expect(batch->completion_tags.size() == batch->segments.size(), "missing segment completion tags");
  std::set<std::uint16_t> unique;
  for (std::size_t segment = 0; segment < 1024; ++segment) {
    const auto tag = tp_completion_tag(segment);
    expect(tag && std::isfinite(xinfer::rpu::fp16_to_fp32(tag)) && unique.insert(tag).second,
           "completion tag is zero, nonfinite, or repeated");
    if (segment < batch->completion_tags.size())
      expect(batch->completion_tags[segment] == tag, "prepared segment has the wrong completion tag");
  }
  rejects([&] { (void)tp_completion_tag(1024); }, "SDK segment bound");
  auto mark = [](TpBatch& target, std::uint16_t tag) {
    std::array<std::uint16_t,16> words; words.fill(tag);
    for (unsigned core = 0; core < 8; ++core)
      tp_write(target.completion->spm_pointer(core), words.data(), sizeof(words));
  };
  const auto current = batch->completion_tags[1], previous = batch->completion_tags[0];
  mark(*batch, current); tp_verify_completion(*batch, current); ++checks;
  for (unsigned core = 0; core < 8; ++core) {
    for (unsigned lane = 0; lane < 16; ++lane) {
      mark(*batch, current);
      auto* words = static_cast<std::uint16_t*>(batch->completion->spm_pointer(core));
      words[lane] = 0;
      rejects([&] { tp_verify_completion(*batch, current); }, "completion marker");
    }
    mark(*batch, current);
    std::array<std::uint16_t,16> stale; stale.fill(previous);
    tp_write(batch->completion->spm_pointer(core), stale.data(), sizeof(stale));
    rejects([&] { tp_verify_completion(*batch, current); }, "completion marker");
  }
  mark(*batch, previous);
  rejects([&] { tp_verify_completion(*batch, current); }, "completion marker");
  // Forty-eight collectives with producer gaps retain 97 independent segments.
  // The tag change does not admit a forty-ninth reduction or a 1025th node.
  std::vector<Handle> chain;
  for (unsigned i = 0; i < 48; ++i) { chain.push_back(fill); chain.push_back(reduce); }
  chain.push_back(fill);
  const auto full = h.keep(tp_batch(ctx, chain));
  auto full_batch = get<TpBatch>(full, Kind::TpBatch, "full batch");
  expect(full_batch->segments.size() == 97 && full_batch->completion_tags.size() == 97 &&
         full_batch->collective_count == 48 && full_batch->segments.size() <= full_batch->nodes.size(),
         "completion tags changed the 48-collective envelope");
  rejects([&] { h.keep(tp_batch(ctx, std::vector<Handle>(49, reduce))); }, "exceeds48");
  rejects([&] { h.keep(tp_batch(ctx, std::vector<Handle>(1025, fill))); }, "1..1024");
  const auto single_id = h.keep(tp_batch(ctx, {fill}));
  auto single = get<TpBatch>(single_id, Kind::TpBatch, "single batch");
  expect(single->segments.size() == 1 && single->completion_tags.size() == 1,
         "single-segment completion metadata is incomplete");
  for (const bool asynchronous : {false, true}) {
    for (unsigned replay = 0; replay < 2; ++replay) {
      // Simulate only stale control words, never any device arithmetic. The
      // host SDK fixture returns success without writing a completion marker.
      mark(*single, single->completion_tags[0]);
      rhino_lkn::next_enqueue_status = 0;
      const auto enqueues = rhino_lkn::enqueue_count;
      if (asynchronous) {
        tp_execute(single_id, true);
        rejects([&] { tp_wait(single_id); }, "completion marker");
      } else rejects([&] { tp_execute(single_id, false); }, "completion marker");
      expect(rhino_lkn::enqueue_count == enqueues + 1, "stale replay did not reach the SDK fixture");
      for (unsigned core = 0; core < 8; ++core) {
        std::array<std::uint16_t,16> words;
        tp_read(words.data(), single->completion->spm_pointer(core), sizeof(words));
        expect(std::all_of(words.begin(), words.end(), [](auto value) { return value == 0; }),
               "logical execution did not clear its stale completion marker on every core");
      }
    }
  }
}
void collective_window_guards() {
  Handles h;
  const auto ctx = h.keep(tp_context_create(true));
  const auto input = h.keep(tp_empty(ctx, {1,1024}, "float16", "spm", "row_major", "partial", -1, "none"));
  const auto output = h.keep(tp_empty(ctx, {1,1024}, "float16", "spm", "row_major", "replicated", -1, "none"));
  const auto fill = h.keep(tp_prepare(ctx, "fill_f16", {"out"}, {input}, {}, {}, {"value"}, {0.125}));
  const auto reduce = h.keep(tp_prepare(ctx, "all_reduce_residual", {"input","out"}, {input,output}, {}, {}, {}, {}));
  std::vector<Handle> chain;
  for (unsigned i = 0; i < 48; ++i) { chain.push_back(fill); chain.push_back(reduce); }
  chain.push_back(fill);
  const auto allocations = tp_context_stats(ctx)[1];
  const auto denied = h.keep(tp_context_create(false));
  rejects([&] { h.keep(tp_batch_with_window(denied, chain, 8)); }, "candidate admission");
  for (const auto window : {0,1,16})
    rejects([&] { h.keep(tp_batch_with_window(ctx, chain, window)); }, "[2,15]");
  rejects([&] { h.keep(tp_batch_with_window(ctx, std::vector<Handle>(49, reduce), 8)); }, "exceeds48");
  rejects([&] { h.keep(tp_batch_with_window(ctx, std::vector<Handle>(1025, fill), 8)); }, "1..1024");
  expect(tp_context_stats(ctx)[1] == allocations, "rejected window graph allocated completion storage");
  for (unsigned mode : {0U,2U,3U}) {
    xinfer::rpu::ar_cap_fixture_mode = mode;
    expect(tp_ar_window_max() == 0, "missing/quarantined/unknown metadata advertised AR capability");
    rejects([&] { h.keep(tp_batch_with_window(ctx, chain, 4)); }, "calibrated AR");
    expect(tp_context_stats(ctx)[1] == allocations, "missing capability allocated completion storage");
  }
  xinfer::rpu::ar_cap_fixture_mode = 1;
  expect(tp_ar_window_max() == 15, "calibrated AR fixture capability changed");
  const auto unmodified = h.keep(tp_batch(ctx, chain));
  expect(tp_batch_stats(unmodified)[2] == 97, "default collective segmentation changed");
  for (const unsigned window : {2U,4U,8U,12U,15U}) {
    const auto id = h.keep(tp_batch_with_window(ctx, chain, window));
    const auto batch = get<TpBatch>(id, Kind::TpBatch, "window");
    expect(batch->segments.size() == 2 * ((48 + window - 1) / window) && batch->nodes.size() == chain.size() &&
           batch->collective_count == 48 && batch->completion_tags.size() == batch->segments.size(),
           "window initializer/body count or logical operation order changed");
    unsigned occurrences = 0;
    for (const auto& binding : batch->kernel_bindings) {
      if (binding.node->request.op != "all_reduce_residual") continue;
      expect(binding.window_index == static_cast<int>(occurrences % window) &&
             binding.counter_slot == 12 && binding.counters.empty(),
             "window body resets an atomic bank or loses its counter occurrence");
      expect(std::all_of(binding.kernel->written_regs.begin(), binding.kernel->written_regs.end(),
                         [](unsigned index) { return index < 128; }),
             "window body wrote an atomic register");
      ++occurrences;
    }
    expect(occurrences == 48, "window omitted a collective occurrence");
    unsigned initializers = 0;
    for (const auto& kernel : batch->kernels) {
      if (std::find(kernel->written_regs.begin(), kernel->written_regs.end(), 226) == kernel->written_regs.end())
        continue;
      ++initializers;
      for (unsigned counter = 0; counter <= 50; ++counter)
        expect(std::count(kernel->written_regs.begin(), kernel->written_regs.end(), 128 + 2 * counter) == 1,
               "window initializer did not clear its complete atomic-register bank exactly once");
    }
    expect(initializers == (48 + window - 1) / window, "window initialization is absent or duplicated");
    // Rebind must preserve private entry/count/indices when synchronizing the
    // original public node. Fault injection avoids simulating device arithmetic.
    tp_rebind(reduce, {}, {}, {}, {}, {}, {});
    rhino_lkn::next_enqueue_status = 7;
    rejects([&] { tp_execute(id, false); }, "SDK status 7");
    for (const auto& binding : batch->kernel_bindings)
      expect(binding.revision == binding.node->revision, "window rebind failed to publish its revision");
    for (const auto& binding : batch->kernel_bindings)
      if (binding.window_index >= 0)
        expect(std::all_of(binding.kernel->written_regs.begin(), binding.kernel->written_regs.end(),
                           [](unsigned reg) { return reg < 128; }),
               "mutable window synchronization wrote an atomic register");
    for (const bool asynchronous : {false,true}) {
      const auto enqueues = rhino_lkn::enqueue_count;
      rhino_lkn::next_enqueue_status = 0;
      if (asynchronous) {
        tp_execute(id, true);
        rejects([&] { tp_wait(id); }, "completion marker");
      } else rejects([&] { tp_execute(id, false); }, "completion marker");
      expect(rhino_lkn::enqueue_count == enqueues + 1,
             "window body reached the SDK after initialization missed a completion marker");
    }
  }
  const auto gathered = h.keep(tp_empty(ctx, {128,1024}, "float16", "spm", "row_major", "replicated", -1, "none"));
  const auto shard = h.keep(tp_empty(ctx, {128,128}, "float16", "spm", "row_major", "sharded", 1, "none"));
  const auto gather = h.keep(tp_prepare(ctx, "all_gather_f16", {"input","out"}, {shard,gathered}, {}, {}, {}, {}));
  const auto bounded = h.keep(tp_batch_with_window(ctx, {reduce,gather,reduce}, 8));
  expect(tp_batch_stats(bounded)[2] == 5, "all-gather lost its hard completion boundary");
  // A graph of individually hard boundaries can exhaust the SDK-segment cap
  // even while its logical-op and counted-collective budgets remain valid.
  std::vector<Handle> over;
  for (unsigned i = 0; i < 48; ++i) { over.push_back(reduce); over.push_back(gather); }
  over.resize(1024, gather);
  const auto before = tp_context_stats(ctx)[1];
  rejects([&] { h.keep(tp_batch_with_window(ctx, over, 8)); }, "exceeds1024 SDK segments");
  expect(tp_context_stats(ctx)[1] == before, "SDK-segment budget rejection allocated a completion owner");
}
}  // namespace

namespace {
void m16_collective_window_guards() {
  using namespace xinfer::rpu;
  Handles h;
  const auto ctx = h.keep(tp_context_create(true));
  auto tensor = [&](std::vector<std::int64_t> shape, const char* role, std::int64_t axis = -1) {
    return h.keep(tp_empty(ctx, shape, "float16", "spm", "row_major", role, axis, "none"));
  };
  const auto src = tensor({16,256}, "sharded", 1), partial = tensor({16,1024}, "partial"),
      residual = tensor({16,1024}, "replicated"), stream = tensor({16,1024}, "replicated"),
      norm = tensor({16,1024}, "replicated");
  std::vector<std::uint16_t> zeros(8*1024*256, 0), row(3*1024, 0);
  const auto weight = h.keep(tp_weight(ctx, zeros.data(), {8,1024,256}, "tp8_nk16"));
  const auto gate = h.keep(tp_weight(ctx, row.data(), {3,1024}, "row_major"));
  const auto next = h.keep(tp_weight(ctx, row.data(), {3,1024}, "row_major"));
  const auto down = h.keep(tp_prepare(ctx, "down_allreduce_rmsnorm_m16n1024k256_acc32",
      {"src","partial","residual","stream","norm","weight","gate_row","next_row"},
      {src,partial,residual,stream,norm,weight,gate,next}, {}, {}, {}, {}));
  const auto input = tensor({1,1024}, "partial"), output = tensor({1,1024}, "replicated");
  const auto ar = h.keep(tp_prepare(ctx, "all_reduce_residual", {"input","out"}, {input,output}, {}, {}, {}, {}));
  const auto memory = tp_context_stats(ctx), pool = tp_pool_stats(ctx);
  const auto submitted = rhino_lkn::enqueue_count;
  for (unsigned mode : {0U,2U,3U}) {
    m16_cap_fixture_mode = mode;
    expect(tp_down_norm_window_max() == 0, "unqualified metadata advertised M16");
    rejects([&] { h.keep(tp_batch_with_window(ctx, {down}, 4)); }, "requested calibrated M16");
    expect(tp_context_stats(ctx) == memory && tp_pool_stats(ctx) == pool &&
           rhino_lkn::enqueue_count == submitted, "M16cap rejection allocated or submitted");
  }
  m16_cap_fixture_mode = 1;
  expect(tp_ar_window_max() == 15 && tp_down_norm_window_max() == 4, "family caps conflated");
  rejects([&] { h.keep(tp_batch_with_window(ctx, {down}, 8)); }, "requested calibrated M16");
  for (const auto& ids : {std::vector<Handle>{ar,down}, std::vector<Handle>{down,ar}})
    rejects([&] { h.keep(tp_batch_with_window(ctx, ids, 4)); }, "cannot mix AR and M16");
  expect(tp_context_stats(ctx) == memory && tp_pool_stats(ctx) == pool &&
         rhino_lkn::enqueue_count == submitted, "mixed-family rejection allocated or submitted");
  const auto default_id = h.keep(tp_batch(ctx, {ar,down}));
  expect(tp_batch_stats(default_id)[2] == 2, "default1 mixed-family compatibility changed");
  const auto id = h.keep(tp_batch_with_window(ctx, std::vector<Handle>(5, down), 4));
  expect(tp_batch_stats(id) == std::vector<std::int64_t>({5,9,4,0,0,5}), "M16W4 ragged plan changed");
  const auto batch = get<TpBatch>(id, Kind::TpBatch, "batch");
  unsigned occurrence = 0;
  for (const auto& binding : batch->kernel_bindings) {
    if (!binding.node || binding.node->request.op != "down_allreduce_rmsnorm_m16n1024k256_acc32") continue;
    const auto j = occurrence++ % 4;
    const auto packet = abi::collective_window(binding.node->lowered.plan, j);
    expect(binding.window_index == static_cast<int>(j) && binding.counters.empty() &&
           packet.register_count == 37 && packet.regs[24] == 34 + j &&
           packet.regs[33] == 2 + 2*j && packet.regs[36] == 3 + 2*j, "M16 calibrated packet changed");
    for (auto reg : binding.kernel->written_regs) expect(reg < 128, "M16 body writes atomicbank");
  }
  expect(occurrence == 5, "M16 tail occurrence lost");
  m16_cap_fixture_mode = 0;
}

void fused_rebind_guards() {
  Handles h;
  const auto ctx = h.keep(tp_context_create(true));
  auto tensor = [&](const char* role, std::vector<std::int64_t> dims = {1,1024}, int axis = -1) {
    return h.keep(tp_empty(ctx, dims, "float16", "spm", "row_major", role, axis, "none"));
  };
  const auto partial = tensor("partial"), first_out = tensor("replicated"),
      second_out = tensor("replicated"), independent = tensor("replicated");
  const auto first = h.keep(tp_prepare(ctx, "all_reduce_residual", {"input","out"},
      {partial,first_out}, {"counter"}, {50}, {}, {}));
  const auto second = h.keep(tp_prepare(ctx, "all_reduce_residual", {"input","out","residual"},
      {partial,second_out,first_out}, {"counter","with_residual"}, {49,1}, {}, {}));
  const auto fused = h.keep(tp_batch_fused(ctx, {first,second}));
  expect(tp_batch_stats(fused)[2] == 1 && tp_batch_stats(fused)[5] == 2,
         "valid fused chain lost its single segment or collectives");
  tp_rebind(second, {"residual"}, {independent}, {}, {}, {}, {});
  const auto submitted = rhino_lkn::enqueue_count;
  const auto synced = rhino_lkn::mutable_sync_count;
  for (bool asynchronous : {false,true}) {
    rejects([&] { tp_execute(fused, asynchronous); }, "downstream of the previous collective");
    expect(rhino_lkn::enqueue_count == submitted && rhino_lkn::mutable_sync_count == synced,
           "disconnected fused replay reached SDK synchronization or submission");
  }
  const auto memory = tp_context_stats(ctx);
  rejects([&] { h.keep(tp_batch_fused(ctx, {first,second})); }, "downstream");
  expect(tp_context_stats(ctx) == memory, "rejected fused chain allocated completion storage");
  tp_rebind(second, {"residual"}, {first_out}, {}, {}, {}, {});
  rhino_lkn::next_enqueue_status = 9;
  rejects([&] { tp_execute(fused, false); }, "SDK status 9");
  expect(rhino_lkn::enqueue_count == submitted + 1,
         "restoring a valid dependency did not make the existing fused batch executable");
  const auto source = tensor("sharded", {16,128}, 1), gathered = tensor("replicated", {16,1024});
  const auto gather = h.keep(tp_prepare(ctx, "all_gather_f16", {"input","out"},
                                      {source,gathered}, {}, {}, {}, {}));
  rejects([&] { h.keep(tp_batch_fused(ctx, {gather})); }, "followed by its DMA burst");
  const auto copy_out = tensor("replicated", {16,1024});
  const auto copy = h.keep(tp_prepare(ctx, "copy_f16", {"input","out"},
      {gathered,copy_out}, {"mode"}, {0}, {}, {}));
  h.keep(tp_batch_fused(ctx, {gather,copy}));
}
}  // namespace

// Family entry points through the production prepare/rebind/selection path.
void family_guards() {
  std::vector<std::uint16_t> zeros(128*128, 0), ones(128, 0x3c00);
  {
  Handles h;
  auto ctx = h.keep(tp_context_create(true));
  auto tensor = [&](std::vector<std::int64_t> dims) {
    return h.keep(tp_empty(ctx, dims, "float16", "spm", "row_major", "local", -1, "none"));
  };
  auto x = tensor({128,128}), out = tensor({128,128}), weight = tensor({128});
  tp_upload(x, zeros.data(), {128,128}, "float16");
  tp_upload(weight, ones.data(), {128}, "float16");
  // rmsnorm mode=2 (gated) is the catalog entry rmsnorm_gdn_f16.
  auto family = h.keep(tp_prepare(ctx, "rmsnorm", {"x","out","weight"}, {x,out,weight},
                                  {"mode"}, {2}, {}, {}));
  auto legacy = h.keep(tp_prepare(ctx, "rmsnorm_gdn_f16", {"x","out","weight"}, {x,out,weight},
                                  {}, {}, {}, {}));
  const auto fs = tp_selection(family), ls = tp_selection(legacy);
  expect(fs.size() == 4 && fs[0] == "rmsnorm_gdn_f16" && fs[1] == "rmsnorm_gdn_f16" &&
         fs[2] == "rmsnorm", "family selection reports kernel, entry and family");
  expect(ls[0] == fs[0] && ls[1] == fs[1] && ls[2].empty(), "catalog selection has no family");
  auto fn = get<TpNode>(family, Kind::TpNode, "family node");
  auto ln = get<TpNode>(legacy, Kind::TpNode, "legacy node");
  expect(fn->lowered.plan.entry == ln->lowered.plan.entry && fn->lowered.plan.regs == ln->lowered.plan.regs &&
         fn->lowered.plan.grid == ln->lowered.plan.grid, "family and catalog packets differ");
  // In-place alias: one tensor bound under two semantic names.
  auto inplace = h.keep(tp_prepare(ctx, "rmsnorm", {"x","out","weight"}, {x,x,weight},
                                   {"mode"}, {2}, {}, {}));
  expect(get<TpNode>(inplace, Kind::TpNode, "inplace")->tensors.at("x") ==
         get<TpNode>(inplace, Kind::TpNode, "inplace")->tensors.at("out"), "alias lost");
  // Rebind accepts the family's own names, rejects selectors and unknown names.
  auto other = tensor({128,128});
  tp_rebind(family, {"out"}, {other}, {}, {}, {}, {});
  expect(fn->tensors.at("out")->desc.identity == get<TpTensor>(other, Kind::TpTensor, "o")->desc.identity,
         "family rebind did not rebind out");
  rejects([&] { tp_rebind(family, {}, {}, {"mode"}, {0}, {}, {}); },
          "mode selects the kernel and is fixed at prepare");
  rejects([&] { tp_rebind(family, {"input"}, {other}, {}, {}, {}, {}); },
          "rmsnorm rebind: unknown tensor argument input");
  // Family errors name the family and the violated condition.
  rejects([&] { h.keep(tp_prepare(ctx, "rmsnorm", {"x","out","weight"}, {x,out,weight},
                                  {"mode"}, {9}, {}, {})); }, "RPU TP rmsnorm: mode must be 0");
  xinfer::rpu::family_admission_fixture = true;
  expect(tp_selection(family)[3] == "candidate", "selection reports the kernel admission");
  }
  // Admission is checked against the selected kernel, at prepare. One TP8
  // arena exists at a time, so the strict context follows the first's release.
  Handles h;
  auto strict = h.keep(tp_context_create(false));
  auto sx = h.keep(tp_empty(strict, {128,128}, "float16", "spm", "row_major", "local", -1, "none"));
  auto so = h.keep(tp_empty(strict, {128,128}, "float16", "spm", "row_major", "local", -1, "none"));
  auto sw = h.keep(tp_empty(strict, {128}, "float16", "spm", "row_major", "local", -1, "none"));
  tp_upload(sx, zeros.data(), {128,128}, "float16");
  tp_upload(sw, ones.data(), {128}, "float16");
  rejects([&] { h.keep(tp_prepare(strict, "rmsnorm", {"x","out","weight"}, {sx,so,sw},
                                  {"mode"}, {2}, {}, {})); },
          "selects candidate kernel rmsnorm_gdn_f16; create the context with allow_candidate=True");
  xinfer::rpu::family_admission_fixture = false;
}

int main(int argc, char** argv) {
  try {
    const std::string mode = argc == 2 ? argv[1] : "all";
    if (mode == "all" || mode == "submission") submission_guards();
    if (mode == "all" || mode == "finite") finite_guards();
    if (mode == "all" || mode == "memory") memory_guards();
    if (mode == "all" || mode == "mutable") { mutable_guards(); rebind_memo_guards(); }
    if (mode == "all" || mode == "fused") fused_rebind_guards();
    // sigmoid_table mode is gone (table accounting no longer exists).
    if (mode == "all" || mode == "completion_tags") completion_tag_guards();
    if (mode == "all" || mode == "family") family_guards();
    if (mode == "all" || mode == "collective_windows") {
      collective_window_guards(); m16_collective_window_guards();
    }
    if (mode == "all" || mode == "state")
      for (bool asynchronous : {false, true})
        for (bool missing_completion : {false, true}) state_failure(asynchronous, missing_completion);
    std::cout << "Production runtime host checks: " << checks << " passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
