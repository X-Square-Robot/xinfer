// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/tp_contract.hpp"
#include "common/rpu/tp_execution.hpp"
#include "common/rpu/tp_family.hpp"
#include "common/rpu/library.hpp"
#include "common/rpu/tp_cache_layout.hpp"
#include "common/rpu/tp_memory.hpp"
#include "common/rpu/elementwise_abi.hpp"
#include <algorithm>
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <map>
#include <future>
#include <set>
#include <dirent.h>
#include <sys/resource.h>

namespace xinfer::rpu::execution {
namespace {
struct TpTensor;
struct TpContext final : Resource {
  explicit TpContext(bool candidates) : Resource(Kind::TpContext), allow_candidate(candidates) {}
  ~TpContext() override {
    if (::getpid() != process) { for (auto& p : arenas) p.release(); return; }
    std::lock_guard<std::recursive_mutex> lock(device_state().mutex);
    for (unsigned c = 0; c < 8; ++c) if (arenas[c]) {
      arenas[c].reset(); device_state().allocated_bytes[c] -= abi::working_bytes;
    }
  }
  const bool allow_candidate;
  std::array<std::unique_ptr<rhino_lkn::LocalSPM_t>, 8> arenas;
  std::map<std::string, std::weak_ptr<rhino_lkn::Program_t>> programs;
  std::map<std::size_t, std::size_t> free_ranges;
  std::size_t peak_allocated = 0;
  void initialize() {
    if (arenas[0]) return;
    auto& device = device_state();
    for (auto n : device.allocated_bytes)
      require(n == 0, "TP8 arena requires the per-core SPM budget to be free");
    std::array<std::unique_ptr<rhino_lkn::LocalSPM_t>, 8> candidate;
    for (unsigned c = 0; c < 8; ++c) {
      candidate[c] = std::make_unique<rhino_lkn::LocalSPM_t>(abi::working_bytes,
          rhino_lkn::read_write, rhino_lkn::kStride256B, 256, c);
      require(candidate[c]->get_cpu_ptr() && candidate[c]->get_rpu_addr() % 256 == 0,
              "invalid TP8 SPM mapping");
      require(candidate[c]->get_rpu_addr() <= abi::spm_bytes - abi::working_bytes,
              "TP8 arena exceeds the device SPM address range");
      abi::detail::span(candidate[c]->get_rpu_addr(), abi::working_bytes);
      require(c == 0 || candidate[c]->get_rpu_addr() == candidate[0]->get_rpu_addr(),
              "SDK cannot provide equal TP8 arena offsets");
    }
    arenas = std::move(candidate);
    for (auto& n : device.allocated_bytes) n += abi::working_bytes;
    free_ranges.emplace(0, abi::working_bytes);
  }
  std::size_t allocate(std::size_t bytes) {
    initialize();
    for (auto it = free_ranges.begin(); it != free_ranges.end(); ++it) {
      if (it->second < bytes) continue;
      const auto base = it->first, remaining = it->second - bytes;
      free_ranges.erase(it);
      if (remaining) free_ranges.emplace(base + bytes, remaining);
      std::size_t available = 0;
      for (const auto& range : free_ranges) available += range.second;
      peak_allocated = std::max(peak_allocated, abi::working_bytes - available);
      return base;
    }
    throw std::invalid_argument("TP8 SPM arena capacity exceeded");
  }
  void free(std::size_t offset, std::size_t bytes) {
    auto next = free_ranges.lower_bound(offset);
    if (next != free_ranges.end() && offset + bytes == next->first) {
      bytes += next->second; free_ranges.erase(next);
    }
    next = free_ranges.lower_bound(offset);
    if (next != free_ranges.begin()) {
      auto previous = std::prev(next);
      if (previous->first + previous->second == offset) {
        offset = previous->first; bytes += previous->second; free_ranges.erase(previous);
      }
    }
    free_ranges.emplace(offset, bytes);
  }
};
struct TpTensor final : Resource {
  explicit TpTensor(std::shared_ptr<TpContext> ctx)
      : Resource(Kind::TpTensor), context(std::move(ctx)) {}
  ~TpTensor() override {
    if (::getpid() != process) { ddr_memory.release(); return; }
    std::lock_guard<std::recursive_mutex> lock(device_state().mutex);
    if (allocated) context->free(offset, allocated);
    ddr_memory.reset();
  }
  std::shared_ptr<TpContext> context;
  std::shared_ptr<TpTensor> owner;
  std::shared_ptr<std::vector<std::weak_ptr<TpTensor>>> aliases =
      std::make_shared<std::vector<std::weak_ptr<TpTensor>>>();
  tp::Tensor desc;
  std::size_t offset = 0, allocated = 0;
  std::unique_ptr<rhino_lkn::HostDDR_t> ddr_memory;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> valid_ranges;
  rhino_lkn::HostDDR_t& ddr_buffer() const {
    return owner ? owner->ddr_buffer() : *ddr_memory;
  }
  void* ddr_pointer() const {
    return static_cast<unsigned char*>(ddr_buffer().get_cpu_ptr()) + offset;
  }
  void* spm_pointer(unsigned core) const {
    return static_cast<unsigned char*>(context->arenas[core]->get_cpu_ptr()) + offset;
  }
};
struct TpNode final : Resource {
  explicit TpNode(std::shared_ptr<TpContext> ctx) : Resource(Kind::TpNode), context(std::move(ctx)) {}
  std::shared_ptr<TpContext> context;
  tp::Request request;
  tp::Lowered lowered;
  std::map<std::string, std::shared_ptr<TpTensor>> tensors;
  std::map<std::string, std::uint64_t> generations;
  std::uint64_t revision = 0;
  // Family entry point the caller used ("" for a catalog entry name) and the
  // caller's argument names -> the catalog entry's names, used by rebind.
  std::string family;
  std::map<std::string, std::string> argument_names;
  std::set<std::string> selectors;
  // Reuse validated lowerings for the same tensors, generations and scalar bits.
  struct RebindMemo {
    std::string key;
    tp::Request request;
    tp::Lowered lowered;
    std::map<std::string, std::shared_ptr<TpTensor>> tensors;
  };
  std::vector<RebindMemo> rebind_memos;
};
struct TpBatch;
void tp_verify_completion(TpBatch& batch, std::uint16_t expected);
void tp_invalidate_writes(TpBatch& batch);
struct TpBatch final : Resource {
  explicit TpBatch(std::shared_ptr<TpContext> ctx) : Resource(Kind::TpBatch), context(std::move(ctx)) {}
  ~TpBatch() override {
    if (::getpid() != process) {
      for (auto& q : segments) q.release();
      for (auto& p : kernels) p.release();
      (void)new std::future<void>(std::move(pending));
      // The inherited SDK mappings must not be destructed in a forked child.
      (void)new std::vector<std::shared_ptr<rhino_lkn::Program_t>>(std::move(programs));
      return;
    }
    std::lock_guard<std::recursive_mutex> lock(device_state().mutex);
    if (in_flight) {
      // A worker uses this object only while its future is retained here.
      // Waiting before field destruction keeps even a closed handle alive.
      try { pending.get(); }
      catch (...) {
        try { tp_invalidate_writes(*this); } catch (...) {}
      }
      in_flight = device_state().in_flight = false;
    }
    segments.clear(); kernels.clear(); programs.clear();
  }
  std::shared_ptr<TpContext> context;
  std::vector<std::shared_ptr<TpNode>> nodes;
  std::vector<std::shared_ptr<rhino_lkn::Program_t>> programs;
  std::vector<std::unique_ptr<rhino_lkn::Kernel_t>> kernels;
  std::vector<std::unique_ptr<rhino_lkn::Queue_t>> segments;
  std::vector<std::uint16_t> completion_tags;
  struct KernelBinding {
    std::shared_ptr<TpNode> node;
    rhino_lkn::Kernel_t* kernel;
    std::size_t segment;
    int counter_slot;
    int window_index;
    std::vector<unsigned> counters;
    std::uint64_t revision;
  };
  struct DmaBinding {
    std::shared_ptr<TpNode> node;
    std::size_t segment;
    unsigned id, core;
    std::uint64_t revision;
  };
  std::vector<KernelBinding> kernel_bindings;
  std::vector<DmaBinding> dma_bindings;
  unsigned next_dma_id = 0;
  std::future<void> pending;
  std::exception_ptr failure;
  std::shared_ptr<TpTensor> completion;
  bool in_flight = false;
  bool fused = false;
  std::size_t dma_count = 0, barrier_count = 0, collective_count = 0;
};
void tp_require_batch_size(const std::vector<Handle>& ids) {
  require(!ids.empty() && ids.size() <= 1024,
          "TP batch must contain 1..1024 explicit operations");
}
void tp_idle() {
  require(!device_state().in_flight, "RPU device has outstanding work; wait before TP access");
}
void tp_require_descriptor_headroom() {
  rlimit limits{};
  require(::getrlimit(RLIMIT_NOFILE, &limits) == 0, "cannot inspect process file descriptor limit");
  if (limits.rlim_cur == RLIM_INFINITY) return;
  DIR* directory = ::opendir("/proc/self/fd");
  require(directory != nullptr, "file descriptor budget exhausted before SDK allocation");
  std::size_t count = 0;
  while (auto* entry = ::readdir(directory))
    if (std::strcmp(entry->d_name, ".") && std::strcmp(entry->d_name, "..")) ++count;
  ::closedir(directory);
  // Include this inspection's descriptor in the conservative count and leave
  // room for SDK DMA-BUF export/mapping and diagnostic error propagation.
  require(count < limits.rlim_cur && limits.rlim_cur - count >= 16,
          "insufficient file descriptor headroom for SDK allocation; admit a larger process RLIMIT_NOFILE budget");
}
std::size_t tp_itemsize(const std::string& dtype) {
  require(dtype == "float16" || dtype == "int32", "TP dtype must be float16 or int32");
  return dtype == "float16" ? 2 : 4;
}
void tp_write(void* target, const void* source, std::size_t bytes) {
  if (bytes % 8 == 0) { write_spm(target, source, bytes); return; }
  std::vector<unsigned char> padded((bytes + 7) & ~std::size_t(7), 0);
  std::memcpy(padded.data(), source, bytes); write_spm(target, padded.data(), padded.size());
}
void tp_read(void* target, const void* source, std::size_t bytes) {
  if (bytes % 8 == 0) { read_spm(target, source, bytes); return; }
  std::vector<unsigned char> padded((bytes + 7) & ~std::size_t(7));
  read_spm(padded.data(), source, padded.size()); std::memcpy(target, padded.data(), bytes);
}
std::uint64_t tp_state_capacity(const TpTensor& t) {
  if (t.desc.state_kind != "cache") return 1;
  if (t.desc.layout == "causal_k_cache" || t.desc.layout == "causal_v_cache") return 780;
  return t.desc.shape.at(1);
}
void tp_stats(TpTensor& t, const void* data, std::size_t count) {
  auto& d = t.desc; d.values_known = true; d.finite = true;
  d.minimum = std::numeric_limits<double>::infinity(); d.maximum = -d.minimum;
  if (d.dtype == "float16") {
    const auto* values = static_cast<const std::uint16_t*>(data);
    std::uint16_t minimum_bits = 0x7c00U, maximum_bits = 0xfc00U;
    std::uint16_t minimum_key = 0xfc00U, maximum_key = 0x03ffU;
    bool all_finite = true;
#if defined(__aarch64__) && defined(__ARM_NEON)
    // Only integer AdvSIMD operations touch input bits before the original
    // nonfinite fallback. Unsigned ordered keys merge +0/-0 at key0x8000.
    const auto magnitude_mask = vdupq_n_u16(0x7fffU);
    const auto exponent_limit = vdupq_n_u16(0x7c00U);
    const auto sign_bit = vdupq_n_u16(0x8000U);
    const auto zero = vdupq_n_u16(0);
    auto low = vdupq_n_u16(minimum_key);
    auto high = vdupq_n_u16(maximum_key);
    auto nonfinite = zero;
    const std::size_t vector_end = count & ~std::size_t(7);
    for (std::size_t i = 0; i < vector_end; i += 8) {
      const auto value = vld1q_u16(values + i);
      const auto magnitude = vandq_u16(value, magnitude_mask);
      nonfinite = vorrq_u16(nonfinite, vcgeq_u16(magnitude, exponent_limit));
      const auto negative_mask = vreinterpretq_u16_s16(
          vshrq_n_s16(vreinterpretq_s16_u16(value), 15));
      auto key = veorq_u16(value, vorrq_u16(negative_mask, sign_bit));
      key = vbslq_u16(vceqq_u16(magnitude, zero), sign_bit, key);
      low = vminq_u16(low, key); high = vmaxq_u16(high, key);
    }
    all_finite = vmaxvq_u16(nonfinite) == 0;
    minimum_key = vminvq_u16(low); maximum_key = vmaxvq_u16(high);
    for (std::size_t i = vector_end; all_finite && i < count; ++i) {
      const std::uint16_t value = values[i];
      const auto magnitude = value & 0x7fffU;
      if (magnitude >= 0x7c00U) { all_finite = false; break; }
      const std::uint16_t key = magnitude == 0U ? 0x8000U
          : (value & 0x8000U) ? (value ^ 0xffffU) : (value ^ 0x8000U);
      minimum_key = std::min(minimum_key, key);
      maximum_key = std::max(maximum_key, key);
    }
    if (all_finite) {
      minimum_bits = minimum_key ^ ((minimum_key & 0x8000U) ? 0x8000U : 0xffffU);
      maximum_bits = maximum_key ^ ((maximum_key & 0x8000U) ? 0x8000U : 0xffffU);
      if (minimum_key == 0x8000U || maximum_key == 0x8000U) {
        // Equal numeric zeros must keep the first original sign, exactly as
        // the strict scalar updates do. This pass occurs only at a zero extremum.
        for (std::size_t i = 0; i < count; ++i) if ((values[i] & 0x7fffU) == 0U) {
          if (minimum_key == 0x8000U) minimum_bits = values[i];
          if (maximum_key == 0x8000U) maximum_bits = values[i];
          break;
        }
      }
    }
#else
    for (std::size_t i = 0; i < count; ++i) {
      const std::uint16_t value = values[i];
      const auto magnitude = value & 0x7fffU;
      // Keep the original floating-point evaluation order, including fenv,
      // for every array containing an infinity or NaN. No FP conversion has
      // occurred in this integer-only scan before falling back below.
      if (magnitude >= 0x7c00U) { all_finite = false; break; }
      // Both zeros compare equal: strict updates retain the first zero's sign.
      const std::uint16_t key = magnitude == 0U ? 0x8000U
          : (value & 0x8000U) ? (value ^ 0xffffU) : (value ^ 0x8000U);
      if (key < minimum_key) { minimum_key = key; minimum_bits = value; }
      if (key > maximum_key) { maximum_key = key; maximum_bits = value; }
    }
#endif
    if (all_finite) {
      d.minimum = static_cast<double>(fp16_to_fp32(minimum_bits));
      d.maximum = static_cast<double>(fp16_to_fp32(maximum_bits));
      return;
    }
  }
  for (std::size_t i = 0; i < count; ++i) {
    const double v = d.dtype == "float16"
        ? static_cast<double>(fp16_to_fp32(static_cast<const std::uint16_t*>(data)[i]))
        : static_cast<double>(static_cast<const std::int32_t*>(data)[i]);
    d.finite = d.finite && std::isfinite(v);
    d.minimum = std::min(d.minimum, v); d.maximum = std::max(d.maximum, v);
  }
}
void tp_invalidate_contents(TpTensor& t) {
  for (auto it = t.aliases->begin(); it != t.aliases->end();) {
    if (auto alias = it->lock()) {
      alias->desc.values_known = alias->desc.finite = false;
      alias->valid_ranges.clear();
      ++it;
    } else it = t.aliases->erase(it);
  }
}
void tp_invalidate_writes(TpBatch& batch) {
  // Generic operators (fill/binary/copy) may write a state/history owner without
  // declaring specialized StateAccess intervals. Failure must invalidate every
  // written owner, not only the explicit cache/state transitions.
  for (const auto& node : batch.nodes) {
    for (const auto& name : node->lowered.writes)
      tp_invalidate_contents(*node->tensors.at(name));
    for (const auto& access : node->lowered.state_accesses)
      if (access.write) tp_invalidate_contents(*node->tensors.at(access.tensor));
  }
}
std::shared_ptr<TpTensor> tp_allocate(const std::shared_ptr<TpContext>& context,
    const std::vector<std::int64_t>& dims, const std::string& dtype,
    const std::string& storage, const std::string& layout_name,
    const std::string& distribution_name, std::int64_t axis, const std::string& state_kind) {
  tp_require_descriptor_headroom();
  auto result = std::make_shared<TpTensor>(context);
  auto& d = result->desc;
  d.shape = dims; d.dtype = dtype; d.storage = storage; d.layout = layout_name;
  d.distribution = distribution_name; d.shard_axis = axis; d.state_kind = state_kind;
  const auto size = tp_itemsize(dtype); d.bytes = tp::elements(d) * size;
  require(d.bytes <= (1ULL << 40), "TP allocation byte size exceeds address range");
  require(storage == "spm" || storage == "ddr", "TP storage must be spm or ddr");
  require(state_kind == "none" || state_kind == "cache" || state_kind == "history" || state_kind == "state",
          "unsupported TP state kind");
  const std::set<std::string> layouts{"row_major", "nk16", "tp8_nk16", "vision_k_cache", "vision_v_cache", "causal_k_cache", "causal_v_cache"};
  require(layouts.count(layout_name), "unknown TP physical layout");
  if (tp::is_cache_layout(layout_name)) {
    require(storage == "ddr" && dtype == "float16" && state_kind == "cache",
            "cache layout requires FP16 DDR cache owner");
    tp::cache_offset(d, 0, 0, 0);
  }
  require(state_kind != "cache" || tp::is_cache_layout(layout_name),
          "cache owner requires an explicit cache layout");
  if (storage == "spm") {
    const std::set<std::string> roles{"local", "replicated", "sharded", "partial", "single", "owner0", "sequence_shard"};
    require(roles.count(distribution_name), "invalid SPM distribution");
    require(layout_name == "row_major", "SPM requires contiguous row_major layout");
    if (distribution_name == "sharded" || distribution_name == "sequence_shard")
      require(axis >= 0 && axis < static_cast<std::int64_t>(dims.size()), "sharded tensor requires valid shard_axis");
    else require(axis == -1, "unsharded tensor must not specify shard_axis");
    require(d.bytes <= abi::working_bytes, "SPM tensor exceeds7 MiB");
    result->allocated = (d.bytes + 255) & ~std::size_t(255);
    // Set allocation ownership only after allocate succeeds.
    const auto allocation = result->allocated; result->allocated = 0;
    result->offset = context->allocate(allocation); result->allocated = allocation;
    d.address = context->arenas[0]->get_rpu_addr() + result->offset;
    if (distribution_name == "single" || distribution_name == "owner0") {
      // Core0-only producers leave these cores untouched, while generic TP8
      // consumers rely on their zero contribution. Establish it on allocation,
      // including arena reuse; core0 retains ordinary empty-buffer semantics.
      const std::vector<unsigned char> zero(d.bytes, 0);
      for (unsigned c = 1; c < 8; ++c)
        tp_write(result->spm_pointer(c), zero.data(), d.bytes);
    }
  } else {
    require(distribution_name == "shared" && axis == -1, "DDR is shared with no shard_axis");
    result->ddr_memory = std::make_unique<rhino_lkn::HostDDR_t>((d.bytes + 255) & ~std::size_t(255),
        rhino_lkn::read_write, rhino_lkn::kStride256B, 256);
    require(result->ddr_pointer(), "DDR allocation returned null mapping");
    d.address = result->ddr_memory->get_rpu_addr(); abi::detail::ddr_span(d.address, d.bytes);
  }
  return result;
}
void tp_lower(TpNode& node) {
  auto& r = node.request;
  if (tp::lower_copy(r, node.lowered)) {
    // DMA has no kernel entry or candidate ABI to load.
  } else if (r.op == "silu_mul_fp16") {
    const auto& out = r.tensor("out"); const auto& x = r.tensor("x"); const auto& y = r.tensor("y");
    for (auto* t : {&out, &x, &y}) { tp::fp16(*t); tp::layout(*t, "row_major"); }
    tp::require(out.shape == x.shape && x.shape == y.shape && out.distribution == x.distribution &&
        x.distribution == y.distribution && out.shard_axis == x.shard_axis && x.shard_axis == y.shard_axis,
        "SiLU tensor shape/distribution mismatch");
    tp::require(x.distribution == "local" || x.distribution == "replicated" || x.distribution == "sharded",
                "SiLU cannot consume an unreduced partial or single-core binding");
    tp::writable(out);
    const auto elements = static_cast<std::uint32_t>(tp::elements(x));
    const abi::SiluMulArgs args{tp::spm(out), tp::spm(x), tp::spm(y), elements, 8};
    node.lowered.plan = abi::silu_mul(args);
    node.lowered.writes = {"out"};
  } else require(tp::lower_operator(r, node.lowered), "unsupported TP operator");
  r.finish();
  for (const auto& name : node.lowered.writes) {
    const auto& t = node.tensors.at(name); tp::writable(t->desc);
    if (node.lowered.transfer_mode < 0 && node.lowered.plan.cores == 1 && t->desc.storage == "spm")
      require(t->desc.distribution == "single", "single-core output must have single distribution");
  }
}
void tp_check_node(const TpNode& node) {
  for (const auto& item : node.generations)
    tp::require(node.tensors.at(item.first)->desc.generation == item.second,
                "stale prepared state generation: " + item.first);
  for (const auto& rule : node.request.value_requirements) {
    const auto& d = node.tensors.at(rule.tensor)->desc;
    tp::require(d.values_known && d.finite && d.minimum >= rule.minimum && d.maximum <= rule.maximum,
                "replay input violates value constraint: " + rule.tensor);
  }
}
void tp_append(TpBatch& batch, rhino_lkn::Queue_t& queue,
               const abi::LaunchPlan& plan, const std::vector<unsigned>& counters,
               const std::shared_ptr<TpNode>& node = nullptr, int counter_slot = -1,
               int window_index = -1) {
  const char* entry = abi::name(plan.entry);
  auto program = batch.context->programs[entry].lock();
  if (!program) {
    tp_require_descriptor_headroom();
    program = std::make_shared<rhino_lkn::Program_t>();
    if (load_program(*program, entry, batch.context->allow_candidate))
      throw std::runtime_error(std::string("TP prepare ") + entry + ": " + last_error());
    batch.context->programs[entry] = program;
  }
  auto kernel = std::make_unique<rhino_lkn::Kernel_t>(*program, entry);
  check(abi::apply(*kernel, plan, false), "TP ABI binding");
  for (unsigned axis = 0; axis < 3; ++axis)
    check(kernel->set_regs(64 + axis, plan.grid[axis]), "TP grid register");
  for (auto counter : counters) {
    require(counter >= 128 && counter <= 254 && counter % 2 == 0, "invalid TP atomic register ID");
    check(kernel->set_regs(counter, std::uint32_t{0}), "TP counter reset packet");
  }
  require(plan.cores == 1 || plan.cores == 8, "TP adapter supports core0 or fixedTP8");
  const std::vector<std::uint8_t> core_ids = plan.cores == 8
      ? std::vector<std::uint8_t>{0,1,2,3,4,5,6,7} : std::vector<std::uint8_t>{0};
  const std::vector<std::uint16_t> grid{plan.grid[0],plan.grid[1],plan.grid[2]};
  if (node) {
    check(queue.add_kernel_mutable(*kernel, grid, core_ids), "TP add mutable kernel");
    batch.kernel_bindings.push_back({node, kernel.get(), batch.segments.size(), counter_slot,
                                     window_index, counters, node->revision});
  } else check(queue.add_kernel_to_stream(*kernel, grid, core_ids, 0), "TP add marker kernel");
  batch.programs.push_back(std::move(program)); batch.kernels.push_back(std::move(kernel));
}
template<class F>
void tp_copy_endpoints(const TpNode& node, unsigned c, F&& append) {
  const auto& src = *node.tensors.at("input");
  const auto& dst = *node.tensors.at("out");
  const int mode = node.lowered.transfer_mode;
  const bool src_spm = src.desc.storage == "spm", dst_spm = dst.desc.storage == "spm";
  const auto bytes = src_spm ? src.desc.bytes : dst.desc.bytes;
  const auto transfer_bytes = mode == 3 ? src.desc.bytes : bytes;
  const auto source_offset = src.offset + (!src_spm && mode == 0 ? c * transfer_bytes : 0);
  const auto target_offset = dst.offset + (!dst_spm && mode == 0 ? c * transfer_bytes : 0);
  if (src_spm && dst_spm)
    append(*src.context->arenas[c], source_offset, *dst.context->arenas[c], target_offset, transfer_bytes);
  else if (src_spm)
    append(*src.context->arenas[c], source_offset, dst.ddr_buffer(), target_offset, transfer_bytes);
  else if (dst_spm)
    append(src.ddr_buffer(), source_offset, *dst.context->arenas[c], target_offset, transfer_bytes);
  else append(src.ddr_buffer(), source_offset, dst.ddr_buffer(), target_offset, transfer_bytes);
}
void tp_append_copy(TpBatch& batch, rhino_lkn::Queue_t& queue,
                    const std::shared_ptr<TpNode>& node) {
  const int mode = node->lowered.transfer_mode;
  const unsigned copies = mode == 0 || mode == 1 ? 8 : 1;
  auto append = [&](unsigned c, int channel) {
    tp_copy_endpoints(*node, c, [&](const auto& src, auto src_offset, auto& dst, auto dst_offset, auto size) {
      check(queue.add_dma_kernel_mutable(src, src_offset, dst, dst_offset, size, channel), "TP append mutable DMA");
    });
    batch.dma_bindings.push_back({node, batch.segments.size(), batch.next_dma_id++, c, node->revision});
  };
  if (copies == 1) {
    // The original's single-transfer discipline: channel 0 shares the
    // compute stream (stream 0), so program order already serializes the
    // transfer against adjacent compute launches — no barrier packets at
    // all (rpu_memcpy's plain-DMA convention; the model's vision
    // segment runs its lone DMAs on channel 0 the same way).
    append(0, 0);
    batch.dma_count += 1;
    return;
  }
  // The original's canonical burst: per-core copy c rides channel c
  // (stream 2*c), so eight DMA engines transfer concurrently instead of
  // serializing one channel. Channel 0's copy stays on the compute stream
  // and needs no fence; fence every other DMA stream against compute on
  // both sides of the burst (7+7 events, rpu_memcpy's canonical form).
  for (unsigned c = 1; c < copies; ++c) queue.add_barrier(2 * c, 0);
  for (unsigned c = 0; c < copies; ++c) append(c, static_cast<int>(c));
  for (unsigned c = 1; c < copies; ++c) queue.add_barrier(0, 2 * c);
  batch.dma_count += copies;
  batch.barrier_count += 2 * (copies - 1);
}
void tp_sync_bindings(TpBatch& batch) {
  std::set<std::size_t> changed;
  for (auto& binding : batch.kernel_bindings) {
    if (binding.revision == binding.node->revision) continue;
    auto plan = binding.node->lowered.plan;
    if (binding.window_index >= 0) plan = abi::collective_window(plan, binding.window_index);
    else if (binding.counter_slot >= 0)
      // The build assigned this epoch a dedicated pool slot (50..3);
      // recover it from the stored zeroing registers {0,1,own}.
      plan.regs[binding.counter_slot] = (binding.counters.back() - 128) / 2;
    check(abi::apply(*binding.kernel, plan, false), "TP update kernel binding");
    for (unsigned axis = 0; axis < 3; ++axis)
      check(binding.kernel->set_regs(64 + axis, plan.grid[axis]), "TP update grid");
    for (auto counter : binding.counters)
      check(binding.kernel->set_regs(counter, std::uint32_t{0}), "TP update counter reset");
    changed.insert(binding.segment);
  }
  for (auto& binding : batch.dma_bindings) {
    if (binding.revision == binding.node->revision) continue;
    auto& queue = *batch.segments[binding.segment];
    tp_copy_endpoints(*binding.node, binding.core,
        [&](const auto& src, auto src_offset, auto& dst, auto dst_offset, auto size) {
          check(queue.update_dma_kernel(binding.id, src, src_offset, dst, dst_offset, size),
                "TP update DMA binding");
        });
    changed.insert(binding.segment);
  }
  for (auto index : changed)
    check(batch.segments[index]->sync_mutable_params(), "TP sync mutable parameters");
  for (auto& binding : batch.kernel_bindings) binding.revision = binding.node->revision;
  for (auto& binding : batch.dma_bindings) binding.revision = binding.node->revision;
}
std::uint16_t tp_completion_tag(std::size_t segment) {
  // Both execution modes admit at most 1024 SDK segments. Window mode counts
  // its extra initialization segments before allocating any SDK queue.
  // These raw FP16 encodings are distinct, finite, and nonzero (1..~2).
  require(segment < 1024, "TP completion tag exceeds the SDK segment bound");
  return static_cast<std::uint16_t>(0x3c00 + segment);
}
void tp_verify_completion(TpBatch& batch, std::uint16_t expected) {
  std::array<std::uint16_t, 16> marker;
  bool complete = true;
  for (unsigned core = 0; core < 8; ++core) {
    tp_read(marker.data(), batch.completion->spm_pointer(core), sizeof(marker));
    complete = complete && std::all_of(marker.begin(), marker.end(),
                                      [expected](auto v) { return v == expected; });
  }
  if (!complete) {
    throw std::runtime_error("RPU TP batch did not reach its completion marker on every core; "
                             "all cores must publish the expected marker before results are consumed");
  }
}
}  // namespace

Handle tp_context_create(bool allow_candidate) {
  return tp_context_create_with_pool(allow_candidate, 128);
}
Handle tp_context_create_with_pool(bool allow_candidate, std::int64_t retained_segments) {
  require(registry().process == ::getpid(), "RPU handles cannot be used after fork");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  require(retained_segments >= 128 && retained_segments <= 8192 && retained_segments % 64 == 0,
          "retained segment pool must be a multiple of 64 in [128,8192]");
  // Prepared collective segments outlive individual submissions, so the
  // arena pool must be configured before weight/program loading (idempotent).
  // Windowed body segments at model scale exceed the small class' 256KB
  // instruction budget and would exhaust the scarce large arenas; whole-stage
  // queues (one fused queue per stage, ~2-4MB of instructions) need a class
  // at the model's own max-buildable-segment size. Sixteen 4MB/32MB arenas
  // cover the whole-stage request shape (vision + two cached prefix
  // signatures + up to eight action buckets) with spare; the 8-64MB class
  // stays as fallback.
  const rhino_lkn::GraphArenaRequirement requirements[] = {
      {64U * 1024U, 256U * 1024U, static_cast<std::uint32_t>(retained_segments), 0},
      {256U * 1024U, 2U * 1024U * 1024U, 32, 0},
      {4U * 1024U * 1024U, 32U * 1024U * 1024U, 16, 0},
      {8U * 1024U * 1024U, 64U * 1024U * 1024U, 4, 0}};
  const auto pool_status = rhino_lkn::ConfigureGraphArenaPool(requirements, 4);
  if (pool_status != rhino_lkn::GRAPH_ARENA_POOL_SUCCESS)
    throw std::runtime_error("RPU TP graph arena configuration failed (SDK status " +
        std::to_string(pool_status) + "); create the TP context before loading SDK weights/programs");
  return insert(std::make_shared<TpContext>(allow_candidate));
}
Handle tp_empty(Handle ctx, const std::vector<std::int64_t>& dims, const std::string& dtype,
    const std::string& storage, const std::string& layout_name, const std::string& distribution_name,
    std::int64_t axis, const std::string& state_kind) {
  auto context = get<TpContext>(ctx, Kind::TpContext, "TP context");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  require(layout_name != "tp8_nk16" && layout_name != "nk16", "packed weights require prepare_weight");
  auto t = tp_allocate(context, dims, dtype, storage, layout_name, distribution_name, axis, state_kind);
  const auto id = insert(t); t->desc.identity = id; t->aliases->push_back(t); return id;
}
Handle tp_view(Handle id, const std::vector<std::int64_t>& dims,
    std::int64_t offset, const std::string& distribution, std::int64_t axis) {
  auto parent = get<TpTensor>(id, Kind::TpTensor, "TP tensor");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  auto descriptor = tp::contiguous_view(parent->desc, dims, offset, distribution, axis);
  auto view = std::make_shared<TpTensor>(parent->context);
  view->owner = parent;
  view->desc = std::move(descriptor);
  view->offset = parent->offset + static_cast<std::size_t>(offset);
  view->aliases = parent->aliases;
  view->aliases->push_back(view);
  return insert(std::move(view));
}
void tp_upload(Handle id, const void* data, const std::vector<std::int64_t>& dims, const std::string& dtype) {
  auto t = get<TpTensor>(id, Kind::TpTensor, "TP tensor");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  require(data && dtype == t->desc.dtype && !t->desc.immutable, "invalid upload dtype/data or immutable tensor");
  require(t->desc.layout == "row_major" || tp::is_cache_layout(t->desc.layout), "unsupported upload layout");
  const auto bytes = t->desc.bytes, count = tp::elements(t->desc);
  if (t->desc.storage == "ddr") {
    require(dims == t->desc.shape, "DDR upload shape mismatch");
    // A failed CPU-access flush can follow a partial or complete write. Never
    // leave the previous upload proof or initialized state range usable.
    tp_invalidate_contents(*t);
    check(t->ddr_buffer().begin_cpu_access(t->offset, bytes), "DDR begin write");
    if (tp::is_cache_layout(t->desc.layout)) {
      const auto* src = static_cast<const std::uint16_t*>(data);
      auto* dst = static_cast<std::uint16_t*>(t->ddr_pointer());
      for (unsigned c = 0; c < 8; ++c)
        for (unsigned row = 0; row < t->desc.shape[1]; ++row)
          for (unsigned col = 0; col < t->desc.shape[2]; ++col)
            dst[tp::cache_offset(t->desc, c, row, col)] = src[(c * t->desc.shape[1] + row) * t->desc.shape[2] + col];
    } else std::memcpy(t->ddr_pointer(), data, bytes);
    check(t->ddr_buffer().end_cpu_access(t->offset, bytes), "DDR end write");
    tp_stats(*t, data, count);
  } else {
    auto expanded = t->desc.shape; expanded.insert(expanded.begin(), 8);
    const bool per_core = dims == expanded;
    require(per_core || dims == t->desc.shape, "SPM upload requires local shape or [8,*local_shape]");
    const auto* source = static_cast<const unsigned char*>(data);
    if (per_core && t->desc.distribution == "replicated")
      for (unsigned c = 1; c < 8; ++c)
        require(std::memcmp(source, source + c * bytes, bytes) == 0, "replicated upload differs across cores");
    if (per_core && (t->desc.distribution == "owner0" || t->desc.distribution == "single"))
      for (std::size_t i = bytes; i < 8 * bytes; ++i)
        require(source[i] == 0, "single/owner0 upload requires zero nonowner cores");
    std::vector<unsigned char> zero(bytes, 0);
    tp_invalidate_contents(*t);
    for (unsigned c = 0; c < 8; ++c) {
      const auto* src = source + (per_core ? c * bytes : 0);
      if (c && (t->desc.distribution == "owner0" || t->desc.distribution == "single")) src = zero.data();
      tp_write(t->spm_pointer(c), src, bytes);
    }
    tp_stats(*t, data, count * (per_core ? 8 : 1));
    if (t->desc.distribution == "owner0" || t->desc.distribution == "single") {
      t->desc.minimum = std::min(t->desc.minimum, 0.0); t->desc.maximum = std::max(t->desc.maximum, 0.0);
    }
  }
  if (t->desc.state_kind != "none") t->valid_ranges = {{0, tp_state_capacity(*t)}};
}
void tp_download(void* data, Handle id, const std::vector<std::int64_t>& dims, const std::string& dtype) {
  auto t = get<TpTensor>(id, Kind::TpTensor, "TP tensor");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  require(data && dtype == t->desc.dtype, "invalid download data or dtype");
  require(t->desc.layout == "row_major" || tp::is_cache_layout(t->desc.layout), "packed weights have no row-major download");
  const auto bytes = t->desc.bytes;
  if (t->desc.storage == "ddr") {
    require(dims == t->desc.shape, "DDR download shape mismatch");
    check(t->ddr_buffer().begin_cpu_access(t->offset, bytes), "DDR begin read");
    // Refresh CPU-visible data after device completion and before reading.
    rhino_lkn::RpuDdrFlushDirect(t->ddr_pointer(), bytes);
    if (tp::is_cache_layout(t->desc.layout)) {
      const auto* src = static_cast<const std::uint16_t*>(t->ddr_pointer());
      auto* dst = static_cast<std::uint16_t*>(data);
      for (unsigned c = 0; c < 8; ++c)
        for (unsigned row = 0; row < t->desc.shape[1]; ++row)
          for (unsigned col = 0; col < t->desc.shape[2]; ++col)
            dst[(c * t->desc.shape[1] + row) * t->desc.shape[2] + col] = src[tp::cache_offset(t->desc, c, row, col)];
    } else std::memcpy(data, t->ddr_pointer(), bytes);
    check(t->ddr_buffer().end_cpu_access(t->offset, bytes), "DDR end read");
  } else {
    auto expanded = t->desc.shape; expanded.insert(expanded.begin(), 8);
    require(dims == expanded, "SPM download must preserve all eight physical cores");
    for (unsigned c = 0; c < 8; ++c)
      tp_read(static_cast<unsigned char*>(data) + c * bytes, t->spm_pointer(c), bytes);
  }
}
Handle tp_weight(Handle ctx, const void* data, const std::vector<std::int64_t>& dims, const std::string& packing) {
  auto context = get<TpContext>(ctx, Kind::TpContext, "TP context");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  require(data && (packing == "tp8_nk16" || packing == "nk16" || packing == "row_major"), "unsupported prepared weight layout");
  if (packing == "row_major") {
    auto t = tp_allocate(context, dims, "float16", "ddr", packing, "shared", -1, "none");
    check(t->ddr_memory->begin_cpu_access(0, t->desc.bytes), "table begin write");
    std::memcpy(t->ddr_pointer(), data, t->desc.bytes);
    check(t->ddr_memory->end_cpu_access(0, t->desc.bytes), "table end write");
    t->desc.immutable = true; tp_stats(*t, data, tp::elements(t->desc));
    const auto id = insert(t); t->desc.identity = id; t->aliases->push_back(t); return id;
  }
  const bool tp8 = packing == "tp8_nk16";
  require((tp8 && (dims.size() == 3 || dims.size() == 4)) || (!tp8 && dims.size() == 2), "prepared weight rank mismatch");
  const auto n = dims[dims.size()-2], k = dims.back();
  require(n > 0 && k > 0 && n % 16 == 0 && k % 16 == 0, "weight N/K must be positive multiples of16");
  if (tp8) require(dims[dims.size()-3] == 8, "TP weight requires exactly8 core shards");
  auto t = tp_allocate(context, dims, "float16", "ddr", packing, "shared", -1, "none");
  const auto planes = dims.size() == 4 ? dims[0] : 1;
  const int cores = tp8 ? 8 : 1;
  const auto* source = static_cast<const std::uint16_t*>(data);
  if (dims == std::vector<std::int64_t>{8,1056,2048})
    for (unsigned c = 0; c < 8; ++c)
      for (unsigned row = 1026; row < 1056; ++row) {
        if (row == 1040 || row == 1041) continue;
        for (unsigned col = 0; col < 2048; ++col)
          require((source[(c * 1056 + row) * 2048 + col] & 0x7fff) == 0,
                  "Language ZBA weight padding rows must be zero");
      }
  auto* target = static_cast<std::uint16_t*>(t->ddr_pointer());
  check(t->ddr_memory->begin_cpu_access(0, t->desc.bytes), "weight begin write");
  for (std::int64_t p = 0; p < planes; ++p)
    for (std::int64_t bn = 0; bn < n / 16; ++bn)
      for (std::int64_t bk = 0; bk < k / 16; ++bk)
        for (int c = 0; c < cores; ++c)
          for (int nn = 0; nn < 16; ++nn)
            for (int kk = 0; kk < 16; ++kk) {
              const auto dst = (((((p * (n/16) + bn) * (k/16) + bk) * cores + c) * 16 + nn) * 16 + kk);
              target[dst] = source[((p * cores + c) * n + bn * 16 + nn) * k + bk * 16 + kk];
            }
  check(t->ddr_memory->end_cpu_access(0, t->desc.bytes), "weight end write");
  t->desc.immutable = true; tp_stats(*t, data, tp::elements(t->desc));
  const auto id = insert(t); t->desc.identity = id; t->aliases->push_back(t); return id;
}
void tp_reset(Handle id) {
  auto t = get<TpTensor>(id, Kind::TpTensor, "TP tensor");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  require(t->desc.state_kind != "none" && !t->desc.immutable, "reset requires a mutable state/cache/history owner");
  require(t->desc.generation < std::numeric_limits<std::uint64_t>::max(), "state generation exhausted");
  std::vector<unsigned char> zeros(t->desc.bytes, 0);
  // Even a failed reset may modify memory. Invalidate captured plans before
  // writing, and publish initialized zero contents only after all writes pass.
  ++t->desc.generation;
  tp_invalidate_contents(*t);
  if (t->desc.storage == "spm") {
    for (unsigned c = 0; c < 8; ++c) tp_write(t->spm_pointer(c), zeros.data(), t->desc.bytes);
  } else {
    check(t->ddr_memory->begin_cpu_access(0, t->desc.bytes), "state begin reset");
    std::memcpy(t->ddr_pointer(), zeros.data(), t->desc.bytes);
    check(t->ddr_memory->end_cpu_access(0, t->desc.bytes), "state end reset");
  }
  t->desc.values_known = t->desc.finite = true;
  t->desc.minimum = t->desc.maximum = 0;
  if (t->desc.state_kind != "cache") t->valid_ranges.emplace_back(0, 1);
}
void tp_invalidate(Handle id) {
  auto t = get<TpTensor>(id, Kind::TpTensor, "TP tensor");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  require(t->desc.state_kind != "none" && !t->desc.immutable,
          "invalidate requires a mutable state/cache/history owner");
  // A new request invalidates logical contents without touching storage or
  // changing captured addresses/generations. A retained initial producer must
  // establish every interval before any reader; reset remains a separate,
  // generation-changing recovery operation.
  tp_invalidate_contents(*t);
}
// Rewrite a family request (gemm, rmsnorm, ...) into its catalog entry before
// validation. The node keeps the caller's family and argument names so that
// rebind accepts the same names prepare did.
void tp_resolve_family(TpNode& node) {
  auto& r = node.request;
  if (!tp::is_family(r.op)) return;
  node.family = r.op;
  std::map<std::string, std::string> renames;  // caller name -> catalog name
  tp::resolve_family(r, &renames, &node.selectors);
  // Aliased bindings (out=x) are distinct names for one tensor; rename by name.
  std::map<std::string, std::shared_ptr<TpTensor>> tensors;
  std::map<std::string, std::uint64_t> generations;
  for (const auto& item : node.tensors) {
    const auto found = renames.find(item.first);
    const auto& name = found == renames.end() ? item.first : found->second;
    node.argument_names.emplace(item.first, name);
    tensors.emplace(name, item.second);
    const auto captured = node.generations.find(item.first);
    if (captured != node.generations.end()) generations.emplace(name, captured->second);
  }
  node.tensors = std::move(tensors);
  node.generations = std::move(generations);
}
// Admission belongs to the kernel that will run, not to the name the caller
// passed: a family call or an automatic variant (K128) may select a
// candidate kernel. Reject it here instead of at the first batch.
// Admission is based on the selected kernel: family calls and automatic variants
// may resolve from a named entry to its candidate kernel.
void tp_require_admission(const TpNode& node) {
  if (node.lowered.transfer_mode >= 0) return;  // DMA copies load no kernel.
  const std::string selected = abi::name(node.lowered.plan.entry);
  for (std::size_t i = 0; i < kernel_count(); ++i) {
    if (selected != kernel_name(i)) continue;
    const std::string status = kernel_execution_status(i);
    tp::require(status != "quarantined", "TP prepare " + selected + ": kernel is quarantined");
    const std::string admission = kernel_admission(i);
    tp::require(admission != "candidate" || node.context->allow_candidate,
            "TP prepare " + node.request.op + ": selects candidate kernel " + selected +
            "; create the context with allow_candidate=True");
    return;
  }
}
Handle tp_prepare(Handle ctx, const std::string& op, const std::vector<std::string>& names,
    const std::vector<Handle>& ids, const std::vector<std::string>& ikeys,
    const std::vector<std::int64_t>& ints, const std::vector<std::string>& fkeys,
    const std::vector<double>& floats) {
  auto context = get<TpContext>(ctx, Kind::TpContext, "TP context");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  require(names.size() == ids.size() && ikeys.size() == ints.size() && fkeys.size() == floats.size(), "TP named parameter count mismatch");
  auto node = std::make_shared<TpNode>(context); node->request.op = op;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    require(names[i].rfind("__xinfer_", 0) != 0, "reserved internal tensor argument");
    auto t = get<TpTensor>(ids[i], Kind::TpTensor, "TP tensor");
    require(t->context == context, "TP tensor belongs to another context");
    require(node->tensors.emplace(names[i], t).second, "duplicate TP tensor name");
    node->request.tensors.emplace(names[i], t->desc);
    if (t->desc.state_kind != "none") node->generations.emplace(names[i], t->desc.generation);
  }
  for (std::size_t i = 0; i < ints.size(); ++i)
    require(node->request.integers.emplace(ikeys[i], ints[i]).second, "duplicate TP integer name");
  for (std::size_t i = 0; i < floats.size(); ++i)
    require(node->request.scalars.emplace(fkeys[i], floats[i]).second, "duplicate TP scalar name");
  tp_resolve_family(*node);
  tp_lower(*node);
  tp_require_admission(*node);
  return insert(std::move(node));
}
std::vector<std::string> tp_selection(Handle id) {
  auto node = get<TpNode>(id, Kind::TpNode, "TP operation");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex);
  if (node->lowered.transfer_mode >= 0) return {"dma", node->request.op, node->family, ""};
  const std::string kernel = abi::name(node->lowered.plan.entry);
  std::string admission;
  for (std::size_t i = 0; i < kernel_count(); ++i)
    if (kernel == kernel_name(i)) {
      admission = kernel_admission(i);
      break;
    }
  return {kernel, node->request.op, node->family, admission};
}
void tp_rebind(Handle id, const std::vector<std::string>& caller_names,
    const std::vector<Handle>& ids, const std::vector<std::string>& ikeys,
    const std::vector<std::int64_t>& ints, const std::vector<std::string>& fkeys,
    const std::vector<double>& floats) {
  auto node = get<TpNode>(id, Kind::TpNode, "TP operation");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  // A family operation accepts the argument names its prepare() used.
  std::vector<std::string> names = caller_names;
  if (!node->family.empty()) {
    for (auto& name : names) {
      const auto found = node->argument_names.find(name);
      tp::require(found != node->argument_names.end(),
              node->family + " rebind: unknown tensor argument " + name);
      name = found->second;
    }
    for (const auto& key : ikeys)
      tp::require(!node->selectors.count(key),
              node->family + " rebind: " + key + " selects the kernel and is fixed at prepare");
  }
  require(names.size() == ids.size() && ikeys.size() == ints.size() && fkeys.size() == floats.size(),
          "TP rebind named parameter count mismatch");
  require(node->revision != std::numeric_limits<std::uint64_t>::max(), "operation revision exhausted");
  // Reset cannot be repaired by rebinding a captured generation. Its entire
  // stateful plan must be rebuilt explicitly.
  for (const auto& captured : node->generations)
    tp::require(node->tensors.at(captured.first)->desc.generation == captured.second,
                "stale prepared state generation: " + captured.first);
  TpNode candidate(node->context);
  candidate.request.op = node->request.op;
  candidate.request.integers = node->request.integers;
  candidate.request.scalars = node->request.scalars;
  candidate.tensors = node->tensors;
  std::set<std::string> seen;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    require(names[i].rfind("__xinfer_", 0) != 0, "reserved internal tensor argument");
    require(seen.insert(names[i]).second, "duplicate rebind tensor");
    require(candidate.tensors.count(names[i]), "rebind cannot add a new tensor argument");
    auto t = get<TpTensor>(ids[i], Kind::TpTensor, "TP tensor");
    require(t->context == node->context, "rebound tensor belongs to another context");
    const auto& previous = node->tensors.at(names[i]);
    require((previous->desc.state_kind == "none" && t->desc.state_kind == "none") || t == previous,
            "rebind cannot replace a captured state owner");
    candidate.tensors[names[i]] = std::move(t);
  }
  seen.clear();
  for (std::size_t i = 0; i < ints.size(); ++i) {
    require(seen.insert(ikeys[i]).second, "duplicate rebind integer");
    candidate.request.integers[ikeys[i]] = ints[i];
  }
  seen.clear();
  for (std::size_t i = 0; i < floats.size(); ++i) {
    require(seen.insert(fkeys[i]).second, "duplicate rebind scalar");
    candidate.request.scalars[fkeys[i]] = floats[i];
  }
  for (const auto& item : candidate.tensors) candidate.request.tensors[item.first] = item.second->desc;
  candidate.generations = node->generations;
  // Binary fields are process-local and lossless. The memo retains its
  // tensors, so a cached instance address cannot be recycled for another view.
  std::string memo_key;
  memo_key.reserve(192);
  auto append = [&](const auto& value) {
    memo_key.append(reinterpret_cast<const char*>(&value), sizeof(value));
  };
  auto append_name = [&](const std::string& name) {
    append(name.size()); memo_key += name;
  };
  append_name(candidate.request.op);
  append(candidate.tensors.size());
  for (const auto& item : candidate.tensors) {
    append_name(item.first);
    const auto* instance = item.second.get();
    append(instance); append(instance->desc.generation);
  }
  append(candidate.request.integers.size());
  for (const auto& item : candidate.request.integers) {
    append_name(item.first); append(item.second);
  }
  append(candidate.request.scalars.size());
  for (const auto& item : candidate.request.scalars) {
    append_name(item.first); append(item.second);
  }
  for (const auto& memo : node->rebind_memos) {
    if (memo.key == memo_key) {
      // Uploads and device writes change value proofs without changing the
      // instance or generation. Recheck those live proofs before publishing.
      candidate.request.value_requirements = memo.request.value_requirements;
      tp_check_node(candidate);
      node->request = memo.request;
      node->lowered = memo.lowered;
      node->tensors = memo.tensors;
      ++node->revision;
      return;
    }
  }
  tp_lower(candidate);
  const auto& before = node->lowered;
  const auto& after = candidate.lowered;
  require(before.transfer_mode == after.transfer_mode && before.writes == after.writes &&
          before.collective == after.collective && before.zero_counters == after.zero_counters,
          "rebind cannot change operation writes, transfer mode or collective structure");
  if (before.transfer_mode < 0)
    require(before.plan.entry == after.plan.entry && before.plan.cores == after.plan.cores &&
            before.plan.grid == after.plan.grid && before.plan.register_count == after.plan.register_count,
            "rebind cannot change kernel entry, core set, grid or register ABI");
  else {
    for (const auto* name : {"input", "out"}) {
      const auto& a = node->tensors.at(name)->desc;
      const auto& b = candidate.tensors.at(name)->desc;
      require(a.shape == b.shape && a.storage == b.storage && a.distribution == b.distribution &&
                  a.shard_axis == b.shard_axis, "rebind cannot change DMA shape or TP identity");
    }
  }
  // Publish only after every contract check passes. Batches retain this node
  // and lazily synchronize their own mutable SDK packets before the next run.
  node->request = candidate.request;
  node->lowered = candidate.lowered;
  node->tensors = candidate.tensors;
  ++node->revision;
  if (node->rebind_memos.size() >= 8) node->rebind_memos.erase(node->rebind_memos.begin());
  node->rebind_memos.push_back(
      {std::move(memo_key), node->request, node->lowered, node->tensors});
}
Handle tp_batch(Handle ctx, const std::vector<Handle>& ids) {
  auto context = get<TpContext>(ctx, Kind::TpContext, "TP context");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  tp_require_batch_size(ids);
  auto batch = std::make_shared<TpBatch>(context);
  batch->completion = tp_allocate(context, {16}, "float16", "spm", "row_major", "local", -1, "none");
  const auto marker_address = static_cast<std::uint32_t>(batch->completion->desc.address);
  std::unique_ptr<rhino_lkn::Queue_t> current;
  auto finish_segment = [&]() {
    if (!current) return;
    const auto tag = tp_completion_tag(batch->segments.size());
    tp_append(*batch, *current, abi::fill({marker_address, 16, tag}), {});
    check(current->build_batch(), "TP build segment");
    batch->completion_tags.push_back(tag);
    batch->segments.push_back(std::move(current));
  };
  auto start_segment = [&]() {
    if (current) return;
    current = std::make_unique<rhino_lkn::Queue_t>(8);
    current->set_flush_icache(true); current->set_broadcast_mode(true);
    batch->next_dma_id = 0;
  };
  unsigned reduce_epoch = 0;
  std::set<std::uint64_t> prior_writes;
  for (auto id : ids) {
    auto node = get<TpNode>(id, Kind::TpNode, "TP operation");
    require(node->context == context, "TP operation belongs to another context"); tp_check_node(*node);
    for (const auto& rule : node->request.value_requirements)
      require(!prior_writes.count(node->tensors.at(rule.tensor)->desc.identity),
              "value-constrained input cannot depend on an unproven device-produced range");
    for (const auto& name : node->lowered.writes)
      prior_writes.insert(node->tensors.at(name)->desc.identity);
    auto plan = node->lowered.plan; auto counters = node->lowered.zero_counters;
    int global_slot = -1;
    if (plan.entry == abi::Entry::AllReduceResidual) global_slot = 12;
    else if (plan.entry == abi::Entry::AllReduceResidualLayerNormFusion) global_slot = 9;
    else if (plan.entry == abi::Entry::DownAllReduceRmsnormFusion ||
             plan.entry == abi::Entry::DownAllReduceRmsnormFusionM16 ||
             plan.entry == abi::Entry::DownAllReduceRmsnormHwM16) global_slot = 24;
    if (global_slot >= 0) {
      require(reduce_epoch < 48, "TP batch exceeds48 all-reduce epochs");
      // Static counter pool discipline: assign counters 50..3 to launch
      // epochs in descending order. Each epoch's zeroing prefix reaches only
      // its own lower slot, so no later launch can disturb an earlier
      // in-flight counter; the shared-50 reset across a long batch is what
      // this excludes. Reuse across batches remains graph-ordered.
      const unsigned global = static_cast<unsigned>(50 - reduce_epoch);
      ++reduce_epoch;
      plan.regs[global_slot] = global;
      counters = {128, 130, 128 + 2 * global};
    }
    if (node->lowered.collective) finish_segment();
    start_segment();
    if (node->lowered.transfer_mode >= 0) tp_append_copy(*batch, *current, node);
    else tp_append(*batch, *current, plan, counters, node, global_slot);
    if (node->lowered.collective) finish_segment();
    batch->nodes.push_back(std::move(node));
  }
  finish_segment();
  require(batch->completion_tags.size() == batch->segments.size() &&
              batch->segments.size() <= ids.size(), "TP segment completion metadata mismatch");
  batch->collective_count = reduce_epoch;
  return insert(std::move(batch));

}
std::int64_t tp_ar_window_max() {
  // Calibrated AR collective-window capability; 0 means unavailable.
  // The query does not initialize a device or materialize a Program_t.
  for (std::size_t i = 0; i < kernel_count(); ++i) {
    if (std::string(kernel_name(i)) == "all_reduce_residual_window") {
      const char* status = kernel_execution_status(i);
      const char* admission = kernel_admission(i);
      return status && !status[0] && admission &&
          std::string(admission) == "candidate" ? 15 : 0;
    }
  }
  return 0;
}
std::int64_t tp_down_norm_window_max() {
  // Calibrated M16 DownNorm window capability (windows 2..4); 0 means
  // unavailable. Only the HW window kernel qualifies the cap now.
  // The query does not initialize a device or materialize a Program_t.
  const char* name = "down_allreduce_rmsnorm_hw_window_m16n1024k256_acc32";
  std::int64_t cap = 0;
  for (std::size_t i = 0; i < kernel_count(); ++i) {
    if (std::string(kernel_name(i)) == name) {
      const char* status = kernel_execution_status(i);
      const char* admission = kernel_admission(i);
      if (status && !status[0] && admission &&
          std::string(admission) == "candidate")
        cap = std::max(cap, std::int64_t(4));
    }
  }
  return cap;
}
Handle tp_batch_with_window(Handle ctx, const std::vector<Handle>& ids,
                            std::int64_t collective_window) {
  auto context = get<TpContext>(ctx, Kind::TpContext, "TP context");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  require(context->allow_candidate, "collective windows require explicit candidate admission");
  require(collective_window >= 2 && collective_window <= 15,
          "collective_window must be in [2,15]");
  const auto ar_cap = tp_ar_window_max();
  const auto m16_cap = tp_down_norm_window_max();
  require(ar_cap >= collective_window || m16_cap >= collective_window,
          "backend lacks calibrated AR or M16 DownNorm collective-window capability");
  tp_require_batch_size(ids);
  auto counter_slot = [](abi::Entry entry) {
    if (entry == abi::Entry::AllReduceResidual) return 12;
    if (entry == abi::Entry::AllReduceResidualLayerNormFusion) return 9;
    if (entry == abi::Entry::DownAllReduceRmsnormFusion ||
        entry == abi::Entry::DownAllReduceRmsnormFusionM16 ||
        entry == abi::Entry::DownAllReduceRmsnormHwM16) return 24;
    return -1;
  };
  auto eligible = [](abi::Entry entry) {
    return entry == abi::Entry::AllReduceResidual ||
        entry == abi::Entry::DownAllReduceRmsnormHwM16;
  };
  struct Segment {
    std::vector<std::shared_ptr<TpNode>> nodes;
    std::vector<int> indices;
    unsigned occurrences = 0;
  };
  std::vector<Segment> layout;
  Segment current;
  auto finish_layout = [&] {
    if (current.nodes.empty()) return;
    layout.push_back(std::move(current)); current = {};
  };
  unsigned reductions = 0;
  bool has_ar = false, has_m16 = false;
  std::set<std::uint64_t> prior_writes;
  // Validate the complete graph, counter and segment budgets before creating
  // a completion owner, loading a program or allocating an SDK queue.
  for (auto id : ids) {
    auto node = get<TpNode>(id, Kind::TpNode, "TP operation");
    require(node->context == context, "TP operation belongs to another context");
    tp_check_node(*node);
    for (const auto& rule : node->request.value_requirements)
      require(!prior_writes.count(node->tensors.at(rule.tensor)->desc.identity),
              "value-constrained input cannot depend on an unproven device-produced range");
    for (const auto& name : node->lowered.writes)
      prior_writes.insert(node->tensors.at(name)->desc.identity);
    const auto entry = node->lowered.plan.entry;
    if (entry == abi::Entry::AllReduceResidual) {
      has_ar = true;
      require(ar_cap >= collective_window, "backend lacks requested calibrated AR collective-window capability");
    }
    if (entry == abi::Entry::DownAllReduceRmsnormHwM16) {
      has_m16 = true;
      require(m16_cap >= collective_window, "backend lacks requested calibrated M16 DownNorm collective-window capability");
    }
    require(!(has_ar && has_m16), "collective windows cannot mix AR and M16 DownNorm in one batch");

    if (counter_slot(entry) >= 0) {
      require(reductions < 48, "TP batch exceeds48 all-reduce epochs");
      ++reductions;
    }
    const bool can_window = eligible(entry);
    const bool hard_boundary = !can_window && (node->lowered.collective ||
        std::any_of(node->lowered.zero_counters.begin(), node->lowered.zero_counters.end(),
                    [](unsigned reg) { return reg > 130; }));
    if (hard_boundary) {
      finish_layout();
      layout.push_back({{node}, {-1}, 0});
      continue;
    }
    if (can_window && current.occurrences == static_cast<unsigned>(collective_window))
      finish_layout();
    current.nodes.push_back(node);
    current.indices.push_back(can_window ? static_cast<int>(current.occurrences++) : -1);
  }
  finish_layout();
  std::size_t segments = layout.size();
  for (const auto& segment : layout) segments += segment.occurrences != 0;
  require(segments <= 1024, "TP window execution exceeds1024 SDK segments including initialization");

  auto batch = std::make_shared<TpBatch>(context);
  batch->completion = tp_allocate(context, {16}, "float16", "spm", "row_major", "local", -1, "none");
  const auto marker = static_cast<std::uint32_t>(batch->completion->desc.address);
  auto queue = [&] {
    auto result = std::make_unique<rhino_lkn::Queue_t>(8);
    result->set_flush_icache(true); result->set_broadcast_mode(true);
    batch->next_dma_id = 0;
    return result;
  };
  auto retain = [&](std::unique_ptr<rhino_lkn::Queue_t> segment,
                    const std::vector<unsigned>& counters) {
    const auto tag = tp_completion_tag(batch->segments.size());
    tp_append(*batch, *segment, abi::fill({marker, 16, tag}), counters);
    check(segment->build_batch(), "TP build window segment");
    batch->completion_tags.push_back(tag);
    batch->segments.push_back(std::move(segment));
  };
  std::vector<unsigned> initialize;
  for (unsigned counter = 0; counter <= 50; ++counter) initialize.push_back(128 + 2 * counter);
  // Mirror of the layout pass reductions sequence, so segmented
  // collectives inside a windowed batch keep their assigned slot.
  unsigned epoch = 0;
  for (const auto& segment : layout) {
    if (segment.occurrences) {
      // Complete counter initialization in a separate queue before the body.
      // Body collectives preserve these counters. Unique local pairs 2..29
      // and global counters 34..48 avoid intra-window reuse.
      retain(queue(), initialize);
    }
    auto body = queue();
    for (std::size_t i = 0; i < segment.nodes.size(); ++i) {
      const auto& node = segment.nodes[i];
      auto plan = node->lowered.plan;
      auto counters = node->lowered.zero_counters;
      const auto slot = counter_slot(plan.entry);
      const auto index = segment.indices[i];
      if (index >= 0) {
        plan = abi::collective_window(plan, index);
        counters.clear();
      } else if (slot >= 0) {
        const unsigned global = static_cast<unsigned>(50 - epoch);
        plan.regs[slot] = global;
        counters = {128, 130, 128 + 2 * global};
      }
      if (slot >= 0) ++epoch;
      if (node->lowered.transfer_mode >= 0) tp_append_copy(*batch, *body, node);
      else tp_append(*batch, *body, plan, counters, node, slot, index);
      batch->nodes.push_back(node);
    }
    retain(std::move(body), {});
  }
  require(batch->segments.size() == segments && batch->nodes.size() == ids.size(),
          "TP window segment metadata mismatch");
  batch->collective_count = reductions;
  return insert(std::move(batch));
}
// Batch-wide ordering must be checked again after a node is rebound.
unsigned tp_validate_fused(const std::vector<std::shared_ptr<TpNode>>& nodes) {
  auto fused_counter_slot = [](abi::Entry entry) {
    if (entry == abi::Entry::DownAllReduceRmsnormHwM16) return 24;
    if (entry == abi::Entry::AllReduceResidualLayerNormFusion) return 9;
    if (entry == abi::Entry::AllReduceResidual) return 12;
    return -1;
  };
  // Whole-stage counter discipline (the model's action and vision
  // execution models): one retained queue for the entire stage,
  // per-occurrence static counter assignment from the counter pool pinned
  // by each request (action DownNorm 3..49; vision AR/ARR 3..50 — the
  // merger re-arms 50,49 after the tower's descending 50..3, the same-id
  // reuse distance the action queue already proved), per-launch zeroing
  // {0,1,own}, and no isolation segments or initialization sweeps.
  // Invariants checked below: consecutive emitting collectives never share
  // an id, and every collective reads — transitively — a tensor the chain
  // of the previous collective produced, so pool reuse and the locals 0/1
  // shared by every launch stay graph-ordered. Reads are approximated from
  // below by consumed tensors minus declared writes; a read-modify-write
  // alias therefore does not count as ordering evidence.
  std::set<std::uint64_t> prior_writes;
  std::set<std::uint64_t> descendants;
  unsigned collectives = 0;
  int previous_id = -1;
  bool gather_pending = false;
  for (const auto& node : nodes) {
    // The counterless owner-push all_gather is admitted only in the
    // original's own whole-stage shape: its gathered output is consumed
    // strictly behind the following DMA burst, which absorbs cross-core
    // skew exactly as the model's per-layer weight streaming does
    // (its vision segment runs all_gather barrier-free the same way).
    require(!gather_pending || node->lowered.transfer_mode >= 0,
            "fused all_gather must be immediately followed by its DMA burst");
    gather_pending = false;
    for (const auto& rule : node->request.value_requirements)
      require(!prior_writes.count(node->tensors.at(rule.tensor)->desc.identity),
              "value-constrained input cannot depend on an unproven device-produced range");
    for (const auto& name : node->lowered.writes)
      prior_writes.insert(node->tensors.at(name)->desc.identity);
    const auto slot = fused_counter_slot(node->lowered.plan.entry);
    bool joins = false;
    if (collectives != 0) {
      for (const auto& name : node->request.used_tensors) {
        if (std::find(node->lowered.writes.begin(), node->lowered.writes.end(), name) !=
            node->lowered.writes.end()) continue;
        if (descendants.count(node->tensors.at(name)->desc.identity)) { joins = true; break; }
      }
    }
    if (slot >= 0) {
      const auto assigned = static_cast<int>(node->lowered.plan.regs[slot]);
      // The action DownNorm queue pins 3..49; the vision tower walks the
      // full 50..3 and its merger re-arms 50,49, so the vision families
      // admit the whole counter pool.
      const int pool_hi = slot == 24 ? 49 : 50;
      require(assigned >= 3 && assigned <= pool_hi,
              "fused batch requires the request integer counter inside the family's counter pool "
              "(DownNorm 3..49, vision AR/ARR 3..50)");
      require(assigned != previous_id,
              "consecutive fused collective launches must not share a counter id");
      if (collectives != 0)
        require(joins, "fused collective reuse must stay downstream of the previous collective");
      previous_id = assigned;
      descendants.clear();
      for (const auto& name : node->lowered.writes)
        descendants.insert(node->tensors.at(name)->desc.identity);
      ++collectives;
    } else {
      const bool gather = node->lowered.plan.entry == abi::Entry::AllGather;
      require(gather || !node->lowered.collective,
              ("fused whole-stage batches support only the counted M16 DownNorm and vision "
               "AR/ARR families plus the counterless owner-push all_gather "
               "(offending op: " + node->request.op + ")").c_str());
      // Non-collective launches may zero only the transient locals 0/1
      // (encoded 128/130) — the per-launch scratch discipline
      // (qkv_rope zeroes {0,1}). Pool ids stay owned by the collective
      // occurrences; zeroing one here would clobber a pinned counter.
      for (const auto zeroed : node->lowered.zero_counters)
        require(zeroed == 128 || zeroed == 130,
                ("fused non-collective " + node->request.op +
                 " may zero only the transient counters 0 and 1").c_str());
      if (joins)
        for (const auto& name : node->lowered.writes)
          descendants.insert(node->tensors.at(name)->desc.identity);
      gather_pending = gather;
    }
  }
  require(!gather_pending, "fused all_gather must be immediately followed by its DMA burst");
  return collectives;
}
Handle tp_batch_fused(Handle ctx, const std::vector<Handle>& ids) {
  auto context = get<TpContext>(ctx, Kind::TpContext, "TP context");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  tp_require_batch_size(ids);
  require(context->allow_candidate, "fused whole-stage batches require explicit candidate admission");
  std::vector<std::shared_ptr<TpNode>> nodes;
  nodes.reserve(ids.size());
  for (auto id : ids) {
    auto node = get<TpNode>(id, Kind::TpNode, "TP operation");
    require(node->context == context, "TP operation belongs to another context");
    tp_check_node(*node);
    nodes.push_back(std::move(node));
  }
  const auto collectives = tp_validate_fused(nodes);
  auto batch = std::make_shared<TpBatch>(context);
  batch->completion = tp_allocate(context, {16}, "float16", "spm", "row_major", "local", -1, "none");
  const auto marker = static_cast<std::uint32_t>(batch->completion->desc.address);
  auto queue = std::make_unique<rhino_lkn::Queue_t>(8);
  queue->set_flush_icache(true); queue->set_broadcast_mode(true);
  batch->next_dma_id = 0;
  for (auto& node : nodes) {
    if (node->lowered.transfer_mode >= 0) tp_append_copy(*batch, *queue, node);
    else tp_append(*batch, *queue, node->lowered.plan, node->lowered.zero_counters, node);
    batch->nodes.push_back(std::move(node));
  }
  const auto tag = tp_completion_tag(0);
  tp_append(*batch, *queue, abi::fill({marker, 16, tag}), {});
  check(queue->build_batch(), "TP build fused segment");
  batch->completion_tags.push_back(tag);
  batch->segments.push_back(std::move(queue));
  require(batch->segments.size() == 1 && batch->nodes.size() == ids.size(),
          "TP fused segment metadata mismatch");
  batch->collective_count = collectives;
  batch->fused = true;
  return insert(std::move(batch));
}
std::vector<std::int64_t> tp_batch_stats(Handle id) {
  auto batch = get<TpBatch>(id, Kind::TpBatch, "TP batch");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  return {static_cast<std::int64_t>(batch->nodes.size()),
          static_cast<std::int64_t>(batch->kernels.size()),
          static_cast<std::int64_t>(batch->segments.size()),
          static_cast<std::int64_t>(batch->dma_count),
          static_cast<std::int64_t>(batch->barrier_count),
          static_cast<std::int64_t>(batch->collective_count)};
}
std::vector<std::int64_t> tp_context_stats(Handle id) {
  auto context = get<TpContext>(id, Kind::TpContext, "TP context");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  if (!context->arenas[0]) return {0, 0, 0, 0};
  std::size_t available = 0, largest = 0;
  for (const auto& range : context->free_ranges) {
    available += range.second;
    largest = std::max(largest, range.second);
  }
  return {static_cast<std::int64_t>(abi::working_bytes),
          static_cast<std::int64_t>(abi::working_bytes - available),
          static_cast<std::int64_t>(context->peak_allocated),
          static_cast<std::int64_t>(largest)};
}
std::vector<std::int64_t> tp_pool_stats(Handle id) {
  get<TpContext>(id, Kind::TpContext, "TP context");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  rhino_lkn::GraphArenaPoolStats stats{};
  check(rhino_lkn::QueryGraphArenaPoolStats(&stats), "TP query graph arena pool");
  return {static_cast<std::int64_t>(stats.capacity), static_cast<std::int64_t>(stats.current),
          static_cast<std::int64_t>(stats.peak), static_cast<std::int64_t>(stats.graph_retained),
          static_cast<std::int64_t>(stats.ephemeral), static_cast<std::int64_t>(stats.capacity_bytes),
          static_cast<std::int64_t>(stats.current_bytes), static_cast<std::int64_t>(stats.peak_bytes),
          static_cast<std::int64_t>(stats.graph_retained_bytes), static_cast<std::int64_t>(stats.ephemeral_bytes)};
}
void tp_execute(Handle id, bool asynchronous) {
  auto batch = get<TpBatch>(id, Kind::TpBatch, "TP batch");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex); tp_idle();
  std::set<std::uint64_t> prior_writes;
  for (const auto& node : batch->nodes) {
    tp_check_node(*node);
    for (const auto& rule : node->request.value_requirements)
      require(!prior_writes.count(node->tensors.at(rule.tensor)->desc.identity),
              "value-constrained input cannot depend on an unproven device-produced range");
    for (const auto& name : node->lowered.writes)
      prior_writes.insert(node->tensors.at(name)->desc.identity);
  }
  if (batch->fused) (void)tp_validate_fused(batch->nodes);
  batch->failure = nullptr;
  try { tp_sync_bindings(*batch); }
  catch (...) { batch->failure = std::current_exception(); throw; }
  // Simulate explicit state effects in submission order, before any enqueue.
  // This validates insert->attention within one batch without host round trips.
  using Ranges = std::vector<std::pair<std::uint64_t, std::uint64_t>>;
  std::map<TpTensor*, Ranges> future;
  for (const auto& node : batch->nodes) for (const auto& access : node->lowered.state_accesses) {
    auto* tensor = node->tensors.at(access.tensor).get();
    tp::require(tensor->desc.state_kind != "none" && access.begin < access.end &&
                access.end <= tp_state_capacity(*tensor), "invalid state access interval: " + access.tensor);
    auto inserted = future.emplace(tensor, tensor->valid_ranges);
    auto& intervals = inserted.first->second;
    if (!access.write) {
      tp::require(std::any_of(intervals.begin(), intervals.end(), [&](const auto& range) {
        return range.first <= access.begin && range.second >= access.end;
      }), "read outside initialized state/cache range: " + access.tensor);
    } else {
      intervals.emplace_back(access.begin, access.end); std::sort(intervals.begin(), intervals.end());
      Ranges merged;
      for (const auto& range : intervals) {
        if (!merged.empty() && range.first <= merged.back().second)
          merged.back().second = std::max(merged.back().second, range.second);
        else merged.push_back(range);
      }
      intervals = std::move(merged);
    }
  }
  batch->failure = nullptr;
  for (auto& item : future) item.first->valid_ranges = std::move(item.second);
  std::set<const void*> invalidated_alias_groups;
  for (const auto& node : batch->nodes)
    for (const auto& name : node->lowered.writes) {
      // Preserve state intervals validated above, while invalidating every
      // alias's numeric proof before any device write can occur.
      auto& tensor = *node->tensors.at(name);
      if (!invalidated_alias_groups.insert(tensor.aliases.get()).second) continue;
      for (auto it = tensor.aliases->begin(); it != tensor.aliases->end();) {
        if (auto alias = it->lock()) {
          alias->desc.values_known = alias->desc.finite = false;
          ++it;
        } else it = tensor.aliases->erase(it);
      }
    }
  device_state().in_flight = batch->in_flight = true;
  auto run = [raw = batch.get()] {
    // Poison once per logical replay, including a single-segment replay.
    // Every following segment has a distinct fixed tag, so a preceding
    // segment's success cannot satisfy a later completion check. Keep the
    // all-core check and stop before submitting the next segment on error.
    const std::array<std::uint16_t, 16> zero{};
    for (unsigned core = 0; core < 8; ++core)
      tp_write(raw->completion->spm_pointer(core), zero.data(), sizeof(zero));
    for (std::size_t index = 0; index < raw->segments.size(); ++index) {
      const auto& segment = raw->segments[index];
      const auto status = segment->enqueue_batch(true);
      if (status) { segment->finish(); check(status, "TP enqueue segment"); }
      tp_verify_completion(*raw, raw->completion_tags[index]);
    }
  };
  try {
    if (asynchronous) batch->pending = std::async(std::launch::async, std::move(run));
    else { run(); batch->in_flight = device_state().in_flight = false; }
  } catch (...) {
    batch->in_flight = device_state().in_flight = false;
    tp_invalidate_writes(*batch);
    batch->failure = std::current_exception(); throw;
  }
}
void tp_wait(Handle id) {
  auto batch = get<TpBatch>(id, Kind::TpBatch, "TP batch");
  std::lock_guard<std::recursive_mutex> lock(device_state().mutex);
  if (batch->in_flight) {
    try { batch->pending.get(); }
    catch (...) {
      tp_invalidate_writes(*batch);
      batch->failure = std::current_exception();
    }
    batch->in_flight = device_state().in_flight = false;
  }
  if (batch->failure) std::rethrow_exception(batch->failure);
}
}  // namespace xinfer::rpu::execution
