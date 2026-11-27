// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/execution.hpp"
#include "common/rpu/kernel_abi.hpp"
#include "common/rpu/single_core_contract.hpp"
#include "rhino_launch_buffer.h"
#include "rhino_launch_kernel.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace xinfer::rpu::execution {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
void check(std::int64_t status, const char* operation) {
  if (status) throw std::runtime_error(std::string(operation) +
                                     " failed: SDK status " + std::to_string(status));
}
enum class Kind { Context, Buffer, Weight, Execution, TpContext, TpTensor, TpNode, TpBatch };
struct Resource {
  explicit Resource(Kind value) : kind(value) {}
  virtual ~Resource() = default;
  const Kind kind;
  const pid_t process = ::getpid();
};
struct DeviceState {
  std::recursive_mutex mutex;
  bool in_flight = false;
  std::array<std::size_t, 8> allocated_bytes{};
};
DeviceState& device_state() {
  // Process-lifetime state must outlive registry teardown regardless of which
  // API initialized its function-local static first.
  static auto* state = new DeviceState;
  return *state;
}
struct Context final : Resource {
  Context(std::uint8_t core_id, bool candidates)
      : Resource(Kind::Context), mutex(device_state().mutex),
        core(core_id), allow_candidate(candidates),
        allocated_bytes(device_state().allocated_bytes[core_id]),
        in_flight(device_state().in_flight) {}
  // Serialize device submissions until finish(), including submissions to
  // different cores. Multi-core work is submitted within one graph or batch.
  std::recursive_mutex& mutex;
  const std::uint8_t core;
  const bool allow_candidate;
  std::size_t& allocated_bytes;
  bool& in_flight;
};
struct Buffer final : Resource {
  Buffer(std::shared_ptr<Context> ctx, std::size_t r, std::size_t c)
      : Resource(Kind::Buffer), context(std::move(ctx)), rows(r), cols(c),
        bytes(r * c * 2), allocation_bytes((bytes + 255) & ~std::size_t(255)) {}
  ~Buffer() override {
    if (::getpid() != process) { memory.release(); return; }
    // Allocation accounting is atomic with creation/release. Executions retain
    // shared ownership, so no queued command can reference freed SPM.
    std::lock_guard<std::recursive_mutex> lock(context->mutex);
    if (memory) {
      memory.reset();
      context->allocated_bytes -= allocation_bytes;
    }
  }
  std::shared_ptr<Context> context;
  const std::size_t rows, cols, bytes, allocation_bytes;
  std::unique_ptr<rhino_lkn::LocalSPM_t> memory;
  std::uint32_t offset() const {
    return static_cast<std::uint32_t>(memory->get_rpu_addr());
  }
};
struct Weight final : Resource {
  Weight(std::shared_ptr<Context> ctx, std::size_t n_, std::size_t k_)
      : Resource(Kind::Weight), context(std::move(ctx)), n(n_), k(k_) {}
  ~Weight() override {
    if (::getpid() != process) { memory.release(); return; }
    std::lock_guard<std::recursive_mutex> lock(context->mutex);
    memory.reset();
  }
  std::shared_ptr<Context> context;
  const std::size_t n, k;
  std::unique_ptr<rhino_lkn::HostDDR_t> memory;
};
struct Prepared final : Resource {
  explicit Prepared(std::shared_ptr<Context> ctx)
      : Resource(Kind::Execution), context(std::move(ctx)) {}
  ~Prepared() override {
    if (::getpid() != process) {
      queue.release(); kernel.release(); program.release(); return;
    }
    // The SDK finish API returns void. A submitted queue must outlive completion,
    // including when Python closes/drops the last execution reference.
    std::lock_guard<std::recursive_mutex> lock(context->mutex);
    if (in_flight) {
      queue->finish();
      in_flight = false;
      context->in_flight = false;
    }
    queue.reset(); kernel.reset(); program.reset();
  }
  std::shared_ptr<Context> context;
  std::vector<std::shared_ptr<Resource>> dependencies;
  std::unique_ptr<rhino_lkn::Program_t> program;
  std::unique_ptr<rhino_lkn::Kernel_t> kernel;
  std::unique_ptr<rhino_lkn::Queue_t> queue;
  bool in_flight = false;
};
struct Registry {
  const pid_t process = ::getpid();
  std::mutex mutex;
  Handle next = 1;
  std::unordered_map<Handle, std::shared_ptr<Resource>> resources;
};
Registry& registry() { static Registry value; return value; }
Handle insert(std::shared_ptr<Resource> resource) {
  auto& r = registry();
  require(r.process == ::getpid(), "RPU registry cannot be used after fork");
  std::lock_guard<std::mutex> lock(r.mutex);
  require(r.next < std::numeric_limits<Handle>::max(), "RPU handle space exhausted");
  const auto id = r.next++;
  r.resources.emplace(id, std::move(resource));
  return id;
}
template<class T> std::shared_ptr<T> get(Handle id, Kind kind, const char* name) {
  auto& r = registry();
  require(r.process == ::getpid(), "RPU registry cannot be used after fork");
  std::lock_guard<std::mutex> lock(r.mutex);
  auto found = r.resources.find(id);
  if (found == r.resources.end() || found->second->kind != kind)
    throw std::invalid_argument(std::string(name) + " is a closed, invalid or wrong-kind RPU handle");
  require(found->second->process == ::getpid(), "RPU handles cannot be used after fork");
  return std::static_pointer_cast<T>(found->second);
}
void idle(const Context& context) {
  // Independent of weak/shared ownership: a last-reference destructor can be
  // waiting on this mutex while its queue remains live on the device.
  require(!context.in_flight,
          "RPU device has an outstanding execution; wait before reuse or host copy");
}
void same_context(const std::shared_ptr<Context>& context,
                  const std::shared_ptr<Context>& other) {
  require(context == other, "all RPU operands must belong to the same context");
}
void shape(const Buffer& b, std::size_t rows, std::size_t cols, const char* name) {
  if (b.rows != rows || b.cols != cols)
    throw std::invalid_argument(std::string(name) + " has incompatible shape");
}
void fence() {
#if defined(__aarch64__)
  asm volatile("dsb sy" ::: "memory");
#else
  __sync_synchronize();
#endif
}
void write_spm(void* target, const void* source, std::size_t bytes) {
  // SPM mapping is device memory: use naturally aligned volatile 64-bit
  // transactions, never libc's SIMD memcpy into the mapping.
  auto* dst = static_cast<volatile std::uint64_t*>(target);
  auto* src = static_cast<const unsigned char*>(source);
  for (std::size_t i = 0; i < bytes / 8; ++i) {
    std::uint64_t word;
    std::memcpy(&word, src + i * 8, 8); dst[i] = word;
  }
  fence();
}
void read_spm(void* target, const void* source, std::size_t bytes) {
  fence();
  auto* src = static_cast<const volatile std::uint64_t*>(source);
  auto* dst = static_cast<unsigned char*>(target);
  for (std::size_t i = 0; i < bytes / 8; ++i) {
    const std::uint64_t word = src[i]; std::memcpy(dst + i * 8, &word, 8);
  }
  fence();
}
Handle prepare(const std::shared_ptr<Context>& context, const abi::LaunchPlan& plan,
               std::vector<std::shared_ptr<Resource>> dependencies) {
  // Caller holds the context mutex through validation and queue construction.
  idle(*context);
  auto result = std::make_shared<Prepared>(context);
  result->dependencies = std::move(dependencies);
  result->program = std::make_unique<rhino_lkn::Program_t>();
  const char* entry = abi::name(plan.entry);
  const auto status = load_program(*result->program, entry, context->allow_candidate);
  if (status) {
    throw std::runtime_error(std::string("RPU prepare ") + entry + ": " + last_error());
  }
  result->kernel = std::make_unique<rhino_lkn::Kernel_t>(*result->program, entry);
  check(abi::apply(*result->kernel, plan), "RPU register binding");
  // Queue_t's argument bounds physical core IDs [0, core_num), rather than
  // counting the cores in this launch. A queue for core 7 must manage 8 IDs.
  // The actual launch still selects exactly one physical core below.
  result->queue = std::make_unique<rhino_lkn::Queue_t>(context->core + 1);
  result->queue->set_flush_icache(true);
  check(result->queue->add_kernel(*result->kernel,
        {plan.grid[0], plan.grid[1], plan.grid[2]}, {context->core}), "RPU add_kernel");
  check(result->queue->build_batch(), "RPU build_batch");
  return insert(std::move(result));
}
}  // namespace

Handle context_create(std::int64_t core, bool allow_candidate) {
  require(core >= 0 && core < 8, "RPU core must be in [0,7]");
  return insert(std::make_shared<Context>(static_cast<std::uint8_t>(core), allow_candidate));
}
void release(Handle id) {
  std::shared_ptr<Resource> resource;
  {
    auto& r = registry();
    require(r.process == ::getpid(), "RPU registry cannot be used after fork");
    std::lock_guard<std::mutex> lock(r.mutex);
    auto found = r.resources.find(id);
    require(found != r.resources.end(), "cannot release closed or invalid RPU handle");
    require(found->second->process == ::getpid(), "RPU handles cannot be released after fork");
    resource = std::move(found->second);
    r.resources.erase(found);
  }
  // Destruction, possibly waiting for execution, must happen outside registry lock.
}
Handle buffer_empty(Handle context_id, std::int64_t rows, std::int64_t cols) {
  auto context = get<Context>(context_id, Kind::Context, "context");
  require(rows > 0 && rows <= static_cast<std::int64_t>(abi::working_bytes / 2) &&
          cols > 0 && cols <= static_cast<std::int64_t>(abi::working_bytes / 2),
          "RPU buffer rows and cols must be positive and fit the SPM budget");
  const auto bytes = static_cast<std::uint64_t>(rows) * cols * 2;
  require(bytes <= abi::working_bytes && bytes % 32 == 0,
          "RPU buffer requires a multiple of 16 FP16 elements within 7 MiB");
  auto result = std::make_shared<Buffer>(context, rows, cols);
  {
    std::lock_guard<std::recursive_mutex> lock(context->mutex);
    idle(*context);
    require(result->allocation_bytes <= abi::working_bytes - context->allocated_bytes,
            "RPU core buffers exceed 7 MiB SPM budget");
    result->memory = std::make_unique<rhino_lkn::LocalSPM_t>(result->allocation_bytes,
        rhino_lkn::read_write, rhino_lkn::kStride256B, 256, context->core);
    context->allocated_bytes += result->allocation_bytes;
    require(result->memory->get_cpu_ptr() != nullptr &&
            reinterpret_cast<std::uintptr_t>(result->memory->get_cpu_ptr()) % 8 == 0 &&
            result->memory->get_rpu_addr() < abi::spm_bytes,
            "SDK returned an invalid or unaligned SPM mapping");
    abi::detail::span(result->offset(), result->allocation_bytes);
  }
  return insert(std::move(result));
}
void buffer_upload(Handle id, const std::uint16_t* data,
                   std::size_t rows, std::size_t cols) {
  auto buffer = get<Buffer>(id, Kind::Buffer, "buffer");
  require(data != nullptr, "upload data must not be null");
  shape(*buffer, rows, cols, "upload tensor");
  std::lock_guard<std::recursive_mutex> lock(buffer->context->mutex);
  idle(*buffer->context);
  write_spm(buffer->memory->get_cpu_ptr(), data, buffer->bytes);
}
void buffer_download(std::uint16_t* data, Handle id,
                     std::size_t rows, std::size_t cols) {
  auto buffer = get<Buffer>(id, Kind::Buffer, "buffer");
  require(data != nullptr, "download data must not be null");
  shape(*buffer, rows, cols, "download tensor");
  std::lock_guard<std::recursive_mutex> lock(buffer->context->mutex);
  idle(*buffer->context);
  read_spm(data, buffer->memory->get_cpu_ptr(), buffer->bytes);
}
Handle prepare_weight(Handle context_id, const std::uint16_t* data,
                      std::size_t n, std::size_t k) {
  auto context = get<Context>(context_id, Kind::Context, "context");
  auto packed = single_core::pack_action_io_weight(data, n, k);
  auto result = std::make_shared<Weight>(context, n, k);
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  idle(*context);
  result->memory = std::make_unique<rhino_lkn::HostDDR_t>(packed.size() * 2,
      rhino_lkn::read_only, rhino_lkn::kStride256B, 256);
  require(result->memory->get_cpu_ptr() != nullptr, "SDK returned null DDR mapping");
  abi::detail::ddr_span(result->memory->get_rpu_addr(), packed.size() * 2);
  std::memcpy(result->memory->get_cpu_ptr(), packed.data(), packed.size() * 2);
  check(result->memory->flush(), "RPU packed weight flush");
  return insert(std::move(result));
}
Handle prepare_gemm(Handle output, Handle input, Handle weight, Handle bias) {
  auto out = get<Buffer>(output, Kind::Buffer, "output");
  auto src = get<Buffer>(input, Kind::Buffer, "input");
  auto w = get<Weight>(weight, Kind::Weight, "weight");
  auto b = get<Buffer>(bias, Kind::Buffer, "bias");
  auto context = out->context;
  same_context(context, src->context); same_context(context, w->context);
  same_context(context, b->context);
  require(single_core::supported_gemm_shape(src->rows, w->n, w->k),
          "GEMM requires M=16 or 32 and (N,K)=(1024,64) or (64,1024)");
  shape(*src, src->rows, w->k, "input"); shape(*out, src->rows, w->n, "output");
  shape(*b, 1, w->n, "bias");
  auto plan = abi::io_gemm({w->n == 1024 ? abi::IoGemmKind::InputProjection
                                           : abi::IoGemmKind::OutputProjection,
      out->offset(), src->offset(), w->memory->get_rpu_addr(), b->offset(),
      static_cast<std::uint32_t>(src->rows)});
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  return prepare(context, plan, {out, src, w, b});
}
Handle prepare_layernorm(Handle output, Handle input, Handle weight, Handle bias,
                         double epsilon) {
  auto out = get<Buffer>(output, Kind::Buffer, "output");
  auto src = get<Buffer>(input, Kind::Buffer, "input");
  auto w = get<Buffer>(weight, Kind::Buffer, "weight");
  auto b = bias ? get<Buffer>(bias, Kind::Buffer, "bias") : nullptr;
  auto context = out->context;
  same_context(context, src->context); same_context(context, w->context);
  if (b) same_context(context, b->context);
  shape(*out, src->rows, src->cols, "output"); shape(*w, 1, src->cols, "weight");
  if (b) shape(*b, 1, src->cols, "bias");
  const float eps32 = static_cast<float>(epsilon);
  require(std::isfinite(epsilon) && std::isfinite(eps32) &&
          eps32 >= 1.0e-5F && eps32 <= 65504.0F,
          "LayerNorm epsilon requires finite FP32 value in [1e-5,65504]");
  auto plan = abi::layernorm({out->offset(), src->offset(), w->offset(),
      b ? b->offset() : 0, static_cast<std::uint32_t>(src->rows),
      static_cast<std::uint32_t>(src->cols), b != nullptr, eps32});
  std::vector<std::shared_ptr<Resource>> dependencies{out, src, w};
  if (b) dependencies.push_back(b);
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  return prepare(context, plan, std::move(dependencies));
}
Handle prepare_silu_mul(Handle output, Handle x, Handle y) {
  auto out = get<Buffer>(output, Kind::Buffer, "output");
  auto lhs = get<Buffer>(x, Kind::Buffer, "x");
  auto rhs = get<Buffer>(y, Kind::Buffer, "y");
  auto context = out->context;
  same_context(context, lhs->context); same_context(context, rhs->context);
  shape(*out, lhs->rows, lhs->cols, "output");
  shape(*rhs, lhs->rows, lhs->cols, "y");
  auto plan = abi::silu_mul({out->offset(), lhs->offset(), rhs->offset(),
      static_cast<std::uint32_t>(lhs->rows * lhs->cols), 1});
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  return prepare(context, plan, {out, lhs, rhs});
}
void execute(Handle id, bool asynchronous) {
  auto prepared = get<Prepared>(id, Kind::Execution, "execution");
  auto context = prepared->context;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  idle(*context);
  context->in_flight = true;
  prepared->in_flight = true;
  std::int64_t status = 0;
  try {
    status = prepared->queue->enqueue_batch(!asynchronous);
  } catch (...) {
    // The SDK rejected submission before exposing a status. Drain any partial
    // enqueue, then restore the device contract for the caller's next attempt.
    prepared->in_flight = false;
    context->in_flight = false;
    try { prepared->queue->finish(); } catch (...) {}
    throw;
  }
  if (status) {
    prepared->in_flight = false;
    context->in_flight = false;
    prepared->queue->finish();
    check(status, "RPU enqueue_batch");
  }
  if (!asynchronous) {
    prepared->in_flight = false;
    context->in_flight = false;
  }
}
void wait(Handle id) {
  auto prepared = get<Prepared>(id, Kind::Execution, "execution");
  std::lock_guard<std::recursive_mutex> lock(prepared->context->mutex);
  if (prepared->in_flight) {
    prepared->queue->finish();
    prepared->in_flight = false;
    prepared->context->in_flight = false;
  }
}
}  // namespace xinfer::rpu::execution

// Uses the same checked handle registry, device lock and allocation accounting
// as single-core executions. No independent queue can bypass the device-wide in-flight guard.
#include "tp_execution.inc"
