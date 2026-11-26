// The rhino_lkn namespace and API signatures correspond to the external
// proprietary Rhino Launch SDK provided by HUIXI-AI. This minimal in-memory
// test double supports host-only tests. SDK licensing is governed by the
// terms provided by HUIXI-AI.
// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Authored host-only SDK doubles. No vendor code, device kernels, numerical
// emulation, or board access. Queues expose only submission/failure ordering;
// successful returns deliberately leave completion markers untouched.
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <new>
#include <stdexcept>
#include <vector>

namespace rhino_lkn {
enum Flag { read_write, read_only };
enum Stride { kStride256B };
inline int next_enqueue_status = 0;
inline bool throw_on_enqueue = false;
inline unsigned enqueue_count = 0;
inline int next_dma_status = 0;
inline unsigned dma_count = 0;
inline unsigned dma_update_count = 0, mutable_sync_count = 0;
inline int next_sync_status = 0;
class LocalSPM_t {
 public:
  LocalSPM_t(std::size_t bytes, Flag, Stride, std::uint64_t, int)
      : pointer_(::operator new(bytes, std::align_val_t(256))) {}
  ~LocalSPM_t() { ::operator delete(pointer_, std::align_val_t(256)); }
  void* get_cpu_ptr() const { return pointer_; }
  std::uint64_t get_rpu_addr() const { return 0; }
 private:
  void* pointer_;
};
class HostDDR_t {
 public:
  HostDDR_t(std::size_t bytes, Flag, Stride, std::uint64_t)
      : pointer_(::operator new(bytes, std::align_val_t(256))) {
    static std::uint64_t next = 0x10000000;
    address_ = next; next += (bytes + 255) & ~std::uint64_t(255);
  }
  ~HostDDR_t() { ::operator delete(pointer_, std::align_val_t(256)); }
  void* get_cpu_ptr() const { return pointer_; }
  std::uint64_t get_rpu_addr() const { return address_; }
  int flush() { return 0; }
  int begin_cpu_access(std::size_t, std::size_t) { return 0; }
  int end_cpu_access(std::size_t, std::size_t) { return 0; }
 private:
  void* pointer_;
  std::uint64_t address_;
};
class Program_t {
 public:
  int create_with_binary_file(const std::string& path, const char*) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("test asset cannot be read");
    return 0;
  }
};
class Kernel_t {
 public:
  Kernel_t(Program_t&, const char*) {}
  int reset_regs() { return 0; }
  template<class T> int set_regs(unsigned index, T) { written_regs.push_back(index); return 0; }
  std::vector<unsigned> written_regs;
};
class Queue_t {
 public:
  explicit Queue_t(unsigned) {}
  void set_flush_icache(bool) {}
  void set_broadcast_mode(bool) {}
  int add_kernel(Kernel_t&, std::vector<std::uint16_t>,
                 std::vector<std::uint8_t>) { return 0; }
  int add_kernel_to_stream(Kernel_t&, std::vector<std::uint16_t>,
                           std::vector<std::uint8_t>, std::uint8_t) { return 0; }
  int add_kernel_mutable(Kernel_t&, std::vector<std::uint16_t>,
                          std::vector<std::uint8_t>) { return 0; }
  int build_batch() { return 0; }
  void add_barrier(unsigned, unsigned) {}
  template<class S, class D>
  int add_dma_kernel(const S&, std::size_t, D&, std::size_t, std::size_t, int) {
    ++dma_count;
    const auto result = next_dma_status;
    next_dma_status = 0;
    return result;
  }
  template<class S, class D>
  int add_dma_kernel_mutable(const S& src, std::size_t so, D& dst, std::size_t to,
                             std::size_t size, int channel) {
    const auto result = add_dma_kernel(src, so, dst, to, size, channel);
    if (!result) ++mutable_dmas_;
    return result;
  }
  template<class S, class D>
  int update_dma_kernel(unsigned id, const S&, std::size_t, D&, std::size_t, std::size_t) {
    if (id >= mutable_dmas_) throw std::runtime_error("invalid mutable DMA slot");
    ++dma_update_count;
    return 0;
  }
  int sync_mutable_params() {
    ++mutable_sync_count;
    auto result = next_sync_status; next_sync_status = 0; return result;
  }
  void finish() {}
  int enqueue_batch(bool) {
    ++enqueue_count;
    if (throw_on_enqueue) {
      throw_on_enqueue = false;
      throw std::runtime_error("test enqueue_batch exception");
    }
    const int result = next_enqueue_status;
    next_enqueue_status = 0;
    return result;
  }
 private:
  unsigned mutable_dmas_ = 0;
};
struct GraphArenaRequirement {
  std::uint64_t program_bytes, queue_bytes;
  std::uint32_t count, flags;
};
struct GraphArenaPoolStats {
  std::uint64_t capacity, current, peak, graph_retained, ephemeral;
  std::uint64_t capacity_bytes, current_bytes, peak_bytes, graph_retained_bytes, ephemeral_bytes;
};
inline int QueryGraphArenaPoolStats(GraphArenaPoolStats* stats) {
  *stats = {}; return 0;
}
enum GraphArenaPoolResult { GRAPH_ARENA_POOL_SUCCESS };
inline GraphArenaPoolResult ConfigureGraphArenaPool(
    const GraphArenaRequirement*, std::size_t) { return GRAPH_ARENA_POOL_SUCCESS; }
inline void RpuDdrFlushDirect(void*, std::size_t) {}
}  // namespace rhino_lkn
