// Pieces shared by the SM120 MXFP8 kernels that work on 256-column tiles of the nk_tile256 packing
// (tile_mxfp8_sm120.cu, streamk_mxfp8_sm120.cu): the packed-weight geometry, barrier / load / MMA instructions, the
// TMA map of the activation and the SM count.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "common/tensor_map.h"
#include "common/tvm_ffi_cuda_tensor.h"

namespace xinfer::gemm::sm120::mxfp8_tile {

constexpr int kBN = 256, kBK = 64, kWarps = 8, kThreads = kWarps * 32;
constexpr int kWBytes = kBN * kBK, kWBlock = kWBytes + 1024;  // one (column tile, 64-K step) of the packed weight

__device__ __forceinline__ uint32_t Shared(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void BarrierInit(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ bool BarrierTest(uint32_t bar, uint32_t parity) {
  uint32_t done;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(done) : "r"(bar), "r"(parity) : "memory");
  return done != 0;
}
__device__ __forceinline__ bool BarrierTry(uint32_t bar, uint32_t parity) {
  uint32_t done;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(done) : "r"(bar), "r"(parity) : "memory");
  return done != 0;
}
__device__ __forceinline__ void ArriveElect(uint32_t bar) {
  asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n @p mbarrier.arrive.shared.b64 _, [%0];\n}\n"
               ::"r"(bar) : "memory");
}
__device__ __forceinline__ void LoadMatrix4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ uint32_t LoadShared32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ void LoadShared128(uint32_t addr, uint32_t* v) {
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "r"(addr));
}
// d += A (m16 x k32 E4M3) . B (n8 x k32 E4M3) with one E8M0 scale per row / column; kBlock picks the 32-K slot of
// the scale words.
template <int kBlock>
__device__ __forceinline__ void Mma(float* d, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t a_scale,
                                    uint32_t b_scale) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(a_scale), "h"((uint16_t)kBlock),
        "h"((uint16_t)0), "r"(b_scale), "h"((uint16_t)kBlock), "h"((uint16_t)0));
}
// Four FP32 values to four E4M3 bytes (low byte first), round to nearest, saturating.
__device__ __forceinline__ uint32_t ToE4M3x4(float a, float b, float c, float d) {
  uint32_t r;
  asm("{\n .reg .b16 lo, hi;\n cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      " mov.b32 %0, {lo, hi};\n}\n" : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}

// TMA map of the E4M3 activation [rows, k] with a box of box_rows x 64 bytes (64-byte swizzle).
inline CUtensorMap MapA(const void* x, int64_t rows, int64_t k, int box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(k)};
  cuuint32_t box[2] = {kBK, static_cast<cuuint32_t>(box_rows)};
  cuuint32_t elem[2] = {1, 1};
  if (xinfer::ffi::TensorMapEncodeTiled()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(x), dims, strides,
                                            box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    throw std::runtime_error("cuTensorMapEncodeTiled failed");
  }
  return m;
}

inline int SmCount(int device) {
  static int counts[64] = {};
  if (device < 0 || device >= 64) throw std::runtime_error("device index out of range");
  if (counts[device] == 0) {
    xinfer::ffi::CheckCudaError(cudaDeviceGetAttribute(&counts[device], cudaDevAttrMultiProcessorCount, device),
                                  "cudaDeviceGetAttribute");
  }
  return counts[device];
}

}  // namespace xinfer::gemm::sm120::mxfp8_tile
