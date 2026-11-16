// attention_kvcache: split-KV small-M backend (generic CUDA, see attention_kvcache_splitkv.cuh).
//
//   attention_kvcache_splitkv                 launch (partial + PDL-chained finalize)
//   attention_kvcache_splitkv_workspace_size  workspace bytes
//
// Called by ../attn.py, which allocates the output and workspace. Every hard constraint is
// re-validated here.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <utility>

#include "common/cuda_pdl.cuh"
#include "common/tvm_ffi_cuda_tensor.h"
#include "attention_kvcache_splitkv.cuh"

namespace xinfer::attention_kvcache {
namespace {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;
namespace ffi = xinfer::ffi;

[[noreturn]] void Fail(const std::string& what) {
  throw std::runtime_error("attention_kvcache_splitkv: " + what);
}

void CheckRank(const TensorView& t, int ndim, const char* name) {
  if (t.ndim() != ndim) Fail(std::string(name) + " must be rank-" + std::to_string(ndim));
}

// [B, rows, heads, D] BF16 with a dense last dimension and 16-byte aligned rows / heads
void CheckHeads(const TensorView& t, const TensorView& ref, int64_t b, int64_t rows, int64_t heads, int64_t d,
                const char* name) {
  CheckRank(t, 4, name);
  ffi::CheckSameDevice(t, ref, name);
  ffi::CheckBFloat16(t, name);
  if (t.size(0) != b || t.size(1) != rows || t.size(2) != heads || t.size(3) != d) {
    Fail(std::string(name) + " must be [" + std::to_string(b) + ", " + std::to_string(rows) + ", " +
         std::to_string(heads) + ", " + std::to_string(d) + "]");
  }
  if (t.stride(3) != 1 || t.stride(0) % 8 || t.stride(1) % 8 || t.stride(2) % 8 ||
      reinterpret_cast<uintptr_t>(t.data_ptr()) % 16) {
    Fail(std::string(name) + " needs a dense last dimension, strides that are multiples of 8 elements and a "
                             "16-byte aligned base");
  }
}

constexpr int64_t kMaxM = 32;

struct Variant {
  int d, g, chunk;
};

constexpr int kSm120Batch = 4;  // SM120 merge: chunk partials in flight per warp (same result for any value)

template <int D, int G, int CHUNK, int ROWS, int WG = 1>
void Launch(SplitkvParams p, bool pdl, cudaStream_t stream) {
  p.row_groups = (G * p.M + ROWS - 1) / ROWS;
  const int rows_pad = (std::min(G * p.M, ROWS) + 15) & ~15;
  const size_t smem = SplitkvSmemBytes<D, CHUNK, WG>(rows_pad);
  auto* partial = splitkv_partial_kernel<D, G, CHUNK, ROWS, WG>;
  int device = 0;
  ffi::CheckCudaError(cudaGetDevice(&device), "cudaGetDevice");
  int major = 0;
  ffi::CheckCudaError(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
                      "cudaDeviceGetAttribute");
  // merge: two chunk partials in flight per warp on SM110, four elsewhere (fastest measured; same result)
  auto* finalize =
      major == 11 ? (p.out_quant == kOutBf16          ? splitkv_finalize_kernel<D, G, CHUNK, kOutBf16, 2>
                     : p.out_quant == kOutMxfp8Blk32 ? splitkv_finalize_kernel<D, G, CHUNK, kOutMxfp8Blk32, 2>
                                                     : splitkv_finalize_kernel<D, G, CHUNK, kOutMxfp8Blk128, 2>)
                  : (p.out_quant == kOutBf16          ? splitkv_finalize_kernel<D, G, CHUNK, kOutBf16, kSm120Batch>
                     : p.out_quant == kOutMxfp8Blk32 ? splitkv_finalize_kernel<D, G, CHUNK, kOutMxfp8Blk32, kSm120Batch>
                                                     : splitkv_finalize_kernel<D, G, CHUNK, kOutMxfp8Blk128, kSm120Batch>);
  static int attribute_set_on = -1;  // device whose function attribute is set (per instantiation)
  if (attribute_set_on != device) {
    ffi::CheckCudaError(cudaFuncSetAttribute(partial, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(SplitkvSmemBytes<D, CHUNK, WG>(ROWS))),
                        "attention_kvcache_splitkv shared-memory attribute");
    if constexpr (WG == 2) {  // two CTAs of the larger tile per SM need the whole shared-memory carveout
      ffi::CheckCudaError(
          cudaFuncSetAttribute(partial, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared),
          "attention_kvcache_splitkv carveout attribute");
    }
    attribute_set_on = device;
  }
  const dim3 grid(p.nchunk, p.Hkv * p.row_groups, p.B);
  const dim3 block(ROWS * 2 * WG);  // ROWS / 16 warps per group: all load, those with rows compute
  if (pdl) {
    xinfer::cuda::pdl_launch(grid, block, smem, stream, partial, p);
  } else {
    partial<<<grid, block, smem, stream>>>(p);
    ffi::CheckLastCudaError("attention_kvcache_splitkv partial launch");
  }
  const int warps = p.B * p.M * p.Hkv * G;
  // SM120: one warp per block spreads a small merge over more SMs (same result)
  const int wpb = major == 11 ? 8 : 1;
  xinfer::cuda::pdl_launch(dim3((warps + wpb - 1) / wpb), dim3(32 * wpb), 0, stream, finalize, p);
}

int DeviceMajor() {
  int device = 0, major = 0;
  ffi::CheckCudaError(cudaGetDevice(&device), "cudaGetDevice");
  ffi::CheckCudaError(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
                      "cudaDeviceGetAttribute");
  return major;
}

int64_t ChunkCount(int64_t l, int64_t chunk) { return (l + chunk - 1) / chunk; }

int64_t PartialBytes(int64_t b, int64_t m, int64_t hq, int64_t hkv, int64_t d, int64_t l, int64_t chunk) {
  return (b * hkv * ChunkCount(l, chunk) * (hq / hkv) * m * d * 4 + 255) / 256 * 256;
}

int64_t WorkspaceBytes(int64_t b, int64_t m, int64_t hq, int64_t hkv, int64_t d, int64_t l, int64_t chunk) {
  return PartialBytes(b, m, hq, hkv, d, l, chunk) + b * hkv * ChunkCount(l, chunk) * (hq / hkv) * m * 8;
}

// (head_dim, heads per KV head, chunk) of the compiled kernels
bool Supported(int64_t d, int64_t g, int64_t chunk) {
  if (d == 256 && (g == 4 || g == 8)) return chunk == 16 || chunk == 32 || chunk == 64;
  if (d == 128 && g == 8) return chunk == 32 || chunk == 64 || chunk == 128;
  return false;
}

// 64 query rows per CTA: on both architectures more CTAs beat whole-head CTAs (256 rows measured slower or equal)
void Run(const SplitkvParams& p, int64_t d, int64_t g, int64_t chunk, bool pdl, cudaStream_t stream) {
  if (d == 256 && g == 8) {
    if (chunk == 16) Launch<256, 8, 16, 64>(p, pdl, stream);
    if (chunk == 32) Launch<256, 8, 32, 64>(p, pdl, stream);
    if (chunk == 64) Launch<256, 8, 64, 64>(p, pdl, stream);
  } else if (d == 256) {
    if (chunk == 16) Launch<256, 4, 16, 64>(p, pdl, stream);
    if (chunk == 32) Launch<256, 4, 32, 64>(p, pdl, stream);
    if (chunk == 64) Launch<256, 4, 64, 64>(p, pdl, stream);
  } else if (g * p.M <= 32 && DeviceMajor() == 11) {
    // SM110, at most two 16-row tiles: one 32-row CTA with two warp groups per tile, one warp per SMSP (halves
    // the per-warp MMA and softmax chains). Bit-identical to the one-group kernel, so choosing it by M keeps a
    // row's result independent of M. With more rows the SMSPs are already busy and one group is faster.
    if (chunk == 32) Launch<128, 8, 32, 32, 2>(p, pdl, stream);
    if (chunk == 64) Launch<128, 8, 64, 32, 2>(p, pdl, stream);
    if (chunk == 128) Launch<128, 8, 128, 32, 2>(p, pdl, stream);
  } else {
    if (chunk == 32) Launch<128, 8, 32, 64>(p, pdl, stream);
    if (chunk == 64) Launch<128, 8, 64, 64>(p, pdl, stream);
    if (chunk == 128) Launch<128, 8, 128, 64>(p, pdl, stream);
  }
  ffi::CheckLastCudaError("attention_kvcache_splitkv launch");
}

}  // namespace

int64_t attention_kvcache_splitkv_workspace_size(int64_t batch, int64_t m, int64_t num_heads, int64_t num_kv_heads,
                                                 int64_t head_dim, int64_t cache_rows, int64_t chunk) {
  if (batch < 1 || m < 1 || num_kv_heads < 1 || num_heads % num_kv_heads || cache_rows < 1 || chunk < 1) {
    Fail("invalid workspace-size arguments");
  }
  return WorkspaceBytes(batch, m, num_heads, num_kv_heads, head_dim, cache_rows, chunk);
}

void launch_attention_kvcache_splitkv(TensorView out, Optional<TensorView> out_scale, TensorView k_cache,
                                      TensorView v_cache, TensorView workspace, TensorView q, TensorView prefix_len,
                                      Optional<TensorView> k_new, Optional<TensorView> v_new,
                                      Optional<TensorView> flash_mask, Optional<TensorView> cache_batch_idx,
                                      Optional<TensorView> cache_leftpad, int64_t new_pos, bool causal,
                                      double softmax_scale, int64_t chunk, int64_t out_quant, bool pdl) {
  ffi::CheckCuda(q, "q");
  CheckRank(q, 4, "q");
  const int64_t b = q.size(0), m = q.size(1), hq = q.size(2), d = q.size(3);
  CheckRank(k_cache, 4, "k_cache");
  const int64_t l = k_cache.size(1), hkv = k_cache.size(2);
  if (b < 1 || m < 1 || m > kMaxM) Fail("needs batch >= 1 and 1 <= M <= 32, got M=" + std::to_string(m));
  if (hkv < 1 || hq % hkv) Fail("num_heads must be a multiple of num_kv_heads");
  const int64_t g = hq / hkv;
  if (!Supported(d, g, chunk)) {
    Fail("no kernel for head_dim=" + std::to_string(d) + ", heads per KV head=" + std::to_string(g) +
         ", chunk=" + std::to_string(chunk) + " (have 256/4/{16,32,64}, 256/8/{16,32,64}, 128/8/{32,64,128})");
  }
  CheckHeads(q, q, b, m, hq, d, "q");
  // cache_batch_idx: dim 0 of the cache is a pool of rows (any count); otherwise one row per request
  const int64_t pool = cache_batch_idx.has_value() ? k_cache.size(0) : b;
  if (pool < 1) Fail("the cache needs at least one row");
  CheckHeads(k_cache, q, pool, l, hkv, d, "k_cache");
  CheckHeads(v_cache, q, pool, l, hkv, d, "v_cache");
  for (const auto& [opt, name] : {std::pair<const Optional<TensorView>&, const char*>{cache_batch_idx, "cache_batch_idx"},
                                  {cache_leftpad, "cache_leftpad"}}) {
    if (!opt.has_value()) continue;
    const TensorView& t = opt.value();
    CheckRank(t, 1, name);
    ffi::CheckSameDevice(t, q, name);
    ffi::CheckInt32(t, name);
    ffi::CheckContiguous(t, name);
    if (t.size(0) != b) Fail(std::string(name) + " must be [batch]");
  }
  if (k_new.has_value() != v_new.has_value()) Fail("k_new and v_new must be given together");
  if (k_new.has_value()) {
    CheckHeads(k_new.value(), q, b, m, hkv, d, "k_new");
    CheckHeads(v_new.value(), q, b, m, hkv, d, "v_new");
    ffi::CheckNoOverlap(k_cache, "k_cache", k_new.value(), "k_new");
    ffi::CheckNoOverlap(v_cache, "v_cache", v_new.value(), "v_new");
  }
  CheckRank(prefix_len, 1, "prefix_len");
  ffi::CheckSameDevice(prefix_len, q, "prefix_len");
  ffi::CheckInt32(prefix_len, "prefix_len");
  ffi::CheckContiguous(prefix_len, "prefix_len");
  if (prefix_len.size(0) != b) Fail("prefix_len must be [batch]");
  if (new_pos < -1 || (new_pos >= 0 && new_pos + m > l)) {
    Fail("new_pos must be -1 (after the prefix) or satisfy 0 <= new_pos and new_pos + M <= cache rows");
  }
  if (!(softmax_scale > 0.0) || !std::isfinite(softmax_scale)) Fail("softmax_scale must be finite and positive");
  if (out_quant < kOutBf16 || out_quant > kOutMxfp8Blk128) Fail("out_quant must be 0 (BF16), 1 or 2 (MXFP8)");

  // output
  CheckRank(out, 3, "out");
  ffi::CheckSameDevice(out, q, "out");
  ffi::CheckContiguous(out, "out");
  if (out.size(0) != b || out.size(1) != m || out.size(2) != hq * d) Fail("out must be [batch, M, num_heads * head_dim]");
  if (out_quant == kOutBf16) {
    ffi::CheckBFloat16(out, "out");
    if (out_scale.has_value()) Fail("out_scale is only used with MXFP8 output");
  } else {
    ffi::CheckFloat8E4M3Storage(out, "out");
    if (!out_scale.has_value()) Fail("MXFP8 output needs out_scale");
    ffi::CheckSameDevice(out_scale.value(), q, "out_scale");
    ffi::CheckContiguous(out_scale.value(), "out_scale");
    ffi::CheckFloat8E8M0Storage(out_scale.value(), "out_scale");
    if (ffi::NumElements(out_scale.value()) < xinfer::mxfp8::RequiredScaleElements(b * m, hq * d)) {
      Fail("out_scale holds fewer than the SWIZZLE_32_4_4 scale bytes of [batch * M, num_heads * head_dim]");
    }
  }
  ffi::CheckNoOverlap(out, "out", q, "q");
  ffi::CheckNoOverlap(out, "out", k_cache, "k_cache");
  ffi::CheckNoOverlap(out, "out", v_cache, "v_cache");

  SplitkvParams p{};
  if (flash_mask.has_value()) {
    const TensorView& fm = flash_mask.value();
    CheckRank(fm, 4, "flash_mask");
    ffi::CheckSameDevice(fm, q, "flash_mask");
    ffi::CheckInt32(fm, "flash_mask");
    if (fm.size(0) != b || (fm.size(1) != 1 && fm.size(1) != hkv) || fm.size(2) != l || fm.size(3) != 2 ||
        fm.stride(3) != 1 || fm.stride(2) != 2 || reinterpret_cast<uintptr_t>(fm.data_ptr()) % 8) {
      Fail("flash_mask must be int32 [batch, 1 | num_kv_heads, cache rows, 2] with dense last two dims");
    }
    p.flash_mask = ffi::ConstData<int>(fm);
    p.fm_sb = fm.stride(0);
    p.fm_sh = fm.size(1) == 1 ? 0 : fm.stride(1);
  }

  CheckRank(workspace, 1, "workspace");
  ffi::CheckSameDevice(workspace, q, "workspace");
  ffi::CheckUInt8(workspace, "workspace");
  ffi::CheckContiguous(workspace, "workspace");
  const int64_t need = WorkspaceBytes(b, m, hq, hkv, d, l, chunk);
  if (workspace.size(0) < need || reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 256) {
    Fail("workspace must be 256-byte aligned with at least " + std::to_string(need) + " bytes");
  }
  if (b * hkv * ChunkCount(l, chunk) > 65535 * 64) Fail("too many chunks");

  p.q = ffi::ConstData<__nv_bfloat16>(q);
  p.q_sb = q.stride(0);
  p.q_sm = q.stride(1);
  p.q_sh = q.stride(2);
  p.k_cache = ffi::Data<__nv_bfloat16>(k_cache);
  p.v_cache = ffi::Data<__nv_bfloat16>(v_cache);
  p.kc_sb = k_cache.stride(0);
  p.kc_sr = k_cache.stride(1);
  p.kc_sh = k_cache.stride(2);
  p.vc_sb = v_cache.stride(0);
  p.vc_sr = v_cache.stride(1);
  p.vc_sh = v_cache.stride(2);
  if (k_new.has_value()) {
    p.k_new = ffi::ConstData<__nv_bfloat16>(k_new.value());
    p.v_new = ffi::ConstData<__nv_bfloat16>(v_new.value());
    p.kn_sb = k_new.value().stride(0);
    p.kn_sm = k_new.value().stride(1);
    p.kn_sh = k_new.value().stride(2);
    p.vn_sb = v_new.value().stride(0);
    p.vn_sm = v_new.value().stride(1);
    p.vn_sh = v_new.value().stride(2);
  }
  p.prefix_len = ffi::ConstData<int>(prefix_len);
  p.cache_batch_idx = cache_batch_idx.has_value() ? ffi::ConstData<int>(cache_batch_idx.value()) : nullptr;
  p.leftpad = cache_leftpad.has_value() ? ffi::ConstData<int>(cache_leftpad.value()) : nullptr;
  p.pool = static_cast<int>(pool);
  auto* ws = static_cast<unsigned char*>(workspace.data_ptr());
  p.ws_o = reinterpret_cast<float*>(ws);
  p.ws_ml = reinterpret_cast<float2*>(ws + PartialBytes(b, m, hq, hkv, d, l, chunk));
  p.out = out.data_ptr();
  p.out_scale = out_scale.has_value() ? static_cast<uint8_t*>(out_scale.value().data_ptr()) : nullptr;
  p.B = static_cast<int>(b);
  p.M = static_cast<int>(m);
  p.L = static_cast<int>(l);
  p.Hkv = static_cast<int>(hkv);
  p.new_pos = static_cast<int>(new_pos);
  p.causal = causal ? 1 : 0;
  p.nchunk = static_cast<int>(ChunkCount(l, chunk));
  p.out_quant = static_cast<int>(out_quant);
  p.scale_log2 = static_cast<float>(softmax_scale * M_LOG2E);

  Run(p, d, g, chunk, pdl, ffi::Stream(q));
}

}  // namespace xinfer::attention_kvcache

TVM_FFI_DLL_EXPORT_TYPED_FUNC(attention_kvcache_splitkv, xinfer::attention_kvcache::launch_attention_kvcache_splitkv);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(attention_kvcache_splitkv_workspace_size,
                              xinfer::attention_kvcache::attention_kvcache_splitkv_workspace_size);
