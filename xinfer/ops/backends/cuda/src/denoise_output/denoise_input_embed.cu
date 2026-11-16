// Two kernels of a denoising step on SM90+ (thread-block clusters).
//
// cluster_tail (denoise_output_euler, backend "cluster_tail"): K = 1280, up to 80 rows, A <= 32, 1 or 2 output heads.
//   Per row: adaptive RMSNorm with a per-row modulation row (ada_index into [C, K] scale / shift tables), the
//   normalized row rounded to BF16, projection by the row's head (head_index into FP32 [Hh, A, K]), velocity (flow:
//   v = p; x-prediction: v = (p - x) / max(1 - tau, tau_floor)), optional dof blend, Euler step x + dt * v.
//   Grid: one 4-CTA cluster per (head, pair of output dims); CTA rank = K quarter (320 columns). Each cluster takes
//   the rows of its head; the per-row sums of squares and the two dot-product partials are exchanged through
//   distributed shared memory inside the cluster. No CTA waits for another cluster.
// embed (denoise_input_embed): out[row, n] = BF16(override if override_mask[row] else
//   ((proj + add0) + add1) + add2), proj = sum_k x[row, k] * W[head, n, k] (+ sum_k extra[row, k] * W[head, n, A + k]),
//   zero where input_mask[row] == 0. One CTA per 16 output features x 16 rows; no cross-CTA communication.
//
// Reduction order (fixed by K quarter, lane, vector and element; independent of M, of the other rows and of the
// grid), FMA contraction and rounding are those of the original fused step kernel: this file is compiled with
// nvcc's default --fmad and exact division (src/CMakeLists.txt), and the update uses explicit __*_rn intrinsics.
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/tvm_ffi_cuda_tensor.h"

namespace xinfer::denoise_output {
namespace {

namespace cg = cooperative_groups;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;
namespace ffi = xinfer::ffi;

constexpr int kK = 1280, kKs = 4, kKsl = kK / kKs, kRows = 80, kThreads = 256, kMaxA = 32, kMaxHeads = 2, kMaxTables = 4;
constexpr int kFeat = 16;  // embed: output features per CTA
constexpr int kMaxAin = 2 * kMaxA;

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;"); }

struct EmbedParams {
  const float* x;      // [rows][A]
  const float* extra;  // [rows][A] or null
  const float* w;      // [heads][N][Ain]
  const int* head;     // [rows] or null
  const uint8_t* in_mask;  // [rows] or null
  const float* add[3];     // [rows][N] each, or null
  const float* ovr;        // [rows][N] or null
  const uint8_t* ovr_mask;  // [rows]
  __nv_bfloat16* out;      // [rows][N]
  int rows, a, ain, n, heads, nadd;
};

constexpr int kRowTile = kThreads / kFeat;  // embed: rows per CTA (one output per thread)

struct EmbedSmem {
  float swe[kMaxHeads][kFeat][kMaxAin + 1];  // +1: rows of the 16 features fall in different banks
  float sx[kRowTile][kMaxAin];
};

// grid (N / 16, ceil(rows / 16)): thread (row r0 + tid / 16, feature n0 + tid % 16). The addends, the override and
// the masks are loaded before the projection so their latency overlaps it.
__global__ void __launch_bounds__(kThreads) EmbedKernel(const EmbedParams a) {
  extern __shared__ __align__(16) uint8_t smraw[];
  EmbedSmem& s = *reinterpret_cast<EmbedSmem*>(smraw);
  const int tid = threadIdx.x, n0 = blockIdx.x * kFeat, r0 = blockIdx.y * kRowTile;
  for (int i = tid; i < a.heads * kFeat * a.ain; i += kThreads) {  // weights: before the PDL wait
    const int g = i / (kFeat * a.ain), j = (i / a.ain) % kFeat, k = i % a.ain;
    s.swe[g][j][k] = a.w[((size_t)g * a.n + n0 + j) * a.ain + k];
  }
  pdl_wait();
  const int nr = min(kRowTile, a.rows - r0);
  for (int i = tid; i < nr * a.a; i += kThreads) {
    const int r = i / a.a, k = i % a.a;
    s.sx[r][k] = a.x[(size_t)(r0 + r) * a.a + k];
    if (a.extra) s.sx[r][a.a + k] = a.extra[(size_t)(r0 + r) * a.a + k];
  }
  const int rl = tid / kFeat, j = tid % kFeat, r = r0 + rl, n = n0 + j;
  const bool live = rl < nr;
  float addv[3] = {0.f, 0.f, 0.f}, ov = 0.f;
  int hd = 0;
  bool in = true, ovr = false;
  if (live) {
    const size_t o = (size_t)r * a.n + n;
#pragma unroll
    for (int t = 0; t < 3; ++t)
      if (t < a.nadd) addv[t] = a.add[t][o];
    hd = a.head ? a.head[r] : 0;
    in = a.in_mask ? a.in_mask[r] != 0 : true;
    ovr = a.ovr_mask ? a.ovr_mask[r] != 0 : false;
    if (ovr) ov = a.ovr[o];
  }
  __syncthreads();
  if (tid == 0) pdl_launch();
  if (!live) return;
  const float* wr = s.swe[hd][j];
  const float* xr = s.sx[rl];
  float h = 0.f;
  for (int k = 0; k < a.a; ++k) h += xr[k] * wr[k];
  if (a.extra)
    for (int k = 0; k < a.a; ++k) h += xr[a.a + k] * wr[a.a + k];
  if (!in) h = 0.f;
#pragma unroll
  for (int t = 0; t < 3; ++t)
    if (t < a.nadd) h = __fadd_rn(h, addv[t]);
  if (ovr) h = ov;
  a.out[(size_t)r * a.n + n] = __float2bfloat16_rn(h);
}

[[noreturn]] void Fail(const char* op, const std::string& what) { throw std::runtime_error(std::string(op) + ": " + what); }

void CheckRowsOf(const char* op, const TensorView& t, const TensorView& ref, int64_t rows, int64_t inner, const char* name,
                 bool fp32) {
  ffi::CheckSameDevice(t, ref, name);
  ffi::CheckContiguous(t, name);
  if (fp32) ffi::CheckFloat32(t, name);
  if (ffi::NumElements(t) != rows * inner) Fail(op, std::string(name) + " has the wrong number of elements");
}

const uint8_t* MaskU8(const char* op, const Optional<TensorView>& t, const TensorView& ref, int64_t n, const char* name) {
  if (!t.has_value()) return nullptr;
  const TensorView& v = t.value();
  ffi::CheckSameDevice(v, ref, name);
  ffi::CheckContiguous(v, name);
  if (!ffi::IsUInt8(v.dtype()) || ffi::NumElements(v) != n) Fail(op, std::string(name) + " must be uint8 / bool with " + std::to_string(n) + " elements");
  return static_cast<const uint8_t*>(v.data_ptr());
}

const int* IndexI32(const char* op, const Optional<TensorView>& t, const TensorView& ref, int64_t n, const char* name) {
  if (!t.has_value()) return nullptr;
  const TensorView& v = t.value();
  ffi::CheckSameDevice(v, ref, name);
  ffi::CheckContiguous(v, name);
  ffi::CheckInt32(v, name);
  if (ffi::NumElements(v) != n) Fail(op, std::string(name) + " must have " + std::to_string(n) + " elements");
  return static_cast<const int*>(v.data_ptr());
}

void Launch(const void* kernel, dim3 grid, int smem, int cluster, bool pdl, cudaStream_t st, void** args, const char* op) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  int n = 0;
  if (cluster > 1) {
    at[n].id = cudaLaunchAttributeClusterDimension;
    at[n].val.clusterDim.x = cluster, at[n].val.clusterDim.y = 1, at[n].val.clusterDim.z = 1;
    ++n;
  }
  if (pdl) {
    at[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  cfg.attrs = at;
  cfg.numAttrs = n;
  ffi::CheckCudaError(cudaLaunchKernelExC(&cfg, kernel, args), op);
}

}  // namespace

// out BF16 [B, M, N]; x FP32 [B, M, A]; extra FP32 [B, M, A] or none; weight FP32 [heads, N, A (+A with extra)];
// head_index int32 [B, M]; input_mask / override_mask uint8 [B, M]; addends: up to three FP32 [B, M, N];
// override FP32 [B, M, N].
void denoise_input_embed(TensorView out, TensorView x, Optional<TensorView> extra, TensorView weight,
                         Optional<TensorView> head_index, Optional<TensorView> input_mask, Optional<TensorView> add0,
                         Optional<TensorView> add1, Optional<TensorView> add2, Optional<TensorView> override_rows,
                         Optional<TensorView> override_mask, bool pdl) {
  const char* op = "denoise_input_embed";
  ffi::CheckCuda(x, "x");
  ffi::CheckFloat32(x, "x");
  ffi::CheckContiguous(x, "x");
  if (x.ndim() != 3) Fail(op, "x must be FP32 [B, M, A]");
  const int64_t b = x.size(0), m = x.size(1), a = x.size(2), rows = b * m;
  if (rows < 1 || rows > kRows) Fail(op, "needs 1 <= B * M <= 80");
  if (a < 1 || a > kMaxA) Fail(op, "A must be in [1, 32]");
  ffi::CheckFloat32(weight, "weight");
  ffi::CheckContiguous(weight, "weight");
  ffi::CheckSameDevice(weight, x, "weight");
  const int64_t ain = extra.has_value() ? 2 * a : a;
  if (weight.ndim() != 3 || weight.size(0) < 1 || weight.size(0) > kMaxHeads || weight.size(2) != ain) {
    Fail(op, "weight must be FP32 [1 or 2, N, " + std::to_string(ain) + "]");
  }
  const int64_t heads = weight.size(0), n = weight.size(1);
  if (n < kFeat || n % kFeat || n > 65536) Fail(op, "N must be a positive multiple of 16");
  if (heads > 1 && !head_index.has_value()) Fail(op, "two heads need head_index");
  ffi::CheckBFloat16(out, "out");
  CheckRowsOf(op, out, x, rows, n, "out", false);
  if (out.ndim() != 3 || out.size(0) != b || out.size(1) != m || out.size(2) != n) Fail(op, "out must be BF16 [B, M, N]");
  EmbedParams p{};
  p.x = static_cast<const float*>(x.data_ptr());
  if (extra.has_value()) {
    CheckRowsOf(op, extra.value(), x, rows, a, "extra", true);
    p.extra = static_cast<const float*>(extra.value().data_ptr());
  }
  p.w = static_cast<const float*>(weight.data_ptr());
  p.head = IndexI32(op, head_index, x, rows, "head_index");
  p.in_mask = MaskU8(op, input_mask, x, rows, "input_mask");
  const Optional<TensorView>* adds[3] = {&add0, &add1, &add2};
  for (int t = 0; t < 3; ++t) {
    if (!adds[t]->has_value()) break;
    CheckRowsOf(op, adds[t]->value(), x, rows, n, "addend", true);
    p.add[p.nadd++] = static_cast<const float*>(adds[t]->value().data_ptr());
  }
  if (override_rows.has_value() != override_mask.has_value()) Fail(op, "override and override_mask go together");
  if (override_rows.has_value()) {
    CheckRowsOf(op, override_rows.value(), x, rows, n, "override", true);
    p.ovr = static_cast<const float*>(override_rows.value().data_ptr());
    p.ovr_mask = MaskU8(op, override_mask, x, rows, "override_mask");
  }
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.rows = (int)rows, p.a = (int)a, p.ain = (int)ain, p.n = (int)n, p.heads = (int)heads;

  const int smem = (int)sizeof(EmbedSmem);
  static bool attr = false;
  if (!attr) {
    ffi::CheckCudaError(cudaFuncSetAttribute(EmbedKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), op);
    attr = true;
  }
  void* args[] = {&p};
  Launch((const void*)EmbedKernel, dim3((unsigned)(n / kFeat), (unsigned)((rows + kRowTile - 1) / kRowTile)), smem, 1, pdl,
         ffi::Stream(x), args, op);
}

}  // namespace xinfer::denoise_output

TVM_FFI_DLL_EXPORT_TYPED_FUNC(denoise_input_embed, xinfer::denoise_output::denoise_input_embed);
