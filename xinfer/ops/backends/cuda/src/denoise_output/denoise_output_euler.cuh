// denoise_output_euler: one denoising step's tail -- optional pre-normalization of the hidden row, the A x K output head
// (+ bias), dof-masked velocity blend, Euler update of the action and pinned rows.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace xinfer::denoise_output {

enum Prenorm : int { kNone = 0, kRmsNorm = 1, kAdaRms = 2 };

struct Params {
  // rows r = b * M + t; x / out / padding_velocity / pinned_values are contiguous [B, M, A]
  const void* h;          // [B, M, K] BF16 or FP32, last dim contiguous
  int64_t h_sb, h_st;     // element strides of batch / row
  const void* w;          // [A, K] BF16 or FP32, contiguous
  const void* bias;       // [A] (dtype of w) or nullptr
  const void* norm_w;     // rmsnorm: [K]; adarms: scale [K]   (dtype: mod_bf16 ? BF16 : FP32)
  const void* shift;      // adarms: shift [K]
  bool mod_bf16;
  int prenorm;
  float eps;
  const float* x;
  float* out;
  const uint8_t* row_mask;  // [B, M] (0 / 1) or nullptr
  int64_t rm_sb, rm_st;
  const void* dof;          // [B, M, A] or [B, A], uint8 (0 / 1) or FP32, or nullptr
  bool dof_f32;
  int64_t dof_sb, dof_st;
  const float* vpad;        // [B, M, A] (with dof)
  const int* pinned;        // [B] or nullptr
  const float* pinned_values;  // [B, M, A] or nullptr
  float dt;
  const float* dt_t;        // [B] or nullptr (then dt)
  int B, M, A, K;
  float inv_m;  // set by Launch
};

// Launches the fused tail on `stream` (programmatic dependent launch when `pdl`). Supports K == 1024, A <= 32.
cudaError_t Launch(const Params& p, bool h_bf16, bool w_bf16, bool pdl, cudaStream_t stream);

}  // namespace xinfer::denoise_output
