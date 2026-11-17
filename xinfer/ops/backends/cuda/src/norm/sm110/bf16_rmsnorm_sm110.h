// Launcher of the SM110 BF16 RMSNorm (bf16_rmsnorm_sm110.cu), called by the generic rmsnorm (rmsnorm.cu).
// Only builds for SM110 (XINFER_WITH_SM110) contain it.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace bf16_rmsnorm_sm110 {

// True when the kernel handles this hidden size (hidden = 256 * {1..8, 10, 12, 14, 16}).
bool Supported(int64_t hidden);

// out / input: [rows, hidden] packed, weight: [hidden]; 16-byte aligned. Caller checks the device is SM110.
void Launch(__nv_bfloat16* out, const __nv_bfloat16* input, const __nv_bfloat16* weight, int64_t rows,
            int64_t hidden, float epsilon, int block_size, int device_id, cudaStream_t stream);

}  // namespace bf16_rmsnorm_sm110
