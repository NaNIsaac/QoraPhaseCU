#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "../core/pointwise.hpp"
#include "kernels.cuh"

// Device helpers shared by the fp32 and fp16 kernels.

namespace qpcu::cuda {

// One element of the modulus projection on the unnormalised forward FFT. `s` is
// 1/sqrt(n). `out_scale` = s also prepares the unnormalised inverse; the fp16
// path passes 1 and applies s after the inverse instead, to stay in range.
__device__ __forceinline__ float2 project(float2 w, float a, float s, float out_scale, float& acc) {
  const float re = w.x * s;
  const float im = w.y * s;
  const float mag = sqrtf(re * re + im * im);
  float target = mag;
  if (a >= 0.0f) {
    const float d = mag - a;
    acc += d * d;
    target = a;
  }
  const float f = target / (mag + detail::kEps) * out_scale;
  return {re * f, im * f};
}

__device__ __forceinline__ float2 combine(float2 x, float2 p, std::uint8_t inside,
                                          detail::UpdateCoefficients c) {
  return inside ? p : float2{c.a * x.x + c.b * p.x, c.a * x.y + c.b * p.y};
}

__device__ __forceinline__ float block_sum_shuffle(float v) {
  __shared__ float warp_sums[kThreads / 32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  const unsigned lane = threadIdx.x % 32;
  const unsigned warp = threadIdx.x / 32;
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < blockDim.x / 32 ? warp_sums[lane] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  }
  return v;
}

}  // namespace qpcu::cuda
