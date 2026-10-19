#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "../core/pointwise.hpp"
#include "qoraphasecu/recipe.hpp"

// Host-side launchers. Arrays are FFT order; `batch` starts lie back to back with
// stride n, and every kernel puts the start index in blockIdx.y.

namespace qpcu::cuda {

inline constexpr int kThreads = 256;

// Grid width for grid-stride loops over `work` items per start.
int grid_blocks(std::size_t work);

// --- naive: one operation per kernel, one element per thread -----------------

void scale_naive(float2* w, std::size_t n, float s, cudaStream_t stream);
// Per-block sums of (|w| - amp)^2 over usable pixels into partials[blockIdx].
void error_naive(const float2* w, const float* amp, const std::uint8_t* mask, std::size_t n,
                 float* partials, cudaStream_t stream);
int error_naive_blocks(std::size_t n);
void modulus_naive(float2* w, const float* amp, const std::uint8_t* mask, std::size_t n,
                   cudaStream_t stream);
void update_naive(float2* x, const float2* p, const std::uint8_t* support, std::size_t n,
                  detail::UpdateCoefficients c, cudaStream_t stream);

// --- fused ---------------------------------------------------------------------
// `signed_amp` is the amplitude with masked pixels set to -1, so the mask costs
// no extra load. The kernel takes the unnormalised forward FFT and leaves the
// input of an unnormalised inverse FFT that lands on the unitary result.

// Error partials reduced in shared memory, one per block, summed by the host.
void modulus_fused(float2* w, const float* signed_amp, std::size_t n, int batch, float* partials,
                   int blocks, cudaStream_t stream);
// Error reduced with warp shuffles and accumulated into errs[*counter * batch + start].
void modulus_reduced(float2* w, const float* signed_amp, std::size_t n, int batch, double* errs,
                     const int* counter, int blocks, cudaStream_t stream);
// Increments *counter once per launch when it is not null.
void update_fused(float2* x, const float2* p, const std::uint8_t* support, std::size_t n, int batch,
                  detail::UpdateCoefficients c, int* counter, int blocks, cudaStream_t stream);

// --- cuFFT LTO callbacks (callbacks.cu) ---------------------------------------
// The forward transform stores the modulus projection, the inverse stores the
// updated x, so the wave is never read or written by a separate kernel.

// Every element adds its error term to slot offset % kErrorSlots, so that
// concurrent atomics rarely share an address; collect_error sums the slots.
inline constexpr int kErrorSlots = 4096;

struct ModulusCallbackInfo {
  const float* signed_amp;
  float* slots;  // kErrorSlots, zero between iterations
  float scale;   // 1/sqrt(n)
};

// x is a double buffer of two halves of `span` elements; the inverse transform
// writes one half and the callback reads the old x from the other.
struct UpdateCallbackInfo {
  float2* x;
  std::size_t span;
  const std::uint8_t* support;  // `span` elements, same layout as each half
  detail::UpdateCoefficients c;
};

// errs[*counter] += sum of slots; zeroes the slots and increments *counter.
void collect_error(float* slots, double* errs, int* counter, cudaStream_t stream);

// --- setup and shrinkwrap ------------------------------------------------------

// amp * exp(i 2 pi u) / sqrt(n), the input of an unnormalised inverse FFT.
void random_wave(float2* w, const float* amp, std::size_t n, int batch, std::uint64_t seed,
                 std::uint64_t first_start, cudaStream_t stream);
void support_project(float2* x, const std::uint8_t* support, std::size_t n, int batch,
                     cudaStream_t stream);
void magnitude(const float2* x, float2* out, std::size_t n, int batch, cudaStream_t stream);
void real_to_complex(const float* in, float2* out, std::size_t n, cudaStream_t stream);
// Multiplies by the Gaussian transfer function and by 1/n.
void gaussian(float2* w, std::array<std::uint32_t, 3> dims, float sigma, int batch,
              cudaStream_t stream);

struct LevelAccumulator {
  double sum;
  unsigned long long count;
  int max_bits;  // float bits; ordered like the float for non-negative values
};

// Reduces the threshold reference of each start into acc[start] (zeroed here).
void shrinkwrap_level(const float2* blurred, const std::uint8_t* support, std::size_t n, int batch,
                      Threshold method, LevelAccumulator* acc, cudaStream_t stream);
void shrinkwrap_threshold(const float2* blurred, const LevelAccumulator* acc, std::size_t n,
                          int batch, Threshold method, float threshold, std::uint8_t* support,
                          cudaStream_t stream);

// --- fp16 storage --------------------------------------------------------------
// x and the wave are stored as half2 (re, im); arithmetic stays in fp32. The
// amplitude is pre-divided by its maximum so values stay inside fp16 range, and
// the 1/sqrt(n) of the inverse is applied in the update rather than before the
// inverse FFT, where it would push weak pixels into fp16 subnormals. n is even.

void modulus_half(__half2* w, const float* signed_amp, std::size_t n, double* errs,
                  const int* counter, int blocks, cudaStream_t stream);
void update_half(__half2* x, const __half2* p, const std::uint8_t* support, std::size_t n,
                 detail::UpdateCoefficients c, int* counter, int blocks, cudaStream_t stream);
// out = in, zeroed outside the support.
void to_half(const float2* in, const std::uint8_t* support, __half2* out, std::size_t n,
             cudaStream_t stream);
void to_float(const __half2* in, float2* out, std::size_t n, cudaStream_t stream);
void magnitude_half(const __half2* x, float2* out, std::size_t n, cudaStream_t stream);
void to_half(const float2* in, __half2* out, std::size_t n, cudaStream_t stream);

// Mixed precision: x stays fp32 and the update accumulates into it, but also
// writes the fp16 copy `xh` that the next fp16 forward transform reads.
void update_mixed(float2* x, const __half2* p, const std::uint8_t* support, std::size_t n,
                  detail::UpdateCoefficients c, __half2* xh, int* counter, int blocks,
                  cudaStream_t stream);

}  // namespace qpcu::cuda
