#pragma once

// Element-wise formulas shared by the CPU reference and the CUDA kernels, so the
// two backends evaluate the same expressions and can be compared closely.

#include <cstdint>

#if defined(__CUDACC__)
#define QPCU_HD __host__ __device__ __forceinline__
#else
#define QPCU_HD inline
#endif

namespace qpcu::detail {

inline constexpr float kEps = 1e-12f;
inline constexpr float kTwoPi = 6.28318530717958647692f;

QPCU_HD std::uint64_t mix64(std::uint64_t z) {
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// Counter-based uniform in [0, 1): any element of any start can be drawn
// independently, which is what lets the batched GPU path and the CPU loop
// produce bit-identical starting phases.
QPCU_HD float uniform(std::uint64_t seed, std::uint64_t start, std::uint64_t index) {
  const std::uint64_t key = mix64(seed ^ mix64(start + 1));
  return static_cast<float>(mix64(key + index) >> 40) * 0x1p-24f;
}

// Every update in the ladder keeps P_M x inside the support and sets
// a x + b P_M x outside it.
struct UpdateCoefficients {
  float a;
  float b;
};

QPCU_HD UpdateCoefficients er_coefficients() { return {0.0f, 0.0f}; }
QPCU_HD UpdateCoefficients hio_coefficients(float beta) { return {1.0f, -beta}; }
// R_S R_M x is 2 P_M x - x inside and x - 2 P_M x outside, which collapses
// RAAR to P_M x inside and beta x + (1 - 2 beta) P_M x outside.
QPCU_HD UpdateCoefficients raar_coefficients(float beta) { return {beta, 1.0f - 2.0f * beta}; }

// Signed frequency of FFT-order index k on an axis of length n, in cycles/pixel.
QPCU_HD float fftfreq(std::uint32_t k, std::uint32_t n) {
  const int s = k < (n + 1) / 2 ? static_cast<int>(k) : static_cast<int>(k) - static_cast<int>(n);
  return static_cast<float>(s) / static_cast<float>(n);
}

}  // namespace qpcu::detail
