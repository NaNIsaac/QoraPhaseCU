#include "device.cuh"
#include "kernels.cuh"

namespace qpcu::cuda {

namespace {

// Two complex values, 8 bytes, so each thread still issues one wide load.
struct alignas(8) HalfPair {
  __half2 lo;
  __half2 hi;
};

__global__ void modulus_half_kernel(__half2* w, const float* signed_amp, std::size_t n,
                                    double* errs, const int* counter) {
  const float s = rsqrtf(static_cast<float>(n));
  auto* w2 = reinterpret_cast<HalfPair*>(w);
  const auto* a2 = reinterpret_cast<const float2*>(signed_amp);
  float acc = 0.0f;
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t k = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; k < n / 2; k += stride) {
    const HalfPair v = w2[k];
    const float2 a = a2[k];
    const float2 lo = project(__half22float2(v.lo), a.x, s, 1.0f, acc);
    const float2 hi = project(__half22float2(v.hi), a.y, s, 1.0f, acc);
    w2[k] = {__float22half2_rn(lo), __float22half2_rn(hi)};
  }
  const float total = block_sum_shuffle(acc);
  if (threadIdx.x == 0) atomicAdd(&errs[*counter], static_cast<double>(total));
}

__global__ void update_half_kernel(__half2* x, const __half2* p, const std::uint8_t* support,
                                   std::size_t n, detail::UpdateCoefficients c, int* counter) {
  const float s = rsqrtf(static_cast<float>(n));
  auto* x2 = reinterpret_cast<HalfPair*>(x);
  const auto* p2 = reinterpret_cast<const HalfPair*>(p);
  const auto* s2 = reinterpret_cast<const uchar2*>(support);
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t k = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; k < n / 2; k += stride) {
    const uchar2 in = s2[k];
    const HalfPair pv = p2[k];
    float2 plo = __half22float2(pv.lo);
    float2 phi = __half22float2(pv.hi);
    plo = {plo.x * s, plo.y * s};
    phi = {phi.x * s, phi.y * s};
    if (in.x && in.y) {
      x2[k] = {__float22half2_rn(plo), __float22half2_rn(phi)};
      continue;
    }
    const HalfPair xv = x2[k];
    const float2 lo = combine(__half22float2(xv.lo), plo, in.x, c);
    const float2 hi = combine(__half22float2(xv.hi), phi, in.y, c);
    x2[k] = {__float22half2_rn(lo), __float22half2_rn(hi)};
  }
  if (counter && blockIdx.x == 0 && threadIdx.x == 0) ++*counter;
}

__global__ void to_half_kernel(const float2* in, const std::uint8_t* support, __half2* out,
                               std::size_t n) {
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
    out[i] = support[i] ? __float22half2_rn(in[i]) : __float2half2_rn(0.0f);
  }
}

__global__ void to_float_kernel(const __half2* in, float2* out, std::size_t n) {
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
    out[i] = __half22float2(in[i]);
  }
}

__global__ void magnitude_half_kernel(const __half2* x, float2* out, std::size_t n) {
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
    const float2 v = __half22float2(x[i]);
    out[i] = {sqrtf(v.x * v.x + v.y * v.y), 0.0f};
  }
}

}  // namespace

void modulus_half(__half2* w, const float* signed_amp, std::size_t n, double* errs,
                  const int* counter, int blocks, cudaStream_t stream) {
  modulus_half_kernel<<<blocks, kThreads, 0, stream>>>(w, signed_amp, n, errs, counter);
}

void update_half(__half2* x, const __half2* p, const std::uint8_t* support, std::size_t n,
                 detail::UpdateCoefficients c, int* counter, int blocks, cudaStream_t stream) {
  update_half_kernel<<<blocks, kThreads, 0, stream>>>(x, p, support, n, c, counter);
}

void to_half(const float2* in, const std::uint8_t* support, __half2* out, std::size_t n,
             cudaStream_t stream) {
  to_half_kernel<<<grid_blocks(n), kThreads, 0, stream>>>(in, support, out, n);
}

void to_float(const __half2* in, float2* out, std::size_t n, cudaStream_t stream) {
  to_float_kernel<<<grid_blocks(n), kThreads, 0, stream>>>(in, out, n);
}

void magnitude_half(const __half2* x, float2* out, std::size_t n, cudaStream_t stream) {
  magnitude_half_kernel<<<grid_blocks(n), kThreads, 0, stream>>>(x, out, n);
}

}  // namespace qpcu::cuda
