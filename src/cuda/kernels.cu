#include <algorithm>
#include <cmath>

#include "common.cuh"
#include "kernels.cuh"

namespace qpcu::cuda {

using detail::kEps;
using detail::UpdateCoefficients;

namespace {

int sm_count() {
  static const int count = [] {
    int device = 0;
    int n = 0;
    QPCU_CHECK(cudaGetDevice(&device));
    QPCU_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device));
    return n;
  }();
  return count;
}

unsigned blocks_for(std::size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

// --- naive -----------------------------------------------------------------------

__global__ void scale_naive_kernel(float2* w, std::size_t n, float s) {
  const std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
  if (i < n) w[i] = {w[i].x * s, w[i].y * s};
}

__global__ void error_naive_kernel(const float2* w, const float* amp, const std::uint8_t* mask,
                                   std::size_t n, float* partials) {
  __shared__ float buf[kThreads];
  const std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
  float v = 0.0f;
  if (i < n && !mask[i]) {
    const float d = sqrtf(w[i].x * w[i].x + w[i].y * w[i].y) - amp[i];
    v = d * d;
  }
  buf[threadIdx.x] = v;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = buf[0];
}

__global__ void modulus_naive_kernel(float2* w, const float* amp, const std::uint8_t* mask,
                                     std::size_t n) {
  const std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
  if (i >= n) return;
  const float2 v = w[i];
  const float mag = sqrtf(v.x * v.x + v.y * v.y);
  const float f = (mask[i] ? mag : amp[i]) / (mag + kEps);
  w[i] = {v.x * f, v.y * f};
}

__global__ void update_naive_kernel(float2* x, const float2* p, const std::uint8_t* support,
                                    std::size_t n, UpdateCoefficients c) {
  const std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
  if (i >= n) return;
  const float2 pi = p[i];
  if (support[i]) {
    x[i] = pi;
  } else {
    const float2 xi = x[i];
    x[i] = {c.a * xi.x + c.b * pi.x, c.a * xi.y + c.b * pi.y};
  }
}

// --- fused -------------------------------------------------------------------------

// One element of the modulus projection on the unnormalised forward FFT. `s` is
// 1/sqrt(n); applied twice it also prepares the unnormalised inverse.
__device__ __forceinline__ float2 project(float2 w, float a, float s, float& acc) {
  const float re = w.x * s;
  const float im = w.y * s;
  const float mag = sqrtf(re * re + im * im);
  float target = mag;
  if (a >= 0.0f) {
    const float d = mag - a;
    acc += d * d;
    target = a;
  }
  const float f = target / (mag + kEps) * s;
  return {re * f, im * f};
}

__device__ __forceinline__ float2 combine(float2 x, float2 p, std::uint8_t inside,
                                          UpdateCoefficients c) {
  return inside ? p : float2{c.a * x.x + c.b * p.x, c.a * x.y + c.b * p.y};
}

// Grid-stride over pairs of complex values so each thread issues 16-byte loads.
// Only used for even n, where every start is 16-byte aligned.
template <class Body>
__device__ __forceinline__ void for_each_pair(std::size_t n, Body body) {
  const std::size_t pairs = n / 2;
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t k = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; k < pairs; k += stride) {
    body(k);
  }
}

__device__ __forceinline__ float block_sum_tree(float v) {
  __shared__ float buf[kThreads];
  buf[threadIdx.x] = v;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
    __syncthreads();
  }
  return buf[0];
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

template <bool kVector>
__device__ __forceinline__ float modulus_body(float2* w, const float* signed_amp, std::size_t n) {
  const float s = rsqrtf(static_cast<float>(n));
  float acc = 0.0f;
  if constexpr (kVector) {
    auto* w4 = reinterpret_cast<float4*>(w);
    const auto* a2 = reinterpret_cast<const float2*>(signed_amp);
    for_each_pair(n, [&](std::size_t k) {
      const float4 v = w4[k];
      const float2 a = a2[k];
      const float2 lo = project({v.x, v.y}, a.x, s, acc);
      const float2 hi = project({v.z, v.w}, a.y, s, acc);
      w4[k] = {lo.x, lo.y, hi.x, hi.y};
    });
  } else {
    const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
      w[i] = project(w[i], signed_amp[i], s, acc);
    }
  }
  return acc;
}

template <bool kVector>
__global__ void modulus_fused_kernel(float2* w, const float* signed_amp, std::size_t n,
                                     float* partials) {
  const float acc = modulus_body<kVector>(w + blockIdx.y * n, signed_amp, n);
  const float total = block_sum_tree(acc);
  if (threadIdx.x == 0) partials[blockIdx.y * gridDim.x + blockIdx.x] = total;
}

template <bool kVector>
__global__ void modulus_reduced_kernel(float2* w, const float* signed_amp, std::size_t n,
                                       double* errs, const int* counter) {
  const float acc = modulus_body<kVector>(w + blockIdx.y * n, signed_amp, n);
  const float total = block_sum_shuffle(acc);
  if (threadIdx.x == 0) {
    atomicAdd(&errs[static_cast<std::size_t>(*counter) * gridDim.y + blockIdx.y],
              static_cast<double>(total));
  }
}

template <bool kVector>
__global__ void update_fused_kernel(float2* x, const float2* p, const std::uint8_t* support,
                                    std::size_t n, UpdateCoefficients c, int* counter) {
  x += blockIdx.y * n;
  p += blockIdx.y * n;
  support += blockIdx.y * n;
  if constexpr (kVector) {
    auto* x4 = reinterpret_cast<float4*>(x);
    const auto* p4 = reinterpret_cast<const float4*>(p);
    const auto* s2 = reinterpret_cast<const uchar2*>(support);
    for_each_pair(n, [&](std::size_t k) {
      const uchar2 in = s2[k];
      const float4 pv = p4[k];
      // Inside the support the old x is not needed, so skip its load.
      if (in.x && in.y) {
        x4[k] = pv;
        return;
      }
      const float4 xv = x4[k];
      const float2 lo = combine({xv.x, xv.y}, {pv.x, pv.y}, in.x, c);
      const float2 hi = combine({xv.z, xv.w}, {pv.z, pv.w}, in.y, c);
      x4[k] = {lo.x, lo.y, hi.x, hi.y};
    });
  } else {
    const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
      x[i] = combine(x[i], p[i], support[i], c);
    }
  }
  if (counter && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) ++*counter;
}

// --- setup and shrinkwrap -------------------------------------------------------------

__global__ void random_wave_kernel(float2* w, const float* amp, std::size_t n, std::uint64_t seed,
                                   std::uint64_t first_start) {
  const float s = rsqrtf(static_cast<float>(n));
  const std::uint64_t start = first_start + blockIdx.y;
  w += blockIdx.y * n;
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
    const float phase = detail::kTwoPi * detail::uniform(seed, start, i);
    const float a = amp[i] * s;
    w[i] = {a * cosf(phase), a * sinf(phase)};
  }
}

__global__ void support_project_kernel(float2* x, const std::uint8_t* support, std::size_t n) {
  x += blockIdx.y * n;
  support += blockIdx.y * n;
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
    if (!support[i]) x[i] = {0.0f, 0.0f};
  }
}

__global__ void magnitude_kernel(const float2* x, float2* out, std::size_t n) {
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  x += blockIdx.y * n;
  out += blockIdx.y * n;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
    out[i] = {sqrtf(x[i].x * x[i].x + x[i].y * x[i].y), 0.0f};
  }
}

__global__ void real_to_complex_kernel(const float* in, float2* out, std::size_t n) {
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
    out[i] = {in[i], 0.0f};
  }
}

__global__ void gaussian_kernel(float2* w, uint3 dims, float k, float inv_n) {
  const std::size_t n = std::size_t{dims.x} * dims.y * dims.z;
  w += blockIdx.y * n;
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
    const auto ix = static_cast<std::uint32_t>(i % dims.z);
    const auto iy = static_cast<std::uint32_t>((i / dims.z) % dims.y);
    const auto iz = static_cast<std::uint32_t>(i / (std::size_t{dims.z} * dims.y));
    const float fz = detail::fftfreq(iz, dims.x);
    const float fy = detail::fftfreq(iy, dims.y);
    const float fx = detail::fftfreq(ix, dims.z);
    const float g = expf(k * (fz * fz + fy * fy + fx * fx)) * inv_n;
    w[i] = {w[i].x * g, w[i].y * g};
  }
}

__global__ void level_kernel(const float2* blurred, const std::uint8_t* support, std::size_t n,
                             Threshold method, LevelAccumulator* acc) {
  blurred += blockIdx.y * n;
  support += blockIdx.y * n;
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  float sum = 0.0f;
  float count = 0.0f;
  float peak = 0.0f;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
    const float v = blurred[i].x;
    if (method == Threshold::max) {
      peak = fmaxf(peak, v);
    } else if (support[i]) {
      sum += v * v;
      count += 1.0f;
    }
  }
  LevelAccumulator& a = acc[blockIdx.y];
  if (method == Threshold::max) {
    // Non-negative floats order like their bit patterns, and the peak of a
    // blurred magnitude is never negative.
    for (int o = 16; o > 0; o >>= 1) peak = fmaxf(peak, __shfl_down_sync(0xffffffffu, peak, o));
    if (threadIdx.x % 32 == 0) atomicMax(&a.max_bits, __float_as_int(peak));
  } else {
    sum = block_sum_shuffle(sum);
    __syncthreads();
    count = block_sum_shuffle(count);
    if (threadIdx.x == 0) {
      atomicAdd(&a.sum, static_cast<double>(sum));
      atomicAdd(&a.count, static_cast<unsigned long long>(count));
    }
  }
}

__global__ void threshold_kernel(const float2* blurred, const LevelAccumulator* acc, std::size_t n,
                                 Threshold method, float threshold, std::uint8_t* support) {
  const LevelAccumulator& a = acc[blockIdx.y];
  const float level =
      method == Threshold::max
          ? __int_as_float(a.max_bits)
          : static_cast<float>(sqrt(a.sum / fmax(static_cast<double>(a.count), 1.0)));
  const float cut = threshold * level;
  blurred += blockIdx.y * n;
  support += blockIdx.y * n;
  const std::size_t stride = std::size_t{gridDim.x} * blockDim.x;
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += stride) {
    support[i] = blurred[i].x > cut;
  }
}

dim3 grid(int blocks, int batch) {
  return {static_cast<unsigned>(blocks), static_cast<unsigned>(batch), 1};
}

}  // namespace

int grid_blocks(std::size_t work) {
  return static_cast<int>(std::min<std::size_t>(std::max<std::size_t>(blocks_for(work), 1),
                                                static_cast<std::size_t>(sm_count()) * 8));
}

void scale_naive(float2* w, std::size_t n, float s, cudaStream_t stream) {
  scale_naive_kernel<<<blocks_for(n), kThreads, 0, stream>>>(w, n, s);
}

int error_naive_blocks(std::size_t n) { return static_cast<int>(blocks_for(n)); }

void error_naive(const float2* w, const float* amp, const std::uint8_t* mask, std::size_t n,
                 float* partials, cudaStream_t stream) {
  error_naive_kernel<<<blocks_for(n), kThreads, 0, stream>>>(w, amp, mask, n, partials);
}

void modulus_naive(float2* w, const float* amp, const std::uint8_t* mask, std::size_t n,
                   cudaStream_t stream) {
  modulus_naive_kernel<<<blocks_for(n), kThreads, 0, stream>>>(w, amp, mask, n);
}

void update_naive(float2* x, const float2* p, const std::uint8_t* support, std::size_t n,
                  UpdateCoefficients c, cudaStream_t stream) {
  update_naive_kernel<<<blocks_for(n), kThreads, 0, stream>>>(x, p, support, n, c);
}

void modulus_fused(float2* w, const float* signed_amp, std::size_t n, int batch, float* partials,
                   int blocks, cudaStream_t stream) {
  if (n % 2 == 0) {
    modulus_fused_kernel<true>
        <<<grid(blocks, batch), kThreads, 0, stream>>>(w, signed_amp, n, partials);
  } else {
    modulus_fused_kernel<false>
        <<<grid(blocks, batch), kThreads, 0, stream>>>(w, signed_amp, n, partials);
  }
}

void modulus_reduced(float2* w, const float* signed_amp, std::size_t n, int batch, double* errs,
                     const int* counter, int blocks, cudaStream_t stream) {
  if (n % 2 == 0) {
    modulus_reduced_kernel<true>
        <<<grid(blocks, batch), kThreads, 0, stream>>>(w, signed_amp, n, errs, counter);
  } else {
    modulus_reduced_kernel<false>
        <<<grid(blocks, batch), kThreads, 0, stream>>>(w, signed_amp, n, errs, counter);
  }
}

void update_fused(float2* x, const float2* p, const std::uint8_t* support, std::size_t n, int batch,
                  UpdateCoefficients c, int* counter, int blocks, cudaStream_t stream) {
  if (n % 2 == 0) {
    update_fused_kernel<true>
        <<<grid(blocks, batch), kThreads, 0, stream>>>(x, p, support, n, c, counter);
  } else {
    update_fused_kernel<false>
        <<<grid(blocks, batch), kThreads, 0, stream>>>(x, p, support, n, c, counter);
  }
}

void random_wave(float2* w, const float* amp, std::size_t n, int batch, std::uint64_t seed,
                 std::uint64_t first_start, cudaStream_t stream) {
  random_wave_kernel<<<grid(grid_blocks(n), batch), kThreads, 0, stream>>>(w, amp, n, seed,
                                                                           first_start);
}

void support_project(float2* x, const std::uint8_t* support, std::size_t n, int batch,
                     cudaStream_t stream) {
  support_project_kernel<<<grid(grid_blocks(n), batch), kThreads, 0, stream>>>(x, support, n);
}

void magnitude(const float2* x, float2* out, std::size_t n, int batch, cudaStream_t stream) {
  magnitude_kernel<<<grid(grid_blocks(n), batch), kThreads, 0, stream>>>(x, out, n);
}

void real_to_complex(const float* in, float2* out, std::size_t n, cudaStream_t stream) {
  real_to_complex_kernel<<<grid_blocks(n), kThreads, 0, stream>>>(in, out, n);
}

void gaussian(float2* w, std::array<std::uint32_t, 3> dims, float sigma, int batch,
              cudaStream_t stream) {
  const std::size_t n = std::size_t{dims[0]} * dims[1] * dims[2];
  const float pi = detail::kTwoPi / 2;
  const float k = -2.0f * pi * pi * sigma * sigma;
  gaussian_kernel<<<grid(grid_blocks(n), batch), kThreads, 0, stream>>>(
      w, {dims[0], dims[1], dims[2]}, k, 1.0f / static_cast<float>(n));
}

void shrinkwrap_level(const float2* blurred, const std::uint8_t* support, std::size_t n, int batch,
                      Threshold method, LevelAccumulator* acc, cudaStream_t stream) {
  QPCU_CHECK(cudaMemsetAsync(acc, 0, sizeof(LevelAccumulator) * batch, stream));
  level_kernel<<<grid(grid_blocks(n), batch), kThreads, 0, stream>>>(blurred, support, n, method,
                                                                     acc);
}

void shrinkwrap_threshold(const float2* blurred, const LevelAccumulator* acc, std::size_t n,
                          int batch, Threshold method, float threshold, std::uint8_t* support,
                          cudaStream_t stream) {
  threshold_kernel<<<grid(grid_blocks(n), batch), kThreads, 0, stream>>>(blurred, acc, n, method,
                                                                         threshold, support);
}

}  // namespace qpcu::cuda
