#include <cmath>
#include <stdexcept>
#include <vector>

#include "../core/layout.hpp"
#include "common.cuh"
#include "kernels.cuh"
#include "qoraphasecu/operators.hpp"
#include "qoraphasecu/solver.hpp"

namespace qpcu {

bool cuda_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

DeviceInfo cuda_device_info() {
  if (!cuda_available()) throw std::runtime_error("no CUDA device available");
  int device = 0;
  QPCU_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  QPCU_CHECK(cudaGetDeviceProperties(&prop, device));
  // CUDA 13 dropped memoryClockRate from cudaDeviceProp; the attribute remains.
  int clock_khz = 0;
  QPCU_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device));
  DeviceInfo info;
  info.name = prop.name;
  info.sm_major = prop.major;
  info.sm_minor = prop.minor;
  info.memory_bytes = prop.totalGlobalMem;
  // Double data rate: two transfers per clock over the bus.
  info.peak_bandwidth_gbs = 2.0 * clock_khz * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
  return info;
}

namespace cuda {

namespace {

template <class T>
DeviceBuffer<T> upload(std::span<const T> host) {
  DeviceBuffer<T> d(host.size());
  QPCU_CHECK(cudaMemcpy(d.get(), host.data(), host.size_bytes(), cudaMemcpyHostToDevice));
  return d;
}

template <class T>
void download(const DeviceBuffer<T>& d, std::span<T> host) {
  QPCU_CHECK(cudaMemcpy(host.data(), d.get(), host.size_bytes(), cudaMemcpyDeviceToHost));
}

std::vector<int> dims_of(const Shape& shape) {
  std::vector<int> dims;
  for (std::size_t d : shape) dims.push_back(static_cast<int>(d));
  return dims;
}

}  // namespace

float modulus_project(const Problem& problem, std::span<Complex> obj) {
  const std::size_t n = numel(problem.shape);
  if (obj.size() != n) throw std::invalid_argument("obj does not match the problem shape");
  const auto amplitude = detail::to_fft_order<float>(problem.amplitude, problem.shape);
  const auto mask = detail::to_fft_order<std::uint8_t>(problem.mask, problem.shape);
  std::vector<float> signed_amp(n);
  double energy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    signed_amp[i] = mask[i] ? -1.0f : amplitude[i];
    if (!mask[i]) energy += static_cast<double>(amplitude[i]) * amplitude[i];
  }
  auto x = detail::to_fft_order<Complex>(obj, problem.shape);

  Stream stream;
  const auto dims = dims_of(problem.shape);
  const FftPlan plan(dims.data(), static_cast<int>(dims.size()), 1, stream);
  auto d_amp = upload<float>(signed_amp);
  DeviceBuffer<float2> d_x(n);
  QPCU_CHECK(cudaMemcpy(d_x.get(), x.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
  const int blocks = grid_blocks(n / 2 + n % 2);
  DeviceBuffer<float> partials(static_cast<std::size_t>(blocks));
  plan.forward(d_x.get(), d_x.get());
  modulus_fused(d_x.get(), d_amp.get(), n, 1, partials.get(), blocks, stream);
  plan.inverse(d_x.get(), d_x.get());
  stream.sync();

  std::vector<float> host_partials(static_cast<std::size_t>(blocks));
  download(partials, std::span(host_partials));
  QPCU_CHECK(cudaMemcpy(x.data(), d_x.get(), n * sizeof(float2), cudaMemcpyDeviceToHost));
  std::ranges::copy(detail::to_centered<Complex>(x, problem.shape), obj.begin());
  double num = 0.0;
  for (float p : host_partials) num += p;
  return static_cast<float>(std::sqrt(num / (energy + detail::kEps)));
}

void support_project(std::span<const std::uint8_t> support, std::span<Complex> obj) {
  if (obj.size() != support.size()) throw std::invalid_argument("support does not match obj");
  auto d_support = upload(support);
  DeviceBuffer<float2> d_x(obj.size());
  QPCU_CHECK(cudaMemcpy(d_x.get(), obj.data(), obj.size_bytes(), cudaMemcpyHostToDevice));
  Stream stream;
  cuda::support_project(d_x.get(), d_support.get(), obj.size(), 1, stream);
  stream.sync();
  QPCU_CHECK(cudaMemcpy(obj.data(), d_x.get(), obj.size_bytes(), cudaMemcpyDeviceToHost));
}

std::vector<float> gaussian_blur(std::span<const float> image, const Shape& shape, float sigma) {
  const std::size_t n = numel(shape);
  if (image.size() != n) throw std::invalid_argument("image does not match shape");
  Stream stream;
  const auto dims = dims_of(shape);
  const FftPlan plan(dims.data(), static_cast<int>(dims.size()), 1, stream);
  auto d_image = upload(image);
  DeviceBuffer<float2> w(n);
  real_to_complex(d_image.get(), w.get(), n, stream);
  plan.forward(w.get(), w.get());
  const auto d = detail::dims3(shape);
  gaussian(w.get(),
           {static_cast<std::uint32_t>(d[0]), static_cast<std::uint32_t>(d[1]),
            static_cast<std::uint32_t>(d[2])},
           sigma, 1, stream);
  plan.inverse(w.get(), w.get());
  stream.sync();
  std::vector<float2> host(n);
  download(w, std::span(host));
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = host[i].x;
  return out;
}

}  // namespace cuda

}  // namespace qpcu
