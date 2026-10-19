#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qpcu::cuda {

inline void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

inline void check(cufftResult status, const char* what) {
  if (status != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cuFFT error " + std::to_string(status));
  }
}

#define QPCU_CHECK(expr) ::qpcu::cuda::check((expr), #expr)

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) : size_(n) {
    if (n) QPCU_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept
      : ptr_(std::exchange(o.ptr_, nullptr)), size_(std::exchange(o.size_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    std::swap(ptr_, o.ptr_);
    std::swap(size_, o.size_);
    return *this;
  }
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }

  T* get() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

class Stream {
 public:
  Stream() { QPCU_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() { cudaStreamDestroy(s_); }
  operator cudaStream_t() const { return s_; }
  void sync() const { QPCU_CHECK(cudaStreamSynchronize(s_)); }

 private:
  cudaStream_t s_{};
};

// C2C plan over `batch` contiguous transforms of `shape`, bound to `stream`.
class FftPlan {
 public:
  FftPlan(const int* dims, int ndim, int batch, cudaStream_t stream) {
    QPCU_CHECK(cufftPlanMany(&plan_, ndim, const_cast<int*>(dims), nullptr, 1, 0, nullptr, 1, 0,
                             CUFFT_C2C, batch));
    QPCU_CHECK(cufftSetStream(plan_, stream));
  }
  FftPlan(const FftPlan&) = delete;
  FftPlan& operator=(const FftPlan&) = delete;
  ~FftPlan() { cufftDestroy(plan_); }

  void forward(cufftComplex* in, cufftComplex* out) const {
    QPCU_CHECK(cufftExecC2C(plan_, in, out, CUFFT_FORWARD));
  }
  void inverse(cufftComplex* in, cufftComplex* out) const {
    QPCU_CHECK(cufftExecC2C(plan_, in, out, CUFFT_INVERSE));
  }

 private:
  cufftHandle plan_{};
};

// Single fp16 C2C transform; cuFFT supports it for power-of-two extents only.
class HalfFftPlan {
 public:
  template <class Shape>
  HalfFftPlan(const Shape& shape, cudaStream_t stream) {
    std::vector<long long> dims(shape.begin(), shape.end());
    std::size_t workspace = 0;
    QPCU_CHECK(cufftCreate(&plan_));
    QPCU_CHECK(cufftXtMakePlanMany(plan_, static_cast<int>(dims.size()), dims.data(), nullptr, 1, 0,
                                   CUDA_C_16F, nullptr, 1, 0, CUDA_C_16F, 1, &workspace,
                                   CUDA_C_16F));
    QPCU_CHECK(cufftSetStream(plan_, stream));
  }
  HalfFftPlan(const HalfFftPlan&) = delete;
  HalfFftPlan& operator=(const HalfFftPlan&) = delete;
  ~HalfFftPlan() { cufftDestroy(plan_); }

  void forward(void* in, void* out) const {
    QPCU_CHECK(cufftXtExec(plan_, in, out, CUFFT_FORWARD));
  }
  void inverse(void* in, void* out) const {
    QPCU_CHECK(cufftXtExec(plan_, in, out, CUFFT_INVERSE));
  }

 private:
  cufftHandle plan_{};
};

}  // namespace qpcu::cuda
