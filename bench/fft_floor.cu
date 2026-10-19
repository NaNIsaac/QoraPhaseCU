#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "fft_floor.hpp"

namespace {

void check(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

}  // namespace

double fft_pair_ms(const qpcu::Shape& shape, int iters, bool half) {
  std::vector<long long> dims(shape.begin(), shape.end());
  const std::size_t n = qpcu::numel(shape);
  cufftComplex* data = nullptr;
  check(cudaMalloc(&data, n * sizeof(cufftComplex)) == cudaSuccess, "cudaMalloc");
  check(cudaMemset(data, 0, n * sizeof(cufftComplex)) == cudaSuccess, "cudaMemset");
  const cudaDataType type = half ? CUDA_C_16F : CUDA_C_32F;
  cufftHandle plan{};
  std::size_t workspace = 0;
  check(cufftCreate(&plan) == CUFFT_SUCCESS, "cufftCreate");
  check(cufftXtMakePlanMany(plan, static_cast<int>(dims.size()), dims.data(), nullptr, 1, 0, type,
                            nullptr, 1, 0, type, 1, &workspace, type) == CUFFT_SUCCESS,
        "cufftXtMakePlanMany");
  // Replayed from a graph like the graph variants, so launch overhead does not
  // set the floor at small sizes.
  cudaStream_t stream{};
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cufftSetStream(plan, stream);
  constexpr int kPerGraph = 10;
  cudaGraph_t graph{};
  cudaGraphExec_t exec{};
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess, "capture");
  for (int i = 0; i < kPerGraph; ++i) {
    cufftXtExec(plan, data, data, CUFFT_FORWARD);
    cufftXtExec(plan, data, data, CUFFT_INVERSE);
  }
  check(cudaStreamEndCapture(stream, &graph) == cudaSuccess, "end capture");
  check(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess, "instantiate");
  const int launches = std::max(1, iters / kPerGraph);
  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaGraphLaunch(exec, stream);
  cudaEventRecord(start, stream);
  for (int i = 0; i < launches; ++i) cudaGraphLaunch(exec, stream);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cufftDestroy(plan);
  cudaFree(data);
  return ms / (launches * kPerGraph);
}
