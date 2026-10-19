// cuFFT store callbacks for Variant::callback. This file is not part of the
// library build: it is compiled to LTO-IR (cmake/cufft_callbacks.cmake) and
// cuFFT links the functions into its own kernels when the plan is made. cuFFT
// declares them itself with C++ linkage, so they must not be extern "C"
// (plan creation then fails with CUFFT_NVJITLINK_FAILURE).

#include <cufft.h>

#include "device.cuh"

using qpcu::cuda::ModulusCallbackInfo;
using qpcu::cuda::UpdateCallbackInfo;

__device__ void qpcu_store_modulus(void* out, unsigned long long offset, cufftComplex element,
                                   void* info, void*) {
  const auto& in = *static_cast<const ModulusCallbackInfo*>(info);
  float acc = 0.0f;
  static_cast<float2*>(out)[offset] =
      qpcu::cuda::project(element, in.signed_amp[offset], in.scale, in.scale, acc);
  // One element per call leaves nothing to reduce locally. A warp-shuffle sum
  // before a single atomic was slower than spreading plain atomics over slots.
  atomicAdd(&in.slots[offset % qpcu::cuda::kErrorSlots], acc);
}

__device__ void qpcu_store_update(void* out, unsigned long long offset, cufftComplex element,
                                  void* info, void*) {
  const auto& in = *static_cast<const UpdateCallbackInfo*>(info);
  auto* x_new = static_cast<float2*>(out);
  const std::size_t i = static_cast<std::size_t>(x_new - in.x) + offset;
  const std::size_t pos = i < in.span ? i : i - in.span;
  if (in.support[pos]) {
    x_new[offset] = element;
    return;
  }
  const float2 x_old = in.x[i < in.span ? i + in.span : pos];
  x_new[offset] = qpcu::cuda::combine(x_old, element, 0, in.c);
}
