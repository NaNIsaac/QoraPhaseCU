#include "fft.hpp"

#include <pocketfft_hdronly.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>

namespace qpcu::cpu {

Fft::Fft(const Shape& shape)
    : shape_(shape),
      scale_(1.0f / std::sqrt(static_cast<float>(numel(shape)))),
      threads_(std::max(1u, std::thread::hardware_concurrency())) {}

void Fft::forward(std::span<const Complex> in, std::span<Complex> out) const {
  exec(in, out, true);
}

void Fft::inverse(std::span<const Complex> in, std::span<Complex> out) const {
  exec(in, out, false);
}

void Fft::exec(std::span<const Complex> in, std::span<Complex> out, bool forward) const {
  pocketfft::stride_t stride(shape_.size());
  std::ptrdiff_t step = sizeof(Complex);
  for (std::size_t a = shape_.size(); a-- > 0;) {
    stride[a] = step;
    step *= static_cast<std::ptrdiff_t>(shape_[a]);
  }
  pocketfft::shape_t axes(shape_.size());
  std::iota(axes.begin(), axes.end(), std::size_t{0});
  pocketfft::c2c(shape_, stride, stride, axes, forward, in.data(), out.data(), scale_, threads_);
}

}  // namespace qpcu::cpu
