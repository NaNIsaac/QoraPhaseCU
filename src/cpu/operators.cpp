#include "qoraphasecu/operators.hpp"

#include <algorithm>
#include <stdexcept>

#include "../core/layout.hpp"
#include "kernels.hpp"

namespace qpcu::cpu {

float modulus_project(const Problem& problem, std::span<Complex> obj) {
  if (obj.size() != problem.amplitude.size()) {
    throw std::invalid_argument("obj does not match the problem shape");
  }
  const auto amplitude = detail::to_fft_order<float>(problem.amplitude, problem.shape);
  const auto mask = detail::to_fft_order<std::uint8_t>(problem.mask, problem.shape);
  auto x = detail::to_fft_order<Complex>(obj, problem.shape);
  const Fft fft(problem.shape);
  fft.forward(x, x);
  const float err = modulus(x, amplitude, mask, usable_energy(amplitude, mask));
  fft.inverse(x, x);
  std::ranges::copy(detail::to_centered<Complex>(x, problem.shape), obj.begin());
  return err;
}

void support_project(std::span<const std::uint8_t> support, std::span<Complex> obj) {
  if (obj.size() != support.size()) throw std::invalid_argument("support does not match obj");
  for (std::size_t i = 0; i < obj.size(); ++i) {
    if (!support[i]) obj[i] = 0.0f;
  }
}

std::vector<float> gaussian_blur(std::span<const float> image, const Shape& shape, float sigma) {
  if (image.size() != numel(shape)) throw std::invalid_argument("image does not match shape");
  // A circular convolution commutes with the fftshift roll, so no reordering.
  std::vector<float> out(image.size());
  blur(image, shape, Fft(shape), sigma, out);
  return out;
}

}  // namespace qpcu::cpu
