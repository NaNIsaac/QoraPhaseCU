#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "qoraphasecu/problem.hpp"

namespace qpcu::detail {

// Shapes padded to three axes, slowest first, so 2D is {1, ny, nx}.
inline std::array<std::size_t, 3> dims3(const Shape& shape) {
  return shape.size() == 2 ? std::array<std::size_t, 3>{1, shape[0], shape[1]}
                           : std::array<std::size_t, 3>{shape[0], shape[1], shape[2]};
}

// out[i] = in[(i + offset(n)) mod n] along every axis.
template <class T>
std::vector<T> roll(std::span<const T> in, const Shape& shape, bool to_fft_order) {
  const auto d = dims3(shape);
  std::array<std::size_t, 3> off{};
  for (int a = 0; a < 3; ++a) off[a] = to_fft_order ? d[a] / 2 : d[a] - d[a] / 2;
  std::vector<T> out(in.size());
  std::size_t i = 0;
  for (std::size_t z = 0; z < d[0]; ++z) {
    const std::size_t sz = (z + off[0]) % d[0];
    for (std::size_t y = 0; y < d[1]; ++y) {
      const std::size_t sy = (y + off[1]) % d[1];
      const std::size_t row = (sz * d[1] + sy) * d[2];
      for (std::size_t x = 0; x < d[2]; ++x) out[i++] = in[row + (x + off[2]) % d[2]];
    }
  }
  return out;
}

// numpy ifftshift: center-origin to FFT order.
template <class T>
std::vector<T> to_fft_order(std::span<const T> in, const Shape& shape) {
  return roll(in, shape, true);
}

// numpy fftshift: FFT order to center-origin.
template <class T>
std::vector<T> to_centered(std::span<const T> in, const Shape& shape) {
  return roll(in, shape, false);
}

}  // namespace qpcu::detail
