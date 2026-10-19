#pragma once

#include <cstddef>
#include <span>

#include "qoraphasecu/problem.hpp"

namespace qpcu::cpu {

// Unitary (norm="ortho") n-dimensional C2C transform on FFT-order data.
class Fft {
 public:
  explicit Fft(const Shape& shape);

  void forward(std::span<const Complex> in, std::span<Complex> out) const;
  void inverse(std::span<const Complex> in, std::span<Complex> out) const;

 private:
  void exec(std::span<const Complex> in, std::span<Complex> out, bool forward) const;

  Shape shape_;
  float scale_;
  std::size_t threads_;
};

}  // namespace qpcu::cpu
