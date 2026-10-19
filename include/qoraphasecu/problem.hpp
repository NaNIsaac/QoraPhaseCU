#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qpcu {

using Complex = std::complex<float>;

// Row-major extents of a 2D or 3D array, slowest axis first.
using Shape = std::vector<std::size_t>;

std::size_t numel(const Shape& shape);

// Fixed reconstruction inputs. Every array is center-origin: index n / 2 along
// each axis is q = 0 (detector) or r = 0 (object), as in QoraPhase.
struct Problem {
  Shape shape;
  std::vector<float> amplitude;       // sqrt(clip(intensity, 0))
  std::vector<std::uint8_t> mask;     // 1 marks an unusable pixel, left unconstrained
  std::vector<std::uint8_t> support;  // 1 is inside the object
};

// Validates the inputs and derives the amplitude. An empty `support` is replaced
// by autocorrelation_support(intensity, shape, support_threshold); an empty
// `mask` means every pixel is usable.
Problem make_problem(std::span<const float> intensity, Shape shape,
                     std::span<const std::uint8_t> support = {},
                     std::span<const std::uint8_t> mask = {}, float support_threshold = 0.1f);

// |IFFT(intensity)| thresholded at `threshold` times its maximum. The
// autocorrelation is about twice the object size, so this is a loose start that
// shrinkwrap has to tighten.
std::vector<std::uint8_t> autocorrelation_support(std::span<const float> intensity,
                                                  const Shape& shape, float threshold = 0.1f);

// Keeps the part of `support` below its centre of mass along `axis`, which
// breaks the r -> -r symmetry that lets the twin fit as well as the object.
std::vector<std::uint8_t> half_support(std::span<const std::uint8_t> support, const Shape& shape,
                                       int axis = 0);

}  // namespace qpcu
