#include "qoraphasecu/problem.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>

#include "../cpu/fft.hpp"
#include "layout.hpp"

namespace qpcu {

std::size_t numel(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>{});
}

namespace {

void check_shape(const Shape& shape) {
  if (shape.size() != 2 && shape.size() != 3) {
    throw std::invalid_argument("shape must be 2D or 3D, got ndim=" + std::to_string(shape.size()));
  }
  if (std::ranges::any_of(shape, [](std::size_t n) { return n == 0; })) {
    throw std::invalid_argument("shape has a zero-length axis");
  }
}

template <class T>
void check_size(std::span<const T> array, std::size_t n, const char* name) {
  if (!array.empty() && array.size() != n) {
    throw std::invalid_argument(std::string(name) + " has " + std::to_string(array.size()) +
                                " elements, shape has " + std::to_string(n));
  }
}

}  // namespace

Problem make_problem(std::span<const float> intensity, Shape shape,
                     std::span<const std::uint8_t> support, std::span<const std::uint8_t> mask,
                     float support_threshold) {
  check_shape(shape);
  const std::size_t n = numel(shape);
  if (intensity.size() != n) {
    throw std::invalid_argument("intensity has " + std::to_string(intensity.size()) +
                                " elements, shape has " + std::to_string(n));
  }
  check_size(support, n, "support");
  check_size(mask, n, "mask");
  if (!std::ranges::all_of(intensity, [](float v) { return std::isfinite(v); })) {
    throw std::invalid_argument("intensity contains NaN or Inf");
  }
  if (!mask.empty() && std::ranges::all_of(mask, [](std::uint8_t m) { return m != 0; })) {
    throw std::invalid_argument("mask marks every pixel as bad; nothing left to fit");
  }

  Problem p;
  p.amplitude.resize(n);
  std::ranges::transform(intensity, p.amplitude.begin(),
                         [](float v) { return std::sqrt(std::max(v, 0.0f)); });
  p.mask = mask.empty() ? std::vector<std::uint8_t>(n, 0)
                        : std::vector<std::uint8_t>(mask.begin(), mask.end());
  p.support = support.empty() ? autocorrelation_support(intensity, shape, support_threshold)
                              : std::vector<std::uint8_t>(support.begin(), support.end());
  for (auto& s : p.support) s = s != 0;
  for (auto& m : p.mask) m = m != 0;
  p.shape = std::move(shape);
  return p;
}

std::vector<std::uint8_t> autocorrelation_support(std::span<const float> intensity,
                                                  const Shape& shape, float threshold) {
  check_shape(shape);
  const std::size_t n = numel(shape);
  std::vector<Complex> buf(n);
  const auto ordered = detail::to_fft_order(intensity, shape);
  std::ranges::copy(ordered, buf.begin());
  cpu::Fft(shape).inverse(buf, buf);
  std::vector<float> ac(n);
  std::ranges::transform(buf, ac.begin(), [](Complex c) { return std::abs(c); });
  const auto centered = detail::to_centered<float>(ac, shape);
  const float level = threshold * std::ranges::max(centered);
  std::vector<std::uint8_t> support(n);
  std::ranges::transform(centered, support.begin(),
                         [level](float v) { return static_cast<std::uint8_t>(v > level); });
  return support;
}

std::vector<float> simulate_intensity(std::span<const Complex> object, const Shape& shape) {
  check_shape(shape);
  if (object.size() != numel(shape)) throw std::invalid_argument("object does not match shape");
  auto wave = detail::to_fft_order<Complex>(object, shape);
  cpu::Fft(shape).forward(wave, wave);
  const auto centered = detail::to_centered<Complex>(wave, shape);
  std::vector<float> intensity(centered.size());
  std::ranges::transform(centered, intensity.begin(), [](Complex c) { return std::norm(c); });
  return intensity;
}

std::vector<std::uint8_t> half_support(std::span<const std::uint8_t> support, const Shape& shape,
                                       int axis) {
  check_shape(shape);
  if (axis < 0 || axis >= static_cast<int>(shape.size())) {
    throw std::invalid_argument("detwin axis out of range");
  }
  const auto ax = static_cast<std::size_t>(axis);
  std::size_t inner = 1;
  for (std::size_t a = ax + 1; a < shape.size(); ++a) inner *= shape[a];
  const std::size_t len = shape[ax];
  auto coordinate = [&](std::size_t i) { return (i / inner) % len; };

  double moment = 0.0;
  double weight = 0.0;
  for (std::size_t i = 0; i < support.size(); ++i) {
    if (support[i]) {
      moment += static_cast<double>(coordinate(i));
      weight += 1.0;
    }
  }
  const double centre = moment / std::max(weight, 1.0);
  std::vector<std::uint8_t> out(support.size());
  for (std::size_t i = 0; i < support.size(); ++i) {
    out[i] = support[i] && static_cast<double>(coordinate(i)) < centre;
  }
  return out;
}

}  // namespace qpcu
