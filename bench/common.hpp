#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "qoraphasecu/qoraphasecu.hpp"

namespace qpcu::bench {

inline std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, sep);) out.push_back(item);
  return out;
}

inline Shape parse_shape(const std::string& s) {
  Shape shape;
  for (const auto& d : split(s, 'x')) shape.push_back(std::stoul(d));
  return shape;
}

inline std::string shape_name(const Shape& shape) {
  return shape.size() == 2 ? std::to_string(shape[0]) + "^2" : std::to_string(shape[0]) + "^3";
}

// Positive random intensity and a box support over the central half of every
// axis. Timing does not depend on the data, only on the shape and support size.
inline Problem bench_problem(const Shape& shape) {
  const std::size_t n = numel(shape);
  std::vector<float> intensity(n);
  unsigned state = 12345u;
  for (auto& v : intensity) {
    state = state * 1664525u + 1013904223u;
    v = static_cast<float>(state >> 8) / 16777216.0f;
  }
  std::vector<std::uint8_t> support(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t rest = i;
    bool inside = true;
    for (std::size_t a = shape.size(); a-- > 0;) {
      const std::size_t c = rest % shape[a];
      rest /= shape[a];
      inside = inside && c >= shape[a] / 4 && c < 3 * shape[a] / 4;
    }
    support[i] = inside;
  }
  return make_problem(intensity, shape, support);
}

}  // namespace qpcu::bench
