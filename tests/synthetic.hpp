#pragma once

#include <cmath>
#include <vector>

#include "core/layout.hpp"
#include "qoraphasecu/problem.hpp"

namespace qpcu::test {

struct Synthetic {
  Shape shape;
  std::vector<Complex> object;
  std::vector<std::uint8_t> support;
  std::vector<float> intensity;
};

// A right triangle (not centrosymmetric, so the twin does not also fit the
// support) of side m, with an amplitude dip and a phase bump, in an n-wide box.
// 3D stacks the triangle over m slices with a phase that varies along z.
inline Synthetic triangle(Shape shape, std::size_t m) {
  const auto d = detail::dims3(shape);
  const bool is3d = shape.size() == 3;
  Synthetic s{
      shape, std::vector<Complex>(numel(shape)), std::vector<std::uint8_t>(numel(shape)), {}};
  const auto c = static_cast<double>(m - 1) / 2.0;
  std::size_t i = 0;
  for (std::size_t z = 0; z < d[0]; ++z) {
    for (std::size_t y = 0; y < d[1]; ++y) {
      for (std::size_t x = 0; x < d[2]; ++x, ++i) {
        const auto lz = static_cast<double>(z) - static_cast<double>(d[0] / 2 - m / 2);
        const auto ly = static_cast<double>(y) - static_cast<double>(d[1] / 2 - m / 2);
        const auto lx = static_cast<double>(x) - static_cast<double>(d[2] / 2 - m / 2);
        const auto md = static_cast<double>(m);
        const bool in_box =
            ly >= 0 && ly < md && lx >= 0 && lx < md && (!is3d || (lz >= 0 && lz < md));
        if (!in_box || ly + lx >= md) continue;
        const double ry = ly - c;
        const double rx = lx - c;
        const double amp = 1.0 - 0.5 * std::exp(-(rx * rx + ry * ry) / 20.0);
        const double phase = 1.2 * std::exp(-((rx + 4) * (rx + 4) + (ry + 4) * (ry + 4)) / 30.0) +
                             (is3d ? 0.3 * (lz - c) / md : 0.0);
        s.object[i] = std::polar(static_cast<float>(amp), static_cast<float>(phase));
        s.support[i] = 1;
      }
    }
  }
  s.intensity = simulate_intensity(s.object, shape);
  return s;
}

// The 3D triangle with the corner x + y + z >= 1.5 m cut off. The plain 3D
// triangle is separable, f(x, y) g(z), and the z-transform of g has all its zeros
// on the unit circle, where |G| is flat to first order in the zero's radius: there
// objects 2e-4 apart fit the data to 3e-7 (docs/OPTIMIZATION.md, step 7).
inline Synthetic corner(std::size_t n, std::size_t m) {
  Synthetic s = triangle({n, n, n}, m);
  const auto lo = static_cast<double>(n / 2 - m / 2);
  std::size_t i = 0;
  for (std::size_t z = 0; z < n; ++z) {
    for (std::size_t y = 0; y < n; ++y) {
      for (std::size_t x = 0; x < n; ++x, ++i) {
        if (static_cast<double>(z + y + x) - 3 * lo >= 1.5 * static_cast<double>(m)) {
          s.object[i] = 0;
          s.support[i] = 0;
        }
      }
    }
  }
  s.intensity = simulate_intensity(s.object, s.shape);
  return s;
}

inline std::vector<Complex> random_field(std::size_t n, unsigned seed) {
  std::vector<Complex> v(n);
  unsigned state = seed * 2654435761u + 1u;
  auto next = [&] {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
  };
  for (auto& z : v) z = {next(), next()};
  return v;
}

}  // namespace qpcu::test
