#include "qoraphasecu/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "../cpu/fft.hpp"
#include "layout.hpp"

namespace qpcu {

namespace {

// Rolls b by the integer shift maximising |sum ref conj(b(r - s))|, found as the
// peak of the FFT cross-correlation, then fits the global phase.
float translated_error(std::vector<Complex> b, std::span<const Complex> ref, const Shape& shape,
                       const cpu::Fft& fft) {
  const std::size_t n = b.size();
  std::vector<Complex> fr(ref.begin(), ref.end());
  std::vector<Complex> fb = b;
  fft.forward(fr, fr);
  fft.forward(fb, fb);
  for (std::size_t i = 0; i < n; ++i) fr[i] *= std::conj(fb[i]);
  fft.inverse(fr, fr);
  const auto peak = static_cast<std::size_t>(
      std::ranges::max_element(fr, {}, [](Complex c) { return std::norm(c); }) - fr.begin());

  const auto d = detail::dims3(shape);
  const std::size_t s0 = peak / (d[1] * d[2]);
  const std::size_t s1 = (peak / d[2]) % d[1];
  const std::size_t s2 = peak % d[2];
  std::vector<Complex> shifted(n);
  std::size_t i = 0;
  for (std::size_t z = 0; z < d[0]; ++z) {
    for (std::size_t y = 0; y < d[1]; ++y) {
      for (std::size_t x = 0; x < d[2]; ++x) {
        const std::size_t src = (((z + d[0] - s0) % d[0]) * d[1] + (y + d[1] - s1) % d[1]) * d[2] +
                                (x + d[2] - s2) % d[2];
        shifted[i++] = b[src];
      }
    }
  }

  std::complex<double> overlap = 0.0;
  double ref_norm = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    overlap += std::complex<double>(ref[k]) * std::conj(std::complex<double>(shifted[k]));
    ref_norm += std::norm(std::complex<double>(ref[k]));
  }
  const std::complex<double> rotation =
      std::abs(overlap) > 0.0 ? overlap / std::abs(overlap) : std::complex<double>(1.0);
  double diff = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    diff += std::norm(rotation * std::complex<double>(shifted[k]) - std::complex<double>(ref[k]));
  }
  return static_cast<float>(
      std::sqrt(diff / std::max(ref_norm, std::numeric_limits<double>::min())));
}

}  // namespace

float aligned_error(std::span<const Complex> a, std::span<const Complex> reference,
                    const Shape& shape) {
  if (a.size() != numel(shape) || reference.size() != a.size()) {
    throw std::invalid_argument("arrays do not match shape");
  }
  const cpu::Fft fft(shape);
  std::vector<Complex> direct(a.begin(), a.end());

  // conj(a(-r)); any constant offset of the flip is absorbed by the translation.
  const auto d = detail::dims3(shape);
  std::vector<Complex> twin(a.size());
  std::size_t i = 0;
  for (std::size_t z = 0; z < d[0]; ++z) {
    for (std::size_t y = 0; y < d[1]; ++y) {
      for (std::size_t x = 0; x < d[2]; ++x) {
        const std::size_t src =
            (((d[0] - z) % d[0]) * d[1] + (d[1] - y) % d[1]) * d[2] + (d[2] - x) % d[2];
        twin[i++] = std::conj(a[src]);
      }
    }
  }
  return std::min(translated_error(std::move(direct), reference, shape, fft),
                  translated_error(std::move(twin), reference, shape, fft));
}

}  // namespace qpcu
