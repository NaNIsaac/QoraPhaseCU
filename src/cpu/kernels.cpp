#include "kernels.hpp"

#include <algorithm>
#include <cmath>

#include "../core/layout.hpp"

namespace qpcu::cpu {

using detail::kEps;

double usable_energy(std::span<const float> amplitude, std::span<const std::uint8_t> mask) {
  double sum = 0.0;
  for (std::size_t i = 0; i < amplitude.size(); ++i) {
    if (!mask[i]) sum += static_cast<double>(amplitude[i]) * amplitude[i];
  }
  return sum;
}

float modulus(std::span<Complex> wave, std::span<const float> amplitude,
              std::span<const std::uint8_t> mask, double energy) {
  double num = 0.0;
  for (std::size_t i = 0; i < wave.size(); ++i) {
    const float mag = std::abs(wave[i]);
    float target = mag;
    if (!mask[i]) {
      const float d = mag - amplitude[i];
      num += static_cast<double>(d) * d;
      target = amplitude[i];
    }
    wave[i] *= target / (mag + kEps);
  }
  return static_cast<float>(std::sqrt(num / (energy + kEps)));
}

void update(std::span<Complex> x, std::span<const Complex> projected,
            std::span<const std::uint8_t> support, detail::UpdateCoefficients c) {
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = support[i] ? projected[i] : c.a * x[i] + c.b * projected[i];
  }
}

void blur(std::span<const float> image, const Shape& shape, const Fft& fft, float sigma,
          std::span<float> out) {
  const auto d = detail::dims3(shape);
  std::vector<Complex> buf(image.begin(), image.end());
  fft.forward(buf, buf);
  const float k = -2.0f * (detail::kTwoPi / 2) * (detail::kTwoPi / 2) * sigma * sigma;
  std::size_t i = 0;
  for (std::size_t z = 0; z < d[0]; ++z) {
    const float fz =
        detail::fftfreq(static_cast<std::uint32_t>(z), static_cast<std::uint32_t>(d[0]));
    for (std::size_t y = 0; y < d[1]; ++y) {
      const float fy =
          detail::fftfreq(static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(d[1]));
      for (std::size_t x = 0; x < d[2]; ++x, ++i) {
        const float fx =
            detail::fftfreq(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(d[2]));
        buf[i] *= std::exp(k * (fz * fz + fy * fy + fx * fx));
      }
    }
  }
  fft.inverse(buf, buf);
  std::ranges::transform(buf, out.begin(), [](Complex c) { return c.real(); });
}

void shrinkwrap(std::span<const Complex> x, const Shape& shape, const Fft& fft,
                const Shrinkwrap& params, std::span<std::uint8_t> support) {
  std::vector<float> magnitude(x.size());
  std::ranges::transform(x, magnitude.begin(), [](Complex c) { return std::abs(c); });
  blur(magnitude, shape, fft, params.sigma, magnitude);

  float level = 0.0f;
  if (params.method == Threshold::max) {
    level = std::ranges::max(magnitude);
  } else {
    double sum = 0.0;
    double count = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (support[i]) {
        sum += static_cast<double>(magnitude[i]) * magnitude[i];
        count += 1.0;
      }
    }
    level = static_cast<float>(std::sqrt(sum / std::max(count, 1.0)));
  }
  const float cut = params.threshold * level;
  for (std::size_t i = 0; i < x.size(); ++i) support[i] = magnitude[i] > cut;
}

void random_start(std::span<const float> amplitude, std::span<const std::uint8_t> support,
                  const Fft& fft, std::uint64_t seed, std::uint64_t start, std::span<Complex> x) {
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float phase = detail::kTwoPi * detail::uniform(seed, start, i);
    x[i] = amplitude[i] * Complex(std::cos(phase), std::sin(phase));
  }
  fft.inverse(x, x);
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!support[i]) x[i] = 0.0f;
  }
}

}  // namespace qpcu::cpu
