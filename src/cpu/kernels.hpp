#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "../core/pointwise.hpp"
#include "fft.hpp"
#include "qoraphasecu/problem.hpp"
#include "qoraphasecu/recipe.hpp"

// Reference implementations on FFT-order arrays.

namespace qpcu::cpu {

// sum of amplitude^2 over usable pixels: the error denominator.
double usable_energy(std::span<const float> amplitude, std::span<const std::uint8_t> mask);

// In place on the ortho-normalised wave; returns the Fourier error of the input.
float modulus(std::span<Complex> wave, std::span<const float> amplitude,
              std::span<const std::uint8_t> mask, double energy);

void update(std::span<Complex> x, std::span<const Complex> projected,
            std::span<const std::uint8_t> support, detail::UpdateCoefficients c);

void blur(std::span<const float> image, const Shape& shape, const Fft& fft, float sigma,
          std::span<float> out);

void shrinkwrap(std::span<const Complex> x, const Shape& shape, const Fft& fft,
                const Shrinkwrap& params, std::span<std::uint8_t> support);

// Measured amplitude with phases from detail::uniform, back-transformed and
// cut to the support.
void random_start(std::span<const float> amplitude, std::span<const std::uint8_t> support,
                  const Fft& fft, std::uint64_t seed, std::uint64_t start, std::span<Complex> x);

}  // namespace qpcu::cpu
