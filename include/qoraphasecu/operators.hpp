#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "qoraphasecu/problem.hpp"

// Single projections on center-origin arrays, the same functions on both
// backends. The solvers do not go through these: they keep their state in FFT
// order on the device. These exist for tests and for composing new algorithms.

namespace qpcu::cpu {

// Replaces `obj` by P_M obj and returns the Fourier error of the input.
float modulus_project(const Problem& problem, std::span<Complex> obj);

void support_project(std::span<const std::uint8_t> support, std::span<Complex> obj);

// Gaussian blur through the FFT, transfer function exp(-2 (pi sigma f)^2).
std::vector<float> gaussian_blur(std::span<const float> image, const Shape& shape, float sigma);

}  // namespace qpcu::cpu

namespace qpcu::cuda {

float modulus_project(const Problem& problem, std::span<Complex> obj);

void support_project(std::span<const std::uint8_t> support, std::span<Complex> obj);

std::vector<float> gaussian_blur(std::span<const float> image, const Shape& shape, float sigma);

}  // namespace qpcu::cuda
