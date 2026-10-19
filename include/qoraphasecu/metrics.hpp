#pragma once

#include <span>

#include "qoraphasecu/problem.hpp"

namespace qpcu {

// ||a - b|| / ||b|| after bringing `a` onto `reference` by the symmetries no
// iteration can resolve: an integer translation, the conjugate flip
// conj(a(-r)), and a global phase.
float aligned_error(std::span<const Complex> a, std::span<const Complex> reference,
                    const Shape& shape);

}  // namespace qpcu
