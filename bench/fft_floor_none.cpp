#include <stdexcept>

#include "fft_floor.hpp"

double fft_pair_ms(const qpcu::Shape&, int) { throw std::runtime_error("built without CUDA"); }
