#pragma once

#include "qoraphasecu/problem.hpp"

// Milliseconds for one forward plus one inverse in-place cuFFT C2C of `shape`:
// the floor under any variant that keeps cuFFT for the transforms.
double fft_pair_ms(const qpcu::Shape& shape, int iters);
