#pragma once

#include "qoraphasecu/problem.hpp"

// Milliseconds for one forward plus one inverse in-place cuFFT C2C of `shape`,
// fp32 or (`half`) fp16: the floor under any variant that keeps cuFFT.
double fft_pair_ms(const qpcu::Shape& shape, int iters, bool half = false);
