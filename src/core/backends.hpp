#pragma once

#include <memory>

#include "qoraphasecu/solver.hpp"

namespace qpcu::detail {

std::unique_ptr<Solver> make_cpu_solver(const Problem& problem, const Options& options);
std::unique_ptr<Solver> make_cuda_solver(const Problem& problem, const Options& options);

}  // namespace qpcu::detail
