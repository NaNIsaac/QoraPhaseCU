#include "qoraphasecu/solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "backends.hpp"

namespace qpcu {

std::string_view to_string(Backend backend) {
  switch (backend) {
    case Backend::cpu:
      return "cpu";
    case Backend::cuda:
      return "cuda";
  }
  return "?";
}

std::string_view to_string(Variant variant) {
  switch (variant) {
    case Variant::naive:
      return "naive";
    case Variant::fused:
      return "fused";
    case Variant::graph:
      return "graph";
    case Variant::batched:
      return "batched";
  }
  return "?";
}

Backend parse_backend(std::string_view name) {
  if (name == "cpu") return Backend::cpu;
  if (name == "cuda") return Backend::cuda;
  throw std::invalid_argument("unknown backend '" + std::string(name) + "'");
}

Variant parse_variant(std::string_view name) {
  for (Variant v : {Variant::naive, Variant::fused, Variant::graph, Variant::batched}) {
    if (name == to_string(v)) return v;
  }
  throw std::invalid_argument("unknown variant '" + std::string(name) + "'");
}

float Reconstruction::error() const {
  return errors.empty() ? std::numeric_limits<float>::quiet_NaN() : errors.back();
}

std::unique_ptr<Solver> make_solver(const Problem& problem, const Options& options) {
  if (options.starts < 1) throw std::invalid_argument("starts must be >= 1");
  if (options.sync_every < 1) throw std::invalid_argument("sync_every must be >= 1");
  if (problem.shape.size() != 2 && problem.shape.size() != 3) {
    throw std::invalid_argument("problem shape must be 2D or 3D");
  }
  const std::size_t n = numel(problem.shape);
  if (problem.amplitude.size() != n || problem.mask.size() != n || problem.support.size() != n) {
    throw std::invalid_argument(
        "problem arrays do not match its shape; build it with make_problem");
  }
  switch (options.backend) {
    case Backend::cpu:
      return detail::make_cpu_solver(problem, options);
    case Backend::cuda:
      return detail::make_cuda_solver(problem, options);
  }
  throw std::invalid_argument("unknown backend");
}

std::vector<Reconstruction> reconstruct(const Problem& problem, const Recipe& recipe,
                                        const Options& options) {
  auto solver = make_solver(problem, options);
  solver->run(recipe);
  return solver->results();
}

std::size_t best_index(std::span<const Reconstruction> runs) {
  if (runs.empty()) throw std::invalid_argument("no runs");
  // NaN compares false, so a NaN run never displaces a finite one.
  std::size_t best = 0;
  for (std::size_t i = 1; i < runs.size(); ++i) {
    if (runs[i].error() < runs[best].error() || std::isnan(runs[best].error())) best = i;
  }
  return best;
}

}  // namespace qpcu
