// Simulates the 64x64 object of the QoraPhase README, reconstructs it from the
// intensity alone (no support given), and reports how close the best start is
// to the truth.
//
//   qpcu_reconstruct [cpu|cuda]

#include <cmath>
#include <cstdio>
#include <string>

#include "qoraphasecu/qoraphasecu.hpp"

int main(int argc, char** argv) {
  const std::string name = argc > 1 ? argv[1] : (qpcu::cuda_available() ? "cuda" : "cpu");
  const qpcu::Backend backend = qpcu::parse_backend(name);
  constexpr int n = 64;
  constexpr int m = 24;

  // A triangle (not centrosymmetric, so its twin does not fit the same support)
  // with a smooth amplitude dip and a phase bump.
  const qpcu::Shape shape{n, n};
  std::vector<qpcu::Complex> object(n * n);
  for (int y = 0; y < m; ++y) {
    for (int x = 0; x + y < m; ++x) {
      const float ry = static_cast<float>(y) - (m - 1) / 2.0f;
      const float rx = static_cast<float>(x) - (m - 1) / 2.0f;
      const float amp = 1.0f - 0.5f * std::exp(-(rx * rx + ry * ry) / 20.0f);
      const float phase = 1.2f * std::exp(-((rx + 4) * (rx + 4) + (ry + 4) * (ry + 4)) / 30.0f);
      object[(y + n / 2 - m / 2) * n + x + n / 2 - m / 2] = std::polar(amp, phase);
    }
  }

  const auto intensity = qpcu::simulate_intensity(object, shape);
  const auto problem = qpcu::make_problem(intensity, shape);  // autocorrelation support
  const auto runs =
      qpcu::reconstruct(problem, qpcu::default_recipe(), {.backend = backend, .starts = 8});
  const auto& best = runs[qpcu::best_index(runs)];
  std::printf("backend %s: best of %zu starts, Fourier error %.1e after %zu iterations\n",
              name.c_str(), runs.size(), best.error(), best.errors.size());
  std::printf("relative error vs. true object: %.1e\n",
              qpcu::aligned_error(best.object, object, shape));
  return 0;
}
