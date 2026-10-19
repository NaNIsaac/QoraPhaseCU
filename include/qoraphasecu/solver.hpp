#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "qoraphasecu/problem.hpp"
#include "qoraphasecu/recipe.hpp"

namespace qpcu {

enum class Backend { cpu, cuda };

// Steps of the CUDA optimization ladder, kept selectable so they can be
// benchmarked against each other (docs/OPTIMIZATION.md). Ignored by the CPU
// backend.
enum class Variant {
  naive,    // one kernel per operation, error summed on the host every iteration
  fused,    // fused element-wise kernels, error partials still summed on the host
  graph,    // on-device error reduction, iterations replayed from a CUDA graph
  batched,  // graph, with every start in one batched cuFFT plan
};

std::string_view to_string(Backend backend);
std::string_view to_string(Variant variant);
Backend parse_backend(std::string_view name);
Variant parse_variant(std::string_view name);

// Called every `sync_every` iterations with the iteration count so far and the
// latest Fourier error of each start.
using Progress = std::function<void(int iteration, std::span<const float> errors)>;

struct Options {
  Backend backend = Backend::cpu;
  Variant variant = Variant::batched;
  int starts = 1;
  std::uint64_t seed = 0;
  // Iterations per CUDA graph launch, and the progress interval.
  int sync_every = 10;
  Progress progress{};
};

// One reconstructed object, center-origin like the inputs. Defined only up to a
// global phase, a translation and a conjugate flip.
struct Reconstruction {
  std::vector<Complex> object;
  std::vector<std::uint8_t> support;
  std::vector<float> errors;  // Fourier error per iteration

  float error() const;  // final Fourier error, NaN before any iteration
};

// Holds the iteration state of every start. Construction draws the random
// starts; run() may be called repeatedly and continues from the current state.
class Solver {
 public:
  virtual ~Solver() = default;
  virtual void run(const Recipe& recipe) = 0;
  virtual std::vector<Reconstruction> results() const = 0;
};

std::unique_ptr<Solver> make_solver(const Problem& problem, const Options& options = {});

std::vector<Reconstruction> reconstruct(const Problem& problem,
                                        const Recipe& recipe = default_recipe(),
                                        const Options& options = {});

// Index of the run with the lowest final error.
std::size_t best_index(std::span<const Reconstruction> runs);

bool cuda_available();

struct DeviceInfo {
  std::string name;
  int sm_major = 0;
  int sm_minor = 0;
  std::size_t memory_bytes = 0;
  std::size_t l2_bytes = 0;
  double peak_bandwidth_gbs = 0.0;  // 2 * memory clock * bus width
};

// Throws std::runtime_error when no CUDA device is present or the library was
// built without CUDA.
DeviceInfo cuda_device_info();

}  // namespace qpcu
