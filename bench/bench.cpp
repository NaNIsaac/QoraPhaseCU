// Times HIO iterations for each variant and size and prints Markdown tables.
//
//   qpcu_bench [--sizes 256x256,1024x1024,128x128x128] [--iters 200] [--starts 8]
//              [--variants cpu,naive,fused,graph,batched] [--cpu-max 1048576]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "fft_floor.hpp"
#include "qoraphasecu/qoraphasecu.hpp"

namespace {

using qpcu::Backend;
using qpcu::Shape;
using qpcu::Variant;

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, sep);) out.push_back(item);
  return out;
}

Shape parse_shape(const std::string& s) {
  Shape shape;
  for (const auto& d : split(s, 'x')) shape.push_back(std::stoul(d));
  return shape;
}

std::string shape_name(const Shape& shape) {
  return shape.size() == 2 ? std::to_string(shape[0]) + "^2" : std::to_string(shape[0]) + "^3";
}

// Positive random intensity and a box support over the central half of every
// axis. Timing does not depend on the data, only on the shape and support size.
qpcu::Problem bench_problem(const Shape& shape) {
  const std::size_t n = qpcu::numel(shape);
  std::vector<float> intensity(n);
  unsigned state = 12345u;
  for (auto& v : intensity) {
    state = state * 1664525u + 1013904223u;
    v = static_cast<float>(state >> 8) / 16777216.0f;
  }
  std::vector<std::uint8_t> support(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t rest = i;
    bool inside = true;
    for (std::size_t a = shape.size(); a-- > 0;) {
      const std::size_t c = rest % shape[a];
      rest /= shape[a];
      inside = inside && c >= shape[a] / 4 && c < 3 * shape[a] / 4;
    }
    support[i] = inside;
  }
  return qpcu::make_problem(intensity, shape, support);
}

struct Timing {
  double ms_per_iter_per_start;
};

Timing time_hio(const qpcu::Problem& problem, const qpcu::Options& options, int iters) {
  auto solver = qpcu::make_solver(problem, options);
  // Same length as the timed run, so CUDA graphs are captured here and reused.
  solver->run(qpcu::recipe(qpcu::HIO{iters}));
  const auto t0 = std::chrono::steady_clock::now();
  solver->run(qpcu::recipe(qpcu::HIO{iters}));
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return {ms / iters / options.starts};
}

// Bytes per element per iteration, counting each FFT as one read and one write
// of the array (a lower bound for cuFFT, which may take several passes) plus
// every load and store the element-wise kernels issue.
double modeled_bytes(Variant v) {
  constexpr double fft_pair = 2 * 16;
  switch (v) {
    case Variant::naive:
      // scale 16, error 13, modulus 21, scale 16, update 25
      return fft_pair + 16 + 13 + 21 + 16 + 25;
    default:
      // modulus 8 + 4 + 8, update 1 + 8 + 8 + 8
      return fft_pair + 20 + 25;
  }
}

constexpr double kFftPairBytes = 32.0;

double gbs(double bytes_per_elem, std::size_t n, double ms) {
  return bytes_per_elem * static_cast<double>(n) / (ms * 1e-3) / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> sizes{"256x256", "1024x1024", "2048x2048", "128x128x128", "256x256x256"};
  std::vector<std::string> variants{"cpu", "naive", "fused", "graph", "batched"};
  int iters = 200;
  int starts = 8;
  std::size_t cpu_max = std::size_t{1} << 21;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i];
    const std::string value = argv[i + 1];
    if (key == "--sizes") {
      sizes = split(value, ',');
    } else if (key == "--variants") {
      variants = split(value, ',');
    } else if (key == "--iters") {
      iters = std::stoi(value);
    } else if (key == "--starts") {
      starts = std::stoi(value);
    } else if (key == "--cpu-max") {
      cpu_max = std::stoul(value);
    } else {
      std::fprintf(stderr, "unknown option %s\n", key.c_str());
      return 2;
    }
  }

  std::optional<qpcu::DeviceInfo> device;
  if (qpcu::cuda_available()) device = qpcu::cuda_device_info();
  if (device) {
    std::printf("GPU: %s (sm_%d%d, %.1f GiB, L2 %.0f MiB), peak DRAM bandwidth %.0f GB/s\n\n",
                device->name.c_str(), device->sm_major, device->sm_minor,
                static_cast<double>(device->memory_bytes) / (1 << 30),
                static_cast<double>(device->l2_bytes) / (1 << 20), device->peak_bandwidth_gbs);
  }

  std::printf("Single start, HIO, %d timed iterations. ms per iteration (modeled GB/s).\n\n",
              iters);
  std::printf("| size | cuFFT pair |");
  for (const auto& v : variants) {
    if (v != "batched") std::printf(" %s |", v.c_str());
  }
  std::printf("\n|---|---:|");
  for (const auto& v : variants) {
    if (v != "batched") std::printf("---:|");
  }
  std::printf("\n");

  for (const auto& size : sizes) {
    const Shape shape = parse_shape(size);
    const std::size_t n = qpcu::numel(shape);
    const auto problem = bench_problem(shape);
    std::printf("| %s |", shape_name(shape).c_str());
    if (device) {
      const double ms = fft_pair_ms(shape, iters);
      std::printf(" %.3f (%.0f) |", ms, gbs(kFftPairBytes, n, ms));
    } else {
      std::printf(" - |");
    }
    for (const auto& name : variants) {
      if (name == "batched") continue;
      if (name == "cpu") {
        if (n > cpu_max) {
          std::printf(" - |");
          continue;
        }
        const int cpu_iters = std::max(5, iters / 20);
        const auto t = time_hio(problem, {.backend = Backend::cpu}, cpu_iters);
        std::printf(" %.2f |", t.ms_per_iter_per_start);
      } else if (device) {
        const Variant v = qpcu::parse_variant(name);
        const auto t = time_hio(problem, {.backend = Backend::cuda, .variant = v}, iters);
        std::printf(" %.3f (%.0f) |", t.ms_per_iter_per_start,
                    gbs(modeled_bytes(v), n, t.ms_per_iter_per_start));
      } else {
        std::printf(" - |");
      }
    }
    std::printf("\n");
    std::fflush(stdout);
  }

  if (!device) return 0;

  std::printf("\n%d starts, HIO, %d timed iterations. ms per iteration per start.\n\n", starts,
              iters);
  std::printf("| size | graph, looped starts | batched | speedup |\n|---|---:|---:|---:|\n");
  for (const auto& size : sizes) {
    const Shape shape = parse_shape(size);
    const std::size_t n = qpcu::numel(shape);
    // x and support per start, one batched work buffer, and cuFFT scratch of
    // about the same size: skip what would not fit.
    const double need = static_cast<double>(n) * (starts * 9.0 + starts * 8.0 * 2);
    if (need > 0.8 * static_cast<double>(device->memory_bytes)) {
      std::printf("| %s | - | - | does not fit |\n", shape_name(shape).c_str());
      continue;
    }
    const auto problem = bench_problem(shape);
    const auto looped = time_hio(
        problem, {.backend = Backend::cuda, .variant = Variant::graph, .starts = starts}, iters);
    const auto batched = time_hio(
        problem, {.backend = Backend::cuda, .variant = Variant::batched, .starts = starts}, iters);
    std::printf("| %s | %.3f | %.3f | %.2fx |\n", shape_name(shape).c_str(),
                looped.ms_per_iter_per_start, batched.ms_per_iter_per_start,
                looped.ms_per_iter_per_start / batched.ms_per_iter_per_start);
    std::fflush(stdout);
  }

  std::printf("\nGraph variant at %s with a progress callback every N iterations.\n\n",
              sizes.front().c_str());
  std::printf("| N | ms per iteration |\n|---:|---:|\n");
  const auto problem = bench_problem(parse_shape(sizes.front()));
  for (int every : {1, 10, 100}) {
    qpcu::Options o{.backend = Backend::cuda, .variant = Variant::graph, .sync_every = every};
    o.progress = [](int, std::span<const float>) {};
    std::printf("| %d | %.4f |\n", every, time_hio(problem, o, iters).ms_per_iter_per_start);
  }
  return 0;
}
