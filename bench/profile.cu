// Runs HIO iterations inside the NVTX range "iterations", so that
// `ncu --nvtx --nvtx-include "iterations/"` profiles one run of the iterations
// and none of the setup, plan creation or warm-up.
//
//   qpcu_profile <size> <variant> [iters] [starts] [batch]

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <string>

#include "common.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: qpcu_profile <size> <variant> [iters] [starts] [batch]\n");
    return 2;
  }
  const auto problem = qpcu::bench::bench_problem(qpcu::bench::parse_shape(argv[1]));
  const int iters = argc > 3 ? std::stoi(argv[3]) : 10;
  const qpcu::Options options{.backend = qpcu::Backend::cuda,
                              .variant = qpcu::parse_variant(argv[2]),
                              .starts = argc > 4 ? std::stoi(argv[4]) : 1,
                              .batch = argc > 5 ? std::stoi(argv[5]) : 0,
                              .sync_every = iters};
  auto solver = qpcu::make_solver(problem, options);
  // Same length as the profiled run, so its CUDA graph is captured here.
  solver->run(qpcu::recipe(qpcu::HIO{iters}));
  cudaDeviceSynchronize();
  nvtxRangePushA("iterations");
  solver->run(qpcu::recipe(qpcu::HIO{iters}));
  cudaDeviceSynchronize();
  nvtxRangePop();
  return 0;
}
