#include <gtest/gtest.h>

#include <string>

#include "qoraphasecu/metrics.hpp"
#include "qoraphasecu/operators.hpp"
#include "qoraphasecu/solver.hpp"
#include "synthetic.hpp"

namespace qpcu {
namespace {

#define REQUIRE_CUDA() \
  if (!cuda_available()) GTEST_SKIP() << "no CUDA device or built without CUDA"

float relative_diff(std::span<const Complex> a, std::span<const Complex> b) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += std::norm(a[i] - b[i]);
    den += std::norm(b[i]);
  }
  return static_cast<float>(std::sqrt(num / den));
}

TEST(Cuda, ModulusProjectionMatchesCpu) {
  REQUIRE_CUDA();
  const auto s = test::triangle({48, 40}, 12);
  std::vector<std::uint8_t> mask(numel(s.shape), 0);
  mask[5] = 1;
  const Problem p = make_problem(s.intensity, s.shape, s.support, mask);
  auto a = test::random_field(numel(s.shape), 5);
  auto b = a;
  const float ea = cpu::modulus_project(p, a);
  const float eb = cuda::modulus_project(p, b);
  EXPECT_NEAR(ea, eb, 1e-5f * ea);
  EXPECT_LT(relative_diff(b, a), 1e-5f);
  EXPECT_LT(cuda::modulus_project(p, b), 1e-5f);
}

TEST(Cuda, SupportProjectionAndBlurMatchCpu) {
  REQUIRE_CUDA();
  const auto s = test::triangle({16, 20, 12}, 6);
  auto a = test::random_field(numel(s.shape), 6);
  auto b = a;
  cpu::support_project(s.support, a);
  cuda::support_project(s.support, b);
  EXPECT_EQ(a, b);
  std::vector<float> image(numel(s.shape));
  for (std::size_t i = 0; i < image.size(); ++i) image[i] = std::abs(a[i]);
  const auto ca = cpu::gaussian_blur(image, s.shape, 1.0f);
  const auto cb = cuda::gaussian_blur(image, s.shape, 1.0f);
  for (std::size_t i = 0; i < ca.size(); ++i) EXPECT_NEAR(ca[i], cb[i], 1e-5f) << i;
}

}  // namespace

void PrintTo(Variant v, std::ostream* os) { *os << to_string(v); }

namespace {

class CudaVariants : public ::testing::TestWithParam<Variant> {};

// Short runs: phase retrieval is chaotic enough that float rounding differences
// between pocketfft and cuFFT grow over hundreds of HIO iterations.
TEST_P(CudaVariants, AgreesWithCpu) {
  REQUIRE_CUDA();
  const auto s = test::triangle({64, 48}, 16);
  const Problem p = make_problem(s.intensity, s.shape, s.support);
  const Recipe r = recipe(HIO{15}, Shrinkwrap{}, RAAR{10}, detwin(HIO{5}), ER{10});
  const Options cpu_opts{.starts = 2, .seed = 3};
  Options gpu_opts = cpu_opts;
  gpu_opts.backend = Backend::cuda;
  gpu_opts.variant = GetParam();
  gpu_opts.sync_every = 4;
  const auto c = reconstruct(p, r, cpu_opts);
  const auto g = reconstruct(p, r, gpu_opts);
  ASSERT_EQ(g.size(), 2u);
  for (std::size_t k = 0; k < 2; ++k) {
    ASSERT_EQ(g[k].errors.size(), c[k].errors.size());
    for (std::size_t i = 0; i < c[k].errors.size(); ++i) {
      EXPECT_NEAR(g[k].errors[i], c[k].errors[i], 1e-3f * c[k].errors[i] + 1e-6f) << k << ":" << i;
    }
    EXPECT_LT(relative_diff(g[k].object, c[k].object), 1e-3f);
    EXPECT_EQ(g[k].support, c[k].support);
  }
}

// An odd element count takes the scalar path instead of the paired 16-byte loads.
TEST_P(CudaVariants, OddLengthAgreesWithCpu) {
  REQUIRE_CUDA();
  const auto s = test::triangle({21, 15}, 6);
  const Problem p = make_problem(s.intensity, s.shape, s.support);
  const Recipe r = recipe(HIO{10}, Shrinkwrap{}, ER{10});
  Options o{.backend = Backend::cuda, .variant = GetParam(), .starts = 2};
  const auto c = reconstruct(p, r, {.starts = 2});
  const auto g = reconstruct(p, r, o);
  for (std::size_t k = 0; k < 2; ++k) {
    EXPECT_NEAR(g[k].error(), c[k].error(), 1e-3f * c[k].error());
    EXPECT_LT(relative_diff(g[k].object, c[k].object), 1e-3f);
  }
}

TEST_P(CudaVariants, ProgressMatchesTheErrorHistory) {
  REQUIRE_CUDA();
  const auto s = test::triangle({32, 32}, 8);
  const Problem p = make_problem(s.intensity, s.shape, s.support);
  std::vector<int> seen;
  std::vector<float> reported;
  Options o{.backend = Backend::cuda, .variant = GetParam(), .starts = 3, .sync_every = 5};
  o.progress = [&](int it, std::span<const float> errors) {
    seen.push_back(it);
    reported.push_back(errors[2]);
  };
  const auto runs = reconstruct(p, recipe(HIO{7}, ER{8}), o);
  EXPECT_EQ(seen, (std::vector<int>{5, 10, 15}));
  for (std::size_t k = 0; k < seen.size(); ++k) {
    EXPECT_EQ(reported[k], runs[2].errors[static_cast<std::size_t>(seen[k] - 1)]);
  }
}

TEST_P(CudaVariants, Recovers3D) {
  REQUIRE_CUDA();
  const auto s = test::triangle({32, 32, 32}, 10);
  const Problem p = make_problem(s.intensity, s.shape, s.support);
  const auto runs = reconstruct(p, recipe(RAAR{300}, ER{100}),
                                {.backend = Backend::cuda, .variant = GetParam(), .starts = 2});
  float best = 1e9f;
  for (const auto& run : runs) best = std::min(best, aligned_error(run.object, s.object, s.shape));
  EXPECT_LT(best, 1e-2f);
}

INSTANTIATE_TEST_SUITE_P(All, CudaVariants,
                         ::testing::Values(Variant::naive, Variant::fused, Variant::graph,
                                           Variant::batched),
                         [](const auto& p) { return std::string(to_string(p.param)); });

TEST(Cuda, BatchedEqualsLoopedStarts) {
  REQUIRE_CUDA();
  const auto s = test::triangle({64, 64}, 20);
  const Problem p = make_problem(s.intensity, s.shape);
  const Recipe r =
      recipe(RAAR{40}, repeat(3, HIO{20}, Shrinkwrap{.method = Threshold::rms}), ER{20});
  const Options looped{.backend = Backend::cuda, .variant = Variant::graph, .starts = 4, .seed = 9};
  Options batched = looped;
  batched.variant = Variant::batched;
  const auto a = reconstruct(p, r, looped);
  const auto b = reconstruct(p, r, batched);
  for (std::size_t k = 0; k < 4; ++k) {
    EXPECT_LT(relative_diff(b[k].object, a[k].object), 1e-4f) << k;
    EXPECT_NEAR(b[k].error(), a[k].error(), 1e-4f * a[k].error() + 1e-7f) << k;
    EXPECT_EQ(b[k].support, a[k].support) << k;
  }
}

TEST(Cuda, DeviceInfoReportsPeakBandwidth) {
  REQUIRE_CUDA();
  const DeviceInfo info = cuda_device_info();
  EXPECT_FALSE(info.name.empty());
  EXPECT_GT(info.peak_bandwidth_gbs, 10.0);
}

}  // namespace
}  // namespace qpcu
