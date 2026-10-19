#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

#include "qoraphasecu/operators.hpp"
#include "synthetic.hpp"

namespace qpcu {
namespace {

float max_diff(std::span<const Complex> a, std::span<const Complex> b) {
  float m = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

class Operators : public ::testing::TestWithParam<Shape> {};

TEST_P(Operators, ModulusProjectionIsIdempotent) {
  const auto s = test::triangle(GetParam(), 8);
  const Problem p = make_problem(s.intensity, s.shape, s.support);
  auto x = test::random_field(numel(s.shape), 1);
  cpu::modulus_project(p, x);
  const auto once = x;
  const float err = cpu::modulus_project(p, x);
  EXPECT_LT(err, 1e-5f);
  EXPECT_LT(max_diff(x, once), 1e-5f);
}

TEST_P(Operators, MaskedPixelsKeepTheirComputedMagnitude) {
  const auto s = test::triangle(GetParam(), 8);
  std::vector<std::uint8_t> mask(numel(s.shape), 0);
  mask[0] = 1;
  mask[numel(s.shape) / 2] = 1;
  const Problem p = make_problem(s.intensity, s.shape, s.support, mask);
  const Problem unmasked = make_problem(s.intensity, s.shape, s.support);
  auto x = test::random_field(numel(s.shape), 2);
  auto y = x;
  cpu::modulus_project(p, x);
  cpu::modulus_project(unmasked, y);
  EXPECT_GT(max_diff(x, y), 1e-4f);  // the masked pixels were not forced to the data
  EXPECT_LT(cpu::modulus_project(p, x), 1e-5f);
}

TEST_P(Operators, ErrorIsZeroAtTheTrueObject) {
  const auto s = test::triangle(GetParam(), 8);
  const Problem p = make_problem(s.intensity, s.shape, s.support);
  auto x = s.object;
  EXPECT_LT(cpu::modulus_project(p, x), 1e-5f);
  EXPECT_LT(max_diff(x, s.object), 1e-4f);
}

TEST_P(Operators, SupportProjectionIsIdempotentAndZeroOutside) {
  const auto s = test::triangle(GetParam(), 8);
  auto x = test::random_field(numel(s.shape), 3);
  cpu::support_project(s.support, x);
  const auto once = x;
  cpu::support_project(s.support, x);
  EXPECT_EQ(x, once);
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!s.support[i]) {
      EXPECT_EQ(x[i], Complex(0.0f));
    }
  }
}

TEST_P(Operators, BlurPreservesTheSum) {
  const Shape shape = GetParam();
  std::vector<float> image(numel(shape), 0.0f);
  image[numel(shape) / 2 + 3] = 1.0f;
  const auto blurred = cpu::gaussian_blur(image, shape, 1.5f);
  EXPECT_NEAR(std::accumulate(blurred.begin(), blurred.end(), 0.0), 1.0, 1e-5);
  EXPECT_LT(*std::ranges::max_element(blurred), 0.5f);
}

INSTANTIATE_TEST_SUITE_P(Shapes, Operators,
                         ::testing::Values(Shape{32, 32}, Shape{17, 24}, Shape{16, 16, 16}));

}  // namespace
}  // namespace qpcu
