#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "core/layout.hpp"
#include "qoraphasecu/problem.hpp"
#include "synthetic.hpp"

namespace qpcu {
namespace {

TEST(Layout, OrderRoundTripsOnOddAndEvenAxes) {
  const Shape shape{5, 4, 7};
  std::vector<int> v(numel(shape));
  std::iota(v.begin(), v.end(), 0);
  const auto there = detail::to_fft_order<int>(v, shape);
  EXPECT_NE(there, v);
  EXPECT_EQ(detail::to_centered<int>(there, shape), v);
}

TEST(Layout, CenterMovesToOrigin) {
  const Shape shape{6, 5};
  std::vector<int> v(numel(shape), 0);
  v[3 * 5 + 2] = 1;  // (n0 / 2, n1 / 2)
  EXPECT_EQ(detail::to_fft_order<int>(v, shape)[0], 1);
}

TEST(MakeProblem, ClipsNegativeIntensity) {
  const std::vector<float> intensity{4.0f, -1.0f, 9.0f, 0.0f};
  const std::vector<std::uint8_t> support{1, 1, 0, 0};
  const Problem p = make_problem(intensity, {2, 2}, support);
  EXPECT_EQ(p.amplitude, (std::vector<float>{2.0f, 0.0f, 3.0f, 0.0f}));
  EXPECT_EQ(p.mask, (std::vector<std::uint8_t>(4, 0)));
}

TEST(MakeProblem, RejectsBadInputs) {
  const std::vector<float> four(4, 1.0f);
  EXPECT_THROW(make_problem(four, {4}), std::invalid_argument);
  EXPECT_THROW(make_problem(four, {2, 3}), std::invalid_argument);
  const std::vector<std::uint8_t> all_bad(4, 1);
  EXPECT_THROW(make_problem(four, {2, 2}, {}, all_bad), std::invalid_argument);
  std::vector<float> nan = four;
  nan[1] = std::nanf("");
  EXPECT_THROW(make_problem(nan, {2, 2}), std::invalid_argument);
}

TEST(AutocorrelationSupport, CoversTheObject) {
  const auto s = test::triangle({64, 64}, 16);
  const auto support = autocorrelation_support(s.intensity, s.shape, 0.1f);
  std::size_t covered = 0;
  std::size_t inside = 0;
  for (std::size_t i = 0; i < support.size(); ++i) {
    inside += s.support[i];
    covered += s.support[i] && support[i];
  }
  // The autocorrelation is centred, the object is not; most of it must still fall inside.
  EXPECT_GT(covered, inside / 2);
  EXPECT_GT(std::accumulate(support.begin(), support.end(), 0u), inside);
}

TEST(HalfSupport, KeepsTheLowerHalf) {
  const Shape shape{8, 4};
  std::vector<std::uint8_t> support(numel(shape), 0);
  for (std::size_t y = 2; y < 6; ++y) {
    for (std::size_t x = 0; x < 4; ++x) support[y * 4 + x] = 1;
  }
  const auto half = half_support(support, shape, 0);
  for (std::size_t y = 0; y < 8; ++y) {
    for (std::size_t x = 0; x < 4; ++x) {
      EXPECT_EQ(half[y * 4 + x], y == 2 || y == 3 ? 1 : 0) << y << "," << x;
    }
  }
  const auto columns = half_support(support, shape, 1);
  EXPECT_EQ(columns[2 * 4 + 1], 1);
  EXPECT_EQ(columns[2 * 4 + 2], 0);
}

}  // namespace
}  // namespace qpcu
