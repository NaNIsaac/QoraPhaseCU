#include <gtest/gtest.h>

#include "qoraphasecu/metrics.hpp"
#include "qoraphasecu/solver.hpp"
#include "synthetic.hpp"

namespace qpcu {
namespace {

float best_aligned_error(const test::Synthetic& s, const Recipe& r, int starts) {
  const Problem p = make_problem(s.intensity, s.shape, s.support);
  const auto runs = reconstruct(p, r, {.starts = starts, .seed = 7});
  float best = 1e9f;
  for (const auto& run : runs) best = std::min(best, aligned_error(run.object, s.object, s.shape));
  return best;
}

TEST(Recover, AlignedErrorIgnoresShiftPhaseAndTwin) {
  const auto s = test::triangle({32, 32}, 10);
  std::vector<Complex> moved(s.object.size());
  const Complex rotation = std::polar(1.0f, 0.7f);
  for (std::size_t y = 0; y < 32; ++y) {
    for (std::size_t x = 0; x < 32; ++x) {
      const std::size_t src = ((32 - y) % 32) * 32 + (32 - x + 3) % 32;
      moved[y * 32 + x] = rotation * std::conj(s.object[src]);
    }
  }
  EXPECT_LT(aligned_error(moved, s.object, s.shape), 1e-5f);
  EXPECT_GT(aligned_error(test::random_field(moved.size(), 4), s.object, s.shape), 0.5f);
}

TEST(Recover, HioThenErRecovers2D) {
  const auto s = test::triangle({64, 64}, 20);
  EXPECT_LT(best_aligned_error(s, recipe(HIO{300}, ER{100}), 4), 1e-2f);
}

TEST(Recover, ErNeverIncreasesTheError) {
  const auto s = test::triangle({64, 64}, 20);
  const Problem p = make_problem(s.intensity, s.shape, s.support);
  const auto run = reconstruct(p, recipe(ER{100}))[0];
  ASSERT_EQ(run.errors.size(), 100u);
  EXPECT_LT(run.errors.back(), run.errors.front());
  for (std::size_t i = 1; i < run.errors.size(); ++i) {
    EXPECT_LE(run.errors[i], run.errors[i - 1] * (1 + 1e-4f)) << i;
  }
}

TEST(Recover, RaarRecovers3D) {
  const auto s = test::triangle({24, 24, 24}, 8);
  EXPECT_LT(best_aligned_error(s, recipe(RAAR{300}, ER{100}), 2), 1e-2f);
}

TEST(Recover, DefaultRecipeWithoutSupport) {
  const auto s = test::triangle({64, 64}, 24);
  const Problem p = make_problem(s.intensity, s.shape);
  const auto runs = reconstruct(p, default_recipe(), {.starts = 8});
  const auto& best = runs[best_index(runs)];
  EXPECT_LT(best.error(), 1e-3f);
  EXPECT_LT(aligned_error(best.object, s.object, s.shape), 1e-2f);
}

TEST(Solver, ProgressFiresEverySyncInterval) {
  const auto s = test::triangle({32, 32}, 8);
  const Problem p = make_problem(s.intensity, s.shape, s.support);
  std::vector<int> seen;
  Options o{.starts = 2, .sync_every = 4};
  o.progress = [&](int it, std::span<const float> errors) {
    seen.push_back(it);
    EXPECT_EQ(errors.size(), 2u);
  };
  reconstruct(p, recipe(HIO{6}, ER{6}), o);
  EXPECT_EQ(seen, (std::vector<int>{4, 8, 12}));
}

TEST(Solver, RunContinuesFromTheCurrentState) {
  const auto s = test::triangle({32, 32}, 8);
  const Problem p = make_problem(s.intensity, s.shape, s.support);
  auto solver = make_solver(p);
  solver->run(recipe(HIO{10}));
  solver->run(recipe(ER{5}));
  const auto split = solver->results()[0];
  const auto whole = reconstruct(p, recipe(HIO{10}, ER{5}))[0];
  EXPECT_EQ(split.errors, whole.errors);
  EXPECT_EQ(split.object, whole.object);
}

}  // namespace
}  // namespace qpcu
