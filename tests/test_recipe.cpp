#include <gtest/gtest.h>

#include "qoraphasecu/recipe.hpp"

namespace qpcu {
namespace {

TEST(Recipe, RepeatAndDetwinExpandInPlace) {
  const Recipe r = recipe(HIO{5}, repeat(3, RAAR{10}, Shrinkwrap{}), detwin(ER{2}));
  ASSERT_EQ(r.ops.size(), 1u + 6u + 3u);
  EXPECT_TRUE(std::holds_alternative<RAAR>(r.ops[1]));
  EXPECT_TRUE(std::holds_alternative<Shrinkwrap>(r.ops[6]));
  EXPECT_TRUE(std::holds_alternative<DetwinBegin>(r.ops[7]));
  EXPECT_TRUE(std::holds_alternative<DetwinEnd>(r.ops[9]));
  EXPECT_EQ(r.iterations(), 5 + 30 + 2);
}

TEST(Recipe, DefaultMatchesQoraPhase) {
  // RAAR(50), detwin(HIO(20)), RAAR(50), repeat(8, RAAR(50), Shrinkwrap()), ER(200)
  EXPECT_EQ(default_recipe().iterations(), 50 + 20 + 50 + 8 * 50 + 200);
  EXPECT_NO_THROW(validate(default_recipe()));
}

TEST(Recipe, ValidateRejectsMalformed) {
  EXPECT_THROW(validate(Recipe{}), std::invalid_argument);
  EXPECT_THROW(validate(recipe(ER{0})), std::invalid_argument);
  EXPECT_THROW(validate(recipe(DetwinBegin{}, ER{1})), std::invalid_argument);
  EXPECT_THROW(validate(recipe(DetwinEnd{})), std::invalid_argument);
  EXPECT_THROW(validate(recipe(detwin(detwin(ER{1})))), std::invalid_argument);
  EXPECT_THROW(validate(recipe(Shrinkwrap{.threshold = 0.0f})), std::invalid_argument);
}

}  // namespace
}  // namespace qpcu
