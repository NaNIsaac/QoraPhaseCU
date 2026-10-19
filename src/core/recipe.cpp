#include "qoraphasecu/recipe.hpp"

#include <stdexcept>

namespace qpcu {

namespace {

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

}  // namespace

int Recipe::iterations() const {
  int total = 0;
  for (const Op& op : ops) {
    std::visit(Overloaded{[&](const ER& s) { total += s.n; }, [&](const HIO& s) { total += s.n; },
                          [&](const RAAR& s) { total += s.n; }, [](const auto&) {}},
               op);
  }
  return total;
}

Recipe default_recipe() {
  return recipe(RAAR{50}, detwin(HIO{20}), RAAR{50}, repeat(8, RAAR{50}, Shrinkwrap{}), ER{200});
}

void validate(const Recipe& recipe) {
  if (recipe.ops.empty()) throw std::invalid_argument("recipe is empty");
  bool in_detwin = false;
  for (const Op& op : recipe.ops) {
    std::visit(Overloaded{
                   [](const ER& s) {
                     if (s.n < 1) throw std::invalid_argument("ER needs n >= 1");
                   },
                   [](const HIO& s) {
                     if (s.n < 1) throw std::invalid_argument("HIO needs n >= 1");
                   },
                   [](const RAAR& s) {
                     if (s.n < 1) throw std::invalid_argument("RAAR needs n >= 1");
                   },
                   [](const Shrinkwrap& s) {
                     if (!(s.threshold > 0.0f) || !(s.sigma >= 0.0f)) {
                       throw std::invalid_argument("Shrinkwrap needs threshold > 0, sigma >= 0");
                     }
                   },
                   [&](const DetwinBegin&) {
                     if (in_detwin) throw std::invalid_argument("nested detwin");
                     in_detwin = true;
                   },
                   [&](const DetwinEnd&) {
                     if (!in_detwin) throw std::invalid_argument("DetwinEnd without DetwinBegin");
                     in_detwin = false;
                   },
               },
               op);
  }
  if (in_detwin) throw std::invalid_argument("DetwinBegin without DetwinEnd");
}

}  // namespace qpcu
