#pragma once

#include <variant>
#include <vector>

namespace qpcu {

// x <- P_S P_M x
struct ER {
  int n;
};

// x <- P_M x inside the support, x - beta P_M x outside.
struct HIO {
  int n;
  float beta = 0.9f;
};

// x <- (beta / 2)(R_S R_M x + x) + (1 - beta) P_M x
struct RAAR {
  int n;
  float beta = 0.9f;
};

enum class Threshold { max, rms };

// Re-derives the support from the Gaussian-blurred object magnitude. The
// threshold is relative to the blurred maximum (`max`) or to the rms inside the
// current support (`rms`); the two scales do not transfer.
struct Shrinkwrap {
  float threshold = 0.3f;
  float sigma = 1.0f;
  Threshold method = Threshold::max;
};

// Cuts the support in half (see half_support) until the matching DetwinEnd,
// which restores the support saved here.
struct DetwinBegin {
  int axis = 0;
};

struct DetwinEnd {};

using Op = std::variant<ER, HIO, RAAR, Shrinkwrap, DetwinBegin, DetwinEnd>;

// A flat list of operations. repeat() and detwin() expand in place, so a backend
// only ever walks a vector and never recurses.
struct Recipe {
  std::vector<Op> ops;

  // Number of iterations that each produce one error value.
  int iterations() const;
};

namespace detail {
inline void append(Recipe& r, const Recipe& part) {
  r.ops.insert(r.ops.end(), part.ops.begin(), part.ops.end());
}
inline void append(Recipe& r, const Op& op) { r.ops.push_back(op); }
}  // namespace detail

template <class... Parts>
Recipe recipe(const Parts&... parts) {
  Recipe r;
  (detail::append(r, parts), ...);
  return r;
}

template <class... Parts>
Recipe repeat(int n, const Parts&... parts) {
  const Recipe body = recipe(parts...);
  Recipe r;
  for (int i = 0; i < n; ++i) detail::append(r, body);
  return r;
}

template <class... Parts>
Recipe detwin(const Parts&... parts) {
  return recipe(DetwinBegin{}, parts..., DetwinEnd{});
}

// RAAR, a detwin pass, then shrinkwrapped RAAR and ER: QoraPhase's default.
Recipe default_recipe();

// Throws std::invalid_argument on non-positive counts or unbalanced detwin.
void validate(const Recipe& recipe);

}  // namespace qpcu
