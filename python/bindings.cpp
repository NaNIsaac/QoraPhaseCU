#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "qoraphasecu/qoraphasecu.hpp"

namespace nb = nanobind;

namespace {

template <class T>
using Input = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

// (kind, n, beta, threshold, sigma, method, axis), flattened by the Python side.
using OpTuple = std::tuple<std::string, int, float, float, float, std::string, int>;

qpcu::Shape shape_of(const Input<float>& a) {
  qpcu::Shape shape;
  for (std::size_t i = 0; i < a.ndim(); ++i) shape.push_back(a.shape(i));
  return shape;
}

template <class T>
std::span<const T> view(const Input<T>& a) {
  return {a.data(), a.size()};
}

template <class T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& v, const qpcu::Shape& shape) {
  auto* owned = new std::vector<T>(std::move(v));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T>(owned->data(), shape.size(), shape.data(), owner);
}

qpcu::Recipe to_recipe(const std::vector<OpTuple>& ops) {
  qpcu::Recipe r;
  for (const auto& [kind, n, beta, threshold, sigma, method, axis] : ops) {
    if (kind == "er") {
      r.ops.emplace_back(qpcu::ER{n});
    } else if (kind == "hio") {
      r.ops.emplace_back(qpcu::HIO{n, beta});
    } else if (kind == "raar") {
      r.ops.emplace_back(qpcu::RAAR{n, beta});
    } else if (kind == "shrinkwrap") {
      const auto m = method == "rms" ? qpcu::Threshold::rms : qpcu::Threshold::max;
      r.ops.emplace_back(qpcu::Shrinkwrap{threshold, sigma, m});
    } else if (kind == "detwin_begin") {
      r.ops.emplace_back(qpcu::DetwinBegin{axis});
    } else if (kind == "detwin_end") {
      r.ops.emplace_back(qpcu::DetwinEnd{});
    } else {
      throw std::invalid_argument("unknown recipe operation '" + kind + "'");
    }
  }
  return r;
}

nb::list reconstruct(const Input<float>& intensity, std::optional<Input<std::uint8_t>> support,
                     std::optional<Input<std::uint8_t>> mask, const std::vector<OpTuple>& ops,
                     int starts, int batch, std::uint64_t seed, float support_threshold,
                     const std::string& backend, const std::string& variant, int sync_every) {
  const qpcu::Shape shape = shape_of(intensity);
  const qpcu::Recipe recipe = to_recipe(ops);
  const qpcu::Options options{.backend = qpcu::parse_backend(backend),
                              .variant = qpcu::parse_variant(variant),
                              .starts = starts,
                              .batch = batch,
                              .seed = seed,
                              .sync_every = sync_every};
  std::vector<qpcu::Reconstruction> runs;
  {
    nb::gil_scoped_release release;
    const auto problem = qpcu::make_problem(
        view(intensity), shape, support ? view(*support) : std::span<const std::uint8_t>{},
        mask ? view(*mask) : std::span<const std::uint8_t>{}, support_threshold);
    runs = qpcu::reconstruct(problem, recipe, options);
  }
  nb::list out;
  for (auto& run : runs) {
    const qpcu::Shape history{run.errors.size()};
    out.append(nb::make_tuple(to_numpy(std::move(run.object), shape),
                              to_numpy(std::move(run.support), shape),
                              to_numpy(std::move(run.errors), history)));
  }
  return out;
}

}  // namespace

NB_MODULE(_core, m) {
  m.def("reconstruct", &reconstruct, nb::arg("intensity"), nb::arg("support").none(),
        nb::arg("mask").none(), nb::arg("ops"), nb::arg("starts"), nb::arg("batch"),
        nb::arg("seed"), nb::arg("support_threshold"), nb::arg("backend"), nb::arg("variant"),
        nb::arg("sync_every"));
  m.def(
      "autocorrelation_support",
      [](const Input<float>& intensity, float threshold) {
        const auto shape = shape_of(intensity);
        return to_numpy(qpcu::autocorrelation_support(view(intensity), shape, threshold), shape);
      },
      nb::arg("intensity"), nb::arg("threshold"));
  m.def(
      "aligned_error",
      [](const Input<qpcu::Complex>& a, const Input<qpcu::Complex>& reference) {
        qpcu::Shape shape;
        for (std::size_t i = 0; i < a.ndim(); ++i) shape.push_back(a.shape(i));
        return qpcu::aligned_error(view(a), view(reference), shape);
      },
      nb::arg("a"), nb::arg("reference"));
  m.def("cuda_available", &qpcu::cuda_available);
}
