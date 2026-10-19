#include <variant>

#include "../core/backends.hpp"
#include "../core/layout.hpp"
#include "kernels.hpp"

namespace qpcu::detail {

namespace {

struct Start {
  std::vector<Complex> x;
  std::vector<std::uint8_t> support;
  std::vector<std::uint8_t> saved_support;
  std::vector<float> errors;
};

class CpuSolver final : public Solver {
 public:
  CpuSolver(const Problem& problem, const Options& options)
      : shape_(problem.shape),
        fft_(problem.shape),
        amplitude_(to_fft_order<float>(problem.amplitude, shape_)),
        mask_(to_fft_order<std::uint8_t>(problem.mask, shape_)),
        energy_(cpu::usable_energy(amplitude_, mask_)),
        options_(options),
        wave_(amplitude_.size()) {
    const auto support = to_fft_order<std::uint8_t>(problem.support, shape_);
    starts_.resize(static_cast<std::size_t>(options.starts));
    for (std::size_t s = 0; s < starts_.size(); ++s) {
      starts_[s].support = support;
      starts_[s].x.resize(amplitude_.size());
      cpu::random_start(amplitude_, support, fft_, options.seed, s, starts_[s].x);
    }
  }

  void run(const Recipe& recipe) override {
    validate(recipe);
    for (const Op& op : recipe.ops) {
      std::visit([this](const auto& o) { apply(o); }, op);
    }
  }

  std::vector<Reconstruction> results() const override {
    std::vector<Reconstruction> out;
    for (const Start& s : starts_) {
      out.push_back({to_centered<Complex>(s.x, shape_),
                     to_centered<std::uint8_t>(s.support, shape_), s.errors});
    }
    return out;
  }

 private:
  void apply(const ER& o) { iterate(o.n, er_coefficients()); }
  void apply(const HIO& o) { iterate(o.n, hio_coefficients(o.beta)); }
  void apply(const RAAR& o) { iterate(o.n, raar_coefficients(o.beta)); }

  void apply(const Shrinkwrap& o) {
    for (Start& s : starts_) cpu::shrinkwrap(s.x, shape_, fft_, o, s.support);
  }

  void apply(const DetwinBegin& o) {
    for (Start& s : starts_) {
      s.saved_support = s.support;
      const auto centered = to_centered<std::uint8_t>(s.support, shape_);
      s.support = to_fft_order<std::uint8_t>(half_support(centered, shape_, o.axis), shape_);
    }
  }

  void apply(const DetwinEnd&) {
    for (Start& s : starts_) s.support = std::move(s.saved_support);
  }

  void iterate(int n, UpdateCoefficients c) {
    for (int i = 0; i < n; ++i) {
      for (Start& s : starts_) {
        fft_.forward(s.x, wave_);
        s.errors.push_back(cpu::modulus(wave_, amplitude_, mask_, energy_));
        fft_.inverse(wave_, wave_);
        cpu::update(s.x, wave_, s.support, c);
      }
      ++iteration_;
      if (options_.progress && iteration_ % options_.sync_every == 0) report();
    }
  }

  void report() const {
    std::vector<float> latest;
    for (const Start& s : starts_) latest.push_back(s.errors.back());
    options_.progress(iteration_, latest);
  }

  Shape shape_;
  cpu::Fft fft_;
  std::vector<float> amplitude_;
  std::vector<std::uint8_t> mask_;
  double energy_;
  Options options_;
  std::vector<Complex> wave_;
  std::vector<Start> starts_;
  int iteration_ = 0;
};

}  // namespace

std::unique_ptr<Solver> make_cpu_solver(const Problem& problem, const Options& options) {
  return std::make_unique<CpuSolver>(problem, options);
}

}  // namespace qpcu::detail
