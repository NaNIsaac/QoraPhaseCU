#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>
#include <variant>

#include "../core/backends.hpp"
#include "../core/layout.hpp"
#include "common.cuh"
#include "kernels.cuh"

namespace qpcu::detail {

namespace {

using cuda::DeviceBuffer;

class CudaGraph {
 public:
  explicit CudaGraph(cudaGraph_t graph) {
    QPCU_CHECK(cudaGraphInstantiate(&exec_, graph, 0));
    cudaGraphDestroy(graph);
  }
  CudaGraph(const CudaGraph&) = delete;
  CudaGraph& operator=(const CudaGraph&) = delete;
  ~CudaGraph() { cudaGraphExecDestroy(exec_); }
  void launch(cudaStream_t stream) const { QPCU_CHECK(cudaGraphLaunch(exec_, stream)); }

 private:
  cudaGraphExec_t exec_{};
};

std::vector<int> fft_dims(const Shape& shape) {
  std::vector<int> dims;
  for (std::size_t d : shape) dims.push_back(static_cast<int>(d));
  return dims;
}

std::array<std::uint32_t, 3> dims3_u32(const Shape& shape) {
  const auto d = dims3(shape);
  return {static_cast<std::uint32_t>(d[0]), static_cast<std::uint32_t>(d[1]),
          static_cast<std::uint32_t>(d[2])};
}

template <class T>
DeviceBuffer<T> upload(const std::vector<T>& host) {
  DeviceBuffer<T> d(host.size());
  QPCU_CHECK(cudaMemcpy(d.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

class CudaSolver final : public Solver {
 public:
  CudaSolver(const Problem& problem, const Options& options)
      : shape_(problem.shape),
        n_(numel(shape_)),
        options_(options),
        starts_(options.starts),
        batch_(options.variant == Variant::batched ? options.starts : 1),
        groups_(starts_ / batch_),
        blocks_(cuda::grid_blocks(n_ / 2 + n_ % 2)),
        plan_(fft_dims(shape_).data(), static_cast<int>(shape_.size()), batch_, stream_),
        errors_(static_cast<std::size_t>(starts_)) {
    const auto amplitude = to_fft_order<float>(problem.amplitude, shape_);
    const auto mask = to_fft_order<std::uint8_t>(problem.mask, shape_);
    std::vector<float> signed_amp(n_);
    double energy = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
      signed_amp[i] = mask[i] ? -1.0f : amplitude[i];
      if (!mask[i]) energy += static_cast<double>(amplitude[i]) * amplitude[i];
    }
    energy_ = energy;
    amp_ = upload(amplitude);
    mask_ = upload(mask);
    signed_amp_ = upload(signed_amp);

    const auto initial = to_fft_order<std::uint8_t>(problem.support, shape_);
    std::vector<std::uint8_t> supports;
    for (int s = 0; s < starts_; ++s)
      supports.insert(supports.end(), initial.begin(), initial.end());
    support_ = upload(supports);
    saved_ = DeviceBuffer<std::uint8_t>(supports.size());

    x_ = DeviceBuffer<float2>(static_cast<std::size_t>(starts_) * n_);
    w_ = DeviceBuffer<float2>(static_cast<std::size_t>(batch_) * n_);
    const int partial_count =
        options.variant == Variant::naive ? cuda::error_naive_blocks(n_) : batch_ * blocks_;
    partials_ = DeviceBuffer<float>(static_cast<std::size_t>(partial_count));
    host_partials_.resize(static_cast<std::size_t>(partial_count));
    counters_ = DeviceBuffer<int>(static_cast<std::size_t>(groups_));
    level_ = DeviceBuffer<cuda::LevelAccumulator>(static_cast<std::size_t>(batch_));

    for (int g = 0; g < groups_; ++g) {
      cuda::random_wave(w_.get(), amp_.get(), n_, batch_, options.seed,
                        static_cast<std::uint64_t>(g * batch_), stream_);
      plan_.inverse(w_.get(), x(g));
      cuda::support_project(x(g), support(g), n_, batch_, stream_);
    }
    stream_.sync();
  }

  void run(const Recipe& recipe) override {
    validate(recipe);
    if (device_reduce()) {
      const auto needed = static_cast<std::size_t>(groups_) * batch_ * recipe.iterations();
      if (errs_.size() < needed) errs_ = DeviceBuffer<double>(needed);
      if (capacity_ != recipe.iterations()) {
        graphs_.clear();  // captured graphs hold error slot addresses
        capacity_ = recipe.iterations();
      }
      QPCU_CHECK(cudaMemsetAsync(errs_.get(), 0, errs_.size() * sizeof(double), stream_));
      QPCU_CHECK(cudaMemsetAsync(counters_.get(), 0, counters_.size() * sizeof(int), stream_));
    }
    run_iterations_ = 0;
    for (const Op& op : recipe.ops) {
      std::visit([this](const auto& o) { apply(o); }, op);
    }
    if (device_reduce()) collect_errors();
    stream_.sync();
  }

  std::vector<Reconstruction> results() const override {
    std::vector<Complex> x_host(x_.size());
    std::vector<std::uint8_t> s_host(support_.size());
    QPCU_CHECK(
        cudaMemcpy(x_host.data(), x_.get(), x_.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    QPCU_CHECK(cudaMemcpy(s_host.data(), support_.get(), support_.size(), cudaMemcpyDeviceToHost));
    std::vector<Reconstruction> out;
    for (int s = 0; s < starts_; ++s) {
      const std::size_t off = static_cast<std::size_t>(s) * n_;
      out.push_back({to_centered<Complex>(std::span(x_host).subspan(off, n_), shape_),
                     to_centered<std::uint8_t>(std::span(s_host).subspan(off, n_), shape_),
                     errors_[static_cast<std::size_t>(s)]});
    }
    return out;
  }

 private:
  bool device_reduce() const {
    return options_.variant == Variant::graph || options_.variant == Variant::batched;
  }

  float2* x(int g) const { return x_.get() + static_cast<std::size_t>(g) * batch_ * n_; }
  std::uint8_t* support(int g) const {
    return support_.get() + static_cast<std::size_t>(g) * batch_ * n_;
  }
  double* errs(int g) const {
    return errs_.get() + static_cast<std::size_t>(g) * batch_ * capacity_;
  }

  void apply(const ER& o) { iterate(o.n, er_coefficients()); }
  void apply(const HIO& o) { iterate(o.n, hio_coefficients(o.beta)); }
  void apply(const RAAR& o) { iterate(o.n, raar_coefficients(o.beta)); }

  void apply(const Shrinkwrap& o) {
    for (int g = 0; g < groups_; ++g) {
      cuda::magnitude(x(g), w_.get(), n_, batch_, stream_);
      plan_.forward(w_.get(), w_.get());
      cuda::gaussian(w_.get(), dims3_u32(shape_), o.sigma, batch_, stream_);
      plan_.inverse(w_.get(), w_.get());
      cuda::shrinkwrap_level(w_.get(), support(g), n_, batch_, o.method, level_.get(), stream_);
      cuda::shrinkwrap_threshold(w_.get(), level_.get(), n_, batch_, o.method, o.threshold,
                                 support(g), stream_);
    }
  }

  // Runs once per recipe, so a host round trip is cheaper than a kernel for it.
  void apply(const DetwinBegin& o) {
    QPCU_CHECK(cudaMemcpyAsync(saved_.get(), support_.get(), support_.size(),
                               cudaMemcpyDeviceToDevice, stream_));
    std::vector<std::uint8_t> host(support_.size());
    QPCU_CHECK(
        cudaMemcpyAsync(host.data(), support_.get(), host.size(), cudaMemcpyDeviceToHost, stream_));
    stream_.sync();
    for (int s = 0; s < starts_; ++s) {
      const auto part = std::span(host).subspan(static_cast<std::size_t>(s) * n_, n_);
      const auto half = to_fft_order<std::uint8_t>(
          half_support(to_centered<std::uint8_t>(part, shape_), shape_, o.axis), shape_);
      std::ranges::copy(half, part.begin());
    }
    QPCU_CHECK(cudaMemcpy(support_.get(), host.data(), host.size(), cudaMemcpyHostToDevice));
  }

  void apply(const DetwinEnd&) {
    QPCU_CHECK(cudaMemcpyAsync(support_.get(), saved_.get(), support_.size(),
                               cudaMemcpyDeviceToDevice, stream_));
  }

  void iterate(int n, UpdateCoefficients c) {
    for (int done = 0; done < n;) {
      const int len = std::min(n - done, options_.sync_every - iteration_ % options_.sync_every);
      for (int g = 0; g < groups_; ++g) {
        switch (options_.variant) {
          case Variant::naive:
            for (int i = 0; i < len; ++i) step_naive(g, c);
            break;
          case Variant::fused:
            for (int i = 0; i < len; ++i) step_fused(g, c);
            break;
          case Variant::graph:
          case Variant::batched:
            graph(g, len, c).launch(stream_);
            break;
        }
      }
      done += len;
      iteration_ += len;
      run_iterations_ += len;
      if (options_.progress && iteration_ % options_.sync_every == 0) report();
    }
  }

  void step_naive(int g, UpdateCoefficients c) {
    const float s = 1.0f / std::sqrt(static_cast<float>(n_));
    float2* w = w_.get();
    plan_.forward(x(g), w);
    cuda::scale_naive(w, n_, s, stream_);
    cuda::error_naive(w, amp_.get(), mask_.get(), n_, partials_.get(), stream_);
    QPCU_CHECK(cudaMemcpyAsync(host_partials_.data(), partials_.get(),
                               host_partials_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream_));
    cuda::modulus_naive(w, amp_.get(), mask_.get(), n_, stream_);
    plan_.inverse(w, w);
    cuda::scale_naive(w, n_, s, stream_);
    cuda::update_naive(x(g), w, support(g), n_, c, stream_);
    stream_.sync();
    record_host_errors(g, host_partials_.size());
  }

  void step_fused(int g, UpdateCoefficients c) {
    float2* w = w_.get();
    plan_.forward(x(g), w);
    cuda::modulus_fused(w, signed_amp_.get(), n_, batch_, partials_.get(), blocks_, stream_);
    QPCU_CHECK(cudaMemcpyAsync(host_partials_.data(), partials_.get(),
                               host_partials_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               stream_));
    plan_.inverse(w, w);
    cuda::update_fused(x(g), w, support(g), n_, batch_, c, nullptr, blocks_, stream_);
    stream_.sync();
    record_host_errors(g, static_cast<std::size_t>(blocks_));
  }

  void step_device(int g, UpdateCoefficients c) {
    float2* w = w_.get();
    int* counter = counters_.get() + g;
    plan_.forward(x(g), w);
    cuda::modulus_reduced(w, signed_amp_.get(), n_, batch_, errs(g), counter, blocks_, stream_);
    plan_.inverse(w, w);
    cuda::update_fused(x(g), w, support(g), n_, batch_, c, counter, blocks_, stream_);
  }

  // `per_start` consecutive partials belong to each start of the group.
  void record_host_errors(int g, std::size_t per_start) {
    for (int b = 0; b < batch_; ++b) {
      double num = 0.0;
      for (std::size_t k = 0; k < per_start; ++k) num += host_partials_[b * per_start + k];
      errors_[static_cast<std::size_t>(g * batch_ + b)].push_back(to_error(num));
    }
  }

  float to_error(double num) const { return static_cast<float>(std::sqrt(num / (energy_ + kEps))); }

  const CudaGraph& graph(int g, int len, UpdateCoefficients c) {
    const auto key = std::make_tuple(g, len, c.a, c.b);
    if (auto it = graphs_.find(key); it != graphs_.end()) return *it->second;
    QPCU_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < len; ++i) step_device(g, c);
    cudaGraph_t captured{};
    QPCU_CHECK(cudaStreamEndCapture(stream_, &captured));
    return *graphs_.emplace(key, std::make_unique<CudaGraph>(captured)).first->second;
  }

  void collect_errors() {
    std::vector<double> host(static_cast<std::size_t>(groups_) * batch_ * capacity_);
    QPCU_CHECK(cudaMemcpyAsync(host.data(), errs_.get(), host.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_));
    stream_.sync();
    for (int g = 0; g < groups_; ++g) {
      for (int t = 0; t < run_iterations_; ++t) {
        for (int b = 0; b < batch_; ++b) {
          const std::size_t at = (static_cast<std::size_t>(g) * capacity_ + t) * batch_ + b;
          errors_[static_cast<std::size_t>(g * batch_ + b)].push_back(to_error(host[at]));
        }
      }
    }
  }

  void report() {
    std::vector<float> latest(static_cast<std::size_t>(starts_));
    if (device_reduce()) {
      std::vector<double> host(static_cast<std::size_t>(batch_));
      for (int g = 0; g < groups_; ++g) {
        const double* last = errs(g) + static_cast<std::size_t>(run_iterations_ - 1) * batch_;
        QPCU_CHECK(cudaMemcpyAsync(host.data(), last, host.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream_));
        stream_.sync();
        for (int b = 0; b < batch_; ++b) latest[g * batch_ + b] = to_error(host[b]);
      }
    } else {
      for (int s = 0; s < starts_; ++s) latest[s] = errors_[s].back();
    }
    options_.progress(iteration_, latest);
  }

  Shape shape_;
  std::size_t n_;
  Options options_;
  int starts_;
  int batch_;
  int groups_;
  int blocks_;
  cuda::Stream stream_;
  cuda::FftPlan plan_;
  double energy_ = 0.0;
  DeviceBuffer<float> amp_;
  DeviceBuffer<std::uint8_t> mask_;
  DeviceBuffer<float> signed_amp_;
  DeviceBuffer<std::uint8_t> support_;
  DeviceBuffer<std::uint8_t> saved_;
  DeviceBuffer<float2> x_;
  DeviceBuffer<float2> w_;
  DeviceBuffer<float> partials_;
  std::vector<float> host_partials_;
  DeviceBuffer<int> counters_;
  DeviceBuffer<double> errs_;
  DeviceBuffer<cuda::LevelAccumulator> level_;
  std::vector<std::vector<float>> errors_;
  std::map<std::tuple<int, int, float, float>, std::unique_ptr<CudaGraph>> graphs_;
  int capacity_ = 0;
  int iteration_ = 0;
  int run_iterations_ = 0;
};

}  // namespace

std::unique_ptr<Solver> make_cuda_solver(const Problem& problem, const Options& options) {
  if (!cuda_available()) throw std::runtime_error("no CUDA device available");
  return std::make_unique<CudaSolver>(problem, options);
}

}  // namespace qpcu::detail
