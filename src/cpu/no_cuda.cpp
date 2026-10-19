// Linked instead of src/cuda when QORAPHASECU_CUDA is off, so the CUDA entry
// points exist on every build and fail at run time.

#include <stdexcept>

#include "../core/backends.hpp"
#include "qoraphasecu/operators.hpp"

namespace qpcu {

namespace {

[[noreturn]] void unavailable() {
  throw std::runtime_error("qoraphasecu was built without CUDA (QORAPHASECU_CUDA=OFF)");
}

}  // namespace

bool cuda_available() { return false; }

DeviceInfo cuda_device_info() { unavailable(); }

namespace detail {
std::unique_ptr<Solver> make_cuda_solver(const Problem&, const Options&) { unavailable(); }
}  // namespace detail

namespace cuda {
float modulus_project(const Problem&, std::span<Complex>) { unavailable(); }
void support_project(std::span<const std::uint8_t>, std::span<Complex>) { unavailable(); }
std::vector<float> gaussian_blur(std::span<const float>, const Shape&, float) { unavailable(); }
}  // namespace cuda

}  // namespace qpcu
