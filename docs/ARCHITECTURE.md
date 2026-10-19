# Architecture

```
include/qoraphasecu/   public API, the same for both backends
  problem.hpp          Problem, make_problem, autocorrelation_support, half_support,
                       simulate_intensity
  recipe.hpp           ER, HIO, RAAR, Shrinkwrap, DetwinBegin/End, recipe/repeat/detwin
  solver.hpp           Backend, Variant, Options, Reconstruction, Solver, reconstruct
  operators.hpp        single projections in qpcu::cpu and qpcu::cuda, for tests
  metrics.hpp          aligned_error
src/core/              backend-independent code
  pointwise.hpp        formulas shared by CPU loops and CUDA kernels (QPCU_HD)
  layout.hpp           fftshift / ifftshift between center-origin and FFT order
src/cpu/               reference backend on pocketfft
src/cuda/              CUDA backend on cuFFT
  cuda_solver.cu       CudaSolver: buffers, plans, the variants, CUDA graphs
  kernels.cu           fp32 kernels (naive, fused, shrinkwrap, setup)
  half.cu              fp16-storage kernels
  callbacks.cu         cuFFT LTO store callbacks, compiled to LTO-IR and embedded
  device.cuh           device helpers shared by both kernel files
  common.cuh           error checks, RAII device buffer, stream, cuFFT plans
python/                nanobind module _core and the qoraphasecu package
bench/                 qpcu_bench and the QoraPhase comparison
```

## Mapping from QoraPhase

| QoraPhase | QoraPhaseCU |
|---|---|
| `qp.CDI(intensity, support=, mask=)` | `qpcu::make_problem(intensity, shape, support, mask)` |
| `qp.Recipe(...)`, `qp.repeat`, `qp.detwin` | `qpcu::recipe(...)`, `qpcu::repeat`, `qpcu::detwin` |
| `qp.ER(n)`, `qp.HIO(n, beta)`, `qp.RAAR(n, beta)` | `qpcu::ER{n}`, `qpcu::HIO{n, beta}`, `qpcu::RAAR{n, beta}` |
| `qp.Shrinkwrap(threshold, sigma, method)` | `qpcu::Shrinkwrap{threshold, sigma, Threshold::max/rms}` |
| `default_recipe()` | `qpcu::default_recipe()`, the same stages and counts |
| `reconstruct(..., starts=, batch=)` with `vmap` | `Options::starts`, `Options::batch`, `Variant::batched` |
| `Reconstruction.metrics` | `Reconstruction::errors` |

Each operator follows the QoraPhase source: the Fourier error
`sqrt(sum_valid (|F x| - A)^2 / sum_valid A^2)` computed from the forward
transform the projection already paid for; masked pixels keep their computed
magnitude; the support from the autocorrelation thresholded at 0.1 of its
maximum; the half support cut at the centre of mass along an axis; the Gaussian
blur as the transfer function `exp(-2 (pi sigma f)^2)`. What differs is the
random start: QoraPhase draws phases from JAX's PRNG, QoraPhaseCU from a
counter-based hash (`detail::uniform`), so the two libraries start from
different phases for the same seed.

## Recipes are flat

`repeat` and `detwin` expand when the recipe is built, so a recipe is a
`std::vector` of operations and a backend walks it without recursion. Detwin
becomes a `DetwinBegin` that saves the support and installs the half support,
and a `DetwinEnd` that restores the saved one. QoraPhase's `lax.scan` over a
nested recipe is a compile-time device; here the loop is plain C++ and the CUDA
graph plays the role of the compiled body.

## FFT order inside, center-origin outside

All public arrays are center-origin, as in QoraPhase, where the centered
transform is `fftshift(fftn(ifftshift(x)))`. Both backends convert the inputs to
FFT order once, at construction, and convert the results back in `results()`.
Under that relabelling the modulus constraint, the support projection and the
update are unchanged, because they are element-wise, and the shrinkwrap blur is
unchanged because a circular convolution commutes with a roll. The iteration
therefore never shifts: it saves four passes over the array per iteration that a
literal translation of the JAX code would make. Only the detwin cut needs
center-origin coordinates; it converts the support on the host, once per detwin.

## One update for three algorithms

Inside the support every algorithm keeps P_M x. Outside it they keep
`a x + b P_M x` with (a, b) = (0, 0) for ER, (1, -beta) for HIO and
(beta, 1 - 2 beta) for RAAR, the last one being RAAR's reflection formula
expanded. The CPU loop and every CUDA update kernel take `UpdateCoefficients`,
so adding an algorithm of this form needs no new kernel.

## CPU and CUDA agree

`src/core/pointwise.hpp` holds what both backends evaluate: the random phase
`uniform(seed, start, index)`, the update coefficients and the FFT frequencies.
Both backends compute the same expressions in float, the denominator energy in
double, and run the same sequence of operations, so they produce the same
objects to float rounding over short runs. Over hundreds of HIO iterations the
rounding differences between pocketfft and cuFFT grow (phase retrieval is
chaotic), so long runs from the same seed converge to different, equally valid
solutions; the tests compare short runs and check recovery separately.

## The CUDA solver

`CudaSolver` holds all starts' x and support on the device (`starts x n`). A
start group is the set of starts that share one cuFFT plan: one start per group
for the naive, fused, graph and half variants, `Options::batch` (default all)
for the batched variant. For each operation of the recipe, every group runs in
turn, so iteration counts and progress reports line up across starts.

Errors are kept per group in a device array laid out `[iteration][start]`.
Variants with host reduction append each error to the host history as soon as
the round trip completes; graph, batched and half accumulate on the device and
the history is copied once per `run()`. A captured graph bakes in the addresses
of its error slots, so the graph cache is cleared whenever the recipe length
(and with it the layout of the error array) changes.

The callback variant keeps two extra cuFFT plans whose store callbacks do the
modulus projection and the update (src/cuda/callbacks.cu). x is then a double
buffer: the inverse transform writes the new x into one half while its callback
reads the old x from the other, and `parity_` records which half is current.
Graphs are cached per starting half as well. The callbacks are built by
cmake/cufft_callbacks.cmake as a separate nvcc invocation that emits LTO-IR for
each entry of `CMAKE_CUDA_ARCHITECTURES`, and cmake/embed.cmake turns the
fatbin into a C++ byte array linked into the library.

Shrinkwrap runs on the device for every variant: magnitude into the complex work
buffer, forward FFT, Gaussian transfer and 1/n in one kernel, inverse FFT, a
per-start level reduction (max with `atomicMax` on the float bits, which order
like the floats for non-negative values, or rms inside the support with
`atomicAdd`), then the threshold. The fp16 variant keeps fp32 for this, since
it runs rarely and decides the support.

## Python

`python/bindings.cpp` exposes one `reconstruct` that takes numpy arrays and a
recipe flattened to tuples by `python/qoraphasecu/recipe.py`. The GIL is
released while the engine runs. The Python layer mirrors QoraPhase:
`reconstruct(intensity, support=None, mask=None, recipe=None, starts=1,
batch=None, seed=None)` returns a `Reconstruction`, or an `Ensemble` with
`best` and `ranking` when `starts > 1`.

## Tests

`tests/` (GoogleTest) checks the CPU reference directly: modulus projection is
idempotent and leaves masked pixels unconstrained, support projection is
idempotent, ER never raises the error, HIO and RAAR recover a non-centrosymmetric
object in 2D and 3D up to translation, global phase and twin, and the default
recipe recovers it with no support given. The CUDA tests run every variant
against the CPU, check that batched and looped starts agree, cover odd sizes
(the scalar path) and progress reporting, and pin the fp16 accuracy floor. They
skip themselves when no device is present, which is how the CUDA job runs in CI.
