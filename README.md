# QoraPhaseCU

[![Tests](https://github.com/NaNIsaac/QoraPhaseCU/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/QoraPhaseCU/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/NaNIsaac/QoraPhaseCU/blob/main/LICENSE)

**Phase retrieval for coherent diffraction imaging in C++20 and CUDA: the algorithms of [QoraPhase](https://github.com/NaNIsaac/QoraPhase), on cuFFT and hand-written kernels.**

QoraPhaseCU recovers a complex object from a measured 2D or 3D far-field
intensity with ER, HIO, RAAR, shrinkwrap, detwinning and multiple random starts,
following QoraPhase operator for operator. It has a CPU reference backend and a
CUDA backend behind the same C++ API, and a Python API shaped like
`qoraphase.reconstruct`. The CUDA backend is also an optimization study: each
step from a naive port to batched, graph-launched, fp16-storage kernels and cuFFT
LTO callbacks is kept as a selectable variant and measured in
[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).

![Simulated object, its diffraction intensity, and the reconstruction](https://raw.githubusercontent.com/NaNIsaac/QoraPhaseCU/main/docs/reconstruction.png)

The 64×64 object of the QoraPhase README, its noiseless intensity, and the best
of 8 random starts from `qpc.reconstruct(intensity, starts=8)` on the CUDA
backend with no support given (it is derived from the autocorrelation and
shrinkwrapped). Fourier error 2.5e-7, relative error against the true object
7.2e-7. Script: `examples/readme_figure.py`.

## Quick start

Requires CMake 3.28+, a C++20 compiler (MSVC or gcc), and for the GPU backend
the CUDA Toolkit 12.8 or newer.

```bash
cmake --preset cuda-release        # or cpu-release without CUDA
cmake --build --preset cuda-release
ctest --preset cuda-release
```

```cpp
#include <qoraphasecu/qoraphasecu.hpp>

const auto problem = qpcu::make_problem(intensity, {256, 256});  // support derived
const auto runs = qpcu::reconstruct(problem, qpcu::default_recipe(),
                                    {.backend = qpcu::Backend::cuda, .starts = 8});
const auto& best = runs[qpcu::best_index(runs)];                 // best.object, best.errors
```

`examples/reconstruct.cpp` is the same in full. Link with
`target_link_libraries(app PRIVATE qoraphasecu::qoraphasecu)` after
`add_subdirectory`.

Python (numpy in, numpy out), built from a clone:

```bash
pip install .                                               # CPU only
pip install . -C cmake.define.QORAPHASECU_CUDA=ON           # with the CUDA backend
```

```python
import numpy as np
import qoraphasecu as qpc

intensity = np.load("pattern.npy")    # centred 2D or 3D diffraction intensity
result = qpc.reconstruct(intensity)   # pass support=... if you have one
result.amplitude, result.phase, result.errors
```

`backend="auto"` (the default) uses CUDA when a device is present.

## Array conventions

The same as QoraPhase:

| | |
|---|---|
| **Origin** | Every public array is **center-origin**: index `n / 2` is q=0. The FFT order is internal. |
| **`intensity`** | A measured **intensity**, not an amplitude. Internally `sqrt(max(intensity, 0))`. |
| **`mask`** | `1` / `True` marks an **unusable** pixel. Masked pixels are left unconstrained, never forced to zero. |
| **`support`** | `1` / `True` is inside the object. Omit it and an autocorrelation support is derived. |

## Composing a reconstruction

Code order is execution order, and the first argument of an algorithm is its
iteration count.

```cpp
const auto r = qpcu::recipe(
    qpcu::repeat(10, qpcu::RAAR{50, 0.9f}, qpcu::Shrinkwrap{}),
    qpcu::ER{200});
```

```python
recipe = qpc.Recipe(
    qpc.repeat(10, qpc.RAAR(50, beta=0.9), qpc.Shrinkwrap()),
    qpc.ER(200),
)
ensemble = qpc.reconstruct(intensity, recipe=recipe, starts=32, batch=4)
best = ensemble.best
```

`default_recipe()` is QoraPhase's: RAAR(50), detwin(HIO(20)), RAAR(50),
repeat(8, RAAR(50), Shrinkwrap()), ER(200). Shrinkwrap defaults to threshold 0.3
and sigma 1.0 relative to the maximum (`Threshold::rms` is the other option);
QoraPhase's README explains why those are the first numbers to re-tune on real
data.

A reconstruction is determined only up to a global phase, a translation and a
conjugate flip. `qpcu::aligned_error(a, reference, shape)` (Python:
`qpc.aligned_error`) compares two objects after removing all three, to whole
pixels.

## CUDA variants

| variant | what it adds |
|---|---|
| `naive` | one kernel per operation, error summed on the host every iteration |
| `fused` | modulus projection with error partials in one kernel, support and update in another; float4 loads |
| `graph` | warp-shuffle error reduction on the device, iterations replayed from a CUDA graph |
| `batched` (default) | `graph` with `Options::batch` starts per batched cuFFT plan |
| `half` | `graph` with x and the wave in fp16 and fp16 cuFFT; error floor near 1e-3, power-of-two sizes |
| `callback` | `graph` with the modulus projection and the update in cuFFT LTO store callbacks; needs nvJitLink and NVRTC at run time |

HIO, single start, ms per iteration (RTX 5070, AMD Ryzen 5 7500F, Windows 11,
CUDA 13.4; `qpcu_bench --iters 200`):

| size | CPU reference | naive | fused | graph | half | callback |
|---|---:|---:|---:|---:|---:|---:|
| 256^2 | 0.52 | 0.135 | 0.102 | 0.011 | 0.010 | 0.019 |
| 1024^2 | 7.23 | 0.194 | 0.139 | 0.059 | 0.111 | 0.070 |
| 2048^2 | - | 0.742 | 0.527 | 0.452 | 0.448 | 0.519 |
| 128^3 | 16.24 | 0.287 | 0.191 | 0.113 | 0.084 | 0.143 |
| 256^3 | - | 5.581 | 4.184 | 4.115 | 2.299 | 3.251 |

At 256^3 the element-wise kernels of `graph` run at an estimated 83% of the
672 GB/s DRAM peak and cuFFT takes two thirds of the iteration. Folding them
into cuFFT callbacks saves 21% there in full precision, but is 15 to 73% slower
at the sizes whose working set fits the 48 MiB L2. Batching starts into one
plan is 2.5 times faster per start at 256^2 and up to 3 times slower at 1024^2,
where it pushes the working set out of the L2. The derivations, the
multi-start and fp16 accuracy tables, and what was not done are in
[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md). Bandwidth there is modeled from
the bytes each kernel moves; Nsight Compute could not read the GPU counters on
this machine.

Against QoraPhase on the same machine (HIO, ms per iteration, best of five,
`bench/compare_qoraphase.py`). JAX ships CUDA wheels for Linux only, so JAX ran
on the CPU here:

| size | QoraPhase, JAX 0.12 CPU | QoraPhaseCU CPU reference (OpenMP) | QoraPhaseCU CUDA |
|---|---:|---:|---:|
| 256^2 | 0.406 | 0.439 | 0.009 |
| 1024^2 | 5.003 | 6.321 | 0.058 |
| 128^3 | 14.447 | 14.140 | 0.113 |

## What this does not do

- No ensemble alignment or averaging and no input diagnostics; QoraPhase has
  those. `aligned_error` aligns to whole pixels for comparison only.
- No pluggable forward model: the propagator is the unitary FFT.
- No positivity or reality constraint, no partial coherence.
- No file I/O.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the layout and how it
maps onto QoraPhase.

## Development

```bash
cmake --preset cpu-debug && cmake --build --preset cpu-debug && ctest --preset cpu-debug
uv build --wheel && uv venv && uv pip install dist/*.whl pytest && uv run --no-project pytest python/tests
./build/cuda-release/bench/qpcu_bench --sizes 256x256,1024x1024
```
