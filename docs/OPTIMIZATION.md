# Optimization log

Every step of the CUDA ladder stays a selectable `qpcu::Variant`, so the
benchmark compares them on the same build. All numbers below come from
`qpcu_bench --iters 200 --starts 8` (bench/bench.cpp) on one machine:

```
GPU: NVIDIA GeForce RTX 5070 (sm_120, 11.9 GiB, L2 48 MiB), peak DRAM bandwidth 672 GB/s
CPU: AMD Ryzen 5 7500F 6-Core Processor (6 cores, 12 threads)
Windows 11, CUDA 13.4, MSVC 14.44, Release build
```

The peak is `2 x memory clock x bus width` from the device attributes. CUDA 13
removed `memoryClockRate` from `cudaDeviceProp`, so the clock comes from
`cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate)` and the bus width from
`cudaDeviceProp::memoryBusWidth` (see `cuda_device_info`).

The workload is HIO on random data with a box support covering the central half
of each axis; timing does not depend on the data. Times are milliseconds per
iteration, after a warm-up run of the same length so that plans and CUDA graphs
already exist.

**Nsight Compute was not used.** The CUDA toolkit on this machine was installed
without admin rights from conda-forge and does not include `ncu`. Bandwidth here
is therefore *modeled*, not measured: bytes per element per iteration are
counted from the loads and stores each element-wise kernel issues, plus one read
and one write of the array per FFT (16 bytes per element for complex64). cuFFT
takes more passes than that for large transforms, so the FFT part of the model
is a lower bound.

| variant | modeled bytes / element / iteration |
|---|---:|
| naive | 32 (FFTs) + 91 (five element-wise kernels) = 123 |
| fused, graph, batched | 32 (FFTs) + 45 (two element-wise kernels) = 77 |
| half | 16 (fp16 FFTs) + 25 (two element-wise kernels) = 41 |

## Results

Single start. ms per iteration, modeled GB/s in parentheses.

| size | cuFFT pair | fp16 cuFFT pair | cpu | naive | fused | graph | half |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256^2 | 0.007 (285) | 0.007 (157) | 0.66 | 0.127 (63) | 0.105 (48) | 0.010 (487) | 0.010 (275) |
| 1024^2 | 0.043 (775) | 0.100 (168) | 10.86 | 0.182 (707) | 0.139 (580) | 0.062 (1306) | 0.113 (382) |
| 2048^2 | 0.184 (730) | 0.367 (183) | - | 0.744 (694) | 0.523 (617) | 0.446 (723) | 0.451 (381) |
| 128^3 | 0.082 (819) | 0.060 (558) | 25.30 | 0.278 (927) | 0.179 (901) | 0.111 (1458) | 0.086 (998) |
| 256^3 | 2.764 (194) | 1.421 (189) | - | 5.576 (370) | 4.184 (309) | 4.108 (314) | 2.296 (300) |

"cuFFT pair" is one forward and one inverse in-place C2C transform replayed from
a CUDA graph: the floor for any variant that keeps cuFFT. The CPU column is the
reference backend (pocketfft on all threads, element-wise loops on one thread),
skipped above 2^21 elements.

Eight starts. ms per iteration per start.

| size | graph, looped starts | batched, all 8 in one plan | batched, 2 per plan |
|---|---:|---:|---:|
| 256^2 | 0.011 | 0.004 | 0.007 |
| 1024^2 | 0.061 | 0.191 | 0.057 |
| 2048^2 | 0.456 | 0.831 | 0.817 |
| 128^3 | 0.119 | 0.271 | 0.191 |
| 256^3 | 4.079 | 4.156 | 4.159 |

Progress reporting (graph variant, 256^2): a host read-back every N iterations.

| N | ms per iteration |
|---:|---:|
| 1 | 0.0655 |
| 10 | 0.0148 |
| 100 | 0.0108 |

A second full run reproduced every row to within a few percent except the
host-synchronised ones at 256^2, which move between runs (fused at 256^2 read
0.178 ms in one run and 0.105 ms in the other; the N = 10 progress row read 0.059
ms in an earlier run). Those rows are bounded by Windows scheduling of the
host-device round trip, not by the GPU.

## Where the time goes

Subtracting the cuFFT pair from a variant leaves the time spent in its own
kernels and launches. At 256^3 the arrays (134 MB each) are far larger than the
48 MiB L2, so this is a clean DRAM measurement:

| 256^3 | total | minus cuFFT pair | element-wise bytes | effective bandwidth |
|---|---:|---:|---:|---:|
| naive | 5.576 | 2.812 | 91 x 16.8M = 1.53 GB | 543 GB/s (81% of peak) |
| fused | 4.184 | 1.420 | 45 x 16.8M = 0.75 GB | 532 GB/s (79%) |
| graph | 4.108 | 1.344 | 0.75 GB | 562 GB/s (84%) |
| half | 2.296 | 0.875 | 25 x 16.8M = 0.42 GB | 479 GB/s (71%) |

The element-wise kernels already run near the bandwidth limit even in the naive
variant; what changes between steps is how many bytes they have to move and how
much of the iteration is spent waiting on the host. Once the element-wise part
is at 84% of peak, cuFFT is two thirds of the iteration at 256^3, and the only
large remaining lever is the FFT itself.

At 1024^2 and 128^3 one complex array is 8 MB and 16 MB, so the whole working
set (x, wave, amplitude, support) fits in L2 and the modeled bandwidth exceeds
the DRAM peak (1306 and 1458 GB/s). 2048^2 (32 MB per array) sits around the
DRAM peak because it fits only partly.

## Step 1: naive

`Variant::naive`, `step_naive` in src/cuda/cuda_solver.cu. One kernel per
operation, one element per thread, no grid-stride loop:

1. forward cuFFT, then a kernel that scales by 1/sqrt(n) (unitary transform);
2. an error kernel that writes one shared-memory partial sum per block;
3. `cudaMemcpyAsync` of the partials and a stream synchronise; the host sums them;
4. a modulus kernel that reads the wave, amplitude and mask again;
5. inverse cuFFT and a second scaling kernel;
6. the update kernel.

It is correct and matches the CPU reference, but every iteration ends in a
host-device round trip, and the wave is read or written five times outside the
FFTs. At 256^2 an iteration costs 0.127 ms, 18 times the 0.007 ms the two FFTs
need: the GPU mostly waits for the host.

## Step 2: fused element-wise kernels

`Variant::fused`. The modulus projection and the error partial sums become one
kernel, and the support projection plus HIO/RAAR/ER update another:

- The 1/sqrt(n) of both unitary transforms is folded into the modulus kernel,
  which multiplies by it twice, so both scaling passes disappear and the FFTs run
  unnormalised.
- Masked pixels are stored as amplitude -1, so the mask costs no extra load
  (`signed_amp`).
- Each thread handles two complex values per iteration of a grid-stride loop,
  as 16-byte `float4` loads (`for_each_pair`). Odd element counts fall back to a
  scalar loop.
- ER, HIO and RAAR share one update: P_M x inside the support, a x + b P_M x
  outside. RAAR's reflections collapse to a = beta, b = 1 - 2 beta (derivation in
  src/core/pointwise.hpp). The update skips loading x when both elements of a
  pair are inside the support.

Element-wise traffic drops from 91 to 45 bytes per element. At 256^3 that is
1.39 ms less per iteration (5.576 to 4.184 ms, 25%), which is what moving 0.78 GB
less at about 540 GB/s predicts. At 256^2 it barely matters (0.127 to 0.105 ms)
because the host round trip still dominates.

## Step 3: on-device reduction, sparse synchronisation, CUDA graphs

`Variant::graph`:

- The block's error partial is reduced with `__shfl_down_sync` (warp, then the
  warp sums through 8 shared-memory slots) and added with one `atomicAdd` per
  block into a device array `errs[iteration]`. The iteration index is a device
  counter that the update kernel increments, so the kernels' arguments do not
  change between iterations.
- Because no argument changes, `sync_every` iterations are captured once into a
  CUDA graph (`cudaStreamBeginCapture`, cuFFT included) and replayed. Graphs are
  cached per (start group, length, update coefficients).
- The host reads errors only when a progress callback asks for them, every
  `sync_every` iterations, and copies the whole history once at the end of
  `run()`.

At 256^2 the iteration falls from 0.105 to 0.010 ms, 10 times faster and within
0.003 ms of the bare FFT pair. The progress table shows the cost of going back
to the host: reading the latest error after every iteration costs 0.0655 ms per
iteration, every 10 iterations 0.0148 ms, every 100 iterations 0.0108 ms. At
256^3, where a single iteration takes 4 ms, launch and sync overhead was already
negligible and the graph variant gains only 2% over fused.

## Step 4: batched multi-start

`Variant::batched` is the equivalent of QoraPhase's `vmap`: all starts (or
`Options::batch` of them) share one `cufftPlanMany` with a batch count, and every
element-wise kernel takes the start index from `blockIdx.y`. Shrinkwrap reduces
its threshold level per start in the same launch. The tests check that batched
and looped starts give the same objects, errors and supports.

It helps only where the GPU is under-occupied:

- 256^2: 8 starts in one plan take 0.004 ms per start against 0.011 ms looped, 2.8
  times faster. Each 256^2 transform alone cannot fill the GPU.
- 1024^2, 2048^2, 128^3: batching is 1.8 to 3.1 times *slower*. Looped, one
  start's working set fits in the 48 MiB L2 and every pass is served from cache;
  batched, eight starts' arrays (64 MB of x alone at 1024^2) do not, and the same
  passes go to DRAM. Two starts per plan at 1024^2 (16 MB of x) still fit and
  are slightly faster than the loop (0.057 vs 0.061 ms).
- 256^3: no difference (4.079 vs 4.156 ms); a single start already saturates
  DRAM.

So `Options::batch` exists to size the batch to the cache, the same knob as
QoraPhase's `batch=` but for speed rather than memory.

## Step 5: fp16 storage

`Variant::half` keeps the object x and the wave in fp16 complex (`__half2`) and
runs cuFFT in fp16 (`cufftXtMakePlanMany` with `CUDA_C_16F`, power-of-two
extents only). All arithmetic stays in fp32. fp16 tops out at 65504, so the
amplitude is divided by its maximum, which bounds the unnormalised forward
transform by sqrt(n), and the inverse's 1/sqrt(n) is applied in the update
instead of before the inverse transform, where it would push weak pixels into
fp16 subnormals.

Speed depends entirely on how well cuFFT handles fp16 for the shape:

- 256^3: the fp16 FFT pair takes 1.421 ms against 2.764 ms, and the iteration
  2.296 ms against 4.108 ms: 1.8 times faster.
- 1024^2 and 2048^2: the fp16 FFT pair is about twice as *slow* as fp32 (0.100
  vs 0.043 ms, 0.367 vs 0.184 ms), and the variant is no faster or slower than
  graph.

Accuracy, measured with the known support (HIO 400 + ER 100, best of 8 starts)
and with the default recipe from an autocorrelation support:

| object | fp32 Fourier error | fp16 Fourier error | fp32 rel. object error | fp16 rel. object error |
|---|---:|---:|---:|---:|
| 64^2, known support | 2.3e-7 | 8.1e-4 | 2.1e-7 | 7.9e-4 |
| 256^2, known support | 2.3e-7 | 1.3e-3 | 1.9e-7 | 1.3e-3 |
| 64^3, known support | 2.9e-7 | 1.1e-3 | 5.1e-6 | 6.2e-3 |

fp16 holds about three significant digits, and the iteration stalls there:
around 1e-3 in both the Fourier error and the object, instead of float
precision. That is comparable to the noise floor of many measured patterns, but
it is a floor, not a slower convergence; the test `CudaHalf.RecoversToItsPrecisionFloor`
pins it. Mixed use (fp16 for the early, exploratory iterations, fp32 for the
final ER) would keep the speed without the floor; it is not implemented.

## Not done

- **cuFFT callbacks.** Fusing the update into the inverse FFT's store callback
  would remove the wave's write and re-read (about 16 of 77 bytes per element).
  On Windows only LTO callbacks (`cufftXtSetJITCallback`) exist, and they need
  nvJitLink, which the toolkit on this machine does not have. Not attempted.
- **R2C transforms.** The object is complex, so there is no Hermitian symmetry to
  exploit in the iteration. Only the shrinkwrap blur of |x| is real and could use
  R2C; it runs once per 50 iterations in the default recipe.
- **Bit-packed support.** The support is 1 of the 45 element-wise bytes, so
  packing it into bits would save under 2%.

## Against QoraPhase (JAX)

`bench/compare_qoraphase.py` runs HIO through both Python APIs on the same input
and takes the difference between n and 2n iterations, so compilation, setup and
transfers drop out. QoraPhase at commit bde11f4. JAX publishes CUDA wheels for
Linux only, so on this Windows machine JAX ran on the CPU; there is no JAX GPU
number to compare against.

```
JAX 0.12.0 (cpu, cpu) | Python 3.14.5 | AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD
```

| size | QoraPhase JAX (CPU) | QoraPhaseCU cpu | QoraPhaseCU cuda (batched variant, 1 start) |
|---|---:|---:|---:|
| 256^2 | 0.539 | 0.771 | 0.009 |
| 1024^2 | 7.411 | 13.893 | 0.057 |
| 128^3 | 21.617 | 24.478 | 0.111 |

ms per HIO iteration. The CPU reference is slower than XLA on the CPU: its
element-wise loops are single-threaded and not vectorised, by design, since it
exists to check the GPU path. The CUDA backend is 60 to 195 times faster than
JAX on this CPU; against JAX on a GPU the gap would be far smaller, and that was
not measured.
