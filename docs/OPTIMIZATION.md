# Optimization log

Every step of the CUDA ladder stays a selectable `qpcu::Variant`, so the
benchmark compares them on the same build. All numbers below come from
`qpcu_bench --iters 200 --starts 8` (bench/bench.cpp) on one machine:

```
GPU: NVIDIA GeForce RTX 5070 (sm_120, 11.9 GiB, L2 48 MiB), peak DRAM bandwidth 672 GB/s
CPU: AMD Ryzen 5 7500F 6-Core Processor (6 cores, 12 threads)
Windows 11, CUDA 13.4 (cuFFT 12.4, nvJitLink and NVRTC 13.4), MSVC 14.44, Release build
```

The peak is `2 x memory clock x bus width` from the device attributes. CUDA 13
removed `memoryClockRate` from `cudaDeviceProp`, so the clock comes from
`cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate)` and the bus width from
`cudaDeviceProp::memoryBusWidth` (see `cuda_device_info`).

The workload is HIO on random data with a box support covering the central half
of each axis; timing does not depend on the data. Times are milliseconds per
iteration, after a warm-up run of the same length so that plans and CUDA graphs
already exist. Another project's benchmark shares this GPU and CPU; one full run
that overlapped with it was discarded (its CPU column read twice the times
below), and every run reported here was taken with no other benchmark process
running.

**Bandwidth is modeled, not measured.** Nsight Compute 2026.3.1 is installed,
but `ncu` stops with `ERR_NVGPUCTRPERM`: on this Windows machine the GPU
performance counters are restricted to administrators, and that setting was not
changed. So instead of measured DRAM throughput, bytes per element per iteration are
counted from the loads and stores each element-wise kernel issues, plus one read
and one write of the array per FFT (16 bytes per element for complex64). cuFFT
takes more passes than that for large transforms, so the FFT part of the model
is a lower bound.

| variant | modeled bytes / element / iteration |
|---|---:|
| naive | 32 (FFTs) + 91 (five element-wise kernels) = 123 |
| fused, graph, batched | 32 (FFTs) + 45 (two element-wise kernels) = 77 |
| half | 16 (fp16 FFTs) + 25 (two element-wise kernels) = 41 |
| callback | 32 (FFTs) + 13 (amplitude, support and old x read in the callbacks) = 45 |
| mixed, HIO/RAAR iterations | 16 (fp16 FFTs) + 12 (modulus) + 25 (update into fp32 x, fp16 copy) = 53 |

## Results

Single start. ms per iteration, modeled GB/s in parentheses.

| size | cuFFT pair | fp16 cuFFT pair | cpu | naive | fused | graph | half | callback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256^2 | 0.008 (279) | 0.006 (163) | 0.52 | 0.135 (60) | 0.102 (50) | 0.011 (477) | 0.010 (282) | 0.019 (159) |
| 1024^2 | 0.043 (778) | 0.103 (162) | 7.23 | 0.194 (666) | 0.139 (579) | 0.059 (1376) | 0.111 (389) | 0.070 (672) |
| 2048^2 | 0.180 (746) | 0.370 (181) | - | 0.742 (695) | 0.527 (613) | 0.452 (714) | 0.448 (384) | 0.519 (363) |
| 128^3 | 0.081 (824) | 0.060 (562) | 16.24 | 0.287 (900) | 0.191 (845) | 0.113 (1430) | 0.084 (1028) | 0.143 (659) |
| 256^3 | 2.766 (194) | 1.417 (189) | - | 5.581 (370) | 4.184 (309) | 4.115 (314) | 2.299 (299) | 3.251 (232) |

"cuFFT pair" is one forward and one inverse in-place C2C transform replayed from
a CUDA graph: the floor for any variant that keeps cuFFT. The CPU column is the
reference backend (pocketfft on all threads, element-wise loops on OpenMP
threads), skipped above 2^21 elements.

Eight starts. ms per iteration per start.

| size | graph, looped starts | batched, all 8 in one plan | batched, 2 per plan |
|---|---:|---:|---:|
| 256^2 | 0.010 | 0.004 | 0.007 |
| 1024^2 | 0.062 | 0.191 | 0.056 |
| 2048^2 | 0.446 | 0.830 | 0.817 |
| 128^3 | 0.119 | 0.270 | 0.190 |
| 256^3 | 4.109 | 4.164 | 4.167 |

Progress reporting (graph variant, 256^2): a host read-back every N iterations.

| N | ms per iteration |
|---:|---:|
| 1 | 0.0603 |
| 10 | 0.0152 |
| 100 | 0.0106 |

Repeated runs reproduce the GPU rows to within a few percent, except the
host-synchronised ones at 256^2, which move between runs (fused at 256^2 has
read 0.102, 0.105 and 0.178 ms; the N = 10 progress row once read 0.059 ms).
Those rows are bounded by Windows scheduling of the host-device round trip, not
by the GPU. The CPU column times only 10 iterations and moves by about 10%.

## Where the time goes

Subtracting the cuFFT pair from a variant leaves the time spent in its own
kernels and launches. At 256^3 the arrays (134 MB each) are far larger than the
48 MiB L2, so this is a clean DRAM measurement:

| 256^3 | total | minus cuFFT pair | element-wise bytes | effective bandwidth |
|---|---:|---:|---:|---:|
| naive | 5.581 | 2.815 | 91 x 16.8M = 1.53 GB | 542 GB/s (81% of peak) |
| fused | 4.184 | 1.418 | 45 x 16.8M = 0.75 GB | 532 GB/s (79%) |
| graph | 4.115 | 1.349 | 0.75 GB | 560 GB/s (83%) |
| half | 2.299 | 0.882 (minus the fp16 pair) | 25 x 16.8M = 0.42 GB | 476 GB/s (71%) |

The element-wise kernels already run near the bandwidth limit even in the naive
variant; what changes between steps is how many bytes they have to move and how
much of the iteration is spent waiting on the host. Once the element-wise part
is at 83% of peak, cuFFT is two thirds of the iteration at 256^3, and the
remaining levers are the FFT's precision (step 5) or folding the element-wise
work into the FFT's own passes (step 6).

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
FFTs. At 256^2 an iteration costs 0.135 ms, 17 times the 0.008 ms the two FFTs
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
1.40 ms less per iteration (5.581 to 4.184 ms, 25%), which is what moving 0.77 GB
less at about 540 GB/s predicts. At 256^2 it barely matters (0.135 to 0.102 ms)
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

At 256^2 the iteration falls from 0.102 to 0.011 ms, 9 times faster and within
0.003 ms of the bare FFT pair. The progress table shows the cost of going back
to the host: reading the latest error after every iteration costs 0.0603 ms per
iteration, every 10 iterations 0.0152 ms, every 100 iterations 0.0106 ms. At
256^3, where a single iteration takes 4 ms, launch and sync overhead was already
negligible and the graph variant gains only 2% over fused.

## Step 4: batched multi-start

`Variant::batched` is the equivalent of QoraPhase's `vmap`: all starts (or
`Options::batch` of them) share one `cufftPlanMany` with a batch count, and every
element-wise kernel takes the start index from `blockIdx.y`. Shrinkwrap reduces
its threshold level per start in the same launch. The tests check that batched
and looped starts give the same objects, errors and supports.

It helps only where the GPU is under-occupied:

- 256^2: 8 starts in one plan take 0.004 ms per start against 0.010 ms looped, 2.5
  times faster. Each 256^2 transform alone cannot fill the GPU.
- 1024^2, 2048^2, 128^3: batching is 1.9 to 3.1 times *slower*. Looped, one
  start's working set fits in the 48 MiB L2 and every pass is served from cache;
  batched, eight starts' arrays (64 MB of x alone at 1024^2) do not, and the same
  passes go to DRAM. Two starts per plan at 1024^2 (16 MB of x) still fit and
  are slightly faster than the loop (0.056 vs 0.062 ms).
- 256^3: no difference (4.109 vs 4.164 ms); a single start already saturates
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

- 256^3: the fp16 FFT pair takes 1.417 ms against 2.766 ms, and the iteration
  2.299 ms against 4.115 ms: 1.8 times faster.
- 1024^2 and 2048^2: the fp16 FFT pair is about twice as *slow* as fp32 (0.103
  vs 0.043 ms, 0.370 vs 0.180 ms), and the variant is slower than graph at
  1024^2 (0.111 vs 0.059 ms) and level at 2048^2.

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
pins it. Step 7 keeps the fp16 transforms for the exploratory iterations and
moves x and the final ER back to fp32.

## Step 6: cuFFT LTO callbacks

`Variant::callback` moves the element-wise work into cuFFT's own passes. cuFFT
can call a device function on every element a transform stores. On Windows only
the LTO form of these callbacks exists: src/cuda/callbacks.cu is compiled to
LTO-IR (`-gencode arch=compute_XX,code=lto_XX`, cmake/cufft_callbacks.cmake),
embedded in the library as a byte array, and handed to `cufftXtSetJITCallback`
before the plan is made. cuFFT then generates its kernels with NVRTC and links
the callbacks in with nvJitLink, so both libraries must be present at run time;
without NVRTC, plan creation fails with `CUFFT_MISSING_DEPENDENCY`. The
callbacks must have C++ linkage: declared `extern "C"`, they make plan creation
fail with `CUFFT_NVJITLINK_FAILURE`. CMake leaves the variant out when the
toolkit's cufftXt.h has no `cufftXtSetJITCallback` (`QORAPHASECU_CUFFT_LTO`).

A callback belongs to a plan, not to a direction, so the variant makes two more
plans and an iteration is three graph nodes:

1. forward FFT with a store callback that applies the modulus projection to each
   element as cuFFT writes it, and adds the element's error term with an
   `atomicAdd` into one of 4096 float slots (slot = offset mod 4096);
2. a one-block kernel that sums the slots into the error history and clears them;
3. inverse FFT with a store callback that writes the updated object: P_M x
   inside the support, a x + b P_M x outside. It writes one half of a double
   buffer and reads the old x from the other half, since cuFFT does not promise
   to leave its output buffer untouched before the final store. The halves swap
   every iteration, and the CUDA graphs are cached per starting half.

The update coefficients sit in device memory that the callback reads, so a
change of algorithm is one small copy rather than new plans. Modeled traffic is
45 bytes per element against 77 for `graph`.

| size | graph | callback | change |
|---|---:|---:|---:|
| 256^2 | 0.011 | 0.019 | +73% |
| 1024^2 | 0.059 | 0.070 | +19% |
| 2048^2 | 0.452 | 0.519 | +15% |
| 128^3 | 0.113 | 0.143 | +27% |
| 256^3 | 4.115 | 3.251 | -21% |

It wins only at 256^3, by 0.86 ms per iteration, in fp32. The fp16 variant is
faster still there (2.299 ms) but stops at its 1e-3 floor; the callback variant
keeps float precision and passes the same CPU comparison tests as `graph`.
Everywhere else it is slower than the separate kernels.

To see why, the callbacks were cut down one piece at a time in throw-away
builds, each timed in one run with `graph` and the plain cuFFT pair alongside
(ms per iteration):

| callbacks | 1024^2 | 2048^2 | 128^3 | 256^3 |
|---|---:|---:|---:|---:|
| full (as shipped) | 0.067 | 0.515 | 0.139 | 3.250 |
| without the error atomics | 0.063 | 0.514 | 0.137 | 3.236 |
| also without the update's loads (store P_M x) | 0.056 | 0.387 | 0.104 | 2.967 |
| copy only, plus the one-block kernel | 0.051 | 0.304 | 0.097 | 2.816 |
| plain cuFFT pair, no callbacks | 0.043 | 0.182 | 0.081 | 2.765 |
| modulus callback only, update as a kernel | 0.065 | 0.512 | 0.160 | 3.653 |

- **The error sum costs almost nothing** (0.014 ms at 256^3). A callback sees a
  single element, so there is no per-thread partial to reduce. The first
  version summed across the active lanes with cooperative groups and issued one
  atomic per warp into 256 slots; in two early runs, not checked for other load,
  it was slower (3.45 against 3.26 ms at 256^3) than plain atomics spread over
  4096 slots.
- **Callbacks slow cuFFT down by themselves.** Copy-only callbacks cost 0.05 ms
  over the plain pair at 256^3, but 0.016 ms (20%) at 128^3 and 0.12 ms (67%)
  at 2048^2.
- **The update's two loads are expensive at 2048^2.** The support byte and the
  old x (9 bytes per element, 38 MB) add 0.13 ms there, about 300 GB/s, against
  0.27 ms for 151 MB (560 GB/s) at 256^3. A store callback runs in the order in
  which cuFFT's last pass writes, and for the 2D transforms that order is
  apparently not contiguous across a warp, so the one-byte support loads waste
  most of each memory transaction. Without profiler counters this is the
  explanation that fits the numbers, not a measurement.
- **The L2 hides what the callbacks save.** At 1024^2, 2048^2 and 128^3 the
  separate kernels of `graph` read the wave back from the 48 MiB L2 (modeled
  1376 and 1430 GB/s at 1024^2 and 128^3), so removing their passes saves
  little, while the costs above are paid at every size. At 256^3 every pass goes
  to DRAM, the 32 bytes per element that the callbacks avoid are worth about
  0.8 ms, and the overhead is small.
- Fusing only the modulus is the worst of both: it pays the callback overhead
  on the forward plan and still runs the update kernel.

The linking happens when the plans are made. Solver construction, first solver
of each size in the process (ms):

| size | graph | callback |
|---|---:|---:|
| 256^2 | 2 | 413 |
| 1024^2 | 12 | 451 |
| 2048^2 | 41 | 542 |
| 128^3 | 22 | 346 |
| 256^3 | 134 | 249 |

Other runs of the same benchmark gave 159 to 362 ms and 89 to 118 ms for the
callback column, so this cost varies. At 256^3 it is paid back after 130 to 250
iterations; at the other sizes, where the variant is slower anyway, never.

## Step 7: mixed precision

`Variant::mixed` keeps the fast part of step 5 and drops its floor. HIO and RAAR
iterations use the fp16 wave and the fp16 transforms of `half`, but x stays in
fp32: the update reads the fp16 inverse transform, combines it with the fp32 x
in fp32, writes x back in fp32, and also writes an fp16 copy of x that the next
forward transform reads. ER stages, which is how a recipe ends, run the fp32
path of `graph` on the same x. Entering an fp16 stage refreshes the copy from x
once, since an ER stage or a shrinkwrap may have changed x. Shrinkwrap works on
the fp32 x.

The fp16 update moves 25 bytes per element against 13 for `half` (fp32 x read
and written, plus the fp16 copy), so an HIO iteration costs more than in
`half`. One run of `qpcu_bench --variants graph,half,mixed,callback`:

| size | graph | half | mixed |
|---|---:|---:|---:|
| 256^2 | 0.010 | 0.010 | 0.009 |
| 1024^2 | 0.060 | 0.117 | 0.118 |
| 2048^2 | 0.444 | 0.457 | 0.569 |
| 128^3 | 0.111 | 0.083 | 0.092 |
| 256^3 | 4.113 | 2.310 | 2.665 |

At 256^3 an HIO iteration is 1.54 times faster than `graph` (`half`: 1.78). At
1024^2 and 2048^2 the fp16 transforms are slow, as in step 5, and mixed is
slower than `graph`; there the variant has no reason to exist.

Whole recipes at 256^3, one start, a 96-voxel triangle with its known support,
second run of the recipe on the same solver (wall time):

| recipe at 256^3 | graph | half | mixed |
|---|---:|---:|---:|
| HIO 400 + ER 100 | 2.08 s | 1.16 s | 1.49 s |
| `default_recipe()` | 3.09 s | 1.78 s | 2.33 s |

Accuracy, best of 8 starts after HIO 400 + ER 100 with the known support (the
same objects as the step 5 table):

| object | variant | Fourier error | rel. object error |
|---|---|---:|---:|
| 64^2 | graph | 2.3e-7 | 2.1e-7 |
| 64^2 | half | 8.1e-4 | 7.9e-4 |
| 64^2 | mixed | 2.2e-7 | 2.0e-7 |
| 256^2 | graph | 2.3e-7 | 1.9e-7 |
| 256^2 | half | 1.3e-3 | 1.3e-3 |
| 256^2 | mixed | 2.2e-7 | 1.9e-7 |
| 64^3 | graph | 2.7e-7 | 4.7e-6 |
| 64^3 | half | 1.0e-3 | 6.4e-3 |
| 64^3 | mixed | 3.6e-7 | 2.1e-4 |

In 2D the 100 fp32 ER iterations bring mixed to the same error as `graph`. In
3D its Fourier error reaches float level too, but the object is 2.1e-4 from the
truth against 4.7e-6 for `graph`. With 400 ER iterations instead of 100 it
stays at 2.1e-4 (fp32: 1.8e-5), so this is not slow ER convergence: the fp16
stage ends near a slightly different object that fits the data to float
precision.

The difference is in the test object, not in the arithmetic. The 64^3 object is
the 2D triangle extruded over 24 slices with a linear phase along z, so it is
separable, f(x, y) g(z), and its intensity is |F|^2 |G|^2. Along z this is a 1D
phase retrieval problem for g, a box with a linear phase, and every zero of the
z-transform of such a profile lies on the unit circle. Moving one of those zeros
radially by eps changes |G|, up to a constant factor, only at second order: in
numpy, eps = 1e-4 changes g by 3.8e-4 and |G| by 5.5e-7 (eps = 1e-3: 3.8e-3 and
5.5e-5). Objects that differ by a factor depending on z alone therefore fit the
data to float precision. The fp16 iterations leave the object somewhere along
these flat directions, and ER, which follows the gradient of the Fourier error,
has almost nothing to follow there.

Two checks, best of 8 starts after HIO 400 + ER 100 with the known support (one
run of `reconstruct(variant=...)` through Python). "After the z factor" divides
the aligned result by the best factor that depends on z alone and measures what
is left. "Corner cut" removes the voxels with x + y + z >= 36 in the 24-voxel
box, and the 3D phase bump is a Gaussian of 0.8 rad:

| object | variant | Fourier error | rel. object error | after the z factor |
|---|---|---:|---:|---:|
| 64^3 prism (above) | graph | 2.7e-7 | 4.5e-6 | 1.9e-7 |
| 64^3 prism (above) | mixed | 3.3e-7 | 2.4e-4 | 1.9e-7 |
| 64^3 prism, corner cut, 3D phase bump | graph | 3.1e-7 | 2.8e-7 | 1.9e-7 |
| 64^3 prism, corner cut, 3D phase bump | mixed | 3.0e-7 | 3.0e-7 | 1.9e-7 |

All of the mixed error on the prism is a z-only factor; whatever is left is at
the level of `graph`. Once the corner cut makes the object non-separable, mixed
reaches the same object error as `graph`. The fp32 `graph` result on the prism
(4.5e-6) is also mostly the z factor; it is smaller only because the fp32
iterations wander less along the flat directions. The tests
`CudaMixed.ConvergesPastTheHalfFloor` (2D) and `ConvergesPastTheHalfFloor3D`
(the cut prism, object error below 1e-5) pin both cases.

## Not done

- **R2C transforms.** The object is complex, so there is no Hermitian symmetry to
  exploit in the iteration. Only the shrinkwrap blur of |x| is real and could use
  R2C; it runs once per 50 iterations in the default recipe.
- **Bit-packed support.** The support is 1 of the 45 element-wise bytes, so
  packing it into bits would save under 2%. In the callback variant, where the
  support load looks expensive at 2048^2, it might matter more; not tried.
- **Measured bandwidth.** See the note on Nsight Compute at the top.

## CPU reference and QoraPhase (JAX)

The CPU reference runs its element-wise loops (modulus with its error sum,
update, shrinkwrap, random start) under `#pragma omp parallel for`; pocketfft
already used all hardware threads for the transforms. MSVC implements OpenMP
2.0, which is enough for these loops with a signed index. The magnitude is
`sqrt(norm(z))`, the expression the CUDA kernels use, instead of `std::abs`,
which goes through `hypot`. Without OpenMP (`QORAPHASECU_OPENMP=OFF`, or a
compiler without it) the loops run on one thread.

`qpcu_bench --variants cpu`, single runs of 10 iterations, ms per iteration:

| size | loops on one thread (`OMP_NUM_THREADS=1`) | loops on OpenMP threads |
|---|---:|---:|
| 256^2 | 0.60 | 0.49 |
| 1024^2 | 9.77 | 7.66 |
| 2048^2 | 37.11 | 32.03 |
| 128^3 | 22.71 | 17.57 |
| 256^3 | 165.77 | 157.34 |

The gain is 5 to 23%: the transforms take most of the time. Setting the OpenMP
threads to the 6 physical cores, or `OMP_WAIT_POLICY=PASSIVE` so that idle
OpenMP threads do not spin while pocketfft's threads work, moved the times by
less than the run-to-run spread.

`bench/compare_qoraphase.py` runs HIO through both Python APIs on the same input
and takes the difference between n and 2n iterations, so compilation, setup and
transfers drop out. The best of five such differences is reported, because
other load on the machine only ever adds time; single differences moved by up
to a factor of 1.8 between runs. QoraPhase at commit bde11f4. JAX publishes
CUDA wheels for Linux only, so on this Windows machine JAX ran on the CPU; there
is no JAX GPU number to compare against.

```
JAX 0.12.0 (cpu, cpu) | Python 3.14.5 | AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD
```

| size | QoraPhase JAX (CPU) | QoraPhaseCU cpu, OpenMP | QoraPhaseCU cpu, `OMP_NUM_THREADS=1` | QoraPhaseCU cuda (batched variant, 1 start) |
|---|---:|---:|---:|---:|
| 256^2 | 0.406 | 0.439 | 0.487 | 0.009 |
| 1024^2 | 5.003 | 6.321 | 7.920 | 0.058 |
| 128^3 | 14.447 | 14.140 | 18.449 | 0.113 |

ms per HIO iteration. With its loops threaded, the CPU reference is 8% slower
than XLA at 256^2, 26% slower at 1024^2 and level at 128^3; with one-thread
loops it is 20 to 58% slower. The CUDA backend is 45 to 128 times faster than
JAX on this CPU; against JAX on a GPU the gap would be far smaller, and that was
not measured.
