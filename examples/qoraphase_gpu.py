"""A QoraPhase recipe run by QoraPhase (JAX) and by QoraPhaseCU (CUDA), timed end to end.

The recipe is written once with ``qoraphase`` and handed to both libraries;
QoraPhaseCU's ensemble then goes back into ``qoraphase.Ensemble.average``.
Needs the CUDA build of qoraphasecu and the QoraPhase release:

    uv pip install --group qoraphase
    python examples/qoraphase_gpu.py [--sizes 256x256,128x128x128] [--starts 4]
"""

import argparse
import platform
import time
from importlib.metadata import version

import jax
import numpy as np
import qoraphase as qp
import qoraphasecu as qpc


def synthetic(shape):
    """A corner-cut block (not centrosymmetric) with an amplitude dip and a phase bump."""
    m = shape[0] * 3 // 8
    r = np.indices((m,) * len(shape)) - (m - 1) / 2
    r2 = (r**2).sum(0)
    bump = ((r + m / 6) ** 2).sum(0)
    block = (1 - 0.5 * np.exp(-r2 / m)) * np.exp(1j * 1.2 * np.exp(-bump / (1.5 * m)))
    block *= np.indices(block.shape).sum(0) < m * len(shape) / 2
    obj = np.zeros(shape, np.complex64)
    obj[tuple(slice(n // 2 - m // 2, n // 2 - m // 2 + m) for n in shape)] = block
    far = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))
    return obj, np.abs(obj) > 0, (np.abs(far) ** 2).astype(np.float32)


def best_time(run, repeat):
    result = run()  # compilation, plan creation, first transfers
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = run()
        best = min(best, time.perf_counter() - t0)
    return best, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="256x256,128x128x128")
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if not qpc.cuda_available():
        raise SystemExit("needs the CUDA build of qoraphasecu and a CUDA device")

    print(
        f"qoraphase {version('qoraphase')} on JAX "
        f"{jax.__version__} ({jax.default_backend()}) | Python {platform.python_version()} | "
        f"{platform.processor()}"
    )
    recipe = qp.default_recipe()
    iterations = qpc.from_qoraphase(recipe).ops()
    print(
        f"recipe: qp.default_recipe(), {sum(op[1] for op in iterations)} iterations, "
        f"{args.starts} starts, known support, best of {args.repeat} after a warm-up\n"
    )
    print(
        "| size | QoraPhase JAX s | QoraPhaseCU CUDA s | speed-up | error JAX | error CUDA "
        "| CUDA vs JAX | averaged error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for size in args.sizes.split(","):
        shape = tuple(int(d) for d in size.split("x"))
        obj, support, intensity = synthetic(shape)
        cdi = qp.CDI(intensity, support=support)

        def theirs(cdi=cdi):
            ensemble = cdi.reconstruct(recipe, starts=args.starts)
            ensemble.best.object.block_until_ready()
            return ensemble

        def ours(intensity=intensity, support=support):
            return qpc.reconstruct(intensity, support=support, recipe=recipe, starts=args.starts)

        jax_s, jax_result = best_time(theirs, args.repeat)
        cuda_s, cuda_result = best_time(ours, args.repeat)
        average = cuda_result.to_qoraphase(cdi).average()
        agreement = qpc.aligned_error(cuda_result.best.object, np.asarray(jax_result.best.object))
        label = f"{shape[0]}^{len(shape)}"
        print(
            f"| {label} | {jax_s:.2f} | {cuda_s:.3f} | {jax_s / cuda_s:.0f}x "
            f"| {jax_result.best.error:.1e} | {cuda_result.best.error:.1e} | {agreement:.1e} "
            f"| {average.error:.1e} |",
            flush=True,
        )
        truth = qpc.aligned_error(cuda_result.best.object, obj)
        print(f"  CUDA best vs true object: {truth:.1e}")


if __name__ == "__main__":
    main()
