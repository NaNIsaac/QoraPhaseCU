"""HIO time per iteration: QoraPhase (JAX) against QoraPhaseCU through Python.

Each engine runs HIO(n) and HIO(2n) on the same input after a warm-up call; the
difference divided by n is the per-iteration cost without setup, compilation or
transfers. Needs both packages importable:

    python bench/compare_qoraphase.py [--sizes 256x256,1024x1024,128x128x128]
"""

import argparse
import platform
import time

import jax
import numpy as np
import qoraphase as qp
import qoraphasecu as qpc


def problem(shape):
    rng = np.random.default_rng(0)
    intensity = rng.random(shape, dtype=np.float32)
    support = np.zeros(shape, bool)
    support[tuple(slice(n // 4, 3 * n // 4) for n in shape)] = True
    return intensity, support


def time_qoraphase(intensity, support, n):
    cdi = qp.CDI(intensity, support=support)

    def run(k):
        result = cdi.reconstruct(qp.Recipe(qp.HIO(k)))
        result.object.block_until_ready()

    run(n)
    run(2 * n)
    t0 = time.perf_counter()
    run(n)
    t1 = time.perf_counter()
    run(2 * n)
    t2 = time.perf_counter()
    return ((t2 - t1) - (t1 - t0)) / n * 1e3


def time_qoraphasecu(intensity, support, n, backend):
    def run(k):
        qpc.reconstruct(intensity, support=support, recipe=qpc.Recipe(qpc.HIO(k)), backend=backend)

    run(n)
    t0 = time.perf_counter()
    run(n)
    t1 = time.perf_counter()
    run(2 * n)
    t2 = time.perf_counter()
    return ((t2 - t1) - (t1 - t0)) / n * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="256x256,1024x1024,128x128x128")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    print(
        f"JAX {jax.__version__} ({jax.default_backend()}, {jax.devices()[0].device_kind}) | "
        f"Python {platform.python_version()} | {platform.processor()}\n"
    )
    backends = ["cpu"] + (["cuda"] if qpc.cuda_available() else [])
    header = " | ".join(f"QoraPhaseCU {b}" for b in backends)
    print(f"| size | QoraPhase JAX | {header} |")
    print("|---|---:|" + "---:|" * len(backends))
    for size in args.sizes.split(","):
        shape = tuple(int(d) for d in size.split("x"))
        intensity, support = problem(shape)
        n = args.iters
        jax_ms = time_qoraphase(intensity, support, n)
        ours = [
            time_qoraphasecu(intensity, support, n * (20 if b == "cuda" else 1), b)
            for b in backends
        ]
        cells = " | ".join(f"{ms:.3f}" for ms in ours)
        label = f"{shape[0]}^{len(shape)}"
        print(f"| {label} | {jax_ms:.3f} | {cells} |", flush=True)


if __name__ == "__main__":
    main()
