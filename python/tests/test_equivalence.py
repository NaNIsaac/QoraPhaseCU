"""QoraPhaseCU against QoraPhase from the same starting object, iteration by iteration.

Skipped without QoraPhase (``uv sync --group qoraphase``). The reference is QoraPhase's JAX run.
When the installed QoraPhase ships ``qoraphase.testing`` (its NumPy complex128 reference), the
bound is ``FACTOR`` times the measured complex64 rounding floor of the case; otherwise it is the
fixed ``BOUND``, which is about ten times the largest distance measured between the two
libraries on these cases.

Runs are short on purpose: HIO and RAAR amplify float32 rounding differences to 1e-2 or more
within about 50 iterations, for any pair of correct single-precision implementations.

``BACKENDS`` lists the QoraPhaseCU backends to check; CUDA joins when a device is present.
"""

import numpy as np
import pytest
import qoraphasecu as qpc

qp = pytest.importorskip("qoraphase")
reference = getattr(qp, "testing", None)
if reference is None:
    try:
        import qoraphase.testing as reference
    except ImportError:
        reference = None

FACTOR = 10.0
BOUND = 1e-4
BACKENDS = ["cpu", *(["cuda"] if qpc.cuda_available() else [])]


def run_qoraphase(recipe, intensity, support, init, mask=None):
    import jax
    import jax.numpy as jnp
    from qoraphase.core.state import ReconState

    problem = qp.CDI(intensity, support=support, mask=mask)
    start = jnp.where(problem.support, jnp.asarray(init), 0)
    state = ReconState(start, problem.support, jnp.int32(0), jax.random.key(0))
    state, errors = jax.jit(lambda s, p: recipe.run(s, p))(state, problem)
    return np.asarray(state.obj), np.asarray(state.support), np.asarray(errors)


def case(shape, seed=0, masked=0.0):
    """An object inside a box support, its intensity, a random-phase start and a mask."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.arange(n) - n // 2 for n in shape], indexing="ij")
    inside = sum((g / (n / 4)) ** 2 for g, n in zip(grids, shape, strict=True)) < 1
    obj = np.where(inside, (1 + 0.3 * rng.random(shape)) * np.exp(1j * rng.random(shape)), 0)
    fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))
    intensity = (np.abs(fft) ** 2).astype(np.float32)
    support = np.zeros(shape, bool)
    support[tuple(slice(n // 4 - 1, 3 * n // 4 + 1) for n in shape)] = True
    phase = np.exp(2j * np.pi * rng.random(shape))
    init = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(np.sqrt(intensity) * phase), norm="ortho"))
    mask = rng.random(shape) < masked
    return intensity, support, init.astype(np.complex64), mask if masked else None


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def assert_follows_qoraphase(recipe, intensity, support, init, mask, backend):
    theirs, their_support, their_errors = run_qoraphase(recipe, intensity, support, init, mask)
    ours = qpc.reconstruct(
        intensity, support=support, mask=mask, recipe=recipe, init=init, backend=backend
    )
    bound = BOUND
    if reference is not None:
        floor = reference.floor(recipe, intensity, support, init, mask=mask)
        bound = FACTOR * floor + 1e-6
    distance = relative(ours.object, theirs)
    assert distance < bound, f"{distance:.2e} against {bound:.2e}"
    np.testing.assert_array_equal(ours.support, their_support)
    np.testing.assert_allclose(ours.errors, their_errors, atol=max(bound, 1e-5))


SHAPES = [(32, 32), (33, 31), (16, 16, 16), (15, 16, 17)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize(
    "stage", [qp.ER(10), qp.HIO(10, beta=0.7), qp.RAAR(10, beta=0.85)], ids=["ER", "HIO", "RAAR"]
)
def test_single_stages(backend, shape, stage):
    intensity, support, init, mask = case(shape)
    assert_follows_qoraphase(qp.Recipe(stage), intensity, support, init, mask, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_masked_pixels(backend):
    intensity, support, init, mask = case((32, 32), seed=1, masked=0.2)
    recipe = qp.Recipe(qp.HIO(5), qp.ER(5))
    assert_follows_qoraphase(recipe, intensity, support, init, mask, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_long_er(backend):
    """ER does not amplify rounding: 200 iterations stay on the reference."""
    intensity, support, init, mask = case((33, 31), seed=2)
    assert_follows_qoraphase(qp.Recipe(qp.ER(200)), intensity, support, init, mask, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["max", "rms"])
def test_support_updates_and_detwin(backend, method):
    intensity, support, init, mask = case((32, 32), seed=3)
    recipe = qp.Recipe(
        qp.ER(3),
        qp.detwin(qp.HIO(3), axis=1),
        qp.repeat(2, qp.RAAR(2), qp.Shrinkwrap(threshold=0.2, method=method)),
        qp.ER(3),
    )
    assert_follows_qoraphase(recipe, intensity, support, init, mask, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_global_phase_carries_through(backend):
    intensity, support, init, _ = case((32, 32), seed=4)
    recipe = qpc.Recipe(qpc.RAAR(5), qpc.ER(5))
    turn = np.exp(0.7j).astype(np.complex64)
    a = qpc.reconstruct(intensity, support=support, recipe=recipe, init=init, backend=backend)
    b = qpc.reconstruct(
        intensity, support=support, recipe=recipe, init=turn * init, backend=backend
    )
    assert relative(b.object, turn * a.object) < 1e-5
    np.testing.assert_allclose(b.errors, a.errors, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_twin_start_gives_the_twin(backend):
    """``conj(x(-r))`` has the same intensity; with the mirrored support the run mirrors."""

    def twin(a):
        flipped = a[(slice(None, None, -1),) * a.ndim]
        return np.roll(flipped, 1, axis=tuple(range(a.ndim)))

    intensity, support, init, _ = case((32, 32), seed=5)
    support[10, 9] = False  # so that the mirrored support differs
    recipe = qpc.Recipe(qpc.HIO(5), qpc.ER(5))
    a = qpc.reconstruct(intensity, support=support, recipe=recipe, init=init, backend=backend)
    b = qpc.reconstruct(
        intensity, support=twin(support), recipe=recipe, init=np.conj(twin(init)), backend=backend
    )
    assert relative(b.object, np.conj(twin(a.object))) < 1e-5
    np.testing.assert_allclose(b.errors, a.errors, atol=1e-6)
