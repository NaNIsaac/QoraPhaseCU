"""Against the QoraPhase release (``uv sync --group qoraphase``); skipped without it.

The two libraries draw their random starts differently, so random-start runs are
compared by what they converge to. From a shared ``init`` they are compared
iteration by iteration, but only over a few iterations: HIO and RAAR amplify
float32 rounding differences to 1e-2 or more within about 50 iterations.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import qoraphasecu as qpc

qp = pytest.importorskip("qoraphase")

THRESHOLD = 1e-3


def test_default_recipe_converts_to_ours():
    assert qpc.from_qoraphase(qp.default_recipe()).ops() == qpc.default_recipe().ops()


def test_parameters_and_nesting_survive_conversion():
    theirs = qp.Recipe(
        qp.HIO(5, beta=0.7),
        qp.detwin(qp.RAAR(3, beta=0.8), axis=1),
        qp.repeat(2, qp.ER(4), qp.Shrinkwrap(threshold=0.25, sigma=2.0, method="rms")),
    )
    ours = qpc.Recipe(
        qpc.HIO(5, beta=0.7),
        qpc.detwin(qpc.RAAR(3, beta=0.8), axis=1),
        qpc.repeat(2, qpc.ER(4), qpc.Shrinkwrap(threshold=0.25, sigma=2.0, method="rms")),
    )
    assert qpc.from_qoraphase(theirs).ops() == ours.ops()
    assert qpc.from_qoraphase(qp.ER(7)).ops() == qpc.Recipe(qpc.ER(7)).ops()


@dataclass(frozen=True)
class DampedER(qp.ER):
    def step(self, state, problem):
        new, err = super().step(state, problem)
        return new._replace(obj=0.5 * (new.obj + state.obj)), err


def test_stages_without_a_counterpart_are_rejected(triangle):
    _, support, intensity = triangle
    with pytest.raises(TypeError, match="DampedER has no QoraPhaseCU counterpart"):
        qpc.from_qoraphase(qp.Recipe(qp.HIO(5), qp.repeat(2, DampedER(3))))
    with pytest.raises(TypeError, match="no QoraPhaseCU counterpart"):
        qpc.reconstruct(intensity, support=support, recipe=DampedER(3))


def test_reconstruct_takes_their_recipe(triangle):
    _, support, intensity = triangle
    direct = qpc.reconstruct(intensity, support=support, recipe=qp.Recipe(qp.HIO(20), qp.ER(5)))
    converted = qpc.reconstruct(
        intensity, support=support, recipe=qpc.Recipe(qpc.HIO(20), qpc.ER(5))
    )
    np.testing.assert_array_equal(direct.errors, converted.errors)


@pytest.mark.parametrize("support_given", [True, False], ids=["support", "autocorrelation"])
def test_both_libraries_converge_to_the_same_object(triangle, support_given):
    obj, support, intensity = triangle
    support = support if support_given else None
    recipe = qp.Recipe(qp.HIO(300), qp.ER(100)) if support_given else qp.default_recipe()
    theirs = qp.reconstruct(intensity, support=support, recipe=recipe, starts=4).best
    ours = qpc.reconstruct(intensity, support=support, recipe=recipe, starts=4).best
    assert theirs.error < THRESHOLD and ours.error < THRESHOLD
    theirs_object = np.asarray(theirs.object)
    assert qpc.aligned_error(ours.object, theirs_object) < 1e-2
    assert qpc.aligned_error(ours.object, obj) < 1e-2


def test_results_plug_into_qoraphase(triangle):
    _, support, intensity = triangle
    ensemble = qpc.reconstruct(
        intensity, support=support, recipe=qpc.Recipe(qpc.HIO(300), qpc.ER(100)), starts=3
    )
    run = ensemble.best.to_qoraphase()
    assert isinstance(run, qp.Reconstruction)
    assert run.error == ensemble.best.error
    np.testing.assert_array_equal(np.asarray(run.object), ensemble.best.object)

    theirs = ensemble.to_qoraphase(qp.CDI(intensity, support=support))
    assert isinstance(theirs, qp.Ensemble) and theirs.ranking == ensemble.ranking
    average = theirs.average()
    assert average.metrics.shape == (1,) and average.error < THRESHOLD


@pytest.mark.parametrize(
    "stage", [qp.ER(10), qp.HIO(10, beta=0.7), qp.RAAR(10, beta=0.85)], ids=["ER", "HIO", "RAAR"]
)
def test_shared_start_follows_qoraphase_step_by_step(triangle, stage):
    import jax
    import jax.numpy as jnp
    from qoraphase.core.state import ReconState

    _, support, intensity = triangle
    rng = np.random.default_rng(1)
    phase = np.exp(2j * np.pi * rng.random(intensity.shape))
    init = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.sqrt(intensity) * phase), norm="ortho"))
    init = np.where(support, init, 0).astype(np.complex64)

    problem = qp.CDI(intensity, support=support)
    state = ReconState(
        obj=jnp.asarray(init),
        support=problem.support,
        iteration=jnp.int32(0),
        key=jax.random.key(0),
    )
    theirs, their_errors = qp.Recipe(stage).run(state, problem)
    ours = qpc.reconstruct(
        intensity, support=support, recipe=qp.Recipe(stage), init=init, backend="cpu"
    )
    theirs = np.asarray(theirs.obj)
    assert np.linalg.norm(ours.object - theirs) / np.linalg.norm(theirs) < 1e-4
    np.testing.assert_allclose(ours.errors, np.asarray(their_errors), atol=1e-5)
