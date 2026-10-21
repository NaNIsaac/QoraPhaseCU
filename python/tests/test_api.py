import numpy as np
import pytest
import qoraphasecu as qpc


def test_single_start_returns_a_reconstruction(triangle):
    obj, support, intensity = triangle
    result = qpc.reconstruct(
        intensity, support=support, recipe=qpc.Recipe(qpc.HIO(30), qpc.ER(20)), backend="cpu"
    )
    assert isinstance(result, qpc.Reconstruction)
    assert result.object.shape == obj.shape and result.object.dtype == np.complex64
    assert result.support.dtype == bool
    assert result.errors.shape == (50,)
    assert result.error == result.errors[-1]


def test_default_recipe_recovers_without_support(triangle):
    obj, _, intensity = triangle
    ensemble = qpc.reconstruct(intensity, starts=8, backend="cpu")
    assert isinstance(ensemble, qpc.Ensemble) and len(ensemble) == 8
    assert ensemble.best.error < 1e-3
    assert qpc.aligned_error(ensemble.best.object, obj) < 1e-2
    assert ensemble.ranking[0] == min(range(8), key=lambda i: ensemble[i].error)


def test_recipe_flattens_like_qoraphase():
    recipe = qpc.default_recipe()
    kinds = [op[0] for op in recipe.ops()]
    assert kinds.count("raar") == 1 + 1 + 8
    assert kinds.count("shrinkwrap") == 8
    assert kinds[1:4] == ["detwin_begin", "hio", "detwin_end"]


def test_support_shape_is_checked(triangle):
    _, _, intensity = triangle
    with pytest.raises(ValueError, match=r"support\.shape"):
        qpc.reconstruct(intensity, support=np.ones((3, 3), bool), backend="cpu")


def test_engine_errors_surface_as_value_errors(triangle):
    _, support, intensity = triangle
    with pytest.raises(ValueError):
        qpc.reconstruct(intensity, support=support, recipe=qpc.Recipe(qpc.ER(0)), backend="cpu")
    with pytest.raises(ValueError):
        qpc.reconstruct(intensity, support=support, starts=4, batch=3, backend="cpu")


def test_negative_intensity_warns(triangle):
    _, support, intensity = triangle
    noisy = intensity - 0.5 * intensity.max()
    with pytest.warns(UserWarning, match="negative"):
        qpc.reconstruct(noisy, support=support, recipe=qpc.Recipe(qpc.ER(1)), backend="cpu")


def test_autocorrelation_support_covers_the_object(triangle):
    _, support, intensity = triangle
    derived = qpc.autocorrelation_support(intensity)
    assert derived.dtype == bool and derived.sum() > support.sum()


@pytest.mark.skipif(not qpc.cuda_available(), reason="no CUDA device or CPU-only build")
@pytest.mark.parametrize("variant", ["naive", "fused", "graph", "batched"])
def test_cuda_matches_cpu(triangle, variant):
    _, support, intensity = triangle
    recipe = qpc.Recipe(qpc.HIO(20), qpc.Shrinkwrap(), qpc.ER(10))
    cpu = qpc.reconstruct(intensity, support=support, recipe=recipe, starts=2, backend="cpu")
    gpu = qpc.reconstruct(
        intensity, support=support, recipe=recipe, starts=2, backend="cuda", variant=variant
    )
    for a, b in zip(cpu, gpu, strict=True):
        np.testing.assert_allclose(b.errors, a.errors, rtol=1e-3)
        assert np.linalg.norm(b.object - a.object) < 1e-3 * np.linalg.norm(a.object)
