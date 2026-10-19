"""``reconstruct``: array in, reconstruction out, shaped like ``qoraphase.reconstruct``."""

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import _core
from .recipe import Recipe, default_recipe

NEGATIVE_WARN_FRACTION = 0.01


@dataclass(frozen=True)
class Reconstruction:
    """One reconstructed object, defined up to a global phase, a translation and a twin."""

    object: np.ndarray
    support: np.ndarray
    errors: np.ndarray

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.object)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.object)

    @property
    def error(self) -> float:
        """Final Fourier error; the ranking key."""
        return float(self.errors[-1]) if self.errors.size else float("nan")


@dataclass(frozen=True)
class Ensemble:
    """The runs of a multi-start reconstruction, not mutually aligned."""

    runs: tuple

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def __getitem__(self, index):
        return self.runs[index]

    @property
    def ranking(self) -> tuple:
        """Run indices ordered best (lowest error) first."""
        return tuple(sorted(range(len(self.runs)), key=lambda i: self.runs[i].error))

    @property
    def best(self) -> Reconstruction:
        return self.runs[self.ranking[0]]


def cuda_available() -> bool:
    return _core.cuda_available()


def autocorrelation_support(intensity, threshold: float = 0.1) -> np.ndarray:
    """Loose support from the thresholded autocorrelation; shrinkwrap tightens it."""
    intensity = np.ascontiguousarray(intensity, dtype=np.float32)
    return _core.autocorrelation_support(intensity, threshold).astype(bool)


def aligned_error(a, reference) -> float:
    """``||a - reference|| / ||reference||`` after removing shift, global phase and twin."""
    a = np.ascontiguousarray(a, dtype=np.complex64)
    reference = np.ascontiguousarray(reference, dtype=np.complex64)
    if a.shape != reference.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {reference.shape}")
    return _core.aligned_error(a, reference)


def _flag(name, array, shape):
    if array is None:
        return None
    array = np.ascontiguousarray(array, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name}.shape {array.shape} != intensity.shape {shape}")
    return array.view(np.uint8)


def reconstruct(
    intensity,
    *,
    support=None,
    mask=None,
    recipe: Recipe | None = None,
    starts: int = 1,
    batch: int | None = None,
    seed: int | None = None,
    support_threshold: float = 0.1,
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    variant: Literal["naive", "fused", "graph", "batched", "half"] = "batched",
    sync_every: int = 10,
) -> Reconstruction | Ensemble:
    """Array in, reconstruction out.

    Args:
        intensity: measured intensity, center-origin, 2D or 3D.
        support: boolean, ``True`` inside the object. ``None`` derives an
            autocorrelation support.
        mask: boolean, ``True`` marks *unusable* pixels.
        recipe: ``None`` uses :func:`default_recipe`.
        starts: number of random starts; >1 returns an :class:`Ensemble`.
        batch: starts per batched cuFFT plan (``variant="batched"``); ``None``
            puts all of them in one.
        seed: seed of the random starting phases; ``None`` means 0.
        backend: ``"auto"`` picks CUDA when a device is present.
        variant: CUDA optimisation step, see docs/OPTIMIZATION.md.
    """
    intensity = np.ascontiguousarray(intensity, dtype=np.float32)
    if intensity.ndim not in (2, 3):
        raise ValueError(f"intensity must be 2D or 3D, got ndim={intensity.ndim}")
    negative = float(np.mean(intensity < 0))
    if negative > NEGATIVE_WARN_FRACTION:
        warnings.warn(
            f"{negative:.1%} of intensity pixels are negative and will be clipped to 0; "
            "consider masking them instead",
            stacklevel=2,
        )
    if backend == "auto":
        backend = "cuda" if cuda_available() else "cpu"
    recipe = default_recipe() if recipe is None else recipe
    runs = _core.reconstruct(
        intensity,
        _flag("support", support, intensity.shape),
        _flag("mask", mask, intensity.shape),
        recipe.ops(),
        starts,
        0 if batch is None else batch,
        0 if seed is None else seed,
        support_threshold,
        backend,
        variant,
        sync_every,
    )
    results = tuple(Reconstruction(obj, sup.astype(bool), err) for obj, sup, err in runs)
    return results[0] if starts == 1 else Ensemble(results)
