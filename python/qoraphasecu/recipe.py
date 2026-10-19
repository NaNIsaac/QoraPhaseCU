"""Recipe stages with the names and defaults of QoraPhase.

A recipe flattens to a list of operation tuples for the C++ engine; ``repeat``
and ``detwin`` expand in place there, as in the C++ API.
"""

from dataclasses import dataclass
from typing import Literal


def _op(kind, n=0, beta=0.0, threshold=0.0, sigma=0.0, method="max", axis=0):
    return (kind, n, beta, threshold, sigma, method, axis)


@dataclass(frozen=True)
class ER:
    """``x <- P_S P_M x``."""

    n: int

    def ops(self):
        return [_op("er", self.n)]


@dataclass(frozen=True)
class HIO:
    """``x <- P_M x`` inside the support, ``x - beta P_M x`` outside."""

    n: int
    beta: float = 0.9

    def ops(self):
        return [_op("hio", self.n, self.beta)]


@dataclass(frozen=True)
class RAAR:
    """``x <- (beta/2)(R_S R_M x + x) + (1 - beta) P_M x``."""

    n: int
    beta: float = 0.9

    def ops(self):
        return [_op("raar", self.n, self.beta)]


@dataclass(frozen=True)
class Shrinkwrap:
    """Threshold the Gaussian-blurred magnitude; ``method`` is ``"max"`` or ``"rms"``."""

    threshold: float = 0.3
    sigma: float = 1.0
    method: Literal["max", "rms"] = "max"

    def __post_init__(self):
        if self.method not in ("max", "rms"):
            raise ValueError(f"method must be 'max' or 'rms', got {self.method!r}")

    def ops(self):
        return [_op("shrinkwrap", threshold=self.threshold, sigma=self.sigma, method=self.method)]


class Recipe:
    """Ordered sequence of stages. Code order is execution order."""

    def __init__(self, *stages):
        self.stages = tuple(stages)

    def __len__(self):
        return len(self.stages)

    def ops(self):
        return [op for stage in self.stages for op in stage.ops()]


@dataclass(frozen=True)
class _Repeat:
    n: int
    body: Recipe

    def ops(self):
        return self.body.ops() * self.n


@dataclass(frozen=True)
class _Detwin:
    body: Recipe
    axis: int

    def ops(self):
        return [_op("detwin_begin", axis=self.axis), *self.body.ops(), _op("detwin_end")]


def repeat(n: int, *stages) -> _Repeat:
    """``repeat(10, RAAR(50), Shrinkwrap())``"""
    return _Repeat(n, Recipe(*stages))


def detwin(*stages, axis: int = 0) -> _Detwin:
    """Run *stages* against the support cut at its centre of mass, then restore it."""
    return _Detwin(Recipe(*stages), axis)


def default_recipe() -> Recipe:
    """RAAR, a detwin pass, then shrinkwrapped RAAR and ER: QoraPhase's default."""
    return Recipe(
        RAAR(50),
        detwin(HIO(20)),
        RAAR(50),
        repeat(8, RAAR(50), Shrinkwrap()),
        ER(200),
    )
