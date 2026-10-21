"""QoraPhaseCU: C++/CUDA phase retrieval for CDI with the algorithms of QoraPhase.

Array in, reconstruction out::

    import qoraphasecu as qpc
    result = qpc.reconstruct(intensity, support=support)

All public arrays are center-origin, ``mask=True`` means an unusable pixel, and
``intensity`` is a measured intensity (not an amplitude).
"""

import os
import sys

# Python on Windows no longer searches PATH for the DLLs of an extension, so a
# CUDA build would not find cuFFT without this.
if sys.platform == "win32" and "CUDA_PATH" in os.environ:
    _bin = os.path.join(os.environ["CUDA_PATH"], "bin")
    if os.path.isdir(_bin):
        os.add_dll_directory(_bin)

from .api import (
    Ensemble,
    Reconstruction,
    aligned_error,
    autocorrelation_support,
    cuda_available,
    reconstruct,
)
from .interop import from_qoraphase
from .recipe import ER, HIO, RAAR, Recipe, Shrinkwrap, default_recipe, detwin, repeat

__all__ = [
    "ER",
    "HIO",
    "RAAR",
    "Ensemble",
    "Recipe",
    "Reconstruction",
    "Shrinkwrap",
    "aligned_error",
    "autocorrelation_support",
    "cuda_available",
    "default_recipe",
    "detwin",
    "from_qoraphase",
    "reconstruct",
    "repeat",
]
