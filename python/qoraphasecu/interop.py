"""Conversion to and from QoraPhase objects (tested against ``qoraphase==0.1.0``).

``qoraphase`` is imported only inside these functions: it pulls in JAX, which
QoraPhaseCU does not otherwise need.
"""

import sys

from .recipe import ER, HIO, RAAR, Recipe, Shrinkwrap, _Detwin, _Repeat

SUPPORTED = "ER, HIO, RAAR, Shrinkwrap, Recipe, repeat and detwin"


def is_qoraphase(obj) -> bool:
    """True for QoraPhase stages, user subclasses included."""
    # A QoraPhase object can only exist once the package is imported, so this
    # never triggers the import itself.
    module = sys.modules.get("qoraphase.recipes.recipe")
    return module is not None and isinstance(obj, module.Stage)


def from_qoraphase(recipe) -> Recipe:
    """Convert a QoraPhase recipe, or a single QoraPhase stage, into a :class:`Recipe`.

    Raises ``TypeError`` for a stage without a QoraPhaseCU counterpart, which
    includes subclasses of the built-in stages: their ``step`` may differ.
    """
    import qoraphase as qp

    stages = recipe.stages if type(recipe) is qp.Recipe else (recipe,)
    return Recipe(*(_stage(stage, qp) for stage in stages))


def _stage(stage, qp):
    from qoraphase.recipes.recipe import Detwin, Repeat

    convert = {
        qp.ER: lambda s: ER(s.n),
        qp.HIO: lambda s: HIO(s.n, float(s.beta)),
        qp.RAAR: lambda s: RAAR(s.n, float(s.beta)),
        qp.Shrinkwrap: lambda s: Shrinkwrap(float(s.threshold), float(s.sigma), s.method),
        qp.Recipe: from_qoraphase,
        Repeat: lambda s: _Repeat(s.n, from_qoraphase(s.body)),
        Detwin: lambda s: _Detwin(from_qoraphase(s.body), s.axis),
    }.get(type(stage))
    if convert is None:
        name = f"{type(stage).__module__}.{type(stage).__qualname__}"
        raise TypeError(f"{name} has no QoraPhaseCU counterpart; supported stages: {SUPPORTED}")
    return convert(stage)
