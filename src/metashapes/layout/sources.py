# metashapes/layout/sources.py
# Atom sources: *what* shape goes at a layout site.

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

from metashapes.generators.config import GeneratorConfig
from metashapes.generators.registry import SHAPE_SAMPLER_REGISTRY
import metashapes.generators.samplers  # noqa: F401  (registers the samplers)
from metashapes.lattice.basis import Lattice
from metashapes.shape.base import Shape, is_empty_bounds
from metashapes.shape.primitives import Bar
from metashapes.shape.registry import SHAPE_REGISTRY
from metashapes.shape.transforms import Translate

__all__ = ["AtomSource", "SamplerSource", "ParametricSource", "Regions"]

# Primitives with infinite extent cannot be layout atoms.
_INFINITE = (Bar,)


class AtomSource(ABC):
    """Produces the atom for a site.

    ``atom(x, y, rng)`` returns a shape centred at the origin (the generator
    translates it to the site), or ``None`` to leave the site empty.
    Infeasible samples may raise ``RuntimeError``/``ValueError``; the
    generator records the message as a rejection reason.

    ``deterministic`` sources return the same atom for the same site, so
    the generator doesn't retry a rejected site with them.
    """
    deterministic: bool = False

    @abstractmethod
    def atom(self, x: float, y: float, rng) -> Shape | None:
        raise NotImplementedError


def _recentre(shape: Shape) -> Shape:
    b = shape.bounds()
    if is_empty_bounds(b):
        return shape
    (x0, y0), (x1, y1) = b
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        raise ValueError("infinite_extent")
    return Translate(shape, -0.5 * (x0 + x1), -0.5 * (y0 + y1))


class SamplerSource(AtomSource):
    """Random atoms from the existing per-primitive samplers.

    Each sampler is called with a small rectangular *reference box*
    (``atom_box``, the meta-atom scale) instead of the whole domain, so its
    default size ranges stay at atom scale. The result is recentred on its
    bbox centre. ``allowed_shapes``, ``shape_weights``,
    ``fixed_shape_params``, ``shape_param_ranges``, ``min_shape_size``,
    ``min_feature_size`` and ``max_tries_per_shape`` of ``atom_config`` are
    used exactly as by ``RandomUnitCellGenerator``. Its lattice and
    shape-count fields are ignored.

    ``Bar`` (infinite extent) is rejected; with an empty ``allowed_shapes``
    every registered finite primitive is used.
    """

    def __init__(self, atom_config: GeneratorConfig, atom_box: tuple[float, float]):
        w, h = float(atom_box[0]), float(atom_box[1])
        if not (w > 0 and h > 0):
            raise ValueError("atom_box must be positive")
        self.config = atom_config
        self.atom_box = (w, h)
        self.lattice = Lattice.rectangular(w, h)
        allowed = list(atom_config.allowed_shapes)
        for name in allowed:
            if name not in SHAPE_SAMPLER_REGISTRY:
                raise ValueError(f"No sampler registered for shape '{name}'")
            if issubclass(SHAPE_SAMPLER_REGISTRY[name].shape_class, _INFINITE):
                raise ValueError(f"'{name}' has infinite extent and cannot be a layout atom")
        if not allowed:
            allowed = [n for n, s in SHAPE_SAMPLER_REGISTRY.items()
                       if not issubclass(s.shape_class, _INFINITE)]
        self.names = allowed
        weights = [atom_config.shape_weights.get(n, 1.0) for n in allowed]
        if atom_config.shape_weights and all(wt == 0 for wt in weights):
            raise ValueError("At least one shape weight must be > 0")
        self.weights = weights if atom_config.shape_weights else None

    def atom(self, x, y, rng):
        if self.weights is None:
            name = rng.choice(self.names)
        else:
            name = rng.choices(self.names, weights=self.weights, k=1)[0]
        shape = SHAPE_SAMPLER_REGISTRY[name].sample(rng, self.lattice, self.config)
        return _recentre(shape)


class ParametricSource(AtomSource):
    """Field-driven atoms: shape parameters as functions of position.

    ``params`` maps constructor arguments to constants or to callables
    ``f(x, y)`` evaluated at the site, e.g.
    ``ParametricSource(Ellipse, {"axes": lambda x, y: (r(x, y),) * 2})``.
    If the shape takes a ``center`` and none is given, it is ``(0, 0)``, so
    the site lands on the shape's own centre.
    """

    def __init__(self, shape_cls: type[Shape] | str, params: dict[str, Any],
                 *, deterministic: bool = True):
        if isinstance(shape_cls, str):
            try:
                shape_cls = SHAPE_REGISTRY[shape_cls]
            except KeyError as e:
                raise ValueError(f"Unknown shape type: {shape_cls}") from e
        if issubclass(shape_cls, _INFINITE):
            raise ValueError(f"{shape_cls.__name__} has infinite extent and cannot be a layout atom")
        self.shape_cls = shape_cls
        self.params = dict(params)
        self.deterministic = bool(deterministic)
        sig = inspect.signature(shape_cls.__init__).parameters
        self._needs_center = "center" in sig and "center" not in self.params

    def atom(self, x, y, rng):
        kw = {k: (v(x, y) if callable(v) else v) for k, v in self.params.items()}
        if self._needs_center:
            kw["center"] = (0.0, 0.0)
        return self.shape_cls(**kw)


class Regions(AtomSource):
    """Different sources in different sub-areas of the domain.

    ``regions`` is a list of ``(region, source)``; the first region with
    ``region.sdf(x, y) <= 0`` decides. Sites in no region use ``default``,
    or stay empty if it is ``None``.
    """

    def __init__(self, regions: Sequence[tuple[Shape, AtomSource]],
                 default: AtomSource | None = None):
        self.regions = list(regions)
        self.default = default
        sources = [s for _, s in self.regions] + ([default] if default is not None else [])
        self.deterministic = all(s.deterministic for s in sources)

    def source_at(self, x: float, y: float) -> AtomSource | None:
        import torch
        for region, source in self.regions:
            with torch.no_grad():
                d = region.sdf(torch.tensor(float(x)), torch.tensor(float(y)))
            if d.item() <= 0:
                return source
        return self.default

    def atom(self, x, y, rng):
        source = self.source_at(x, y)
        return None if source is None else source.atom(x, y, rng)
