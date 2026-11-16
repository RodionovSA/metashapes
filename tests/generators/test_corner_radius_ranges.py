# tests/generators/test_corner_radius_ranges.py
#
# A corner-radius range that fits only some draws (the valid range depends
# on the rest of the shape) must make the sampler redraw, not raise
# "incompatible ranges".

import random

import pytest

from metashapes.generators.random import RandomGeneratorConfig
from metashapes.generators.registry import SHAPE_SAMPLER_REGISTRY
from metashapes.generators.samplers.utils import resolve_radius
from metashapes.lattice.basis import Lattice

LATTICE = Lattice.rectangular(3.0, 3.0)

# (shape, param, range that only larger draws can satisfy)
CASES = [
    ("Rectangle", "corner_radius", (0.5, 0.7)),
    ("IsoscelesTrapezoid", "corner_radius", (0.3, 0.4)),
    ("ConvexQuad", "corner_radius", (0.2, 0.3)),
    ("RegularPolygon", "corner_radius", (0.4, 0.6)),
    ("Triangle", "corner_radius", (0.1, 0.15)),
    ("Star", "outer_corner_radius", (0.15, 0.25)),
    ("Star", "inner_corner_radius", (0.3, 0.5)),
    ("Cross", "outer_corner_radius", (0.3, 0.4)),
    ("Cross", "inner_corner_radius", (0.3, 0.4)),
    ("TShape", "outer_corner_radius", (0.3, 0.4)),
    ("TShape", "inner_corner_radius", (0.3, 0.4)),
]


def _sample(name, cfg, n, seed=0):
    rng = random.Random(seed)
    sampler = SHAPE_SAMPLER_REGISTRY[name]
    return [sampler.sample(rng, LATTICE, cfg) for _ in range(n)]


@pytest.mark.parametrize("name,param,rng_range", CASES)
def test_partial_range_redraws(name, param, rng_range):
    cfg = RandomGeneratorConfig(shape_param_ranges={name: {param: rng_range}},
                                max_tries_per_shape=2000)
    for s in _sample(name, cfg, n=30):
        v = getattr(s, param).item()
        assert rng_range[0] - 1e-5 <= v <= rng_range[1] + 1e-5


@pytest.mark.parametrize("name,param", [("Rectangle", "corner_radius"),
                                        ("Star", "outer_corner_radius"),
                                        ("Cross", "inner_corner_radius")])
def test_impossible_range_exhausts_tries(name, param):
    cfg = RandomGeneratorConfig(shape_param_ranges={name: {param: (50.0, 60.0)}},
                                max_tries_per_shape=50)
    with pytest.raises(RuntimeError, match="too_large_for_canvas"):
        _sample(name, cfg, n=1)


def test_resolve_radius():
    rng = random.Random(0)
    assert resolve_radius(rng, fixed_value=0.3, user_range=(5, 6), default_range=(0, 1)) == 0.3
    assert resolve_radius(rng, fixed_value=None, user_range=(2, 3), default_range=(0, 1)) is None
    assert 0.5 <= resolve_radius(rng, fixed_value=None, user_range=(0.5, 3),
                                 default_range=(0, 1)) <= 1
    with pytest.raises(ValueError, match="invalid range"):
        resolve_radius(rng, fixed_value=None, user_range=(3, 2), default_range=(0, 1))
