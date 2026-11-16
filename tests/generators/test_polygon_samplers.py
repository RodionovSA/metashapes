# tests/generators/test_polygon_samplers.py
#
# Direct tests for TriangleSampler and StarSampler
# (generators/samplers/polygons.py), exercised without the generator's
# retry / validation loop so a sampler that emits bad shapes can't hide
# behind it.

import math
import random

import pytest

from metashapes.lattice.basis import Lattice
from metashapes.generators.random import RandomGeneratorConfig, RandomUnitCellGenerator
from metashapes.generators.registry import SHAPE_SAMPLER_REGISTRY
from metashapes.generators.samplers.polygons import TriangleSampler, StarSampler
from metashapes.generators.samplers.utils import lattice_inner_bounds
from metashapes.shape.primitives import Triangle, Star

N_SAMPLES = 300
TOL = 1e-5

LATTICES = {
    "square": Lattice.rectangular(3.0, 3.0),
    "wide": Lattice.rectangular(4.0, 1.5),
    "hexagonal": Lattice.hexagonal(3.0),
}

SAMPLERS = {
    "Triangle": (TriangleSampler, Triangle),
    "Star": (StarSampler, Star),
}


def _samples(sampler_cls, lattice, n=N_SAMPLES, seed=0, **cfg):
    config = RandomGeneratorConfig(**cfg)
    rng = random.Random(seed)
    sampler = sampler_cls()
    return [sampler.sample(rng, lattice, config) for _ in range(n)]


def _scalar(t):
    return t.detach().item()


def _inradius(tri):
    return tri._inradius().item()


# ---------------------------------------------------------------------------
# shared properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(SAMPLERS))
def test_registered_under_shape_name(name):
    sampler_cls, _ = SAMPLERS[name]
    assert isinstance(SHAPE_SAMPLER_REGISTRY[name], sampler_cls)


@pytest.mark.parametrize("name", list(SAMPLERS))
@pytest.mark.parametrize("lattice_name", list(LATTICES))
def test_never_raises_with_default_config(name, lattice_name):
    # every sample must be constructible -- the primitive's __init__
    # validation is stricter than a naive corner-radius bound
    sampler_cls, shape_cls = SAMPLERS[name]
    for s in _samples(sampler_cls, LATTICES[lattice_name], n=1000):
        assert type(s) is shape_cls


@pytest.mark.parametrize("name", list(SAMPLERS))
@pytest.mark.parametrize("lattice_name", list(LATTICES))
def test_shape_fits_inside_lattice_bounds(name, lattice_name):
    sampler_cls, _ = SAMPLERS[name]
    lattice = LATTICES[lattice_name]
    x0, y0, x1, y1 = lattice_inner_bounds(lattice)
    for s in _samples(sampler_cls, lattice):
        (bx0, by0), (bx1, by1) = s.bounds()
        assert bx0 >= x0 - TOL and by0 >= y0 - TOL
        assert bx1 <= x1 + TOL and by1 <= y1 + TOL


@pytest.mark.parametrize("name", list(SAMPLERS))
@pytest.mark.parametrize("lattice_name", list(LATTICES))
@pytest.mark.parametrize("min_feature", [0.1, 0.2])
def test_respects_min_feature_size(name, lattice_name, min_feature):
    sampler_cls, _ = SAMPLERS[name]
    for s in _samples(sampler_cls, LATTICES[lattice_name], min_feature_size=min_feature):
        assert s.min_feature_size >= min_feature - TOL


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_deterministic_with_seed(name):
    sampler_cls, _ = SAMPLERS[name]
    lattice = LATTICES["square"]
    a = _samples(sampler_cls, lattice, n=10, seed=123)
    b = _samples(sampler_cls, lattice, n=10, seed=123)
    for s1, s2 in zip(a, b):
        assert s1.bounds() == s2.bounds()


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_generator_emits_only_this_shape(name):
    _, shape_cls = SAMPLERS[name]
    config = RandomGeneratorConfig(
        target_count=5, seed=0, allowed_shapes=(name,), fixed_num_shapes=1,
        min_feature_size=0.1,
    )
    result = RandomUnitCellGenerator(config).generate(LATTICES["square"])
    assert result.report.generated == 5
    for cell in result.unit_cells:
        assert isinstance(cell.scene, shape_cls)


# ---------------------------------------------------------------------------
# Triangle-specific
# ---------------------------------------------------------------------------

class TestTriangleSampler:
    def test_params_valid(self):
        for s in _samples(TriangleSampler, LATTICES["square"]):
            alpha, beta = _scalar(s.alpha), _scalar(s.beta)
            assert _scalar(s.base) > 0
            assert alpha > 0 and beta > 0
            assert alpha + beta < 180.0
            assert 0.0 <= _scalar(s.corner_radius) < _inradius(s)

    def test_default_angle_ranges(self):
        for s in _samples(TriangleSampler, LATTICES["square"]):
            alpha, beta = _scalar(s.alpha), _scalar(s.beta)
            assert 20.0 - TOL <= alpha <= 140.0 + TOL
            assert 20.0 - TOL <= beta <= 140.0 + TOL
            # apex angle never below 20 degrees
            assert alpha + beta <= 160.0 + TOL

    def test_fixed_params_respected(self):
        cfg = dict(fixed_shape_params={
            "Triangle": {"base": 1.2, "alpha": 60.0, "beta": 60.0,
                         "angle": 15.0, "corner_radius": 0.05},
        })
        for s in _samples(TriangleSampler, LATTICES["square"], n=20, **cfg):
            assert _scalar(s.base) == pytest.approx(1.2)
            assert _scalar(s.alpha) == pytest.approx(60.0)
            assert _scalar(s.beta) == pytest.approx(60.0)
            assert _scalar(s.angle) == pytest.approx(15.0)
            assert _scalar(s.corner_radius) == pytest.approx(0.05)

    def test_param_ranges_respected(self):
        cfg = dict(shape_param_ranges={
            "Triangle": {"base": (0.8, 1.0), "alpha": (40.0, 50.0),
                         "beta": (70.0, 80.0), "angle": (0.0, 10.0)},
        })
        for s in _samples(TriangleSampler, LATTICES["square"], **cfg):
            assert 0.8 - TOL <= _scalar(s.base) <= 1.0 + TOL
            assert 40.0 - TOL <= _scalar(s.alpha) <= 50.0 + TOL
            assert 70.0 - TOL <= _scalar(s.beta) <= 80.0 + TOL
            assert 0.0 - TOL <= _scalar(s.angle) <= 10.0 + TOL

    def test_oversized_corner_radius_is_clamped_below_inradius(self):
        cfg = dict(fixed_shape_params={"Triangle": {"corner_radius": 10.0}})
        for s in _samples(TriangleSampler, LATTICES["square"], n=50, **cfg):
            assert _scalar(s.corner_radius) < _inradius(s)

    def test_sliver_triangles_rejected_under_min_feature(self):
        # a 1.0 base with 20/20 degree base angles has inradius ~0.088;
        # fixing those params with min_feature=0.3 is infeasible
        cfg = dict(
            min_feature_size=0.3,
            fixed_shape_params={"Triangle": {"base": 1.0, "alpha": 20.0, "beta": 20.0}},
        )
        with pytest.raises(RuntimeError, match="triangle_too_large_for_canvas"):
            _samples(TriangleSampler, LATTICES["square"], n=1, **cfg)

    def test_min_shape_size_bounds_base(self):
        for s in _samples(TriangleSampler, LATTICES["square"], min_shape_size=0.9):
            assert _scalar(s.base) >= 0.9 - TOL

    def test_infeasible_min_shape_size_raises(self):
        with pytest.raises(ValueError):
            _samples(TriangleSampler, LATTICES["square"], n=1, min_shape_size=10.0)


# ---------------------------------------------------------------------------
# Star-specific
# ---------------------------------------------------------------------------

def _star_limits(s):
    """Corner-radius limits exactly as Star.__init__ computes them."""
    R, r, n = _scalar(s.outer_radius), _scalar(s.inner_radius), s.n
    an = math.pi / n
    L = math.sqrt(R * R + r * r - 2.0 * R * r * math.cos(an))
    tan_tip = math.tan(math.asin(r * math.sin(an) / L))
    tan_valley = math.tan(math.asin(min(1.0, R * math.sin(an) / L)))
    return L, tan_tip, tan_valley


class TestStarSampler:
    def test_params_valid(self):
        for s in _samples(StarSampler, LATTICES["square"]):
            R, r = _scalar(s.outer_radius), _scalar(s.inner_radius)
            ocr, icr = _scalar(s.outer_corner_radius), _scalar(s.inner_corner_radius)
            L, tan_tip, tan_valley = _star_limits(s)
            assert 0 < r < R
            assert 0 <= ocr < L * tan_tip
            assert 0 <= icr < L * tan_valley
            assert ocr / tan_tip + icr / tan_valley < L

    def test_valleys_concave(self):
        # r >= R cos(pi/n) would make a convex 2n-gon (no real valleys)
        for s in _samples(StarSampler, LATTICES["square"]):
            R, r = _scalar(s.outer_radius), _scalar(s.inner_radius)
            assert r < R * math.cos(math.pi / s.n)

    def test_inner_radius_range_too_high_for_some_n_retries(self):
        # (0.6, 0.8)*R misses the valid range for n=3 (cap ~0.495); the
        # sampler must retry another n rather than raise.
        cfg = dict(fixed_shape_params={"Star": {"outer_radius": 1.0}},
                   shape_param_ranges={"Star": {"inner_radius": (0.6, 0.8)}})
        for s in _samples(StarSampler, LATTICES["square"], n=50, **cfg):
            assert s.n >= 4
            assert 0.6 <= _scalar(s.inner_radius) < math.cos(math.pi / s.n)

    def test_default_n_range(self):
        ns = {s.n for s in _samples(StarSampler, LATTICES["square"])}
        assert ns == {3, 4, 5, 6, 7}

    def test_default_angle_within_one_period(self):
        for s in _samples(StarSampler, LATTICES["square"]):
            assert 0.0 - TOL <= _scalar(s.angle) <= 360.0 / s.n + TOL

    def test_fixed_params_respected(self):
        cfg = dict(fixed_shape_params={
            "Star": {"n": 5, "outer_radius": 1.0, "inner_radius": 0.5, "angle": 12.0},
        })
        for s in _samples(StarSampler, LATTICES["square"], n=20, **cfg):
            assert s.n == 5
            assert _scalar(s.outer_radius) == pytest.approx(1.0)
            assert _scalar(s.inner_radius) == pytest.approx(0.5)
            assert _scalar(s.angle) == pytest.approx(12.0)

    def test_n_range_respected(self):
        cfg = dict(shape_param_ranges={"Star": {"n": (8, 10)}})
        ns = {s.n for s in _samples(StarSampler, LATTICES["square"], **cfg)}
        assert ns <= {8, 9, 10}
        assert len(ns) > 1

    def test_param_ranges_respected(self):
        cfg = dict(shape_param_ranges={
            "Star": {"outer_radius": (0.9, 1.1), "inner_radius": (0.4, 0.5)},
        })
        for s in _samples(StarSampler, LATTICES["square"], **cfg):
            assert 0.9 - TOL <= _scalar(s.outer_radius) <= 1.1 + TOL
            assert 0.4 - TOL <= _scalar(s.inner_radius) <= 0.5 + TOL

    def test_oversized_corner_radii_clamped_to_constructible(self):
        cfg = dict(fixed_shape_params={
            "Star": {"outer_corner_radius": 10.0, "inner_corner_radius": 10.0},
        })
        # construction itself would raise if the clamp were wrong
        for s in _samples(StarSampler, LATTICES["square"], n=100, **cfg):
            L, tan_tip, tan_valley = _star_limits(s)
            assert _scalar(s.outer_corner_radius) / tan_tip \
                + _scalar(s.inner_corner_radius) / tan_valley < L

    def test_sharp_stars_allowed_under_min_feature(self):
        # corners don't count toward min_feature_size, so fixed sharp
        # corners still sample, with spikes at least min_feature wide
        cfg = dict(
            min_feature_size=0.05,
            fixed_shape_params={"Star": {"outer_corner_radius": 0.0,
                                         "inner_corner_radius": 0.0}},
        )
        for s in _samples(StarSampler, LATTICES["square"], n=50, **cfg):
            assert _scalar(s.outer_corner_radius) == 0.0
            assert _scalar(s.inner_corner_radius) == 0.0
            assert s.min_feature_size >= 0.05 - TOL

    def test_fixed_inner_radius_too_small_for_min_feature(self):
        cfg = dict(min_feature_size=0.2, fixed_shape_params={"Star": {"inner_radius": 0.05}})
        with pytest.raises(RuntimeError, match="star_too_large_for_canvas"):
            _samples(StarSampler, LATTICES["square"], n=1, **cfg)

    def test_fits_cell_when_wide_lattice_limits_radius(self):
        lattice = LATTICES["wide"]
        _, y0, _, y1 = lattice_inner_bounds(lattice)
        for s in _samples(StarSampler, lattice):
            assert 2 * _scalar(s.outer_radius) <= (y1 - y0) + TOL
