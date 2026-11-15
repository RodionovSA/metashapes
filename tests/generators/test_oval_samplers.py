# tests/generators/test_oval_samplers.py
#
# Direct tests for EggSampler and StadiumSampler (generators/samplers/ovals.py).
# These exercise the samplers in isolation, without the generator's retry /
# validation loop, so a sampler that emits bad shapes can't hide behind it.

import math
import random

import pytest

from metashapes.lattice.basis import Lattice
from metashapes.generators.random import RandomGeneratorConfig, RandomUnitCellGenerator
from metashapes.generators.registry import SHAPE_SAMPLER_REGISTRY
from metashapes.generators.samplers.ovals import EggSampler, StadiumSampler
from metashapes.generators.samplers.utils import lattice_inner_bounds
from metashapes.shape.primitives import Egg, Stadium

N_SAMPLES = 200
TOL = 1e-5

LATTICES = {
    "square": Lattice.rectangular(3.0, 3.0),
    "wide": Lattice.rectangular(4.0, 1.5),
    "hexagonal": Lattice.hexagonal(3.0),
}

SAMPLERS = {
    "Egg": (EggSampler, Egg),
    "Stadium": (StadiumSampler, Stadium),
}


def _samples(sampler_cls, lattice, n=N_SAMPLES, seed=0, **cfg):
    config = RandomGeneratorConfig(**cfg)
    rng = random.Random(seed)
    sampler = sampler_cls()
    return [sampler.sample(rng, lattice, config) for _ in range(n)]


def _scalar(t):
    return t.detach().item()


# ---------------------------------------------------------------------------
# shared properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(SAMPLERS))
def test_registered_under_shape_name(name):
    sampler_cls, _ = SAMPLERS[name]
    assert isinstance(SHAPE_SAMPLER_REGISTRY[name], sampler_cls)


@pytest.mark.parametrize("name", list(SAMPLERS))
@pytest.mark.parametrize("lattice_name", list(LATTICES))
def test_returns_correct_type(name, lattice_name):
    sampler_cls, shape_cls = SAMPLERS[name]
    for s in _samples(sampler_cls, LATTICES[lattice_name], n=20):
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
def test_deterministic_with_seed(name):
    sampler_cls, _ = SAMPLERS[name]
    lattice = LATTICES["square"]
    a = _samples(sampler_cls, lattice, n=10, seed=123)
    b = _samples(sampler_cls, lattice, n=10, seed=123)
    for s1, s2 in zip(a, b):
        assert s1.bounds() == s2.bounds()


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_respects_min_shape_size(name):
    sampler_cls, _ = SAMPLERS[name]
    for s in _samples(sampler_cls, LATTICES["square"], min_shape_size=0.8):
        if name == "Egg":
            assert _scalar(s.width) >= 0.8 - TOL
            assert _scalar(s.height) >= 0.8 - TOL
        else:
            assert _scalar(s.length) >= 0.8 - TOL
            assert _scalar(s.width) >= 0.8 - TOL


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_infeasible_min_shape_size_raises(name):
    sampler_cls, _ = SAMPLERS[name]
    with pytest.raises(ValueError):
        _samples(sampler_cls, LATTICES["square"], n=1, min_shape_size=10.0)


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_generator_emits_only_this_shape(name):
    _, shape_cls = SAMPLERS[name]
    config = RandomGeneratorConfig(
        target_count=5, seed=0, allowed_shapes=(name,), fixed_num_shapes=1,
    )
    result = RandomUnitCellGenerator(config).generate(LATTICES["square"])
    assert result.report.generated == 5
    for cell in result.unit_cells:
        assert isinstance(cell.scene, shape_cls)


# ---------------------------------------------------------------------------
# Egg-specific
# ---------------------------------------------------------------------------

class TestEggSampler:
    def test_params_valid(self):
        for s in _samples(EggSampler, LATTICES["square"]):
            assert _scalar(s.width) > 0
            assert _scalar(s.height) > 0
            assert abs(_scalar(s.skew)) < 1.0

    def test_default_skew_range(self):
        skews = [_scalar(s.skew) for s in _samples(EggSampler, LATTICES["square"])]
        assert all(-0.5 - TOL <= k <= 0.5 + TOL for k in skews)
        # actually asymmetric eggs get produced, not just ellipses
        assert any(abs(k) > 0.1 for k in skews)

    def test_fixed_params_respected(self):
        cfg = dict(fixed_shape_params={
            "Egg": {"width": 1.0, "height": 1.4, "skew": 0.3, "angle": 45.0},
        })
        for s in _samples(EggSampler, LATTICES["square"], n=20, **cfg):
            assert _scalar(s.width) == pytest.approx(1.0)
            assert _scalar(s.height) == pytest.approx(1.4)
            assert _scalar(s.skew) == pytest.approx(0.3)
            assert _scalar(s.angle) == pytest.approx(45.0)

    def test_fixed_center_respected(self):
        cfg = dict(fixed_shape_params={
            "Egg": {"center": (1.5, 1.5), "width": 0.8, "height": 0.8},
        })
        for s in _samples(EggSampler, LATTICES["square"], n=10, **cfg):
            assert s.center.tolist() == pytest.approx([1.5, 1.5])

    def test_param_ranges_respected(self):
        cfg = dict(shape_param_ranges={
            "Egg": {"width": (0.5, 0.7), "height": (0.9, 1.1),
                    "skew": (0.2, 0.4), "angle": (10.0, 20.0)},
        })
        for s in _samples(EggSampler, LATTICES["square"], **cfg):
            assert 0.5 - TOL <= _scalar(s.width) <= 0.7 + TOL
            assert 0.9 - TOL <= _scalar(s.height) <= 1.1 + TOL
            assert 0.2 - TOL <= _scalar(s.skew) <= 0.4 + TOL
            assert 10.0 - TOL <= _scalar(s.angle) <= 20.0 + TOL

    def test_impossible_fixed_size_raises_runtime_error(self):
        cfg = dict(fixed_shape_params={"Egg": {"width": 5.0, "height": 5.0}})
        with pytest.raises(RuntimeError, match="egg_too_large_for_canvas"):
            _samples(EggSampler, LATTICES["square"], n=1, **cfg)

    def test_respects_min_feature_size(self):
        for s in _samples(EggSampler, LATTICES["square"], min_feature_size=0.4):
            assert s.min_feature_size >= 0.4 - TOL


# ---------------------------------------------------------------------------
# Stadium-specific
# ---------------------------------------------------------------------------

class TestStadiumSampler:
    def test_params_valid(self):
        for s in _samples(StadiumSampler, LATTICES["square"]):
            length, width = _scalar(s.length), _scalar(s.width)
            assert width > 0
            assert width <= length + TOL

    def test_fixed_params_respected(self):
        cfg = dict(fixed_shape_params={
            "Stadium": {"length": 2.0, "width": 0.6, "angle": 30.0},
        })
        for s in _samples(StadiumSampler, LATTICES["square"], n=20, **cfg):
            assert _scalar(s.length) == pytest.approx(2.0)
            assert _scalar(s.width) == pytest.approx(0.6)
            assert _scalar(s.angle) == pytest.approx(30.0)

    def test_param_ranges_respected(self):
        cfg = dict(shape_param_ranges={
            "Stadium": {"length": (1.5, 2.0), "width": (0.3, 0.5),
                        "angle": (80.0, 100.0)},
        })
        for s in _samples(StadiumSampler, LATTICES["square"], **cfg):
            assert 1.5 - TOL <= _scalar(s.length) <= 2.0 + TOL
            assert 0.3 - TOL <= _scalar(s.width) <= 0.5 + TOL
            assert 80.0 - TOL <= _scalar(s.angle) <= 100.0 + TOL

    def test_respects_min_feature_size(self):
        for s in _samples(StadiumSampler, LATTICES["square"], min_feature_size=0.4):
            assert s.min_feature_size >= 0.4 - TOL

    def test_width_capped_by_cell_height(self):
        # in a short cell the cap diameter can't exceed the cell height
        lattice = LATTICES["wide"]
        _, y0, _, y1 = lattice_inner_bounds(lattice)
        for s in _samples(StadiumSampler, lattice):
            assert _scalar(s.width) <= (y1 - y0) + TOL

    def test_long_stadium_fits_when_rotated(self):
        # length near the diagonal only fits at specific angles; the sampler
        # must reject the rest rather than emit an out-of-cell shape
        cfg = dict(fixed_shape_params={"Stadium": {"length": 3.5, "width": 0.3}})
        lattice = LATTICES["square"]
        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
        for s in _samples(StadiumSampler, lattice, n=30, **cfg):
            (bx0, by0), (bx1, by1) = s.bounds()
            assert bx0 >= x0 - TOL and bx1 <= x1 + TOL
            assert by0 >= y0 - TOL and by1 <= y1 + TOL
            # 3.5 > 3.0 cell side, so it must be tilted off-axis
            a = _scalar(s.angle) % 180.0
            assert 10.0 < a < 170.0 and not math.isclose(a, 90.0, abs_tol=10.0)

    def test_width_greater_than_length_rejected(self):
        cfg = dict(fixed_shape_params={"Stadium": {"length": 0.5, "width": 1.0}})
        with pytest.raises(RuntimeError):
            _samples(StadiumSampler, LATTICES["square"], n=1, **cfg)
