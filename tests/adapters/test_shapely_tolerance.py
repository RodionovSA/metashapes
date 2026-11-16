# tests/adapters/test_shapely_tolerance.py
#
# shape_to_shapely(tolerance=...) bounds the distance between every polygon
# edge and the true outline. Checked against the SDF: vertices lie on the
# zero contour, edge midpoints are within the tolerance of it.

import random

import numpy as np
import pytest
import shapely
import torch

from metashapes.adapters.shapely import shape_to_shapely
from metashapes.adapters.shapely.helpers import arc_quad_segs, tolerance_scope
from metashapes.generators import GeneratorConfig
from metashapes.generators.registry import SHAPE_SAMPLER_REGISTRY
from metashapes.lattice import Lattice
from metashapes.shape import Ellipse, Rectangle, Scale, Translate

FINITE = ["Rectangle", "ConvexQuad", "IsoscelesTrapezoid", "Ellipse", "Egg", "Stadium",
          "RegularPolygon", "Triangle", "Star", "Cross", "TShape"]


def _rings(geom):
    polys = getattr(geom, "geoms", [geom])
    for p in polys:
        yield np.asarray(p.exterior.coords)
        for r in p.interiors:
            yield np.asarray(r.coords)


def _edge_errors(shape, geom):
    """(max |sdf| at vertices, max |sdf| at edge midpoints)."""
    vmax = mmax = 0.0
    for ring in _rings(geom):
        v = torch.as_tensor(ring, dtype=torch.float64)
        mid = 0.5 * (v[1:] + v[:-1])
        with torch.no_grad():
            dv = shape.sdf(v[:, 0], v[:, 1]).abs().max().item()
            dm = shape.sdf(mid[:, 0], mid[:, 1]).abs().max().item()
        vmax, mmax = max(vmax, dv), max(mmax, dm)
    return vmax, mmax


def _n_vertices(geom):
    return sum(len(r) for r in _rings(geom))


def _samples(name, n=15, seed=0):
    rng = random.Random(seed)
    lat = Lattice.rectangular(1000.0, 1000.0)
    cfg = GeneratorConfig(allowed_shapes=(name,), min_feature_size=20, min_shape_size=60)
    return [SHAPE_SAMPLER_REGISTRY[name].sample(rng, lat, cfg).to(torch.float64) for _ in range(n)]


@pytest.mark.parametrize("name", FINITE)
@pytest.mark.parametrize("tol", [0.05, 1.0])
def test_edges_within_tolerance(name, tol):
    for s in _samples(name):
        g = shape_to_shapely(s, tolerance=tol)
        v_err, m_err = _edge_errors(s, g)
        size = max(b - a for a, b in zip(*s.bounds()))
        assert v_err <= 1e-6 * size + 1e-6, (name, v_err)
        assert m_err <= tol * (1 + 1e-6) + 1e-6 * size, (name, m_err, tol)


@pytest.mark.parametrize("name", ["Ellipse", "Egg", "Stadium", "Rectangle", "Star", "Cross"])
def test_looser_tolerance_fewer_vertices(name):
    for s in _samples(name, n=5):
        fine = _n_vertices(shape_to_shapely(s, tolerance=0.01))
        coarse = _n_vertices(shape_to_shapely(s, tolerance=2.0))
        assert coarse <= fine


def test_default_unchanged():
    e = Ellipse(center=(0.0, 0.0), axes=(10.0, 4.0))
    assert _n_vertices(shape_to_shapely(e)) == 65          # historical 64 segments
    assert shape_to_shapely(e).equals(shape_to_shapely(e, tolerance=None))


def test_scale_tightens_child_tolerance():
    # child drawn at 1/s of its final size must use tolerance / s
    e = Ellipse(center=(0.0, 0.0), axes=(10.0, 10.0)).to(torch.float64)
    s = Scale(e, 50.0)
    g = shape_to_shapely(s, tolerance=0.5)
    _, m_err = _edge_errors(s, g)
    assert m_err <= 0.5 * (1 + 1e-6)


def test_composites_within_tolerance():
    a = Ellipse(center=(0.0, 0.0), axes=(300.0, 120.0)).to(torch.float64)
    b = Translate(Rectangle(center=(0.0, 0.0), size=(80.0, 200.0), corner_radius=30.0), 120.0, 0.0
                  ).to(torch.float64)
    for shape in (a | b, a - b, a & b):
        g = shape_to_shapely(shape, tolerance=0.2)
        v_err, m_err = _edge_errors(shape, g)
        # boolean SDFs are exact near the boundary away from crossings;
        # crossing vertices come from Shapely's intersection, not the arcs
        assert m_err <= 0.2 * (1 + 1e-6) + 1e-6


def test_tolerance_scope_restores_and_validates():
    with tolerance_scope(1.0):
        assert arc_quad_segs(100.0, 16) < 16
    assert arc_quad_segs(100.0, 16) == 16
    with pytest.raises(ValueError):
        with tolerance_scope(0.0):
            pass


def test_tiny_radius_single_segment():
    with tolerance_scope(5.0):
        assert arc_quad_segs(1.0, 16) == 1


@pytest.mark.parametrize("params", [
    # convex "valleys" (r >= R cos(pi/n)): rounded by outer_corner_radius
    dict(n=3, outer_radius=50.72, inner_radius=38.84, outer_corner_radius=19.68,
         inner_corner_radius=7.45),
    # tip and valley fillets nearly meeting on each edge (ill-conditioned for
    # buffer-based construction)
    dict(n=7, outer_radius=491.91, inner_radius=348.11, outer_corner_radius=55.52,
         inner_corner_radius=218.12),
])
def test_star_outline_exact(params):
    from metashapes.shape import Star
    s = Star(center=(3.0, -2.0), angle=11.0, **params).to(torch.float64)
    for tol in (1.0, 0.01):
        g = shape_to_shapely(s, tolerance=tol)
        assert g.is_valid
        v_err, m_err = _edge_errors(s, g)
        assert v_err <= 1e-6 and m_err <= tol * (1 + 1e-6)
