# tests/lattice/test_unit_cell.py

import math
import pytest
import torch
import torch.nn as nn

from src.metashapes.lattice.basis import Lattice
from src.metashapes.lattice.unit_cell import UnitCell
from src.metashapes.shape.primitives.polygons import RegularPolygon
from src.metashapes.shape.primitives.periodic import Stripe

from .conftest import make_learnable_polygon


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _square_cell(px=2.0, py=2.0, side=0.5):
    """Rectangular unit cell with a small square at the origin."""
    lattice = Lattice.rectangular(px, py)
    shape = RegularPolygon(center=torch.zeros(2), n=4, side_length=torch.tensor(side))
    return UnitCell(lattice, shape)


def _sdf_at(cell, x, y):
    return cell.sdf(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    ).item()


# ---------------------------------------------------------------------------
# SDF tests
# ---------------------------------------------------------------------------

class TestUnitCellSDF:
    def test_sdf_inside_center(self):
        cell = _square_cell()
        assert _sdf_at(cell, 0.0, 0.0) < 0.0

    def test_sdf_outside(self):
        cell = _square_cell(side=0.5)
        # well outside the small square — SDF > 0
        assert _sdf_at(cell, 0.8, 0.8) > 0.0

    def test_sdf_periodicity(self):
        px, py = 2.0, 3.0
        cell = _square_cell(px=px, py=py)
        x0, y0 = torch.tensor(0.2), torch.tensor(0.1)
        d0 = cell.sdf(x0, y0).item()
        d1 = cell.sdf(x0 + px, y0).item()
        d2 = cell.sdf(x0, y0 + py).item()
        assert d0 == pytest.approx(d1, abs=1e-4)
        assert d0 == pytest.approx(d2, abs=1e-4)

    def test_sdf_hexagonal(self):
        lat = Lattice.hexagonal(2.0, orientation="pointy")
        shape = RegularPolygon(center=torch.zeros(2), n=6, side_length=torch.tensor(0.5))
        cell = UnitCell(lat, shape)
        x = torch.zeros(4)
        y = torch.zeros(4)
        d = cell.sdf(x, y)
        assert d.shape == (4,)
        assert torch.isfinite(d).all()

    def test_sdf_returns_tensor(self):
        cell = _square_cell()
        x = torch.linspace(-1.0, 1.0, 8)
        y = torch.zeros(8)
        d = cell.sdf(x, y)
        assert isinstance(d, torch.Tensor)
        assert d.shape == (8,)


# ---------------------------------------------------------------------------
# Rasterize tests
# ---------------------------------------------------------------------------

class TestUnitCellRasterize:
    def test_rasterize_shape_default(self):
        cell = _square_cell()
        out = cell.rasterize(32, 24)
        assert out.shape == (24, 32)

    def test_rasterize_shape_repeat(self):
        cell = _square_cell()
        n1, n2 = 3, 2
        out = cell.rasterize(16, 12, repeat=(n1, n2))
        assert out.shape == (12 * n2, 16 * n1)

    def test_rasterize_cartesian_shape(self):
        cell = _square_cell()
        out = cell.rasterize(32, 24, cartesian=True)
        assert out.shape == (24, 32)

    def test_rasterize_cartesian_repeat(self):
        cell = _square_cell()
        out = cell.rasterize(16, 12, repeat=(2, 3), cartesian=True)
        assert out.shape == (12 * 3, 16 * 2)

    def test_rasterize_values_finite(self):
        cell = _square_cell()
        out = cell.rasterize(32, 32)
        assert torch.isfinite(out).all()

    def test_rasterize_has_negative_values(self):
        # the interior of the square must appear in the SDF grid
        cell = _square_cell()
        out = cell.rasterize(64, 64)
        assert (out < 0).any()


# ---------------------------------------------------------------------------
# Mask tests
# ---------------------------------------------------------------------------

class TestUnitCellMask:
    def test_mask_hard_binary(self):
        cell = _square_cell()
        m = cell.mask(32, 32, soft=False)
        unique = m.unique()
        for v in unique:
            assert v.item() in {0.0, 1.0}

    def test_mask_shape_matches_rasterize(self):
        cell = _square_cell()
        rast = cell.rasterize(32, 24)
        m = cell.mask(32, 24, soft=False)
        assert m.shape == rast.shape

    def test_mask_soft_range(self):
        cell = _square_cell()
        m = cell.mask(32, 32, soft=True)
        assert (m > 0.0).all()
        assert (m < 1.0).all()

    def test_mask_soft_has_grad_fn(self):
        # requires an nn.Parameter so sigmoid stays in the autograd graph
        cell, _, _ = make_learnable_polygon()
        m = cell.mask(32, 32, soft=True)
        assert m.grad_fn is not None

    def test_mask_soft_custom_softness(self):
        cell = _square_cell()
        m = cell.mask(32, 32, soft=True, softness=0.05)
        assert m.shape == (32, 32)
        assert torch.isfinite(m).all()

    def test_mask_hard_no_grad_fn(self):
        cell = _square_cell()
        m = cell.mask(32, 32, soft=False)
        assert m.grad_fn is None


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

class TestUnitCellBoundary:
    def test_boundary_points_nontrivial(self):
        cell = _square_cell()
        pts = cell.boundary_points(resolution=128)
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        assert pts.shape[0] > 0

    def test_boundary_points_empty_when_full(self):
        # A Stripe much wider than the cell: all points inside, no boundary.
        lat = Lattice.rectangular(1.0, 1.0)
        stripe = Stripe(offset=0.0, width=100.0, axis='x')
        cell = UnitCell(lat, stripe)
        pts = cell.boundary_points(resolution=64)
        assert pts.shape == (0, 2)

    def test_boundary_points_dtype(self):
        cell = _square_cell()
        pts = cell.boundary_points(resolution=64)
        import numpy as np
        assert pts.dtype == np.float64


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestUnitCellSerialization:
    def _sdf_grid(self, cell, n=16):
        xs = torch.linspace(-0.9, 0.9, n)
        ys = torch.linspace(-0.9, 0.9, n)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        return cell.sdf(X, Y)

    def test_round_trip_rectangular(self):
        cell = _square_cell()
        data = cell.to_parametric()
        restored = UnitCell.from_parametric(data)
        d_orig = self._sdf_grid(cell)
        d_rest = self._sdf_grid(restored)
        assert torch.allclose(d_orig, d_rest, atol=1e-5)

    def test_round_trip_hexagonal(self):
        lat = Lattice.hexagonal(2.0, orientation="flat")
        shape = RegularPolygon(center=torch.zeros(2), n=6, side_length=torch.tensor(0.4))
        cell = UnitCell(lat, shape)
        data = cell.to_parametric()
        restored = UnitCell.from_parametric(data)
        d_orig = self._sdf_grid(cell)
        d_rest = self._sdf_grid(restored)
        assert torch.allclose(d_orig, d_rest, atol=1e-5)

    def test_round_trip_preserves_lattice_vectors(self):
        lat = Lattice.rectangular(1.5, 2.5)
        shape = RegularPolygon(center=torch.zeros(2), n=3, side_length=torch.tensor(0.3))
        cell = UnitCell(lat, shape)
        restored = UnitCell.from_parametric(cell.to_parametric())
        assert torch.allclose(restored.lattice.a1, lat.a1, atol=1e-6)
        assert torch.allclose(restored.lattice.a2, lat.a2, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient propagation tests
# ---------------------------------------------------------------------------

class TestUnitCellGradients:
    def test_grad_sdf_wrt_side_length(self):
        cell, side_length, _ = make_learnable_polygon(side_length_val=0.5)
        x = torch.tensor([0.0, 0.1, 0.3])
        y = torch.tensor([0.0, 0.1, 0.2])
        d = cell.sdf(x, y)
        d.sum().backward()
        assert side_length.grad is not None
        assert side_length.grad.abs().item() > 0.0

    def test_grad_sdf_wrt_center(self):
        cell, _, center = make_learnable_polygon(center_val=(0.1, 0.0))
        x = torch.tensor([0.2, 0.3])
        y = torch.tensor([0.1, 0.0])
        d = cell.sdf(x, y)
        d.sum().backward()
        assert center.grad is not None
        assert center.grad.norm().item() > 0.0

    def test_grad_soft_mask_wrt_side_length(self):
        cell, side_length, _ = make_learnable_polygon(side_length_val=0.5)
        m = cell.mask(16, 16, soft=True)
        m.sum().backward()
        assert side_length.grad is not None

    def test_grad_soft_mask_wrt_center(self):
        cell, _, center = make_learnable_polygon(center_val=(0.0, 0.1))
        m = cell.mask(16, 16, soft=True)
        m.sum().backward()
        assert center.grad is not None

    def test_grad_flows_through_periodicity(self):
        """Gradient flows from a point displaced by one lattice vector."""
        px = 2.0
        cell, side_length, _ = make_learnable_polygon(side_length_val=0.5)
        # Query displaced by one full period — the periodic copy carries the SDF
        x = torch.tensor([0.0 + px])
        y = torch.tensor([0.0])
        d = cell.sdf(x, y)
        d.sum().backward()
        assert side_length.grad is not None
        assert side_length.grad.abs().item() > 0.0

    def test_hard_mask_requires_grad_false(self):
        cell, side_length, _ = make_learnable_polygon()
        m = cell.mask(16, 16, soft=False)
        assert not m.requires_grad

    def test_grad_soft_mask_wrt_side_length_nonzero_pixels(self):
        """At least some pixels must have non-trivial gradient contribution."""
        cell, side_length, _ = make_learnable_polygon(side_length_val=0.5)
        m = cell.mask(32, 32, soft=True)
        # Only accumulate gradient from near-boundary pixels (0.05 < m < 0.95)
        boundary_region = m[(m > 0.05) & (m < 0.95)]
        if boundary_region.numel() > 0:
            boundary_region.sum().backward()
            assert side_length.grad is not None
            assert side_length.grad.abs().item() > 0.0


# ---------------------------------------------------------------------------
# to_shapely tests
# ---------------------------------------------------------------------------

class TestToShapely:
    def test_rectangle_clips_to_cell(self):
        from src.metashapes.shape.primitives.quads import Rectangle
        lattice = Lattice.rectangular(2.0, 2.0)
        shape = Rectangle(center=torch.zeros(2), size=torch.tensor([0.8, 0.8]))
        cell = UnitCell(lattice, shape)
        geom = cell.to_shapely()
        assert not geom.is_empty
        assert geom.area > 0.0
        # result must lie within the unit cell parallelogram
        from shapely.geometry import Polygon
        a1 = lattice.a1.tolist()
        a2 = lattice.a2.tolist()
        cell_poly = Polygon([(0, 0), (a1[0], a1[1]),
                             (a1[0]+a2[0], a1[1]+a2[1]), (a2[0], a2[1])])
        assert geom.difference(cell_poly).is_empty

    def test_stripe_clips_to_cell(self):
        lattice = Lattice.rectangular(2.0, 3.0)
        # offset=1.5 centers the stripe in the [0,3] cell; y ∈ [1.0, 2.0] fully inside
        shape = Stripe(offset=torch.tensor(1.5), width=torch.tensor(1.0), axis="x")
        cell = UnitCell(lattice, shape)
        geom = cell.to_shapely()
        assert not geom.is_empty
        assert geom.area > 0.0
        # stripe width=1.0, cell Lx=2.0 → intersection area = 2.0 * 1.0
        assert abs(geom.area - 2.0 * 1.0) < 0.01

    def test_shape_outside_cell_is_empty(self):
        from src.metashapes.shape.primitives.quads import Rectangle
        lattice = Lattice.rectangular(2.0, 2.0)
        # Rectangle centered far outside the unit cell
        shape = Rectangle(center=torch.tensor([100.0, 100.0]),
                          size=torch.tensor([0.5, 0.5]))
        cell = UnitCell(lattice, shape)
        geom = cell.to_shapely()
        assert geom.is_empty


# ---------------------------------------------------------------------------
# extent() tests
# ---------------------------------------------------------------------------

class TestExtent:
    def test_rectangular_unit_cell(self):
        cell = UnitCell(Lattice.rectangular(2.0, 3.0), _square_cell().scene)
        xmin, xmax, ymin, ymax = cell.extent()
        assert xmin == pytest.approx(0.0)
        assert xmax == pytest.approx(2.0)
        assert ymin == pytest.approx(0.0)
        assert ymax == pytest.approx(3.0)

    def test_rectangular_repeat(self):
        cell = UnitCell(Lattice.rectangular(2.0, 3.0), _square_cell().scene)
        xmin, xmax, ymin, ymax = cell.extent(repeat=(2, 3))
        assert xmin == pytest.approx(0.0)
        assert xmax == pytest.approx(4.0)
        assert ymin == pytest.approx(0.0)
        assert ymax == pytest.approx(9.0)

    def test_hexagonal_pointy_width(self):
        # Pointy hex: a1=(a,0), a2=(a/2, a√3/2) → corners at (0,0),(a,0),(a/2,…),(3a/2,…)
        # AABB width = xmax - xmin = 3a/2
        a = 2.0
        cell = UnitCell(Lattice.hexagonal(a, orientation="pointy"), _square_cell().scene)
        xmin, xmax, ymin, ymax = cell.extent()
        assert xmax - xmin == pytest.approx(1.5 * a, abs=1e-5)
        assert ymax > ymin

    def test_hexagonal_repeat_n1_grows_width(self):
        # For repeat=(n1, 1), x-extent grows as n1*a + a/2 (last corner at n1*a1 + a2)
        a = 1.0
        cell = UnitCell(Lattice.hexagonal(a, orientation="pointy"), _square_cell().scene)
        xmin1, xmax1, _, _ = cell.extent(repeat=(1, 1))
        xmin3, xmax3, _, _ = cell.extent(repeat=(3, 1))
        # repeat=(3,1): corners at (0,0),(3a,0),(a/2,…),(7a/2,…) → width = 7a/2
        assert xmax3 - xmin3 == pytest.approx(3.5 * a, abs=1e-5)
        assert xmax3 - xmin3 > xmax1 - xmin1

    def test_returns_plain_floats(self):
        cell = _square_cell()
        result = cell.extent()
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)

    def test_default_repeat_is_1_1(self):
        cell = _square_cell()
        assert cell.extent() == cell.extent(repeat=(1, 1))
