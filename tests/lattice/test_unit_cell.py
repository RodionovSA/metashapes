# tests/lattice/test_unit_cell.py

import math
import pytest
import torch
import torch.nn as nn

from metashapes.lattice.basis import Lattice
from metashapes.lattice.unit_cell import UnitCell
from metashapes.shape.primitives.polygons import RegularPolygon
from metashapes.shape.primitives.stripes import Bar

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

    def test_rasterize_cartesian_matches_fractional_pixel_pitch(self):
        # cartesian=True and cartesian=False should give the same pixel
        # pitch for the same nx/ny -- both should be endpoint-excluded,
        # not one inclusive and the other not.
        cell = _square_cell(px=1.0, py=1.0, side=0.2)
        d_false = cell.rasterize(16, 16, cartesian=False)
        d_true = cell.rasterize(16, 16, cartesian=True)
        assert torch.allclose(d_false, d_true, atol=1e-4)

    def test_rasterize_cartesian_repeat_has_no_duplicate_seam(self):
        # A repeated cartesian=True raster must tile seamlessly: the last
        # row/col of one cell and the first row/col of the next must not be
        # identical duplicates of the same boundary sample.
        cell = _square_cell(px=1.0, py=1.0, side=0.2)
        out = cell.rasterize(4, 4, cartesian=True, repeat=(2, 2))
        assert not torch.allclose(out[3], out[4])
        assert not torch.allclose(out[:, 3], out[:, 4])

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
        # A Bar much wider than the cell: all points inside, no boundary.
        lat = Lattice.rectangular(1.0, 1.0)
        bar = Bar(offset=0.0, width=100.0, axis='x')
        cell = UnitCell(lat, bar)
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

    def test_grad_soft_mask_wrt_optimizable_lattice_vector(self):
        """A Lattice built with an nn.Parameter a1 is itself optimizable:
        gradient from the soft mask must reach the caller's own tensor.
        """
        a1 = nn.Parameter(torch.tensor([2.0, 0.0]))
        lattice = Lattice(a1=a1, a2=torch.tensor([0.0, 2.0]))
        shape = RegularPolygon(center=(0.0, 0.0), n=4, side_length=0.5)
        cell = UnitCell(lattice, shape)
        m = cell.mask(32, 32, soft=True)
        m.sum().backward()
        assert a1.grad is not None
        assert torch.isfinite(a1.grad).all()


# ---------------------------------------------------------------------------
# to_shapely tests
# ---------------------------------------------------------------------------

class TestToShapely:
    def test_rectangle_clips_to_cell(self):
        from metashapes.shape.primitives.quads import Rectangle
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

    def test_bar_clips_to_cell(self):
        lattice = Lattice.rectangular(2.0, 3.0)
        # offset=1.5 centers the bar in the [0,3] cell; y ∈ [1.0, 2.0] fully inside
        shape = Bar(offset=torch.tensor(1.5), width=torch.tensor(1.0), axis="x")
        cell = UnitCell(lattice, shape)
        geom = cell.to_shapely()
        assert not geom.is_empty
        assert geom.area > 0.0
        # bar width=1.0, cell Lx=2.0 → intersection area = 2.0 * 1.0
        assert abs(geom.area - 2.0 * 1.0) < 0.01

    def test_shape_outside_cell_wraps_periodically(self):
        # to_shapely() (and sdf()/mask()) are fully periodic, so a scene
        # positioned far outside the cell still wraps back in -- it does
        # not read as empty. A rect 50 cells away on a 2.0 lattice must
        # give exactly the same geometry as the same rect placed at the
        # cell origin.
        from metashapes.shape.primitives.quads import Rectangle
        lattice = Lattice.rectangular(2.0, 2.0)
        far = Rectangle(center=torch.tensor([100.3, 100.3]),
                         size=torch.tensor([0.5, 0.5]))
        near = Rectangle(center=torch.tensor([0.3, 0.3]),
                          size=torch.tensor([0.5, 0.5]))
        far_cell = UnitCell(lattice, far)
        near_cell = UnitCell(lattice, near)

        far_geom = far_cell.to_shapely()
        near_geom = near_cell.to_shapely()
        assert not far_geom.is_empty
        assert far_geom.area == pytest.approx(near_geom.area, abs=1e-6)
        assert far_geom.area == pytest.approx(0.25, abs=1e-6)
        # float32 precision at a ~200-unit offset limits how tightly the two
        # polygons can agree; the mask comparison below is the tight check.
        assert far_geom.symmetric_difference(near_geom).area == pytest.approx(0.0, abs=1e-4)

        far_mask = far_cell.mask(64, 64)
        near_mask = near_cell.mask(64, 64)
        assert torch.equal(far_mask, near_mask)

    def test_area_matches_periodic_sdf_for_seam_straddling_shape(self):
        # to_shapely() must account for a shape's wrapped-around part when
        # it crosses the cell boundary. A 0.4x0.4 rectangle straddling the
        # x=0 seam on a 1x1 cell: the true (periodic) area is the full
        # 0.4*0.4=0.16, not the 0.08 a non-periodic clip would give.
        from metashapes.shape.primitives.quads import Rectangle
        lattice = Lattice.rectangular(1.0, 1.0)
        shape = Rectangle(center=torch.tensor([0.0, 0.5]),
                          size=torch.tensor([0.4, 0.4]))
        cell = UnitCell(lattice, shape)
        geom = cell.to_shapely()
        assert geom.area == pytest.approx(0.16, abs=1e-6)

    def test_fully_interior_shape_unaffected_by_periodic_union(self):
        # A shape entirely inside the cell must give the same result as
        # before -- its neighbouring copies fall outside cell_poly and are
        # discarded by the intersection, same as a plain single-copy clip.
        # cell_poly spans [0, px] x [0, py], not centered at the origin, so
        # the shape must be placed mid-cell to actually stay interior.
        from metashapes.shape.primitives.quads import Rectangle
        lattice = Lattice.rectangular(2.0, 2.0)
        shape = Rectangle(center=torch.tensor([1.0, 1.0]), size=torch.tensor([0.5, 0.5]))
        cell = UnitCell(lattice, shape)
        geom = cell.to_shapely()
        assert geom.area == pytest.approx(0.25, abs=1e-6)


# ---------------------------------------------------------------------------
# _offsets_for() / periodic copy search tests
# ---------------------------------------------------------------------------

def _brute_force_sdf(cell, x, y, ring=20):
    """Independent reference: fold into the cell, then search a ring far
    larger than any offset _offsets_for should ever need. Used to verify
    _offsets_for's copy search is actually sufficient, not just internally
    consistent."""
    f1, f2 = cell.lattice.to_fractional(x, y)
    xf, yf = cell.lattice.to_cartesian(f1 % 1.0, f2 % 1.0)
    best = None
    for i in range(-ring, ring + 1):
        for j in range(-ring, ring + 1):
            ox, oy = cell.lattice.offset(i, j)
            d = cell.scene.sdf(xf - ox, yf - oy)
            best = d if best is None else torch.minimum(best, d)
    return best


class TestOffsetsForRing:
    def test_bar_offsets_are_finite_not_nan(self):
        # Bar's bbox has an infinite x-extent; the finite (y) direction's
        # offsets must stay finite too, not get corrupted by mapping the
        # literal +/-inf bbox corners through to_fractional.
        lattice = Lattice.rectangular(1.0, 1.0)
        bar = Bar(offset=torch.tensor(0.0), width=torch.tensor(0.3), axis="x")
        cell = UnitCell(lattice, bar)
        (i0, i1), (j0, j1) = cell._offsets_for(bar)
        assert all(math.isfinite(v) for v in (i0, i1, j0, j1))

    def test_bar_offsets_finite_on_hexagonal_lattice_too(self):
        lattice = Lattice.hexagonal(1.0, orientation="pointy")
        bar = Bar(offset=torch.tensor(0.0), width=torch.tensor(0.3), axis="x")
        cell = UnitCell(lattice, bar)
        (i0, i1), (j0, j1) = cell._offsets_for(bar)
        assert all(math.isfinite(v) for v in (i0, i1, j0, j1))

    def test_mixed_infinite_and_wide_finite_union_covers_the_finite_branch(self):
        # Union(Bar, wide Rectangle): Union.bounds()'s componentwise
        # min/max could let the Bar's -inf/+inf x-extent swallow the
        # Rectangle's own, finite, much wider extent -- confirm the offset
        # range still covers the Rectangle.
        from metashapes.shape.primitives.quads import Rectangle
        from metashapes.shape.boolean import Union

        lattice = Lattice.rectangular(1.0, 1.0)
        bar = Bar(offset=torch.tensor(0.0), width=torch.tensor(0.3), axis="x")
        wide_rect = Rectangle(center=torch.tensor([0.0, 0.0]), size=torch.tensor([5.0, 0.2]))
        union = Union(bar, wide_rect)
        cell = UnitCell(lattice, union)

        (i0, i1), _ = cell._offsets_for(union)
        # The rectangle spans 5 cells in x; a single ring of 1 (offsets
        # -1..1) cannot possibly cover it.
        assert i1 - i0 >= 5

        # And the periodic sdf must actually agree with a much larger
        # brute-force search -- not just report a wide-looking range.
        xs = torch.linspace(-3.0, 3.0, 41)
        X, Y = torch.meshgrid(xs, xs, indexing="xy")
        assert torch.allclose(cell.sdf(X, Y), _brute_force_sdf(cell, X, Y), atol=1e-5)

    def test_offsets_for_bounds_empty_is_zero_zero(self):
        from metashapes.shape.boolean import Intersection
        from metashapes.shape.primitives.quads import Rectangle

        lattice = Lattice.rectangular(1.0, 1.0)
        disjoint = Intersection(
            Rectangle(center=torch.tensor([0.0, 0.0]), size=torch.tensor([0.5, 0.5])),
            Rectangle(center=torch.tensor([5.0, 5.0]), size=torch.tensor([0.5, 0.5])),
        )
        cell = UnitCell(lattice, disjoint)
        assert cell._offsets_for_bounds(disjoint.bounds()) == ((0, 0), (0, 0))


class TestPeriodicSdfMatchesBruteForce:
    """sdf() must agree with a large brute-force reference search for a
    variety of lattices and shape placements -- not just be
    self-consistent with its own (tighter) offset range."""

    @pytest.mark.parametrize("lattice_factory", [
        lambda: Lattice.rectangular(1.0, 1.0),
        lambda: Lattice.hexagonal(1.0, orientation="pointy"),
        lambda: Lattice(a1=torch.tensor([1.0, 0.0]), a2=torch.tensor([0.7, 0.4])),
    ], ids=["rectangular", "hexagonal", "oblique"])
    @pytest.mark.parametrize("center,size", [
        ((0.5, 0.5), (0.2, 0.2)),   # cell-centred, typical after center_scene()
        ((0.0, 0.0), (0.2, 0.2)),   # straddling the seam
        ((0.5, 0.5), (2.5, 0.2)),   # spans multiple cells
        ((10.0, 10.0), (0.3, 0.3)), # far from the cell -- must still wrap in
    ])
    def test_matches_brute_force(self, lattice_factory, center, size):
        from metashapes.shape.primitives.quads import Rectangle

        lattice = lattice_factory()
        shape = Rectangle(center=torch.tensor(center), size=torch.tensor(size))
        cell = UnitCell(lattice, shape)

        xs = torch.linspace(-1.5, 2.5, 40)
        X, Y = torch.meshgrid(xs, xs, indexing="xy")
        got = cell.sdf(X, Y)

        # Reference ring: a fixed margin *on top of* whatever _offsets_for
        # itself reports, not a fixed constant -- a shape placed many cells
        # away (e.g. the (10, 10) case, especially on an oblique lattice)
        # can legitimately need a much wider search than a fixed ring of
        # 20 covers. This keeps the brute force a true superset of what
        # _offsets_for searches, so the comparison still catches an
        # undershoot rather than just failing due to its own insufficient
        # ring.
        (i0, i1), (j0, j1) = cell._offsets_for(shape)
        ring = max(abs(i0), abs(i1), abs(j0), abs(j1)) + 5
        ref = _brute_force_sdf(cell, X, Y, ring=ring)
        assert torch.allclose(got, ref, atol=1e-4), (
            f"max diff = {(got - ref).abs().max().item():.3e}"
        )


class TestOffsetSearchOverhead:
    def test_centered_small_shape_uses_nine_copies(self):
        # A small shape centred in the cell should need only ring-1
        # (9 copies), not a wider search.
        from metashapes.shape.primitives.quads import Rectangle

        lattice = Lattice.rectangular(1.0, 1.0)
        shape = Rectangle(center=torch.tensor([0.5, 0.5]), size=torch.tensor([0.2, 0.2]))
        cell = UnitCell(lattice, shape)
        (i0, i1), (j0, j1) = cell._offsets_for(shape)
        n_copies = (i1 - i0 + 1) * (j1 - j0 + 1)
        assert n_copies == 9


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
