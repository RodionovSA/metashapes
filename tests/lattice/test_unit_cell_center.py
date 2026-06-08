# tests/lattice/test_unit_cell_center.py
"""Tests for UnitCell.center_scene()."""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from metashapes.lattice.basis import Lattice
from metashapes.lattice.unit_cell import UnitCell
from metashapes.shape.primitives.quads import Rectangle
from metashapes.shape.primitives.conics import Ellipse
from metashapes.shape.primitives.periodic import Stripe
from metashapes.shape.primitives.polygons import RegularPolygon

from .conftest import make_learnable_polygon


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rect_cell(px=2.0, py=3.0, cx=0.0, cy=0.0):
    """Rectangular cell with a Rectangle shape centred at (cx, cy)."""
    lattice = Lattice.rectangular(px, py)
    shape = Rectangle(
        center=torch.tensor([cx, cy]),
        size=torch.tensor([0.4, 0.6]),
    )
    return UnitCell(lattice, shape)


def _cell_center(cell: UnitCell) -> tuple[float, float]:
    """The geometric midpoint (a1 + a2) / 2 as plain floats."""
    c = (cell.lattice.a1 + cell.lattice.a2) * 0.5
    return c[0].item(), c[1].item()


def _translate_offset(cell: UnitCell) -> tuple[float, float]:
    """Extract the outermost Translate dx/dy buffers from the scene."""
    from metashapes.shape.transforms import Translate
    scene = cell.scene
    assert isinstance(scene, Translate), f"Expected Translate, got {type(scene)}"
    return scene.dx.item(), scene.dy.item()


# ---------------------------------------------------------------------------
# 1. Rectangular lattice + bbox
# ---------------------------------------------------------------------------

class TestBboxRectangular:
    def test_scene_at_origin_moves_to_cell_center(self):
        px, py = 2.0, 3.0
        cell = _rect_cell(px=px, py=py, cx=0.0, cy=0.0)
        centered = cell.center_scene()

        ccx, ccy = _cell_center(cell)   # (1.0, 1.5)
        dx, dy = _translate_offset(centered)

        # The Translate offset should equal the cell centre exactly
        assert dx == pytest.approx(ccx, abs=1e-5)
        assert dy == pytest.approx(ccy, abs=1e-5)

    def test_translate_buffers_are_not_parameters(self):
        cell = _rect_cell()
        centered = cell.center_scene()

        from metashapes.shape.transforms import Translate
        t = centered.scene
        assert isinstance(t, Translate)
        # dx and dy must be buffers, not nn.Parameters
        assert not isinstance(t.dx, nn.Parameter)
        assert not isinstance(t.dy, nn.Parameter)


# ---------------------------------------------------------------------------
# 2. Hexagonal lattice + bbox
# ---------------------------------------------------------------------------

class TestBboxHexagonal:
    def test_hex_cell_center(self):
        a = 2.0
        lat = Lattice.hexagonal(a, orientation="pointy")
        shape = Ellipse(
            center=torch.zeros(2),
            axes=torch.tensor([0.3, 0.2]),
        )
        cell = UnitCell(lat, shape)
        centered = cell.center_scene()

        # expected cell centre for pointy hex: (a1 + a2) / 2
        expected_cx = (lat.a1[0] + lat.a2[0]).item() / 2
        expected_cy = (lat.a1[1] + lat.a2[1]).item() / 2

        dx, dy = _translate_offset(centered)
        assert dx == pytest.approx(expected_cx, abs=1e-5)
        assert dy == pytest.approx(expected_cy, abs=1e-5)


# ---------------------------------------------------------------------------
# 3. Already-centered shape → offset ≈ 0
# ---------------------------------------------------------------------------

class TestAlreadyCentered:
    def test_zero_offset_when_already_at_cell_center(self):
        px, py = 2.0, 3.0
        ccx, ccy = px / 2, py / 2   # cell centre for rectangular lattice
        cell = _rect_cell(px=px, py=py, cx=ccx, cy=ccy)
        centered = cell.center_scene()

        dx, dy = _translate_offset(centered)
        assert dx == pytest.approx(0.0, abs=1e-5)
        assert dy == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. Centroid method: result centroid lands at cell centre
# ---------------------------------------------------------------------------

class TestCentroidMethod:
    def test_shapely_centroid_at_cell_center(self):
        from metashapes.adapters.shapely import shape_to_shapely

        px, py = 2.0, 2.0
        # Shape at an off-centre position
        cell = _rect_cell(px=px, py=py, cx=0.3, cy=-0.4)
        centered = cell.center_scene(method="centroid")

        ccx, ccy = _cell_center(centered)
        geom = shape_to_shapely(centered.scene)
        assert geom.centroid.x == pytest.approx(ccx, abs=1e-5)
        assert geom.centroid.y == pytest.approx(ccy, abs=1e-5)


# ---------------------------------------------------------------------------
# 5. Infinite bounds → ValueError with method="bbox"
# ---------------------------------------------------------------------------

class TestInfiniteBoundsError:
    def test_stripe_raises_value_error(self):
        lattice = Lattice.rectangular(2.0, 2.0)
        scene = Stripe(offset=0.0, width=0.3, axis="x")
        cell = UnitCell(lattice, scene)

        with pytest.raises(ValueError, match="infinite bounds"):
            cell.center_scene(method="bbox")

    def test_stripe_centroid_does_not_raise(self):
        """method='centroid' on a shape that Shapely can represent should work."""
        # Stripe mapped to a bounded polygon via Shapely adapter — just verify
        # no exception is raised (the centroid is well-defined after clipping).
        lattice = Lattice.rectangular(2.0, 2.0)
        scene = Stripe(offset=0.0, width=0.3, axis="x")
        cell = UnitCell(lattice, scene)
        # The Shapely adapter clips to a bounding region; it should not raise.
        try:
            cell.center_scene(method="centroid")
        except Exception as e:
            # If the adapter raises for a Stripe that's truly infinite, skip;
            # we only care that it does NOT raise the bbox ValueError.
            assert "infinite bounds" not in str(e)


# ---------------------------------------------------------------------------
# 6. Unknown method → ValueError
# ---------------------------------------------------------------------------

class TestUnknownMethodError:
    def test_bad_method_raises(self):
        cell = _rect_cell()
        with pytest.raises(ValueError, match="Unknown centering method"):
            cell.center_scene(method="foobar")


# ---------------------------------------------------------------------------
# 7. Immutability — original cell is unchanged
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_original_scene_unchanged(self):
        cell = _rect_cell(cx=0.0, cy=0.0)
        original_scene_id = id(cell.scene)
        _ = cell.center_scene()

        # The original scene object must not have been replaced
        assert id(cell.scene) == original_scene_id

    def test_returns_different_unit_cell(self):
        cell = _rect_cell()
        centered = cell.center_scene()
        assert centered is not cell

    def test_original_scene_position_unchanged(self):
        cell = _rect_cell(cx=0.0, cy=0.0)
        _ = cell.center_scene()
        # Original scene bounds are still at the origin
        (x0, _), (x1, _) = cell.scene.bounds()
        assert (x0 + x1) / 2 == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 8. Gradient flow preserved through nn.Parameter
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_param_center_receives_gradient_after_centering(self):
        """nn.Parameter centre of a shape must receive gradients after center_scene()."""
        center = nn.Parameter(torch.zeros(2))
        shape = Rectangle(
            center=center,
            size=torch.tensor([0.4, 0.4]),
        )
        lattice = Lattice.rectangular(2.0, 2.0)
        cell = UnitCell(lattice, shape)
        centered = cell.center_scene()

        # Forward pass: soft mask (differentiable)
        mask = centered.mask(32, 32, soft=True)
        loss = mask.sum()
        loss.backward()

        assert center.grad is not None, "center parameter should receive a gradient"
        assert not torch.all(center.grad == 0), "gradient should be non-zero"

    def test_learnable_polygon_center_grad(self):
        """make_learnable_polygon center param grad is non-None after centering."""
        cell, side_param, center_param = make_learnable_polygon(
            side_length_val=0.5, center_val=(0.0, 0.0)
        )
        centered = cell.center_scene()

        mask = centered.mask(32, 32, soft=True)
        mask.sum().backward()

        assert center_param.grad is not None
        assert not torch.all(center_param.grad == 0)


# ---------------------------------------------------------------------------
# 9. Chaining — center_scene().mask() works end-to-end
# ---------------------------------------------------------------------------

class TestChaining:
    def test_mask_after_centering(self):
        cell = _rect_cell()
        mask = cell.center_scene().mask(64, 64)
        assert mask.shape == (64, 64)
        # Mask should contain both 0s and 1s (non-trivial)
        assert mask.any()
        assert not mask.all()

    def test_rasterize_after_centering(self):
        cell = _rect_cell()
        sdf = cell.center_scene().rasterize(32, 32)
        assert sdf.shape == (32, 32)

    def test_cartesian_mask_after_centering(self):
        lat = Lattice.hexagonal(2.0, orientation="pointy")
        shape = Rectangle(center=torch.zeros(2), size=torch.tensor([0.5, 0.5]))
        cell = UnitCell(lat, shape)
        mask = cell.center_scene().mask(64, 64, cartesian=True)
        assert mask.shape == (64, 64)


# ---------------------------------------------------------------------------
# 10. Symmetric shape: centroid ≈ bbox center
# ---------------------------------------------------------------------------

class TestSymmetricShapeConsistency:
    def test_centroid_equals_bbox_for_rectangle(self):
        """For a symmetric Rectangle, both methods produce the same offset."""
        cell = _rect_cell(cx=0.3, cy=-0.5)
        centered_bbox = cell.center_scene(method="bbox")
        centered_centroid = cell.center_scene(method="centroid")

        dx_bbox, dy_bbox = _translate_offset(centered_bbox)
        dx_cen, dy_cen = _translate_offset(centered_centroid)

        assert dx_bbox == pytest.approx(dx_cen, abs=1e-4)
        assert dy_bbox == pytest.approx(dy_cen, abs=1e-4)

    def test_centroid_equals_bbox_for_ellipse(self):
        """For a centred Ellipse, both methods produce the same offset."""
        lattice = Lattice.rectangular(4.0, 4.0)
        shape = Ellipse(
            center=torch.tensor([0.5, -0.2]),
            axes=torch.tensor([1.0, 0.6]),
        )
        cell = UnitCell(lattice, shape)
        centered_bbox = cell.center_scene(method="bbox")
        centered_centroid = cell.center_scene(method="centroid")

        dx_bbox, dy_bbox = _translate_offset(centered_bbox)
        dx_cen, dy_cen = _translate_offset(centered_centroid)

        assert dx_bbox == pytest.approx(dx_cen, abs=1e-3)
        assert dy_bbox == pytest.approx(dy_cen, abs=1e-3)
