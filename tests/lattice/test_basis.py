# tests/lattice/test_basis.py

import math
import pytest
import torch

from metashapes.lattice.basis import Lattice


class TestLatticeConstruction:
    def test_rectangular_vectors(self):
        lat = Lattice.rectangular(3.0, 5.0)
        assert torch.allclose(lat.a1, torch.tensor([3.0, 0.0]))
        assert torch.allclose(lat.a2, torch.tensor([0.0, 5.0]))

    def test_hexagonal_pointy_vectors(self):
        a = 2.0
        lat = Lattice.hexagonal(a, orientation="pointy")
        assert torch.allclose(lat.a1, torch.tensor([a, 0.0]))
        assert torch.allclose(lat.a2, torch.tensor([a / 2, a * math.sqrt(3) / 2]))

    def test_hexagonal_flat_vectors(self):
        a = 2.0
        lat = Lattice.hexagonal(a, orientation="flat")
        assert torch.allclose(lat.a1, torch.tensor([a * math.sqrt(3) / 2, a / 2]))
        assert torch.allclose(lat.a2, torch.tensor([0.0, a]))

    def test_hexagonal_invalid_orientation(self):
        with pytest.raises(ValueError, match="orientation"):
            Lattice.hexagonal(1.0, orientation="diagonal")

    def test_singular_raises(self):
        with pytest.raises(ValueError):
            Lattice(a1=torch.tensor([1.0, 0.0]), a2=torch.tensor([2.0, 0.0]))


class TestLatticeProperties:
    def test_matrix_columns(self):
        lat = Lattice.rectangular(3.0, 5.0)
        M = lat.matrix
        assert torch.allclose(M[:, 0], lat.a1)
        assert torch.allclose(M[:, 1], lat.a2)

    def test_cell_area_rectangular(self):
        px, py = 3.0, 5.0
        lat = Lattice.rectangular(px, py)
        assert lat.cell_area.item() == pytest.approx(px * py, rel=1e-5)

    def test_cell_area_hexagonal(self):
        a = 2.0
        lat = Lattice.hexagonal(a, orientation="pointy")
        expected = a ** 2 * math.sqrt(3) / 2
        assert lat.cell_area.item() == pytest.approx(expected, rel=1e-5)

    def test_device_property(self):
        lat = Lattice.rectangular(1.0, 1.0)
        assert lat.device == torch.device("cpu")

    def test_dtype_property(self):
        lat = Lattice.rectangular(1.0, 1.0)
        assert lat.dtype == torch.float32


class TestLatticeCoordinates:
    def test_round_trip_fractional_cartesian(self):
        lat = Lattice.rectangular(3.0, 4.0)
        x = torch.tensor([0.5, 1.0, -1.5, 2.0])
        y = torch.tensor([0.2, -0.3, 1.1, 0.0])
        f1, f2 = lat.to_fractional(x, y)
        xr, yr = lat.to_cartesian(f1, f2)
        assert torch.allclose(xr, x, atol=1e-5)
        assert torch.allclose(yr, y, atol=1e-5)

    def test_round_trip_oblique(self):
        lat = Lattice.hexagonal(1.0, orientation="pointy")
        x = torch.tensor([0.3, -0.2, 0.8])
        y = torch.tensor([0.1,  0.5, -0.1])
        f1, f2 = lat.to_fractional(x, y)
        xr, yr = lat.to_cartesian(f1, f2)
        assert torch.allclose(xr, x, atol=1e-5)
        assert torch.allclose(yr, y, atol=1e-5)

    def test_fractional_of_a1(self):
        lat = Lattice.rectangular(3.0, 5.0)
        f1, f2 = lat.to_fractional(lat.a1[0:1], lat.a1[1:2])
        assert f1.item() == pytest.approx(1.0, abs=1e-5)
        assert f2.item() == pytest.approx(0.0, abs=1e-5)

    def test_fractional_of_a2(self):
        lat = Lattice.rectangular(3.0, 5.0)
        f1, f2 = lat.to_fractional(lat.a2[0:1], lat.a2[1:2])
        assert f1.item() == pytest.approx(0.0, abs=1e-5)
        assert f2.item() == pytest.approx(1.0, abs=1e-5)

    def test_cartesian_of_unit_f1(self):
        lat = Lattice.rectangular(3.0, 5.0)
        x, y = lat.to_cartesian(torch.tensor([1.0]), torch.tensor([0.0]))
        assert torch.allclose(torch.stack([x, y], dim=-1), lat.a1.unsqueeze(0), atol=1e-5)

    def test_cartesian_of_unit_f2(self):
        lat = Lattice.rectangular(3.0, 5.0)
        x, y = lat.to_cartesian(torch.tensor([0.0]), torch.tensor([1.0]))
        assert torch.allclose(torch.stack([x, y], dim=-1), lat.a2.unsqueeze(0), atol=1e-5)


class TestLatticeOffset:
    def test_offset_origin(self):
        lat = Lattice.rectangular(2.0, 3.0)
        off = lat.offset(0, 0)
        assert torch.allclose(off, torch.zeros(2))

    def test_offset_a1(self):
        lat = Lattice.rectangular(2.0, 3.0)
        assert torch.allclose(lat.offset(1, 0), lat.a1)

    def test_offset_a2(self):
        lat = Lattice.rectangular(2.0, 3.0)
        assert torch.allclose(lat.offset(0, 1), lat.a2)

    def test_offset_negative(self):
        lat = Lattice.rectangular(2.0, 3.0)
        assert torch.allclose(lat.offset(-1, -1), -lat.a1 - lat.a2)

    def test_offset_combined(self):
        lat = Lattice.rectangular(2.0, 3.0)
        assert torch.allclose(lat.offset(2, -3), 2 * lat.a1 - 3 * lat.a2)


class TestNeighborOffsets:
    def test_ring1_count(self):
        lat = Lattice.rectangular(1.0, 1.0)
        offsets = lat.neighbor_offsets(ring=1)
        assert len(offsets) == 9

    def test_ring2_count(self):
        lat = Lattice.rectangular(1.0, 1.0)
        offsets = lat.neighbor_offsets(ring=2)
        assert len(offsets) == 25

    def test_ring1_contains_origin(self):
        lat = Lattice.rectangular(1.0, 1.0)
        offsets = lat.neighbor_offsets(ring=1)
        assert (0, 0) in offsets

    def test_ring0_is_origin_only(self):
        lat = Lattice.rectangular(1.0, 1.0)
        offsets = lat.neighbor_offsets(ring=0)
        assert offsets == [(0, 0)]
