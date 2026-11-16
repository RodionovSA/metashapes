# tests/gds/test_gds.py

from collections import Counter

import numpy as np
import pytest
import shapely
import torch
from shapely.geometry import Polygon

gdstk = pytest.importorskip("gdstk")

from metashapes import Lattice
from metashapes.adapters.shapely import shape_to_shapely
from metashapes.gds import (GDSAssembly, Units, layout_to_cell, polygons_from_shapely,
                            save_gds, shape_to_gdstk, unit_cell_to_cell)
from metashapes.lattice.unit_cell import UnitCell
from metashapes.layout import Layout, LayoutBatchResult
from metashapes.shape import Bar, Ellipse, Rectangle, Rotate, Scale, ShapeSet, Star, Translate

NM = Units(coord_unit=1e-9)          # model in nm, file in µm with 1 nm grid


def _to_shapely(polys, scale=1.0):
    return shapely.union_all([Polygon(np.asarray(p.points) / scale) for p in polys])


def _read(path):
    lib = gdstk.read_gds(str(path))
    return lib, {c.name: c for c in lib.cells}


def _flat(cell):
    c = cell.copy("flat")
    c.flatten()
    return c.polygons


def _disc(r=40.0):
    return Ellipse(center=(0.0, 0.0), axes=(2 * r, 2 * r))


# --- units / conversion ---------------------------------------------------------
class TestUnits:
    def test_scale_and_grid(self):
        assert NM.scale == pytest.approx(1e-3)
        assert NM.grid_model == pytest.approx(1.0)
        assert Units().grid_model == pytest.approx(1e-3)
        assert NM.default_tolerance() == pytest.approx(1.0)

    def test_validation(self):
        with pytest.raises(ValueError):
            Units(coord_unit=0)
        with pytest.raises(ValueError):
            Units(unit=1e-9, precision=1e-6)


class TestPolygonsFromShapely:
    def test_scaling_and_layer(self):
        polys = polygons_from_shapely(shapely.box(0, 0, 1000, 500), NM, layer=3, datatype=2)
        assert len(polys) == 1
        p = polys[0]
        assert (p.layer, p.datatype) == (3, 2)
        assert p.area() == pytest.approx(0.5)            # 1 µm x 0.5 µm

    def test_hole_becomes_single_cut_polygon(self):
        g = shapely.box(0, 0, 100, 100).difference(shapely.box(30, 30, 60, 60))
        polys = polygons_from_shapely(g, Units())
        assert len(polys) == 1
        assert sum(p.area() for p in polys) == pytest.approx(g.area)

    def test_multipolygon_collection_and_empty(self):
        g = shapely.GeometryCollection([shapely.box(0, 0, 1, 1), shapely.box(3, 0, 4, 1),
                                        shapely.LineString([(0, 0), (5, 5)])])
        assert len(polygons_from_shapely(g, Units())) == 2
        assert polygons_from_shapely(Polygon(), Units()) == []

    def test_subgrid_pieces_dropped(self):
        g = shapely.box(0, 0, 0.5, 0.5)                   # 0.5 nm square on a 1 nm grid
        assert polygons_from_shapely(g, NM) == []


class TestShapeToGdstk:
    def test_area_matches_within_tolerance(self):
        e = Ellipse(center=(10.0, 0.0), axes=(400.0, 200.0)).to(torch.float64)
        polys = shape_to_gdstk(e, NM, tolerance=0.5)
        area_nm2 = sum(p.area() for p in polys) / NM.scale ** 2
        perim = shape_to_shapely(e).length
        assert abs(area_nm2 - np.pi * 200 * 100) <= 0.5 * perim

    def test_default_tolerance_is_1nm(self):
        coarse = shape_to_gdstk(_disc(5000.0), NM, tolerance=50.0)[0]
        fine = shape_to_gdstk(_disc(5000.0), NM)[0]
        assert len(fine.points) > len(coarse.points)


# --- cells ------------------------------------------------------------------------
class TestUnitCellToCell:
    def test_one_period_unclipped(self):
        # disc crosses the cell boundary: exported whole, not wrapped
        uc = UnitCell(Lattice.rectangular(100.0, 100.0), Ellipse(center=(95.0, 50.0), axes=(40.0, 40.0)))
        cell = unit_cell_to_cell(uc, "uc", NM, tolerance=0.2)
        assert len(cell.polygons) == 1
        (x0, _), (x1, _) = cell.polygons[0].bounding_box()
        assert x1 == pytest.approx(0.115, abs=1e-3)       # sticks out past x=100 nm

    def test_infinite_scene_is_wrapped(self):
        uc = UnitCell(Lattice.rectangular(500.0, 500.0), Bar(offset=250.0, width=100.0, axis="x"))
        cell = unit_cell_to_cell(uc, "g", NM)
        (x0, y0), (x1, y1) = cell.bounding_box()
        assert (x0, x1) == pytest.approx((0.0, 0.5))
        assert (y0, y1) == pytest.approx((0.2, 0.3))


class TestLayoutToCell:
    def _layout(self, extra=()):
        atoms = [Translate(_disc(), x, y) for x in (-200.0, 0.0, 200.0) for y in (-200.0, 0.0, 200.0)]
        atoms += list(extra)
        return Layout(Rectangle(center=(0.0, 0.0), size=(1000.0, 1000.0)), ShapeSet(atoms))

    def test_dedupe_uses_references(self):
        unique = Translate(Rectangle(center=(0.0, 0.0), size=(60.0, 30.0)), 350.0, 350.0)
        cell = layout_to_cell(self._layout([unique]), "L", NM, tolerance=0.5)
        assert len(cell.references) == 9
        assert len({r.cell.name for r in cell.references}) == 1
        assert len(cell.polygons) == 1                  # the unique atom stays flat
        no = layout_to_cell(self._layout([unique]), "L", NM, tolerance=0.5, dedupe=False)
        assert len(no.references) == 0 and len(no.polygons) == 10

    def test_dedupe_snaps_to_grid(self):
        a = Translate(Ellipse(center=(0.0, 0.0), axes=(80.0, 80.0)), 0.0, 0.0)
        b = Translate(Ellipse(center=(0.0, 0.0), axes=(80.0 + 1e-4, 80.0)), 300.0, 0.0)
        lay = Layout(Rectangle(center=(0.0, 0.0), size=(1000.0, 1000.0)), ShapeSet([a, b]))
        assert len(layout_to_cell(lay, "L", NM).references) == 2

    def _two(self, a, b):
        lay = Layout(Rectangle(center=(0.0, 0.0), size=(2000.0, 2000.0)),
                     ShapeSet([Translate(a, -400.0, 0.0), Translate(b, 400.0, 0.0)]))
        return layout_to_cell(lay, "L", NM, tolerance=0.5)

    def test_dedupe_keeps_different_rotations_apart(self):
        rect = lambda: Rectangle(center=(0.0, 0.0), size=(200.0, 50.0))
        cell = self._two(Rotate(rect(), 0.1), Rotate(rect(), 0.4))
        assert len(cell.references) == 0 and len(cell.polygons) == 2
        # 1e-3 rad moves the 100 nm half-length by 0.1 nm, below the 1 nm grid: merged
        cell = self._two(Rotate(rect(), 0.1), Rotate(rect(), 0.1 + 1e-3))
        assert len(cell.references) == 2
        # 0.05 rad moves the corner by ~5 nm: must stay separate
        cell = self._two(Rotate(rect(), 0.1), Rotate(rect(), 0.15))
        assert len(cell.references) == 0
        # the primitive's own angle parameter too
        a = Rectangle(center=(0.0, 0.0), size=(200.0, 50.0), angle=0.1)
        b = Rectangle(center=(0.0, 0.0), size=(200.0, 50.0), angle=0.4)
        assert len(self._two(a, b).references) == 0

    def test_dedupe_keeps_different_scales_apart(self):
        cell = self._two(Scale(_disc(), 1.5), Scale(_disc(), 2.0))
        assert len(cell.references) == 0 and len(cell.polygons) == 2

    def test_clips_only_crossing_atoms(self):
        crossing = Translate(Rectangle(center=(0.0, 0.0), size=(100.0, 40.0)), 500.0, 0.0)
        outside = Translate(_disc(), 900.0, 0.0)
        cell = layout_to_cell(self._layout([crossing, outside]), "L", NM, tolerance=0.5)
        flat = _flat(cell)
        assert len(flat) == 10
        xs = np.concatenate([p.points[:, 0] for p in flat])
        assert xs.max() == pytest.approx(0.5, abs=1e-6)  # clipped at the domain edge (500 nm)
        clipped = [p for p in cell.polygons]
        assert clipped[0].area() == pytest.approx(0.05 * 0.04, rel=1e-6)

    def test_no_clip_keeps_everything(self):
        lay = self._layout([Translate(_disc(), 900.0, 0.0)])
        lay.clip = False
        assert len(_flat(layout_to_cell(lay, "L", NM))) == 10

    def test_area_matches_shapely_layout(self):
        atoms = [Translate(Star(center=(0.0, 0.0), n=5, outer_radius=50.0, inner_radius=25.0,
                                outer_corner_radius=5.0, inner_corner_radius=3.0), x, 0.0)
                 for x in (-300.0, 0.0, 300.0)]
        atoms.append(Translate(_disc(30.0), 480.0, 0.0))
        lay = Layout(Ellipse(center=(0.0, 0.0), axes=(1000.0, 1000.0)), ShapeSet(atoms))
        ref = lay.to_shapely().area
        got = _to_shapely(_flat(layout_to_cell(lay, "L", NM, tolerance=0.05)), NM.scale).area
        assert got == pytest.approx(ref, rel=2e-3)

    def test_empty(self):
        lay = Layout(Rectangle(center=(0.0, 0.0), size=(10.0, 10.0)), ShapeSet([]))
        assert len(layout_to_cell(lay, "E", NM).polygons) == 0


# --- assembly ---------------------------------------------------------------------
def _lens(domain_r=500.0):
    atoms = [Translate(_disc(), x, y) for x in (-150.0, 150.0) for y in (-150.0, 150.0)]
    return Layout(Ellipse(center=(0.0, 0.0), axes=(2 * domain_r, 2 * domain_r)), ShapeSet(atoms))


def _hex_cell():
    return UnitCell(Lattice.hexagonal(300.0), Ellipse(center=(150.0, 86.6), axes=(200.0, 120.0)))


class TestPlacement:
    def test_layout_at_moves_own_frame(self):
        chip = GDSAssembly(coord_unit=1e-9)
        p = chip.place(_lens(), at=(1000.0, -500.0))
        assert chip.footprint(p).bounds == pytest.approx((500.0, -1000.0, 1500.0, 0.0), abs=1e-6)

    def test_array_anchor_center_and_origin(self):
        uc = UnitCell(Lattice.rectangular(100.0, 50.0), Rectangle(center=(50.0, 25.0), size=(20.0, 10.0)))
        chip = GDSAssembly(coord_unit=1e-9)
        c = chip.place_array(uc, at=(0.0, 0.0), repeats=(10, 4), name="c")
        o = chip.place_array(uc, at=(5000.0, 0.0), repeats=(10, 4), anchor="origin", name="o")
        assert c.offset == pytest.approx((-500.0, -100.0))
        assert o.offset == pytest.approx((5000.0, 0.0))
        assert chip.footprint(c).bounds == pytest.approx((-500.0, -100.0, 500.0, 100.0))

    def test_array_footprint_covers_protruding_atoms(self):
        uc = UnitCell(Lattice.rectangular(100.0, 100.0), Ellipse(center=(95.0, 50.0), axes=(40.0, 40.0)))
        chip = GDSAssembly(coord_unit=1e-9)
        p = chip.place_array(uc, repeats=(3, 3), anchor="origin")
        x0, y0, x1, y1 = chip.footprint(p).bounds
        assert x1 == pytest.approx(315.0) and x0 == pytest.approx(0.0)

    def test_names_and_type_errors(self):
        chip = GDSAssembly(coord_unit=1e-9)
        chip.place(_lens(), name="lens A")
        assert chip.placements[0].name == "lens_A"
        with pytest.raises(ValueError, match="duplicate"):
            chip.place(_lens(), name="lens_A")
        with pytest.raises(TypeError):
            chip.place(_hex_cell())
        with pytest.raises(TypeError):
            chip.place_array(_lens(), repeats=(2, 2))
        with pytest.raises(ValueError):
            chip.place_array(_hex_cell(), repeats=(0, 2))
        with pytest.raises(ValueError):
            chip.place_array(_hex_cell(), repeats=(2, 2), anchor="corner")

    def test_place_grid_batches(self):
        chip = GDSAssembly(coord_unit=1e-9)
        batch = LayoutBatchResult(layouts=[_lens(), _lens(), _lens()], report=None)
        ps = chip.place_grid(batch, origin=(0.0, 0.0), pitch=(2000.0, 2000.0), cols=2,
                             texts=["a", "b", "c"])
        centres = [c for p in ps for c in shapely.centroid(chip.footprint(p)).coords[0]]
        assert centres == pytest.approx([0.0, 0.0, 2000.0, 0.0, 0.0, -2000.0], abs=1e-6)
        cells = chip.place_grid([_hex_cell(), _hex_cell()], origin=(0.0, 5000.0),
                                pitch=(5000.0, 5000.0), cols=2, repeats=(4, 4), name_prefix="h")
        b = chip.footprint(cells[1]).bounds
        assert 0.5 * (b[0] + b[2]) == pytest.approx(5000.0)
        with pytest.raises(ValueError, match="repeats"):
            chip.place_grid([_hex_cell()], pitch=(1.0, 1.0), cols=1, name_prefix="z")


class TestCheck:
    def test_overlap_spacing_touching(self):
        chip = GDSAssembly(coord_unit=1e-9)
        sq = lambda: Layout(Rectangle(center=(0.0, 0.0), size=(100.0, 100.0)), ShapeSet([]))
        chip.place(sq(), (0.0, 0.0), name="a")
        chip.place(sq(), (100.0, 0.0), name="b")          # touching
        chip.place(sq(), (130.0, 150.0), name="c")        # 50 above b
        assert chip.check() == []                          # touching is not an overlap
        msgs = chip.check(min_spacing=60.0)                # ...but is 0 apart
        assert len(msgs) == 3 and all(m.startswith("spacing") for m in msgs)
        assert any("'b' and 'c' are 50 apart" in m for m in msgs)
        chip.place(sq(), (50.0, 0.0), name="d")           # overlaps a and b
        assert sum(m.startswith("overlap") for m in chip.check()) == 2

    def test_write_raises_and_allow_overlap(self, tmp_path):
        chip = GDSAssembly(coord_unit=1e-9)
        chip.place(_lens(), (0.0, 0.0), name="a")
        chip.place(_lens(), (500.0, 0.0), name="b")
        with pytest.raises(ValueError, match="overlap"):
            chip.write_gds(tmp_path / "x.gds")
        chip.write_gds(tmp_path / "x.gds", allow_overlap=True)
        assert (tmp_path / "x.gds").exists()

    def test_geometry_check_clears_conservative_footprints(self):
        # atoms protrude past the cell edge, so the hull footprints of two
        # arrays placed edge to edge overlap, but the actual atoms don't
        # disc spans x in [75, 115] of a 100-wide cell: the left array's hull
        # reaches x=315, the right array's atoms start at x=375
        uc = UnitCell(Lattice.rectangular(100.0, 100.0), Ellipse(center=(95.0, 50.0), axes=(40.0, 40.0)))
        chip = GDSAssembly(coord_unit=1e-9)
        chip.place_array(uc, (0.0, 0.0), repeats=(3, 3), anchor="origin", name="l")
        chip.place_array(uc, (300.0, 0.0), repeats=(3, 3), anchor="origin", name="r")
        assert any(m.startswith("overlap") for m in chip.check())
        assert chip.check(geometry=True) == []
        # a real collision survives the geometry check: right atoms at [305, 345]
        chip2 = GDSAssembly(coord_unit=1e-9)
        chip2.place_array(uc, (0.0, 0.0), repeats=(3, 3), anchor="origin", name="l")
        chip2.place_array(uc, (230.0, 0.0), repeats=(3, 3), anchor="origin", name="r")
        assert any(m.startswith("overlap") for m in chip2.check(geometry=True))


class TestWrite:
    def test_cell_tree_and_shared_cells(self, tmp_path):
        lens = _lens()
        chip = GDSAssembly(coord_unit=1e-9)
        chip.place(lens, (0.0, 0.0), name="l1")
        chip.place(lens, (3000.0, 0.0), name="l2")              # same object -> one cell
        chip.place_array(_hex_cell(), (0.0, 4000.0), repeats=(5, 4), name="hex")
        chip.write_gds(tmp_path / "c.gds")
        lib, cells = _read(tmp_path / "c.gds")
        top = cells["TOP"]
        names = Counter(r.cell.name for r in top.references)
        assert names == {"l1": 2, "hex": 4}                     # hexagonal: one AREF per row
        assert "l2" not in cells
        assert all(r.repetition.size == 5 for r in top.references if r.cell.name == "hex")

    @pytest.mark.parametrize("lat", [Lattice.rectangular(300.0, 200.0), Lattice.hexagonal(300.0),
                                     Lattice(a1=(260.0, 150.0), a2=(0.0, 300.0)),
                                     Lattice(a1=(260.0, 150.0), a2=(-100.0, 280.0))])
    def test_array_positions(self, tmp_path, lat):
        uc = UnitCell(lat, Ellipse(center=(50.0, 50.0), axes=(60.0, 40.0)))
        chip = GDSAssembly(coord_unit=1e-9)
        chip.place_array(uc, repeats=(7, 5), anchor="origin", name="a")
        chip.write_gds(tmp_path / "a.gds")
        _, cells = _read(tmp_path / "a.gds")
        got = []
        for ref in cells["TOP"].references:
            offs = ref.repetition.get_offsets() if ref.repetition.size else np.zeros((1, 2))
            got += list(np.asarray(ref.origin) + offs)
        a1 = lat.a1.double().numpy() * 1e-3
        a2 = lat.a2.double().numpy() * 1e-3
        want = np.array([i * a1 + j * a2 for i in range(7) for j in range(5)])
        got = np.array(got)
        assert len(got) == 35
        d = np.abs(got[:, None, :] - want[None]).max(-1)
        assert d.min(1).max() < 1e-3 and (d.min(0) < 1e-3).all()   # within the 1 nm grid

    def test_flattened_geometry_matches_model(self, tmp_path):
        lens = _lens()
        chip = GDSAssembly(coord_unit=1e-9, tolerance=0.1)
        chip.place(lens, (2000.0, 0.0), name="l")
        chip.write_gds(tmp_path / "g.gds")
        _, cells = _read(tmp_path / "g.gds")
        got = _to_shapely(_flat(cells["TOP"]), 1e-3)
        want = shapely.affinity.translate(lens.to_shapely(), 2000.0, 0.0)
        # differences come from snapping to the 1 nm grid: <= perimeter * 1 nm
        assert got.symmetric_difference(want).area < want.length * 1.0

    def test_text_on_text_layer_only(self, tmp_path):
        chip = GDSAssembly(coord_unit=1e-9, text_layer=(63, 5))
        chip.place(_lens(), name="l", text="lens A")
        chip.write_gds(tmp_path / "t.gds")
        _, cells = _read(tmp_path / "t.gds")
        labels = cells["TOP"].labels
        assert [(lb.text, lb.layer, lb.texttype) for lb in labels] == [("lens A", 63, 5)]
        assert labels[0].origin == pytest.approx((0.0, 0.0))
        assert all(p.layer == 0 for p in _flat(cells["TOP"]))

    def test_text_as_polygons(self, tmp_path):
        chip = GDSAssembly(coord_unit=1e-9, text_layer=(63, 0), text_as_polygons=True)
        chip.place(_lens(), name="l", text="A1", text_anchor="top")
        chip.write_gds(tmp_path / "t.gds")
        _, cells = _read(tmp_path / "t.gds")
        assert any(p.layer == 63 for p in cells["TOP"].polygons)
        assert not cells["TOP"].labels

    def test_layers(self, tmp_path):
        chip = GDSAssembly(coord_unit=1e-9)
        chip.place(_lens(), name="l", layer=(7, 3))
        chip.write_gds(tmp_path / "l.gds")
        _, cells = _read(tmp_path / "l.gds")
        assert {(p.layer, p.datatype) for p in _flat(cells["TOP"])} == {(7, 3)}

    def test_vertex_limit(self, tmp_path):
        big = Layout(Rectangle(center=(0.0, 0.0), size=(1e6, 1e6)),
                     ShapeSet([_disc(4e5)]))
        chip = GDSAssembly(coord_unit=1e-9, tolerance=0.01, max_points=8190)
        chip.place(big, name="b")
        chip.write_gds(tmp_path / "v.gds")
        _, cells = _read(tmp_path / "v.gds")
        assert all(len(p.points) <= 8190 for p in _flat(cells["TOP"]))

    def test_save_gds(self, tmp_path):
        save_gds(tmp_path / "s.gds", _lens(), coord_unit=1e-9)
        save_gds(tmp_path / "a.gds", _hex_cell(), repeats=(3, 3), coord_unit=1e-9)
        with pytest.raises(ValueError, match="repeats"):
            save_gds(tmp_path / "b.gds", _hex_cell(), coord_unit=1e-9)
        _, cells = _read(tmp_path / "a.gds")
        assert "array" in cells


def test_workers_match_serial(monkeypatch):
    import metashapes.gds.cells as cells
    monkeypatch.setattr(cells, "_MIN_PER_WORKER", 2)
    atoms = [Translate(Star(center=(0.0, 0.0), n=5, outer_radius=40.0 + i, inner_radius=20.0,
                            outer_corner_radius=4.0, inner_corner_radius=2.0), 120.0 * i, 0.0)
             for i in range(12)]
    lay = Layout(Rectangle(center=(660.0, 0.0), size=(2000.0, 400.0)), ShapeSet(atoms))
    ser = layout_to_cell(lay, "L", NM)
    par = layout_to_cell(lay, "L", NM, workers=3)
    key = lambda c: sorted(np.round(p.points, 9).tobytes() for p in c.polygons)
    assert len(par.polygons) == 12 and key(par) == key(ser)
