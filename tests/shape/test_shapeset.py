# tests/shape/test_shapeset.py

import math
import random

import pytest
import torch
import torch.nn as nn

from metashapes.shape import (
    Shape, ShapeSet, Union, Difference, Rectangle, Ellipse, Star, Translate, Rotate, Bar,
    make_learnable,
)
from metashapes.shape.base import EMPTY_BOUNDS
from metashapes.shape.spatial import SpatialHash, ragged_arange


def _atoms(n=30, seed=0, extent=100.0, stars=False):
    rng = random.Random(seed)
    atoms = []
    for i in range(n):
        x, y = rng.uniform(0, extent), rng.uniform(0, extent)
        a, b = rng.uniform(3, 8), rng.uniform(3, 8)
        ang = rng.uniform(0, 90)
        kind = i % 3
        if kind == 2 and not stars:
            atoms.append(Translate(Ellipse(center=(0.0, 0.0), axes=(b, a)), x, y))
        elif kind == 0:
            atoms.append(Rectangle(center=(x, y), size=(a, b), angle=ang, corner_radius=0.5))
        elif kind == 1:
            atoms.append(Translate(Ellipse(center=(0.0, 0.0), axes=(a, b), angle=ang), x, y))
        else:
            atoms.append(Translate(Rotate(Star(center=(0.0, 0.0), n=5, outer_radius=a,
                                               inner_radius=0.5 * a), ang), x, y))
    return atoms


def _union(atoms):
    u = atoms[0]
    for a in atoms[1:]:
        u = Union(u, a)
    return u


@pytest.fixture
def scene():
    atoms = _atoms()
    return ShapeSet(atoms), _union(atoms)


@pytest.fixture
def grid():
    xs = torch.linspace(-30.0, 130.0, 161)
    ys = torch.linspace(-25.0, 125.0, 151)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    return xs, ys, X, Y


# ---------------------------------------------------------------------------
# SpatialHash
# ---------------------------------------------------------------------------

class TestSpatialHash:
    def test_ragged_arange(self):
        assert ragged_arange(torch.tensor([2, 0, 3])).tolist() == [0, 1, 0, 1, 2]
        assert ragged_arange(torch.tensor([], dtype=torch.long)).numel() == 0

    def test_pairs_match_brute_force(self):
        torch.manual_seed(0)
        lo = torch.rand(50, 2) * 100
        boxes = torch.cat([lo, lo + torch.rand(50, 2) * 10], dim=1)
        px, py = torch.rand(2000) * 120 - 10, torch.rand(2000) * 120 - 10
        idx = SpatialHash(boxes, cell_size=7.0)
        for pad in (0.0, 3.0, 25.0):
            a, p = idx.pairs(px, py, pad)
            got = set(zip(a.tolist(), p.tolist()))
            inside = ((px[None] >= boxes[:, 0:1] - pad) & (px[None] <= boxes[:, 2:3] + pad)
                      & (py[None] >= boxes[:, 1:2] - pad) & (py[None] <= boxes[:, 3:4] + pad))
            want = set(map(tuple, torch.nonzero(inside).tolist()))
            assert got == want
            assert a.tolist() == sorted(a.tolist())

    def test_invalid_boxes_skipped(self):
        boxes = torch.tensor([[0, 0, 1, 1], [math.inf, math.inf, -math.inf, -math.inf],
                              [5, 5, 6, 6]], dtype=torch.float64)
        idx = SpatialHash(boxes)
        assert len(idx) == 2
        a, _ = idx.pairs(torch.tensor([0.5, 5.5]), torch.tensor([0.5, 5.5]), 0.0)
        assert sorted(a.tolist()) == [0, 2]

    def test_grid_windows(self):
        idx = SpatialHash(torch.tensor([[2.0, 3.0, 4.0, 5.0]]))
        xs = torch.arange(10.0)
        ys = torch.arange(8.0)
        (atom, i0, i1, j0, j1), = idx.grid_windows(xs, ys, pad=1.0)
        assert (atom, i0, i1, j0, j1) == (0, 1, 6, 2, 7)


# ---------------------------------------------------------------------------
# ShapeSet SDF
# ---------------------------------------------------------------------------

class TestShapeSetSDF:
    def test_exact_points_match_union(self, scene, grid):
        ss, u = scene
        _, _, X, Y = grid
        assert torch.allclose(ss.sdf(X, Y), u.sdf(X, Y), atol=1e-4)

    def test_exact_grid_match_union_including_empty_regions(self, scene, grid):
        ss, u = scene
        xs, ys, X, Y = grid
        # far outside every atom: exercises the expanding-radius search
        far_x = torch.linspace(-500.0, 600.0, 23)
        far_y = torch.linspace(-400.0, 450.0, 19)
        FX, FY = torch.meshgrid(far_x, far_y, indexing="xy")
        assert torch.allclose(ss.sdf_grid(xs, ys), u.sdf(X, Y), atol=1e-4)
        assert torch.allclose(ss.sdf_grid(far_x, far_y), u.sdf(FX, FY), atol=1e-3)
        assert torch.allclose(ss.sdf(FX, FY), u.sdf(FX, FY), atol=1e-3)

    @pytest.mark.parametrize("halo", [0.5, 2.0, 10.0])
    def test_capped_matches_clamped_union(self, scene, grid, halo):
        ss, u = scene
        xs, ys, X, Y = grid
        ref = u.sdf(X, Y).clamp(max=halo)
        assert torch.allclose(ss.sdf(X, Y, halo=halo), ref, atol=1e-5)
        assert torch.allclose(ss.sdf_grid(xs, ys, halo=halo), ref, atol=1e-5)
        # beyond the halo the value is exactly +halo
        far = u.sdf(X, Y) > halo
        assert torch.all(ss.sdf_grid(xs, ys, halo=halo)[far] == halo)

    def test_hard_masks_identical(self, scene, grid):
        ss, u = scene
        xs, ys, X, Y = grid
        ref = u.sdf(X, Y) <= 0
        assert torch.equal(ss.sdf_grid(xs, ys, halo=1.0) <= 0, ref)
        assert torch.equal(ss.sdf(X, Y, halo=1.0) <= 0, ref)

    def test_stored_halo_is_default(self, scene, grid):
        _, u = scene
        ss = ShapeSet(_atoms(), halo=3.0)
        _, _, X, Y = grid
        assert torch.allclose(ss.sdf(X, Y), u.sdf(X, Y).clamp(max=3.0), atol=1e-5)
        # explicit per-call override wins
        assert torch.allclose(ss.sdf(X, Y, halo=math.inf), u.sdf(X, Y), atol=1e-4)

    def test_broadcast_and_scalar_inputs(self, scene):
        ss, u = scene
        x = torch.linspace(0, 100, 7)[None, :]
        y = torch.linspace(0, 100, 5)[:, None]
        assert ss.sdf(x, y).shape == (5, 7)
        assert torch.allclose(ss.sdf(x, y), u.sdf(x, y), atol=1e-4)
        p = ss.sdf(torch.tensor(50.0), torch.tensor(50.0))
        assert p.shape == ()

    def test_overlapping_atoms_merge(self):
        a = Rectangle(center=(0.0, 0.0), size=(2.0, 2.0))
        b = Rectangle(center=(1.0, 0.0), size=(2.0, 2.0))
        ss = ShapeSet([a, b])
        xs = torch.linspace(-3, 4, 71)
        ys = torch.linspace(-2, 2, 41)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        assert torch.allclose(ss.sdf_grid(xs, ys), Union(a, b).sdf(X, Y), atol=1e-5)

    def test_empty_set(self):
        ss = ShapeSet([])
        assert len(ss) == 0
        assert ss.bounds() == EMPTY_BOUNDS
        assert ss.min_feature_size is None
        assert torch.all(torch.isinf(ss.sdf(torch.zeros(3), torch.zeros(3))))
        assert torch.all(ss.sdf(torch.zeros(3), torch.zeros(3), halo=2.0) == 2.0)
        assert ss.sdf_grid(torch.arange(4.0), torch.arange(3.0), halo=1.0).shape == (3, 4)

    def test_rejects_infinite_atoms(self):
        with pytest.raises(ValueError, match="infinite"):
            ShapeSet([Bar(offset=0.0, width=1.0)])

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            ShapeSet([], halo=0.0)
        with pytest.raises(ValueError):
            ShapeSet([], skin=-1.0)
        with pytest.raises(TypeError):
            ShapeSet([1.0])

    def test_bound_sdf_atoms(self, grid):
        # Star's SDF underestimates the distance far away. ShapeSet may skip
        # such an atom, so it can exceed the Union chain, but it stays a
        # lower bound on the true distance with the same sign.
        from shapely import distance, points
        from metashapes.adapters.shapely import shape_to_shapely
        atoms = _atoms(stars=True)
        ss, u = ShapeSet(atoms), _union(atoms)
        xs, ys, X, Y = grid
        got = ss.sdf_grid(xs, ys)
        ref = u.sdf(X, Y)
        assert torch.equal(got <= 0, ref <= 0)
        assert torch.all(got >= ref - 1e-4)
        outside = ref > 0.5
        true = torch.as_tensor(distance(shape_to_shapely(u), points(X[outside].numpy(), Y[outside].numpy())),
                               dtype=got.dtype)
        assert torch.all(got[outside] <= true + 1e-2)
        assert torch.allclose(ss.sdf(X, Y), got, atol=1e-4)

    def test_non_tight_bound_atom(self):
        # Difference: SDF is a bound, still exact sign
        d = Difference(Rectangle(center=(0.0, 0.0), size=(4.0, 4.0)),
                       Ellipse(center=(0.0, 0.0), axes=(2.0, 2.0)))
        ss = ShapeSet([d, Rectangle(center=(10.0, 0.0), size=(1.0, 1.0))])
        xs = torch.linspace(-5, 12, 69)
        ys = torch.linspace(-4, 4, 33)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        ref = torch.minimum(d.sdf(X, Y), ss[1].sdf(X, Y))
        assert torch.equal(ss.sdf_grid(xs, ys, halo=0.5) <= 0, ref <= 0)

    def test_float64(self, grid):
        atoms = _atoms(n=10)
        ss = ShapeSet(atoms).to(dtype=torch.float64)
        u = _union(atoms)
        xs, ys, X, Y = grid
        xs, ys, X, Y = xs.double(), ys.double(), X.double(), Y.double()
        out = ss.sdf_grid(xs, ys)
        assert out.dtype == torch.float64
        assert torch.allclose(out, u.sdf(X, Y), atol=1e-8)


# ---------------------------------------------------------------------------
# Geometry, container, serialization
# ---------------------------------------------------------------------------

class TestShapeSetGeometry:
    def test_container_protocol(self, scene):
        ss, _ = scene
        assert len(ss) == 30
        assert ss[0] is list(ss)[0]
        assert isinstance(ss, Shape)

    def test_bounds_and_feature_size(self, scene):
        ss, u = scene
        (a, b), (c, d) = u.bounds()
        assert sum(ss.bounds(), ()) == pytest.approx((a, b, c, d))
        a = Rectangle(center=(0.0, 0.0), size=(2.0, 3.0))
        b = Ellipse(center=(5.0, 0.0), axes=(1.0, 4.0))
        assert ShapeSet([a, b]).min_feature_size == pytest.approx(1.0)
        assert ShapeSet([a, Union(a, b)]).min_feature_size is None

    def test_round_trip(self, scene, grid):
        ss, _ = scene
        data = ss.to_parametric()
        assert data["type"] == "ShapeSet" and len(data["shapes"]) == 30
        restored = Shape.from_parametric(data)
        assert isinstance(restored, ShapeSet)
        xs, ys, _, _ = grid
        assert torch.allclose(restored.sdf_grid(xs, ys), ss.sdf_grid(xs, ys), atol=1e-6)

    def test_round_trip_keeps_options(self):
        ss = ShapeSet([Rectangle(center=(0.0, 0.0), size=(1.0, 1.0))],
                      halo=4.0, cell_size=2.0, skin=0.5)
        r = ShapeSet.from_parametric(ss.to_parametric())
        assert (r.halo, r.cell_size, r.skin) == (4.0, 2.0, 0.5)

    def test_shapely(self, scene):
        from metashapes.adapters.shapely import shape_to_shapely
        ss, u = scene
        assert shape_to_shapely(ss).area == pytest.approx(shape_to_shapely(u).area, rel=1e-6)

    def test_as_unit_cell_scene(self):
        from metashapes import Lattice, UnitCell
        atoms = [Rectangle(center=(0.25, 0.25), size=(0.2, 0.2)),
                 Rectangle(center=(0.75, 0.75), size=(0.2, 0.2))]
        lat = Lattice.rectangular(1.0, 1.0)
        a = UnitCell(lat, ShapeSet(atoms)).rasterize(32, 32)
        b = UnitCell(lat, _union(atoms)).rasterize(32, 32)
        assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradients and index staleness
# ---------------------------------------------------------------------------

def _gradcheck_atoms(p):
    def t(*v):
        return torch.stack([torch.as_tensor(x, dtype=torch.float64) for x in v])
    f64 = lambda v: torch.tensor(v, dtype=torch.float64)
    return [
        Translate(Ellipse(center=t(0.0, 0.0), axes=t(p[2], 1.5), angle=f64(0.0)), p[0], p[1]),
        Rectangle(center=t(p[3], 0.5), size=t(1.0, 1.2), angle=f64(20.0),
                  corner_radius=f64(0.2)),
    ]


class TestGradients:
    def test_gradcheck_stamping(self):
        xs = torch.linspace(-2.0, 5.0, 15, dtype=torch.float64)
        ys = torch.linspace(-2.0, 3.0, 11, dtype=torch.float64)

        def f(p):
            ss = ShapeSet(_gradcheck_atoms(p))
            return ss.sdf_grid(xs, ys, halo=50.0)

        p = torch.tensor([0.13, 0.27, 2.1, 3.03], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(f, (p,), eps=1e-6, atol=1e-5)

    def test_gradcheck_points_exact(self):
        g = torch.Generator().manual_seed(0)
        px = torch.rand(40, generator=g, dtype=torch.float64) * 30 - 10
        py = torch.rand(40, generator=g, dtype=torch.float64) * 30 - 10

        def f(p):
            return ShapeSet(_gradcheck_atoms(p)).sdf(px, py)

        p = torch.tensor([0.13, 0.27, 2.1, 3.03], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(f, (p,), eps=1e-6, atol=1e-5)

    def test_grad_matches_union(self):
        atoms = _atoms(n=12)
        make_learnable(nn.ModuleList(atoms))
        xs = torch.linspace(-10.0, 110.0, 60)
        ys = torch.linspace(-10.0, 110.0, 60)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        ss = ShapeSet(atoms)
        g1 = torch.autograd.grad(torch.sigmoid(-ss.sdf_grid(xs, ys, halo=20.0)).sum(),
                                 list(ss.parameters()))
        g2 = torch.autograd.grad(torch.sigmoid(-_union(atoms).sdf(X, Y)).sum(),
                                 list(ss.parameters()))
        for a, b in zip(g1, g2):
            assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)

    def test_index_rebuilds_after_drift(self):
        r = Rectangle(center=(0.0, 0.0), size=(1.0, 1.0))
        t = Translate(r, nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0)))
        ss = ShapeSet([t, Rectangle(center=(50.0, 0.0), size=(1.0, 1.0))], skin=2.0)
        x = torch.tensor([20.0])
        y = torch.tensor([0.0])
        assert ss.sdf(x, y, halo=1.0).item() == 1.0
        with torch.no_grad():
            t.dx.fill_(0.5)            # within skin/2: no rebuild needed
        assert not ss.rebuild_index()
        with torch.no_grad():
            t.dx.fill_(20.0)           # far beyond the skin
        assert ss.sdf(x, y, halo=1.0).item() < 0     # index followed the atom
        assert not ss.rebuild_index()                  # already rebuilt by the query

    def test_buffers_need_forced_rebuild(self):
        t = Translate(Rectangle(center=(0.0, 0.0), size=(1.0, 1.0)), 0.0, 0.0)
        ss = ShapeSet([t])
        x, y = torch.tensor([20.0]), torch.tensor([0.0])
        assert ss.sdf(x, y, halo=1.0).item() == 1.0
        t.dx.fill_(20.0)
        assert ss.rebuild_index(force=True)
        assert ss.sdf(x, y, halo=1.0).item() < 0


# ---------------------------------------------------------------------------
# make_learnable
# ---------------------------------------------------------------------------

class TestMakeLearnable:
    def test_names(self):
        ss = ShapeSet(_atoms(n=6))
        names = make_learnable(ss, names=("dx", "dy"))
        assert names and all(n.endswith((".dx", ".dy")) for n in names)
        # Translate atoms are 1 and 2 mod 3
        assert {n.split(".")[1] for n in names} == {"1", "2", "4", "5"}
        assert all(p.requires_grad for p in ss.parameters())

    def test_select_indices_and_predicate(self):
        ss = ShapeSet(_atoms(n=6))
        names = make_learnable(ss, select=[0, 3])
        assert {n.split(".")[1] for n in names} == {"0", "3"}
        ss2 = ShapeSet(_atoms(n=6))
        names2 = make_learnable(ss2, select=lambda i, a: isinstance(a, Rectangle))
        assert {n.split(".")[1] for n in names2} == {"0", "3"}
        with pytest.raises(IndexError):
            make_learnable(ShapeSet(_atoms(n=2)), select=[5])

    def test_values_preserved_and_gradients_flow(self):
        r = Rectangle(center=(1.0, 2.0), size=(3.0, 4.0))
        before = r.sdf(torch.tensor([0.3]), torch.tensor([0.7]))
        make_learnable(r, names=("center",))
        assert isinstance(r.center, nn.Parameter)
        assert not isinstance(r.size, nn.Parameter)
        d = r.sdf(torch.tensor([0.3]), torch.tensor([0.7]))
        assert torch.equal(d.detach(), before)
        d.sum().backward()
        assert r.center.grad is not None

    def test_no_recurse(self):
        t = Translate(Rectangle(center=(0.0, 0.0), size=(1.0, 1.0)), 1.0, 2.0)
        names = make_learnable(t, recurse=False)
        assert sorted(names) == ["dx", "dy"]
        assert not isinstance(t.shape.center, nn.Parameter)

    def test_works_on_unit_cell(self):
        from metashapes import Lattice, UnitCell
        cell = UnitCell(Lattice.rectangular(1.0, 1.0),
                        Rectangle(center=(0.5, 0.5), size=(0.3, 0.3)))
        make_learnable(cell.scene)
        assert len(list(cell.parameters())) == 4
        assert not any(p is cell.lattice.a1 for p in cell.parameters())
