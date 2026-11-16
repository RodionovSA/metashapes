# tests/layout/test_layout.py

import math

import pytest
import torch
import torch.nn as nn

from metashapes import Layout
from metashapes.shape import (
    Shape, ShapeSet, Intersection, Union, Rectangle, Ellipse, Translate, make_learnable,
)


def _atoms():
    # 5x4 grid of small shapes; two of them poke outside the domain
    atoms = []
    for i in range(5):
        for j in range(4):
            x, y = 10.0 + 20.0 * i, 12.0 + 25.0 * j
            if (i + j) % 2:
                atoms.append(Rectangle(center=(x, y), size=(8.0, 6.0), angle=15.0))
            else:
                atoms.append(Translate(Ellipse(center=(0.0, 0.0), axes=(9.0, 7.0)), x, y))
    atoms.append(Rectangle(center=(99.0, 50.0), size=(10.0, 10.0)))    # crosses x = 100
    atoms.append(Translate(Ellipse(center=(0.0, 0.0), axes=(6.0, 6.0)), 50.0, -1.0))  # crosses y = 0
    return atoms


def _union(atoms):
    u = atoms[0]
    for a in atoms[1:]:
        u = Union(u, a)
    return u


@pytest.fixture
def domain():
    return Rectangle(center=(50.0, 50.0), size=(100.0, 100.0))


@pytest.fixture
def layout(domain):
    return Layout(domain, ShapeSet(_atoms()))


class TestConstruction:
    def test_basic(self, layout):
        assert len(layout) == 22
        assert len(layout.shapes) == 22
        assert layout.extent() == pytest.approx((0.0, 100.0, 0.0, 100.0))

    def test_plain_shape_scene(self, domain):
        lay = Layout(domain, Rectangle(center=(50.0, 50.0), size=(10.0, 10.0)))
        assert len(lay) == 1
        assert lay.mask(50, 50).sum().item() == pytest.approx(25.0)

    def test_rejects_infinite_or_empty_domain(self):
        from metashapes.shape import Bar
        with pytest.raises(ValueError):
            Layout(Bar(offset=0.0, width=1.0), ShapeSet([]))
        empty = Intersection(Rectangle(center=(0.0, 0.0), size=(1.0, 1.0)),
                             Rectangle(center=(10.0, 0.0), size=(1.0, 1.0)))
        with pytest.raises(ValueError):
            Layout(empty, ShapeSet([]))

    def test_strict(self, domain):
        with pytest.raises(ValueError, match="not fully inside"):
            Layout(domain, ShapeSet(_atoms()), strict=True)
        Layout(domain, ShapeSet(_atoms()[:-2]), strict=True)


class TestRendering:
    def test_clip_equals_intersection(self, layout, domain):
        X, Y = layout.grid(120, 110)
        ref = Intersection(_union(_atoms()), domain).sdf(X, Y)
        assert torch.allclose(layout.rasterize(120, 110), ref, atol=1e-4)
        assert torch.allclose(layout.sdf(X, Y), ref, atol=1e-4)

    def test_no_clip(self, domain):
        lay = Layout(domain, ShapeSet(_atoms()), clip=False)
        X, Y = lay.grid(64, 64)
        assert torch.allclose(lay.rasterize(64, 64), _union(_atoms()).sdf(X, Y), atol=1e-4)

    def test_grid_convention(self, layout):
        X, Y = layout.grid(4, 2)
        assert X.shape == (2, 4)
        assert X[0].tolist() == pytest.approx([0.0, 25.0, 50.0, 75.0])
        assert Y[:, 0].tolist() == pytest.approx([0.0, 50.0])

    def test_resolution_arguments(self, layout):
        assert layout.rasterize(pixel_size=2.0).shape == (50, 50)
        assert layout.rasterize(40).shape == (40, 40)
        assert layout.rasterize(ny=20).shape == (20, 20)
        with pytest.raises(ValueError):
            layout.rasterize()
        with pytest.raises(ValueError):
            layout.rasterize(10, 10, pixel_size=1.0)

    def test_bounds_window(self, layout):
        full = layout.rasterize(pixel_size=1.0)
        win = layout.rasterize(pixel_size=1.0, bounds=((20.0, 30.0), (60.0, 50.0)))
        assert win.shape == (20, 40)
        assert torch.allclose(win, full[30:50, 20:60], atol=1e-5)

    def test_hard_mask_matches_intersection(self, layout, domain):
        X, Y = layout.grid(pixel_size=0.5)
        ref = (Intersection(_union(_atoms()), domain).sdf(X, Y) <= 0).float()
        assert torch.equal(layout.mask(pixel_size=0.5), ref)

    def test_soft_mask(self, layout, domain):
        X, Y = layout.grid(pixel_size=1.0)
        ref = torch.sigmoid(-Intersection(_union(_atoms()), domain).sdf(X, Y) / 1.0)
        m = layout.mask(pixel_size=1.0, soft=True)
        assert torch.allclose(m, ref, atol=5e-4)
        m2 = layout.mask(pixel_size=1.0, soft=True, softness=0.5)
        ref2 = torch.sigmoid(-Intersection(_union(_atoms()), domain).sdf(X, Y) / 0.5)
        assert torch.allclose(m2, ref2, atol=5e-4)

    def test_capped_rasterize(self, layout, domain):
        X, Y = layout.grid(80, 80)
        ref = torch.maximum(_union(_atoms()).sdf(X, Y).clamp(max=3.0), domain.sdf(X, Y))
        assert torch.allclose(layout.rasterize(80, 80, halo=3.0), ref, atol=1e-5)


class TestTiles:
    @pytest.mark.parametrize("overlap", [0.0, 3.0])
    def test_tiles_stitch_to_full_raster(self, layout, overlap):
        full = layout.rasterize(pixel_size=1.0)
        out = torch.full_like(full, float("nan"))
        tiles = list(layout.tiles(30.0, pixel_size=1.0, overlap=overlap))
        assert len(tiles) == 16
        for t in tiles:
            r = layout.rasterize(pixel_size=1.0, bounds=t.bounds)
            assert r.shape == (t.ny, t.nx)
            out[t.rows, t.cols] = r[t.crop]
        assert not torch.isnan(out).any()
        assert torch.allclose(out, full, atol=1e-5)

    def test_tile_grid(self, layout):
        t = next(iter(layout.tiles(25.0, pixel_size=5.0)))
        assert t.X.shape == (5, 5)
        assert t.X[0, 1].item() == pytest.approx(5.0)
        assert t.core == t.bounds

    def test_tiled_gradients_accumulate(self, domain):
        atoms = _atoms()
        make_learnable(nn.ModuleList(atoms), names=("dx", "dy", "center"))
        lay = Layout(domain, ShapeSet(atoms))
        lay.mask(pixel_size=1.0, soft=True).sum().backward()
        full = [p.grad.clone() for p in lay.parameters()]
        lay.zero_grad()
        for t in lay.tiles(40.0, pixel_size=1.0):
            m = lay.mask(pixel_size=1.0, bounds=t.bounds, soft=True)
            m.sum().backward()
        for g, p in zip(full, lay.parameters()):
            assert torch.allclose(g, p.grad, atol=1e-3, rtol=1e-4)


class TestShapelyAndSerialization:
    def test_to_shapely(self, layout, domain):
        from metashapes.adapters.shapely import shape_to_shapely
        geom = layout.to_shapely()
        ref = shape_to_shapely(_union(_atoms())).intersection(shape_to_shapely(domain))
        assert geom.area == pytest.approx(ref.area, rel=1e-6)
        assert shape_to_shapely(domain).buffer(1e-9).contains(geom)

    def test_round_trip(self, layout):
        data = layout.to_parametric()
        assert data["type"] == "Layout"
        lay2 = Layout.from_parametric(data)
        assert lay2.clip and len(lay2) == len(layout)
        assert torch.allclose(lay2.rasterize(50, 50), layout.rasterize(50, 50), atol=1e-6)

    def test_yaml(self, layout, tmp_path):
        from metashapes.adapters.yaml import save_layouts, load_layouts
        p = tmp_path / "layouts.yaml"
        save_layouts(p, layout)
        (back,) = load_layouts(p)
        assert torch.allclose(back.rasterize(40, 40), layout.rasterize(40, 40), atol=1e-6)
        assert math.isinf(back.scene.halo)
        save_layouts(p, [layout, layout])
        assert len(load_layouts(p)) == 2

    def test_device_dtype_follow_module(self, layout):
        lay = layout.to(dtype=torch.float64)
        assert lay.rasterize(10, 10).dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
    def test_cuda(self, layout):
        cpu = layout.rasterize(pixel_size=1.0)
        gpu = layout.to("cuda").rasterize(pixel_size=1.0)
        assert gpu.is_cuda
        assert torch.allclose(gpu.cpu(), cpu, atol=1e-4)
