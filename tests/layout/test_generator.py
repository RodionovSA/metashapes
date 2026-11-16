# tests/layout/test_generator.py

import math
import itertools
import random

import pytest
import torch

from metashapes import Lattice
from metashapes.adapters.shapely import shape_to_shapely
from metashapes.generators import GeneratorConfig
from metashapes.layout import (
    ExplicitSites, GeometryHash, LatticeSites, Layout, LayoutAnalyzer, LayoutBatchResult,
    LayoutConfig, LayoutGenerator, LayoutValidator, ParametricSource, RandomSites, Regions,
    SamplerSource,
)
from metashapes.shape import Ellipse, Rectangle, ShapeSet, Translate


@pytest.fixture
def domain():
    return Rectangle(center=(0.0, 0.0), size=(1000.0, 800.0))


def _sampler(**kw):
    cfg = GeneratorConfig(allowed_shapes=("Rectangle", "Ellipse", "Cross"),
                          min_feature_size=20.0, **kw)
    return SamplerSource(cfg, atom_box=(100.0, 100.0))


def _brute(layout, margin, min_gap):
    """Containment and pairwise gaps by brute-force Shapely."""
    dom = shape_to_shapely(layout.domain)
    inner = dom.buffer(-margin) if margin else dom
    geoms = [shape_to_shapely(a) for a in layout.shapes]
    assert all(inner.contains(g) for g in geoms)
    if min_gap is not None:
        for a, b in itertools.combinations(geoms, 2):
            assert a.distance(b) >= min_gap - 1e-9
    return geoms


# ---------------------------------------------------------------------------
# GeometryHash
# ---------------------------------------------------------------------------

class TestGeometryHash:
    def test_conflicts_match_brute_force(self):
        from shapely.geometry import Point
        rng = random.Random(1)
        h = GeometryHash()
        geoms = []
        for _ in range(200):
            g = Point(rng.uniform(0, 500), rng.uniform(0, 500)).buffer(rng.uniform(3, 12))
            geoms.append(g)
            h.insert(g)
        for _ in range(100):
            q = Point(rng.uniform(0, 500), rng.uniform(0, 500)).buffer(rng.uniform(3, 12))
            for gap in (0.0, 5.0, 30.0):
                if gap > 0:
                    want = any(q.distance(g) < gap for g in geoms)
                else:
                    want = any(q.intersects(g) and not q.touches(g) for g in geoms)
                assert h.conflicts(q, gap) == want

    def test_zero_gap_allows_touching(self):
        from shapely.geometry import box
        h = GeometryHash()
        h.insert(box(0, 0, 1, 1))
        assert not h.conflicts(box(1, 0, 2, 1), 0.0)
        assert h.conflicts(box(0.9, 0, 2, 1), 0.0)
        assert h.conflicts(box(1, 0, 2, 1), 0.1)


# ---------------------------------------------------------------------------
# Sites and sources
# ---------------------------------------------------------------------------

class TestSites:
    def test_lattice_sites_inside_domain(self, domain):
        pts = LatticeSites(Lattice.rectangular(100.0, 100.0), origin=(50.0, 50.0)).points(domain)
        assert len(pts) == 10 * 8
        assert all(abs(x) <= 500 and abs(y) <= 400 for x, y in pts)

    def test_hex_sites_in_circle(self):
        dom = Ellipse(center=(0.0, 0.0), axes=(1000.0, 1000.0))
        pts = LatticeSites(Lattice.hexagonal(50.0)).points(dom)
        assert all(math.hypot(x, y) <= 500 + 1e-6 for x, y in pts)
        # hex density: 2 / (sqrt(3) a^2) per unit area
        expected = math.pi * 500 ** 2 * 2 / (math.sqrt(3) * 50 ** 2)
        assert len(pts) == pytest.approx(expected, rel=0.03)

    def test_jitter_and_random_reproducible(self, domain):
        s = LatticeSites(Lattice.rectangular(100.0, 100.0), jitter=5.0)
        a = list(s.candidates(domain, random.Random(3)))
        b = list(s.candidates(domain, random.Random(3)))
        assert a == b and a != s.points(domain)
        r = RandomSites()
        c = list(itertools.islice(r.candidates(domain, random.Random(0)), 50))
        d = list(itertools.islice(r.candidates(domain, random.Random(0)), 50))
        assert c == d
        assert all(abs(x) <= 500 and abs(y) <= 400 for x, y in c)

    def test_random_sites_respect_domain_shape(self):
        dom = Ellipse(center=(0.0, 0.0), axes=(100.0, 100.0))
        pts = list(itertools.islice(RandomSites().candidates(dom, random.Random(0)), 300))
        assert all(math.hypot(x, y) <= 50 + 1e-6 for x, y in pts)

    def test_explicit_sites(self, domain):
        assert list(ExplicitSites([(1, 2), (3, 4)]).candidates(domain, None)) == [(1.0, 2.0), (3.0, 4.0)]


class TestSources:
    def test_sampler_source_is_centred(self):
        src = _sampler()
        rng = random.Random(0)
        for _ in range(20):
            (x0, y0), (x1, y1) = src.atom(0.0, 0.0, rng).bounds()
            assert (x0 + x1) / 2 == pytest.approx(0.0, abs=1e-4)
            assert (y0 + y1) / 2 == pytest.approx(0.0, abs=1e-4)
            assert x1 - x0 <= 100.0 + 1e-4 and y1 - y0 <= 100.0 + 1e-4

    def test_sampler_source_rejects_bar(self):
        with pytest.raises(ValueError, match="infinite"):
            SamplerSource(GeneratorConfig(allowed_shapes=("Bar",)), atom_box=(10, 10))
        assert "Bar" not in SamplerSource(GeneratorConfig(), atom_box=(10, 10)).names

    def test_parametric_source(self):
        src = ParametricSource(Ellipse, {"axes": lambda x, y: (10.0 + x, 5.0), "angle": 30.0})
        e = src.atom(2.0, 0.0, None)
        assert e.axes.tolist() == pytest.approx([12.0, 5.0])
        assert e.center.tolist() == [0.0, 0.0]
        assert src.deterministic
        assert isinstance(ParametricSource("Rectangle", {"size": (1, 2)}).atom(0, 0, None), Rectangle)

    def test_regions(self):
        left = Rectangle(center=(-250.0, 0.0), size=(500.0, 800.0))
        rect = ParametricSource(Rectangle, {"size": (10.0, 10.0)})
        ell = ParametricSource(Ellipse, {"axes": (10.0, 10.0)})
        src = Regions([(left, rect)], default=ell)
        assert isinstance(src.atom(-100.0, 0.0, None), Rectangle)
        assert isinstance(src.atom(100.0, 0.0, None), Ellipse)
        assert Regions([(left, rect)]).atom(100.0, 0.0, None) is None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestGenerator:
    @pytest.mark.parametrize("min_gap,margin", [(15.0, 0.0), (0.0, 25.0), (40.0, 10.0)])
    def test_random_constraints_hold(self, domain, min_gap, margin):
        cfg = LayoutConfig(seed=1, min_gap=min_gap, edge_margin=margin, target_num_shapes=40,
                           max_consecutive_failures=300)
        lay = LayoutGenerator(cfg, RandomSites(), _sampler()).generate_one(domain)
        assert len(lay) == 40
        _brute(lay, margin, min_gap)
        assert LayoutAnalyzer(min_gap=min_gap, edge_margin=margin).check(lay) == []
        assert all(a.min_feature_size >= 20.0 for a in lay.shapes)
        assert lay.violations(margin=margin) == []

    def test_min_gap_none_allows_overlap(self, domain):
        cfg = LayoutConfig(seed=0, min_gap=None, target_num_shapes=150)
        lay = LayoutGenerator(cfg, RandomSites(), _sampler()).generate_one(domain)
        assert len(lay) == 150
        assert LayoutAnalyzer().metrics(lay).min_gap == 0.0
        _brute(lay, 0.0, None)

    def test_same_seed_same_output(self, domain):
        def run(seed):
            cfg = LayoutConfig(seed=seed, target_count=2, min_gap=10.0, target_num_shapes=25)
            b = LayoutGenerator(cfg, RandomSites(), _sampler()).generate(domain)
            return [lay.to_parametric() for lay in b.layouts]
        assert run(7) == run(7)
        assert run(7) != run(8)

    def test_batch_report(self, domain):
        cfg = LayoutConfig(seed=0, target_count=3, min_gap=10.0, target_num_shapes=20)
        b = LayoutGenerator(cfg, RandomSites(), _sampler()).generate(domain)
        r = b.report
        assert (r.requested, r.generated, r.failed, r.total_attempts) == (3, 3, 0, 3)
        assert r.parameter_ranges["num_shapes"] == (20, 20)
        assert len(r.metadata["layouts"]) == 3
        assert r.metadata["sites"] == "RandomSites"
        lo, hi = r.parameter_ranges["fill_fraction"]
        assert 0 < lo <= hi < 1

    def test_failed_layouts_are_counted(self, domain):
        class Never(LayoutValidator):
            def validate(self, layout):
                return "nope"
        cfg = LayoutConfig(seed=0, target_count=2, max_tries_per_layout=2, target_num_shapes=3)
        b = LayoutGenerator(cfg, RandomSites(), _sampler(), validator=Never()).generate(domain)
        assert (b.report.generated, b.report.failed, b.report.total_attempts) == (0, 2, 4)
        assert b.report.failure_reasons == {"nope": 4}
        with pytest.raises(RuntimeError, match="nope"):
            LayoutGenerator(cfg, RandomSites(), _sampler(), validator=Never()).generate_one(domain)

    def test_fill_fraction_stop(self, domain):
        cfg = LayoutConfig(seed=0, min_gap=5.0, target_fill_fraction=0.10)
        b = LayoutGenerator(cfg, RandomSites(), _sampler()).generate(domain)
        lay = b.layouts[0]
        st = b.report.metadata["layouts"][0]
        assert 0.10 <= st["fill_fraction"] < 0.13
        assert not st["saturated"]
        assert lay.to_shapely().area / 800_000 == pytest.approx(st["fill_fraction"], rel=1e-6)

    def test_saturation_stop(self):
        small = Rectangle(center=(0.0, 0.0), size=(200.0, 200.0))
        cfg = LayoutConfig(seed=0, min_gap=10.0, max_consecutive_failures=200)
        src = ParametricSource(Ellipse, {"axes": (40.0, 40.0)})
        b = LayoutGenerator(cfg, RandomSites(), src).generate(small)
        st = b.report.metadata["layouts"][0]
        assert st["saturated"] and b.report.metadata["saturated"] == 1
        # a 200x200 box holds at most 16 discs of 40 with 10 gaps (4x4 packing)
        assert 3 <= st["n_shapes"] <= 16
        _brute(b.layouts[0], 0.0, 10.0)

    def test_field_driven_lattice(self):
        dom = Ellipse(center=(0.0, 0.0), axes=(2000.0, 2000.0))
        radius = lambda x, y: 20.0 + 30.0 * (math.hypot(x, y) / 1000.0)
        src = ParametricSource(Ellipse, {"axes": lambda x, y: (2 * radius(x, y),) * 2})
        sites = LatticeSites(Lattice.hexagonal(150.0))
        lay = LayoutGenerator(LayoutConfig(edge_margin=20.0), sites, src).generate_one(dom)
        pts = sites.points(dom)
        assert 0 < len(lay) < len(pts)          # edge sites that don't fit stay empty
        for atom in lay.shapes:
            r = math.hypot(atom.dx.item(), atom.dy.item())
            assert atom.shape.axes[0].item() == pytest.approx(2 * radius(atom.dx.item(), atom.dy.item()), rel=1e-5)
            assert r <= 1000.0
        _brute(lay, 20.0, None)

    def test_lattice_sites_retry_random_source(self, domain):
        sites = LatticeSites(Lattice.rectangular(100.0, 100.0), origin=(50.0, 50.0))
        cfg = LayoutConfig(seed=0, min_gap=5.0, max_tries_per_site=5)
        b = LayoutGenerator(cfg, sites, _sampler()).generate(domain)
        st = b.report.metadata["layouts"][0]
        assert st["sites_tried"] == len(sites.points(domain))
        assert st["n_shapes"] <= st["sites_tried"]
        _brute(b.layouts[0], 0.0, 5.0)

    def test_regions_layout(self, domain):
        left = Rectangle(center=(-250.0, 0.0), size=(500.0, 800.0))
        src = Regions([(left, ParametricSource(Rectangle, {"size": (30.0, 30.0)}))],
                      default=ParametricSource(Ellipse, {"axes": (30.0, 30.0)}))
        sites = LatticeSites(Lattice.rectangular(100.0, 100.0), origin=(50.0, 50.0))
        lay = LayoutGenerator(LayoutConfig(), sites, src).generate_one(domain)
        assert len(lay) == 80
        for atom in lay.shapes:
            want = Rectangle if atom.dx.item() <= 0 else Ellipse
            assert isinstance(atom.shape, want)

    def test_outputs_are_plain(self, domain):
        cfg = LayoutConfig(seed=0, target_num_shapes=5)
        lay = LayoutGenerator(cfg, RandomSites(), _sampler()).generate_one(domain)
        assert len(list(lay.parameters())) == 0
        assert all(isinstance(a, Translate) for a in lay.shapes)
        assert all(not isinstance(a.shape, Translate) for a in lay.shapes)  # folded

    def test_merge(self, domain):
        g = LayoutGenerator(LayoutConfig(seed=0, target_num_shapes=5), RandomSites(), _sampler())
        a, b = g.generate(domain), g.generate(domain)
        m = LayoutBatchResult.merge(a, b)
        assert len(m.layouts) == 2
        assert m.report.requested == 2 and m.report.generated == 2
        assert LayoutBatchResult.merge(a) is a
        with pytest.raises(ValueError):
            LayoutBatchResult.merge()

    def test_config_validation(self):
        for bad in (dict(target_count=0), dict(target_fill_fraction=0.0),
                    dict(target_fill_fraction=1.5), dict(edge_margin=-1.0),
                    dict(min_gap=-1.0), dict(max_tries_per_site=0)):
            with pytest.raises(ValueError):
                LayoutConfig(**bad)


# ---------------------------------------------------------------------------
# Serialization of generated layouts
# ---------------------------------------------------------------------------

class TestSerialization:
    @pytest.fixture
    def batch(self, domain):
        cfg = LayoutConfig(seed=0, target_count=2, min_gap=10.0, target_num_shapes=30)
        return LayoutGenerator(cfg, RandomSites(), _sampler()).generate(domain)

    def test_yaml_batch(self, batch, tmp_path):
        from metashapes.adapters.yaml import save_layout_batch, load_layout_batch, load_layouts
        p = tmp_path / "batch.yaml"
        save_layout_batch(p, batch)
        back = load_layout_batch(p)
        assert back.report.generated == 2
        assert back.report.parameter_ranges == batch.report.parameter_ranges
        assert len(load_layouts(p)) == 2
        for a, b in zip(batch.layouts, back.layouts):
            assert torch.allclose(a.rasterize(64, 64), b.rasterize(64, 64), atol=1e-5)

    def test_npz(self, batch, tmp_path):
        from metashapes.adapters.npz import save_layouts_npz, load_layouts_npz
        p = tmp_path / "layouts.npz"
        save_layouts_npz(p, batch.layouts)
        back = load_layouts_npz(p)
        assert len(back) == 2
        for a, b in zip(batch.layouts, back):
            assert [type(s.shape) for s in a.shapes] == [type(s.shape) for s in b.shapes]
            assert torch.allclose(a.rasterize(64, 64), b.rasterize(64, 64), atol=1e-5)

    def test_npz_mixed_atoms(self, tmp_path):
        from metashapes.adapters.npz import save_layouts_npz, load_layouts_npz
        from metashapes.shape import Star, Union
        atoms = [
            Rectangle(center=(0.0, 0.0), size=(2.0, 3.0), angle=10.0),
            Translate(Star(center=(0.0, 0.0), n=5, outer_radius=2.0, inner_radius=1.0), 5.0, 0.0),
            Translate(Star(center=(0.0, 0.0), n=6, outer_radius=2.0, inner_radius=1.0), 5.0, 5.0),
            Union(Rectangle(center=(9.0, 0.0), size=(1.0, 1.0)), Ellipse(center=(9.5, 0.0), axes=(1.0, 2.0)),
                  smooth=True, k=0.3),
        ]
        lay = Layout(Rectangle(center=(5.0, 2.0), size=(20.0, 12.0)), ShapeSet(atoms, halo=7.0, skin=0.5))
        p = tmp_path / "mixed.npz"
        save_layouts_npz(p, lay)
        (back,) = load_layouts_npz(p)
        assert back.scene.halo == 7.0 and back.scene.skin == 0.5
        assert back.shapes[2].shape.n == 6 and isinstance(back.shapes[2].shape.n, int)
        assert back.shapes[3].smooth is True
        assert torch.allclose(back.rasterize(80, 48), lay.rasterize(80, 48), atol=1e-5)
