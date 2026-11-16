# tests/layout/test_analysis.py

import pytest
import torch
import torch.nn as nn

from metashapes.layout import Layout, LayoutAnalyzer, LayoutMetrics, make_learnable
from metashapes.shape import Ellipse, Rectangle, ShapeSet, Translate


@pytest.fixture
def domain():
    return Rectangle(center=(50.0, 50.0), size=(100.0, 100.0))


def _sq(x, y, s=10.0):
    return Rectangle(center=(x, y), size=(s, s))


class TestMetrics:
    def test_basic(self, domain):
        lay = Layout(domain, ShapeSet([_sq(20, 20), _sq(40, 20), _sq(80, 80)]))
        m = LayoutAnalyzer().metrics(lay)
        assert isinstance(m, LayoutMetrics)
        assert m.n_shapes == 3
        assert m.fill_fraction == pytest.approx(300 / 10_000)
        assert m.min_gap == pytest.approx(10.0)
        assert m.min_feature_size == pytest.approx(10.0)
        assert m.min_edge_clearance == pytest.approx(15.0)
        assert (m.n_outside, m.n_clipped_features) == (0, 0)

    def test_overlap_and_identical_atoms_have_zero_gap(self, domain):
        assert LayoutAnalyzer().metrics(
            Layout(domain, ShapeSet([_sq(20, 20), _sq(25, 20)]))).min_gap == 0.0
        assert LayoutAnalyzer().metrics(
            Layout(domain, ShapeSet([_sq(20, 20), _sq(20, 20)]))).min_gap == 0.0

    def test_single_and_empty(self, domain):
        m = LayoutAnalyzer().metrics(Layout(domain, ShapeSet([_sq(20, 20)])))
        assert m.min_gap is None and m.n_shapes == 1
        m0 = LayoutAnalyzer().metrics(Layout(domain, ShapeSet([])))
        assert (m0.n_shapes, m0.fill_fraction, m0.min_edge_clearance) == (0, 0.0, None)

    def test_outside_and_clipped(self, domain):
        atoms = [
            _sq(50, 50),
            _sq(101, 50),           # mostly outside: clipped piece 4 wide -> sliver
            _sq(97, 20),            # clipped piece 8 wide
            _sq(150, 50),           # entirely outside: not "clipped", but outside
        ]
        lay = Layout(domain, ShapeSet(atoms))
        m = LayoutAnalyzer(min_feature_size=6.0).metrics(lay)
        assert m.n_outside == 3
        assert m.n_clipped_features == 1
        assert m.min_edge_clearance == 0.0
        # stricter threshold: both clipped pieces are slivers
        assert LayoutAnalyzer(min_feature_size=9.0).metrics(lay).n_clipped_features == 2
        # without a threshold the atom's own feature size (10) is used
        assert LayoutAnalyzer().metrics(lay).n_clipped_features == 2

    def test_edge_margin(self, domain):
        lay = Layout(domain, ShapeSet([_sq(10, 50), _sq(50, 50)]))
        assert LayoutAnalyzer(edge_margin=4.0).metrics(lay).n_outside == 0
        assert LayoutAnalyzer(edge_margin=6.0).metrics(lay).n_outside == 1
        assert lay.violations(margin=6.0) == [0]


class TestCheck:
    def test_clean(self, domain):
        lay = Layout(domain, ShapeSet([_sq(20, 20), _sq(40, 20)]))
        a = LayoutAnalyzer(min_gap=5.0, min_feature_size=5.0, edge_margin=5.0)
        assert a.check(lay) == [] and a.validate(lay) is None
        assert lay.check(margin=5.0, min_gap=5.0, min_feature_size=5.0) == []

    def test_reasons(self, domain):
        lay = Layout(domain, ShapeSet([_sq(20, 20), _sq(28, 20, 4.0), _sq(102, 50)]))
        msgs = lay.check(min_gap=5.0, min_feature_size=6.0)
        text = " | ".join(msgs)
        assert "outside_domain" in text
        assert "clipped_feature" in text
        assert "min_feature_size" in text
        assert "min_gap" in text
        assert LayoutAnalyzer(min_gap=5.0).validate(lay).startswith("outside_domain")

    def test_zero_gap_allows_touching(self, domain):
        touching = Layout(domain, ShapeSet([_sq(20, 20), _sq(30, 20)]))
        overlapping = Layout(domain, ShapeSet([_sq(20, 20), _sq(29, 20)]))
        assert touching.check(min_gap=0.0) == []
        assert any("overlap" in m for m in overlapping.check(min_gap=0.0))


class TestContainmentPenalty:
    def test_zero_when_contained(self, domain):
        lay = Layout(domain, ShapeSet([_sq(50, 50), _sq(20, 30)]))
        assert lay.containment_penalty(margin=2.0, pixel_size=0.5).item() == pytest.approx(0.0, abs=1e-3)

    def test_measures_outside_area(self, domain):
        # 10x10 square centred on x=100: 5 x 10 outside, linear ramp d in [0, 5]
        lay = Layout(domain, ShapeSet([_sq(100, 50)]))
        p = lay.containment_penalty(pixel_size=0.1, softness=0.05).item()
        assert p == pytest.approx(10 * 5 * 5 / 2, rel=0.05)
        # margin moves the boundary inward
        pm = lay.containment_penalty(margin=2.0, pixel_size=0.1, softness=0.05).item()
        assert pm == pytest.approx(10 * 7 * 7 / 2, rel=0.05)

    def test_requires_resolution(self, domain):
        lay = Layout(domain, ShapeSet([_sq(50, 50)]))
        with pytest.raises(ValueError):
            lay.containment_penalty()

    def test_gradient_pushes_atom_inside(self, domain):
        atom = Translate(Ellipse(center=(0.0, 0.0), axes=(10.0, 10.0)), 98.0, 50.0)
        lay = Layout(domain, ShapeSet([atom], skin=1.0))
        make_learnable(lay.scene, names=("dx", "dy"))
        opt = torch.optim.SGD(lay.parameters(), lr=0.05)
        start = lay.containment_penalty(margin=1.0, pixel_size=0.5).item()
        for _ in range(60):
            opt.zero_grad()
            loss = lay.containment_penalty(margin=1.0, pixel_size=0.5)
            loss.backward()
            opt.step()
        assert atom.dx.item() < 98.0
        assert lay.containment_penalty(margin=1.0, pixel_size=0.5).item() < 0.05 * start
        assert lay.violations(margin=1.0) == []

    def test_only_border_atoms_are_evaluated(self, domain):
        calls = []

        class Spy(Rectangle):
            def sdf(self, x, y):
                calls.append(1)
                return super().sdf(x, y)

        lay = Layout(domain, ShapeSet([Spy(center=(50.0, 50.0), size=(5.0, 5.0)),
                                       Spy(center=(98.0, 50.0), size=(5.0, 5.0))]))
        lay.containment_penalty(pixel_size=1.0)
        assert len(calls) == 1


class TestDriftedParameters:
    def test_shapely_paths_project_first(self, domain):
        # An optimizer step can leave a size negative until the next sdf();
        # the Shapely-based methods must project before converting.
        r = Rectangle(center=(0.0, 0.0), size=(10.0, 2.0))
        lay = Layout(domain, ShapeSet([Translate(r, 50.0, 50.0), _sq(20, 20)]))
        make_learnable(lay.scene, names=("size",))
        with torch.no_grad():
            r.size[1] = -1.0
        assert lay.violations() == []
        assert r.size[1].item() > 0
        with torch.no_grad():
            r.size[1] = -1.0
        assert LayoutAnalyzer(min_gap=1.0).check(lay) == []
        with torch.no_grad():
            r.size[1] = -1.0
        assert not lay.to_shapely().is_empty


class TestOptimizationLoop:
    def test_layout_optimizes_with_tiles(self, domain):
        """Grow ellipse axes toward a target fill using tiled soft masks."""
        atoms = [Translate(Ellipse(center=(0.0, 0.0), axes=(8.0, 8.0)), x, y)
                 for x in (25.0, 75.0) for y in (25.0, 75.0)]
        lay = Layout(domain, ShapeSet(atoms, skin=2.0))
        make_learnable(lay.scene, names=("axes",))
        opt = torch.optim.Adam(lay.parameters(), lr=0.5)
        target = 0.10
        for _ in range(40):
            opt.zero_grad()
            fill = 0.0
            for t in lay.tiles(50.0, pixel_size=1.0):
                m = lay.mask(bounds=t.bounds, pixel_size=1.0, soft=True)
                fill = fill + m.sum() / 10_000
            ((fill - target) ** 2).backward()
            opt.step()
        got = lay.mask(pixel_size=1.0).mean().item()
        assert got == pytest.approx(target, abs=0.01)
