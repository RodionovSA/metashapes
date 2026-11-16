# metashapes/layout
# Non-periodic layouts: a finite domain filled with many shapes.

from metashapes.shape.learnable import make_learnable
from metashapes.shape.shapeset import ShapeSet

from .layout import Layout, Tile
from .config import LayoutConfig
from .sites import Sites, LatticeSites, RandomSites, ExplicitSites
from .sources import AtomSource, SamplerSource, ParametricSource, Regions
from .spatial import GeometryHash
from .generator import LayoutGenerator, LayoutBatchResult, LayoutValidator
from .analysis import LayoutAnalyzer, LayoutMetrics

__all__ = [
    "Layout",
    "Tile",
    "ShapeSet",
    "make_learnable",
    "LayoutConfig",
    "Sites",
    "LatticeSites",
    "RandomSites",
    "ExplicitSites",
    "AtomSource",
    "SamplerSource",
    "ParametricSource",
    "Regions",
    "GeometryHash",
    "LayoutGenerator",
    "LayoutBatchResult",
    "LayoutValidator",
    "LayoutAnalyzer",
    "LayoutMetrics",
]
