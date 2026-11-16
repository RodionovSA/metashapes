# metashapes/gds/__init__.py
# GDS export (needs the optional dependency gdstk). See docs/gds_design.md.

from .convert import Units, polygons_from_shapely
from .cells import layout_to_cell, shape_to_gdstk, unit_cell_geometry, unit_cell_to_cell
from .assembly import GDSAssembly, Placement, save_gds

__all__ = [
    "Units",
    "polygons_from_shapely",
    "shape_to_gdstk",
    "unit_cell_geometry",
    "unit_cell_to_cell",
    "layout_to_cell",
    "GDSAssembly",
    "Placement",
    "save_gds",
]
