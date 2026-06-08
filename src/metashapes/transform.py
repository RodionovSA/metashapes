#metashapes/transform.py

# This module provides functions to transform shapes between vector and raster formats.

import numpy as np
import gdstk
import pya

from skimage.draw import polygon as sk_polygon
from skimage import measure
from skimage.filters import gaussian as gaussian_filter

import shapely
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from shapely.geometry.base import BaseGeometry

from typing import TYPE_CHECKING, List
from .lattice import Canvas  

def shapely_to_numpy(geom: 'BaseGeometry', canvas: 'Canvas') -> np.ndarray:
    if geom.is_empty:
        return np.zeros((canvas.H, canvas.W), dtype=bool)

    xs, ys = canvas.grid()
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    mask = shapely.contains_xy(geom, xs, ys) | shapely.intersects_xy(geom.boundary, xs, ys)
    return np.asarray(mask, dtype=bool)

def numpy_to_shapely(img: np.ndarray, 
                     canvas: 'Canvas',
                     *,
                     simp_coeff: float = 0.5,
                     sfd: bool = True,
                     gaussian: bool = False,
                     gauss_sigma: float = 0.6,
                     verbose: bool = False) -> 'BaseGeometry':
    """
    Create a shapely geometry from a binary numpy array.
    Parameters:
        img: 2D binary numpy array where True/1 indicates the shape.
        canvas: Canvas object defining the spatial mapping.
        simp_coeff: Simplification coefficient (scales with pixel size). Reduce number of vertices.
        sfd: If True, apply signed distance function before contouring. **Requires scipy.ndimage**.
        gaussian: If True, apply Gaussian smoothing before contouring.
        gauss_sigma: Standard deviation for Gaussian kernel if gaussian is True.
        verbose: If True, print image reconstruction error.
    Returns:
        A new shapely geometry (Polygon or MultiPolygon).
    """
    if not isinstance(canvas, Canvas):
        raise TypeError("canvas must be an instance of Canvas.")
    if not isinstance(img, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if img.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    if img.dtype != bool:
        try:
            img = img.astype(bool)
        except:
            raise ValueError("Input image must be of boolean type.")
    if img.shape != (canvas.H, canvas.W):
        raise ValueError(f"Input image shape must match canvas shape {(canvas.H, canvas.W)}.")
    
    px = canvas.dx
    py = canvas.dy
    origin = (canvas.x0 + 0.5*px, canvas.y0 + 0.5*py)
    
    image = img.copy()  # avoid modifying input
    
    # Optional pre-processing
    if bool(sfd):
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            raise ImportError("scipy is required for signed distance function (sfd) processing.")
        
        d_out = distance_transform_edt(image < 0.5)   # distance to background
        d_in  = distance_transform_edt(image > 0.5)    # distance to foreground
        image   = d_out - d_in                   # negative inside, positive outside

    if bool(gaussian):
        image = gaussian_filter(image.astype(float), 
                                sigma=float(gauss_sigma), 
                                preserve_range=True, mode="nearest")  # ~0.6–1.0 px works well

    contours = measure.find_contours(image.astype(float), 0.5, 
                                     fully_connected='high',
                                     positive_orientation='high')

    polys = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # Convert (row, col) -> (x, y) in a bottom-left, y-up world frame
        col = contour[:, 1]
        row = contour[:, 0]
        x = origin[0] + (col - 0.5) * px
        y = origin[1] + (canvas.H - row - 0.5) * py  # <-- flip Y using image height
        coords = np.column_stack([x, y])
        poly = Polygon(coords)
        if poly.is_valid and poly.area > 0:
            polys.append(poly.simplify(tolerance=float(simp_coeff) * min(px, py), preserve_topology=True))

    if not polys:
        return MultiPolygon()

    # Split by orientation: in a y-up frame, CCW = exterior, CW = hole
    outers = []
    holes  = []
    for p in polys:
        if LinearRing(p.exterior.coords).is_ccw:
            outers.append(p)
        else:
            holes.append(p)

    # Fallback: if everything came CW, treat them as outers
    if not outers and holes:
        outers, holes = holes, []

    # For each hole, attach it to the *smallest-area* outer that contains it
    # (handles islands-within-holes robustly)
    holes_for_outer = {i: [] for i in range(len(outers))}
    for h in holes:
        rp = h.representative_point()
        parent_idx = None
        parent_area = float("inf")
        for i, o in enumerate(outers):
            if o.contains(rp) and o.area < parent_area:
                parent_idx, parent_area = i, o.area
        if parent_idx is None:
            # no containing outer → treat as standalone polygon
            outers.append(h)
            holes_for_outer[len(outers) - 1] = []
        else:
            holes_for_outer[parent_idx].append(list(h.exterior.coords))

    # Build final polygons with interiors
    assembled = []
    for i, o in enumerate(outers):
        assembled.append(Polygon(list(o.exterior.coords), holes_for_outer[i]))
        
    result = MultiPolygon(assembled)

    if bool(verbose):
        # Optional: report reconstruction error
        img_recon = shapely_to_numpy(result, canvas)
        err = np.mean((img.astype(float) - img_recon.astype(float))**2)/np.mean(img.astype(float)**2)
        print(f"Image reconstruction error: {err:.6f} (0 = perfect, 1 = total mismatch)")

    return result

# gdstk
def shapely_to_gdstk(geom: 'BaseGeometry') -> List[gdstk.Polygon]:
    """
    Convert a shapely geometry to a gdstk polygon or a list of polygons.
    """
    # Handle empty geometries
    if geom.is_empty:
        return []
    
    parts = getattr(geom, 'geoms', [geom])
    out: List[gdstk.Polygon] = []
    for g in parts:
        # Get exterior
        ex = np.asarray(g.exterior.coords, dtype=float)
        outer = [gdstk.Polygon(ex)]
        
        # Get holes
        holes = [np.asarray(ring.coords, dtype=float) for ring in g.interiors]
        holes_polys = [gdstk.Polygon(h) for h in holes]
        
        if holes_polys:
            result = gdstk.boolean(outer, holes_polys, 'not') or []
            out.extend(result)
        else:
            out.extend(outer)

    return out 

def gdstk_to_shapely(poly: List[gdstk.Polygon]) -> 'BaseGeometry':
    """
    Convert a gdstk polygon to a shapely geometry.
    """
    poly = poly if isinstance(poly, list) else [poly]
    if not poly:
        return shapely.geometry.MultiPolygon()

    out_polys = []
    for p in poly:
        if not isinstance(p, gdstk.Polygon):
            raise TypeError("Input must be a gdstk Polygon or a list of gdstk Polygons.")
        pts = np.asarray(p.points, dtype=float)
        if pts.shape[0] < 3:
            raise ValueError("gdstk Polygon must have at least 3 points.")

        # exterior
        exterior = pts

        # holes: gdstk Polygon.holes may contain arrays or gdstk.Polygon objects
        holes_raw = getattr(p, "holes", []) or []
        holes = []
        for h in holes_raw:
            if isinstance(h, gdstk.Polygon):
                hpts = np.asarray(h.points, dtype=float)
            else:
                hpts = np.asarray(h, dtype=float)
            if hpts.shape[0] >= 3:
                holes.append(hpts)

        shapely_poly = shapely.geometry.Polygon(exterior, holes)
        if not shapely_poly.is_valid:
            shapely_poly = shapely_poly.buffer(0)
        if not shapely_poly.is_empty and shapely_poly.area > 0:
            out_polys.append(shapely_poly)

    if not out_polys:
        return shapely.geometry.MultiPolygon()
    if len(out_polys) == 1:
        return out_polys[0]
    return shapely.geometry.MultiPolygon(out_polys)
    
#Klayout
def shapely_to_klayout(geom: 'BaseGeometry') -> List['pya.Polygon']:
    """
    Convert a shapely geometry to a KLayout polygon or a list of polygons.
    """
    # Handle empty geometries
    if geom.is_empty:
        return []
    
    parts = getattr(geom, 'geoms', [geom])
    dreg = pya.Region()
    for g in parts:
        # Get exterior
        ex = np.asarray(g.exterior.coords, dtype=float)
        ex = ex[:-1]
        dpoly = pya.DPolygon([pya.DPoint(x, y) for x, y in ex])
        dreg += pya.Region(dpoly)
        
        # Add holes
        for ring in g.interiors:
            hole = np.asarray(ring.coords, dtype=float)[:-1]
            hpoly = pya.DPolygon([pya.DPoint(x, y) for x, y in hole])
            dreg -= pya.Region(hpoly)

   
    dreg = dreg.merged()
    out: List[pya.Polygon] = [p for p in dreg.each()]

    return out 
    
def klayout_to_shapely(kl_poly) -> 'BaseGeometry':
    raise NotImplementedError("KLayout conversion not implemented yet.")