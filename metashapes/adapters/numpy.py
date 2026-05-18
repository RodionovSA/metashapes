# metashapes/adapters/numpy.py
# OBSOLETE: This adapter is no longer used internally. Will be removed. 

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from metashapes.lattice.canvas import Canvas
from metashapes.patterned_layer import PatternedLayer
from metashapes.shape.base import Shape
from metashapes.shape.primitives import (
    Rectangle,
    Ellipse,
    RegularPolygon,
    Cross,
    Ring,
    Moon,
    RoundedRectangle,
    RoundedRegularPolygon,
    RoundedCross,
    RoundedMoon,
)
from metashapes.shape.boolean import Union, Intersection, Difference
from metashapes.shape.transforms import Translate, Rotate, Scale

def shape_to_numpy(shape: Shape, 
                   canvas: Canvas, 
                   *, 
                   dtype=bool,
                   soft: bool = False,
                   soft_mode: str = "sigmoid",
                   softness: float | None = None) -> np.ndarray:
    """
    Rasterize a symbolic Shape directly into a NumPy mask.

    Parameters:
        shape: Symbolic shape tree.
        canvas: Output raster canvas.
        dtype: Output dtype, typically bool or np.uint8.
        soft: If True, return a soft mask instead of a hard binary mask.
        soft_mode:
            Smoothing mode:
                - "sigmoid": signed-distance sigmoid smoothing
                - "fourier": Gaussian low-pass filtering
        softness: Edge smoothing scale in world units. If None, defaults to one pixel.
    Returns:
        2D array of shape (H, W).
    """
    if not isinstance(shape, Shape):
        raise TypeError("shape must be a Shape")
    if not isinstance(canvas, Canvas):
        raise TypeError("canvas must be a Canvas")

    x, y = _meshgrid(canvas)
    mask = _eval(shape, x, y, canvas)

    if not soft:
        if dtype is bool:
            return mask
        return mask.astype(dtype)

    if softness is None:
        softness = min(canvas.dx, canvas.dy)

    if soft_mode == "sigmoid":
        soft_mask = _soften_mask_sigmoid(mask, canvas, softness)
    elif soft_mode == "fourier":
        soft_mask = _soften_mask_fourier(mask, canvas, softness)
    else:
        raise ValueError("soft_mode must be 'sigmoid' or 'fourier'")

    if dtype is bool:
        return soft_mask >= 0.5
    return soft_mask.astype(dtype)

# ----- Eval functions -----

def _eval_rectangle(shape: Rectangle, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx, cy = shape.center
    w, h = shape.size

    x_local, y_local = _inverse_rotate(x, y, shape.angle, origin=shape.center)
    return (
        (np.abs(x_local - cx) <= w / 2)
        & (np.abs(y_local - cy) <= h / 2)
    )


def _eval_ellipse(shape: Ellipse, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx, cy = shape.center
    a, b = shape.axes

    x_local, y_local = _inverse_rotate(x, y, shape.angle, origin=shape.center)
    xr = x_local - cx
    yr = y_local - cy

    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def _eval_regular_polygon(shape: RegularPolygon, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx, cy = shape.center
    n = shape.n
    s = shape.side_length

    R = s / (2.0 * np.sin(np.pi / n))
    phi = np.pi / 2.0 + np.deg2rad(shape.angle)

    vertices = np.array([
        (
            cx + R * np.cos(2.0 * np.pi * k / n + phi),
            cy + R * np.sin(2.0 * np.pi * k / n + phi),
        )
        for k in range(n)
    ])

    return _point_in_convex_polygon(x, y, vertices)


def _eval_cross(shape: Cross, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    vertical = Rectangle(center=shape.center, size=(shape.width, shape.size))
    horizontal = Rectangle(center=shape.center, size=(shape.size, shape.width))
    mask = _eval_rectangle(vertical, x, y) | _eval_rectangle(horizontal, x, y)

    if shape.angle != 0:
        x_local, y_local = _inverse_rotate(x, y, shape.angle, origin=shape.center)
        return _eval_cross(Cross(center=shape.center, size=shape.size, width=shape.width, angle=0.0), x_local, y_local)

    return mask


def _eval_ring(shape: Ring, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    outer = Ellipse(center=shape.center, axes=shape.outer_axes, angle=shape.angle)
    inner = Ellipse(center=shape.center, axes=shape.inner_axes, angle=shape.angle)
    return _eval_ellipse(outer, x, y) & (~_eval_ellipse(inner, x, y))


def _eval_moon(shape: Moon, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx, cy = shape.center
    r = shape.radius

    x_local, y_local = _inverse_rotate(x, y, shape.angle, origin=shape.center)

    main = (x_local - cx) ** 2 + (y_local - cy) ** 2 <= r ** 2

    cut_cx = cx + 2.0 * r * (1.0 - shape.cut_ratio)
    cut = (x_local - cut_cx) ** 2 + (y_local - cy) ** 2 <= r ** 2

    return main & (~cut)

def _eval_rounded_rectangle(shape: RoundedRectangle, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx, cy = shape.center
    w, h = shape.size
    r = shape.radius

    x_local, y_local = _inverse_rotate(x, y, shape.angle, origin=shape.center)
    xr = x_local - cx
    yr = y_local - cy

    d = _sdf_box(xr, yr, w / 2 - r, h / 2 - r) - r
    return d <= 0.0

def _eval_rounded_regular_polygon(
    shape: RoundedRegularPolygon,
    x: np.ndarray,
    y: np.ndarray,
    canvas: Canvas,
) -> np.ndarray:
    """
    Evaluate a rounded regular polygon by first building the plain polygon mask,
    then rounding the composed mask.
    """
    base = RegularPolygon(
        center=shape.center,
        n=shape.n,
        side_length=shape.side_length,
        angle=shape.angle,
    )

    mask = _eval(base, x, y, canvas)
    return _round_mask(mask, radius=shape.radius, canvas=canvas, mode="outer")


def _eval_rounded_cross(
    shape: RoundedCross,
    x: np.ndarray,
    y: np.ndarray,
    canvas: Canvas,
) -> np.ndarray:
    """
    Evaluate a rounded cross by first building the plain cross mask,
    then rounding the composed shape so both inner and outer corners
    are handled correctly.
    """
    base = Cross(
        center=shape.center,
        size=shape.size,
        width=shape.width,
        angle=shape.angle,
    )

    mask = _eval(base, x, y, canvas)
    return _round_mask(mask, radius=shape.radius, canvas=canvas, mode="both")

def _eval_rounded_moon(
    shape: RoundedMoon,
    x: np.ndarray,
    y: np.ndarray,
    canvas: Canvas,
) -> np.ndarray:
    """
    Evaluate a rounded moon by first building the plain moon mask,
    then rounding the composed mask.
    """
    base = Moon(
        center=shape.center,
        radius=shape.radius,
        cut_ratio=shape.cut_ratio,
        angle=shape.angle,
        resolution=shape.resolution,
    )

    mask = _eval(base, x, y, canvas)
    return _round_mask(
        mask,
        radius=shape.rounding_radius,
        canvas=canvas,
        mode="outer",
    )

def _eval(shape: Shape, x: np.ndarray, y: np.ndarray, canvas: Canvas) -> np.ndarray:
    if isinstance(shape, Rectangle):
        return _eval_rectangle(shape, x, y)

    if isinstance(shape, Ellipse):
        return _eval_ellipse(shape, x, y)

    if isinstance(shape, RegularPolygon):
        return _eval_regular_polygon(shape, x, y)

    if isinstance(shape, Cross):
        return _eval_cross(shape, x, y)

    if isinstance(shape, Ring):
        return _eval_ring(shape, x, y)

    if isinstance(shape, Moon):
        return _eval_moon(shape, x, y)

    if isinstance(shape, Union):
        return _eval(shape.left, x, y, canvas) | _eval(shape.right, x, y, canvas)

    if isinstance(shape, Intersection):
        return _eval(shape.left, x, y, canvas) & _eval(shape.right, x, y, canvas)

    if isinstance(shape, Difference):
        return _eval(shape.left, x, y, canvas) & (~_eval(shape.right, x, y, canvas))

    if isinstance(shape, Translate):
        x_new, y_new = _inverse_translate(x, y, shape.dx, shape.dy)
        return _eval(shape.shape, x_new, y_new, canvas)
    
    if isinstance(shape, Rotate):
        x_new, y_new = _inverse_rotate(x, y, shape.angle, shape.origin)
        return _eval(shape.shape, x_new, y_new, canvas)

    if isinstance(shape, Scale):
        x_new, y_new = _inverse_scale(x, y, shape.sx, shape.sy, shape.origin)
        return _eval(shape.shape, x_new, y_new, canvas)

    if isinstance(shape, RoundedRectangle):
        return _eval_rounded_rectangle(shape, x, y)

    if isinstance(shape, RoundedRegularPolygon):
        return _eval_rounded_regular_polygon(shape, x, y, canvas)

    if isinstance(shape, RoundedCross):
        return _eval_rounded_cross(shape, x, y, canvas)

    if isinstance(shape, RoundedMoon):
        return _eval_rounded_moon(shape, x, y, canvas)

    raise TypeError(f"Unsupported shape type: {type(shape).__name__}")

# ----- Helper functions -----
def _meshgrid(canvas: Canvas) -> tuple[np.ndarray, np.ndarray]:
    """
    Return world-coordinate pixel-center grids X, Y of shape (H, W).
    """
    x = canvas.x_centers
    y = canvas.y_centers
    return np.meshgrid(x, y)

def _rotate_coords(
    x: np.ndarray,
    y: np.ndarray,
    angle_deg,
    origin,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate coordinates by +angle around origin.
    """
    ox, oy = origin

    a = np.deg2rad(angle_deg)
    c = np.cos(a)
    s = np.sin(a)

    xr = x - ox
    yr = y - oy

    x_new = ox + c * xr - s * yr
    y_new = oy + s * xr + c * yr
    return x_new, y_new


def _inverse_translate(x, y, dx, dy):
    return x - dx, y - dy


def _inverse_rotate(x, y, angle_deg, origin):
    return _rotate_coords(x, y, -angle_deg, origin)


def _inverse_scale(x, y, sx, sy, origin):
    if origin == "center":
        ox, oy = 0.0, 0.0
    elif origin == "centroid":
        raise ValueError("origin='centroid' is not supported in numpy adapter")
    else:
        ox, oy = origin

    return ox + (x - ox) / sx, oy + (y - oy) / sy


def _point_in_convex_polygon(x, y, vertices: np.ndarray) -> np.ndarray:
    """
    Vectorized inside-test for a convex polygon.

    Parameters:
        x, y:
            Arrays of shape (H, W)
        vertices:
            Array of shape (N, 2), counter-clockwise ordered
    """
    inside = np.ones_like(x, dtype=bool)
    n = len(vertices)

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        inside &= (cross <= 0)

    return inside


def _length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(x * x + y * y)

def _sdf_box(x: np.ndarray, y: np.ndarray, hx, hy) -> np.ndarray:
    """
    Signed distance to axis-aligned box centered at origin,
    with half-sizes hx, hy.
    """
    qx = np.abs(x) - hx
    qy = np.abs(y) - hy

    outside = _length(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside

def _dilate_mask(mask: np.ndarray, radius: float, canvas: Canvas) -> np.ndarray:
    """
    Binary dilation by a disk of radius `radius` in world units.
    """
    dist = distance_transform_edt(~mask, sampling=(canvas.dy, canvas.dx))
    return dist <= radius


def _erode_mask(mask: np.ndarray, radius: float, canvas: Canvas) -> np.ndarray:
    """
    Binary erosion by a disk of radius `radius` in world units.
    """
    dist = distance_transform_edt(mask, sampling=(canvas.dy, canvas.dx))
    return dist > radius


def _round_mask(
    mask: np.ndarray,
    radius: float,
    canvas: Canvas,
    mode: str = "both",
) -> np.ndarray:
    """
    Round mask corners by binary morphology.

    mode:
        - "outer": round convex/outward corners (opening)
        - "inner": round concave/inward corners (closing)
        - "both": closing then opening
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return mask

    if mode not in {"outer", "inner", "both"}:
        raise ValueError("mode must be 'outer', 'inner', or 'both'")

    out = mask.copy()

    # closing: rounds/fills inner concave corners
    if mode in {"inner", "both"}:
        out = _erode_mask(_dilate_mask(out, radius, canvas), radius, canvas)

    # opening: rounds outer convex corners
    if mode in {"outer", "both"}:
        out = _dilate_mask(_erode_mask(out, radius, canvas), radius, canvas)

    return out

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _soften_mask_sigmoid(
    mask: np.ndarray,
    canvas: Canvas,
    softness: float,
) -> np.ndarray:
    """
    Convert a hard binary mask into a soft mask using signed distance + sigmoid.

    Parameters:
        mask:
            Binary mask.
        canvas:
            Raster canvas.
        softness:
            Transition width in world units. Smaller values = sharper edge.

    Returns:
        Float mask in [0, 1].
    """
    if softness <= 0:
        raise ValueError("softness must be positive")

    # distance outside object
    d_out = distance_transform_edt(~mask, sampling=(canvas.dy, canvas.dx))
    # distance inside object
    d_in = distance_transform_edt(mask, sampling=(canvas.dy, canvas.dx))

    # signed distance: negative inside, positive outside
    sdf = d_out - d_in

    # inside should be near 1, outside near 0
    return _sigmoid(-sdf / softness)


def _soften_mask_fourier(
    mask: np.ndarray,
    canvas: Canvas,
    softness: float,
) -> np.ndarray:
    """
    Smooth a hard mask by Gaussian low-pass filtering in Fourier space.

    Parameters:
        mask:
            Binary mask.
        canvas:
            Raster canvas.
        softness:
            Smoothing scale in world units.

    Returns:
        Float mask in [0, 1].
    """
    if softness <= 0:
        raise ValueError("softness must be positive")

    arr = mask.astype(float)

    ky = np.fft.fftfreq(canvas.H, d=canvas.dy)
    kx = np.fft.fftfreq(canvas.W, d=canvas.dx)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    # Gaussian low-pass in frequency domain
    sigma = softness
    filt = np.exp(-2.0 * (np.pi**2) * sigma**2 * K2)

    arr_f = np.fft.fft2(arr)
    arr_lp = np.fft.ifft2(arr_f * filt).real

    return np.clip(arr_lp, 0.0, 1.0)