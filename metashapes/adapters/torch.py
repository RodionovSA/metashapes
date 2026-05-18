# metashapes/adapters/torch.py
# OBSOLETE: This adapter is no longer used internally. Will be removed. 

from __future__ import annotations

import math
from turtle import shape

import torch
import torch.nn.functional as F

from metashapes.lattice.canvas import Canvas
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

def shape_to_torch(
    shape: Shape,
    canvas: Canvas,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    soft: bool = False,
    soft_mode: str = "sigmoid",
    softness: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Rasterize a symbolic Shape into a torch mask.

    Parameters:
        shape: Symbolic shape tree.
        canvas: Output raster canvas.
        dtype: Torch dtype of the output mask.
        device: Torch device.
        soft: If True, return a soft mask instead of a hard binary mask.
        soft_mode: sigmoid (support differentiable optimization) or fourier (support anti-aliasing). 
        softness: Edge smoothing scale in world units. If None, defaults to one pixel.

    Returns:
        Tensor of shape (H, W).
        
    Notes:
    soft_mode behavior:
        - "sigmoid": Signed-distance sigmoid smoothing. Intended for
          differentiable optimization and preserves gradient flow from
          shape parameters to the mask. Smaller `softness` makes the mask
          sharper but can lead to vanishing gradients, while larger
          `softness` gives smoother gradients but blurs geometry more.
        - "fourier": Gaussian low-pass filtering of a hard mask in Fourier
          space. Useful for anti-aliasing and smooth visual filtering, but
          generally not suitable for gradient-based optimization because
          hard thresholding largely destroys gradients before filtering.
    """
    if not isinstance(shape, Shape):
        raise TypeError("shape must be a Shape")
    if not isinstance(canvas, Canvas):
        raise TypeError("canvas must be a Canvas")

    device = torch.device(device)

    if softness is None:
        softness = min(canvas.dx, canvas.dy)

    if soft_mode not in {"sigmoid", "fourier"}:
        raise ValueError("soft_mode must be 'sigmoid' or 'fourier'")

    x, y = _meshgrid(canvas, dtype=dtype, device=device)
    sdf = _eval(shape, x, y, dtype=dtype, device=device)

    if not soft:
        return (sdf <= 0).to(dtype)

    if softness is None:
        softness = min(canvas.dx, canvas.dy)

    if soft_mode == "sigmoid":
        soft_mask = _soften_mask_sigmoid(sdf, softness)
    elif soft_mode == "fourier":
        hard_mask = (sdf <= 0).to(dtype)
        soft_mask = _soften_mask_fourier(hard_mask, canvas, softness)
    else:
        raise ValueError("soft_mode must be 'sigmoid' or 'fourier'")

    return soft_mask.to(dtype)

# ----- Eval functions -----

def _eval_rectangle(
    shape: Rectangle,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cx = _as_tensor(shape.center[0], dtype=dtype, device=device)
    cy = _as_tensor(shape.center[1], dtype=dtype, device=device)
    w = _as_tensor(shape.size[0], dtype=dtype, device=device)
    h = _as_tensor(shape.size[1], dtype=dtype, device=device)

    x_local, y_local = _inverse_rotate(
        x, y, shape.angle, shape.center, dtype=dtype, device=device
    )
    xr = x_local - cx
    yr = y_local - cy

    return _sdf_box(xr, yr, w / 2, h / 2)


def _eval_ellipse(
    shape: Ellipse,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cx = _as_tensor(shape.center[0], dtype=dtype, device=device)
    cy = _as_tensor(shape.center[1], dtype=dtype, device=device)
    a = _as_tensor(shape.axes[0], dtype=dtype, device=device)
    b = _as_tensor(shape.axes[1], dtype=dtype, device=device)

    x_local, y_local = _inverse_rotate(
        x, y, shape.angle, shape.center, dtype=dtype, device=device
    )
    xr = x_local - cx
    yr = y_local - cy

    eps = torch.as_tensor(1e-12, dtype=dtype, device=device)

    phi = (xr / (a + eps)) ** 2 + (yr / (b + eps)) ** 2 - 1.0
    grad_norm = torch.sqrt(
        (2.0 * xr / (a * a + eps)) ** 2 +
        (2.0 * yr / (b * b + eps)) ** 2 + eps
    )
    return phi / grad_norm

def _eval_regular_polygon(
    shape: RegularPolygon,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cx = _as_tensor(shape.center[0], dtype=dtype, device=device)
    cy = _as_tensor(shape.center[1], dtype=dtype, device=device)
    n = shape.n
    s = _as_tensor(shape.side_length, dtype=dtype, device=device)

    pi = torch.as_tensor(torch.pi, dtype=dtype, device=device)
    R = s / (2.0 * torch.sin(pi / n))
    phi = pi / 2 + _as_tensor(shape.angle, dtype=dtype, device=device) * (pi / 180.0)

    verts = []
    for k in range(n):
        ang = 2.0 * pi * k / n + phi
        vx = cx + R * torch.cos(ang)
        vy = cy + R * torch.sin(ang)
        verts.append((vx, vy))

    inside_field = None
    unsigned_dist = None

    for k in range(n):
        x1, y1 = verts[k]
        x2, y2 = verts[(k + 1) % n]

        ex = x2 - x1
        ey = y2 - y1

        # inward normal for CCW polygon
        nx = ey
        ny = -ex
        norm = torch.sqrt(nx * nx + ny * ny)
        nx = nx / norm
        ny = ny / norm

        dk = (x - x1) * nx + (y - y1) * ny
        inside_field = dk if inside_field is None else torch.maximum(inside_field, dk)

        # distance to segment
        wx = x - x1
        wy = y - y1
        vv = ex * ex + ey * ey
        t = (wx * ex + wy * ey) / vv
        t = torch.clamp(t, 0.0, 1.0)

        px = x1 + t * ex
        py = y1 + t * ey

        dist_k = torch.sqrt((x - px) ** 2 + (y - py) ** 2)
        unsigned_dist = dist_k if unsigned_dist is None else torch.minimum(unsigned_dist, dist_k)

    inside = inside_field <= 0
    return torch.where(inside, -unsigned_dist, unsigned_dist)

def _eval_cross(
    shape: Cross,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cx = _as_tensor(shape.center[0], dtype=dtype, device=device)
    cy = _as_tensor(shape.center[1], dtype=dtype, device=device)
    size = _as_tensor(shape.size, dtype=dtype, device=device)
    width = _as_tensor(shape.width, dtype=dtype, device=device)

    x_local, y_local = _inverse_rotate(
        x, y, shape.angle, shape.center, dtype=dtype, device=device
    )

    px = torch.abs(x_local - cx)
    py = torch.abs(y_local - cy)

    a = size / 2.0
    b = width / 2.0
    zero = torch.zeros((), dtype=dtype, device=device)

    inside = ((px <= a) & (py <= b)) | ((px <= b) & (py <= a))

    ty1 = torch.minimum(torch.maximum(py, zero), b)
    d1 = torch.sqrt((px - a) ** 2 + (py - ty1) ** 2)

    tx2 = torch.minimum(torch.maximum(px, zero), b)
    d2 = torch.sqrt((py - a) ** 2 + (px - tx2) ** 2)

    ty3 = torch.minimum(torch.maximum(py, b), a)
    d3 = torch.sqrt((px - b) ** 2 + (py - ty3) ** 2)

    tx4 = torch.minimum(torch.maximum(px, b), a)
    d4 = torch.sqrt((py - b) ** 2 + (px - tx4) ** 2)

    unsigned_dist = torch.minimum(torch.minimum(d1, d2), torch.minimum(d3, d4))
    return torch.where(inside, -unsigned_dist, unsigned_dist)

def _eval_ring(
    shape: Ring,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    outer = Ellipse(
        center=shape.center,
        axes=shape.outer_axes,
        angle=shape.angle,
        resolution=shape.resolution,
    )
    inner = Ellipse(
        center=shape.center,
        axes=shape.inner_axes,
        angle=shape.angle,
        resolution=shape.resolution,
    )

    return _eval(Difference(outer, inner), x, y, dtype=dtype, device=device)

def _eval_moon(
    shape: Moon,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cx = _as_tensor(shape.center[0], dtype=dtype, device=device)
    cy = _as_tensor(shape.center[1], dtype=dtype, device=device)
    r = _as_tensor(shape.radius, dtype=dtype, device=device)
    cut_ratio = _as_tensor(shape.cut_ratio, dtype=dtype, device=device)

    main = Ellipse(
        center=shape.center,
        axes=(shape.radius, shape.radius),
        angle=0.0,
        resolution=shape.resolution,
    )

    cut_cx = cx + 2.0 * r * (1.0 - cut_ratio)
    cut = Ellipse(
        center=(cut_cx, cy),
        axes=(shape.radius, shape.radius),
        angle=0.0,
        resolution=shape.resolution,
    )

    base = Difference(main, cut)

    if shape.angle != 0:
        base = Rotate(base, angle=shape.angle, origin=shape.center)

    return _eval(base, x, y, dtype=dtype, device=device)

def _eval_rounded_rectangle(
    shape: RoundedRectangle,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cx = _as_tensor(shape.center[0], dtype=dtype, device=device)
    cy = _as_tensor(shape.center[1], dtype=dtype, device=device)
    w = _as_tensor(shape.size[0], dtype=dtype, device=device)
    h = _as_tensor(shape.size[1], dtype=dtype, device=device)
    r = _as_tensor(shape.radius, dtype=dtype, device=device)

    x_local, y_local = _inverse_rotate(
        x, y, shape.angle, shape.center, dtype=dtype, device=device
    )
    xr = x_local - cx
    yr = y_local - cy

    return _sdf_box(xr, yr, w / 2 - r, h / 2 - r) - r

def _eval_rounded_regular_polygon(
    shape: RoundedRegularPolygon,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    r = _as_tensor(shape.radius, dtype=dtype, device=device)
    s = _as_tensor(shape.side_length, dtype=dtype, device=device)

    pi = torch.as_tensor(torch.pi, dtype=dtype, device=device)
    shrink = 2.0 * r * torch.tan(pi / shape.n)
    s_inset = s - shrink

    inset_shape = RegularPolygon(
        center=shape.center,
        n=shape.n,
        side_length=s_inset,
        angle=shape.angle,
    )

    return _eval_regular_polygon(inset_shape, x, y, dtype=dtype, device=device) - r


def _eval(
    shape: Shape,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    
    if isinstance(shape, Rectangle):
        return _eval_rectangle(
            shape, x, y,
            dtype=dtype,
            device=device,
        )

    if isinstance(shape, Ellipse):
        return _eval_ellipse(
            shape, x, y,
            dtype=dtype,
            device=device,
        )
        
    if isinstance(shape, RegularPolygon):
        return _eval_regular_polygon(shape, x, y, dtype=dtype, device=device)

    if isinstance(shape, Cross):
        return _eval_cross(shape, x, y, dtype=dtype, device=device)

    if isinstance(shape, Ring):
        return _eval_ring(shape, x, y, dtype=dtype, device=device)

    if isinstance(shape, Moon):
        return _eval_moon(shape, x, y, dtype=dtype, device=device)
    
    if isinstance(shape, RoundedRectangle):
        return _eval_rounded_rectangle(shape, x, y, dtype=dtype, device=device)

    if isinstance(shape, RoundedRegularPolygon):
        return _eval_rounded_regular_polygon(shape, x, y, dtype=dtype, device=device)
    
    if isinstance(shape, RoundedCross):
        #return _eval_rounded_cross(shape, x, y, dtype=dtype, device=device)
        raise NotImplementedError("RoundedCross SDF is not implemented yet")

    if isinstance(shape, RoundedMoon):
        #return _eval_rounded_moon(shape, x, y, dtype=dtype, device=device)
        raise NotImplementedError("RoundedMoon SDF is not implemented yet")

    if isinstance(shape, Translate):
        x_new, y_new = _inverse_translate(
            x, y, shape.dx, shape.dy, dtype=dtype, device=device
        )
        return _eval(
            shape.shape, x_new, y_new,
            dtype=dtype,
            device=device,
        )

    if isinstance(shape, Rotate):
        x_new, y_new = _inverse_rotate(
            x, y, shape.angle, shape.origin, dtype=dtype, device=device
        )
        return _eval(
            shape.shape, x_new, y_new,
            dtype=dtype,
            device=device,
        )

    if isinstance(shape, Scale):
        x_new, y_new = _inverse_scale(
            x, y, shape.sx, shape.sy, shape.origin, dtype=dtype, device=device
        )
        return _eval(
            shape.shape, x_new, y_new,
            dtype=dtype,
            device=device,
        )

    if isinstance(shape, Union):
        d1 = _eval(shape.left, x, y, dtype=dtype, device=device)
        d2 = _eval(shape.right, x, y, dtype=dtype, device=device)
        return torch.minimum(d1, d2)

    if isinstance(shape, Intersection):
        d1 = _eval(shape.left, x, y, dtype=dtype, device=device)
        d2 = _eval(shape.right, x, y, dtype=dtype, device=device)
        return torch.maximum(d1, d2)

    if isinstance(shape, Difference):
        d1 = _eval(shape.left, x, y, dtype=dtype, device=device)
        d2 = _eval(shape.right, x, y, dtype=dtype, device=device)
        return torch.maximum(d1, -d2)

    raise TypeError(f"Unsupported shape type: {type(shape).__name__}")

# ----- Helper functions -----

def _as_tensor(value, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Convert a scalar-like value to a torch tensor on the requested device/dtype.

    If `value` is already a tensor, it is only moved/cast and remains in the
    autograd graph.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def _meshgrid(
    canvas: Canvas,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return world-coordinate pixel-center grids X, Y of shape (H, W).
    """
    x = canvas.x0 + (torch.arange(canvas.W, dtype=dtype, device=device) + 0.5) * canvas.dx
    y = canvas.y1 - (torch.arange(canvas.H, dtype=dtype, device=device) + 0.5) * canvas.dy
    return torch.meshgrid(y, x, indexing="ij")[1], torch.meshgrid(y, x, indexing="ij")[0]


def _sigmoid(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


def _inverse_translate(
    x: torch.Tensor,
    y: torch.Tensor,
    dx,
    dy,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx_t = _as_tensor(dx, dtype=dtype, device=device)
    dy_t = _as_tensor(dy, dtype=dtype, device=device)
    return x - dx_t, y - dy_t


def _inverse_rotate(
    x: torch.Tensor,
    y: torch.Tensor,
    angle_deg,
    origin,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ox = _as_tensor(origin[0], dtype=dtype, device=device)
    oy = _as_tensor(origin[1], dtype=dtype, device=device)
    a = _as_tensor(angle_deg, dtype=dtype, device=device) * (math.pi / 180.0)

    c = torch.cos(-a)
    s = torch.sin(-a)

    xr = x - ox
    yr = y - oy

    x_new = ox + c * xr - s * yr
    y_new = oy + s * xr + c * yr
    return x_new, y_new

def _inverse_scale(
    x: torch.Tensor,
    y: torch.Tensor,
    sx,
    sy,
    origin,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ox = _as_tensor(origin[0], dtype=dtype, device=device)
    oy = _as_tensor(origin[1], dtype=dtype, device=device)
    sx_t = _as_tensor(sx, dtype=dtype, device=device)
    sy_t = _as_tensor(sy, dtype=dtype, device=device)

    return ox + (x - ox) / sx_t, oy + (y - oy) / sy_t

def _sdf_box(
    x: torch.Tensor,
    y: torch.Tensor,
    hx: torch.Tensor,
    hy: torch.Tensor,
) -> torch.Tensor:
    """
    Signed distance to an axis-aligned box centered at origin
    with half-sizes hx, hy.
    """
    qx = torch.abs(x) - hx
    qy = torch.abs(y) - hy

    outside = torch.sqrt(torch.clamp(qx, min=0.0) ** 2 + torch.clamp(qy, min=0.0) ** 2)
    inside = torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx))
    return outside + inside

# ----- Softening functions -----
def _soften_mask_sigmoid(sdf: torch.Tensor, smoothness) -> torch.Tensor:
    """
    Convert signed distance to a soft occupancy mask in [0, 1].
    Negative inside -> near 1, positive outside -> near 0.
    """
    eps = torch.as_tensor(1e-12, dtype=sdf.dtype, device=sdf.device)
    s = torch.clamp(_as_tensor(smoothness, dtype=sdf.dtype, device=sdf.device), min=eps)

    out = torch.zeros_like(sdf)
    inside = sdf <= -s
    band = torch.abs(sdf) < s

    out[inside] = 1.0
    out[band] = 0.5 * (1.0 - sdf[band] / s)
    return out

def _soften_mask_fourier(
    mask: torch.Tensor,
    canvas: Canvas,
    softness,
) -> torch.Tensor:
    """
    Smooth a mask by Gaussian low-pass filtering in Fourier space.

    Parameters:
        mask:
            Tensor of shape (H, W).
        canvas:
            Raster canvas.
        softness:
            Smoothing scale in world units.

    Returns:
        Float mask in [0, 1].
    """
    softness_t = _as_tensor(softness, dtype=mask.dtype, device=mask.device)

    ky = torch.fft.fftfreq(canvas.H, d=canvas.dy, device=mask.device, dtype=mask.dtype)
    kx = torch.fft.fftfreq(canvas.W, d=canvas.dx, device=mask.device, dtype=mask.dtype)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    K2 = KX**2 + KY**2

    filt = torch.exp(-2.0 * (torch.pi**2) * softness_t**2 * K2)

    mask_f = torch.fft.fft2(mask)
    mask_lp = torch.fft.ifft2(mask_f * filt).real

    return torch.clamp(mask_lp, 0.0, 1.0)