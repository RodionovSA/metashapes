# sdflib/ovals.py
# SDF functions for ovals

import torch
from typing import Callable


def CircleSDF(r) -> Callable:
    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q = x*x + y*y
        p = torch.sqrt(q.clamp_min(torch.finfo(q.dtype).tiny)) # safe grads through the origin
        return p - r
    return f


def _ellipse_closest_point(px: torch.Tensor, py: torch.Tensor,
                            a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest point on the boundary of an axis-aligned ellipse (semi-axes
    a, b, centered at the origin) to a first-quadrant point (px, py >= 0).

    Returns (rx, ry), both >= 0 -- the nearest point, folded into the first
    quadrant consistent with (px, py). Closed-form solve (Quilez); the
    near-circular case (a ~= b) is handled by radial projection instead of
    the general solve, which is singular as b^2 - a^2 -> 0.

    Both general-solve branches are formulated to keep their derivatives
    finite everywhere a query can land, not just their values -- each has
    a root/power computation that is otherwise singular at an ordinary,
    non-rare input (see the inline comments on each branch for the exact
    form used).

    Shared by EllipseSDF (whole-ellipse distance) and EggSDF (which
    additionally needs the *point*, not just the distance, to build its
    two-arc SDF).
    """
    swap = px > py
    px2 = torch.where(swap, py, px)
    py2 = torch.where(swap, px, py)
    ax = torch.where(swap, b, a)
    by = torch.where(swap, a, b)

    eps = torch.finfo(px.dtype).eps

    # handle circle / near-circle separately to avoid division by zero.
    # Relative threshold: c ~ (m^2+n^2)/3 grows without bound as l -> 0, so
    # an absolute threshold lets c^3 (below) overflow float32 well before
    # routing the query to this branch. Radial projection's own error near
    # a circle is bounded by |a - b| / 2, i.e. <= ~5e-5 * r0 here.
    l = by * by - ax * ax
    circle_mask = torch.abs(l) < torch.clamp(1e-4 * (by * by + ax * ax), min=eps)
    r0 = 0.5 * (ax + by)

    # radial projection onto the circle of radius r0; near the origin the
    # direction (px2, py2)/r is undefined (both components ~0), so fall
    # back to an arbitrary fixed direction there -- any direction gives the
    # exact magnitude for Ellipse, and Egg only reaches this branch with a
    # near-zero query, where the two arcs meet anyway.
    r2 = px2 * px2 + py2 * py2
    degenerate_center = r2 < eps
    r = torch.sqrt(r2 + eps)
    ux = torch.where(degenerate_center, torch.ones_like(px2), px2 / r)
    uy = torch.where(degenerate_center, torch.zeros_like(py2), py2 / r)
    circle_rx = r0 * ux
    circle_ry = r0 * uy

    # general ellipse branch
    l_safe = torch.where(circle_mask, torch.ones_like(l), l)

    m = ax * px2 / l_safe
    n = by * py2 / l_safe
    m2 = m * m
    n2 = n * n
    k = m * n  # >= 0 always: m and n both carry l_safe's sign

    c = (m2 + n2 - 1.0) / 3.0
    c3 = c * c * c
    d = c3 + k * k
    g = m + m * n2

    sqrt3 = torch.sqrt(torch.as_tensor(3.0, dtype=px.dtype, device=px.device))
    co = torch.zeros_like(px2)

    # branch 1: d < 0 (three real roots)
    mask1 = (d < 0.0) & (~circle_mask)
    if torch.any(mask1):
        # c^3 < 0 throughout this branch, so acos((c^3 + 2*k^2)/c^3) is
        # algebraically identical to 2*asin(k/|c|^1.5)/3 -- and asin, unlike
        # acos, is smooth at its input's zero, which is where any on-axis
        # query (m == 0 or n == 0 => k == 0) lands.
        absc = torch.abs(c)
        c_pow = absc * torch.sqrt(absc)  # |c|^1.5
        pow_ok = mask1 & (c_pow > 0)
        ratio = torch.where(
            pow_ok, k / torch.where(pow_ok, c_pow, torch.ones_like(c_pow)),
            torch.zeros_like(c_pow),
        )
        h = 2.0 * torch.asin(torch.clamp(ratio, -1.0, 1.0)) / 3.0
        s = torch.cos(h)
        t = torch.sin(h) * sqrt3

        # sqrt has an infinite derivative at exactly 0; guard both arms.
        arg_x = torch.clamp(-c * (s + t + 2.0) + m2, min=0.0)
        arg_y = torch.clamp(-c * (s - t + 2.0) + m2, min=0.0)
        x_ok = mask1 & (arg_x > 0)
        y_ok = mask1 & (arg_y > 0)
        rx1 = torch.where(x_ok, torch.sqrt(torch.where(x_ok, arg_x, torch.ones_like(arg_x))),
                          torch.zeros_like(arg_x))
        ry1 = torch.where(y_ok, torch.sqrt(torch.where(y_ok, arg_y, torch.ones_like(arg_y))),
                          torch.zeros_like(arg_y))

        denom = rx1 * ry1
        denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)

        co1 = (ry1 + torch.sign(l_safe) * rx1 + torch.abs(g) / denom - m) * 0.5
        co = torch.where(mask1, co1, co)

    # branch 2: d >= 0 (one real root)
    mask2 = (~mask1) & (~circle_mask)
    if torch.any(mask2):
        # With P = sqrt(d) + k (>= 0), qp = P^2 and qm = (c^3/P)^2
        # algebraically, so their cube roots are s = P^(2/3) and
        # u = c^2 / P^(2/3) = c^2 / s -- avoiding cbrt of a value that
        # passes through 0, whose derivative diverges there.
        d_pos = torch.clamp(d, min=0.0)
        d_ok = mask2 & (d_pos > 0)
        sd = torch.where(d_ok, torch.sqrt(torch.where(d_ok, d_pos, torch.ones_like(d_pos))),
                         torch.zeros_like(d_pos))
        p = sd + k
        p_ok = mask2 & (p > 0)
        s = torch.where(p_ok, torch.pow(torch.where(p_ok, p, torch.ones_like(p)), 2.0 / 3.0),
                        torch.zeros_like(p))
        u = torch.where(p_ok, c * c / torch.where(p_ok, s, torch.ones_like(s)),
                        torch.zeros_like(c))

        rx2 = -s - u - 4.0 * c + 2.0 * m2
        ry2 = (s - u) * sqrt3
        rm = torch.sqrt(torch.clamp(rx2 * rx2 + ry2 * ry2, min=eps))

        denom = torch.sqrt(torch.clamp(rm - rx2, min=eps))
        co2 = (ry2 / denom + 2.0 * g / rm - m) * 0.5
        co = torch.where(mask2, co2, co)

    # numerical safety
    co = torch.clamp(co, -1.0, 1.0)
    si = torch.sqrt(torch.clamp(1.0 - co * co, min=0.0))

    gen_rx = ax * co
    gen_ry = by * si

    rx_folded = torch.where(circle_mask, circle_rx, gen_rx)
    ry_folded = torch.where(circle_mask, circle_ry, gen_ry)

    # un-swap back to the (a, b) orientation
    rx = torch.where(swap, ry_folded, rx_folded)
    ry = torch.where(swap, rx_folded, ry_folded)
    return rx, ry


def EllipseSDF(a, b) -> Callable:
    """
    SDF for an axis-aligned ellipse centered at the origin.

    Parameters:
        a: semi-axis along x
        b: semi-axis along y
    """
    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # work by symmetry in first quadrant
        px = torch.abs(x)
        py = torch.abs(y)

        rx, ry = _ellipse_closest_point(px, py, a, b)

        eps = torch.finfo(x.dtype).eps
        magnitude = torch.sqrt((rx - px) ** 2 + (ry - py) ** 2 + eps)

        # exact sign from the implicit ellipse equation, evaluated at the
        # query itself -- independent of the closest-point solve, so it
        # can't disagree with it near the boundary the way a proxy sign
        # test (e.g. comparing py against the solved ry) can.
        sign = torch.sign((px / a) ** 2 + (py / b) ** 2 - 1.0)

        return sign * magnitude
    return f


def EggSDF(a, b_top, b_bot) -> Callable:
    """
    SDF for an egg centered at the origin: two half-ellipses joined at the
    x-axis.

    Parameters:
        a: semi-axis along x (shared by both halves)
        b_top: semi-axis of the upper half (y >= 0)
        b_bot: semi-axis of the lower half (y < 0)
    """
    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        px = torch.abs(x)
        py = torch.abs(y)

        # The Egg boundary is two arcs sharing endpoints (+-a, 0): the top
        # (y >= 0) follows ellipse (a, b_top), the bottom follows
        # (a, b_bot). Near the seam the true nearest boundary point can lie
        # on the *other* arc (e.g. a point just above y=0 can be much closer
        # to a point on the lower arc than to anything reachable on the
        # upper one), so both arcs are evaluated for every point rather than
        # only the one matching the point's own sign.
        #
        # For each arc, `_ellipse_closest_point` gives the nearest point on
        # its *full* supporting ellipse, folded into the first quadrant; its
        # y-coordinate is always >= 0, so reflecting it onto that arc's own
        # half-plane (+ry for the top arc, -ry for the bottom) and measuring
        # distance from the true (unfolded) query point is exact whenever
        # the query is already on that side, and a close approximation
        # otherwise (worst observed residual ~1.3% of the true distance, for
        # points far from the seam with strongly asymmetric b_top/b_bot).
        # Not a proven-exact closed form.
        eps = torch.finfo(x.dtype).eps

        rx_t, ry_t = _ellipse_closest_point(px, py, a, b_top)
        mag_top = torch.sqrt((rx_t - px) ** 2 + (ry_t - y) ** 2 + eps)

        rx_b, ry_b = _ellipse_closest_point(px, py, a, b_bot)
        mag_bot = torch.sqrt((rx_b - px) ** 2 + (-ry_b - y) ** 2 + eps)

        magnitude = torch.minimum(mag_top, mag_bot)

        b_eff = torch.where(y >= 0, b_top, b_bot)
        sign = torch.sign((px / a) ** 2 + (y / b_eff) ** 2 - 1.0)

        return sign * magnitude
    return f


def StadiumSDF(half_span, r) -> Callable:
    """
    SDF for a stadium (capsule) centered at the origin, long axis along x.

    Parameters:
        half_span: half length of the straight segment between cap centers
        r: cap radius (half the width)
    """
    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps
        dx = torch.clamp(torch.abs(x) - half_span, min=0.0)
        return torch.sqrt(dx * dx + y * y + eps) - r
    return f
