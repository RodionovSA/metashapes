# metashapes/generators/samplers/quads.py
# This module defines samplers for quadrilateral shapes

from __future__ import annotations

import numpy as np

from metashapes.lattice.basis import Lattice
from metashapes.generators.samplers.base import ShapeSampler
from metashapes.generators.registry import register_shape_sampler
from metashapes.shape.primitives import Rectangle, ConvexQuad, IsoscelesTrapezoid
from metashapes.generators.samplers.utils import (
    get_all_fixed_param,
    get_all_param_range,
    intersect_ranges,
    lattice_inner_bounds,
    resolve_param,
    resolve_pair_param,
    sample_center_in_bounds,
)


@register_shape_sampler
class RectangleSampler(ShapeSampler):
    shape_class = Rectangle

    def sample(self, rng, lattice: Lattice, config) -> Rectangle:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
        iw, ih = x1 - x0, y1 - y0

        for _ in range(config.max_tries_per_shape):
            width, height = resolve_pair_param(
                rng,
                fixed_value=fixed['size'],
                user_range=ranges['size'],
                default_x_range=intersect_ranges((min_size, iw), (min_feature, iw)),
                default_y_range=intersect_ranges((min_size, ih), (min_feature, ih)),
            )

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 360.0),
            )

            corner_radius = resolve_param(
                rng,
                fixed_value=fixed['corner_radius'],
                user_range=ranges['corner_radius'],
                default_range=(0.0, min(width, height) / 2.0),
            )

            hx = width / 2.0
            hy = height / 2.0
            theta = np.deg2rad(angle)

            dx = abs(hx * np.cos(theta)) + abs(hy * np.sin(theta))
            dy = abs(hx * np.sin(theta)) + abs(hy * np.cos(theta))

            if dx > iw / 2 or dy > ih / 2:
                continue

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + dx, x1 - dx),
                y_bounds=(y0 + dy, y1 - dy),
            )

            return Rectangle(
                center=(cx, cy),
                size=(width, height),
                angle=angle,
                corner_radius=corner_radius,
            )

        raise RuntimeError("rectangle_too_large_for_canvas")


@register_shape_sampler
class IsoscelesTrapezoidSampler(ShapeSampler):
    shape_class = IsoscelesTrapezoid

    def sample(self, rng, lattice: Lattice, config) -> IsoscelesTrapezoid:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
        iw, ih = x1 - x0, y1 - y0

        for _ in range(config.max_tries_per_shape):
            bottom_width = resolve_param(
                rng,
                fixed_value=fixed['bottom_width'],
                user_range=ranges['bottom_width'],
                default_range=intersect_ranges((min_size, iw), (min_feature, iw)),
            )

            top_width = resolve_param(
                rng,
                fixed_value=fixed['top_width'],
                user_range=ranges['top_width'],
                default_range=intersect_ranges((min_size, iw), (min_feature, iw)),
            )

            height = resolve_param(
                rng,
                fixed_value=fixed['height'],
                user_range=ranges['height'],
                default_range=intersect_ranges((min_size, ih), (min_feature, ih)),
            )

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 360.0),
            )

            # max corner radius is limited by the smallest of bottom_width, top_width, height
            max_cr = min(bottom_width, top_width, height) / 2.0
            corner_radius = resolve_param(
                rng,
                fixed_value=fixed['corner_radius'],
                user_range=ranges['corner_radius'],
                default_range=(0.0, max_cr),
            )

            # bounding box: half-width = max base / 2, half-height = height / 2
            hx = max(bottom_width, top_width) / 2.0
            hy = height / 2.0
            theta = np.deg2rad(angle)

            dx = abs(hx * np.cos(theta)) + abs(hy * np.sin(theta))
            dy = abs(hx * np.sin(theta)) + abs(hy * np.cos(theta))

            if dx > iw / 2 or dy > ih / 2:
                continue

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + dx, x1 - dx),
                y_bounds=(y0 + dy, y1 - dy),
            )

            return IsoscelesTrapezoid(
                center=(cx, cy),
                bottom_width=bottom_width,
                top_width=top_width,
                height=height,
                angle=angle,
                corner_radius=corner_radius,
            )

        raise RuntimeError("trapezoid_too_large_for_canvas")


@register_shape_sampler
class ConvexQuadSampler(ShapeSampler):
    shape_class = ConvexQuad

    def sample(self, rng, lattice: Lattice, config) -> ConvexQuad:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = lattice_inner_bounds(lattice)
        iw, ih = x1 - x0, y1 - y0

        for _ in range(config.max_tries_per_shape):
            # sample u and v as axis-aligned half-vectors; global angle rotates them
            ux, vy = resolve_pair_param(
                rng,
                fixed_value=fixed['u'] if fixed['u'] is not None else None,
                user_range=ranges['u'],
                default_x_range=intersect_ranges((min_size / 2, iw / 2), (min_feature / 2, iw / 2)),
                default_y_range=intersect_ranges((min_size / 2, ih / 2), (min_feature / 2, ih / 2)),
            )

            # v is sampled independently if not fixed
            if fixed['v'] is not None:
                vx_val, vy_val = fixed['v']
            else:
                vx_val = 0.0
                vy_val = vy

            u = (ux, 0.0)
            v = (vx_val, vy_val)

            alpha = resolve_param(
                rng,
                fixed_value=fixed['alpha'],
                user_range=ranges['alpha'],
                default_range=(-0.4, 0.4),
            )

            beta = resolve_param(
                rng,
                fixed_value=fixed['beta'],
                user_range=ranges['beta'],
                default_range=(-0.4, 0.4),
            )

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 360.0),
            )

            max_cr = min(ux, abs(vy_val)) * 0.4
            corner_radius = resolve_param(
                rng,
                fixed_value=fixed['corner_radius'],
                user_range=ranges['corner_radius'],
                default_range=(0.0, max_cr),
            )

            # conservative bounding box: largest extent from center across all 4 vertices
            # vertices (before global rotation): center ± (1+|alpha|)*ux ± (1+|beta|)*vy
            ext_x = (1.0 + abs(alpha)) * ux + (1.0 + abs(beta)) * abs(vx_val)
            ext_y = (1.0 + abs(alpha)) * 0.0 + (1.0 + abs(beta)) * abs(vy_val)
            theta = np.deg2rad(angle)
            dx = abs(ext_x * np.cos(theta)) + abs(ext_y * np.sin(theta))
            dy = abs(ext_x * np.sin(theta)) + abs(ext_y * np.cos(theta))

            if dx > iw / 2 or dy > ih / 2:
                continue

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + dx, x1 - dx),
                y_bounds=(y0 + dy, y1 - dy),
            )

            return ConvexQuad(
                center=(cx, cy),
                u=u,
                v=v,
                alpha=alpha,
                beta=beta,
                angle=angle,
                corner_radius=corner_radius,
            )

        raise RuntimeError("convex_quad_too_large_for_canvas")
