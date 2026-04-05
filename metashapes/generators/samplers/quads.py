# metashapes/generators/samplers/quads.py
# This module defines samplers for quadrilateral shapes

from __future__ import annotations

import numpy as np

from metashapes.canvas import Canvas
from metashapes.generators.samplers.base import ShapeSampler
from metashapes.generators.samplers.utils import get_fixed_param, sample_param
from metashapes.generators.registry import register_shape_sampler
from metashapes.shape.primitives import Rectangle
from metashapes.generators.samplers.utils import (
    get_fixed_param,
    get_param_range,
    intersect_ranges,
    resolve_param,
    resolve_pair_param,
    sample_center_in_bounds,
)

@register_shape_sampler("Rectangle")
class RectangleSampler(ShapeSampler):
    shape_name = "Rectangle"

    def sample(self, rng, canvas: Canvas, config) -> Rectangle:
        shape_name = self.shape_name

        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed_size = get_fixed_param(config, shape_name, "size")
        fixed_angle = get_fixed_param(config, shape_name, "angle")
        fixed_center = get_fixed_param(config, shape_name, "center")
        fixed_corner_radius = get_fixed_param(config, shape_name, "corner_radius")

        size_range = get_param_range(config, shape_name, "size")
        angle_range = get_param_range(config, shape_name, "angle")
        center_range = get_param_range(config, shape_name, "center")
        corner_radius_range = get_param_range(config, shape_name, "corner_radius")

        for _ in range(config.max_tries_per_shape):
            width, height = resolve_pair_param(
                rng,
                fixed_value=fixed_size,
                user_range=size_range,
                default_x_range=intersect_ranges(
                    (min_size, canvas.Lx),
                    (min_feature, canvas.Lx),
                ),
                default_y_range=intersect_ranges(
                    (min_size, canvas.Ly),
                    (min_feature, canvas.Ly),
                ),
            )

            angle = resolve_param(
                rng,
                fixed_value=fixed_angle,
                user_range=angle_range,
                default_range=(0.0, 360.0),
            )

            corner_radius = resolve_param(
                rng,
                fixed_value=fixed_corner_radius,
                user_range=corner_radius_range,
                default_range=(0.0, min(width, height) / 2.0),
            )

            hx = width / 2.0
            hy = height / 2.0
            theta = np.deg2rad(angle)

            dx = abs(hx * np.cos(theta)) + abs(hy * np.sin(theta))
            dy = abs(hx * np.sin(theta)) + abs(hy * np.cos(theta))

            if dx > 0.5 * canvas.Lx or dy > 0.5 * canvas.Ly:
                continue

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed_center,
                center_range=center_range,
                x_bounds=(-0.5 * canvas.Lx + dx, 0.5 * canvas.Lx - dx),
                y_bounds=(-0.5 * canvas.Ly + dy, 0.5 * canvas.Ly - dy),
            )

            return Rectangle(
                center=(cx, cy),
                size=(width, height),
                angle=angle,
                corner_radius=corner_radius,
            )

        raise RuntimeError("rectangle_too_large_for_canvas")