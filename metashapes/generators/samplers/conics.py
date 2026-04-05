# metashapes/generators/samplers/conics.py
# This module defines samplers for conic shapes

from __future__ import annotations

import numpy as np

from metashapes.canvas import Canvas
from metashapes.generators.registry import register_shape_sampler
from metashapes.generators.samplers.base import ShapeSampler
from metashapes.generators.samplers.utils import (
    get_fixed_param,
    get_param_range,
    intersect_ranges,
    resolve_param,
    resolve_pair_param,
    sample_center_in_bounds,
)
from metashapes.shape.primitives import Ellipse


@register_shape_sampler("Ellipse")
class EllipseSampler(ShapeSampler):
    def sample(self, rng, canvas: Canvas, config) -> Ellipse:
        shape_name = "Ellipse"

        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed_axes = get_fixed_param(config, shape_name, "axes")
        fixed_angle = get_fixed_param(config, shape_name, "angle")
        fixed_center = get_fixed_param(config, shape_name, "center")

        axes_range = get_param_range(config, shape_name, "axes")
        angle_range = get_param_range(config, shape_name, "angle")
        center_range = get_param_range(config, shape_name, "center")

        for _ in range(config.max_tries_per_shape):
            ax, ay = resolve_pair_param(
                rng,
                fixed_value=fixed_axes,
                user_range=axes_range,
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

            hx = 0.5 * ax
            hy = 0.5 * ay
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

            return Ellipse(
                center=(cx, cy),
                axes=(ax, ay),
                angle=angle,
            )

        raise RuntimeError("ellipse_too_large_for_canvas")