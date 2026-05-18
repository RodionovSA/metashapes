# metashapes/generators/samplers/junctions.py
# This module defines samplers for junction shapes (Cross, TShape)

from __future__ import annotations

import numpy as np

from metashapes.lattice.canvas import Canvas
from metashapes.generators.registry import register_shape_sampler
from metashapes.generators.samplers.base import ShapeSampler
from metashapes.generators.samplers.utils import (
    get_all_fixed_param,
    get_all_param_range,
    intersect_ranges,
    resolve_param,
    sample_center_in_bounds,
)
from metashapes.shape.primitives import Cross, TShape


@register_shape_sampler
class CrossSampler(ShapeSampler):
    shape_class = Cross

    def sample(self, rng, canvas: Canvas, config) -> Cross:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = canvas.inner_bounds
        iw, ih = x1 - x0, y1 - y0
        # bounding box of a cross is length x length
        max_length = min(iw, ih)

        for _ in range(config.max_tries_per_shape):
            length = resolve_param(
                rng,
                fixed_value=fixed['length'],
                user_range=ranges['length'],
                default_range=intersect_ranges((min_size, max_length), (min_feature, max_length)),
            )

            # width < length; also width >= min_feature
            width = resolve_param(
                rng,
                fixed_value=fixed['width'],
                user_range=ranges['width'],
                default_range=intersect_ranges((min_feature, length), (min_size, length)),
            )

            if width >= length:
                continue

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 90.0),  # 90° symmetry
            )

            # outer_corner_radius must be < width / 2
            max_ocr = width / 2.0 - 1e-6
            if max_ocr <= 0:
                continue
            outer_corner_radius = resolve_param(
                rng,
                fixed_value=fixed['outer_corner_radius'],
                user_range=ranges['outer_corner_radius'],
                default_range=(0.0, max_ocr),
            )

            # inner_corner_radius must be <= length/2 - width/2 - outer_corner_radius
            max_icr = length / 2.0 - width / 2.0 - outer_corner_radius
            if max_icr < 0:
                continue
            inner_corner_radius = resolve_param(
                rng,
                fixed_value=fixed['inner_corner_radius'],
                user_range=ranges['inner_corner_radius'],
                default_range=(0.0, max_icr),
            )

            # bounding box is length x length, rotated
            half = length / 2.0
            theta = np.deg2rad(angle)
            dx = abs(half * np.cos(theta)) + abs(half * np.sin(theta))
            dy = abs(half * np.sin(theta)) + abs(half * np.cos(theta))

            if dx > iw / 2 or dy > ih / 2:
                continue

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + dx, x1 - dx),
                y_bounds=(y0 + dy, y1 - dy),
            )

            return Cross(
                center=(cx, cy),
                length=length,
                width=width,
                angle=angle,
                outer_corner_radius=outer_corner_radius,
                inner_corner_radius=inner_corner_radius,
            )

        raise RuntimeError("cross_too_large_for_canvas")


@register_shape_sampler
class TShapeSampler(ShapeSampler):
    shape_class = TShape

    def sample(self, rng, canvas: Canvas, config) -> TShape:
        min_size = config.min_shape_size or 0.1
        min_feature = config.min_feature_size or 0.0

        fixed = get_all_fixed_param(config, self.shape_class)
        ranges = get_all_param_range(config, self.shape_class)

        x0, y0, x1, y1 = canvas.inner_bounds
        iw, ih = x1 - x0, y1 - y0
        max_length = min(iw, ih)

        for _ in range(config.max_tries_per_shape):
            length = resolve_param(
                rng,
                fixed_value=fixed['length'],
                user_range=ranges['length'],
                default_range=intersect_ranges((min_size, max_length), (min_feature, max_length)),
            )

            # width < length; also width >= min_feature
            width = resolve_param(
                rng,
                fixed_value=fixed['width'],
                user_range=ranges['width'],
                default_range=intersect_ranges((min_feature, length), (min_size, length)),
            )

            if width >= length:
                continue

            angle = resolve_param(
                rng,
                fixed_value=fixed['angle'],
                user_range=ranges['angle'],
                default_range=(0.0, 360.0),
            )

            # outer_corner_radius must be < width / 2
            max_ocr = width / 2.0 - 1e-6
            if max_ocr <= 0:
                continue
            outer_corner_radius = resolve_param(
                rng,
                fixed_value=fixed['outer_corner_radius'],
                user_range=ranges['outer_corner_radius'],
                default_range=(0.0, max_ocr),
            )

            # inner_corner_radius must be <= length/2 - width/2 - outer_corner_radius
            max_icr = length / 2.0 - width / 2.0 - outer_corner_radius
            if max_icr < 0:
                continue
            inner_corner_radius = resolve_param(
                rng,
                fixed_value=fixed['inner_corner_radius'],
                user_range=ranges['inner_corner_radius'],
                default_range=(0.0, max_icr),
            )

            # bounding box is length x length, rotated
            half = length / 2.0
            theta = np.deg2rad(angle)
            dx = abs(half * np.cos(theta)) + abs(half * np.sin(theta))
            dy = abs(half * np.sin(theta)) + abs(half * np.cos(theta))

            if dx > iw / 2 or dy > ih / 2:
                continue

            cx, cy = sample_center_in_bounds(
                rng,
                fixed_center=fixed['center'],
                center_range=ranges['center'],
                x_bounds=(x0 + dx, x1 - dx),
                y_bounds=(y0 + dy, y1 - dy),
            )

            return TShape(
                center=(cx, cy),
                length=length,
                width=width,
                angle=angle,
                outer_corner_radius=outer_corner_radius,
                inner_corner_radius=inner_corner_radius,
            )

        raise RuntimeError("tshape_too_large_for_canvas")
