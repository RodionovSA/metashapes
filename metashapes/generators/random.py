# metashapes/generators/random.py
# This module implements a simple random unit-cell generator.

from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np
from shapely.affinity import translate

from metashapes.generators.base import UnitCellGenerator
from metashapes.generators.config import GeneratorConfig
from metashapes.generators.validator import UnitCellValidator
from metashapes.generators.registry import SHAPE_SAMPLER_REGISTRY
import metashapes.generators.samplers  


from metashapes.lattice.unit_cell import UnitCell
from metashapes.lattice.canvas import Canvas
from metashapes.shape.primitives import Rectangle
from metashapes import Shape
from metashapes.adapters.shapely import remove_holes
from metashapes.utils import periodic_shift_skipped


@dataclass(slots=True)
class RandomGeneratorConfig(GeneratorConfig):
    """
    First concrete generator config.

    For now it only adds one flag:
    - require_nonempty_after_clip:
        reject cells that become empty after clipping to the unit-cell box
    """
    require_nonempty_after_clip: bool = True


class RandomUnitCellGenerator(UnitCellGenerator):
    """
    Random unit-cell generator.

    Current responsibilities:
    1. choose number of shapes
    2. sample shape types from allowed_shapes
    3. create raw shapes
    4. build UnitCell
    5. validate
    """

    config: RandomGeneratorConfig

    def __init__(self, config: RandomGeneratorConfig, validator: UnitCellValidator | None = None) -> None:
            super().__init__(config, validator=validator)
            self.config = config

    def _generate_one(self, canvas: "Canvas") -> UnitCell:
        n_shapes = self._sample_num_shapes()

        accepted_shapes = []

        for _ in range(n_shapes):
            shape = self._sample_shape_with_constraints(accepted_shapes, canvas)
            accepted_shapes.append(shape)

        combined_shape = self._combine_shapes(accepted_shapes)

        # Aggregate periodic shifts from all accepted shapes so the UnitCell
        # knows which directions should not be clipped at the boundary.
        periodic_shifts: set[tuple[int, int]] = set()
        for s in accepted_shapes:
            periodic_shifts |= s.allowed_self_periodic_shifts

        cell = UnitCell(combined_shape, canvas, periodic_shifts=periodic_shifts)
        return cell
    
    def _sample_shape_with_constraints(self, accepted_shapes: list, canvas: "Canvas"):
        for _ in range(self.config.max_tries_per_shape):
            shape_name = self._sample_shape_name()
            candidate = self._sample_shape(shape_name, canvas)

            if self._is_shape_valid(candidate, accepted_shapes, canvas):
                return candidate
            
        raise RuntimeError("shape_placement_failed")
            
    def _is_shape_valid(self, candidate, accepted_shapes: list, canvas: "Canvas") -> bool:
        candidate_geom = remove_holes(candidate.to_shapely())
        
        # Check that shape is larger than min_shape_size 
        min_shape_size = self.config.min_shape_size
        if min_shape_size is not None:
            minx, miny, maxx, maxy = candidate_geom.bounds
            sx = maxx - minx
            sy = maxy - miny
            if sx < min_shape_size or sy < min_shape_size:
                return False
            
        # Check that shape does not have features smaller than min_feature_size
        min_feature_size = self.config.min_feature_size
        if min_feature_size is not None:
            fs = candidate.min_feature_size
            if fs is not None and fs < min_feature_size:
                return False
        
        # Check minimum gap to already accepted shapes
        min_gap = self.config.min_gap or 0.0
        shifts = [(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1)]

        # candidate vs its own periodic images
        allowed = candidate.allowed_self_periodic_shifts
        for ix, iy in shifts:
            if periodic_shift_skipped(ix, iy, allowed):
                continue

            shifted = translate(candidate_geom, xoff=ix * canvas.Lx, yoff=iy * canvas.Ly)
            if candidate_geom.distance(shifted) <= min_gap:
                return False

        # candidate vs already accepted shapes and their periodic images
        for shape in accepted_shapes:
            geom = remove_holes(shape.to_shapely())
            for ix, iy in shifts + [(0, 0)]:
                shifted = translate(geom, xoff=ix * canvas.Lx, yoff=iy * canvas.Ly)
                if candidate_geom.distance(shifted) <= min_gap:
                    return False

        return True

    def _sample_num_shapes(self) -> int:
        if self.config.fixed_num_shapes is not None:
            return self.config.fixed_num_shapes

        return self._rng.randint(
            self.config.min_num_shapes,
            self.config.max_num_shapes,
        )

    def _sample_shape_name(self) -> str:
        allowed = list(self.config.allowed_shapes) or list(SHAPE_SAMPLER_REGISTRY.keys())

        weights_cfg = self.config.shape_weights

        if not weights_cfg:
            return self._rng.choice(allowed)

        names = allowed
        weights = [weights_cfg.get(name, 1.0) for name in names]

        if all(w == 0 for w in weights):
            raise ValueError("At least one shape weight must be > 0")

        return self._rng.choices(names, weights=weights, k=1)[0]

    def _sample_shape(self, shape_name: str, canvas: Canvas):
        try:
            sampler = SHAPE_SAMPLER_REGISTRY[shape_name]
        except KeyError as e:
            raise NotImplementedError(
                f"No sampler registered for shape '{shape_name}'"
            ) from e

        return sampler.sample(self._rng, canvas, self.config)
            
    def _combine_shapes(self, shapes: list):
        if not shapes:
            raise RuntimeError("No shapes to combine")

        shape = shapes[0]
        for other in shapes[1:]:
            shape = shape.union(other)
        return shape
