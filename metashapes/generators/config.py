# metashapes/generators/config.py
# This module defines the GeneratorConfig dataclass, which encapsulates configuration parameters for unit cells generation.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GeneratorConfig:
    # reproducibility
    seed: int | None = None

    # batch generation
    target_count: int = 1

    # retry policy
    max_tries_per_shape: int = 100
    max_tries_per_cell: int = 100

    # shapes per unit cell
    fixed_num_shapes: int | None = None
    min_num_shapes: int = 1
    max_num_shapes: int = 3

    # allowed primitive names,
    allowed_shapes: tuple[str, ...] = ()
    
    # shape sampling weights 
    shape_weights: dict[str, float] = field(default_factory=dict)
    
    # fixed parameters for specific shapes
    fixed_shape_params: dict[str, dict[str, object]] = field(default_factory=dict)
    
    # parameter ranges for specific shapes (overrides global ones)
    shape_param_ranges: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)

    # geometry constraints
    min_shape_size: float | None = None
    min_feature_size: float | None = None
    min_gap: float | None = None

    # optional free-form extras for future growth
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_count < 1:
            raise ValueError("target_count must be >= 1")

        if self.max_tries_per_cell < 1:
            raise ValueError("max_tries_per_cell must be >= 1")
        
        if self.max_tries_per_shape < 1:
            raise ValueError("max_tries_per_shape must be >= 1")

        if self.fixed_num_shapes is not None:
            if self.fixed_num_shapes < 0:
                raise ValueError("fixed_num_shapes must be >= 0")
        else:
            if self.min_num_shapes < 0:
                raise ValueError("min_num_shapes must be >= 0")
            if self.max_num_shapes < 0:
                raise ValueError("max_num_shapes must be >= 0")
            if self.min_num_shapes > self.max_num_shapes:
                raise ValueError("min_num_shapes must be <= max_num_shapes")

        if any(v < 0 for v in self.shape_weights.values()):
            raise ValueError("shape_weights values must be >= 0")
        
        if self.min_shape_size is not None and self.min_shape_size < 0:
            raise ValueError("min_shape_size must be >= 0")

        if self.min_feature_size is not None and self.min_feature_size < 0:
            raise ValueError("min_feature_size must be >= 0")

        if self.min_gap is not None and self.min_gap < 0:
            raise ValueError("min_gap must be >= 0")