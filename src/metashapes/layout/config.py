# metashapes/layout/config.py
# LayoutConfig: settings for LayoutGenerator.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["LayoutConfig"]


@dataclass(slots=True)
class LayoutConfig:
    """Settings for :class:`~metashapes.layout.LayoutGenerator`.

    Stopping (whichever comes first):
        ``target_num_shapes``, ``target_fill_fraction`` (sum of atom areas
        over domain area), exhaustion of exhaustible sites, or
        ``max_consecutive_failures`` rejected candidates in a row for
        random sites (the saturation signal).

    Constraints on every candidate:
        ``edge_margin`` clearance from the domain boundary (containment is
        always enforced), ``min_gap`` between atoms (``None``: atoms may
        overlap and merge; ``0.0``: touching allowed, overlap not),
        ``min_feature_size`` and ``min_shape_size`` (bbox side).

    Outputs are plain shapes (buffers only); see ``make_learnable``.
    """
    seed: int | None = None
    target_count: int = 1                       # layouts per batch

    # stopping
    target_num_shapes: int | None = None
    target_fill_fraction: float | None = None
    max_consecutive_failures: int = 1000
    max_tries_per_site: int = 10
    max_tries_per_layout: int = 3

    # constraints
    edge_margin: float = 0.0
    min_gap: float | None = None
    min_feature_size: float | None = None
    min_shape_size: float | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_count < 1:
            raise ValueError("target_count must be >= 1")
        if self.target_num_shapes is not None and self.target_num_shapes < 1:
            raise ValueError("target_num_shapes must be >= 1")
        if self.target_fill_fraction is not None and not 0 < self.target_fill_fraction <= 1:
            raise ValueError("target_fill_fraction must be in (0, 1]")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        if self.max_tries_per_site < 1:
            raise ValueError("max_tries_per_site must be >= 1")
        if self.max_tries_per_layout < 1:
            raise ValueError("max_tries_per_layout must be >= 1")
        if self.edge_margin < 0:
            raise ValueError("edge_margin must be >= 0")
        for name in ("min_gap", "min_feature_size", "min_shape_size"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be >= 0")
