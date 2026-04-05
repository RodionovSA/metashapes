# metashapes/generators/samplers/base.py
# This module defines the base class for shape samplers, which are responsible for generating random shapes. 

from __future__ import annotations

from abc import ABC, abstractmethod

from metashapes.canvas import Canvas
from metashapes.shape.base import Shape


class ShapeSampler(ABC):
    shape_name: str

    @abstractmethod
    def sample(self, rng, canvas: Canvas, config) -> Shape:
        raise NotImplementedError