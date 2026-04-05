# metashapes/generators/validator.py
# This module defines the UnitCellValidator API.
# Validation mostly work through Shapely adapter. 

from __future__ import annotations

from abc import ABC, abstractmethod

from metashapes.unit_cell import UnitCell


class UnitCellValidator(ABC):
    @abstractmethod
    def validate(self, cell: UnitCell) -> str | None:
        """
        Return None if valid, otherwise a short failure reason string.
        """
        raise NotImplementedError


class DefaultUnitCellValidator(UnitCellValidator):
    def validate(self, cell: UnitCell) -> str | None:
        shape = cell.filled_region
        geom = shape.to_shapely()

        if geom.is_empty:
            return "empty"

        if geom.area <= 0:
            return "zero_area"

        return None