# MetaShapes

**MetaShapes** is a Python library for working with 2D periodic geometries in the context of metasurface design.
It is built on a **Shape + Lattice = UnitCell** model: shapes are defined analytically via Signed Distance Functions (SDFs), making them differentiable and composable via boolean operations. A `UnitCell` pairs a `Lattice` (periodicity) with any `Shape` tree to produce a periodic structure.

The library supports simulation workflows, inverse design, geometry analysis, and fabrication-ready layout generation.

## Features

- **SDF-based primitives** backed by `nn.Module` — differentiable by default, ready for PyTorch-based inverse design
- **Boolean operations** (`|` Union, `&` Intersection, `-` Difference) and transforms (Translate, Rotate, Scale)
- **Rectangular and hexagonal lattices** via `Lattice.rectangular` / `Lattice.hexagonal`
- **Random dataset generator** with geometric constraints (min gap, min feature size, fill fraction)
- **Analysis**: fill fraction, min gap, min feature size via `UnitCellAnalyzer`
- **Serialization**: YAML save/load; Shapely export; GDS export (via `transform.shapely_to_gdstk`, full adapter coming soon)

## Installation

```bash
pip install git+https://github.com/RodionovSA/Metashapes
```

## Quick Start

```python
from metashapes import UnitCell, Lattice
from metashapes.shape import Rectangle

lattice = Lattice.rectangular(400, 400)
rect = Rectangle(center=(0, 0), size=(100, 60), angle=0.0)
cell = UnitCell(lattice=lattice, scene=rect)

mask = cell.mask(nx=128, ny=128)  # binary mask as torch.Tensor
sdf  = cell.rasterize(nx=128, ny=128)  # signed-distance grid
```

Available primitives: `Rectangle`, `ConvexQuad`, `IsoscelesTrapezoid`, `Ellipse`, `RegularPolygon`, `Cross`, `TShape`, `Bar`.

## Boolean Operations

Shapes compose with Python operators:

```python
from metashapes.shape import Ellipse, Rectangle

big  = Ellipse(center=(0, 0), axes=(150, 100))
hole = Rectangle(center=(0, 0), size=(60, 40))
ring = big - hole        # Difference
# also: big | hole  (Union),  big & hole  (Intersection)

cell = UnitCell(lattice=lattice, scene=ring)
```

## Random Generation

```python
from metashapes import Lattice
from metashapes.generators import RandomUnitCellGenerator, RandomGeneratorConfig

config = RandomGeneratorConfig(
    target_count=100,
    allowed_shapes=("Rectangle", "Ellipse", "Cross"),
    min_num_shapes=1,
    max_num_shapes=2,
    min_gap=10.0,
    min_feature_size=30.0,
    lattice_L_range=(300.0, 500.0),
    seed=42,
)

gen   = RandomUnitCellGenerator(config)
batch = gen.generate(Lattice.rectangular(400, 400))
cells = batch.unit_cells  # list[UnitCell]
```

## Analysis

```python
from metashapes import UnitCellAnalyzer

analyzer = UnitCellAnalyzer(min_gap=10.0, min_feature_size=20.0)
metrics  = analyzer.metrics(cell)
print(metrics.fill_fraction, metrics.min_gap, metrics.min_feature_size)

failures = analyzer.check(cell)   # list of violated constraint strings
```

## YAML Serialization

```python
from metashapes.adapters.yaml import save_unit_cells, load_unit_cells

save_unit_cells("cells.yaml", batch.unit_cells)
cells = load_unit_cells("cells.yaml")
```

## Requirements

- Python 3.9+
- NumPy >= 1.26
- PyTorch >= 2.0
- Shapely >= 2.0
- PyYAML >= 6.0
- gdstk *(optional, for GDS export)*

## License

MIT License. See [LICENSE](LICENSE) for details.
