# Graph Report - Metashapes  (2026-05-27)

## Corpus Check
- 71 files · ~34,630 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 847 nodes · 2722 edges · 37 communities (27 shown, 10 thin omitted)
- Extraction: 82% EXTRACTED · 18% INFERRED · 0% AMBIGUOUS · INFERRED: 486 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cd43175b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Shape Primitives Core|Shape Primitives Core]]
- [[_COMMUNITY_Random Generator & Lattice|Random Generator & Lattice]]
- [[_COMMUNITY_Shape Analysis & SDF Concepts|Shape Analysis & SDF Concepts]]
- [[_COMMUNITY_Unit Cell Analyzer|Unit Cell Analyzer]]
- [[_COMMUNITY_Conic Shape Primitives|Conic Shape Primitives]]
- [[_COMMUNITY_YAML Serialization Tests|YAML Serialization Tests]]
- [[_COMMUNITY_Shapely Transform Tests|Shapely Transform Tests]]
- [[_COMMUNITY_Random Generator Logic|Random Generator Logic]]
- [[_COMMUNITY_Shapely Compound Shape Tests|Shapely Compound Shape Tests]]
- [[_COMMUNITY_Generator Integration Tests|Generator Integration Tests]]
- [[_COMMUNITY_Generator Base Classes|Generator Base Classes]]
- [[_COMMUNITY_Periodic Stripe Shape|Periodic Stripe Shape]]
- [[_COMMUNITY_Shapely Adapter Modules|Shapely Adapter Modules]]
- [[_COMMUNITY_YAML & Unit Cell Serialization|YAML & Unit Cell Serialization]]
- [[_COMMUNITY_Unit Cell Tests|Unit Cell Tests]]
- [[_COMMUNITY_Shapely Adapter Layer|Shapely Adapter Layer]]
- [[_COMMUNITY_PyTorch Differentiability|PyTorch Differentiability]]
- [[_COMMUNITY_Lattice SDF Tests|Lattice SDF Tests]]
- [[_COMMUNITY_Coordinate Transform Bridge|Coordinate Transform Bridge]]
- [[_COMMUNITY_Boolean Shape Tests|Boolean Shape Tests]]
- [[_COMMUNITY_Periodic Unit Cell Ops|Periodic Unit Cell Ops]]
- [[_COMMUNITY_Mask Tests|Mask Tests]]
- [[_COMMUNITY_Generator Validation|Generator Validation]]
- [[_COMMUNITY_Lattice Basis Rationale|Lattice Basis Rationale]]
- [[_COMMUNITY_Lattice Basis Rationale B|Lattice Basis Rationale B]]
- [[_COMMUNITY_Lattice Basis Rationale C|Lattice Basis Rationale C]]
- [[_COMMUNITY_Validator Rationale|Validator Rationale]]
- [[_COMMUNITY_Generator Base Rationale|Generator Base Rationale]]
- [[_COMMUNITY_Analysis Test Suite|Analysis Test Suite]]
- [[_COMMUNITY_Primitives Init|Primitives Init]]
- [[_COMMUNITY_Sampler Utils|Sampler Utils]]

## God Nodes (most connected - your core abstractions)
1. `UnitCell` - 114 edges
2. `Rectangle` - 81 edges
3. `Lattice` - 80 edges
4. `shape_to_shapely()` - 77 edges
5. `Shape` - 64 edges
6. `UnitCellAnalyzer` - 64 edges
7. `Stripe` - 57 edges
8. `Ellipse` - 54 edges
9. `RegularPolygon` - 52 edges
10. `Union` - 50 edges

## Surprising Connections (you probably didn't know these)
- `SDF-based differentiable shapes concept` --conceptually_related_to--> `Ellipse`  [INFERRED]
  README.md → metashapes/shape/primitives/conics.py
- `TestRectangleSafeExtraction` --uses--> `UnitCell`  [INFERRED]
  tests/adapters/test_shapely_safe_extraction.py → metashapes/lattice/unit_cell.py
- `TestEllipseSafeExtraction` --uses--> `UnitCell`  [INFERRED]
  tests/adapters/test_shapely_safe_extraction.py → metashapes/lattice/unit_cell.py
- `TestStadiumSafeExtraction` --uses--> `UnitCell`  [INFERRED]
  tests/adapters/test_shapely_safe_extraction.py → metashapes/lattice/unit_cell.py
- `TestRegularPolygonSafeExtraction` --uses--> `UnitCell`  [INFERRED]
  tests/adapters/test_shapely_safe_extraction.py → metashapes/lattice/unit_cell.py

## Communities (37 total, 10 thin omitted)

### Community 0 - "Shape Primitives Core"
Cohesion: 0.08
Nodes (12): square_in_rect(), min_feature_size(), General triangle defined by two base angles and the base length (ASA).      Para, Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, (A, B, C) as (x, y) tensor pairs, CCW, centroid at origin., Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, RegularPolygon, Star (+4 more)

### Community 1 - "Random Generator & Lattice"
Cohesion: 0.20
Nodes (30): Constraint-Based Unit Cell Generation, RandomUnitCellGenerator._sample_shape, register_shape_sampler(), SHAPE_SAMPLER_REGISTRY, Constraint-based Parameter Sampling Pattern, EggSampler, EllipseSampler, StadiumSampler (+22 more)

### Community 2 - "Shape Analysis & SDF Concepts"
Cohesion: 0.06
Nodes (43): CellMetrics, _compute_min_gap, Rotation Guard for Infinite-Extent Shapes, Parametric Serialization (to_parametric / from_parametric), Smooth Boolean Operations via Polynomial Blending, _bbox_size(), Compute all metrics for a single cell., True if this periodic shift should be skipped for self-gap measurement.      A s (+35 more)

### Community 3 - "Unit Cell Analyzer"
Cohesion: 0.09
Nodes (14): _leaf_shapes, UnitCellAnalyzer, Return a list of constraint violation descriptions.         An empty list means, Generator-compatible interface.          Returns the first constraint violation, Compute metrics for every cell in a batch., Find groups of cells that have identical SDFs (within tolerance).          Each, Yield leaf (primitive) shapes from a composed shape tree.      Union nodes are d, Computes metrics for :class:`~metashapes.unit_cell.UnitCell` objects and     val (+6 more)

### Community 4 - "Conic Shape Primitives"
Cohesion: 0.09
Nodes (16): TestCompoundShapesToShapely, Egg, Ellipse, Ellipse.min_feature_size, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium (+8 more)

### Community 5 - "YAML Serialization Tests"
Cohesion: 0.22
Nodes (10): _make_batch_result(), TestSaveBatchResult, _check_version(), load_batch_result(), _load_cell(), Load a :class:`~metashapes.generators.report.GenerationBatchResult`     from a Y, Recursively convert tuples → lists so yaml.dump produces clean YAML., Save a :class:`~metashapes.generators.report.GenerationBatchResult`     (unit ce (+2 more)

### Community 6 - "Shapely Transform Tests"
Cohesion: 0.07
Nodes (30): SDF Convention: Negative Inside, Positive Outside, Cross, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic symmetric cross.      Parameters:         center: (cx, cy)         leng, TShape, An infinite stripe spanning the full unit cell along one axis.      The stripe i, Stripe, ConvexQuad (+22 more)

### Community 7 - "Random Generator Logic"
Cohesion: 0.13
Nodes (11): RandomUnitCellGenerator._generate_one, _has_infinite_bounds(), RandomGeneratorConfig, Return True if the shape has infinite spatial extent (e.g. Stripe)., First concrete generator config.      For now it only adds one flag:     - requi, Random unit-cell generator.      Current responsibilities:     1. choose number, _AlwaysFailValidator, TestLatticeVariants (+3 more)

### Community 8 - "Shapely Compound Shape Tests"
Cohesion: 0.07
Nodes (34): _centroid(), TestBooleansToShapely, TestPrimitivesToShapely, TestTransformsToShapely, Shapely Adapter Pattern, difference_to_shapely(), intersection_to_shapely(), union_to_shapely() (+26 more)

### Community 9 - "Generator Integration Tests"
Cohesion: 0.10
Nodes (15): ABC, UnitCellAnalyzer.validate, _gen(), _sdf_grid(), test_each_shape_type_generates(), TestBasicGeneration, TestConstraints, TestReport (+7 more)

### Community 10 - "Generator Base Classes"
Cohesion: 0.14
Nodes (13): _Dumper, YAML Dumper that puts 'type' first, then sorts remaining keys., _generate_one(), Build metadata dict attached to every GenerationReport., Summarise key parameter ranges across generated cells., Base API for periodic unit-cell generators.      Generation pipeline for each ca, Return a (possibly rescaled) lattice for a single cell.          Uniform scaling, UnitCellGenerator (+5 more)

### Community 11 - "Periodic Stripe Shape"
Cohesion: 0.08
Nodes (12): nn.Module Subclassing for Differentiability, Lattice, Cartesian translation for lattice cell (i, j)., In-plane periodicity of the unit cell. Fixed (non-optimizable).     Defined by t, Cartesian (x, y) -> fractional (f1, f2)., Fractional (f1, f2) -> Cartesian (x, y)., cartesian_grid(), fractional_grid() (+4 more)

### Community 12 - "Shapely Adapter Modules"
Cohesion: 0.12
Nodes (15): p(), pv(), center_scene(method='centroid') must work even if params have grad., Verifies that calling shape_to_shapely (which does .detach().cpu())         does, Shorthand: make a scalar nn.Parameter., Shorthand: make a vector nn.Parameter., TestCrossSafeExtraction, TestEllipseSafeExtraction (+7 more)

### Community 13 - "YAML & Unit Cell Serialization"
Cohesion: 0.09
Nodes (13): method='centroid' on a shape that Shapely can represent should work., TestInfiniteBoundsError, TestToShapely, TestUnitCellSerialization Test Class, Rasterize the periodic structure into a mask. Shape [ny·n2, nx·n1].          sof, Return world-coordinate points on the material boundary (zero-level-set)., A periodic structure: one Lattice + one Shape (the scene).      The lattice owns, Axis-aligned Cartesian bounding box of the supercell.          Returns ``(xmin, (+5 more)

### Community 14 - "Unit Cell Tests"
Cohesion: 0.12
Nodes (6): Rectangular unit cell with a small square at the origin., _square_cell(), TestExtent, TestUnitCellBoundary, TestUnitCellMask, TestUnitCellRasterize

### Community 15 - "Shapely Adapter Layer"
Cohesion: 0.10
Nodes (16): _cell_center(), Rectangular cell with a Rectangle shape centred at (cx, cy)., For a symmetric Rectangle, both methods produce the same offset., For a centred Ellipse, both methods produce the same offset., The geometric midpoint (a1 + a2) / 2 as plain floats., Extract the outermost Translate dx/dy buffers from the scene., _rect_cell(), TestAlreadyCentered (+8 more)

### Community 16 - "PyTorch Differentiability"
Cohesion: 0.18
Nodes (6): make_learnable_polygon(), Return (UnitCell, side_length param, center param) with nn.Parameters., Gradient flows from a point displaced by one lattice vector., At least some pixels must have non-trivial gradient contribution., TestUnitCellGradients, TestUnitCellGradients Test Class

### Community 18 - "Coordinate Transform Bridge"
Cohesion: 0.20
Nodes (9): Convert a shapely geometry to a gdstk polygon or a list of polygons., Convert a gdstk polygon to a shapely geometry., Convert a shapely geometry to a KLayout polygon or a list of polygons., Create a shapely geometry from a binary numpy array.     Parameters:         img, gdstk_to_shapely, numpy_to_shapely, shapely_to_gdstk, shapely_to_klayout (+1 more)

### Community 19 - "Boolean Shape Tests"
Cohesion: 0.10
Nodes (4): TestLatticeCoordinates Test Class, TestLatticeOffset, TestLatticeProperties Test Class, TestNeighborOffsets

### Community 21 - "Mask Tests"
Cohesion: 0.28
Nodes (8): _rect_cell(), _sdf_grid(), TestSaveLoadUnitCells, load_unit_cells(), Save one or more unit cells to a YAML file.      Parameters     ----------     p, Load unit cells from a YAML file saved by :func:`save_unit_cells`     or :func:`, save_unit_cells(), Test YAML Adapter

### Community 22 - "Generator Validation"
Cohesion: 0.40
Nodes (3): nn.Parameter centre of a shape must receive gradients after center_scene()., make_learnable_polygon center param grad is non-None after centering., TestGradientFlow

## Knowledge Gaps
- **10 isolated node(s):** `TestAnalyze Test Class`, `TestUnitCellGradients Test Class`, `UnitCellAnalyzer.find_duplicates`, `Constraint-Based Unit Cell Generation`, `Ellipse.min_feature_size` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UnitCell` connect `YAML & Unit Cell Serialization` to `Shape Primitives Core`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `YAML Serialization Tests`, `Random Generator Logic`, `Generator Integration Tests`, `Generator Base Classes`, `Periodic Stripe Shape`, `Shapely Adapter Modules`, `Unit Cell Tests`, `Shapely Adapter Layer`, `PyTorch Differentiability`, `Lattice SDF Tests`, `Mask Tests`, `Generator Validation`?**
  _High betweenness centrality (0.256) - this node is a cross-community bridge._
- **Why does `Lattice` connect `Periodic Stripe Shape` to `Random Generator & Lattice`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `YAML Serialization Tests`, `Shapely Transform Tests`, `Random Generator Logic`, `Generator Integration Tests`, `Generator Base Classes`, `YAML & Unit Cell Serialization`, `Unit Cell Tests`, `PyTorch Differentiability`, `Lattice SDF Tests`, `Boolean Shape Tests`, `Mask Tests`?**
  _High betweenness centrality (0.172) - this node is a cross-community bridge._
- **Why does `Shape` connect `Shape Analysis & SDF Concepts` to `Shape Primitives Core`, `Random Generator & Lattice`, `Unit Cell Analyzer`, `Conic Shape Primitives`, `Shapely Transform Tests`, `Random Generator Logic`, `Shapely Compound Shape Tests`, `Generator Base Classes`, `Periodic Stripe Shape`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Are the 57 inferred relationships involving `UnitCell` (e.g. with `TestRectangleSafeExtraction` and `TestEllipseSafeExtraction`) actually correct?**
  _`UnitCell` has 57 INFERRED edges - model-reasoned connections that need verification._
- **Are the 48 inferred relationships involving `Rectangle` (e.g. with `TestLeafShapes` and `TestCellMetrics`) actually correct?**
  _`Rectangle` has 48 INFERRED edges - model-reasoned connections that need verification._
- **Are the 42 inferred relationships involving `Lattice` (e.g. with `TestLeafShapes` and `TestCellMetrics`) actually correct?**
  _`Lattice` has 42 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `Shape` (e.g. with `TestPrimitivesToShapely` and `TestBooleansToShapely`) actually correct?**
  _`Shape` has 17 INFERRED edges - model-reasoned connections that need verification._