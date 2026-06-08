# Graph Report - metashapes  (2026-06-08)

## Corpus Check
- 71 files · ~34,630 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1020 nodes · 3734 edges · 50 communities (37 shown, 13 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 616 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bd15b64f`
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
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Lattice Basis Rationale|Lattice Basis Rationale]]
- [[_COMMUNITY_Lattice Basis Rationale B|Lattice Basis Rationale B]]
- [[_COMMUNITY_Lattice Basis Rationale C|Lattice Basis Rationale C]]
- [[_COMMUNITY_Validator Rationale|Validator Rationale]]
- [[_COMMUNITY_Generator Base Rationale|Generator Base Rationale]]
- [[_COMMUNITY_Analysis Test Suite|Analysis Test Suite]]
- [[_COMMUNITY_Primitives Init|Primitives Init]]
- [[_COMMUNITY_Sampler Utils|Sampler Utils]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]

## God Nodes (most connected - your core abstractions)
1. `UnitCell` - 127 edges
2. `Lattice` - 125 edges
3. `shape_to_shapely()` - 106 edges
4. `Rectangle` - 95 edges
5. `Shape` - 84 edges
6. `Ellipse` - 73 edges
7. `assert_inside()` - 73 edges
8. `Translate` - 69 edges
9. `Stripe` - 67 edges
10. `RegularPolygon` - 65 edges

## Surprising Connections (you probably didn't know these)
- `UnitCellAnalyzer.validate` --semantically_similar_to--> `UnitCellValidator`  [INFERRED] [semantically similar]
  metashapes/analysis.py → src/metashapes/generators/validator.py
- `SDF-based differentiable shapes concept` --conceptually_related_to--> `Ellipse`  [INFERRED]
  README.md → src/metashapes/shape/primitives/conics.py
- `ConvexQuad` --semantically_similar_to--> `IsoscelesTrapezoid`  [INFERRED] [semantically similar]
  src/metashapes/shape/primitives/quads.py → metashapes/shape/primitives/quads.py
- `shape_to_shapely()` --implements--> `Shapely Adapter Pattern`  [INFERRED]
  src/metashapes/adapters/shapely/dispatch.py → metashapes/adapters/shapely/dispatch.py
- `TestLeafShapes` --uses--> `CellMetrics`  [INFERRED]
  tests/test_analysis.py → src/metashapes/analysis.py

## Communities (50 total, 13 thin omitted)

### Community 0 - "Shape Primitives Core"
Cohesion: 0.12
Nodes (4): General triangle defined by two base angles and the base length (ASA).      Para, (A, B, C) as (x, y) tensor pairs, CCW, centroid at origin., Triangle, TestTriangle

### Community 1 - "Random Generator & Lattice"
Cohesion: 0.14
Nodes (36): register_shape_sampler(), Lattice, Cartesian translation for lattice cell (i, j)., In-plane periodicity of the unit cell. Fixed (non-optimizable).     Defined by t, Cartesian (x, y) -> fractional (f1, f2)., Fractional (f1, f2) -> Cartesian (x, y)., Constraint-based Parameter Sampling Pattern, sample() (+28 more)

### Community 2 - "Shape Analysis & SDF Concepts"
Cohesion: 0.05
Nodes (59): TestCrossSafeExtraction, TestEllipseSafeExtraction, TestIsoscelesTrapezoidSafeExtraction, TestRegularPolygonSafeExtraction, TestStadiumSafeExtraction, TestStripeSafeExtraction, TestUnitCellSafeExtraction, CellMetrics (+51 more)

### Community 3 - "Unit Cell Analyzer"
Cohesion: 0.11
Nodes (17): _leaf_shapes, UnitCellAnalyzer, UnitCellAnalyzer.validate, _leaf_shapes(), Return a list of constraint violation descriptions.         An empty list means, Generator-compatible interface.          Returns the first constraint violation, Compute metrics for every cell in a batch., Find groups of cells that have identical SDFs (within tolerance).          Each (+9 more)

### Community 4 - "Conic Shape Primitives"
Cohesion: 0.06
Nodes (17): Egg, Ellipse, Ellipse.min_feature_size, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium (+9 more)

### Community 5 - "YAML Serialization Tests"
Cohesion: 0.25
Nodes (8): _make_batch_result(), TestSaveBatchResult, load_batch_result(), Load a :class:`~metashapes.generators.report.GenerationBatchResult`     from a Y, Save a :class:`~metashapes.generators.report.GenerationBatchResult`     (unit ce, save_batch_result(), _sdf_grid(), TestYAMLSaving

### Community 6 - "Shapely Transform Tests"
Cohesion: 0.13
Nodes (8): Cross, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic symmetric cross.      Parameters:         center: (cx, cy)         leng, TShape, assert_inside(), All points should have SDF < tol (inside or on boundary)., TestCross, TestTShape

### Community 7 - "Random Generator Logic"
Cohesion: 0.18
Nodes (10): RandomUnitCellGenerator._generate_one, RandomGeneratorConfig, RandomUnitCellGenerator, First concrete generator config.      For now it only adds one flag:     - requi, Random unit-cell generator.      Current responsibilities:     1. choose number, _AlwaysFailValidator, TestLatticeVariants, TestMergeBatchResults (+2 more)

### Community 8 - "Shapely Compound Shape Tests"
Cohesion: 0.14
Nodes (28): difference_to_shapely(), intersection_to_shapely(), union_to_shapely(), egg_to_shapely(), ellipse_to_shapely(), stadium_to_shapely(), as_float(), as_list() (+20 more)

### Community 9 - "Generator Integration Tests"
Cohesion: 0.11
Nodes (11): ABC, _gen(), test_each_shape_type_generates(), TestBasicGeneration, TestConstraints, TestReport, TestShapeCount, TestShapeTypes (+3 more)

### Community 10 - "Generator Base Classes"
Cohesion: 0.14
Nodes (13): _generate_one(), Build metadata dict attached to every GenerationReport., Summarise key parameter ranges across generated cells., Base API for periodic unit-cell generators.      Generation pipeline for each ca, Return a (possibly rescaled) lattice for a single cell.          Uniform scaling, UnitCellGenerator, GeneratorConfig, _has_infinite_bounds() (+5 more)

### Community 11 - "Periodic Stripe Shape"
Cohesion: 0.21
Nodes (11): cell_area(), device(), dtype(), hexagonal(), matrix(), rectangular(), cartesian_grid(), fractional_grid() (+3 more)

### Community 12 - "Shapely Adapter Modules"
Cohesion: 0.13
Nodes (8): p(), pv(), center_scene(method='centroid') must work even if params have grad., Verifies that calling shape_to_shapely (which does .detach().cpu())         does, Shorthand: make a scalar nn.Parameter., Shorthand: make a vector nn.Parameter., TestRectangleSafeExtraction, TestTransformSafeExtraction

### Community 13 - "YAML & Unit Cell Serialization"
Cohesion: 0.15
Nodes (9): TestToShapely, Rasterize the periodic structure into a mask. Shape [ny·n2, nx·n1].          sof, Return world-coordinate points on the material boundary (zero-level-set)., A periodic structure: one Lattice + one Shape (the scene).      The lattice owns, Axis-aligned Cartesian bounding box of the supercell.          Returns ``(xmin,, Number of periodic copies to search per lattice direction.          A finite sha, Periodic signed distance of the scene at Cartesian (x, y).          Minimum over, Periodic SDF sampled over a supercell.          repeat=(n1, n2) — tile n1 cells (+1 more)

### Community 14 - "Unit Cell Tests"
Cohesion: 0.14
Nodes (5): Rectangular unit cell with a small square at the origin., _square_cell(), TestExtent, TestUnitCellMask, TestUnitCellRasterize

### Community 15 - "Shapely Adapter Layer"
Cohesion: 0.08
Nodes (19): _cell_center(), nn.Parameter centre of a shape must receive gradients after center_scene()., make_learnable_polygon center param grad is non-None after centering., Rectangular cell with a Rectangle shape centred at (cx, cy)., For a symmetric Rectangle, both methods produce the same offset., For a centred Ellipse, both methods produce the same offset., The geometric midpoint (a1 + a2) / 2 as plain floats., Extract the outermost Translate dx/dy buffers from the scene. (+11 more)

### Community 16 - "PyTorch Differentiability"
Cohesion: 0.24
Nodes (6): nn.Module Subclassing for Differentiability, make_learnable_polygon(), Return (UnitCell, side_length param, center param) with nn.Parameters., Gradient flows from a point displaced by one lattice vector., At least some pixels must have non-trivial gradient contribution., TestUnitCellGradients

### Community 17 - "Lattice SDF Tests"
Cohesion: 0.20
Nodes (3): square_in_rect(), _sdf_at(), TestUnitCellSDF

### Community 18 - "Coordinate Transform Bridge"
Cohesion: 0.16
Nodes (15): gdstk_to_shapely(), klayout_to_shapely(), numpy_to_shapely(), Convert a shapely geometry to a gdstk polygon or a list of polygons., Convert a gdstk polygon to a shapely geometry., Convert a shapely geometry to a KLayout polygon or a list of polygons., Create a shapely geometry from a binary numpy array.     Parameters:         img, shapely_to_gdstk() (+7 more)

### Community 19 - "Boolean Shape Tests"
Cohesion: 0.06
Nodes (5): TestLatticeConstruction, TestLatticeCoordinates, TestLatticeOffset, TestLatticeProperties, TestNeighborOffsets

### Community 21 - "Mask Tests"
Cohesion: 0.18
Nodes (13): _rect_cell(), _sdf_grid(), TestSaveLoadUnitCells, _check_version(), _Dumper, _load_cell(), load_unit_cells(), Recursively convert tuples → lists so yaml.dump produces clean YAML. (+5 more)

### Community 22 - "Generator Validation"
Cohesion: 0.12
Nodes (5): _centroid(), TestPrimitivesToShapely, Return a new UnitCell with the scene translated to the cell centre.          The, Shapely Adapter Pattern, shape_to_shapely()

### Community 23 - "Community 23"
Cohesion: 0.13
Nodes (3): An infinite stripe spanning the full unit cell along one axis.      The stripe i, Stripe, TestStripe

### Community 24 - "Community 24"
Cohesion: 0.14
Nodes (4): Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, RegularPolygon, TestRegularPolygon

### Community 25 - "Community 25"
Cohesion: 0.16
Nodes (6): TestTransformsToShapely, IsoscelesTrapezoid, min_feature_size(), Symbolic rectangle.      Parameters:         center: (cx, cy)         size: (wid, Rectangle, TestRectangle

### Community 37 - "Analysis Test Suite"
Cohesion: 0.19
Nodes (3): Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, Star, TestStar

### Community 40 - "Community 40"
Cohesion: 0.17
Nodes (3): ConvexQuad, Symbolic convex quadrilateral with optional rounded corners.      The quad is bu, TestConvexQuad

### Community 41 - "Community 41"
Cohesion: 0.35
Nodes (8): SDF Convention: Negative Inside, Positive Outside, assert_bounds_contain(), assert_outside(), assert_round_trip(), All points should have SDF > -tol (outside or on boundary)., Serialise → deserialise and verify SDF equality on a small grid., All points should lie inside (or on) the reported bounding box., sdf_at()

### Community 42 - "Community 42"
Cohesion: 0.27
Nodes (3): IsoscelesTrapezoid, Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), TestIsoscelesTrapezoid

### Community 48 - "Community 48"
Cohesion: 0.67
Nodes (3): Constraint-Based Unit Cell Generation, RandomUnitCellGenerator._sample_shape, SHAPE_SAMPLER_REGISTRY

## Knowledge Gaps
- **8 isolated node(s):** `UnitCellAnalyzer.find_duplicates`, `Constraint-Based Unit Cell Generation`, `Ellipse.min_feature_size`, `Primitives __init__ Module`, `Sampler Utility Functions` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Lattice` connect `Random Generator & Lattice` to `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `YAML Serialization Tests`, `Random Generator Logic`, `Generator Integration Tests`, `Generator Base Classes`, `Periodic Stripe Shape`, `Shapely Adapter Modules`, `YAML & Unit Cell Serialization`, `Unit Cell Tests`, `Shapely Adapter Layer`, `PyTorch Differentiability`, `Lattice SDF Tests`, `Boolean Shape Tests`, `Mask Tests`, `Community 25`, `Community 45`, `Community 46`, `Community 47`?**
  _High betweenness centrality (0.188) - this node is a cross-community bridge._
- **Why does `Shape` connect `Shape Analysis & SDF Concepts` to `Shape Primitives Core`, `Random Generator & Lattice`, `Unit Cell Analyzer`, `Conic Shape Primitives`, `Shapely Transform Tests`, `Random Generator Logic`, `Shapely Compound Shape Tests`, `Generator Base Classes`, `Periodic Stripe Shape`, `PyTorch Differentiability`, `Generator Validation`, `Community 23`, `Community 24`, `Community 25`, `Analysis Test Suite`, `Community 40`, `Community 41`, `Community 42`, `Community 43`, `Community 44`?**
  _High betweenness centrality (0.159) - this node is a cross-community bridge._
- **Why does `UnitCell` connect `YAML & Unit Cell Serialization` to `Random Generator & Lattice`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `YAML Serialization Tests`, `Random Generator Logic`, `Generator Integration Tests`, `Generator Base Classes`, `Periodic Stripe Shape`, `Shapely Adapter Modules`, `Community 45`, `Unit Cell Tests`, `Shapely Adapter Layer`, `PyTorch Differentiability`, `Lattice SDF Tests`, `Community 47`, `Community 46`, `Mask Tests`, `Generator Validation`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Are the 60 inferred relationships involving `UnitCell` (e.g. with `_Dumper` and `CellMetrics`) actually correct?**
  _`UnitCell` has 60 INFERRED edges - model-reasoned connections that need verification._
- **Are the 73 inferred relationships involving `Lattice` (e.g. with `CellMetrics` and `UnitCellAnalyzer`) actually correct?**
  _`Lattice` has 73 INFERRED edges - model-reasoned connections that need verification._
- **Are the 47 inferred relationships involving `Rectangle` (e.g. with `Shape` and `.sample()`) actually correct?**
  _`Rectangle` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `Shape` (e.g. with `Ellipse` and `Cross`) actually correct?**
  _`Shape` has 17 INFERRED edges - model-reasoned connections that need verification._