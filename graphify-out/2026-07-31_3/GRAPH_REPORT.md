# Graph Report - .  (2026-07-31)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1065 nodes · 3861 edges · 52 communities (38 shown, 14 thin omitted)
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 799 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `af9293fa`
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
- [[_COMMUNITY_Community 11|Community 11]]
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
- [[_COMMUNITY_Test Package Root|Test Package Root]]
- [[_COMMUNITY_Shape Test Init|Shape Test Init]]
- [[_COMMUNITY_Lattice Test Init|Lattice Test Init]]
- [[_COMMUNITY_Adapters Test Init|Adapters Test Init]]
- [[_COMMUNITY_Generators Test Init|Generators Test Init]]
- [[_COMMUNITY_Lattice Basis Rationale|Lattice Basis Rationale]]
- [[_COMMUNITY_Lattice Basis Rationale B|Lattice Basis Rationale B]]
- [[_COMMUNITY_Lattice Basis Rationale C|Lattice Basis Rationale C]]
- [[_COMMUNITY_Lattice Package Init|Lattice Package Init]]
- [[_COMMUNITY_Validator Rationale|Validator Rationale]]
- [[_COMMUNITY_Analysis Test Suite|Analysis Test Suite]]
- [[_COMMUNITY_Primitives Init|Primitives Init]]
- [[_COMMUNITY_Sampler Utils|Sampler Utils]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]

## God Nodes (most connected - your core abstractions)
1. `UnitCell` - 131 edges
2. `Lattice` - 124 edges
3. `Rectangle` - 115 edges
4. `shape_to_shapely()` - 110 edges
5. `Ellipse` - 92 edges
6. `Shape` - 85 edges
7. `RegularPolygon` - 85 edges
8. `Bar` - 80 edges
9. `assert_inside()` - 71 edges
10. `Translate` - 70 edges

## Surprising Connections (you probably didn't know these)
- `UnitCellAnalyzer.validate` --semantically_similar_to--> `UnitCellValidator`  [INFERRED] [semantically similar]
  metashapes/analysis.py → src/metashapes/generators/validator.py
- `ConvexQuad` --semantically_similar_to--> `IsoscelesTrapezoid`  [INFERRED] [semantically similar]
  src/metashapes/shape/primitives/quads.py → metashapes/shape/primitives/quads.py
- `SDF-based differentiable shapes concept` --conceptually_related_to--> `Ellipse`  [INFERRED]
  README.md → src/metashapes/shape/primitives/conics.py
- `TestCellMetrics` --uses--> `Union`  [INFERRED]
  tests/test_analysis.py → src/metashapes/shape/boolean.py
- `TestMetrics` --uses--> `Union`  [INFERRED]
  tests/test_analysis.py → src/metashapes/shape/boolean.py

## Communities (52 total, 14 thin omitted)

### Community 0 - "Shape Primitives Core"
Cohesion: 0.06
Nodes (45): TestBarSafeExtraction, TestCrossSafeExtraction, TestIsoscelesTrapezoidSafeExtraction, TestRegularPolygonSafeExtraction, TestStadiumSafeExtraction, TestStripeSafeExtraction, TestUnitCellSafeExtraction, TestTransformsToShapely (+37 more)

### Community 1 - "Random Generator & Lattice"
Cohesion: 0.06
Nodes (34): _centroid(), TestBooleansToShapely, TestPrimitivesToShapely, Return a new UnitCell with the scene translated to the cell centre.          The, Shapely Adapter Pattern, difference_to_shapely(), intersection_to_shapely(), union_to_shapely() (+26 more)

### Community 2 - "Shape Analysis & SDF Concepts"
Cohesion: 0.08
Nodes (33): CellMetrics, _compute_min_gap, _leaf_shapes, UnitCellAnalyzer, UnitCellAnalyzer.validate, Rotation Guard for Infinite-Extent Shapes, CellMetrics, _compute_min_gap() (+25 more)

### Community 3 - "Unit Cell Analyzer"
Cohesion: 0.13
Nodes (39): Constraint-Based Unit Cell Generation, RandomUnitCellGenerator._sample_shape, register_shape_sampler(), SHAPE_SAMPLER_REGISTRY, Lattice, Cartesian translation for lattice cell (i, j)., In-plane periodicity of the unit cell. Fixed (non-optimizable).     Defined by t, Cartesian (x, y) -> fractional (f1, f2). (+31 more)

### Community 4 - "Conic Shape Primitives"
Cohesion: 0.07
Nodes (15): Egg, Ellipse, Ellipse.min_feature_size, Egg shape: two half-ellipses joined at the x-axis.      Parameters:         cent, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Symbolic ellipse.      Parameters:         center: (cx, cy)         axes: full s, Stadium (discorectangle/capsule): a rectangle with semicircular caps.      Param, Stadium (+7 more)

### Community 5 - "YAML Serialization Tests"
Cohesion: 0.06
Nodes (16): nn.Module Subclassing for Differentiability, make_learnable_polygon(), Return (UnitCell, side_length param, center param) with nn.Parameters., square_in_rect(), Rectangular unit cell with a small square at the origin., Gradient flows from a point displaced by one lattice vector., At least some pixels must have non-trivial gradient contribution., _sdf_at() (+8 more)

### Community 6 - "Shapely Transform Tests"
Cohesion: 0.12
Nodes (10): Cross, Symbolic T-shape.      Parameters:         center: (cx, cy)         length: full, Symbolic symmetric cross.      Parameters:         center: (cx, cy)         leng, TShape, Shape, assert_inside(), assert_round_trip(), Serialise → deserialise and verify SDF equality on a small grid. (+2 more)

### Community 7 - "Random Generator Logic"
Cohesion: 0.11
Nodes (12): ABC, _gen(), _sdf_grid(), test_each_shape_type_generates(), TestBasicGeneration, TestConstraints, TestReport, TestShapeCount (+4 more)

### Community 8 - "Shapely Compound Shape Tests"
Cohesion: 0.17
Nodes (11): GeneratorConfig, RandomUnitCellGenerator._generate_one, RandomGeneratorConfig, RandomUnitCellGenerator, First concrete generator config.      For now it only adds one flag:     - requi, Random unit-cell generator.      Current responsibilities:     1. choose number, _AlwaysFailValidator, TestLatticeVariants (+3 more)

### Community 9 - "Generator Integration Tests"
Cohesion: 0.06
Nodes (5): TestLatticeConstruction, TestLatticeCoordinates, TestLatticeOffset, TestLatticeProperties, TestNeighborOffsets

### Community 10 - "Generator Base Classes"
Cohesion: 0.12
Nodes (4): General triangle defined by two base angles and the base length (ASA).      Para, (A, B, C) as (x, y) tensor pairs, CCW, centroid at origin., Triangle, TestTriangle

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (13): _generate_one(), Build metadata dict attached to every GenerationReport., Summarise key parameter ranges across generated cells., Base API for periodic unit-cell generators.      Generation pipeline for each ca, Return a (possibly rescaled) lattice for a single cell.          Uniform scaling, UnitCellGenerator, GeneratorConfig, _has_infinite_bounds() (+5 more)

### Community 12 - "Shapely Adapter Modules"
Cohesion: 0.12
Nodes (3): Bar, An infinite bar spanning the full unit cell along one axis.      The bar is unbo, TestBar

### Community 13 - "YAML & Unit Cell Serialization"
Cohesion: 0.18
Nodes (13): _rect_cell(), _sdf_grid(), TestSaveLoadUnitCells, _check_version(), _Dumper, _load_cell(), load_unit_cells(), Recursively convert tuples → lists so yaml.dump produces clean YAML. (+5 more)

### Community 14 - "Unit Cell Tests"
Cohesion: 0.14
Nodes (4): Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, Symbolic regular polygon.      Parameters:         center: (cx, cy)         n: N, RegularPolygon, TestRegularPolygon

### Community 15 - "Shapely Adapter Layer"
Cohesion: 0.14
Nodes (8): p(), pv(), center_scene(method='centroid') must work even if params have grad., Shorthand: make a scalar nn.Parameter., Shorthand: make a vector nn.Parameter., TestEllipseSafeExtraction, TestRectangleSafeExtraction, TestTransformSafeExtraction

### Community 16 - "PyTorch Differentiability"
Cohesion: 0.15
Nodes (6): TestCompoundShapesToShapely, IsoscelesTrapezoid, Symbolic rectangle.      Parameters:         center: (cx, cy)         size: (wid, Symbolic isosceles trapezoid.      Parameters:         center: (cx, cy), Rectangle, TestRectangle

### Community 17 - "Lattice SDF Tests"
Cohesion: 0.17
Nodes (3): ConvexQuad, Symbolic convex quadrilateral with optional rounded corners.      The quad is bu, TestConvexQuad

### Community 18 - "Coordinate Transform Bridge"
Cohesion: 0.19
Nodes (3): Regular n-pointed star.      Parameters:         center: (cx, cy)         n: num, Star, TestStar

### Community 19 - "Boolean Shape Tests"
Cohesion: 0.26
Nodes (7): _make_batch_result(), TestSaveBatchResult, load_batch_result(), Load a :class:`~metashapes.generators.report.GenerationBatchResult`     from a Y, Save a :class:`~metashapes.generators.report.GenerationBatchResult`     (unit ce, save_batch_result(), TestYAMLSaving

### Community 20 - "Periodic Unit Cell Ops"
Cohesion: 0.21
Nodes (11): cell_area(), device(), dtype(), hexagonal(), matrix(), rectangular(), cartesian_grid(), fractional_grid() (+3 more)

### Community 21 - "Mask Tests"
Cohesion: 0.12
Nodes (16): Analysis, Boolean Operations, code:bash (pip install git+https://github.com/RodionovSA/Metashapes), code:python (from metashapes import UnitCell, Lattice), code:python (from metashapes.shape import Ellipse, Rectangle), code:python (from metashapes import Lattice), code:python (from metashapes import UnitCellAnalyzer), code:python (from metashapes.adapters.yaml import save_unit_cells, load_u) (+8 more)

### Community 22 - "Generator Validation"
Cohesion: 0.16
Nodes (15): gdstk_to_shapely(), klayout_to_shapely(), numpy_to_shapely(), Convert a shapely geometry to a gdstk polygon or a list of polygons., Convert a gdstk polygon to a shapely geometry., Convert a shapely geometry to a KLayout polygon or a list of polygons., Create a shapely geometry from a binary numpy array.     Parameters:         img, shapely_to_gdstk() (+7 more)

### Community 23 - "Community 23"
Cohesion: 0.17
Nodes (8): Rasterize the periodic structure into a mask. Shape [ny·n2, nx·n1].          sof, Return world-coordinate points on the material boundary (zero-level-set)., A periodic structure: one Lattice + one Shape (the scene).      The lattice owns, Axis-aligned Cartesian bounding box of the supercell.          Returns ``(xmin,, Number of periodic copies to search per lattice direction.          A finite sha, Periodic signed distance of the scene at Cartesian (x, y).          Minimum over, Periodic SDF sampled over a supercell.          repeat=(n1, n2) — tile n1 cells, UnitCell

### Community 24 - "Community 24"
Cohesion: 0.19
Nodes (8): For a symmetric Rectangle, both methods produce the same offset., For a centred Ellipse, both methods produce the same offset., Extract the outermost Translate dx/dy buffers from the scene., TestAlreadyCentered, TestBboxHexagonal, TestSymmetricShapeConsistency, TestUnknownMethodError, _translate_offset()

### Community 25 - "Community 25"
Cohesion: 0.21
Nodes (5): Rectangular cell with a Rectangle shape centred at (cx, cy)., _rect_cell(), TestBboxRectangular, TestChaining, TestImmutability

### Community 26 - "Test Package Root"
Cohesion: 0.15
Nodes (12): code:bash (graphify query "<your question>"       # any codebase questi), code:bash (source .venv/bin/activate), code:block3 (metashapes/), graphify, Key Modules, Metashapes — Developer Guide, Navigating This Codebase, New Shape Primitive Addition (+4 more)

### Community 28 - "Lattice Test Init"
Cohesion: 0.35
Nodes (7): SDF Convention: Negative Inside, Positive Outside, assert_bounds_contain(), assert_outside(), All points should have SDF < tol (inside or on boundary)., All points should have SDF > -tol (outside or on boundary)., All points should lie inside (or on) the reported bounding box., sdf_at()

### Community 31 - "Lattice Basis Rationale"
Cohesion: 0.40
Nodes (3): nn.Parameter centre of a shape must receive gradients after center_scene()., make_learnable_polygon center param grad is non-None after centering., TestGradientFlow

### Community 32 - "Lattice Basis Rationale B"
Cohesion: 0.50
Nodes (3): _cell_center(), The geometric midpoint (a1 + a2) / 2 as plain floats., TestCentroidMethod

## Knowledge Gaps
- **27 isolated node(s):** `allow`, `PreToolUse`, `code:bash (graphify query "<your question>"       # any codebase questi)`, `Overview`, `code:bash (source .venv/bin/activate)` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Shape` connect `Shape Primitives Core` to `Random Generator & Lattice`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `Conic Shape Primitives`, `YAML Serialization Tests`, `Shapely Transform Tests`, `Shapely Compound Shape Tests`, `Generator Base Classes`, `Community 11`, `Shapely Adapter Modules`, `Unit Cell Tests`, `PyTorch Differentiability`, `Lattice SDF Tests`, `Coordinate Transform Bridge`, `Periodic Unit Cell Ops`, `Shape Test Init`, `Lattice Test Init`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Why does `Lattice` connect `Unit Cell Analyzer` to `Shape Primitives Core`, `Shape Analysis & SDF Concepts`, `YAML Serialization Tests`, `Random Generator Logic`, `Shapely Compound Shape Tests`, `Generator Integration Tests`, `Community 11`, `YAML & Unit Cell Serialization`, `Shapely Adapter Layer`, `PyTorch Differentiability`, `Boolean Shape Tests`, `Periodic Unit Cell Ops`, `Community 23`, `Community 24`, `Community 25`, `Adapters Test Init`, `Generators Test Init`, `Lattice Basis Rationale`, `Lattice Basis Rationale B`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Why does `UnitCell` connect `Community 23` to `Shape Primitives Core`, `Random Generator & Lattice`, `Shape Analysis & SDF Concepts`, `Unit Cell Analyzer`, `YAML Serialization Tests`, `Random Generator Logic`, `Shapely Compound Shape Tests`, `Community 11`, `YAML & Unit Cell Serialization`, `Shapely Adapter Layer`, `Boolean Shape Tests`, `Periodic Unit Cell Ops`, `Community 24`, `Community 25`, `Adapters Test Init`, `Generators Test Init`, `Lattice Basis Rationale`, `Lattice Basis Rationale B`, `Validator Rationale`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Are the 64 inferred relationships involving `UnitCell` (e.g. with `CellMetrics` and `UnitCellAnalyzer`) actually correct?**
  _`UnitCell` has 64 INFERRED edges - model-reasoned connections that need verification._
- **Are the 74 inferred relationships involving `Lattice` (e.g. with `CellMetrics` and `UnitCellAnalyzer`) actually correct?**
  _`Lattice` has 74 INFERRED edges - model-reasoned connections that need verification._
- **Are the 69 inferred relationships involving `Rectangle` (e.g. with `Shape` and `TestLeafShapes`) actually correct?**
  _`Rectangle` has 69 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `shape_to_shapely()` (e.g. with `.test_bar_gap_is_perpendicular_only()` and `.test_bar_x_height()`) actually correct?**
  _`shape_to_shapely()` has 5 INFERRED edges - model-reasoned connections that need verification._